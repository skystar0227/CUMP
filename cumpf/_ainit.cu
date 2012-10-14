/* Inner functions for init functions.

Copyright 2012 Takatoshi Nakayama.

This file is part of the CUMP Library.

The CUMP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

The CUMP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the CUMP Library.  If not, see http://www.gnu.org/licenses/.  */


#include <cuda_runtime_api.h>
#include "include/cump/cump.cuh"
#include "cump-impl.h"


__global__
void  cumpf_array_init_kernel (cump::mpf_array_t  r, cump_uint32  n, cump_int32  prec)
{
  int  i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    {
      cump::mpf::Float <typename cump::mpf_array_t::Pointer>  f (r [i]);
      f._mp_prec () = prec;
      f._mp_size () = 0;
      f._mp_exp () = 0;
    }
}  // cumpf_array_init_kernel ()


extern "C"  void  __cumpf_array_init (cumpf_array_ptr  r, cump_uint32  n, cump_size_t  prec)
{
  std::size_t  interval;
  r->_dev =
    (*__cump_allocate_2D_func) (&interval, n * CUMP_LIMB_BYTES, __CUMPF_ARRAY_ELEMSIZE (prec));
  r->_int = interval;

  unsigned int const  maxBlocksPerDim = 65535u;
  dim3 const  threads (128u);
  dim3  blocks;
  cump_uint32  _n = n / threads.x;
  blocks.y = _n / (maxBlocksPerDim + 1u) + 1u;
  blocks.x = _n / blocks.y + 1u;
  cumpf_array_init_kernel <<<blocks, threads>>> (r, n, prec);
  cudaThreadSynchronize ();
}  // __cumpf_array_init ()
