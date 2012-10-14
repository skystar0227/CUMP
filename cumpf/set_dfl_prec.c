/* cumpf_set_default_prec -- set default precision in bits.

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
#include "cump.h"
#include "cump-impl.h"


cump_size_t  __cump_host_default_fp_limb_precision = __CUMPF_BITS_TO_PREC (53);

void  cumpf_set_default_prec (cump_bitcnt_t  prec_in_bits)
{
  cump_size_t  prec = __CUMPF_BITS_TO_PREC (prec_in_bits);
  __cump_host_default_fp_limb_precision = prec;
  cudaMemcpyToSymbol
  ( "__cump_default_fp_limb_precision", &prec
  , sizeof (prec), 0, cudaMemcpyHostToDevice
  );
}
