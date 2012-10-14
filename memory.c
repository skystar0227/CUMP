/* Memory allocation routines.

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


#if defined (_DEBUG) || defined (DEBUG)
#include <stdio.h>
#endif
#include <cuda_runtime_api.h>
#include "cump.h"
#include "cump-impl.h"


void*  __cump_default_allocate (size_t);
void*  __cump_default_allocate_2D (size_t*, size_t, size_t);
/*void*  __cump_default_reallocate (void*, size_t, size_t);*/
void    __cump_default_free (void*, size_t);

void  __cump_default_memcpy (void*, void const *, size_t);
void  __cump_default_memcpy_h2d (void*, void const *, size_t);
void  __cump_default_memcpy_d2h (void*, void const *, size_t);
void  __cump_default_memcpy_d2d (void*, void const *, size_t);
void  __cump_default_memcpy_2D (void*, size_t, void const *, size_t, size_t, size_t);
void  __cump_default_memcpy_2D_h2d (void*, size_t, void const *, size_t, size_t, size_t);
void  __cump_default_memcpy_2D_d2h (void*, size_t, void const *, size_t, size_t, size_t);
void  __cump_default_memcpy_2D_d2d (void*, size_t, void const *, size_t, size_t, size_t);


void*  (*__cump_allocate_func) (size_t)
  = __cump_default_allocate;
void*  (*__cump_allocate_2D_func) (size_t*, size_t, size_t)
  = __cump_default_allocate_2D;
/*
void*  (*__cump_reallocate_func) (void*, size_t, size_t)
  = __cump_default_reallocate;
*/
void   (*__cump_free_func) (void*, size_t)
  = __cump_default_free;
void  (*__cump_memcpy_func) (void*, void const *, size_t)
  = __cump_default_memcpy;
void  (*__cump_memcpy_h2d_func) (void*, void const *, size_t)
  = __cump_default_memcpy_h2d;
void  (*__cump_memcpy_d2h_func) (void*, void const *, size_t)
  = __cump_default_memcpy_d2h;
void  (*__cump_memcpy_d2d_func) (void*, void const *, size_t)
  = __cump_default_memcpy_d2d;
void  (*__cump_memcpy_2D_func) (void*, size_t, void const *, size_t, size_t, size_t)
  = __cump_default_memcpy_2D;
void  (*__cump_memcpy_2D_h2d_func) (void*, size_t, void const *, size_t, size_t, size_t)
  = __cump_default_memcpy_2D_h2d;
void  (*__cump_memcpy_2D_d2h_func) (void*, size_t, void const *, size_t, size_t, size_t)
  = __cump_default_memcpy_2D_d2h;
void  (*__cump_memcpy_2D_d2d_func) (void*, size_t, void const *, size_t, size_t, size_t)
  = __cump_default_memcpy_2D_d2d;


#if defined (_DEBUG) || defined (DEBUG)
#define SAFE_CALL(call)                             \
do                                                  \
  {                                                 \
    cudaError_t e = call;                           \
    if (e != cudaSuccess)                           \
      {                                             \
        fprintf                                     \
        ( stderr, "Device memory error (%s): %s.\n" \
        , __func__, cudaGetErrorString (e)          \
        );                                          \
        exit (EXIT_FAILURE);                        \
      }                                             \
  }                                                 \
while(0)
#else
#define SAFE_CALL(call)  (call)
#endif


void*  __cump_default_allocate (size_t  size)
{
  void  *ret;
  SAFE_CALL (cudaMalloc (&ret, size));
  return  ret;
}


void*  __cump_default_allocate_2D (size_t  *pitch, size_t  width, size_t  height)
{
  void  *ret;
  SAFE_CALL (cudaMallocPitch (&ret, pitch, width, height));
  return  ret;
}


/*void*  __cump_default_reallocate (void  *oldptr, size_t  old_size, size_t  new_size)
{
  void  *ret = 0;
  return  ret;
}*/


void  __cump_default_free (void  *blk_ptr, size_t  blk_size)
{
  SAFE_CALL (cudaFree (blk_ptr));
}


void  __cump_default_memcpy (void  *dst, void const  *src, size_t  size)
{
  SAFE_CALL (cudaMemcpy (dst, src, size, cudaMemcpyDefault));
}


void  __cump_default_memcpy_h2d (void  *dst, void const  *src, size_t  size)
{
  SAFE_CALL (cudaMemcpy (dst, src, size, cudaMemcpyHostToDevice));
}


void  __cump_default_memcpy_d2h (void  *dst, void const  *src, size_t  size)
{
  SAFE_CALL (cudaMemcpy (dst, src, size, cudaMemcpyDeviceToHost));
}


void  __cump_default_memcpy_d2d (void  *dst, void const  *src, size_t  size)
{
  SAFE_CALL (cudaMemcpy (dst, src, size, cudaMemcpyDeviceToDevice));
}


void  __cump_default_memcpy_2D
( void  *dst, size_t  dpitch, void const  *src, size_t  spitch
, size_t  width, size_t  height
)
{
  SAFE_CALL (cudaMemcpy2D (dst, dpitch, src, spitch, width, height, cudaMemcpyDefault));
}


void  __cump_default_memcpy_2D_h2d
( void  *dst, size_t  dpitch, void const  *src, size_t  spitch
, size_t  width, size_t  height
)
{
  SAFE_CALL (cudaMemcpy2D (dst, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
}


void  __cump_default_memcpy_2D_d2h
( void  *dst, size_t  dpitch, void const  *src, size_t  spitch
, size_t  width, size_t  height
)
{
  SAFE_CALL (cudaMemcpy2D (dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost));
}


void  __cump_default_memcpy_2D_d2d
( void  *dst, size_t  dpitch, void const  *src, size_t  spitch
, size_t  width, size_t  height
)
{
  SAFE_CALL (cudaMemcpy2D (dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice));
}
