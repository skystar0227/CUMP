/* Include file for internal CUMP types, definitions and inlines.

   THE CONTENTS OF THIS FILE ARE FOR INTERNAL USE AND ARE ALMOST CERTAIN TO
   BE SUBJECT TO INCOMPATIBLE CHANGES IN FUTURE CUMP RELEASES.

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


#ifndef CUMP_IMPL_H_


#include "include/cump/def.h"


#define __CUMP_EXTERN_INLINE  static inline

#define CUMP_LIMB_BYTES  (sizeof (cump_limb_t))


extern  void*  (*__cump_allocate_func) (size_t);
extern  void*  (*__cump_allocate_2D_func) (size_t*, size_t, size_t);
extern  void*  (*__cump_reallocate_func) (void*, size_t, size_t);
extern  void   (*__cump_free_func) (void*, size_t);

extern  void  (*__cump_memcpy_func) (void*, void const *, size_t);
extern  void  (*__cump_memcpy_h2d_func) (void*, void const *, size_t);
extern  void  (*__cump_memcpy_d2h_func) (void*, void const *, size_t);
extern  void  (*__cump_memcpy_d2d_func) (void*, void const *, size_t);
extern  void  (*__cump_memcpy_2D_func) (void*, size_t, void const *, size_t, size_t, size_t);
extern  void  (*__cump_memcpy_2D_h2d_func) (void*, size_t, void const *, size_t, size_t, size_t);
extern  void  (*__cump_memcpy_2D_d2h_func) (void*, size_t, void const *, size_t, size_t, size_t);
extern  void  (*__cump_memcpy_2D_d2d_func) (void*, size_t, void const *, size_t, size_t, size_t);

extern  cump_size_t  __cump_host_default_fp_limb_precision;


#if defined (__cplusplus)
extern "C"
{
#endif


__CUMP_EXTERN_INLINE  void  __cumpf_get_header (cumpf_header  *h, cumpf_srcptr  p)
{
  (*__cump_memcpy_d2h_func) (h, p->_dev, sizeof (*h));
}


__CUMP_EXTERN_INLINE  cump_size_t  __cumpf_get_prec (cumpf_srcptr  p)
{
  cump_int32  prec;
  (*__cump_memcpy_d2h_func) (&prec, p->_dev, sizeof (prec));
  return  prec;
}


__CUMP_EXTERN_INLINE  void  __cumpf_init (cumpf_ptr  r, cump_size_t  prec)
{
  cumpf_header  hd;
  hd._mp_exp = 0;
  hd._mp_size = 0;
  hd._mp_prec = prec;
  r->_dev = (*__cump_allocate_func) (__CUMPF_ALLOCSIZE (prec));
  (*__cump_memcpy_h2d_func) (r->_dev, &hd, sizeof (hd));
}


__CUMP_EXTERN_INLINE  cump_size_t  __cumpf_array_get_prec (cumpf_array_srcptr  p)
{
  cump_int32  prec;
  (*__cump_memcpy_d2h_func) (&prec, p->_dev, sizeof (prec));
  return  prec;
}


void  __cumpf_array_init (cumpf_array_ptr, cump_uint32, cump_size_t);


__CUMP_EXTERN_INLINE
void  CUMPN_COPY_FROM_ARRAY (cump_ptr  dst, char const  *src, size_t  p, cump_size_t  n)
{
  if (n)
    {
      cump_limb_t  x = *(cump_srcptr) src;
      while (--n)
        {
          *dst++ = x;
          x = *(cump_srcptr) (src += p);
        }
      *dst++ = x;
    }
}

__CUMP_EXTERN_INLINE
void  CUMPN_COPY_TO_ARRAY (char  *dst, size_t  p, cump_srcptr  src, cump_size_t  n)
{
  if (n)
    {
      cump_limb_t  x = *src++;
      while (--n)
        {
          *(cump_ptr) dst = x;
          dst += p;
          x = *src++;
        }
      *(cump_ptr) dst = x;
    }
}


#if defined (__cplusplus)
} /* extern "C" */
#endif


#define CUMP_IMPL_H_
#endif
