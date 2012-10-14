/* Definitions for CUMP functions.

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


#ifndef CUMP_H_


#include <stddef.h> /* size_t */


/* General macros following GMP */
#define CUMP_LIMB_BITS  (8 * sizeof (cump_limb_t))
#define CUMP_NAIL_BITS  0
#define CUMP_NUMB_BITS  (CUMP_LIMB_BITS - CUMP_NAIL_BITS)
#define CUMP_NUMB_MASK  ((~ __CUMP_CAST (cump_limb_t, 0)) >> CUMP_NAIL_BITS)
#define CUMP_NUMB_MAX   CUMP_NUMB_MASK
#define CUMP_NAIL_MASK  (~ CUMP_NUMB_MASK)


#if defined (__cplusplus)
#define __CUMP_CAST(type, expr)  static_cast <type> (expr)
#define __CUMP_NOTHROW  /*throw ()*/
#else
#define __CUMP_CAST(type, expr)  ((type) (expr))
#define __CUMP_NOTHROW
#endif


typedef  int       cump_int32;
typedef  long int  cump_int64;
typedef  unsigned int       cump_uint32;
typedef  unsigned long int  cump_uint64;

#ifdef __CUMP_SHORT_LIMB
typedef  cump_uint32  cump_limb_t;
typedef  cump_int32   cump_limb_signed_t;
#else
typedef  cump_uint64  cump_limb_t;
typedef  cump_int64   cump_limb_signed_t;
#endif
typedef  cump_uint64  cump_bitcnt_t;

typedef  cump_int64   cump_size_t;
typedef  cump_int64   cump_exp_t;

typedef  cump_limb_t        *cump_ptr;
typedef  cump_limb_t const  *cump_srcptr;


typedef
  struct
  {
    void  *_dev;
  }
  __cumpf_struct;

typedef  __cumpf_struct        *cumpf_ptr, cumpf_t [1];
typedef  __cumpf_struct const  *cumpf_srcptr;


typedef
  struct
  {
    size_t  _int;
    void  *_dev;
  }
  __cumpf_array;

typedef  __cumpf_array        *cumpf_array_ptr, cumpf_array_t [1];
typedef  __cumpf_array const  *cumpf_array_srcptr;


#if defined (__cplusplus)
extern "C"
{
#endif

void  cumpf_set_default_prec (cump_bitcnt_t  prec_in_bits);
cump_bitcnt_t  cumpf_get_default_prec (void);

void  cumpf_init (cumpf_ptr  r);
void  cumpf_init2 (cumpf_ptr  r, cump_bitcnt_t  prec_in_bits);
void  cumpf_init_set (cumpf_ptr  r, cumpf_srcptr  s);
void  cumpf_clear (cumpf_ptr  m);
void  cumpf_set (cumpf_ptr  r, cumpf_srcptr  u);

void  cumpf_array_init (cumpf_array_ptr  r, cump_uint32  n);
void  cumpf_array_init2 (cumpf_array_ptr  r, cump_uint32  n, cump_bitcnt_t  prec_in_bits);
void  cumpf_array_clear (cumpf_array_ptr  m);
void  cumpf_array_set (cumpf_array_ptr  r, cumpf_array_srcptr  u, cump_uint32  n);

#ifdef __GMP_H__
void  cumpf_init_set_mpf (cumpf_ptr  r, mpf_srcptr  s);
void  cumpf_set_mpf (cumpf_ptr  r, mpf_srcptr  u);
void  mpf_set_cumpf (mpf_ptr  r, cumpf_srcptr  u);

void  cumpf_array_init_set_mpf (cumpf_array_ptr  r, mpf_t  *sa, cump_uint32  n);
void  cumpf_array_set_mpf (cumpf_array_ptr  r, mpf_t  *ua, cump_uint32  n);
void  mpf_array_set_cumpf (mpf_t  *ra, cumpf_array_srcptr  u, cump_uint32  n);
#endif /* __GMP_H__ */

#if defined (__cplusplus)
} /* extern "C" */
#endif


#define CUMP_H_
#endif
