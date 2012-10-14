/* Include file for internal CUMP types and definitions.

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


#ifndef CUMP_DEF_H_


typedef
  struct
  {
    cump_int32  _mp_prec;
    cump_int32  _mp_size;
    cump_exp_t  _mp_exp;
  }
  cumpf_header;


#define __CUMPF_ALLOCSIZE(prec)       (((prec) + 1) * sizeof (cump_limb_t) + sizeof (cumpf_header))
#define __CUMPF_ARRAY_ELEMSIZE(prec)  ((prec) + (1 + sizeof (cumpf_header) / sizeof (cump_limb_t)))

#define ABS(x)  ((x) >= 0 ? (x) : -(x))
#define __CUMP_MAX(h,i) ((h) > (i) ? (h) : (i))

#define __CUMPF_BITS_TO_PREC(n)  \
  ((cump_size_t) ((__CUMP_MAX (53, n) + 2 * CUMP_NUMB_BITS - 1) / CUMP_NUMB_BITS))
#define __CUMPF_PREC_TO_BITS(n)  ((cump_bitcnt_t) (n) * CUMP_NUMB_BITS - CUMP_NUMB_BITS)


#define CUMP_DEF_H_
#endif
