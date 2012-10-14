/* cumpf_array_init_set_mpf -- Initialize an array of floats and assign it from
   an array of mpf_t.

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


#include <stdlib.h>
#include <gmp.h>
#include "cump.h"
#include "cump-impl.h"


void  __cumpf_array_init_set_mpf
( char  *rp, size_t  sLine, cump_size_t  prec
, cump_size_t  *pSize, cump_exp_t  *pExp, mpf_srcptr  s
)
{
  cump_srcptr  sp;
  cump_size_t  size, ssize;

  ssize = s->_mp_size;
  size = ABS (ssize);

  sp = (cump_srcptr) s->_mp_d;

  if (size > prec)
    {
      sp += size - prec;
      size = prec;
    }

  *pExp = s->_mp_exp;
  *pSize = ssize >= 0 ? size : -size;

  CUMPN_COPY_TO_ARRAY (rp, sLine, sp, size);
}


void  cumpf_array_init_set_mpf (cumpf_array_ptr  r, mpf_t  *sa, cump_uint32  n)
{
  cump_size_t  prec = __cump_host_default_fp_limb_precision;
  size_t  width = n * CUMP_LIMB_BYTES;
  size_t  height = __CUMPF_ARRAY_ELEMSIZE (prec);
  size_t  interval;
  char  *dp, *hp;
  int  i;

  dp = (char*) (*__cump_allocate_2D_func) (&interval, width, height);
  hp = (char*) malloc (width * height);

  for (i = 0;  i < n;  ++i)
    {
      char  *p = hp + i * CUMP_LIMB_BYTES;
      cump_size_t  s;
      cump_exp_t  e;
      __cumpf_array_set_mpf (p + width * 2, width, prec + 1, &s, &e, sa [i]);
      *(cump_int32*) p = prec;
      *(cump_int32*) (p + sizeof (cump_int32)) = s;
      *(cump_exp_t*) (p + width) = e;
    }

  (*__cump_memcpy_2D_h2d_func) (dp, interval, hp, width, width, height);

  free (hp);

  r->_int = interval;
  r->_dev = dp;
}
