/* cumpf_array_set_mpf -- Assign an array of floats from an array of mpf_t.

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


void  __cumpf_array_set_mpf
( char  *rp, size_t  sLine, cump_size_t  prec
, cump_size_t  *pSize, cump_exp_t  *pExp, mpf_srcptr  u
)
{
  cump_srcptr  up;
  cump_size_t  size, asize;

  size = u->_mp_size;
  asize = ABS (size);

  up = (cump_srcptr) u->_mp_d;

  if (asize > prec)
    {
      up += asize - prec;
      asize = prec;
    }

  *pExp = u->_mp_exp;
  *pSize = size >= 0 ? asize : -asize;

  CUMPN_COPY_TO_ARRAY (rp, sLine, up, asize);
}


void  cumpf_array_set_mpf (cumpf_array_ptr  r, mpf_t  *ua, cump_uint32  n)
{
  size_t  width, height;
  cump_size_t  prec;
  char  *hp;
  int  i;

  prec = __cumpf_array_get_prec (r);
  width = n * CUMP_LIMB_BYTES;
  height = __CUMPF_ARRAY_ELEMSIZE (prec);
  hp = (char*) malloc (width * height);

  for (i = 0;  i < n;  ++i)
    {
      char  *p = hp + i * CUMP_LIMB_BYTES;
      cump_size_t  s;
      cump_exp_t  e;
      __cumpf_array_set_mpf (p + width * 2, width, prec + 1, &s, &e, ua [i]);
      *(cump_uint32*) p = prec;
      *(cump_uint32*) (p + sizeof (cump_uint32)) = s;
      *(cump_exp_t*) (p + width) = e;
    }

  (*__cump_memcpy_2D_h2d_func) (r->_dev, r->_int, hp, width, width, height);

  free (hp);
}
