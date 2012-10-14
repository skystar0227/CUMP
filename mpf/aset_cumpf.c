/* mpf_array_set_cumpf -- Assign an array of mpf_t from an array of float.

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


void  __mpf_array_set_cumpf
( mpf_ptr  r, cump_size_t  size, cump_exp_t  exp
, char const  *up, size_t  sLine
)
{
  cump_ptr  rp;
  cump_size_t  asize;
  cump_size_t  prec;

  prec = r->_mp_prec + 1;	/* lie not to lose precision in assignment */
  asize = ABS (size);

  rp = (cump_ptr) r->_mp_d;

  if (asize > prec)
    {
      up += (asize - prec) * sLine;
      asize = prec;
    }

  r->_mp_exp = exp;
  r->_mp_size = size >= 0 ? asize : -asize;

  CUMPN_COPY_FROM_ARRAY (rp, up, sLine, asize);
}


void  mpf_array_set_cumpf (mpf_t  *ra, cumpf_array_srcptr  u, cump_uint32  n)
{
  size_t  width, height;
  cump_size_t  prec;
  char  *dp;
  int  i;

  prec = __cumpf_array_get_prec (u);
  width = n * CUMP_LIMB_BYTES;
  height = __CUMPF_ARRAY_ELEMSIZE (prec);
  dp = (char*) malloc (width * height);

  (*__cump_memcpy_2D_d2h_func) (dp, width, u->_dev, u->_int, width, height);

  for (i = 0;  i < n;  ++i)
    {
      char const  *p = dp + i * CUMP_LIMB_BYTES;
      cump_size_t  s = *(cump_int32*) (p + sizeof (cump_int32));
      cump_exp_t  e = *(cump_exp_t*) (p + width);
      __mpf_array_set_cumpf (ra [i], s, e, p + width * 2, width);
    }

  free (dp);
}
