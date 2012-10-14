/* cumpf_init_set_mpf -- Initialize a float and assign it from an mpf_t.

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


#include <gmp.h>
#include "cump.h"
#include "cump-impl.h"


void  __cumpf_set_mpf (cumpf_ptr  r, mpf_srcptr  u, cumpf_header  *h);

void  cumpf_init_set_mpf (cumpf_ptr  r, mpf_srcptr  s)
{
  cumpf_header  hd;
  cump_size_t  prec = __cump_host_default_fp_limb_precision;
  r->_dev = (*__cump_allocate_func) (__CUMPF_ALLOCSIZE (prec));
  hd._mp_prec = prec;
  hd._mp_size = s->_mp_size;
  hd._mp_exp = s->_mp_exp;
  __cumpf_set_mpf (r, s, &hd);
  (*__cump_memcpy_h2d_func) (r->_dev, &hd, sizeof (hd));
}
