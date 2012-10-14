/* cumpf_set_mpf -- Assign a float from an mpf_t.

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

void  cumpf_set_mpf (cumpf_ptr  r, mpf_srcptr  u)
{
  cumpf_header  hd;
  hd._mp_prec = __cumpf_get_prec (r);
  hd._mp_size = u->_mp_size;
  hd._mp_exp = u->_mp_exp;
  __cumpf_set_mpf (r, u, &hd);
  (*__cump_memcpy_h2d_func) (r->_dev, &hd, sizeof (hd));
}
