/* mpf_set_cumpf -- Assign an mpf_t from a float.

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


void  __mpf_set_cumpf (mpf_ptr  r, cumpf_srcptr  u, cumpf_header  *h);

void  mpf_set_cumpf (mpf_ptr  r, cumpf_srcptr  u)
{
  cumpf_header  hd;
  __cumpf_get_header (&hd, u);
  hd._mp_prec = r->_mp_prec;
  __mpf_set_cumpf (r, u, &hd);
  r->_mp_size = hd._mp_size;
  r->_mp_exp = hd._mp_exp;
}
