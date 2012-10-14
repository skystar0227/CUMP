/* cumpf_set -- Assign a float from another float.

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


#include "cump.h"
#include "cump-impl.h"


void  __cumpf_set (cumpf_ptr  r, cumpf_srcptr  u, cumpf_header  *h);

void  cumpf_set (cumpf_ptr  r, cumpf_srcptr  u)
{
  cumpf_header  hd;
  __cumpf_get_header (&hd, u);
  hd._mp_prec = __cumpf_get_prec (r);
  __cumpf_set (r, u, &hd);
  (*__cump_memcpy_h2d_func) (r->_dev, &hd, sizeof (hd));
}
