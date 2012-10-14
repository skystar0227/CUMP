/* cumpf_array_set -- Assign an array of floats from another array of floats.

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
#include "cump.h"
#include "cump-impl.h"


void  cumpf_array_set (cumpf_array_ptr  r, cumpf_array_srcptr  u, cump_uint32  n)
{
  (*__cump_memcpy_2D_d2d_func)
  ( r->_dev, r->_int, u->_dev, u->_int, n * CUMP_LIMB_BYTES
  , __CUMPF_ARRAY_ELEMSIZE (__cumpf_array_get_prec (r))
  );
}
