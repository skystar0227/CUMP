/* cumpf_array_init2 -- Make a new array of multiple precision numbers with value 0.

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


void  cumpf_array_init2 (cumpf_array_ptr  r, cump_uint32  n, cump_bitcnt_t  prec_in_bits)
{
  __cumpf_array_init (r, n, __CUMPF_BITS_TO_PREC (prec_in_bits));
}
