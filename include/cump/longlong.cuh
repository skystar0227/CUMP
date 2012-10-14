/* longlong.h -- inlines for mixed size 32/64 bit arithmetic.

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


#ifndef CUMP_LONGLONG_CUH_


CUMP_FUNCTYPE
void  umul_ppmm (cump_uint32  &w1, cump_uint32  &w0, cump_uint32  u, cump_uint32  v)
{
  w0 = u * v;
  w1 = __umulhi (u, v);
}

CUMP_FUNCTYPE
void  umul_ppmm (cump_uint64  &w1, cump_uint64  &w0, cump_uint64  u, cump_uint64  v)
{
  w0 = u * v;
  w1 = __umul64hi (u, v);
}


#define CUMP_LONGLONG_CUH_
#endif
