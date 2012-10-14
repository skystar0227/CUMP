## Process this file with automake to generate Makefile.in


# Copyright 2012 Takatoshi Nakayama.
#
# This file is part of the CUMP Library.
#
# The CUMP Library is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# The CUMP Library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with the CUMP Library.  If not, see http://www.gnu.org/licenses/.

AM_CPPFLAGS = -D__CUMP_WITHIN_CUMP -I@CUDA_DIR@/include -I$(top_srcdir)

if SPECIFY_GMP_DIR
  AM_CPPFLAGS += -I@GMP_DIR@
endif

noinst_LIBRARIES = libmpf.a
libmpf_a_SOURCES = aset_cumpf.c set_cumpf.c
