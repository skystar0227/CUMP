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

CU = @CU@
CUFLAGS = @CUFLAGS@ -cuda
SUFFIXES = .cu

.cu.$(OBJEXT):
	$(CU) $(CUFLAGS) $(AM_CPPFLAGS) -o $*.cpp $<
	$(CXX) $(CXXFLAGS) $(AM_CPPFLAGS) -c -o $@ $*.cpp

if USE_GMP
  CUMPF_OBJECTS_WITH_GMP = \
    iset_mpf.$(OBJEXT) set_mpf.$(OBJEXT) \
    aiset_mpf.$(OBJEXT) aset_mpf.$(OBJEXT)
endif

noinst_LIBRARIES = libcumpf.a
libcumpf_a_SOURCES = \
  set_dfl_prec.c get_dfl_prec.c \
  init.c init2.c iset.c clear.c set.c \
  ainit.c ainit2.c aclear.c aset.c \
  _set.cpp _ainit.cu
EXTRA_libcumpf_a_SOURCES = \
  aiset_mpf.c aset_mpf.c \
  iset_mpf.c set_mpf.c
libcumpf_a_DEPENDENCIES = $(CUMPF_OBJECTS_WITH_GMP)
libcumpf_a_LIBADD = $(libcumpf_a_DEPENDENCIES)
