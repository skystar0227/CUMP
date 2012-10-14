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

SUBDIRS = cumpf demos include mpf

include_HEADERS = cump.h
lib_LIBRARIES = libcump.a

AM_CPPFLAGS = -D__CUMP_WITHIN_CUMP -I@CUDA_DIR@/include

if SPECIFY_GMP_DIR
  AM_CPPFLAGS += -I@GMP_DIR@
endif

CUMPF_OBJECTS = \
  cumpf/set_dfl_prec.$(OBJEXT) cumpf/get_dfl_prec.$(OBJEXT) \
  cumpf/init.$(OBJEXT) cumpf/init2.$(OBJEXT) cumpf/iset.$(OBJEXT) \
  cumpf/clear.$(OBJEXT) cumpf/set.$(OBJEXT) \
  cumpf/ainit.$(OBJEXT) cumpf/ainit2.$(OBJEXT) \
  cumpf/aclear.$(OBJEXT) cumpf/aset.$(OBJEXT) \
  cumpf/_ainit.$(OBJEXT) cumpf/_set.$(OBJEXT)

CUMPF_OBJECTS_WITH_GMP = \
  cumpf/iset_mpf.$(OBJEXT) cumpf/set_mpf.$(OBJEXT) \
  cumpf/aiset_mpf.$(OBJEXT) cumpf/aset_mpf.$(OBJEXT)

MPF_OBJECTS = mpf/aset_cumpf.$(OBJEXT) mpf/set_cumpf.$(OBJEXT)

OBJECTS = $(CUMPF_OBJECTS)

if USE_GMP
  OBJECTS += $(CUMPF_OBJECTS_WITH_GMP) $(MPF_OBJECTS)
endif

libcump_a_SOURCES = cump.h cump-impl.h memory.c
libcump_a_DEPENDENCIES = $(OBJECTS)
libcump_a_LIBADD = $(libcump_a_DEPENDENCIES)
