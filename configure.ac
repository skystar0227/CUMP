#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.


define(CUMP_COPYRIGHT,[[

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
along with the CUMP Library.  If not, see http://www.gnu.org/licenses/.
]])

AC_COPYRIGHT(CUMP_COPYRIGHT)
AH_TOP(/*CUMP_COPYRIGHT*/)

AC_PREREQ([2.63])
AC_INIT([CUMP], [1.0.1], [nakayama@hpcs.cs.tsukuba.ac.jp], cump)
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([cump-impl.h])
AC_CONFIG_HEADERS([config.h])

# CUDA compiler.
if test -z $CU ; then
  CU="nvcc"
fi

if test -z $CUFLAGS ; then
  CUFLAGS="-arch=sm_20 $CXXFLAGS"
fi

AC_SUBST(CU)
AC_SUBST(CUFLAGS)

AC_ARG_VAR(CU, [CUDA compiler command])
AC_ARG_VAR(CUFLAGS, [CUDA compiler flags])

# CUDA toolkit install directory.
AC_ARG_WITH(cuda,
  AC_HELP_STRING([--with-cuda[[=DIR]]],
    [specify CUDA toolkit install directory [[default=/usr/local/cuda]]]),
  [CUDA_DIR="$withval"], [CUDA_DIR="/usr/local/cuda"])

AC_SUBST(CUDA_DIR)

# GNU MP include file directory.
AC_ARG_WITH(gmp,
  AC_HELP_STRING([--with-gmp[[=DIR]]],
    [use GNU MP (specify GNU MP include file directory) [[default=yes]]]),
  [case $withval in
yes|no) ;;
*) GMP_DIR="$withval"; with_gmp="yes";;
esac],
  [with_gmp="yes"])

AM_CONDITIONAL(USE_GMP, test x$with_gmp = xyes)
AM_CONDITIONAL(SPECIFY_GMP_DIR, test ! x$GMP_DIR = x)
AC_SUBST(GMP_DIR)

# Checks for programs.
AC_PROG_RANLIB
AC_PROG_CXX
AC_PROG_CC

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([stddef.h stdlib.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC

AC_CONFIG_FILES([Makefile
                 cumpf/Makefile
                 demos/Makefile
                 include/Makefile
                 mpf/Makefile])
AC_OUTPUT
