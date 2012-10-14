/* Inner functions for set functions.

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


namespace
{


enum CmpResult  {LARGER, EQUAL, SMALLER};

template <CmpResult, typename, typename>  struct Setter;

template <typename  Dst, typename  Src>
struct Setter <EQUAL, Dst, Src>
{
  static  void  set (Dst  *rp, Src  *up, cumpf_header  *h)
  {
    cump_size_t  prec = h->_mp_prec + 1;	/* lie not to lose precision in assignment */
    cump_size_t  size = h->_mp_size;
    cump_size_t  asize = ABS (size);

    if (asize > prec)
      {
        up += asize - prec;
        asize = prec;
      }

    h->_mp_size = size >= 0 ? asize : -asize;
    if (asize)  {(*__cump_memcpy_func) (rp, up, asize * sizeof (*up));}
  }
};

template <typename  Dst, typename  Src>
struct Setter <LARGER, Dst, Src>
{
  static  void  set (Dst  *rp, Src  *up, cumpf_header  *h)
  {
    cump_size_t  prec = h->_mp_prec + 1;	/* lie not to lose precision in assignment */
    cump_size_t  size = h->_mp_size;
    cump_size_t  asize = ABS (size);
    if (asize)
      {
        int const  scale = sizeof (*rp) / sizeof (*up);
        Dst const  zero = 0;

        cump_exp_t  uexp = h->_mp_exp;
        cump_exp_t  auexp = ABS (uexp);
        int  expPad = scale - auexp % scale;
        int  bExpPad = expPad != scale;
        auexp /= scale;
        cump_exp_t  rexp = uexp >= 0 ? auexp + bExpPad : -auexp;
        cump_size_t  ausize = bExpPad ? asize + expPad : asize;
        cump_size_t  exSize = ausize % scale;
        cump_size_t  arsize = ausize / scale + (exSize != 0);
        char  *rp_ = static_cast <char*> (static_cast <void*> (rp));

        if (bExpPad)  {(*__cump_memcpy_func) (rp + (arsize - 1), &zero, sizeof (zero));}
        if (arsize > prec)
          {
            up += (arsize - prec) * scale - exSize;
            arsize = prec;
          }
        else
        if (exSize)
          {
            (*__cump_memcpy_func) (rp_, &zero, sizeof (zero));
            rp_ += (arsize * scale - ausize) * sizeof (*up);
          }
        h->_mp_exp = rexp;
        (*__cump_memcpy_func) (rp_, up, asize * sizeof (*up));
        asize = arsize;
      }
    h->_mp_size = size >= 0 ? asize : -asize;
  }
};

template <typename  Dst, typename  Src>
struct Setter<SMALLER, Dst, Src>
{
  static  void  set (Dst  *rp, Src  *up, cumpf_header  *h)
  {
    cump_size_t  prec = h->_mp_prec + 1;	/* lie not to lose precision in assignment */
    cump_size_t  size = h->_mp_size;
    cump_size_t  asize = ABS (size);
    if (asize)
      {
        int const  scale = sizeof (*up) / sizeof (*rp);
        Dst const  zero = 0;

        cump_exp_t  uexp = h->_mp_exp;
        cump_exp_t  rexp = uexp * scale;
        cump_size_t  ausize = asize;
        cump_size_t  arsize = ausize * scale;
        Dst  d [scale];
        char const  *up_ = static_cast <char const *> (static_cast <void const *> (up));

        (*__cump_memcpy_func) (d, up + (ausize - 1), sizeof (d));
          {
            Dst  *dp = d + (scale - 1);
            if (*dp == zero)
              {
                int  i = 1;
                while (*--dp == zero)  {++i;}
                arsize -= i;
                rexp -= i;
              }
          }
        (*__cump_memcpy_func) (d, up, sizeof (d));
          {
            Dst  *dp = d;
            if (*dp == zero)
              {
                int  i = 1;
                while (*++dp == zero)  {++i;}
                arsize -= i;
                up_ += i * sizeof (*rp);
              }
          }
        if (arsize > prec)
          {
            up_ += (arsize - prec) * sizeof (*rp);
            arsize = prec;
          }
        h->_mp_exp = rexp;
        (*__cump_memcpy_func) (rp, up_, arsize * sizeof (*rp));
        asize = arsize;
      }
    h->_mp_size = size >= 0 ? asize : -asize;
  }
};

template <typename  Dst, typename  Src>
struct Translater :
  Setter
  < (sizeof (Dst) > sizeof (Src) ? LARGER  :
     sizeof (Dst) < sizeof (Src) ? SMALLER : EQUAL)
  , Dst, Src
  >
{};

inline  ::mp_ptr     getLimbs (::mpf_ptr     x)  {return  x->_mp_d;}
inline  ::mp_srcptr  getLimbs (::mpf_srcptr  x)  {return  x->_mp_d;}

inline  ::cump_ptr  getLimbs (::cumpf_ptr  x)
{
  void  *p = static_cast <char*> (x->_dev) + sizeof (::cumpf_header);
  return  static_cast < ::cump_ptr> (p);
}

inline  ::cump_srcptr  getLimbs (::cumpf_srcptr  x)
{
  void const  *p = static_cast <char const *> (x->_dev) + sizeof (::cumpf_header);
  return  static_cast < ::cump_srcptr> (p);
}


} // namespce


extern "C"
{


void  __cumpf_set (::cumpf_ptr  r, ::cumpf_srcptr  u, ::cumpf_header  *h)
{
  Translater < ::cump_limb_t, ::cump_limb_t const>::set (getLimbs (r), getLimbs (u), h);
}

void  __cumpf_set_mpf (::cumpf_ptr  r, ::mpf_srcptr  u, ::cumpf_header  *h)
{
  Translater < ::cump_limb_t, ::mp_limb_t const>::set (getLimbs (r), getLimbs(u), h);
}

void  __mpf_set_cumpf (::mpf_ptr  r, ::cumpf_srcptr  u, ::cumpf_header  *h)
{
  Translater < ::mp_limb_t, ::cump_limb_t const>::set (getLimbs (r), getLimbs (u), h);
}


} // extern "C"
