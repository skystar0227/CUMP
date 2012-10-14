/* Definitions and implementations for CUMP device functions.

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


#ifndef CUMP_CUH_


#define CUMP_FUNCTYPE  __device__
#define CUMP_HOSTCTOR  __host__
#define CUMP_CONSTANT  __constant__


#include <cstddef>  // std::ptrdiff_t, std::size_t
#include "cump.h"
#include "def.h"


#ifndef __CUMP_WITHIN_CUMP
CUMP_CONSTANT  cump_size_t  __cump_default_fp_limb_precision = __CUMPF_BITS_TO_PREC (53);
#endif  // __CUMP_WITHIN_CUMP


namespace  // CUMP's classes and functions for device code are always in anonymous namespace
{
namespace cump
{


#include "pointer_traits.hpp"
#include "utility.cuh"
#include "strideptr.cuh"


static  std::size_t const  kLimbBits = CUMP_LIMB_BITS;
static  std::size_t const  kNailBits = CUMP_NAIL_BITS;
static  std::size_t const  kNumBits  = CUMP_NUMB_BITS;
static  std::size_t const  kNumMask  = CUMP_NUMB_MASK;
static  std::size_t const  kNumMax   = CUMP_NUMB_MAX;
static  std::size_t const  kNailMask = CUMP_NAIL_MASK;
static  std::size_t const  kFloatMaxPrec = 1040;


using ::cump_int32;
using ::cump_uint32;
using ::cump_int64;
using ::cump_uint64;

using ::cumpf_ptr;
using ::cumpf_srcptr;
using ::cumpf_array_ptr;
using ::cumpf_array_srcptr;

typedef  ::cump_limb_t    mp_limb_t;
typedef  ::cump_bitcnt_t  mp_bitcnt_t;
typedef  ::cump_size_t    mp_size_t;
typedef  ::cump_exp_t     mp_exp_t;


namespace interface
{


template <typename Limbs>
class Storage
{
  typedef  utility::PointerTraits <Limbs>  PT_;
  typedef  Storage <typename PT_::RemoveConstPointer>  Storage_;

  friend  class Storage <typename PT_::ConstPointer>;

  template <typename, template <typename, typename>  class, typename, typename>
  friend  class Array;

 protected:
  CUMP_HOSTCTOR CUMP_FUNCTYPE  Storage (typename PT_::Parameter  p)
    : p_ (p)
  {}

  CUMP_FUNCTYPE  Storage () : p_ ()  {}

  CUMP_FUNCTYPE  typename PT_::Parameter  get () const  {return  p_;}

 public:
  CUMP_FUNCTYPE  Storage (Storage_ const  &x) : p_ (x.p_)  {}

  template <typename T>  CUMP_FUNCTYPE  bool  operator == (Storage <T>) const  {return  false;}
  template <typename T>  CUMP_FUNCTYPE  bool  operator != (Storage <T>) const  {return  true;}
  CUMP_FUNCTYPE  bool  operator == (Storage <Limbs> const  &x) const  {return  p_ == x.p_;}
  CUMP_FUNCTYPE  bool  operator != (Storage <Limbs> const  &x) const  {return  p_ != x.p_;}

  CUMP_FUNCTYPE  void  swap (Storage  &x)  {utility::swap (p_, x.p_);}

 private:
  Limbs  p_;
};  // Storage


template
< typename HostPtr
, typename Limb = mp_limb_t
, typename Limbs = Limb*
>
class Variable
  : public Storage <Limbs>
{
  typedef  Storage <Limbs>  Storage_;

 public:
  CUMP_HOSTCTOR  Variable (HostPtr  p)
    : Storage_ (static_cast <Limb*> (p->_dev))
  {}

  CUMP_FUNCTYPE  Variable () : Storage_ ()  {}
};  // Variable


template <typename, typename>  struct Common;
template <typename, typename>  struct Transposed;


template
< typename HostPtr
, template <typename, typename>  class Layout = Transposed
, typename Limb = mp_limb_t
, typename Index = std::ptrdiff_t
>
class Array
  : public Layout <Limb, Index>
{
  typedef  Storage <typename Layout <Limb, Index>::Pointer>  Element_;

 public:
  CUMP_HOSTCTOR  Array (HostPtr  p)
    : p_ (static_cast <Limb*> (p->_dev))
    , l_ (p->_int/sizeof (Limb))
  {}

  CUMP_FUNCTYPE  Element_  operator [] (Index  n) const  {return  at (p_, l_, n);}

 private:
  Limb  *p_;
  Index  l_;
};  // Array


template <typename Limb, typename Index>
struct Common
{
  typedef  Limb*  Pointer;

  CUMP_FUNCTYPE static  Pointer  at (Limb  *p, Index  l, Index  n)  {return  p + l * n;}
};  // Common


template <typename Limb, typename Index>
struct Transposed
{
  typedef  utility::StridePointer <Limb*>  Pointer;

  CUMP_FUNCTYPE static  Pointer  at (Limb  *p, Index  l, Index  n)  {return  Pointer (p + n, l);}
};  // Transposed


}  // namespace interface


namespace utility
{


template <typename Limbs>
CUMP_FUNCTYPE  void  swap (interface::Storage <Limbs>  &x, interface::Storage <Limbs>  &y)
{
  x.swap (y);
}  // swap ()


}  // namespace utility


typedef  interface::Variable <cumpf_srcptr>  mpf_t;
typedef  interface::Array <cumpf_array_srcptr>  mpf_array_t;
typedef  interface::Array <cumpf_array_srcptr, interface::Common>  mpf_common_array_t;
typedef  interface::Array <cumpf_array_srcptr, interface::Transposed>  mpf_transposed_array_t;
typedef
  interface::Variable <cumpf_srcptr, mp_limb_t const>
  mpf_src_t;
typedef
  interface::Array <cumpf_array_srcptr, interface::Transposed, mp_limb_t const>
  mpf_array_src_t;
typedef
  interface::Array <cumpf_array_srcptr, interface::Common, mp_limb_t const>
  mpf_common_array_src_t;
typedef
  interface::Array <cumpf_array_srcptr, interface::Transposed, mp_limb_t const>
  mpf_transposed_array_src_t;


#define ASSERT(expr)
//#define CUMPN_SAME_OR_SEPARATE_P
//#define CUMPN_SAME_OR_SEPARATE2_P
//#define CUMPN_SAME_OR_INCR_P
//#define CUMPN_SAME_OR_DECR_P
//#define CUMPN_OVERLAP_P


// TMP schemes
#define TMP_SDECL      TMP_DECL
#define TMP_DECL       mp_limb_t  tmp [kFloatMaxPrec]
#define TMP_SMARK      TMP_MARK
#define TMP_MARK
#define TMP_SALLOC(n)  TMP_ALLOC(n)
#define TMP_BALLOC(n)  TMP_ALLOC(n)
#define TMP_ALLOC(n)   tmp
#define TMP_SFREE      TMP_FREE
#define TMP_FREE

/* Allocating various types. */
//#define TMP_ALLOC_TYPE(n,type)   ((type*)TMP_ALLOC((n)*sizeof(type)))
//#define TMP_SALLOC_TYPE(n,type)  ((type*)TMP_SALLOC((n)*sizeof(type)))
//#define TMP_BALLOC_TYPE(n,type)  ((type*)TMP_BALLOC((n)*sizeof(type)))
#define TMP_ALLOC_LIMBS(n)       TMP_ALLOC(n)
#define TMP_SALLOC_LIMBS(n)      TMP_SALLOE(n)
#define TMP_BALLOC_LIMBS(n)      TMP_BALLOC(n)
//#define TMP_ALLOC_MP_PTRS(n)     TMP_ALLOC_TYPE(n,mp_ptr)
//#define TMP_SALLOC_MP_PTRS(n)    TMP_SALLOC_TYPE(n,mp_ptr)
//#define TMP_BALLOC_MP_PTRS(n)    TMP_BALLOC_TYPE(n,mp_ptr)


#define CUMP_1_PTRTYPES  typename P1
#define CUMP_2_PTRTYPES  CUMP_1_PTRTYPES, typename P2
#define CUMP_3_PTRTYPES  CUMP_2_PTRTYPES, typename P3

#define CUMP_FUNCTMPL1  template <CUMP_1_PTRTYPES> CUMP_FUNCTYPE
#define CUMP_FUNCTMPL2  template <CUMP_2_PTRTYPES> CUMP_FUNCTYPE
#define CUMP_FUNCTMPL3  template <CUMP_3_PTRTYPES> CUMP_FUNCTYPE


#define mp_ptr(n)     P##n
#define mp_srcptr(n)  P##n


namespace mpn
{


CUMP_FUNCTMPL2  void  copy (mp_ptr(1)  dst, mp_srcptr(2)  src, mp_size_t  n)
{
  if (n)
    {
      mp_limb_t  x = *src;
      while (--n)
        {
          *dst++ = x;
          x = *++src;
        }
      *dst = x;
    }
}  // copy ()


CUMP_FUNCTMPL2
void  copy_rest (mp_ptr(1)  dst, mp_srcptr(2)  src, mp_size_t  size,  mp_size_t  start)
{
  ASSERT (size>=0);
  ASSERT (start>=0);
  ASSERT (start<=size);
  ASSERT (CUMPN_SAME_OR_SEPARATE_P (dst, src, size));
  // __CUMP_CUCRAY_Pragma ("_CRI ivdep");
  for (mp_size_t  j = start;  j < size;  ++j)  {dst [j] = src [j];}
}  // copy_rest ()


CUMP_FUNCTMPL1  void  zero (mp_ptr(1)  dst, mp_size_t  n)
{
  ASSERT (n>=0);
  if (n)  {*dst=0;  while (--n)  {*++dst=0;}}
}  // zero ()


CUMP_FUNCTMPL1  void  swap_ptr (mp_ptr(1)  &xp, mp_size_t  &xs, mp_ptr(1)  &yp, mp_size_t  &ys)
{
  using utility::swap;
  swap (xp, yp);
  swap (xs, ys);
}  // swap_ptr ()


CUMP_FUNCTMPL1
void  swap_srcptr (mp_srcptr(1)  &xp, mp_size_t  &xs, mp_srcptr(1)  &yp, mp_size_t  &ys)
{
  using utility::swap;
  swap (xp, yp);
  swap (xs, ys);
}  // swap_srcptr ()


template
< bool  (&test) (mp_limb_t&, mp_limb_t)
, class  AorS
, CUMP_3_PTRTYPES
>
CUMP_FUNCTYPE
int  aors (mp_ptr(1)  wp, mp_srcptr(2)  xp, mp_size_t  xsize, mp_srcptr(3)  yp, mp_size_t  ysize)
{
  ASSERT (ysize >= 0);
  ASSERT (xsize >= ysize);
  ASSERT (CUMPN_SAME_OR_SEPARATE2_P (wp, xsize, xp, xsize));
  ASSERT (CUMPN_SAME_OR_SEPARATE2_P (wp, xsize, yp, ysize));
  mp_size_t  i = ysize;
  if (i != 0)
    {
      mp_limb_t  x;
      if (AorS::func (wp, xp, yp, i))
        {
          do
            {
              if (i >= xsize)  {return  1;}
              x = xp [i];
            }
          while (test (wp [i++], x));
        }
    }
  if (wp != xp)  {copy_rest (wp, xp, xsize, i);}
  return  0;
}  // aors ()


template
< mp_limb_t  (&op) (mp_limb_t, mp_limb_t)
, bool  (&cb) (mp_limb_t, mp_limb_t, mp_limb_t)
, CUMP_2_PTRTYPES
>
CUMP_FUNCTYPE  int  aors_1 (mp_ptr(1)  dst, mp_srcptr(2)  src, mp_size_t  n, mp_limb_t  v)
{
  ASSERT (n>=1);
  ASSERT (CUMPN_SAME_OR_SEPARATE_P (dst, src, n));
  mp_limb_t  x = src [0];
  mp_limb_t  r = op (x, v);
  dst [0] = (kNailBits == 0 ? r : r & kNumMask);
  if (kNailBits == 0 ? cb (r, x, v) : r >> kNumBits == 0)
    {
      for (mp_size_t  i=1; i<n;)
        {
          x = src [i];
          r = op (x, 1);
          dst [i] = (kNailBits == 0 ? r : r & kNumMask);
          ++i;
          if (!(kNailBits == 0 ? cb (r, x, 1) : r >> kNumBits == 0))
            {
              if (src != dst)  {copy_rest (dst, src, n, i);}
              return  0;
            }
        }
      return  1;
    }
  else
    {
      if (src != dst)  {copy_rest (dst, src, n, 1);}
      return  0;
    }
}


CUMP_FUNCTMPL2  void  copyi (mp_ptr(1), mp_srcptr(2), mp_size_t);
CUMP_FUNCTMPL2  void  copyd (mp_ptr(1), mp_srcptr(2), mp_size_t);
CUMP_FUNCTMPL2  mp_limb_t  neg (mp_ptr(1), mp_srcptr(2), mp_size_t);
CUMP_FUNCTMPL2  mp_limb_t  rshift  (mp_ptr(1), mp_srcptr(2), mp_size_t, unsigned int);
CUMP_FUNCTMPL2  mp_limb_t  lshift  (mp_ptr(1), mp_srcptr(2), mp_size_t, unsigned int);
CUMP_FUNCTMPL2  mp_limb_t  lshiftc (mp_ptr(1), mp_srcptr(2), mp_size_t, unsigned int);
CUMP_FUNCTMPL3  mp_limb_t  add   (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL2  mp_limb_t  add_1 (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_limb_t) __CUMP_NOTHROW;
CUMP_FUNCTMPL3  mp_limb_t  add_n (mp_ptr(1), mp_srcptr(2), mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL3  mp_limb_t  sub   (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL2  mp_limb_t  sub_1 (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_limb_t) __CUMP_NOTHROW;
CUMP_FUNCTMPL3  mp_limb_t  sub_n (mp_ptr(1), mp_srcptr(2), mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL3  mp_limb_t  addmul   (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_srcptr(3));
CUMP_FUNCTMPL2  mp_limb_t  addmul_1 (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_limb_t);
CUMP_FUNCTMPL3  mp_limb_t  mul   (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL2  mp_limb_t  mul_1 (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_limb_t);
CUMP_FUNCTMPL3  mp_limb_t  mul_2 (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_srcptr(3));
CUMP_FUNCTMPL3  void  mul_n (mp_ptr(1), mp_srcptr(2), mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL3  void  mul_basecase (mp_ptr(1), mp_srcptr(2), mp_size_t, mp_srcptr(3), mp_size_t);
CUMP_FUNCTMPL2  void  sqr (mp_ptr(1), mp_srcptr(2), mp_size_t);
CUMP_FUNCTMPL2  void  sqr_basecase (mp_ptr(1), mp_srcptr(2), mp_size_t);
CUMP_FUNCTMPL2  int  cmp (mp_srcptr(1), mp_srcptr(2), mp_size_t) __CUMP_NOTHROW;


/* mpn functions */
#include "mpn/generic/add.impl"
#include "mpn/generic/add_1.impl"
#include "mpn/generic/add_n.impl"
#include "mpn/generic/addmul_1.impl"
#include "mpn/generic/cmp.impl"
#include "mpn/generic/copyd.impl"
#include "mpn/generic/copyi.impl"
#include "mpn/generic/lshift.impl"
#include "mpn/generic/lshiftc.impl"
#include "mpn/generic/mul_1.impl"
#include "mpn/CUDA/addmul.impl"
#include "mpn/CUDA/mul_2.impl"
#include "mpn/CUDA/mul_basecase.impl"
//#include "mpn/generic/mul_basecase.impl"
#include "mpn/generic/mul.impl"
#include "mpn/generic/neg.impl"
#include "mpn/generic/rshift.impl"
#include "mpn/generic/sub.impl"
#include "mpn/generic/sub_1.impl"
#include "mpn/generic/sub_n.impl"


}  // namespace mpn


#define CUMPN_COPY         mpn::copy
#define CUMPN_ZERO         mpn::zero
#define CUMPN_PTR_SWAP     mpn::swap_ptr
#define CUMPN_SRCPTR_SWAP  mpn::swap_srcptr


namespace mpf
{


using utility::swap;


template <typename Limbs>
class ConstFloat
  : public interface::Storage <Limbs>
{
  typedef  interface::Storage <Limbs>  Storage_;
  typedef  utility::PointerTraits <Limbs>  PT_;
  typedef  cump_int32  Int_;
  typedef  mp_exp_t  Exp_;

  using Storage_::get;

 public:
  template <typename Limbs_>
  CUMP_FUNCTYPE  ConstFloat (interface::Storage <Limbs_> const  &x)
    : Storage_ (x)
  {}

  CUMP_FUNCTYPE  Int_  _mp_prec () const  {return  *ptrcast_ <Int_> (get ());}

  CUMP_FUNCTYPE  Int_  _mp_size () const
  {
    return  *ptrcast_ <Int_> (increment_by_ <sizeof (Int_)> (get ()));
  }

  CUMP_FUNCTYPE  Exp_  _mp_exp () const
  {
    return  *ptrcast_ <Exp_> (increment_by_ <2 * sizeof (Int_)> (get ()));
  }

  CUMP_FUNCTYPE  Limbs  _mp_d () const
  {
    return  increment_and_align_ <2 * sizeof (Int_) + sizeof (Exp_)> (get ());
  }

 private:
  template <typename T>
  CUMP_FUNCTYPE static  T const *  ptrcast_ (void const  *p)
  {
    return  static_cast <T const *> (p);
  }  // ptrcast_ ()

  template <std::size_t  bytes>
  CUMP_FUNCTYPE static  void const *  increment_by_ (typename PT_::Parameter  p)
  {
    std::size_t const  x = bytes / sizeof (*p);
    std::size_t const  y = bytes % sizeof (*p);
    return  ptrcast_ <char> (x == 0 ? p : p + x) + y;
  }  // increment_by_ ()

  template <std::size_t  bytes>
  CUMP_FUNCTYPE static  Limbs  increment_and_align_ (typename PT_::Parameter  p)
  {
    std::size_t const  x = bytes / sizeof (*p);
    std::size_t const  y = bytes % sizeof (*p);
    std::size_t const  offset = x + (y == 0 ? 0l : 1l);
    return  p + offset;
  }  // increment_and_align_ ()
};  // ConstFloat


template <typename Limbs>
class Float
  : public interface::Storage <Limbs>
{
  typedef  interface::Storage <Limbs>  Storage_;
  typedef  utility::PointerTraits <Limbs>  PT_;
  typedef  cump_int32  Int_;
  typedef  mp_exp_t  Exp_;

  using Storage_::get;

 public:
  template <typename Limbs_>
  CUMP_FUNCTYPE  Float (interface::Storage <Limbs_> const  &x)
    : Storage_ (x)
  {}

  CUMP_FUNCTYPE  Int_&  _mp_prec ()  {return  *ptrcast_ <Int_> (get ());}

  CUMP_FUNCTYPE  Int_&  _mp_size ()
  {
    return  *ptrcast_ <Int_> (increment_by_ <sizeof (Int_)> (get ()));
  }

  CUMP_FUNCTYPE  Exp_&  _mp_exp ()
  {
    return  *ptrcast_ <Exp_> (increment_by_ <2 * sizeof (Int_)> (get ()));
  }

  CUMP_FUNCTYPE  Limbs  _mp_d ()
  {
    return  increment_and_align_ <2 * sizeof (Int_) + sizeof (Exp_)> (get ());
  }

 private:
  template <typename T>
  CUMP_FUNCTYPE static  T*  ptrcast_ (void  *p)  {return  static_cast <T*> (p);}

  template <std::size_t  bytes>
  CUMP_FUNCTYPE static  void*  increment_by_ (typename PT_::Parameter  p)
  {
    std::size_t const  x = bytes / sizeof (*p);
    std::size_t const  y = bytes % sizeof (*p);
    return  ptrcast_ <char> (x == 0 ? p : p + x) + y;
  }  // increment_by_ ()

  template <std::size_t  bytes>
  CUMP_FUNCTYPE static  Limbs  increment_and_align_ (typename PT_::Parameter  p)
  {
    std::size_t const  x = bytes / sizeof (*p);
    std::size_t const  y = bytes % sizeof (*p);
    std::size_t const  offset = x + (y == 0 ? 0l : 1l);
    return  p + offset;
  }  // increment_and_align_ ()
};  // Float


#define float_ptr(n)     mpf::Float <P##n>
#define float_srcptr(n)  mpf::ConstFloat <P##n>


CUMP_FUNCTYPE  void  clear (mpf_t&);
CUMP_FUNCTYPE  void  init (mpf_t&);
CUMP_FUNCTYPE  void  init2 (mpf_t&, mp_bitcnt_t);
CUMP_FUNCTMPL1  void  init_set (mpf_t&, float_srcptr(1));
CUMP_FUNCTMPL2  void  set (float_ptr(1), float_srcptr(2));
CUMP_FUNCTMPL2  void  neg (float_ptr(1), float_srcptr(2));

template <bool, CUMP_3_PTRTYPES>
CUMP_FUNCTYPE  void  add (float_ptr(1), float_srcptr(2), float_srcptr(3));

template <bool, CUMP_3_PTRTYPES>
CUMP_FUNCTYPE  void  sub (float_ptr(1), float_srcptr(2), float_srcptr(3));

template <bool, CUMP_2_PTRTYPES>
CUMP_FUNCTYPE  void  add (float_ptr(1), float_srcptr(2), float_srcptr(2));

template <bool, CUMP_2_PTRTYPES>
CUMP_FUNCTYPE  void  sub (float_ptr(1), float_srcptr(2), float_srcptr(2));

CUMP_FUNCTMPL3  void  mul (float_ptr(1), float_srcptr(2), float_srcptr(3));


}  // namespace mpf


#define mpf_ptr(n)     interface::Storage <P##n>
#define mpf_srcptr(n)  interface::Storage <P##n> const &


CUMP_FUNCTYPE  void  mpf_clear (mpf_t&);
CUMP_FUNCTYPE  void  mpf_init (mpf_t&);
CUMP_FUNCTYPE  void  mpf_init2 (mpf_t&, mp_bitcnt_t);
CUMP_FUNCTMPL1  void  mpf_init_set (mpf_t&, mpf_srcptr(1));
CUMP_FUNCTMPL2  void  mpf_set (mpf_ptr(1), mpf_srcptr(2));
CUMP_FUNCTMPL2  void  mpf_neg (mpf_ptr(1), mpf_srcptr(2));
CUMP_FUNCTMPL3  void  mpf_add (mpf_ptr(1), mpf_srcptr(2), mpf_srcptr(3));
CUMP_FUNCTMPL3  void  mpf_sub (mpf_ptr(1), mpf_srcptr(2), mpf_srcptr(3));
CUMP_FUNCTMPL3  void  mpf_mul (mpf_ptr(1), mpf_srcptr(2), mpf_srcptr(3));


#ifndef __CUMP_WITHIN_CUMP
/* mpf functions */
//#include "mpf/clear.impl"
//#include "mpf/init.impl"
//#include "mpf/init2.impl"
//#include "mpf/iset.impl"
#include "mpf/set.impl"
#include "mpf/neg.impl"
#include "mpf/add.impl"
#include "mpf/sub.impl"
#include "mpf/mul.impl"
#endif  // __CUMP_WITHIN_CUMP


#undef mpf_srcptr
#undef mpf_ptr


#undef float_srcptr
#undef float_ptr


#undef CUMPN_COPY
#undef CUMPN_ZERO
#undef CUMPN_PTR_SWAP
#undef CUMPN_SRCPTR_SWAP


#undef mp_srcptr
#undef mp_ptr


#undef CUMP_FUNCTMPL3
#undef CUMP_FUNCTMPL2
#undef CUMP_FUNCTMPL1

#undef CUMP_3_PTRTYPES
#undef CUMP_2_PTRTYPES
#undef CUMP_1_PTRTYPES


//#undef TMP_BALLOC_MP_PTRS
//#undef TMP_SALLOC_MP_PTRS
//#undef TMP_ALLOC_MP_PTRS
#undef TMP_BALLOC_LIMBS
#undef TMP_SALLOC_LIMBS
#undef TMP_ALLOC_LIMBS
//#undef TMP_BALLOC_TYPE
//#undef TMP_SALLOC_TYPE
//#undef TMP_ALLOC_TYPE

#undef TMP_FREE
#undef TMP_SFREE
#undef TMP_ALLOC
#undef TMP_BALLOC
#undef TMP_SALLOC
#undef TMP_MARK
#undef TMP_SMARK
#undef TMP_DECL
#undef TMP_SDECL


//#undef CUMPN_OVERLAP_P
//#undef CUMPN_SAME_OR_DECR_P
//#undef CUMPN_SAME_OR_INCR_P
//#undef CUMPN_SAME_OR_SEPARATE2_P
//#undef CUMPN_SAME_OR_SEPARATE_P
#undef ASSERT


}  // namespace cump
}  // namespace


#undef CUMP_CONSTANT
#undef CUMP_HOSTCTOR
#undef CUMP_FUNCTYPE


#define CUMP_CUH_
#endif
