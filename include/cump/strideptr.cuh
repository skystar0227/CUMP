/* StridePointer -- a pointer wrapper class for coalessed access.

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


#ifndef  CUMP_STRIDEPTR_CUH_


namespace utility
{


template
< typename P
, typename D = std::ptrdiff_t
>
class StridePointer
{
  typedef  PointerTraits <P>  PT_;

  friend  struct StridePointer <typename PT_::ConstPointer, D>;
  friend  struct PointerTraits <StridePointer>;

  // typedefs for PointerTraits
  typedef  typename PT_::Pointee  Pointee;
  typedef  StridePointer <typename PT_::ConstPointer, D>  ConstPointer;
  typedef  StridePointer <typename PT_::RemoveConstPointer, D>  RemoveConstPointer;

 public:
  CUMP_FUNCTYPE  StridePointer ()
    : p_ (), l_ ()
  {}

  CUMP_FUNCTYPE explicit  StridePointer (typename PT_::Parameter  p, D  l = 1)
    : p_ (p), l_ (l)
  {}

  CUMP_FUNCTYPE  StridePointer (RemoveConstPointer const  &x)
    : p_ (x.p_), l_ (x.l_)
  {}

  CUMP_FUNCTYPE  operator typename PT_::Parameter () const  {return  p_;}

  CUMP_FUNCTYPE  Pointee&  operator * () const  {return  *p_;}

  CUMP_FUNCTYPE  StridePointer&  operator ++ ()
  {
    p_ += l_;
    return  *this;
  }

  CUMP_FUNCTYPE  StridePointer&  operator -- ()
  {
    p_ -= l_;
    return  *this;
  }

  CUMP_FUNCTYPE  StridePointer&  operator += (D  n)
  {
    p_ += l_ * n;
    return  *this;
  }

  CUMP_FUNCTYPE  StridePointer&  operator -= (D  n)
  {
    p_ -= l_ * n;
    return  *this;
  }

  template <typename T>  CUMP_FUNCTYPE  bool  operator == (T*) const  {return  false;}
  CUMP_FUNCTYPE  bool  operator == (StridePointer const  &x) const  {return  p_ == x.p_;}
  CUMP_FUNCTYPE  bool  operator < (StridePointer const  &x) const  {return  p_ < x.p_;}

  CUMP_FUNCTYPE  D  operator - (StridePointer const  &x) const  {return  p_ - x.p_;}



  CUMP_FUNCTYPE  P  operator -> ()  {return  p_;}
  CUMP_FUNCTYPE  typename PT_::Parameter  operator -> () const  {return  p_;}
  CUMP_FUNCTYPE  Pointee&  operator [] (D  n) const  {return  p_ [l_ * n];}

  CUMP_FUNCTYPE  StridePointer  operator ++ (int)
  {
    StridePointer  tmp (*this);
    p_ += l_;
    return  tmp;
  }

  CUMP_FUNCTYPE  StridePointer  operator -- (int)
  {
    StridePointer  tmp (*this);
    p_ -= l_;
    return  tmp;
  }

  CUMP_FUNCTYPE  StridePointer  operator + (D  n) const
  {
    StridePointer  tmp (*this);
    tmp += n;
    return  tmp;
  }

  CUMP_FUNCTYPE  StridePointer  operator - (D  n) const
  {
    StridePointer  tmp (*this);
    tmp -= n;
    return  tmp;
  }

  template <typename T>  CUMP_FUNCTYPE  bool  operator != (T*) const  {return  true;}
  CUMP_FUNCTYPE  bool  operator != (StridePointer const  &x) const  {return  p_ != x.p_;}
  CUMP_FUNCTYPE  bool  operator > (StridePointer const  &x) const  {return  p_ > x.p_;}
  CUMP_FUNCTYPE  bool  operator <= (StridePointer const  &x) const  {return  p_ <= x.p_;}
  CUMP_FUNCTYPE  bool  operator >= (StridePointer const  &x) const  {return  p_ >= x.p_;}



  CUMP_FUNCTYPE  void  swap (StridePointer  &x)
  {
    using utility::swap;
    swap (p_, x.p_);
    swap (l_, x.l_);
  }

 private:
  P  p_;
  D  l_;
};  // StridePointer


template <typename P, typename D>
CUMP_FUNCTYPE  void  swap (StridePointer <P, D>  &x, StridePointer <P, D>  &y)  {x.swap (y);}


}  // namespace utility


#define  CUMP_STRIDEPTR_CUH_
#endif
