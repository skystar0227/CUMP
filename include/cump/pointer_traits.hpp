/* PointerTraits -- a class template with typedefs about template parameter of
   a pointer type.

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


#ifndef  CUMP_POINTER_TRAITS_HPP_


namespace utility
{


namespace _
{


template <typename T>
struct PointerTraitsImpl
{
  typedef  T  Pointee;
  typedef  T  *Pointer;
  typedef  T const  *ConstPointer;
  typedef  T* const  Parameter;
};  // PointerTraitsImpl


}  // namespace _


template <typename>  struct PointerTraits;


template <typename P>  struct PointerTraits <P const> : PointerTraits <P>  {};
template <typename P>  struct PointerTraits <P&> : PointerTraits <P>  {};


template <typename T>
struct PointerTraits <T*>
  : _::PointerTraitsImpl <T>
{
  typedef  T  *RemoveConstPointer;
};  // PointerTraits <T*>


template <typename T>
struct PointerTraits <T const *>
  : _::PointerTraitsImpl <T const>
{
  typedef  T  *RemoveConstPointer;
};  // PointerTraits <T const *>


template <typename P>
struct PointerTraits
{
  typedef  typename P::Pointee  Pointee;
  typedef  P  Pointer;
  typedef  typename P::ConstPointer  ConstPointer;
  typedef  P const  &Parameter;

  typedef  typename P::RemoveConstPointer  RemoveConstPointer;
};  // PointerTraits


}  // namespace utility


#define  CUMP_POINTER_TRAITS_HPP_
#endif
