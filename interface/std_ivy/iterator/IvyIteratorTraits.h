/**
 * @file IvyIteratorTraits.h
 * @brief Iterator trait helpers and aliases for std_ivy iterator categories.
 */
#ifndef IVYITERATORTRAITS_H
#define IVYITERATORTRAITS_H


#include "IvyBasicTypes.h"
#include "std_ivy/iterator/IvyIteratorPrimitives.h"


namespace std_ivy{
  /**
   * @brief Primary iterator traits template extracting nested iterator typedefs.
   * @tparam Iterator Iterator type providing STL-like member typedefs.
   */
  template<typename Iterator> struct iterator_traits{
    /** @brief Iterator value type. */
    typedef typename Iterator::value_type value_type;
    /** @brief Iterator pointer type. */
    typedef typename Iterator::pointer pointer;
    /** @brief Iterator reference type. */
    typedef typename Iterator::reference reference;
    /** @brief Iterator distance type. */
    typedef typename Iterator::difference_type difference_type;
    /** @brief Iterator category tag. */
    typedef typename Iterator::iterator_category iterator_category;
  };
  /**
   * @brief Pointer specialization of iterator traits.
   * @tparam T Pointee type.
   */
  template<typename T> struct iterator_traits<T*>{
    /** @brief Iterator value type. */
    typedef T value_type;
    /** @brief Iterator pointer type. */
    typedef T* pointer;
    /** @brief Iterator reference type. */
    typedef T& reference;
    /** @brief Iterator distance type. */
    typedef IvyTypes::ptrdiff_t difference_type;
    /** @brief Iterator category tag for raw pointers. */
    typedef random_access_iterator_tag iterator_category;
  };
}


#endif
