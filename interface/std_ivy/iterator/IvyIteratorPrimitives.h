#ifndef IVYITERATORPRIMITIVES_H
#define IVYITERATORPRIMITIVES_H


#include "IvyBasicTypes.h"
#include "std_ivy/IvyTypeTraits.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  // Iterator primitives
  struct input_iterator_tag{};
  struct output_iterator_tag{};
  struct forward_iterator_tag : public input_iterator_tag{};
  struct bidirectional_iterator_tag : public forward_iterator_tag{};
  struct random_access_iterator_tag : public bidirectional_iterator_tag{};
  struct contiguous_iterator_tag : public random_access_iterator_tag{};
  using stashing_iterator_tag = void; // Dummy tag to recognize iterators that cannot be reversed (CUDA-style solution)

  // Base class for iterators
  template<typename Category, typename T, typename Distance = IvyTypes::ptrdiff_t, typename Pointer = T*, typename Reference = T&> struct iterator{
    using value_type = T;
    using pointer = Pointer;
    using reference = Reference;
    using difference_type = Distance;
    using iterator_category = Category;
  };

  // For the reverse iterator implementation (CUDA-style solution)
  template<typename T, typename = void> struct stashing_iterator : std_ttraits::false_type{};
  template<typename T> struct stashing_iterator<T, std_ttraits::void_t<typename T::stashing_iterator_tag>> : std_ttraits::true_type{};
  template<typename T> inline constexpr bool stashing_iterator_v = stashing_iterator<T>::value;
}

#endif


#endif
