#ifndef IVYFIND_H
#define IVYFIND_H


#ifdef __USE_CUDA__

#include "std_ivy/IvyIterator.h"


namespace std_ivy{
  template<typename Iterator, typename T> __HOST_DEVICE__ constexpr Iterator find(Iterator first, Iterator last, T const& v){
    typename std_iter::iterator_traits<Iterator>::difference_type count = std_iter::distance(first, last) >> 2;
    for (; count > 0; --count){
      if (*first == v) return first;
      ++first;
      if (*first == v) return first;
      ++first;
      if (*first == v) return first;
      ++first;
      if (*first == v) return first;
      ++first;
    }
    switch (std_iter::distance(first, last)){
    case 3:
      if (*first == v) return first;
      ++first;
    case 2:
      if (*first == v) return first;
      ++first;
    case 1:
      if (*first == v) return first;
      ++first;
    case 0:
    default:
      return last;
    }
  }

  template<typename Iterator, class UnaryPredicate> __HOST_DEVICE__ constexpr Iterator find_if(Iterator first, Iterator last, UnaryPredicate p){
    typename std_iter::iterator_traits<Iterator>::difference_type count = std_iter::distance(first, last) >> 2;
    for (; count > 0; --count){
      if (p(*first)) return first;
      ++first;
      if (p(*first)) return first;
      ++first;
      if (p(*first)) return first;
      ++first;
      if (p(*first)) return first;
      ++first;
    }
    switch (std_iter::distance(first, last)){
    case 3:
      if (p(*first)) return first;
      ++first;
    case 2:
      if (p(*first)) return first;
      ++first;
    case 1:
      if (p(*first)) return first;
      ++first;
    case 0:
    default:
      return last;
    }
  }

  template<typename Iterator, class UnaryPredicate> __HOST_DEVICE__ constexpr Iterator find_if_not(Iterator first, Iterator last, UnaryPredicate p){
    typename std_iter::iterator_traits<Iterator>::difference_type count = std_iter::distance(first, last) >> 2;
    for (; count > 0; --count){
      if (!p(*first)) return first;
      ++first;
      if (!p(*first)) return first;
      ++first;
      if (!p(*first)) return first;
      ++first;
      if (!p(*first)) return first;
      ++first;
    }
    switch (std_iter::distance(first, last)){
    case 3:
      if (!p(*first)) return first;
      ++first;
    case 2:
      if (!p(*first)) return first;
      ++first;
    case 1:
      if (!p(*first)) return first;
      ++first;
    case 0:
    default:
      return last;
    }
  }
}

#endif


#endif
