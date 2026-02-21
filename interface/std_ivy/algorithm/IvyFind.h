/**
 * @file IvyFind.h
 * @brief Linear search algorithms for generic iterators in the std_ivy namespace.
 */
#ifndef IVYFIND_H
#define IVYFIND_H


#include "std_ivy/IvyIterator.h"


namespace std_ivy{
  /**
   * @brief Find the first element equal to a target value.
   * @tparam Iterator Iterator type that supports dereference and increment.
   * @tparam T Comparable value type.
   * @param first Beginning of the input range.
   * @param last End of the input range.
   * @param v Target value to match.
   * @return Iterator to the first matching element or @p last if not found.
   */
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

  /**
   * @brief Find the first element satisfying a unary predicate.
   * @tparam Iterator Iterator type that supports dereference and increment.
   * @tparam UnaryPredicate Predicate invocable with dereferenced iterator values.
   * @param first Beginning of the input range.
   * @param last End of the input range.
   * @param p Predicate used for matching.
   * @return Iterator to the first matching element or @p last if not found.
   */
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

  /**
   * @brief Find the first element that does not satisfy a unary predicate.
   * @tparam Iterator Iterator type that supports dereference and increment.
   * @tparam UnaryPredicate Predicate invocable with dereferenced iterator values.
   * @param first Beginning of the input range.
   * @param last End of the input range.
   * @param p Predicate used for filtering.
   * @return Iterator to the first non-matching element or @p last if not found.
   */
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
