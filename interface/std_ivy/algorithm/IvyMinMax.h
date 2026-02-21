/**
 * @file IvyMinMax.h
 * @brief Min/max utility functions and comparators for host/device code.
 */
#ifndef IVYMINMAX_H
#define IVYMINMAX_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyInitializerList.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/iterator/IvyIteratorTraits.h"


namespace std_ivy{
  /**
   * @brief Return the smaller of two values using operator>.
   */
  template<typename T> __HOST_DEVICE__ T const& min(T const& x, T const& y){ return (x>y ? y : x); }
  /**
   * @brief Return the larger of two values using operator>.
   */
  template<typename T> __HOST_DEVICE__ T const& max(T const& x, T const& y){ return (x>y ? x : y); }

  /**
   * @brief Compute minimum and maximum into reference outputs.
   */
  template<typename T> __HOST_DEVICE__ void minmax(T const& x, T const& y, T& __RESTRICT__ i, T& __RESTRICT__ a){ if (y<x){ i=y; a=x; } else{ i=x; a=y; } }
  /**
   * @brief Compute minimum and maximum addresses into pointer outputs.
   */
  template<typename T> __HOST_DEVICE__ void minmax(T const& x, T const& y, T* __RESTRICT__ i, T* __RESTRICT__ a){ if (y<x){ i=&y; a=&x; } else{ i=&x; a=&y; } }
  /**
   * @brief Compute minimum and maximum using a custom comparator into reference outputs.
   */
  template<typename T, typename C> __HOST_DEVICE__ void minmax(T const& x, T const& y, C comp, T& __RESTRICT__ i, T& __RESTRICT__ a){ if (comp(y, x)){ i=y; a=x; } else{ i=x; a=y; } }
  /**
   * @brief Compute minimum and maximum using a custom comparator into pointer outputs.
   */
  template<typename T, typename C> __HOST_DEVICE__ void minmax(T const& x, T const& y, C comp, T* __RESTRICT__ i, T* __RESTRICT__ a){ if (comp(y, x)){ i=&y; a=&x; } else{ i=&x; a=&y; } }

  /**
   * @brief Return the ordered pair of minimum and maximum values.
   */
  template<typename T> __HOST_DEVICE__ __CPP_CONSTEXPR__ std_util::pair<T, T> minmax(T const& x, T const& y){ return (x>y ? std_util::pair<T, T>(y, x) : std_util::pair<T, T>(x, y)); }
  /**
   * @brief Return the ordered pair of minimum and maximum values with a custom comparator.
   */
  template<typename T, typename C> __HOST_DEVICE__ __CPP_CONSTEXPR__ std_util::pair<T, T> minmax(T const& x, T const& y, C comp){
    return (C(y, x) ? std_util::pair<T, T>(y, x) : std_util::pair<T, T>(x, y));
  }


  /**
   * @brief Find iterators to the minimum and maximum elements in a range.
   * @tparam ForwardIt Forward iterator type.
   * @tparam C Comparator type.
   * @param first Beginning of the range.
   * @param last End of the range.
   * @param comp Comparator returning true when first argument is ordered before second.
   * @return Pair of iterators to minimum and maximum elements.
   */
  template<typename ForwardIt, typename C> __HOST_DEVICE__ __CPP_CONSTEXPR__
  std_util::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last, C comp){
    auto min = first, max = first;
    if (first == last || ++first == last) return { min, max };
    if (comp(*first, *min)) min = first;
    else max = first;
    while (++first != last){
      auto next = first;
      if (++next == last){
        if (comp(*first, *min)) min = first;
        else if (!comp(*first, *max)) max = first;
        break;
      }
      else{
        if (comp(*next, *first)){
          if (comp(*next, *min)) min = next;
          if (!comp(*first, *max)) max = first;
        }
        else{
          if (comp(*first, *min)) min = first;
          if (!comp(*next, *max)) max = next;
        }
        first = next;
      }
    }
    return { min, max };
  }
  /**
   * @brief Find iterators to minimum and maximum elements using default less-than ordering.
   */
  template<typename ForwardIt> __HOST_DEVICE__ __CPP_CONSTEXPR__
  std_util::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last){
    using value_type = typename std_ivy::iterator_traits<ForwardIt>::value_type;
    return minmax_element(first, last, std_fcnal::less<value_type>());
  }

  /**
   * @brief Compute minimum and maximum values from an initializer list.
   */
  template<typename T> __HOST_DEVICE__ __CPP_CONSTEXPR__ std::pair<T, T> minmax(std_ilist::initializer_list<T> ilist){
    auto p = minmax_element(ilist.begin(), ilist.end());
    return std_util::pair(*p.first, *p.second);
  }
  /**
   * @brief Compute minimum and maximum values from an initializer list using a custom comparator.
   */
  template<typename T, typename C> __HOST_DEVICE__ __CPP_CONSTEXPR__ std::pair<T, T> minmax(std_ilist::initializer_list<T> ilist, C comp){
    auto p = minmax_element(ilist.begin(), ilist.end(), comp);
    return std_util::pair(*p.first, *p.second);
  }
}


#endif
