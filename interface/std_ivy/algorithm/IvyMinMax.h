#ifndef IVYMINMAX_H
#define IVYMINMAX_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyInitializerList.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/iterator/IvyIteratorTraits.h"


namespace std_ivy{
  template<typename T> __HOST_DEVICE__ T const& min(T const& x, T const& y){ return (x>y ? y : x); }
  template<typename T> __HOST_DEVICE__ T const& max(T const& x, T const& y){ return (x>y ? x : y); }

  template<typename T> __HOST_DEVICE__ void minmax(T const& x, T const& y, T& __RESTRICT__ i, T& __RESTRICT__ a){ if (y<x){ i=y; a=x; } else{ i=x; a=y; } }
  template<typename T> __HOST_DEVICE__ void minmax(T const& x, T const& y, T* __RESTRICT__ i, T* __RESTRICT__ a){ if (y<x){ i=&y; a=&x; } else{ i=&x; a=&y; } }
  template<typename T, typename C> __HOST_DEVICE__ void minmax(T const& x, T const& y, C comp, T& __RESTRICT__ i, T& __RESTRICT__ a){ if (comp(y, x)){ i=y; a=x; } else{ i=x; a=y; } }
  template<typename T, typename C> __HOST_DEVICE__ void minmax(T const& x, T const& y, C comp, T* __RESTRICT__ i, T* __RESTRICT__ a){ if (comp(y, x)){ i=&y; a=&x; } else{ i=&x; a=&y; } }

  template<typename T> __HOST_DEVICE__ __CPP_CONSTEXPR__ std_util::pair<T, T> minmax(T const& x, T const& y){ return (x>y ? std_util::pair<T, T>(y, x) : std_util::pair<T, T>(x, y)); }
  template<typename T, typename C> __HOST_DEVICE__ __CPP_CONSTEXPR__ std_util::pair<T, T> minmax(T const& x, T const& y, C comp){
    return (C(y, x) ? std_util::pair<T, T>(y, x) : std_util::pair<T, T>(x, y));
  }


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
  template<typename ForwardIt> __HOST_DEVICE__ __CPP_CONSTEXPR__
  std_util::pair<ForwardIt, ForwardIt> minmax_element(ForwardIt first, ForwardIt last){
    using value_type = typename std_ivy::iterator_traits<ForwardIt>::value_type;
    return minmax_element(first, last, std_fcnal::less<value_type>());
  }

  template<typename T> __HOST_DEVICE__ __CPP_CONSTEXPR__ std::pair<T, T> minmax(std_ilist::initializer_list<T> ilist){
    auto p = minmax_element(ilist.begin(), ilist.end());
    return std_util::pair(*p.first, *p.second);
  }
  template<typename T, typename C> __HOST_DEVICE__ __CPP_CONSTEXPR__ std::pair<T, T> minmax(std_ilist::initializer_list<T> ilist, C comp){
    auto p = minmax_element(ilist.begin(), ilist.end(), comp);
    return std_util::pair(*p.first, *p.second);
  }
}


#endif
