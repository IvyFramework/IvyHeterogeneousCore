#ifndef IVYMULTIPLIES_H
#define IVYMULTIPLIES_H

#include "config/IvyCompilerConfig.h"


#ifdef __USE_CUDA__

#include "std_ivy/IvyUtility.h"


namespace std_ivy{
  template<typename T = void> struct multiplies{
    __CUDA_HOST_DEVICE__ constexpr T operator()(T const& x, T const& y) const{ return x * y; }
  };
  template<> struct multiplies<void>{
    template<typename T, typename U> __CUDA_HOST_DEVICE__ constexpr auto operator()(T&& lhs, U&& rhs) const -> decltype(std_util::forward<T>(lhs) * std_util::forward<U>(rhs)){
      return std_util::forward<T>(lhs) * std_util::forward<U>(rhs);
    }
  };
}

#endif


#endif
