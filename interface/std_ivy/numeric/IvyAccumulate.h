#ifndef IVYACCUMULATE_H
#define IVYACCUMULATE_H


#include "config/IvyCompilerConfig.h"


#ifdef __USE_CUDA__

#include "std_ivy/IvyUtility.h"


namespace std_ivy{
  template<typename Iterator, typename T> constexpr T accumulate(Iterator first, Iterator last, T init){
    T res(init);
    for (; first!=last; ++first){ res = std_util::move(res) + *first; }
    return res;
  }
  template<typename Iterator, typename T, typename BinaryOperation> constexpr T accumulate(Iterator first, Iterator last, T init, BinaryOperation op){
    T res(init);
    for (; first!=last; ++first){ res = op(std_util::move(res), *first); }
    return res;
  }
}

#endif


#endif
