/**
 * @file IvyAccumulate.h
 * @brief Accumulation utilities analogous to std::accumulate for std_ivy iterators.
 */
#ifndef IVYACCUMULATE_H
#define IVYACCUMULATE_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"


namespace std_ivy{
  /**
   * @brief Fold a range with addition starting from an initial value.
   * @tparam Iterator Input iterator type.
   * @tparam T Accumulator type.
   * @param first Beginning of the range.
   * @param last End of the range.
   * @param init Initial accumulator value.
   * @return Final accumulated value.
   */
  template<typename Iterator, typename T> __HOST_DEVICE__ constexpr T accumulate(Iterator first, Iterator last, T init){
    T res(init);
    for (; first!=last; ++first){ res = std_util::move(res) + *first; }
    return res;
  }
  /**
   * @brief Fold a range with a custom binary operation starting from an initial value.
   * @tparam Iterator Input iterator type.
   * @tparam T Accumulator type.
   * @tparam BinaryOperation Binary operation type.
   * @param first Beginning of the range.
   * @param last End of the range.
   * @param init Initial accumulator value.
   * @param op Binary operation applied at each step.
   * @return Final accumulated value.
   */
  template<typename Iterator, typename T, typename BinaryOperation> __HOST_DEVICE__ constexpr T accumulate(Iterator first, Iterator last, T init, BinaryOperation op){
    T res(init);
    for (; first!=last; ++first){ res = op(std_util::move(res), *first); }
    return res;
  }
}


#endif
