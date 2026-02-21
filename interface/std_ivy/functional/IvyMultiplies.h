/**
 * @file IvyMultiplies.h
 * @brief Multiplication functor utilities in the std_ivy namespace.
 */
#ifndef IVYMULTIPLIES_H
#define IVYMULTIPLIES_H

#include "config/IvyCompilerConfig.h"


#ifdef __USE_CUDA__

#include "std_ivy/IvyUtility.h"


namespace std_ivy{
  /**
   * @brief Binary multiplication functor for a concrete value type.
   * @tparam T Operand and result type.
   */
  template<typename T = void> struct multiplies{
    /**
     * @brief Multiply two values of type @p T.
     * @param x Left operand.
     * @param y Right operand.
     * @return Product of @p x and @p y.
     */
    __HOST_DEVICE__ constexpr T operator()(T const& x, T const& y) const{ return x * y; }
  };
  /**
   * @brief Transparent multiplication functor that supports mixed operand types.
   */
  template<> struct multiplies<void>{
    /**
     * @brief Multiply two forwarded operands.
     * @tparam T Left operand type.
     * @tparam U Right operand type.
     * @param lhs Left operand.
     * @param rhs Right operand.
     * @return Product expression with perfect forwarding.
     */
    template<typename T, typename U> __HOST_DEVICE__ constexpr auto operator()(T&& lhs, U&& rhs) const -> decltype(std_util::forward<T>(lhs) * std_util::forward<U>(rhs)){
      return std_util::forward<T>(lhs) * std_util::forward<U>(rhs);
    }
  };
}

#endif


#endif
