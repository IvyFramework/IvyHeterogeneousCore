#ifndef IVYBASICTYPES_H
#define IVYBASICTYPES_H


#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyLimits.h"


namespace IvyTypes{
  typedef unsigned long long int size_t;
  typedef long long int signed_size_t;
  typedef long long int ptrdiff_t;

  template<typename T> struct convert_to_floating_point{
    static constexpr int d10_t = std_limits::numeric_limits<T>::digits10;
    static constexpr int d10_f = std_limits::numeric_limits<float>::digits10;
    static constexpr int d10_d = std_limits::numeric_limits<double>::digits10;
#ifndef __LONG_DOUBLE_FORBIDDEN__
    static constexpr int d10_ld = std_limits::numeric_limits<long double>::digits10;
#endif

    static constexpr bool use_native_type = !std_ttraits::is_arithmetic_v<T>;
    static constexpr bool convert_to_float = !use_native_type && d10_t<=d10_f;
#ifndef __LONG_DOUBLE_FORBIDDEN__
    static constexpr bool convert_to_double = !use_native_type && d10_t<=d10_d;
#endif

    using type = std_ttraits::conditional_t<
      use_native_type, T,
      std_ttraits::conditional_t<
        convert_to_float, float,
#ifndef __LONG_DOUBLE_FORBIDDEN__
        std_ttraits::conditional_t<
          convert_to_double, double, long double
        >
#else
        double
#endif
      >
    >;
  };
  template<typename T> using convert_to_floating_point_t = typename convert_to_floating_point<T>::type;
#define FLOAT_TYPE(TYPE) IvyTypes::convert_to_floating_point_t<TYPE>

  template<typename T, bool is_unsigned = false> struct convert_to_integral_precision{
    using char_type = std_ttraits::conditional_t<is_unsigned, unsigned char, signed char>;
    using short_type = std_ttraits::conditional_t<is_unsigned, unsigned short, short>;
    using int_type = std_ttraits::conditional_t<is_unsigned, unsigned int, int>;
#ifndef __LONG_INT_FORBIDDEN__
    using lint_type = std_ttraits::conditional_t<is_unsigned, unsigned long int, long int>;
#endif
    using llint_type = std_ttraits::conditional_t<is_unsigned, unsigned long long int, long long int>;

    static constexpr int d10_t = std_limits::numeric_limits<T>::digits10;
    static constexpr int d10_char = std_limits::numeric_limits<char_type>::digits10;
    static constexpr int d10_short = std_limits::numeric_limits<short_type>::digits10;
    static constexpr int d10_int = std_limits::numeric_limits<int_type>::digits10;
#ifndef __LONG_INT_FORBIDDEN__
    static constexpr int d10_lint = std_limits::numeric_limits<lint_type>::digits10;
#endif

    static constexpr bool use_native_type = std_ttraits::is_integral_v<T> || !std_ttraits::is_arithmetic_v<T>;
    static constexpr bool convert_to_char = !use_native_type && d10_t<=d10_char;
    static constexpr bool convert_to_short = !use_native_type && d10_t<=d10_short;
    static constexpr bool convert_to_int = !use_native_type && d10_t<=d10_int;
#ifndef __LONG_INT_FORBIDDEN__
    static constexpr bool convert_to_lint = !use_native_type && d10_t<=d10_lint;
#endif

    using type = std_ttraits::conditional_t<
      use_native_type, T,
      std_ttraits::conditional_t<
        convert_to_char, char_type,
        std_ttraits::conditional_t<
          convert_to_short, short_type,
          std_ttraits::conditional_t<
            convert_to_int, int_type,
#ifndef __LONG_INT_FORBIDDEN__
            std_ttraits::conditional_t<
              convert_to_lint, lint_type,
#endif
              llint_type
#ifndef __LONG_INT_FORBIDDEN__
            >
#endif
          >
        >
      >
    >;
  };
  template<typename T, bool is_unsigned=false>
  using convert_to_integral_precision_t = typename convert_to_integral_precision<T, is_unsigned>::type;
#define INT_TYPE(TYPE) IvyTypes::convert_to_integral_precision_t<TYPE>
#define UINT_TYPE(TYPE) IvyTypes::convert_to_integral_precision_t<TYPE, true>

  using type_rank_t = unsigned short;
  template<typename T> struct type_rank : public std_ttraits::integral_constant<
    type_rank_t,
    std_ttraits::is_same_v<T, void> ? 0 :
#ifndef __LONG_DOUBLE_FORBIDDEN__
    std_ttraits::is_same_v<T, long double> ? 1 :
#endif
    std_ttraits::is_same_v<T, double> ? 2 :
    std_ttraits::is_same_v<T, float> ? 3 :
    std_ttraits::is_same_v<T, long long int> ? 4 :
#ifndef __LONG_INT_FORBIDDEN__
    std_ttraits::is_same_v<T, long int> ? 5 :
#endif
    std_ttraits::is_same_v<T, int> ? 6 :
    std_ttraits::is_same_v<T, short> ? 7 :
    std_ttraits::is_same_v<T, signed char> ? 8 :
    std_ttraits::is_same_v<T, unsigned long long int> ? 9 :
#ifndef __LONG_INT_FORBIDDEN__
    std_ttraits::is_same_v<T, unsigned long int> ? 10 :
#endif
    std_ttraits::is_same_v<T, unsigned int> ? 11 :
    std_ttraits::is_same_v<T, unsigned short> ? 12 :
    std_ttraits::is_same_v<T, unsigned char> ? 13 :
    std_ttraits::is_same_v<T, bool> ? 14 : 15
  >{};
  template<typename T> inline constexpr type_rank_t type_rank_v = type_rank<T>::value;
  template<typename T = void, typename U = void> struct type_rank_sum : public std_ttraits::integral_constant<
    type_rank_t,
    type_rank_v<T> + type_rank_v<U>
  >{};
  template<typename T = void, typename U = void> inline constexpr type_rank_t type_rank_sum_v = type_rank_sum<T, U>::value;
  template<typename T, typename U, typename S> struct type_rank_sum<T, type_rank_sum<U, S>> : public std_ttraits::integral_constant<
    type_rank_t,
    type_rank_v<T> + type_rank_sum_v<U, S>
  >{};
  template<typename T, typename U, typename S> struct type_rank_sum<type_rank_sum<T, U>, S> : public std_ttraits::integral_constant<
    type_rank_t,
    type_rank_v<S> + type_rank_sum_v<T, U>
  >{};
  template<template <typename...> typename T, typename U, typename... Args> struct type_rank<T<U, Args...>> : public std_ttraits::integral_constant<
    type_rank_t,
    type_rank_v<U> + type_rank_sum_v<Args...>
  >{};
#define TYPE_RANK(TYPE) IvyTypes::type_rank_v<TYPE>
}


#endif
