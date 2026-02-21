#ifndef IVYPRINTOUT_H
#define IVYPRINTOUT_H

/**
 * @file IvyPrintout.h
 * @brief Type-aware lightweight print helpers for host/device builds.
 */


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCstdio.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyInitializerList.h"


namespace std_ivy{
  /** @brief Primary printout traits template; specialized per printable type. */
  template<typename T> struct value_printout{};
  /** @brief Print boolean values as textual true/false. */
  template<> struct value_printout<bool>{ static __HOST_DEVICE__ void print(bool const& x){ __PRINT_INFO__((x ? "true" : "false")); } };
  /** @brief Print unsigned char as unsigned integer. */
  template<> struct value_printout<unsigned char>{ static __HOST_DEVICE__ void print(unsigned char const& x){ __PRINT_INFO__("%u", x); } };
  /** @brief Print signed char as character. */
  template<> struct value_printout<signed char>{ static __HOST_DEVICE__ void print(signed char const& x){ __PRINT_INFO__("%c", x); } };
  /** @brief Print unsigned short as unsigned integer. */
  template<> struct value_printout<unsigned short>{ static __HOST_DEVICE__ void print(unsigned short const& x){ __PRINT_INFO__("%u", x); } };
  /** @brief Print short as signed integer. */
  template<> struct value_printout<short>{ static __HOST_DEVICE__ void print(short const& x){ __PRINT_INFO__("%d", x); } };
  /** @brief Print unsigned int as unsigned integer. */
  template<> struct value_printout<unsigned int>{ static __HOST_DEVICE__ void print(unsigned int const& x){ __PRINT_INFO__("%u", x); } };
  /** @brief Print int as signed integer. */
  template<> struct value_printout<int>{ static __HOST_DEVICE__ void print(int const& x){ __PRINT_INFO__("%d", x); } };
#ifndef __LONG_INT_FORBIDDEN__
  /** @brief Print unsigned long int as unsigned long integer. */
  template<> struct value_printout<unsigned long int>{ static __HOST_DEVICE__ void print(unsigned long const& x){ __PRINT_INFO__("%lu", x); } };
  /** @brief Print long int as signed long integer. */
  template<> struct value_printout<long int>{ static __HOST_DEVICE__ void print(long const& x){ __PRINT_INFO__("%ld", x); } };
#endif
  /** @brief Print unsigned long long int as unsigned long long integer. */
  template<> struct value_printout<unsigned long long int>{ static __HOST_DEVICE__ void print(unsigned long long const& x){ __PRINT_INFO__("%llu", x); } };
  /** @brief Print long long int as signed long long integer. */
  template<> struct value_printout<long long int>{ static __HOST_DEVICE__ void print(long long const& x){ __PRINT_INFO__("%lld", x); } };
  /** @brief Print float in fixed-point format. */
  template<> struct value_printout<float>{ static __HOST_DEVICE__ void print(float const& x){ __PRINT_INFO__("%f", x); } };
  /** @brief Print double in fixed-point format. */
  template<> struct value_printout<double>{ static __HOST_DEVICE__ void print(double const& x){ __PRINT_INFO__("%lf", x); } };
#ifndef __LONG_DOUBLE_FORBIDDEN__
  /** @brief Print long double in fixed-point format. */
  template<> struct value_printout<long double>{ static __HOST_DEVICE__ void print(long double const& x){ __PRINT_INFO__("%Lf", x); } };
#endif
  /** @brief Print mutable C-string values. */
  template<> struct value_printout<char*>{ static __HOST_DEVICE__ void print(char* const& x){ __PRINT_INFO__("%s", x); } };
  /** @brief Print immutable C-string values. */
  template<> struct value_printout<char const*>{ static __HOST_DEVICE__ void print(char const* const& x){ __PRINT_INFO__("%s", x); } };
  /** @brief Forward const-qualified printing to non-const value_printout specialization. */
  template<typename T> struct value_printout<T const>{ static __HOST_DEVICE__ void print(T const& x){ value_printout<T>::print(x); } };

  /** @brief Print std_util::pair as `(first, second)`. */
  template<typename T, typename U> struct value_printout<std_util::pair<T, U>>{
    static __HOST_DEVICE__ void print(std_util::pair<T, U> const& x){
      __PRINT_INFO__("(");
      value_printout<T>::print(x.first); __PRINT_INFO__(", "); value_printout<U>::print(x.second);
      __PRINT_INFO__(")");
    }
  };
  /** @brief Print initializer_list as `{ elem0, elem1, ... }` or `(empty)`. */
  template<typename T> struct value_printout<std_ilist::initializer_list<T>>{
    static __HOST_DEVICE__ void print(std_ilist::initializer_list<T> const& x){
      if (x.size() == 0) __PRINT_INFO__("(empty)");
      else{
        __PRINT_INFO__("{ ");
        for (auto it = x.begin(); it != x.end(); ++it){
          value_printout<T>::print(*it);
          if ((it + 1) != x.end()) __PRINT_INFO__(", ");
        }
        __PRINT_INFO__(" }");
      }
    }
  };
  /**
   * @brief Print a value using the corresponding `value_printout` specialization.
   * @tparam T Value type.
   * @param var Value to print.
   * @param put_endl Append newline if true.
   */
  template<typename T> __HOST_DEVICE__ void print_value(T const& var, bool put_endl = true){
    value_printout<T>::print(var);
    if (put_endl) __PRINT_INFO__("\n");
  }
}


#endif
