#ifndef IVYPRINTOUT_H
#define IVYPRINTOUT_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCstdio.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyInitializerList.h"


namespace std_ivy{
  template<typename T> struct value_printout{};
  template<> struct value_printout<bool>{ static __HOST_DEVICE__ void print(bool const& x){ __PRINT_INFO__((x ? "true" : "false")); } };
  template<> struct value_printout<unsigned char>{ static __HOST_DEVICE__ void print(unsigned char const& x){ __PRINT_INFO__("%u", x); } };
  template<> struct value_printout<signed char>{ static __HOST_DEVICE__ void print(signed char const& x){ __PRINT_INFO__("%c", x); } };
  template<> struct value_printout<unsigned short>{ static __HOST_DEVICE__ void print(unsigned short const& x){ __PRINT_INFO__("%u", x); } };
  template<> struct value_printout<short>{ static __HOST_DEVICE__ void print(short const& x){ __PRINT_INFO__("%d", x); } };
  template<> struct value_printout<unsigned int>{ static __HOST_DEVICE__ void print(unsigned int const& x){ __PRINT_INFO__("%u", x); } };
  template<> struct value_printout<int>{ static __HOST_DEVICE__ void print(int const& x){ __PRINT_INFO__("%d", x); } };
#ifndef __LONG_INT_FORBIDDEN__
  template<> struct value_printout<unsigned long int>{ static __HOST_DEVICE__ void print(unsigned long const& x){ __PRINT_INFO__("%lu", x); } };
  template<> struct value_printout<long int>{ static __HOST_DEVICE__ void print(long const& x){ __PRINT_INFO__("%ld", x); } };
#endif
  template<> struct value_printout<unsigned long long int>{ static __HOST_DEVICE__ void print(unsigned long long const& x){ __PRINT_INFO__("%llu", x); } };
  template<> struct value_printout<long long int>{ static __HOST_DEVICE__ void print(long long const& x){ __PRINT_INFO__("%lld", x); } };
  template<> struct value_printout<float>{ static __HOST_DEVICE__ void print(float const& x){ __PRINT_INFO__("%f", x); } };
  template<> struct value_printout<double>{ static __HOST_DEVICE__ void print(double const& x){ __PRINT_INFO__("%lf", x); } };
#ifndef __LONG_DOUBLE_FORBIDDEN__
  template<> struct value_printout<long double>{ static __HOST_DEVICE__ void print(long double const& x){ __PRINT_INFO__("%Lf", x); } };
#endif
  template<> struct value_printout<char*>{ static __HOST_DEVICE__ void print(char* const& x){ __PRINT_INFO__("%s", x); } };
  template<> struct value_printout<char const*>{ static __HOST_DEVICE__ void print(char const* const& x){ __PRINT_INFO__("%s", x); } };
  template<typename T> struct value_printout<T const>{ static __HOST_DEVICE__ void print(T const& x){ value_printout<T>::print(x); } };

  template<typename T, typename U> struct value_printout<std_util::pair<T, U>>{
    static __HOST_DEVICE__ void print(std_util::pair<T, U> const& x){
      __PRINT_INFO__("(");
      value_printout<T>::print(x.first); __PRINT_INFO__(", "); value_printout<U>::print(x.second);
      __PRINT_INFO__(")");
    }
  };
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


  template<typename T> __HOST_DEVICE__ void print_value(T const& var, bool put_endl = true){
    value_printout<T>::print(var);
    if (put_endl) __PRINT_INFO__("\n");
  }
}


#endif
