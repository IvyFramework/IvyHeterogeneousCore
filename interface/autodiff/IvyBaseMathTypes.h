#ifndef IVYBASEMATHTYPES_H
#define IVYBASEMATHTYPES_H


#include "IvyBasicTypes.h"
#include "autodiff/IvyBaseConstant.h"
#include "autodiff/IvyBaseVariable.h"
#include "autodiff/IvyBaseComplexVariable.h"
#include "autodiff/IvyBaseTensor.h"
#include "autodiff/IvyBaseFunction.h"


namespace IvyMath{
  using namespace IvyTypes;

  template<class T> inline constexpr bool is_constant_valued_v = std_ttraits::is_base_of_v<IvyBaseConstant, T>;
  template<class T> inline constexpr bool is_variable_real_valued_v = std_ttraits::is_base_of_v<IvyBaseVariable, T>;
  template<class T> inline constexpr bool is_real_valued_v = is_constant_valued_v<T> || is_variable_real_valued_v<T>;
  template<class T> inline constexpr bool is_complex_valued_v = std_ttraits::is_base_of_v<IvyBaseComplexVariable, T>;
  template<class T> inline constexpr bool is_tensor_valued_v = std_ttraits::is_base_of_v<IvyBaseTensor, T>;
  template<class T> inline constexpr bool is_function_valued_v = std_ttraits::is_base_of_v<IvyBaseFunction, T>;

  // Convert to floating point if the type is complex. Otherwise, keep integers and floating points the same type.
  template<typename T> struct convert_to_floating_point_if_complex{ using type = T; };
  template<typename T> using convert_to_floating_point_if_complex_t = typename convert_to_floating_point_if_complex<T>::type;

  DEFINE_HAS_CALL(value);

  template<typename T> struct unpack_if_function_type{
    template<typename U> static auto get_unpacked_type(int) -> typename U::value_t;
    template<typename U> static auto get_unpacked_type(...) -> void;
    using type = std_ttraits::conditional_t<
      is_function_valued_v<T>,
      decltype(get_unpacked_type<T>(0)),
      T
    >;
  };
  template<typename T> using unpack_if_function_t = typename unpack_if_function_type<T>::type;

  template<typename T> struct reduced_value_type{
    template<typename U> static auto test_vtype(int) -> typename U::value_t;
    template<typename U> static auto test_vtype(...) -> void;
    using type = std_ttraits::conditional_t<
      has_value_call_v<T>,
      decltype(test_vtype<T>(0)),
      T
    >;
  };
  template<typename T> using reduced_value_t = typename reduced_value_type<T>::type;
  /*
  The value of reduced_value_t is supposed to be
  the value_t type of the class if there is one, or an arithmetic type otherwise.
  This means if
  - T = arithmetic type, reduced_value_t = T.
  - T = Ivy(Constant,'')Variable<U>, reduced_value_t = U (arithmetic type).
  - T = IvyComplexVariable<U>, reduced_value_t = IvyComplexVariable<U>.
  - T = IvyTensor<U>, reduced_value_t = IvyTensor<U>.
  - T = IvyFunction<U>, reduced_value_t = IvyFunction<U>::value_t, which is an IvyVariable/ComplexVariable/Tensor<R>.
  */

  template<typename T> struct reduced_data_type{
    template<typename U> static auto test_dtype(int) -> typename U::dtype_t;
    template<typename U> static auto test_dtype(...) -> void;
    using type = std_ttraits::conditional_t<
      has_value_call_v<T>,
      decltype(test_dtype<T>(0)),
      T
    >;
  };
  template<typename T> using reduced_data_t = typename reduced_data_type<T>::type;
  /*
  The value of reduced_data_t is supposed to be
  the dtype_t type of the class if there is one, or an arithmetic type otherwise.
  This means if
  - T = arithmetic type, reduced_data_t = T.
  - T = Ivy(Constant,'',Complex)Variable<U>, reduced_data_t = U (arithmetic type).
  - T = IvyTensor<U>, reduced_data_t = U (which is not necessarily an arithmetic type).
  - T = IvyFunction<U>, reduced_data_t = IvyFunction<U>::dtype_t.
  */

  template<typename T> struct is_pointer : std_ttraits::false_type{};
  template<typename T> struct is_pointer< IvyThreadSafePtr_t<T> > : std_ttraits::true_type{};
  template<typename T> inline constexpr bool is_pointer_v = is_pointer<T>::value;
  template<typename T> using pointer_t = typename T::element_type;

  template<typename T> struct elevateToFcnPtr_if_ptr{ using type = reduced_value_t<T>; };
  template<typename T> using elevateToFcnPtr_if_ptr_t = typename elevateToFcnPtr_if_ptr<T>::type;

}


#endif
