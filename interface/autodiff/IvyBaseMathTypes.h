#ifndef IVYBASEMATHTYPES_H
#define IVYBASEMATHTYPES_H


#include "config/IvyCompilerConfig.h"
#include "IvyBasicTypes.h"
#include "autodiff/base_types/IvyTypeTags.h"


namespace IvyMath{
  using namespace IvyTypes;

  // Convert to floating point if the type is complex. Otherwise, keep integers and floating points the same type.
  template<typename T> struct convert_to_floating_point_if_complex{ using type = T; };
  template<typename T> using convert_to_floating_point_if_complex_t = typename convert_to_floating_point_if_complex<T>::type;
  template<typename T> struct convert_to_floating_point_if_complex<IvyThreadSafePtr_t<T>>{
    using type = IvyThreadSafePtr_t<convert_to_floating_point_if_complex_t<T>>;
  };

  DEFINE_HAS_CALL(value);

  /*
  reduced_value_type:
  The value of reduced_value_t is supposed to be
  the value_t type of the class if there is one, or an arithmetic type otherwise.
  This means if
  - T = arithmetic type, reduced_value_t = T.
  - T = Ivy(Constant,'')Variable<U>, reduced_value_t = U (arithmetic type).
  - T = IvyComplexVariable<U>, reduced_value_t = IvyComplexVariable<U>.
  - T = IvyTensor<U>, reduced_value_t = IvyTensor<U>.
  - T = IvyFunction<U>, reduced_value_t = IvyFunction<U>::value_t, which is an IvyVariable/ComplexVariable/Tensor<R>.
  */
  template<typename T> struct reduced_value_type{
    template<typename U> static __CUDA_HOST_DEVICE__ auto test_vtype(int) -> typename U::value_t;
    template<typename U> static __CUDA_HOST_DEVICE__ auto test_vtype(...) -> T;
    using type = decltype(test_vtype<T>(0));
  };
  template<typename T> using reduced_value_t = typename reduced_value_type<T>::type;

  /*
  reduced_data_type:
  The type reduced_data_t is supposed to be
  the dtype_t type of the class if there is one, or an arithmetic type otherwise.
  This means if
  - T = arithmetic type, reduced_data_t = T.
  - T = Ivy(Constant,'',Complex)Variable<U>, reduced_data_t = U (arithmetic type).
  - T = IvyTensor<U>, reduced_data_t = U (which is not necessarily an arithmetic type).
  - T = IvyFunction<U>, reduced_data_t = IvyFunction<U>::dtype_t.
  */
  template<typename T> struct reduced_data_type{
    template<typename U> static __CUDA_HOST_DEVICE__ auto test_dtype(int) -> typename U::dtype_t;
    template<typename U> static __CUDA_HOST_DEVICE__ auto test_dtype(...) -> T;
    using type = decltype(test_dtype<T>(0));
  };
  template<typename T> using reduced_data_t = typename reduced_data_type<T>::type;

  template<typename T> struct elevateToFcnPtr_if_ptr{ using type = reduced_value_t<T>; };
  template<typename T> using elevateToFcnPtr_if_ptr_t = typename elevateToFcnPtr_if_ptr<T>::type;


  /*
  fundamental_data_t:
  The type fundamental_data_t is supposed to be the lowest baseline arithmetic type of each class.
  Conceptually, this is different from reduced_data_t in that it is the lowest baseline type,
  not necessarily the dtype_t of the data in IvyFunction or IvyTensor themselves.
  In other words, fundamental_data_t looks for the dtype_t recursively in IvyFunction and IvyTensor
  while reduced_data_t looks for the dtype_t only in the top level class.
  It is specialized further for IvyFunction and IvyTensor later on.
  To give a concrete example, if T = IvyFunction<IvyTensor<IvyVariable<double>>>,
  - fundamental_data_t<T> = double, but
  - reduced_data_t<T> = IvyTensor<IvyVariable<double>>.
  */
  template<typename T> struct fundamental_data_type{
    using type = reduced_data_t<T>;
  };
  template<typename T> using fundamental_data_t = typename fundamental_data_type<T>::type;

  /*
  convert_to_floating_point_fundamental_t:
  This is a shorthand to acquire the fundamental data type and convert it to floating point.
  Its result is a fundamental floating-point data type.
  */
  template<typename T> using convert_to_floating_point_fundamental_t = convert_to_floating_point_t<fundamental_data_t<T>>;

  /*
  convert_to_real_t:
  The type of convert_to_real_t is supposed to be the corresponding real value of a class.
  */
  template<typename T> struct convert_to_real_type{
    using type = T;
  };
  template<typename T> using convert_to_real_t = typename convert_to_real_type<T>::type;
  template<typename T> struct convert_to_real_type<IvyThreadSafePtr_t<T>>{
    using type = IvyThreadSafePtr_t<convert_to_real_t<T>>;
  };
  template<typename T> struct convert_to_real_type<T const>{
    using type = convert_to_real_t<T> const;
  };

  /*
  unpack_if_function_t:
  If T is a function type, unpack_if_function_t<T> is the value_t of T::value_t.
  Otherwise, it is T itself.
  */
  template<typename T> struct unpack_if_function_type{
    using type = std_ttraits::conditional_t<is_function_v<T>, reduced_value_t<T>, T>;
  };
  template<typename T> using unpack_if_function_t = typename unpack_if_function_type<T>::type;
  template<typename T, typename Operability = get_operability_t<T>> struct unpack_function_input{
    using value_t = T;
    static __CUDA_HOST_DEVICE__ value_t const& get(T const& t){ return t; }
  };
  template<typename T> struct unpack_function_input<T, function_value_tag>{
    using value_t = typename T::value_t;
    static __CUDA_HOST_DEVICE__ value_t const& get(T const& t){ return t.value(); }
  };

  /*
  more_precise_fundamental_t:
  The type more_precise_fundamental_t is supposed to be the more precise of the fundamental types of two classes.
  */
  template<typename T, typename U> struct more_precise_fundamental{
    using ctype_T = unpack_if_function_t<T>;
    using ctype_U = unpack_if_function_t<U>;
    using dtype_T = fundamental_data_t<ctype_T>;
    using dtype_U = fundamental_data_t<ctype_U>;
    using type = std_ttraits::conditional_t<TYPE_RANK(dtype_T) < TYPE_RANK(dtype_U), dtype_T, dtype_U>;
  };
  template<typename T, typename U> using more_precise_fundamental_t = typename more_precise_fundamental<T, U>::type;
}


#endif
