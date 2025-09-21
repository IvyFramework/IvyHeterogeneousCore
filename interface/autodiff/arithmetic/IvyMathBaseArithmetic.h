#ifndef IVYMATHBASEARITHMETIC_H
#define IVYMATHBASEARITHMETIC_H


#include "autodiff/arithmetic/IvyMathBaseArithmetic.hh"
#include "std_ivy/IvyCmath.h"
#include "autodiff/special_functions/cerf/IvyCerf.h"


namespace IvyMath{
  /****************/
  /* 1D FUNCTIONS */
  /****************/

  // Get real part of a variable
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ RealFcnal<T, domain_tag>::value_t RealFcnal<T, domain_tag>::eval(T const& x){ return x; }
  template<typename T>
  __HOST_DEVICE__ RealFcnal<T, complex_domain_tag>::value_t RealFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Re(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x){ return RealFcnal<T>::eval(x); }

  // Get imaginary part of a variable
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr ImagFcnal<T, domain_tag>::value_t ImagFcnal<T, domain_tag>::eval(T const& x){ return Zero<value_t>(); }
  template<typename T>
  __HOST_DEVICE__ ImagFcnal<T, complex_domain_tag>::value_t ImagFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Im(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x){ return ImagFcnal<T>::eval(x); }

  // Test to check whether value is an integer
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsIntegerFcnal<T, domain_tag>::value_t IsIntegerFcnal<T, domain_tag>::eval(T const& x){
    if constexpr (std_ttraits::is_integral_v<T>) return true;
    using IU_t = convert_to_integral_precision_t<T>;
    return x==__STATIC_CAST__(T, __ENCAPSULATE__(__STATIC_CAST__(IU_t, x)));
  }
  template<typename T>
  __HOST_DEVICE__ IsIntegerFcnal<T, complex_domain_tag>::value_t IsIntegerFcnal<T, complex_domain_tag>::eval(T const& x){ return IsIntegerFcnal::eval(unpack_function_input_reduced<T>::get(x).Re()) && unpack_function_input_reduced<T>::get(x).Im()==Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x){ return IsIntegerFcnal<T>::eval(x); }

  // Test to check whether value is real
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsRealFcnal<T, domain_tag>::value_t IsRealFcnal<T, domain_tag>::eval(T const& x){ return true; }
  template<typename T>
  __HOST_DEVICE__ IsRealFcnal<T, complex_domain_tag>::value_t IsRealFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Im()==Zero<fundamental_data_t<T>>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x){ return IsRealFcnal<T>::eval(x); }

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsImaginaryFcnal<T, domain_tag>::value_t IsImaginaryFcnal<T, domain_tag>::eval(T const& x){ return false; }
  template<typename T>
  __HOST_DEVICE__ IsImaginaryFcnal<T, complex_domain_tag>::value_t IsImaginaryFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Re()==Zero<T>() && unpack_function_input_reduced<T>::get(x).Im()!=Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x){ return IsImaginaryFcnal<T>::eval(x); }

  // NEGATION
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr NegateFcnal<T, domain_tag>::value_t NegateFcnal<T, domain_tag>::eval(T const& x){ return -x; }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::value_t NegateFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(-unpack_function_input_reduced<T>::get(x)); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::grad_t NegateFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::value_t NegateFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& xx = unpack_function_input_reduced<T>::get(x);
    return value_t(-xx.Re(), -xx.Im());
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::grad_t NegateFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x){ return NegateFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_arithmetic_v<T> && !is_pointer_v<T>)> __HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x){ return Negate(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> Negate(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyNegate<typename T::element_type>>(def_mem_type, nullptr, IvyNegate(x));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> operator-(T const& x){
    return Negate(x);
  }

  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ MultInverseFcnal<T, domain_tag>::value_t MultInverseFcnal<T, domain_tag>::eval(T const& x){ return One<fndtype_t>()/x; }
  template<typename T>
  __HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::value_t MultInverseFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(MultInverse(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::grad_t MultInverseFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -MultInverse(x*x);
  }
  template<typename T>
  __HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::value_t MultInverseFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = unpack_function_input_reduced<T>::get(x).norm();
    auto const phi = unpack_function_input_reduced<T>::get(x).phase();
    value_t res; res.set_absval_phase(MultInverse(r), -phi);
    return res;
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::grad_t MultInverseFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -MultInverse(x*x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x){ return MultInverseFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultInverse<typename T::element_type>::base_t> MultInverse(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyMultInverse<typename T::element_type>>(def_mem_type, nullptr, IvyMultInverse(x));
  }

  // SQUARE ROOT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SqrtFcnal<T, domain_tag>::value_t SqrtFcnal<T, domain_tag>::eval(T const& x){ return std_math::sqrt(x); }
  template<typename T>
  __HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::value_t SqrtFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SqrtFcnal<dtype_t>::eval(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::grad_t SqrtFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Pow(x, Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), MinusOneHalf<fndtype_t>()));
  }
  template<typename T>
  __HOST_DEVICE__ SqrtFcnal<T, complex_domain_tag>::value_t SqrtFcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res;
    auto const& re = unpack_function_input_reduced<T>::get(x).Re();
    auto const& im = unpack_function_input_reduced<T>::get(x).Im();
    auto R = SqrtFcnal<dtype_t>::eval(re*re + im*im);
    dtype_t phi = std_math::atan2(im, re);
    res.set_absval_phase(R, phi*OneHalf<dtype_t>);
    return res;
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ SqrtFcnal<T, complex_domain_tag>::grad_t SqrtFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Pow(x, Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), MinusOneHalf<fndtype_t>()));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x){ return SqrtFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySqrt<typename T::element_type>::base_t> Sqrt(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySqrt<typename T::element_type>>(def_mem_type, nullptr, IvySqrt(x));
  }

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ AbsFcnal<T, domain_tag>::value_t AbsFcnal<T, domain_tag>::eval(T const& x){ return std_math::abs(x); }
  template<typename T>
  __HOST_DEVICE__ AbsFcnal<T, real_domain_tag>::value_t AbsFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Abs(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T>
  __HOST_DEVICE__ AbsFcnal<T, complex_domain_tag>::value_t AbsFcnal<T, complex_domain_tag>::eval(T const& x){
    return value_t(unpack_function_input_reduced<T>::get(x).norm());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x){ return AbsFcnal<T>::eval(x); }

  // COMPLEX PHASE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr PhaseFcnal<T, domain_tag>::value_t PhaseFcnal<T, domain_tag>::eval(T const& x){ return value_t(Zero<dtype_t>); }
  template<typename T>
  __HOST_DEVICE__ PhaseFcnal<T, complex_domain_tag>::value_t PhaseFcnal<T, complex_domain_tag>::eval(T const& x){ return value_t(unpack_function_input_reduced<T>::get(x).phase()); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x){ return PhaseFcnal<T>::eval(x); }

  // CONJUGATION
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ConjugateFcnal<T, domain_tag>::value_t ConjugateFcnal<T, domain_tag>::eval(T const& x){ value_t res(unpack_function_input_reduced<T>::get(x)); conjugate(res); return res; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x){ return ConjugateFcnal<T>::eval(x); }

  // EXPONENTIAL
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ExpFcnal<T, domain_tag>::value_t ExpFcnal<T, domain_tag>::eval(T const& x){ return std_math::exp(x); }
  template<typename T>
  __HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::value_t ExpFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Exp(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::grad_t ExpFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(x);
  }
  template<typename T>
  __HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::value_t ExpFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = Exp(unpack_function_input_reduced<T>::get(x).Re());
    auto const& im = unpack_function_input_reduced<T>::get(x).Im();
    return value_t(r*Cos(im), r*Sin(im));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::grad_t ExpFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x){ return ExpFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyExp<typename T::element_type>::base_t> Exp(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyExp<typename T::element_type>>(def_mem_type, nullptr, IvyExp(x));
  }

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ LogFcnal<T, domain_tag>::value_t LogFcnal<T, domain_tag>::eval(T const& x){ return std_math::log(x); }
  template<typename T>
  __HOST_DEVICE__ LogFcnal<T, real_domain_tag>::value_t LogFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ LogFcnal<T, real_domain_tag>::grad_t LogFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x);
  }
  template<typename T>
  __HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::value_t LogFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = unpack_function_input_reduced<T>::get(x).norm();
    auto const phi = unpack_function_input_reduced<T>::get(x).phase();
    return value_t(Log(r), phi);
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::grad_t LogFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x){ return LogFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLog<typename T::element_type>::base_t> Log(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyLog<typename T::element_type>>(def_mem_type, nullptr, IvyLog(x));
  }

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ Log10Fcnal<T, domain_tag>::value_t Log10Fcnal<T, domain_tag>::eval(T const& x){ return Log(x)/LogTen<value_t>(); }
  template<typename T>
  __HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::value_t Log10Fcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log10(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::grad_t Log10Fcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x) / Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), LogTen<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::value_t Log10Fcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res = Log(unpack_function_input_reduced<T>::get(x));
    return res/LogTen<dtype_t>();
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::grad_t Log10Fcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return MultInverse(x) / Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), LogTen<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x){ return Log10Fcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLog10<typename T::element_type>::base_t> Log10(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyLog10<typename T::element_type>>(def_mem_type, nullptr, IvyLog10(x));
  }

  // SINE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SinFcnal<T, domain_tag>::value_t SinFcnal<T, domain_tag>::eval(T const& x){ return std_math::sin(x); }
  template<typename T>
  __HOST_DEVICE__ SinFcnal<T, real_domain_tag>::value_t SinFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Sin(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ SinFcnal<T, real_domain_tag>::grad_t SinFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return Cos(x); }
  template<typename T>
  __HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::value_t SinFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    value_t arg(-b, a);
    auto const exp_arg = Exp(arg);
    return (exp_arg - MultInverse(exp_arg))/value_t(Zero<fndtype_t>(), Two<fndtype_t>());
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::grad_t SinFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return Cos(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x){ return SinFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySin<typename T::element_type>::base_t> Sin(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySin<typename T::element_type>>(def_mem_type, nullptr, IvySin(x));
  }

  // COSINE
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CosFcnal<T, domain_tag>::value_t CosFcnal<T, domain_tag>::eval(T const& x){ return std_math::cos(x); }
  template<typename T>
  __HOST_DEVICE__ CosFcnal<T, real_domain_tag>::value_t CosFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Cos(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ CosFcnal<T, real_domain_tag>::grad_t CosFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return -Sin(x); }
  template<typename T>
  __HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::value_t CosFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    value_t arg(-b, a);
    auto const exp_arg = Exp(arg);
    return (exp_arg + MultInverse(exp_arg))/Two<fndtype_t>();
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::grad_t CosFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return -Sin(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x){ return CosFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCos<typename T::element_type>::base_t> Cos(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCos<typename T::element_type>>(def_mem_type, nullptr, IvyCos(x));
  }

  // TANGENT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ TanFcnal<T, domain_tag>::value_t TanFcnal<T, domain_tag>::eval(T const& x){ return Sin(x)/Cos(x); }
  template<typename T, typename domain_tag> template<typename X_t>
  __HOST_DEVICE__ TanFcnal<T, domain_tag>::grad_t TanFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ auto r = Sec(x); return r*r; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x){ return TanFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyTan<typename T::element_type>::base_t> Tan(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyTan<typename T::element_type>>(def_mem_type, nullptr, IvyTan(x));
  }

  // SECANT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SecFcnal<T, domain_tag>::value_t SecFcnal<T, domain_tag>::eval(T const& x){ return MultInverse(Cos(x)); }
  template<typename T, typename domain_tag> template<typename X_t>
  __HOST_DEVICE__ SecFcnal<T, domain_tag>::grad_t SecFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return Sec(x)*Tan(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x){ return SecFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySec<typename T::element_type>::base_t> Sec(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySec<typename T::element_type>>(def_mem_type, nullptr, IvySec(x));
  }

  // COSECANT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CscFcnal<T, domain_tag>::value_t CscFcnal<T, domain_tag>::eval(T const& x){ return MultInverse(Sin(x)); }
  template<typename T, typename domain_tag> template<typename X_t>
  __HOST_DEVICE__ CscFcnal<T, domain_tag>::grad_t CscFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return -Csc(x)*Cot(x);; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x){ return CscFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCsc<typename T::element_type>::base_t> Csc(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCsc<typename T::element_type>>(def_mem_type, nullptr, IvyCsc(x));
  }

  // COTANGENT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CotFcnal<T, domain_tag>::value_t CotFcnal<T, domain_tag>::eval(T const& x){ return Cos(x)/Sin(x); }
  template<typename T, typename domain_tag> template<typename X_t>
  __HOST_DEVICE__ CotFcnal<T, domain_tag>::grad_t CotFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ auto r = Csc(x); return -r*r; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x){ return CotFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCot<typename T::element_type>::base_t> Cot(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCot<typename T::element_type>>(def_mem_type, nullptr, IvyCot(x));
  }

  // SINH
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SinHFcnal<T, domain_tag>::value_t SinHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex - MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T>
  __HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::value_t SinHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SinH(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::grad_t SinHFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return CosH(x); }
  template<typename T>
  __HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::value_t SinHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto sha = SinH(a);
    auto cb = Cos(b);
    auto cha = CosH(a);
    auto sb = Sin(b);
    return value_t(sha*cb, cha*sb);
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::grad_t SinHFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return CosH(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x){ return SinHFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySinH<typename T::element_type>::base_t> SinH(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySinH<typename T::element_type>>(def_mem_type, nullptr, IvySinH(x));
  }

  // COSH
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CosHFcnal<T, domain_tag>::value_t CosHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex + MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T>
  __HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::value_t CosHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(CosH(unpack_function_input_reduced<T>::get(x))); }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::grad_t CosHFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){ return SinH(x); }
  template<typename T>
  __HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::value_t CosHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto cha = CosH(a);
    auto cb = Cos(b);
    auto sha = SinH(a);
    auto sb = Sin(b);
    return value_t(cha*cb, sha*sb);
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::grad_t CosHFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return SinH(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x){ return CosHFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCosH<typename T::element_type>::base_t> CosH(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCosH<typename T::element_type>>(def_mem_type, nullptr, IvyCosH(x));
  }

  // ERF
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfFcnal<T, domain_tag>::value_t ErfFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erf(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfFcnal<T, real_domain_tag>::value_t ErfFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Erf(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfFcnal<T, real_domain_tag>::grad_t ErfFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfFcnal<T, complex_domain_tag>::value_t ErfFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erf(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfFcnal<T, complex_domain_tag>::grad_t ErfFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ErfFcnal<T>::value_t Erf(T const& x){ return ErfFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErf<typename T::element_type>::base_t> Erf(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyErf<typename T::element_type>>(def_mem_type, nullptr, IvyErf(x));
  }

  // ERFC
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfcFcnal<T, domain_tag>::value_t ErfcFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erfc(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFcnal<T, real_domain_tag>::value_t ErfcFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Erfc(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfcFcnal<T, real_domain_tag>::grad_t ErfcFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFcnal<T, complex_domain_tag>::value_t ErfcFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erfc(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfcFcnal<T, complex_domain_tag>::grad_t ErfcFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ErfcFcnal<T>::value_t Erfc(T const& x){ return ErfcFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErfc<typename T::element_type>::base_t> Erfc(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyErfc<typename T::element_type>>(def_mem_type, nullptr, IvyErfc(x));
  }

  // FADDEEVA
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ FaddeevaFcnal<T, domain_tag>::value_t FaddeevaFcnal<T, domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva(IvyComplexVariable(x));
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFcnal<T, real_domain_tag>::value_t FaddeevaFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Faddeeva(IvyComplexVariable(unpack_function_input_reduced<T>::get(x))));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ FaddeevaFcnal<T, real_domain_tag>::grad_t FaddeevaFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*Faddeeva(x);
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFcnal<T, complex_domain_tag>::value_t FaddeevaFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ FaddeevaFcnal<T, complex_domain_tag>::grad_t FaddeevaFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*Faddeeva(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename FaddeevaFcnal<T>::value_t Faddeeva(T const& x){ return FaddeevaFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyFaddeeva<typename T::element_type>::base_t> Faddeeva(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyFaddeeva<typename T::element_type>>(def_mem_type, nullptr, IvyFaddeeva(x));
  }

  // ERF-FAST
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfFastFcnal<T, domain_tag>::value_t ErfFastFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erf(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfFastFcnal<T, real_domain_tag>::value_t ErfFastFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(ErfFast(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfFastFcnal<T, real_domain_tag>::grad_t ErfFastFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfFastFcnal<T, complex_domain_tag>::value_t ErfFastFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erf_fast(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfFastFcnal<T, complex_domain_tag>::grad_t ErfFastFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ErfFastFcnal<T>::value_t ErfFast(T const& x){ return ErfFastFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErfFast<typename T::element_type>::base_t> ErfFast(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyErfFast<typename T::element_type>>(def_mem_type, nullptr, IvyErfFast(x));
  }

  // ERFC-FAST
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ ErfcFastFcnal<T, domain_tag>::value_t ErfcFastFcnal<T, domain_tag>::eval(T const& x){
    return std_math::erfc(x);
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFastFcnal<T, real_domain_tag>::value_t ErfcFastFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(ErfcFast(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfcFastFcnal<T, real_domain_tag>::grad_t ErfcFastFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T>
  __HOST_DEVICE__ ErfcFastFcnal<T, complex_domain_tag>::value_t ErfcFastFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::erfc_fast(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ ErfcFastFcnal<T, complex_domain_tag>::grad_t ErfcFastFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return -Exp(-x*x)*Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), TwoOverSqrtPi<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename ErfcFastFcnal<T>::value_t ErfcFast(T const& x){ return ErfcFastFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErfcFast<typename T::element_type>::base_t> ErfcFast(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyErfcFast<typename T::element_type>>(def_mem_type, nullptr, IvyErfcFast(x));
  }

  // FADDEEVA-FAST
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, domain_tag>::value_t FaddeevaFastFcnal<T, domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva_fast(IvyComplexVariable(x));
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, real_domain_tag>::value_t FaddeevaFastFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(FaddeevaFast(IvyComplexVariable(unpack_function_input_reduced<T>::get(x))));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, real_domain_tag>::grad_t FaddeevaFastFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*FaddeevaFast(x);
  }
  template<typename T>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, complex_domain_tag>::value_t FaddeevaFastFcnal<T, complex_domain_tag>::eval(T const& x){
    return IvyCerf::faddeeva_fast(unpack_function_input_reduced<T>::get(x));
  }
  template<typename T> template<typename X_t>
  __HOST_DEVICE__ FaddeevaFastFcnal<T, complex_domain_tag>::grad_t FaddeevaFastFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<X_t> const& x){
    return
      Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Zero<fndtype_t>(), TwoOverSqrtPi<fndtype_t>())
      - Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), Two<fndtype_t>())*x*FaddeevaFast(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename FaddeevaFastFcnal<T>::value_t FaddeevaFast(T const& x){ return FaddeevaFastFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyFaddeevaFast<typename T::element_type>::base_t> FaddeevaFast(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyFaddeevaFast<typename T::element_type>>(def_mem_type, nullptr, IvyFaddeevaFast(x));
  }


  /****************/
  /* 2D FUNCTIONS */
  /****************/

  // ADDITION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ AddFcnal<T, U, domain_T, domain_U>::value_t AddFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::gradient( unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x).Im()+unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+y, unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename AddFcnal<T, U>::value_t Add(T const& x, U const& y){ return AddFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y){ return Add(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> Add(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyAdd<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyAdd(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> operator+(T const& x, U const& y){
    return Add(x, y);
  }

  // SUBTRACTION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ SubtractFcnal<T, U, domain_T, domain_U>::value_t SubtractFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x).Im()-unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-unpack_function_input_reduced<U>::get(y).Re(), -unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-y, unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-unpack_function_input_reduced<U>::get(y).Re(), -unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y){ return SubtractFcnal<T, U>::eval(x, y); }
  template<
    typename T, typename U,
    ENABLE_IF_BOOL_IMPL(
      !(is_arithmetic_v<T> && is_arithmetic_v<U>)
      &&
      !is_pointer_v<T> && !is_pointer_v<U>
      &&
      !std_iter::is_contiguous_iterator_v<T> && !std_iter::is_contiguous_iterator_v<U>
    )
  > __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y){ return Subtract(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> Subtract(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySubtract<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvySubtract(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> operator-(T const& x, U const& y){
    return Subtract(x, y);
  }

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, domain_T, domain_U>::value_t MultiplyFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x*y; }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<T>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    using grad_type_U = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<U>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return make_IvyThreadSafePtr<grad_type_U>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const xr = unpack_function_input_reduced<T>::get(x).Re();
    auto const xi = unpack_function_input_reduced<T>::get(x).Im();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    return value_t(xr*yr - xi*yi, xr*yi + xi*yr);
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    switch (ivar){
    case 0:
      return Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return Constant<fndtype_t>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*unpack_function_input_reduced<U>::get(y).Re(), x*unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*y, unpack_function_input_reduced<T>::get(x).Im()*y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<T>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return Complex<fndtype_t>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im()*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_U = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<U>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return make_IvyThreadSafePtr<grad_type_U>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y){ return MultiplyFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y){ return Multiply(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> Multiply(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyMultiply<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyMultiply(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> operator*(T const& x, U const& y){
    return Multiply(x, y);
  }

  // DIVISION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ DivideFcnal<T, U, domain_T, domain_U>::value_t DivideFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x/y; }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<T>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    switch (ivar){
    case 0:
      return Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x*MultInverse(y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()/y, unpack_function_input_reduced<T>::get(x).Im()/y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using grad_type_T = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<T>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    switch (ivar){
    case 0:
      return Complex<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y){ return DivideFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator/(T const& x, U const& y){ return Divide(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> Divide(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyDivide<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyDivide(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> operator/(T const& x, U const& y){
    return Divide(x, y);
  }

  // POWER
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ PowFcnal<T, U, domain_T, domain_U>::value_t PowFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return std_math::pow(x, y); }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Pow(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t PowFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const xn = unpack_function_input_reduced<T>::get(x).norm();
    auto const xp = unpack_function_input_reduced<T>::get(x).phase();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    auto const res_norm = Pow(xn, yr)*Exp(-xp*yi);
    auto const res_phase = xp*yr + yi*Log(xn);
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Pow(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Pow(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const xn = Abs(x);
    auto const xp = x>=0 ? Zero<fndtype_t>() : Pi<fndtype_t>();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    auto const res_norm = Pow(xn, yr)*Exp(-xp*yi);
    auto const res_phase = xp*yr + yi*Log(xn);
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    auto const xn = unpack_function_input_reduced<T>::get(x).norm();
    auto const xp = unpack_function_input_reduced<T>::get(x).phase();
    auto const& yr = y;
    auto const res_norm = Pow(xn, yr);
    auto const res_phase = xp*yr;
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    auto const& xx = unpack_function_input_reduced<T>::get(x);
    auto const xn = Abs(xx);
    auto const xp = xx>=0 ? Zero<fndtype_t>() : Pi<fndtype_t>();
    auto const yr = unpack_function_input_reduced<U>::get(y).Re();
    auto const yi = unpack_function_input_reduced<U>::get(y).Im();
    auto const res_norm = Pow(xn, yr)*Exp(-xp*yi);
    auto const res_phase = xp*yr + yi*Log(xn);
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t PowFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    auto const xn = unpack_function_input_reduced<T>::get(x).norm();
    auto const xp = unpack_function_input_reduced<T>::get(x).phase();
    auto const yr = unpack_function_input_reduced<U>::get(y);
    auto const res_norm = Pow(xn, yr);
    auto const res_phase = xp*yr;
    return value_t(res_norm*Cos(res_phase), res_norm*Sin(res_phase));
  }
  template<typename T, typename U> template<typename X_t, typename Y_t>
  __HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t PowFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y){ return PowFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyPow<typename T::element_type, typename U::element_type>::base_t> Pow(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyPow<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyPow(x, y));
  }


  /******************/
  /* 1D COMPARISONS */
  /******************/

  // NOT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr NotFcnal<T, domain_tag>::value_t NotFcnal<T, domain_tag>::eval(T const& x){ return !x; }
  template<typename T>
  __HOST_DEVICE__ NotFcnal<T, real_domain_tag>::value_t NotFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Not(unpack_function_input_reduced<T>::get(x))); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename NotFcnal<T>::value_t Not(T const& x){ return NotFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNot<typename T::element_type>::base_t> Not(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyNot<typename T::element_type>>(def_mem_type, nullptr, IvyNot(x));
  }


  /******************/
  /* 2D COMPARISONS */
  /******************/

  // EQUALITY
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ EqualFcnal<T, U, domain_T, domain_U>::value_t EqualFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x==y; }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, real_domain_tag, real_domain_tag>::value_t EqualFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Equal(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t EqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()) && Equal(unpack_function_input_reduced<T>::get(x).Im(), unpack_function_input_reduced<U>::get(y).Im()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t EqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t EqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t EqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<U>::get(y).Re(), x) && IsReal(unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t EqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x).Re(), y) && IsReal(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t EqualFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x)) && IsReal(unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ EqualFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t EqualFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Equal(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)) && IsReal(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename EqualFcnal<T, U>::value_t Equal(T const& x, U const& y){ return EqualFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyEqual<typename T::element_type, typename U::element_type>::base_t> Equal(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyEqual<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyEqual(x, y));
  }

  // OR
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ OrFcnal<T, U, domain_T, domain_U>::value_t OrFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x||y; }
  template<typename T, typename U>
  __HOST_DEVICE__ OrFcnal<T, U, real_domain_tag, real_domain_tag>::value_t OrFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Or(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ OrFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t OrFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Or(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ OrFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t OrFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Or(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename OrFcnal<T, U>::value_t Or(T const& x, U const& y){ return OrFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyOr<typename T::element_type, typename U::element_type>::base_t> Or(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyOr<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyOr(x, y));
  }

  // XOR
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ XorFcnal<T, U, domain_T, domain_U>::value_t XorFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x^y; }
  template<typename T, typename U>
  __HOST_DEVICE__ XorFcnal<T, U, real_domain_tag, real_domain_tag>::value_t XorFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Xor(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ XorFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t XorFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(Xor(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ XorFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t XorFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(Xor(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename XorFcnal<T, U>::value_t Xor(T const& x, U const& y){ return XorFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyXor<typename T::element_type, typename U::element_type>::base_t> Xor(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyXor<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyXor(x, y));
  }
  template<typename T, typename U> __HOST_DEVICE__ auto XOr(T const& x, U const& y) -> decltype(Xor(x, y)){ return Xor(x, y); }

  // AND
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ AndFcnal<T, U, domain_T, domain_U>::value_t AndFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x&&y; }
  template<typename T, typename U>
  __HOST_DEVICE__ AndFcnal<T, U, real_domain_tag, real_domain_tag>::value_t AndFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return And(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AndFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t AndFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(And(unpack_function_input_reduced<U>::get(y), x));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ AndFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t AndFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(And(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename AndFcnal<T, U>::value_t And(T const& x, U const& y){ return AndFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAnd<typename T::element_type, typename U::element_type>::base_t> And(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyAnd<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyAnd(x, y));
  }

  // GREATER THAN
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, domain_T, domain_U>::value_t GreaterThanFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x>y; }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, real_domain_tag, real_domain_tag>::value_t GreaterThanFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return GreaterThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t GreaterThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t GreaterThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t GreaterThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t GreaterThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t GreaterThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t GreaterThanFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterThanFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t GreaterThanFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename GreaterThanFcnal<T, U>::value_t GreaterThan(T const& x, U const& y){ return GreaterThanFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyGreaterThan<typename T::element_type, typename U::element_type>::base_t> GreaterThan(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyGreaterThan<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyGreaterThan(x, y));
  }
  template<typename T, typename U> __HOST_DEVICE__ auto GT(T const& x, U const& y) -> decltype(GreaterThan(x, y)){ return GreaterThan(x, y); }

  // LESS THAN
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ LessThanFcnal<T, U, domain_T, domain_U>::value_t LessThanFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x<y; }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, real_domain_tag, real_domain_tag>::value_t LessThanFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return LessThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t LessThanFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t LessThanFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t LessThanFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t LessThanFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t LessThanFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t LessThanFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessThanFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t LessThanFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessThan(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename LessThanFcnal<T, U>::value_t LessThan(T const& x, U const& y){ return LessThanFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLessThan<typename T::element_type, typename U::element_type>::base_t> LessThan(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyLessThan<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyLessThan(x, y));
  }
  template<typename T, typename U> __HOST_DEVICE__ auto LT(T const& x, U const& y) -> decltype(LessThan(x, y)){ return LessThan(x, y); }

  // GREATER THAN OR EQUAL TO
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, domain_T, domain_U>::value_t GreaterOrEqualFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x>=y; }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::value_t GreaterOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return GreaterOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t GreaterOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t GreaterOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t GreaterOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t GreaterOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t GreaterOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ GreaterOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t GreaterOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(GreaterOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename GreaterOrEqualFcnal<T, U>::value_t GreaterOrEqual(T const& x, U const& y){ return GreaterOrEqualFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyGreaterOrEqual<typename T::element_type, typename U::element_type>::base_t> GreaterOrEqual(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyGreaterOrEqual<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyGreaterOrEqual(x, y));
  }
  template<typename T, typename U> __HOST_DEVICE__ auto GE(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y)){ return GreaterOrEqual(x, y); }
  template<typename T, typename U> __HOST_DEVICE__ auto GEQ(T const& x, U const& y) -> decltype(GreaterOrEqual(x, y)){ return GreaterOrEqual(x, y); }

  // LESS THAN OR EQUAL TO
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, domain_T, domain_U>::value_t LessOrEqualFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x<=y; }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::value_t LessOrEqualFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return LessOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t LessOrEqualFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t LessOrEqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(x, unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t LessOrEqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t LessOrEqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(x, unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t LessOrEqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x).Re(), y));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t LessOrEqualFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y).Re()));
  }
  template<typename T, typename U>
  __HOST_DEVICE__ LessOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t LessOrEqualFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(LessOrEqual(unpack_function_input_reduced<T>::get(x).Re(), unpack_function_input_reduced<U>::get(y)));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename LessOrEqualFcnal<T, U>::value_t LessOrEqual(T const& x, U const& y){ return LessOrEqualFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLessOrEqual<typename T::element_type, typename U::element_type>::base_t> LessOrEqual(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyLessOrEqual<typename T::element_type, typename U::element_type>>(def_mem_type, nullptr, IvyLessOrEqual(x, y));
  }
  template<typename T, typename U> __HOST_DEVICE__ auto LE(T const& x, U const& y) -> decltype(LessOrEqual(x, y)){ return LessOrEqual(x, y); }
  template<typename T, typename U> __HOST_DEVICE__ auto LEQ(T const& x, U const& y) -> decltype(LessOrEqual(x, y)){ return LessOrEqual(x, y); }

}


#endif
