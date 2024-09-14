#ifndef IVYMATHBASEARITHMETIC_H
#define IVYMATHBASEARITHMETIC_H


#include "autodiff/arithmetic/IvyMathBaseArithmetic.hh"
#include "std_ivy/IvyCmath.h"


namespace IvyMath{
  // General 1D function implementation
  template<typename T, typename Evaluator>
  __CUDA_HOST__ IvyRegularFunction_1D<T, Evaluator>::IvyRegularFunction_1D(IvyThreadSafePtr_t<T> const& dep) : base_t(), dep(dep){}
  template<typename T, typename Evaluator>
  __CUDA_HOST__ IvyRegularFunction_1D<T, Evaluator>::IvyRegularFunction_1D(IvyRegularFunction_1D<T, Evaluator> const& other) : base_t(other), dep(other.dep){}
  template<typename T, typename Evaluator>
  __CUDA_HOST__ IvyRegularFunction_1D<T, Evaluator>::IvyRegularFunction_1D(IvyRegularFunction_1D<T, Evaluator>&& other) : base_t(std_util::move(other)), dep(std_util::move(other.dep)){}
  template<typename T, typename Evaluator>
  __CUDA_HOST__ void IvyRegularFunction_1D<T, Evaluator>::eval() const{
    *(this->output) = evaluator_t::eval(unpack_function_input<T, Evaluator>::get(*dep));
  }
  template<typename T, typename Evaluator>
  __CUDA_HOST__ bool IvyRegularFunction_1D<T, Evaluator>::depends_on(IvyBaseNode const* node) const{
    return (base_t::depends_on(node) || IvyMath::depends_on(dep, node));
  }
  template<typename T, typename Evaluator>
  __CUDA_HOST__ IvyThreadSafePtr_t<typename IvyRegularFunction_1D<T, Evaluator>::grad_t> IvyRegularFunction_1D<T, Evaluator>::gradient(
    IvyThreadSafePtr_t<IvyBaseNode> const& var
  ) const{
    auto grad_dep = function_gradient<T, Evaluator>::get(*dep, var);
    return evaluator_t::gradient(dep)*grad_dep;
  }

  // General 2D function implementation
  template<typename T, typename U, typename Evaluator>
  __CUDA_HOST__ IvyRegularFunction_2D<T, U, Evaluator>::IvyRegularFunction_2D(IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y) : base_t(), x(x), y(y){}
  template<typename T, typename U, typename Evaluator>
  __CUDA_HOST__ IvyRegularFunction_2D<T, U, Evaluator>::IvyRegularFunction_2D(IvyRegularFunction_2D<T, U, Evaluator> const& other) : base_t(other), x(other.x), y(other.y){}
  template<typename T, typename U, typename Evaluator>
  __CUDA_HOST__ IvyRegularFunction_2D<T, U, Evaluator>::IvyRegularFunction_2D(IvyRegularFunction_2D<T, U, Evaluator>&& other) :
    base_t(std_util::move(other)), x(std_util::move(other.x)), y(std_util::move(other.y)){}
  template<typename T, typename U, typename Evaluator>
  __CUDA_HOST__ void IvyRegularFunction_2D<T, U, Evaluator>::eval() const{
    *(this->output) = evaluator_t::eval(unpack_function_input<T>::get(*x), unpack_function_input<U>::get(*y));
  }
  template<typename T, typename U, typename Evaluator>
  __CUDA_HOST__ bool IvyRegularFunction_2D<T, U, Evaluator>::depends_on(IvyBaseNode const* node) const{
    return (base_t::depends_on(node) || IvyMath::depends_on(x, node) || IvyMath::depends_on(y, node));
  }
  template<typename T, typename U, typename Evaluator>
  __CUDA_HOST__ IvyThreadSafePtr_t<typename IvyRegularFunction_2D<T, U, Evaluator>::grad_t> IvyRegularFunction_2D<T, U, Evaluator>::gradient(
    IvyThreadSafePtr_t<IvyBaseNode> const& var
  ) const{
    auto grad_x = function_gradient<T>::get(*x, var);
    auto grad_y = function_gradient<U>::get(*y, var);
    return evaluator_t::gradient(0, x, y)*grad_x + evaluator_t::gradient(1, x, y)*grad_y;
  }


  /****************/
  /* 1D FUNCTIONS */
  /****************/

  // Get real part of a variable
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ RealFcnal<T, domain_tag>::value_t RealFcnal<T, domain_tag>::eval(T const& x){ return x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ RealFcnal<T, complex_domain_tag>::value_t RealFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Re(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x){ return RealFcnal<T>::eval(x); }

  // Get imaginary part of a variable
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr ImagFcnal<T, domain_tag>::value_t ImagFcnal<T, domain_tag>::eval(T const& x){ return Zero<value_t>(); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ImagFcnal<T, complex_domain_tag>::value_t ImagFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Im(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x){ return ImagFcnal<T>::eval(x); }

  // Test to check whether value is an integer
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr IsIntegerFcnal<T, domain_tag>::value_t IsIntegerFcnal<T, domain_tag>::eval(T const& x){
    if constexpr (std_ttraits::is_integral_v<T>) return true;
    using IU_t = convert_to_integral_precision_t<T>;
    return x==__STATIC_CAST__(T, __ENCAPSULATE__(__STATIC_CAST__(IU_t, x)));
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ IsIntegerFcnal<T, complex_domain_tag>::value_t IsIntegerFcnal<T, complex_domain_tag>::eval(T const& x){ return IsIntegerFcnal::eval(unpack_function_input_reduced<T>::get(x).Re()) && unpack_function_input_reduced<T>::get(x).Im()==Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x){ return IsIntegerFcnal<T>::eval(x); }

  // Test to check whether value is real
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsRealFcnal<T, domain_tag>::value_t IsRealFcnal<T, domain_tag>::eval(T const& x){ return true; }
  template<typename T>
  __CUDA_HOST_DEVICE__ IsRealFcnal<T, complex_domain_tag>::value_t IsRealFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Im()==Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x){ return IsRealFcnal<T>::eval(x); }

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr IsImaginaryFcnal<T, domain_tag>::value_t IsImaginaryFcnal<T, domain_tag>::eval(T const& x){ return false; }
  template<typename T>
  __CUDA_HOST_DEVICE__ IsImaginaryFcnal<T, complex_domain_tag>::value_t IsImaginaryFcnal<T, complex_domain_tag>::eval(T const& x){ return unpack_function_input_reduced<T>::get(x).Re()==Zero<T>() && unpack_function_input_reduced<T>::get(x).Im()!=Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x){ return IsImaginaryFcnal<T>::eval(x); }

  // NEGATION
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ constexpr NegateFcnal<T, domain_tag>::value_t NegateFcnal<T, domain_tag>::eval(T const& x){ return -x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::value_t NegateFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(-unpack_function_input_reduced<T>::get(x)); }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::grad_t NegateFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::value_t NegateFcnal<T, complex_domain_tag>::eval(T const& x){ return value_t(-unpack_function_input_reduced<T>::get(x).Re(), -unpack_function_input_reduced<T>::get(x).Im()); }
  template<typename T>
  __HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::grad_t NegateFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x){ return NegateFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_arithmetic_v<T> && !is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x){ return Negate(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> Negate(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyNegate<T>>(def_mem_type, nullptr, IvyNegate(x));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> operator-(T const& x){
    return Negate(x);
  }

  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, domain_tag>::value_t MultInverseFcnal<T, domain_tag>::eval(T const& x){ return One<value_t>()/x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::value_t MultInverseFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(MultInverse(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::grad_t MultInverseFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return -MultInverse(x*x);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::value_t MultInverseFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = MultInverse(unpack_function_input_reduced<T>::get(x).norm());
    auto const phi = unpack_function_input_reduced<T>::get(x).phase();
    value_t res; res.set_absval_phase(MultInverse(r), -phi);
    return res;
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::grad_t MultInverseFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return -MultInverse(x*x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x){ return MultInverseFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultInverse<typename T::element_type>::base_t> MultInverse(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyMultInverse<T>>(def_mem_type, nullptr, IvyMultInverse(x));
  }

  // SQUARE ROOT
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, domain_tag>::value_t SqrtFcnal<T, domain_tag>::eval(T const& x){ return std_math::sqrt(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::value_t SqrtFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SqrtFcnal<dtype_t>::eval(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::grad_t SqrtFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
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
  template<typename T>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, complex_domain_tag>::grad_t SqrtFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return Pow(x, Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), MinusOneHalf<fndtype_t>()));
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x){ return SqrtFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySqrt<typename T::element_type>::base_t> Sqrt(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySqrt<T>>(def_mem_type, nullptr, IvySqrt(x));
  }

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, domain_tag>::value_t AbsFcnal<T, domain_tag>::eval(T const& x){ return std_math::abs(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, real_domain_tag>::value_t AbsFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Abs(unpack_function_input_reduced<T>::get(x)));
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, complex_domain_tag>::value_t AbsFcnal<T, complex_domain_tag>::eval(T const& x){
    return value_t(unpack_function_input_reduced<T>::get(x).norm());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x){ return AbsFcnal<T>::eval(x); }

  // COMPLEX PHASE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr PhaseFcnal<T, domain_tag>::value_t PhaseFcnal<T, domain_tag>::eval(T const& x){ return value_t(Zero<dtype_t>); }
  template<typename T>
  __CUDA_HOST_DEVICE__ PhaseFcnal<T, complex_domain_tag>::value_t PhaseFcnal<T, complex_domain_tag>::eval(T const& x){ return value_t(unpack_function_input_reduced<T>::get(x).phase()); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x){ return PhaseFcnal<T>::eval(x); }

  // CONJUGATION
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ ConjugateFcnal<T, domain_tag>::value_t ConjugateFcnal<T, domain_tag>::eval(T const& x){ value_t res(unpack_function_input_reduced<T>::get(x)); conjugate(res); return res; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x){ return ConjugateFcnal<T>::eval(x); }

  // EXPONENTIAL
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, domain_tag>::value_t ExpFcnal<T, domain_tag>::eval(T const& x){ return std_math::exp(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::value_t ExpFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Exp(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::grad_t ExpFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return Exp(x);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::value_t ExpFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = Exp(unpack_function_input_reduced<T>::get(x).Re());
    auto const& im = unpack_function_input_reduced<T>::get(x).Im();
    return value_t(r*Cos(im), r*Sin(im));
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::grad_t ExpFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return Exp(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x){ return ExpFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyExp<typename T::element_type>::base_t> Exp(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyExp<T>>(def_mem_type, nullptr, IvyExp(x));
  }

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ LogFcnal<T, domain_tag>::value_t LogFcnal<T, domain_tag>::eval(T const& x){ return std_math::log(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, real_domain_tag>::value_t LogFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, real_domain_tag>::grad_t LogFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return MultInverse(x);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::value_t LogFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = Log(unpack_function_input_reduced<T>::get(x).norm());
    auto const phi = unpack_function_input_reduced<T>::get(x).phase();
    return value_t(Log(r), phi);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::grad_t LogFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return MultInverse(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x){ return LogFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLog<typename T::element_type>::base_t> Log(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyLog<T>>(def_mem_type, nullptr, IvyLog(x));
  }

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, domain_tag>::value_t Log10Fcnal<T, domain_tag>::eval(T const& x){ return Log(x)/LogTen<value_t>(); }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::value_t Log10Fcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log10(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::grad_t Log10Fcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return MultInverse(x) / Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), LogTen<fndtype_t>());
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::value_t Log10Fcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res = Log(unpack_function_input_reduced<T>::get(x));
    return res/LogTen<dtype_t>();
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::grad_t Log10Fcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return MultInverse(x) / Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), LogTen<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x){ return Log10Fcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLog10<typename T::element_type>::base_t> Log10(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyLog10<T>>(def_mem_type, nullptr, IvyLog10(x));
  }

  // SINE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SinFcnal<T, domain_tag>::value_t SinFcnal<T, domain_tag>::eval(T const& x){ return std_math::sin(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, real_domain_tag>::value_t SinFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Sin(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, real_domain_tag>::grad_t SinFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return Cos(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::value_t SinFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto sa = Sin(a);
    auto chb = CosH(b);
    auto ca = Cos(a);
    auto shb = SinH(b);
    return value_t(sa*chb, ca*shb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::grad_t SinFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return Cos(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x){ return SinFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySin<typename T::element_type>::base_t> Sin(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySin<T>>(def_mem_type, nullptr, IvySin(x));
  }

  // COSINE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CosFcnal<T, domain_tag>::value_t CosFcnal<T, domain_tag>::eval(T const& x){ return std_math::cos(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, real_domain_tag>::value_t CosFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Cos(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, real_domain_tag>::grad_t CosFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return -Sin(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::value_t CosFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto ca = Cos(a);
    auto chb = CosH(b);
    auto sa = Sin(a);
    auto shb = CosH(b);
    return value_t(ca*chb, -sa*shb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::grad_t CosFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return -Sin(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x){ return CosFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCos<typename T::element_type>::base_t> Cos(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCos<T>>(def_mem_type, nullptr, IvyCos(x));
  }

  // TANGENT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ TanFcnal<T, domain_tag>::value_t TanFcnal<T, domain_tag>::eval(T const& x){ return Sin(x)/Cos(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ TanFcnal<T, domain_tag>::grad_t TanFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ auto r = Sec(x); return r*r; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x){ return TanFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyTan<typename T::element_type>::base_t> Tan(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyTan<T>>(def_mem_type, nullptr, IvyTan(x));
  }

  // SECANT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SecFcnal<T, domain_tag>::value_t SecFcnal<T, domain_tag>::eval(T const& x){ return One<dtype_t>/Cos(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SecFcnal<T, domain_tag>::grad_t SecFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return Sec(x)*Tan(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x){ return SecFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySec<typename T::element_type>::base_t> Sec(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySec<T>>(def_mem_type, nullptr, IvySec(x));
  }

  // COSECANT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CscFcnal<T, domain_tag>::value_t CscFcnal<T, domain_tag>::eval(T const& x){ return One<dtype_t>/Sin(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CscFcnal<T, domain_tag>::grad_t CscFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return -Csc(x)*Cot(x);; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x){ return CscFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCsc<typename T::element_type>::base_t> Csc(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCsc<T>>(def_mem_type, nullptr, IvyCsc(x));
  }

  // COTANGENT
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CotFcnal<T, domain_tag>::value_t CotFcnal<T, domain_tag>::eval(T const& x){ return Cos(x)/Sin(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CotFcnal<T, domain_tag>::grad_t CotFcnal<T, domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ auto r = Csc(x); return -r*r; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x){ return CotFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCot<typename T::element_type>::base_t> Cot(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCot<T>>(def_mem_type, nullptr, IvyCot(x));
  }

  // SINH
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ SinHFcnal<T, domain_tag>::value_t SinHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex - MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::value_t SinHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SinH(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::grad_t SinHFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return CosH(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::value_t SinHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto sha = SinH(a);
    auto cb = Cos(b);
    auto cha = CosH(a);
    auto sb = Sin(b);
    return value_t(sha*cb, cha*sb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::grad_t SinHFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return CosH(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x){ return SinHFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySinH<typename T::element_type>::base_t> SinH(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySinH<T>>(def_mem_type, nullptr, IvySinH(x));
  }

  // COSH
  template<typename T, typename domain_tag>
  __HOST_DEVICE__ CosHFcnal<T, domain_tag>::value_t CosHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex + MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::value_t CosHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(CosH(unpack_function_input_reduced<T>::get(x))); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::grad_t CosHFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){ return SinH(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::value_t CosHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = unpack_function_input_reduced<T>::get(x).Re();
    auto const& b = unpack_function_input_reduced<T>::get(x).Im();
    auto cha = CosH(a);
    auto cb = Cos(b);
    auto sha = SinH(a);
    auto sb = Sin(b);
    return value_t(cha*cb, sha*sb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::grad_t CosHFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return SinH(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x){ return CosHFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCosH<typename T::element_type>::base_t> CosH(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyCosH<T>>(def_mem_type, nullptr, IvyCosH(x));
  }


  /****************/
  /* 2D FUNCTIONS */
  /****************/

  // ADDITION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ AddFcnal<T, U, domain_T, domain_U>::value_t AddFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x).Im()+unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+y, unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)+unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      One<fndtype_t>()
    );
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()+unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
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
  __CUDA_HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y){ return Add(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> Add(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyAdd<T, U>>(def_mem_type, nullptr, IvyAdd(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> operator+(T const& x, U const& y){
    return Add(x, y);
  }

  // SUBTRACTION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ SubtractFcnal<T, U, domain_T, domain_U>::value_t SubtractFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x).Im()-unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-unpack_function_input_reduced<U>::get(y).Re(), -unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-y, unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)-unpack_function_input_reduced<U>::get(y).Re(), -unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()-unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    auto mem_type = (ivar==0 ? x.get_memory_type() : y.get_memory_type());
    auto gpu_stream = (ivar==0 ? x.gpu_stream() : y.gpu_stream());
    return make_IvyThreadSafePtr<typename grad_t::element_type>(
      mem_type, gpu_stream,
      (ivar==0 ? One<fndtype_t>() : MinusOne<fndtype_t>())
    );
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y){ return SubtractFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y){ return Subtract(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> Subtract(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvySubtract<T, U>>(def_mem_type, nullptr, IvySubtract(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> operator-(T const& x, U const& y){
    return Subtract(x, y);
  }

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, domain_T, domain_U>::value_t MultiplyFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x*y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
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
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*unpack_function_input_reduced<U>::get(y).Re() - unpack_function_input_reduced<T>::get(x).Im()+unpack_function_input_reduced<U>::get(y).Im(), unpack_function_input_reduced<T>::get(x).Re()*unpack_function_input_reduced<U>::get(y).Im() + unpack_function_input_reduced<T>::get(x).Im()*unpack_function_input_reduced<U>::get(y).Re());
  }
  template<typename T, typename U>
  __HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    switch (ivar){
    case 0:
      return Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return Constant<fndtype_t>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*unpack_function_input_reduced<U>::get(y).Re(), x*unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*y, unpack_function_input_reduced<T>::get(x).Im()*y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y).Re(), unpack_function_input_reduced<T>::get(x)*unpack_function_input_reduced<U>::get(y).Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    using grad_type_T = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<T>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return make_IvyThreadSafePtr<grad_type_T>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return ComplexVariable<fndtype_t>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()*unpack_function_input_reduced<U>::get(y), unpack_function_input_reduced<T>::get(x).Im()*unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    using grad_type_U = std_ttraits::conditional_t<
      std_ttraits::is_base_of_v<constant_value_tag, get_operability_t<U>>,
      IvyConstant<fndtype_t>, IvyVariable<fndtype_t>
    >;
    switch (ivar){
    case 0:
      return ComplexVariable<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) * y;
    default:
      return make_IvyThreadSafePtr<grad_type_U>(y.get_memory_type(), y.gpu_stream(), One<fndtype_t>()) * x;
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y){ return MultiplyFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y){ return Multiply(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> Multiply(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyMultiply<T, U>>(def_mem_type, nullptr, IvyMultiply(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> operator*(T const& x, U const& y){
    return Multiply(x, y);
  }

  // DIVISION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __HOST_DEVICE__ DivideFcnal<T, U, domain_T, domain_U>::value_t DivideFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x/y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
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
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    switch (ivar){
    case 0:
      return Constant<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/y);
  }
  template<typename T, typename U>
  __HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()/y, unpack_function_input_reduced<T>::get(x).Im()/y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
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
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    switch (ivar){
    case 0:
      return ComplexVariable<fndtype_t>(x.get_memory_type(), x.gpu_stream(), One<fndtype_t>()) / y;
    default:
      return -x/(y*y);
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y){ return DivideFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator/(T const& x, U const& y){ return Divide(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> Divide(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyDivide<T, U>>(def_mem_type, nullptr, IvyDivide(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> operator/(T const& x, U const& y){
    return Divide(x, y);
  }

  // POWER
  template<typename T, typename U, typename domain_T, typename domain_U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, domain_T, domain_U>::value_t PowFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return std_math::pow(x, y); }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return Pow(unpack_function_input_reduced<T>::get(x), unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, real_domain_tag>::grad_t PowFcnal<T, U, real_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_t PowFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x/unpack_function_input_reduced<U>::get(y));
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x)/y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(unpack_function_input_reduced<T>::get(x).Re()/y, unpack_function_input_reduced<T>::get(x).Im()/y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t PowFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_t PowFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t PowFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return unpack_function_input_reduced<T>::get(x)*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ PowFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_t PowFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y){
    using ctype = fundamental_data_t<U>;
    switch (ivar){
    case 0:
      return y*Pow(x, y-Constant<ctype>(y.get_memory_type(), y.gpu_stream(), One<ctype>()));
    default:
      return Log(x)*Pow(x, y);
    }
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y){ return PowFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!(is_arithmetic_v<T>&& is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename PowFcnal<T, U>::value_t operator/(T const& x, U const& y){ return Pow(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>&& is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyPow<typename T::element_type, typename U::element_type>::base_t> Pow(T const& x, U const& y){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr<IvyPow<T, U>>(def_mem_type, nullptr, IvyPow(x, y));
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(is_pointer_v<T>&& is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyPow<typename T::element_type, typename U::element_type>::base_t> operator/(T const& x, U const& y){
    return Pow(x, y);
  }

}


#endif
