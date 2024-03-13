#ifndef IVYMATHBASEARITHMETIC_H
#define IVYMATHBASEARITHMETIC_H


#include "autodiff/arithmetic/IvyMathBaseArithmetic.hh"
#include "std_ivy/IvyCmath.h"


namespace IvyMath{
  // Get real part of a variable
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ RealFcnal<T, domain_tag>::value_t RealFcnal<T, domain_tag>::eval(T const& x){ return x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ RealFcnal<T, complex_domain_tag>::value_t RealFcnal<T, complex_domain_tag>::eval(T const& x){ return x.Re(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x){ return RealFcnal<T>::eval(x); }

  // Get imaginary part of a variable
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr ImagFcnal<T, domain_tag>::value_t ImagFcnal<T, domain_tag>::eval(T const& x){ return Zero<value_t>(); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ImagFcnal<T, complex_domain_tag>::value_t ImagFcnal<T, complex_domain_tag>::eval(T const& x){ return x.Im(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x){ return ImagFcnal<T>::eval(x); }

  // Test to check whether value is an integer
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr IsIntegerFcnal<T, domain_tag>::value_t IsIntegerFcnal<T, domain_tag>::eval(T const& x){
    if (std_ttraits::is_integral_v<T>) return true;
    using IU_t = convert_to_integral_precision_t<T>;
    return x==__STATIC_CAST__(T, __ENCAPSULATE__(__STATIC_CAST__(IU_t, x)));
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ IsIntegerFcnal<T, complex_domain_tag>::value_t IsIntegerFcnal<T, complex_domain_tag>::eval(T const& x){ return IsIntegerFcnal::eval(x.Re()) && x.Im()==Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x){ return IsIntegerFcnal<T>::eval(x); }

  // Test to check whether value is real
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr IsRealFcnal<T, domain_tag>::value_t IsRealFcnal<T, domain_tag>::eval(T const& x){ return true; }
  template<typename T>
  __CUDA_HOST_DEVICE__ IsRealFcnal<T, complex_domain_tag>::value_t IsRealFcnal<T, complex_domain_tag>::eval(T const& x){ return x.Im()==Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x){ return IsRealFcnal<T>::eval(x); }

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr IsImaginaryFcnal<T, domain_tag>::value_t IsImaginaryFcnal<T, domain_tag>::eval(T const& x){ return false; }
  template<typename T>
  __CUDA_HOST_DEVICE__ IsImaginaryFcnal<T, complex_domain_tag>::value_t IsImaginaryFcnal<T, complex_domain_tag>::eval(T const& x){ return x.Re()==Zero<T>() && x.Im()!=Zero<T>(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x){ return IsImaginaryFcnal<T>::eval(x); }

  // NEGATION
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr NegateFcnal<T, domain_tag>::value_t NegateFcnal<T, domain_tag>::eval(T const& x){ return -x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::value_t NegateFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(-x.value()); }
  template<typename T>
  __CUDA_HOST_DEVICE__ NegateFcnal<T, real_domain_tag>::grad_t NegateFcnal<T, real_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::value_t NegateFcnal<T, complex_domain_tag>::eval(T const& x){ return value_t(-x.Re(), -x.Im()); }
  template<typename T>
  __CUDA_HOST_DEVICE__ NegateFcnal<T, complex_domain_tag>::grad_t NegateFcnal<T, complex_domain_tag>::gradient(IvyThreadSafePtr_t<T> const& x){
    return make_IvyThreadSafePtr<typename grad_t::element_type>(x.get_memory_type(), x.gpu_stream(), MinusOne<fndtype_t>());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x){ return NegateFcnal<T>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!std_ttraits::is_arithmetic_v<T> && !is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x){ return Negate(x); }

  template<typename T> class IvyNegate : public IvyFunction<reduced_value_t<T>, get_domain_t<T>>{
  public:
    using base_t = IvyFunction<reduced_value_t<T>, get_domain_t<T>>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
    using grad_t = typename base_t::grad_t;
    using Evaluator = NegateFcnal<unpack_if_function_t<T>>;

  protected:
    IvyThreadSafePtr_t<T> dep;

  public:
    __CUDA_HOST__ IvyNegate(IvyThreadSafePtr_t<T> const& dep) : base_t(), dep(dep){}
    __CUDA_HOST__ IvyNegate(IvyNegate const& other) : base_t(other), dep(other.dep){}
    __CUDA_HOST__ IvyNegate(IvyNegate&& other) : base_t(std::move(other)), dep(std::move(other.dep)){}

    __CUDA_HOST__ void eval() const override{
      //eval_fcn(dep); // The dependent could be a function itself, so we need to evaluate it first.
      *(this->output) = Evaluator::eval(unpack_function_input<T>::get(*dep));
    }
    __CUDA_HOST__ bool depends_on(IvyBaseNode const* node) const override{
      return (base_t::depends_on(node) || IvyMath::depends_on(dep, node));
    }
    __CUDA_HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const override{
      auto grad_dep = function_gradient<T>::get(*dep, var);
      return Evaluator::gradient(dep)*grad_dep;
    }
  };
  template<typename T, ENABLE_IF_BOOL_IMPL(!std_ttraits::is_arithmetic_v<T> && is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<T>::base_t> operator-(T const& x){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    return make_IvyThreadSafePtr(def_mem_type, nullptr, IvyNegate(x));
  }


  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, domain_tag>::value_t MultInverseFcnal<T, domain_tag>::eval(T const& x){ return One<value_t>()/x; }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, domain_tag>::grad_t MultInverseFcnal<T, domain_tag>::gradient(T const& x){ return MinusOne<value_t>()/(x*x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::value_t MultInverseFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(MultInverse(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, real_domain_tag>::grad_t MultInverseFcnal<T, real_domain_tag>::gradient(T const& x){ return value_t(-One<dtype_t>()/(x.value()*x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::value_t MultInverseFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = MultInverse(x.norm());
    auto const phi = x.phase();
    value_t res; res.set_absval_phase(MultInverse(r), -phi);
    return res;
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ MultInverseFcnal<T, complex_domain_tag>::grad_t MultInverseFcnal<T, complex_domain_tag>::gradient(T const& x){
    auto const r = MultInverse(x.norm());
    auto const phi = x.phase();
    auto const rinv = MultInverse(r);
    value_t res; res.set_absval_phase(-rinv*rinv, -Two<dtype_t>()*phi);
    return res;
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x){ return MultInverseFcnal<T>::eval(x); }

  // SQUARE ROOT
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, domain_tag>::value_t SqrtFcnal<T, domain_tag>::eval(T const& x){ return std_math::sqrt(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, domain_tag>::grad_t SqrtFcnal<T, domain_tag>::gradient(T const& x){ return OneHalf<dtype_t>/SqrtFcnal<T>::eval(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::value_t SqrtFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SqrtFcnal<dtype_t>::eval(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, real_domain_tag>::grad_t SqrtFcnal<T, real_domain_tag>::gradient(T const& x){ return value_t(SqrtFcnal<dtype_t>::gradient(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SqrtFcnal<T, complex_domain_tag>::value_t SqrtFcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res;
    auto const& re = x.Re();
    auto const& im = x.Im();
    auto R = SqrtFcnal<dtype_t>::eval(re*re + im*im);
    dtype_t phi = std_math::atan2(im, re);
    res.set_absval_phase(R, phi*OneHalf<dtype_t>);
    return res;
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x){ return SqrtFcnal<T>::eval(x); }

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, domain_tag>::value_t AbsFcnal<T, domain_tag>::eval(T const& x){ return std_math::abs(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, domain_tag>::grad_t AbsFcnal<T, domain_tag>::gradient(T const& x){ return std_math::abs(x)/x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, real_domain_tag>::value_t AbsFcnal<T, real_domain_tag>::eval(T const& x){
    return value_t(Abs(x.value()));
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, real_domain_tag>::grad_t AbsFcnal<T, real_domain_tag>::gradient(T const& x){
    value_t res(x.value());
    auto& r = res.value(); r = std_math::abs(r)/r;
    return res;
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ AbsFcnal<T, complex_domain_tag>::value_t AbsFcnal<T, complex_domain_tag>::eval(T const& x){
    return value_t(x.norm());
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x){ return AbsFcnal<T>::eval(x); }

  // COMPLEX PHASE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr PhaseFcnal<T, domain_tag>::value_t PhaseFcnal<T, domain_tag>::eval(T const& x){ return value_t(Zero<dtype_t>); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr PhaseFcnal<T, domain_tag>::grad_t PhaseFcnal<T, domain_tag>::gradient(T const& x){ return value_t(Zero<dtype_t>); }
  template<typename T>
  __CUDA_HOST_DEVICE__ PhaseFcnal<T, complex_domain_tag>::value_t PhaseFcnal<T, complex_domain_tag>::eval(T const& x){ return value_t(x.phase()); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x){ return PhaseFcnal<T>::eval(x); }

  // CONJUGATION
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ ConjugateFcnal<T, domain_tag>::value_t ConjugateFcnal<T, domain_tag>::eval(T const& x){ value_t res(x.value()); conjugate(res); return res; }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ constexpr ConjugateFcnal<T, domain_tag>::grad_t ConjugateFcnal<T, domain_tag>::gradient(T const& x){ return grad_t(One<dtype_t>); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x){ return ConjugateFcnal<T>::eval(x); }

  // EXPONENTIAL
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, domain_tag>::value_t ExpFcnal<T, domain_tag>::eval(T const& x){ return std_math::exp(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, domain_tag>::grad_t ExpFcnal<T, domain_tag>::gradient(T const& x){ return ExpFcnal<T, domain_tag>::eval(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::value_t ExpFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Exp(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, real_domain_tag>::grad_t ExpFcnal<T, real_domain_tag>::gradient(T const& x){ return ExpFcnal<T, real_domain_tag>::eval(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::value_t ExpFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = Exp(x.Re());
    return value_t(r*std_math::cos(x.Im()), r*std_math::sin(x.Im()));
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ ExpFcnal<T, complex_domain_tag>::grad_t ExpFcnal<T, complex_domain_tag>::gradient(T const& x){ return ExpFcnal<T, complex_domain_tag>::eval(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x){ return ExpFcnal<T>::eval(x); }

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ LogFcnal<T, domain_tag>::value_t LogFcnal<T, domain_tag>::eval(T const& x){ return std_math::log(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ LogFcnal<T, domain_tag>::grad_t LogFcnal<T, domain_tag>::gradient(T const& x){ return One<value_t>()/x; }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, real_domain_tag>::value_t LogFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, real_domain_tag>::grad_t LogFcnal<T, real_domain_tag>::gradient(T const& x){ return value_t(One<dtype_t>()/x.value()); }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::value_t LogFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const r = Log(x.norm());
    auto const phi = x.phase();
    return value_t(Log(r), phi);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ LogFcnal<T, complex_domain_tag>::grad_t LogFcnal<T, complex_domain_tag>::gradient(T const& x){
    return MultInverseFcnal<T, complex_domain_tag>::eval(x);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x){ return LogFcnal<T>::eval(x); }

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, domain_tag>::value_t Log10Fcnal<T, domain_tag>::eval(T const& x){ return Log(x)/LogTen<value_t>(); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, domain_tag>::grad_t Log10Fcnal<T, domain_tag>::gradient(T const& x){ return LogFcnal<T, domain_tag>::gradient(x)/LogTen<value_t>(); }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::value_t Log10Fcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Log10(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, real_domain_tag>::grad_t Log10Fcnal<T, real_domain_tag>::gradient(T const& x){
    value_t res = LogFcnal<T, real_domain_tag>::gradient(x);
    res.value() /= LogTen<dtype_t>();
    return res;
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::value_t Log10Fcnal<T, complex_domain_tag>::eval(T const& x){
    value_t res = Log(x);
    res.Re() /= LogTen<dtype_t>();
    res.Im() /= LogTen<dtype_t>();
    return res;
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ Log10Fcnal<T, complex_domain_tag>::grad_t Log10Fcnal<T, complex_domain_tag>::gradient(T const& x){
    value_t res = LogFcnal<T, complex_domain_tag>::gradient(x);
    res.Re() /= LogTen<dtype_t>();
    res.Im() /= LogTen<dtype_t>();
    return res;
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x){ return Log10Fcnal<T>::eval(x); }

  // SINE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SinFcnal<T, domain_tag>::value_t SinFcnal<T, domain_tag>::eval(T const& x){ return std_math::sin(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SinFcnal<T, domain_tag>::grad_t SinFcnal<T, domain_tag>::gradient(T const& x){ return Cos(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, real_domain_tag>::value_t SinFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Sin(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, real_domain_tag>::grad_t SinFcnal<T, real_domain_tag>::gradient(T const& x){ return Cos(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::value_t SinFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = x.Re();
    auto const& b = x.Im();
    auto sa = Sin(a);
    auto chb = CosH(b);
    auto ca = Cos(a);
    auto shb = SinH(b);
    return value_t(sa*chb, ca*shb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinFcnal<T, complex_domain_tag>::grad_t SinFcnal<T, complex_domain_tag>::gradient(T const& x){ return Cos(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x){ return SinFcnal<T>::eval(x); }

  // COSINE
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CosFcnal<T, domain_tag>::value_t CosFcnal<T, domain_tag>::eval(T const& x){ return std_math::cos(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CosFcnal<T, domain_tag>::grad_t CosFcnal<T, domain_tag>::gradient(T const& x){ return -Sin(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, real_domain_tag>::value_t CosFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(Cos(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, real_domain_tag>::grad_t CosFcnal<T, real_domain_tag>::gradient(T const& x){ return -Sin(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::value_t CosFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = x.Re();
    auto const& b = x.Im();
    auto ca = Cos(a);
    auto chb = CosH(b);
    auto sa = Sin(a);
    auto shb = CosH(b);
    return value_t(ca*chb, -sa*shb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosFcnal<T, complex_domain_tag>::grad_t CosFcnal<T, complex_domain_tag>::gradient(T const& x){ return -Sin(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x){ return CosFcnal<T>::eval(x); }

  // TANGENT
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ TanFcnal<T, domain_tag>::value_t TanFcnal<T, domain_tag>::eval(T const& x){ return Sin(x)/Cos(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ TanFcnal<T, domain_tag>::grad_t TanFcnal<T, domain_tag>::gradient(T const& x){ auto r = Sec(x); return r*r; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x){ return TanFcnal<T>::eval(x); }

  // SECANT
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SecFcnal<T, domain_tag>::value_t SecFcnal<T, domain_tag>::eval(T const& x){ return One<dtype_t>/Cos(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SecFcnal<T, domain_tag>::grad_t SecFcnal<T, domain_tag>::gradient(T const& x){ return Sec(x)*Tan(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x){ return SecFcnal<T>::eval(x); }

  // COSECANT
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CscFcnal<T, domain_tag>::value_t CscFcnal<T, domain_tag>::eval(T const& x){ return One<dtype_t>/Sin(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CscFcnal<T, domain_tag>::grad_t CscFcnal<T, domain_tag>::gradient(T const& x){ return -Csc(x)*Cot(x);; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x){ return CscFcnal<T>::eval(x); }

  // COTANGENT
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CotFcnal<T, domain_tag>::value_t CotFcnal<T, domain_tag>::eval(T const& x){ return Cos(x)/Sin(x); }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CotFcnal<T, domain_tag>::grad_t CotFcnal<T, domain_tag>::gradient(T const& x){ auto r = Csc(x); return -r*r; }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x){ return CotFcnal<T>::eval(x); }

  // SINH
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, domain_tag>::value_t SinHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex - MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, domain_tag>::grad_t SinHFcnal<T, domain_tag>::gradient(T const& x){ return CosH(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::value_t SinHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(SinH(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, real_domain_tag>::grad_t SinHFcnal<T, real_domain_tag>::gradient(T const& x){ return CosH(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::value_t SinHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = x.Re();
    auto const& b = x.Im();
    auto sha = SinH(a);
    auto cb = Cos(b);
    auto cha = CosH(a);
    auto sb = Sin(b);
    return value_t(sha*cb, cha*sb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ SinHFcnal<T, complex_domain_tag>::grad_t SinHFcnal<T, complex_domain_tag>::gradient(T const& x){ return CosH(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x){ return SinHFcnal<T>::eval(x); }

  // COSH
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, domain_tag>::value_t CosHFcnal<T, domain_tag>::eval(T const& x){
    auto ex = Exp(x);
    return (ex + MultInverse(ex))/Two<dtype_t>();
  }
  template<typename T, typename domain_tag>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, domain_tag>::grad_t CosHFcnal<T, domain_tag>::gradient(T const& x){ return SinH(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::value_t CosHFcnal<T, real_domain_tag>::eval(T const& x){ return value_t(CosH(x.value())); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, real_domain_tag>::grad_t CosHFcnal<T, real_domain_tag>::gradient(T const& x){ return SinH(x); }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::value_t CosHFcnal<T, complex_domain_tag>::eval(T const& x){
    auto const& a = x.Re();
    auto const& b = x.Im();
    auto cha = CosH(a);
    auto cb = Cos(b);
    auto sha = SinH(a);
    auto sb = Sin(b);
    return value_t(cha*cb, sha*sb);
  }
  template<typename T>
  __CUDA_HOST_DEVICE__ CosHFcnal<T, complex_domain_tag>::grad_t CosHFcnal<T, complex_domain_tag>::gradient(T const& x){ return CosH(x); }
  template<typename T, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T>)> __CUDA_HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x){ return CosHFcnal<T>::eval(x); }

  // ADDITION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, domain_T, domain_U>::value_t AddFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()+y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::grad_x_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, real_domain_tag>::grad_y_t AddFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(One<reduced_data_t<U>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()+y.Re(), x.Im()+y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_x_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_y_t AddFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(One<reduced_data_t<U>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()+y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x+y.Re(), y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()+y, x.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()+y.Re(), y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_x_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_y_t AddFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(One<reduced_data_t<U>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()+y.value(), x.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_x_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ AddFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_y_t AddFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(One<reduced_data_t<U>>());
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename AddFcnal<T, U>::value_t Add(T const& x, U const& y){ return AddFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL((!std_ttraits::is_arithmetic_v<T> || !std_ttraits::is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y){ return Add(x, y); }

  // SUBTRACTION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, domain_T, domain_U>::value_t SubtractFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x+y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()-y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::grad_x_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::grad_y_t SubtractFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(MinusOne<reduced_data_t<U>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()-y.Re(), x.Im()-y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_x_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_y_t SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(MinusOne<reduced_data_t<U>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()-y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x-y.Re(), -y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()-y, x.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()-y.Re(), -y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_x_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_y_t SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(MinusOne<reduced_data_t<U>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()-y.value(), x.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_x_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(One<reduced_data_t<T>>());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_y_t SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(MinusOne<reduced_data_t<U>>());
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y){ return SubtractFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL((!std_ttraits::is_arithmetic_v<T> || !std_ttraits::is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y){ return Subtract(x, y); }

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, domain_T, domain_U>::value_t MultiplyFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x*y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()*y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::grad_x_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::grad_y_t MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(x.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()*y.Re() - x.Im()+y.Im(), x.Re()*y.Im() + x.Im()*y.Re());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_x_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_y_t MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(x.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()*y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x*y.Re(), x*y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()*y, x.Im()*y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()*y.Re(), x.value()*y.Im());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_x_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_y_t MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(x.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()*y.value(), x.Im()*y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_x_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return grad_x_t(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_y_t MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    return grad_y_t(x.value());
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y){ return MultiplyFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL((!std_ttraits::is_arithmetic_v<T> || !std_ttraits::is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y){ return Multiply(x, y); }

  // DIVISION
  template<typename T, typename U, typename domain_T, typename domain_U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, domain_T, domain_U>::value_t DivideFcnal<T, U, domain_T, domain_U>::eval(T const& x, U const& y){ return x/y; }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()/y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::grad_x_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return MultInverse(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, real_domain_tag>::grad_y_t DivideFcnal<T, U, real_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    auto yinv = MultInverse(y.value());
    auto myinvt = -yinv*yinv;
    return x.value()*myinvt.value();
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x.value()*MultInverse(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_x_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return MultInverse(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::grad_y_t DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    auto yinv = MultInverse(y.value());
    auto myinvt = -yinv*yinv;
    return x.value()*myinvt.value();
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return value_t(x/y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.value()/y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>::eval(T const& x, U const& y){
    return value_t(x.Re()/y, x.Im()/y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::value_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::eval(T const& x, U const& y){
    return x.value()*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_x_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_x(T const& x, U const& y){
    return MultInverse(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::grad_y_t DivideFcnal<T, U, real_domain_tag, complex_domain_tag>::gradient_y(T const& x, U const& y){
    auto yinv = MultInverse(y.value());
    auto myinvt = -yinv*yinv;
    return x.value()*myinvt.value();
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::value_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::eval(T const& x, U const& y){
    return x.value()*MultInverse(y);
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_x_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_x(T const& x, U const& y){
    return MultInverse(y.value());
  }
  template<typename T, typename U>
  __CUDA_HOST_DEVICE__ DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::grad_y_t DivideFcnal<T, U, complex_domain_tag, real_domain_tag>::gradient_y(T const& x, U const& y){
    auto yinv = MultInverse(y.value());
    auto myinvt = -yinv*yinv;
    return x.value()*myinvt.value();
  }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y){ return DivideFcnal<T, U>::eval(x, y); }
  template<typename T, typename U, ENABLE_IF_BOOL_IMPL((!std_ttraits::is_arithmetic_v<T> || !std_ttraits::is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator/(T const& x, U const& y){ return Divide(x, y); }

}


#endif
