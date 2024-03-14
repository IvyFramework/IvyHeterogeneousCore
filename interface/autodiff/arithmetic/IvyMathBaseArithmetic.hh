#ifndef IVYMATHBASEARITHMETIC_HH
#define IVYMATHBASEARITHMETIC_HH


#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyFunction.h"
#include "autodiff/arithmetic/IvyMathConstOps.h"
#include "autodiff/IvyMathTypes.h"


namespace IvyMath{
  // Get real part of a variable
  template<typename T, typename domain_tag = get_domain_t<T>> struct RealFcnal{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct RealFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x);

  // Get imaginary part of a variable
  template<typename T, typename domain_tag = get_domain_t<T>> struct ImagFcnal{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct ImagFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x);

  // Test to check whether value is an integer
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsIntegerFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsIntegerFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x);

  // Test to check whether value is real
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsRealFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsRealFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x);

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsImaginaryFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsImaginaryFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x);

  // NEGATION
  template<typename T, typename domain_tag = get_domain_t<T>> struct NegateFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct NegateFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<T> const& x);
  };
  template<typename T> struct NegateFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<T> const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x);
  template<typename T, ENABLE_IF_BOOL(!is_arithmetic_v<T> && !is_pointer_v<T>)>
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x);

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
    __CUDA_HOST__ IvyNegate(IvyThreadSafePtr_t<T> const& dep);
    __CUDA_HOST__ IvyNegate(IvyNegate const& other);
    __CUDA_HOST__ IvyNegate(IvyNegate&& other);

    __CUDA_HOST__ void eval() const override;
    __CUDA_HOST__ bool depends_on(IvyBaseNode const* node) const override;
    __CUDA_HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const override;
  };
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<T>::base_t> Negate(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<T>::base_t> operator-(T const& x);

  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag = get_domain_t<T>> struct MultInverseFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct MultInverseFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct MultInverseFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x);

  // SQUARE ROOT
  template<typename T, typename domain_tag = get_domain_t<T>> struct SqrtFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct SqrtFcnal<T, real_domain_tag>{
    using dtype_t = reduced_data_t<reduced_value_t<T>>;
    using value_t = IvyVariable<dtype_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct SqrtFcnal<T, complex_domain_tag>{
    using dtype_t = reduced_data_t<reduced_value_t<T>>;
    using value_t = IvyComplexVariable<dtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x);

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag = get_domain_t<T>> struct AbsFcnal{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct AbsFcnal<T, real_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct AbsFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x);

  // COMPLEX PHASE
  template<typename T, typename domain_tag = get_domain_t<T>> struct PhaseFcnal{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ constexpr value_t eval(T const& x);
    static __CUDA_HOST_DEVICE__ constexpr grad_t gradient(T const& x);
  };
  template<typename T> struct PhaseFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x);

  // CONJUGATION
  template<typename T, typename domain_tag = get_domain_t<T>> struct ConjugateFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = IvyConstant<dtype_t>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __CUDA_HOST_DEVICE__ constexpr grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x);

  // EXPONENTIAL
  template<typename T, typename domain_tag = get_domain_t<T>> struct ExpFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct ExpFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct ExpFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x);

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag = get_domain_t<T>> struct LogFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct LogFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct LogFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x);

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag = get_domain_t<T>> struct Log10Fcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct Log10Fcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct Log10Fcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x);

  // SINE
  template<typename T, typename domain_tag = get_domain_t<T>> struct SinFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct SinFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct SinFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x);

  // COSINE
  template<typename T, typename domain_tag = get_domain_t<T>> struct CosFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct CosFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct CosFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x);

  // TANGENT
  template<typename T, typename domain_tag = get_domain_t<T>> struct TanFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x);

  // SECANT
  template<typename T, typename domain_tag = get_domain_t<T>> struct SecFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x);

  // COSECANT
  template<typename T, typename domain_tag = get_domain_t<T>> struct CscFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x);

  // COTANGENT
  template<typename T, typename domain_tag = get_domain_t<T>> struct CotFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x);

  // SINH
  template<typename T, typename domain_tag = get_domain_t<T>> struct SinHFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct SinHFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct SinHFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x);

  // COSH
  template<typename T, typename domain_tag = get_domain_t<T>> struct CosHFcnal{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct CosHFcnal<T, real_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T> struct CosHFcnal<T, complex_domain_tag>{
    using value_t = reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x);

  // ADDITION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct AddFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename AddFcnal<T, U>::value_t Add(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y);

  template<typename T, typename U> class IvyAdd : public IvyFunction<more_precise_reduced_t<T, U>, get_domain_t<more_precise_t<T, U>>>{
  public:
    using base_t = IvyFunction<more_precise_reduced_t<T, U>, get_domain_t<more_precise_t<T, U>>>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
    using grad_t = typename base_t::grad_t;
    using Evaluator = AddFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>;

  protected:
    IvyThreadSafePtr_t<T> x;
    IvyThreadSafePtr_t<U> y;

  public:
    __CUDA_HOST__ IvyAdd(IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
    __CUDA_HOST__ IvyAdd(IvyAdd const& other);
    __CUDA_HOST__ IvyAdd(IvyAdd&& other);

    __CUDA_HOST__ void eval() const override;
    __CUDA_HOST__ bool depends_on(IvyBaseNode const* node) const override;
    __CUDA_HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const override;
  };
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<T, U>::base_t> Add(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<T, U>::base_t> operator+(T const& x, U const& y);

  // SUBTRACTION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct SubtractFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = IvyConstant<reduced_data_t<T>>;
    using grad_y_t = IvyConstant<reduced_data_t<U>>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = IvyConstant<reduced_data_t<T>>;
    using grad_y_t = IvyConstant<reduced_data_t<U>>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = IvyConstant<reduced_data_t<T>>;
    using grad_y_t = IvyConstant<reduced_data_t<U>>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = IvyConstant<reduced_data_t<T>>;
    using grad_y_t = IvyConstant<reduced_data_t<U>>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y);

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct MultiplyFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
  };
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y);

  template<typename T, typename U> class IvyMultiply : public IvyFunction<more_precise_reduced_t<T, U>, get_domain_t<more_precise_t<T, U>>>{
  public:
    using base_t = IvyFunction<more_precise_reduced_t<T, U>, get_domain_t<more_precise_t<T, U>>>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
    using grad_t = typename base_t::grad_t;
    using Evaluator = MultiplyFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>;

  protected:
    IvyThreadSafePtr_t<T> x;
    IvyThreadSafePtr_t<U> y;

  public:
    __CUDA_HOST__ IvyMultiply(IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
    __CUDA_HOST__ IvyMultiply(IvyMultiply const& other);
    __CUDA_HOST__ IvyMultiply(IvyMultiply&& other);

    __CUDA_HOST__ void eval() const override;
    __CUDA_HOST__ bool depends_on(IvyBaseNode const* node) const override;
    __CUDA_HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const override;
  };
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<T, U>::base_t> Multiply(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<T, U>::base_t> operator*(T const& x, U const& y);

  // DIVISION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct DivideFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = reduced_value_t<U>;
    using grad_y_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = reduced_value_t<U>;
    using grad_y_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = reduced_value_t<U>;
    using grad_y_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using grad_x_t = reduced_value_t<U>;
    using grad_y_t = value_t;
    static __CUDA_HOST_DEVICE__ value_t eval(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_x_t gradient_x(T const& x, U const& y);
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ grad_y_t gradient_y(T const& x, U const& y);
  };
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator*(T const& x, U const& y);

}


#endif
