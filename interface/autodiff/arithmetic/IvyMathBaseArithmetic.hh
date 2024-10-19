#ifndef IVYMATHBASEARITHMETIC_HH
#define IVYMATHBASEARITHMETIC_HH


#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyFunction.h"
#include "autodiff/arithmetic/IvyMathConstOps.h"
#include "autodiff/IvyMathTypes.h"


namespace IvyMath{
  /*
  IvyRegularFunction_1D:
  This is a master class for regular 1D functions.
  Unless we have any sepcial cases in return types, i.e., not the return type of reduced_value_t<T> required here,
  this master class should be used with partial specializations for different Evaluator types.
  */
  template<typename T, typename Evaluator, typename precision_type = unpacked_reduced_value_t<T>, typename Domain = get_domain_t<T>>
  class IvyRegularFunction_1D : public IvyFunction<precision_type, Domain>
  {
  public:
    using base_t = IvyFunction<precision_type, Domain>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
    using grad_t = typename base_t::grad_t;
    using evaluator_t = Evaluator;

  protected:
    IvyThreadSafePtr_t<T> dep;

  public:
    __HOST__ IvyRegularFunction_1D(IvyThreadSafePtr_t<T> const& dep);
    __HOST__ IvyRegularFunction_1D(IvyRegularFunction_1D const& other);
    __HOST__ IvyRegularFunction_1D(IvyRegularFunction_1D&& other);

    __HOST__ void eval() const override;
    __HOST__ bool depends_on(IvyBaseNode const* node) const override;
    __HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const override;
  };
  /*
  IvyRegularFunction_2D:
  This is a master class for regular 2D functions.
  Unless we have any sepcial cases in return types, i.e., not the return type of more_precise_reduced_t<T, U> required here,
  this master class should be used with partial specializations for different Evaluator types.
  */
  template<typename T, typename U, typename Evaluator, typename precision_type = more_precise_reduced_t<T, U>, typename Domain = get_domain_t<more_precise_t<T, U>>>
  class IvyRegularFunction_2D : public IvyFunction<precision_type, Domain>
  {
  public:
    using base_t = IvyFunction<precision_type, Domain>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
    using grad_t = typename base_t::grad_t;
    using evaluator_t = Evaluator;

  protected:
    IvyThreadSafePtr_t<T> x;
    IvyThreadSafePtr_t<U> y;

  public:
    __HOST__ IvyRegularFunction_2D(IvyThreadSafePtr_t<T> const& x, IvyThreadSafePtr_t<U> const& y);
    __HOST__ IvyRegularFunction_2D(IvyRegularFunction_2D const& other);
    __HOST__ IvyRegularFunction_2D(IvyRegularFunction_2D&& other);

    __HOST__ void eval() const override;
    __HOST__ bool depends_on(IvyBaseNode const* node) const override;
    __HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const override;
  };

  // Get real part of a variable
  template<typename T, typename domain_tag = get_domain_t<T>> struct RealFcnal{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct RealFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename RealFcnal<T>::value_t Real(T const& x);

  // Get imaginary part of a variable
  template<typename T, typename domain_tag = get_domain_t<T>> struct ImagFcnal{
    using value_t = convert_to_real_t<T>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct ImagFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ImagFcnal<T>::value_t Imag(T const& x);

  // Test to check whether value is an integer
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsIntegerFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsIntegerFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename IsIntegerFcnal<T>::value_t IsInteger(T const& x);

  // Test to check whether value is real
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsRealFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsRealFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename IsRealFcnal<T>::value_t IsReal(T const& x);

  // Test to check whether value is imaginary
  template<typename T, typename domain_tag = get_domain_t<T>> struct IsImaginaryFcnal{
    using value_t = bool;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct IsImaginaryFcnal<T, complex_domain_tag>{
    using value_t = bool;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename IsImaginaryFcnal<T>::value_t IsImaginary(T const& x);

  // NEGATION
  template<typename T, typename domain_tag = get_domain_t<T>> struct NegateFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct NegateFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct NegateFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyNegate = IvyRegularFunction_1D<T, NegateFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename NegateFcnal<T>::value_t Negate(T const& x);
  template<typename T, ENABLE_IF_BOOL(!is_arithmetic_v<T> && !is_pointer_v<T>)>
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ typename NegateFcnal<T>::value_t operator-(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> Negate(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyNegate<typename T::element_type>::base_t> operator-(T const& x);

  // MULTIPLICATIVE INVERSE
  template<typename T, typename domain_tag = get_domain_t<T>> struct MultInverseFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct MultInverseFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct MultInverseFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyMultInverse = IvyRegularFunction_1D<T, MultInverseFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename MultInverseFcnal<T>::value_t MultInverse(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultInverse<typename T::element_type>::base_t> MultInverse(T const& x);

  // SQUARE ROOT
  template<typename T, typename domain_tag = get_domain_t<T>> struct SqrtFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct SqrtFcnal<T, real_domain_tag>{
    using dtype_t = reduced_data_t<unpacked_reduced_value_t<T>>;
    using value_t = IvyVariable<dtype_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct SqrtFcnal<T, complex_domain_tag>{
    using dtype_t = reduced_data_t<unpacked_reduced_value_t<T>>;
    using value_t = IvyComplexVariable<dtype_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvySqrt = IvyRegularFunction_1D<T, SqrtFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SqrtFcnal<T>::value_t Sqrt(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySqrt<typename T::element_type>::base_t> Sqrt(T const& x);

  // ABSOLUTE VALUE
  template<typename T, typename domain_tag = get_domain_t<T>> struct AbsFcnal{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct AbsFcnal<T, real_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct AbsFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AbsFcnal<T>::value_t Abs(T const& x);

  // COMPLEX PHASE
  template<typename T, typename domain_tag = get_domain_t<T>> struct PhaseFcnal{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ constexpr value_t eval(T const& x);
  };
  template<typename T> struct PhaseFcnal<T, complex_domain_tag>{
    using value_t = convert_to_real_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename PhaseFcnal<T>::value_t Phase(T const& x);

  // CONJUGATION
  template<typename T, typename domain_tag = get_domain_t<T>> struct ConjugateFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ConjugateFcnal<T>::value_t Conjugate(T const& x);

  // EXPONENTIAL
  template<typename T, typename domain_tag = get_domain_t<T>> struct ExpFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ExpFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ExpFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyExp = IvyRegularFunction_1D<T, ExpFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ExpFcnal<T>::value_t Exp(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyExp<typename T::element_type>::base_t> Exp(T const& x);

  // LOG (NATURAL LOG)
  template<typename T, typename domain_tag = get_domain_t<T>> struct LogFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct LogFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct LogFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyLog = IvyRegularFunction_1D<T, LogFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename LogFcnal<T>::value_t Log(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLog<typename T::element_type>::base_t> Log(T const& x);

  // LOG10 (BASE=10 LOG)
  template<typename T, typename domain_tag = get_domain_t<T>> struct Log10Fcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct Log10Fcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct Log10Fcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyLog10 = IvyRegularFunction_1D<T, Log10Fcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename Log10Fcnal<T>::value_t Log10(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyLog10<typename T::element_type>::base_t> Log10(T const& x);

  // SINE
  template<typename T, typename domain_tag = get_domain_t<T>> struct SinFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct SinFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct SinFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvySin = IvyRegularFunction_1D<T, SinFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SinFcnal<T>::value_t Sin(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySin<typename T::element_type>::base_t> Sin(T const& x);

  // COSINE
  template<typename T, typename domain_tag = get_domain_t<T>> struct CosFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct CosFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct CosFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyCos = IvyRegularFunction_1D<T, CosFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CosFcnal<T>::value_t Cos(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCos<typename T::element_type>::base_t> Cos(T const& x);

  // TANGENT
  template<typename T, typename domain_tag = get_domain_t<T>> struct TanFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyTan = IvyRegularFunction_1D<T, TanFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename TanFcnal<T>::value_t Tan(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyTan<typename T::element_type>::base_t> Tan(T const& x);

  // SECANT
  template<typename T, typename domain_tag = get_domain_t<T>> struct SecFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvySec = IvyRegularFunction_1D<T, SecFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SecFcnal<T>::value_t Sec(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySec<typename T::element_type>::base_t> Sec(T const& x);

  // COSECANT
  template<typename T, typename domain_tag = get_domain_t<T>> struct CscFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyCsc = IvyRegularFunction_1D<T, CscFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CscFcnal<T>::value_t Csc(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCsc<typename T::element_type>::base_t> Csc(T const& x);

  // COTANGENT
  template<typename T, typename domain_tag = get_domain_t<T>> struct CotFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, domain_tag>>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyCot = IvyRegularFunction_1D<T, CotFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CotFcnal<T>::value_t Cot(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCot<typename T::element_type>::base_t> Cot(T const& x);

  // SINH
  template<typename T, typename domain_tag = get_domain_t<T>> struct SinHFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct SinHFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct SinHFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvySinH = IvyRegularFunction_1D<T, SinHFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SinHFcnal<T>::value_t SinH(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySinH<typename T::element_type>::base_t> SinH(T const& x);

  // COSH
  template<typename T, typename domain_tag = get_domain_t<T>> struct CosHFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct CosHFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct CosHFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyCosH = IvyRegularFunction_1D<T, CosHFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename CosHFcnal<T>::value_t CosH(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyCosH<typename T::element_type>::base_t> CosH(T const& x);

  // ERF
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyErf = IvyRegularFunction_1D<T, ErfFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfFcnal<T>::value_t Erf(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErf<typename T::element_type>::base_t> Erf(T const& x);

  // ERFC
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfcFcnal {
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfcFcnal<T, real_domain_tag> {
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfcFcnal<T, complex_domain_tag> {
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyErfc = IvyRegularFunction_1D<T, ErfcFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfcFcnal<T>::value_t Erfc(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErfc<typename T::element_type>::base_t> Erfc(T const& x);

  // FADDEEVA
  template<typename T, typename domain_tag = get_domain_t<T>> struct FaddeevaFcnal{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct FaddeevaFcnal<T, real_domain_tag>{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct FaddeevaFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyFaddeeva = IvyRegularFunction_1D<
    T,
    FaddeevaFcnal<unpack_if_function_t<T>>,
    unpacked_reduced_value_t< typename FaddeevaFcnal<unpack_if_function_t<T>>::value_t >,
    get_domain_t< typename FaddeevaFcnal<unpack_if_function_t<T>>::value_t >
  >;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename FaddeevaFcnal<T>::value_t Faddeeva(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyFaddeeva<typename T::element_type>::base_t> Faddeeva(T const& x);

  // ERF-FAST
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfFastFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfFastFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfFastFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyErfFast = IvyRegularFunction_1D<T, ErfFastFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfFastFcnal<T>::value_t ErfFast(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErfFast<typename T::element_type>::base_t> ErfFast(T const& x);

  // ERFC-FAST
  template<typename T, typename domain_tag = get_domain_t<T>> struct ErfcFastFcnal{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct ErfcFastFcnal<T, real_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, real_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct ErfcFastFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyErfcFast = IvyRegularFunction_1D<T, ErfcFastFcnal<unpack_if_function_t<T>>>;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename ErfcFastFcnal<T>::value_t ErfcFast(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyErfcFast<typename T::element_type>::base_t> ErfcFast(T const& x);

  // FADDEEVA-FAST
  template<typename T, typename domain_tag = get_domain_t<T>> struct FaddeevaFastFcnal{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x);
  };
  template<typename T> struct FaddeevaFastFcnal<T, real_domain_tag>{
    using value_t = convert_to_complex_t<unpacked_reduced_value_t<T>>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> struct FaddeevaFastFcnal<T, complex_domain_tag>{
    using value_t = unpacked_reduced_value_t<T>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyThreadSafePtr_t<IvyFunction<value_t, complex_domain_tag>>;
    static __HOST_DEVICE__ value_t eval(T const& x);
    template<typename X_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(IvyThreadSafePtr_t<X_t> const& x);
  };
  template<typename T> using IvyFaddeevaFast = IvyRegularFunction_1D<
    T,
    FaddeevaFastFcnal<unpack_if_function_t<T>>,
    unpacked_reduced_value_t< typename FaddeevaFastFcnal<unpack_if_function_t<T>>::value_t >,
    get_domain_t< typename FaddeevaFastFcnal<unpack_if_function_t<T>>::value_t >
  >;
  template<typename T, ENABLE_IF_BOOL(!is_pointer_v<T>)> __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename FaddeevaFastFcnal<T>::value_t FaddeevaFast(T const& x);
  template<typename T, ENABLE_IF_BOOL(is_pointer_v<T>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyFaddeevaFast<typename T::element_type>::base_t> FaddeevaFast(T const& x);

  /****************/
  /* 2D FUNCTIONS */
  /****************/

  // COMPARISON OPERATORS
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct EqualFcnal{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct EqualFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = bool;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> using IvyEqual = IvyRegularFunction_2D<T, U, EqualFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>, bool, get_domain_t<IvyConstant<bool>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename EqualFcnal<T, U>::value_t Equal(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyEqual<typename T::element_type, typename U::element_type>::base_t> Equal(T const& x, U const& y);

  // ADDITION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct AddFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct AddFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> using IvyAdd = IvyRegularFunction_2D<T, U, AddFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AddFcnal<T, U>::value_t Add(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename AddFcnal<T, U>::value_t operator+(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> Add(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyAdd<typename T::element_type, typename U::element_type>::base_t> operator+(T const& x, U const& y);

  // SUBTRACTION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct SubtractFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyConstantPtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct SubtractFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyComplexVariablePtr_t<fndtype_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> using IvySubtract = IvyRegularFunction_2D<T, U, SubtractFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t Subtract(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename SubtractFcnal<T, U>::value_t operator-(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> Subtract(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvySubtract<typename T::element_type, typename U::element_type>::base_t> operator-(T const& x, U const& y);

  // MULTIPLICATION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct MultiplyFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct MultiplyFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> using IvyMultiply = IvyRegularFunction_2D<T, U, MultiplyFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t Multiply(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename MultiplyFcnal<T, U>::value_t operator*(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> Multiply(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyMultiply<typename T::element_type, typename U::element_type>::base_t> operator*(T const& x, U const& y);

  // DIVISION
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct DivideFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct DivideFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> using IvyDivide = IvyRegularFunction_2D<T, U, DivideFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t Divide(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(!(is_arithmetic_v<T> && is_arithmetic_v<U>) && !is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename DivideFcnal<T, U>::value_t operator/(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> Divide(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T> && is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyDivide<typename T::element_type, typename U::element_type>::base_t> operator/(T const& x, U const& y);

  // POWER
  template<typename T, typename U, typename domain_T = get_domain_t<T>, typename domain_U = get_domain_t<U>> struct PowFcnal{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, real_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, complex_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, arithmetic_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, real_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, arithmetic_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, complex_domain_tag, arithmetic_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, real_domain_tag, complex_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> struct PowFcnal<T, U, complex_domain_tag, real_domain_tag>{
    using value_t = more_precise_reduced_t<T, U>;
    using dtype_t = reduced_data_t<value_t>;
    using fndtype_t = fundamental_data_t<value_t>;
    using grad_t = IvyFunctionPtr_t<value_t, get_domain_t<more_precise_t<T, U>>>;
    static __HOST_DEVICE__ value_t eval(T const& x, U const& y);
    template<typename X_t, typename Y_t>
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ grad_t gradient(unsigned char ivar, IvyThreadSafePtr_t<X_t> const& x, IvyThreadSafePtr_t<Y_t> const& y);
  };
  template<typename T, typename U> using IvyPow = IvyRegularFunction_2D<T, U, PowFcnal<unpack_if_function_t<T>, unpack_if_function_t<U>>>;
  template<typename T, typename U, ENABLE_IF_BOOL(!is_pointer_v<T> && !is_pointer_v<U>)>
  __INLINE_FCN_FORCE__ __HOST_DEVICE__ typename PowFcnal<T, U>::value_t Pow(T const& x, U const& y);
  template<typename T, typename U, ENABLE_IF_BOOL(is_pointer_v<T>&& is_pointer_v<U>)>
  __HOST_DEVICE__ IvyThreadSafePtr_t<typename IvyPow<typename T::element_type, typename U::element_type>::base_t> Pow(T const& x, U const& y);

}


#endif
