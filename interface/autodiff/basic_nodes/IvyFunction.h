#ifndef IVYFUNCTION_H
#define IVYFUNCTION_H


#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/IvyBaseMathTypes.h"


namespace _fcn_eval{
  template<typename T, ENABLE_IF_BOOL(IvyMath::is_constant_v<T> || IvyMath::is_variable_v<T>)>
  __CUDA_HOST_DEVICE__ void eval(T& fcn){}
  template<typename T, ENABLE_IF_BOOL(!IvyMath::is_tensor_v<T> && IvyMath::is_function_v<T>)>
  __CUDA_HOST__ void eval(T& fcn){ fcn.eval(); }
  template<typename T, ENABLE_IF_BOOL(IvyMath::is_tensor_v<T> && IvyMath::is_function_v<T>)>
  __CUDA_HOST__ void eval(T& fcn){
    for (IvyTensorDim_t i=0; i<fcn.num_elements(); ++i) _fcn_eval::eval(fcn[i]);
  }

  template<typename T, ENABLE_IF_BOOL(IvyMath::is_constant_v<T> || IvyMath::is_variable_v<T>)>
  __CUDA_HOST_DEVICE__ void eval(IvyThreadSafePtr_t<T>& fcn){}
  template<typename T, ENABLE_IF_BOOL(!IvyMath::is_tensor_v<T> && IvyMath::is_function_v<T>)>
  __CUDA_HOST__ void eval(IvyThreadSafePtr_t<T>& fcn){ fcn->eval(); }
  template<typename T, ENABLE_IF_BOOL(IvyMath::is_tensor_v<T> && IvyMath::is_function_v<T>)>
  __CUDA_HOST__ void eval(IvyThreadSafePtr_t<T>& fcn){
    for (IvyTensorDim_t i=0; i<fcn->num_elements(); ++i) _fcn_eval::eval((*fcn)[i]);
  }
};
// Shorthand to eval functions and other objects
#define eval_fcn(fcn) _fcn_eval::eval(fcn)

namespace IvyMath{
  template<typename... Args> class IvyFunction;
  template<typename... Args> using IvyFunctionPtr_t = IvyThreadSafePtr_t< IvyFunction<Args...> >;

  template<typename precision_type, typename Domain, typename... Args>
  struct minimal_domain_type<precision_type, Domain, function_value_tag>{
    using type = IvyFunction<std_ttraits::remove_cv_t<precision_type>, Domain, Args...>;
  };
  template<typename precision_type, typename Domain>
  struct minimal_fcn_output_type<precision_type, Domain, function_value_tag>{
    using dtype_t = reduced_data_t<std_ttraits::remove_cv_t<precision_type>>;
    using type = std_ttraits::conditional_t<
      is_tensor_v<Domain>,
      IvyTensor<dtype_t>, std_ttraits::conditional_t<
        is_complex_v<Domain>,
        IvyComplexVariable<dtype_t>, std_ttraits::conditional_t<
          is_real_v<Domain>,
          IvyVariable<dtype_t>, IvyConstant<dtype_t>
        >
      >
    >;
  };

  template<typename Evaluator> struct evaluator_gradient_type{
    template<typename U> static __CUDA_HOST_DEVICE__ auto test_vtype(int) -> typename Evaluator::grad_t;
    template<typename U> static __CUDA_HOST_DEVICE__ auto test_vtype(...) -> void;
    using type = decltype(test_vtype<Evaluator>(0));
  };
  template<typename Evaluator> using evaluator_gradient_t = typename evaluator_gradient_type<Evaluator>::type;
  template<typename Evaluator> inline constexpr bool has_gradient_v = !std_ttraits::is_void_v<evaluator_gradient_t<Evaluator>>;

  template<typename precision_type, typename Domain, typename Evaluator, bool HasGradient = has_gradient_v<Evaluator>>
  class IvyFunction :
    public IvyBaseNode,
    public Domain,
    public function_value_tag
  {
  public:
    // Domain tag of the function
    using domain_tag = Domain;
    // Type of the output
    using value_t = minimal_fcn_output_t<precision_type, domain_tag>;
    // Data type of the output
    using dtype_t = reduced_data_t<value_t>;
    // Evaluator type
    using evaluator_t = Evaluator;
    // Gradient type
    using grad_t = evaluator_gradient_t<Evaluator>;

  protected:
    value_t output;
    evaluator_t evaluator;

  public:
    IvyFunction() = default;
    __CUDA_HOST__ IvyFunction(value_t const& def_val) : output(def_val){}
    __CUDA_HOST__ IvyFunction(IvyFunction const& other) : output(other.output){}
    __CUDA_HOST__ IvyFunction(IvyFunction&& other) : output(std_util::move(other.output)){}
    ~IvyFunction() = default;

    // Value of the function
    __CUDA_HOST__ value_t& value(){ return output; }
    __CUDA_HOST__ value_t const& value() const{ return output; }

    virtual __CUDA_HOST__ void eval() = 0;

    template<typename Var> IvyThreadSafePtr_t<typename evaluator_t::grad_t> gradient(IvyThreadSafePtr_t<Var> const& var) const{
      return make_IvyThreadSafePtr(var.get_memory_type(), var.gpu_stream(), evaluator.gradient(this, var));
    }
  };
  template<typename precision_type, typename Domain, typename Evaluator>
  class IvyFunction<precision_type, Domain, Evaluator, false> :
    public IvyBaseNode,
    public Domain,
    public function_value_tag
  {
  public:
    // Domain tag of the function
    using domain_tag = Domain;
    // Type of the output
    using value_t = minimal_fcn_output_t<precision_type, domain_tag>;
    // Data type of the output
    using dtype_t = reduced_data_t<value_t>;
    // Evaluator type
    using evaluator_t = Evaluator;

  protected:
    value_t output;

  public:
    IvyFunction() = default;
    __CUDA_HOST__ IvyFunction(value_t const& def_val) : output(def_val){}
    __CUDA_HOST__ IvyFunction(IvyFunction const& other) : output(other.output){}
    __CUDA_HOST__ IvyFunction(IvyFunction&& other) : output(std_util::move(other.output)){}
    ~IvyFunction() = default;

    // Value of the function
    __CUDA_HOST__ value_t& value(){ return output; }
    __CUDA_HOST__ value_t const& value() const{ return output; }
  };

  // Specializations for utilities
  template<typename precision_type, typename Domain, typename Evaluator, bool HasGradient>
  struct IvyNodeSelfRelations<IvyFunction<precision_type, Domain, Evaluator, HasGradient>>{
    static __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(IvyFunction<precision_type, Domain, Evaluator, HasGradient> const& x){
      return HasGradient;
    }
    static __CUDA_HOST_DEVICE__ void conjugate(IvyFunction<precision_type, Domain, Evaluator, HasGradient>& x){
      conjugate(x.value());
    }
    static constexpr bool is_conjugatable = is_conjugatable<typename IvyFunction<precision_type, Domain, Evaluator, HasGradient>::value_t>;
  };

  template<typename... Args>
  struct get_domain<IvyFunction<Args...>>{ using tag = typename IvyFunction<Args...>::domain_tag; };
  template<typename... Args>
  struct fundamental_data_type<IvyFunction<Args...>>{ using type = fundamental_data_t<reduced_data_t<IvyFunction<Args...>>>; };
  template<typename... Args>
  struct convert_to_real_type<IvyFunction<Args...>>{ using type = convert_to_real_t<reduced_value_t<IvyFunction<Args...>>>; };

}


#endif
