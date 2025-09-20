#ifndef IVYFUNCTION_H
#define IVYFUNCTION_H


#include "autodiff/base_types/IvyBaseModifiable.h"
#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/arithmetic/IvyMathConstOps.h"


namespace IvyMath{
  template<typename precision_type, typename Domain, typename GradientDomain=Domain> class IvyFunction;
  template<typename precision_type, typename Domain, typename GradientDomain=Domain>
  using IvyFunctionPtr_t = IvyThreadSafePtr_t< IvyFunction<precision_type, Domain, GradientDomain> >;

  template<typename precision_type, typename Domain, typename GradientDomain>
  struct minimal_domain_type<precision_type, Domain, function_value_tag, GradientDomain>{
    using type = IvyFunction<std_ttraits::remove_cv_t<precision_type>, Domain, GradientDomain>;
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


  template<typename precision_type, typename Domain, typename GradientDomain>
  class IvyFunction{};

  template<typename precision_type, typename Domain>
  class IvyFunction<precision_type, Domain, Domain> :
    public IvyBaseNode,
    public IvyBaseModifiable,
    public Domain,
    public function_value_tag
  {
  public:
    // Domain tag of the function
    using domain_tag = Domain;
    // Type of the output
    using value_t = minimal_fcn_output_t<reduced_data_t<precision_type>, domain_tag, get_operability_t<precision_type>>;
    // Data type of the output
    using dtype_t = reduced_data_t<value_t>;
    // Gradient type
    using grad_t = IvyFunction<precision_type, domain_tag>;

  protected:
    // Output of the function
    // This is a pointer so that we can have a const-qualified eval function,
    // which would const-qualify the pointer, not the value itself.
    // That way, we can have the call to the const-qualified value() function run the evaluation.
    std_mem::unique_ptr<value_t> output;

  public:
    __HOST__ IvyFunction(IvyFunction const& other) :
      IvyBaseModifiable()
    {
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      output = std_mem::make_unique<value_t>(def_mem_type, nullptr, *(other.output));
    }
    __HOST__ IvyFunction(IvyFunction&& other) : IvyBaseModifiable(), output(std_util::move(other.output)){}
    template<typename... Args> __HOST__ IvyFunction(Args&&... default_value_args) :
      IvyBaseModifiable()
    {
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      output = std_mem::make_unique<value_t>(def_mem_type, nullptr, default_value_args...);
    }
    __HOST__ IvyFunction() :
      IvyBaseModifiable()
    {
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      output = std_mem::make_unique<value_t>(def_mem_type, nullptr);
    }
    ~IvyFunction() = default;

    // Function evaluator implementation
    virtual __HOST__ void eval() const = 0;
    // Function gradient implementation
    virtual __HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<IvyBaseNode> const& var) const = 0;
    // Since functions can only exist on the host, there is no issue with having a virtual depends_on call
    virtual __HOST__ bool depends_on(IvyBaseNode const* node) const{ return (this==node); }

    // Value of the function
    __HOST__ value_t const& value() const{ this->eval(); return *output; }

    template<typename Var> __HOST__ IvyThreadSafePtr_t<grad_t> gradient(IvyThreadSafePtr_t<Var> const& var) const{
      IvyThreadSafePtr_t<IvyBaseNode> base_var(var);
      return this->gradient(base_var);
    }
  };

  template<typename precision_type, typename Domain>
  class IvyFunction<precision_type, Domain, undefined_domain_tag> :
    public IvyBaseNode,
    public IvyBaseModifiable,
    public Domain,
    public function_value_tag
  {
  public:
    // Domain tag of the function
    using domain_tag = Domain;
    // Type of the output
    using value_t = minimal_fcn_output_t<reduced_data_t<precision_type>, domain_tag, get_operability_t<precision_type>>;
    // Data type of the output
    using dtype_t = reduced_data_t<value_t>;

  protected:
    // Output of the function
    // This is a pointer so that we can have a const-qualified eval function,
    // which would const-qualify the pointer, not the value itself.
    // That way, we can have the call to the const-qualified value() function run the evaluation.
    std_mem::unique_ptr<value_t> output;

  public:
    __HOST__ IvyFunction(IvyFunction const& other){
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      output = std_mem::make_unique<value_t>(def_mem_type, nullptr, *(other.output));
    }
    __HOST__ IvyFunction(IvyFunction&& other) : output(std_util::move(other.output)){}
    template<typename... Args> __HOST__ IvyFunction(Args&&... default_value_args){
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      output = std_mem::make_unique<value_t>(def_mem_type, nullptr, default_value_args...);
    }
    __HOST__ IvyFunction(){
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      output = std_mem::make_unique<value_t>(def_mem_type, nullptr);
    }
    ~IvyFunction() = default;

    // Function evaluator implementation
    virtual __HOST__ void eval() const = 0;
    // Since functions can only exist on the host, there is no issue with having a virtual depends_on call
    virtual __HOST__ bool depends_on(IvyBaseNode const* node) const{ return (this==node); }

    // Value of the function
    __HOST__ value_t const& value() const{ this->eval(); return *output; }
  };

  template<typename T, typename Domain = get_domain_t<T>, typename Operability = get_operability_t<T>> struct function_gradient{
    using value_t = unpack_if_function_t<T>;
    static __HOST__ IvyThreadSafePtr_t<value_t> get(
      T const& fcn, IvyThreadSafePtr_t<IvyBaseNode> const& var
    ){
      return make_IvyThreadSafePtr<value_t>(
        var.get_memory_type(),
        var.gpu_stream(),
        (var && depends_on(fcn, var.get()) ? One<fundamental_data_t<T>>() : Zero<fundamental_data_t<T>>())
      );
    }
  };
  // TODO: Even if we specialize for the tensor domain, we also need to subspecialize for functions, pointers, and other types in the operabiility argument.
  template<typename T> struct function_gradient<T, tensor_domain_tag, get_operability_t<T>>{
    static __HOST__ IvyThreadSafePtr_t<T> get(
      T const& fcn, IvyThreadSafePtr_t<IvyBaseNode> const& var
    ){
      using dtype_t = typename T::dtype_t;
      T res(fcn.shape(), Zero<dtype_t>());
      for (IvyTensorDim_t i=0; i<fcn.num_elements(); ++i){
        auto grad = function_gradient::get(fcn[i], var);
        res[i] = *grad;
      }
      return make_IvyThreadSafePtr(var.get_memory_type(), var.gpu_stream(), res);
    }
  };
  template<typename T> struct function_gradient<T, get_domain_t<T>, function_value_tag>{
    static __HOST__ IvyThreadSafePtr_t<typename T::grad_t> get(
      T const& fcn, IvyThreadSafePtr_t<IvyBaseNode> const& var
    ){
      return fcn.gradient(var);
    }
  };
  template<typename T, std_mem::IvyPointerType IPT> struct function_gradient<std_mem::IvyUnifiedPtr<T, IPT>>{
    static __HOST__ auto get(
      std_mem::IvyUnifiedPtr<T, IPT> const& fcn, IvyThreadSafePtr_t<IvyBaseNode> const& var
    ){
      return function_gradient<T>::get(*fcn, var);
    }
  };

  // Specializations for utilities
  template<typename... Args>
  struct IvyNodeSelfRelations<IvyFunction<Args...>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyFunction<Args...> const& x){
      return true;
    }
    static __HOST_DEVICE__ void conjugate(IvyFunction<Args...>& x){ conjugate(x.value()); }
    static constexpr bool is_conjugatable = is_conjugatable<typename IvyFunction<Args...>::value_t>;
  };
  template<typename precision_type, typename Domain, typename U>
  struct IvyNodeBinaryRelations<IvyFunction<precision_type, Domain>, U>{
    static __HOST_DEVICE__ bool depends_on(IvyFunction<precision_type, Domain> const& fcn, U* var){
      return fcn.depends_on(var);
    }
  };

  template<typename... Args>
  struct get_domain<IvyFunction<Args...>>{ using tag = typename IvyFunction<Args...>::domain_tag; };
  template<typename... Args>
  struct fundamental_data_type<IvyFunction<Args...>>{ using type = fundamental_data_t<reduced_data_t<IvyFunction<Args...>>>; };
  template<typename... Args>
  struct convert_to_real_type<IvyFunction<Args...>>{ using type = convert_to_real_t<reduced_value_t<IvyFunction<Args...>>>; };

}

namespace std_ivy{
  template<typename... Args>
  struct value_printout<IvyMath::IvyFunction<Args...>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyFunction<Args...> const& var){
      __PRINT_INFO__("Function(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
