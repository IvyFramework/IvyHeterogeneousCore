#ifndef IVYMATHFUNCTIONPRIMITIVES_HH
#define IVYMATHFUNCTIONPRIMITIVES_HH


#include "autodiff/basic_nodes/IvyFunction.h"
#include "autodiff/IvyMathTypes.h"


namespace IvyMath{
  template<
    typename F,
    typename T,
    std_ttraits::enable_if_t<is_function_v<F> && std_ttraits::is_base_of_v<IvyClientManager, T>, bool> = true
  > struct function_data_client_updator{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void update(IvyThreadSafePtr_t<F> const& fcn, IvyThreadSafePtr_t<T> const& dep){
      dep->add_client(fcn);
    }
  };
  template<typename F, typename T>
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void add_fcn_to_clients(IvyThreadSafePtr_t<F> const& fcn, IvyThreadSafePtr_t<T> const& dep){
    function_data_client_updator<F, T>::update(fcn, dep);
  }

  /*
  IvyRegularFunction_1D:
  This is a master class for regular 1D functions.
  Unless we have any special cases in return types, i.e., not the return type of reduced_value_t<T> required here,
  this master class should be used with partial specializations for different Evaluator types.
  */
  template<
    typename T, typename Evaluator,
    typename precision_type = unpacked_reduced_value_t<T>,
    typename Domain = get_domain_t<T>,
    typename GradientDomain = Domain
  > class IvyRegularFunction_1D : public IvyFunction<precision_type, Domain, GradientDomain>
  {
  public:
    using base_t = IvyFunction<precision_type, Domain, GradientDomain>;
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
  template<
    typename T, typename Evaluator,
    typename precision_type,
    typename Domain
  > class IvyRegularFunction_1D<T, Evaluator, precision_type, Domain, undefined_domain_tag> :
    public IvyFunction<precision_type, Domain, undefined_domain_tag>
  {
  public:
    using base_t = IvyFunction<precision_type, Domain, undefined_domain_tag>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
    using evaluator_t = Evaluator;

  protected:
    IvyThreadSafePtr_t<T> dep;

  public:
    __HOST__ IvyRegularFunction_1D(IvyThreadSafePtr_t<T> const& dep);
    __HOST__ IvyRegularFunction_1D(IvyRegularFunction_1D const& other);
    __HOST__ IvyRegularFunction_1D(IvyRegularFunction_1D&& other);

    __HOST__ void eval() const override;
    __HOST__ bool depends_on(IvyBaseNode const* node) const override;
  };
  /*
  IvyRegularFunction_2D:
  This is a master class for regular 2D functions.
  Unless we have any special cases in return types, i.e., not the return type of more_precise_reduced_t<T, U> required here,
  this master class should be used with partial specializations for different Evaluator types.
  */
  template<
    typename T, typename U, typename Evaluator,
    typename precision_type = more_precise_reduced_t<T, U>,
    typename Domain = get_domain_t<more_precise_t<T, U>>,
    typename GradientDomain = Domain
  > class IvyRegularFunction_2D : public IvyFunction<precision_type, Domain, GradientDomain>
  {
  public:
    using base_t = IvyFunction<precision_type, Domain, GradientDomain>;
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
  template<
    typename T, typename U, typename Evaluator,
    typename precision_type,
    typename Domain
  > class IvyRegularFunction_2D<T, U, Evaluator, precision_type, Domain, undefined_domain_tag> :
    public IvyFunction<precision_type, Domain, undefined_domain_tag>
  {
  public:
    using base_t = IvyFunction<precision_type, Domain, undefined_domain_tag>;
    using value_t = typename base_t::value_t;
    using dtype_t = typename base_t::dtype_t;
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
  };

  /*
  IvyConditionalFunction_1D:
  This is a master class typedef for 1D conditionals.
  */
  template<typename T, typename Evaluator>
  using IvyConditionalFunction_1D = IvyRegularFunction_1D<T, Evaluator, bool, real_domain_tag, undefined_domain_tag>;

  /*
  IvyConditionalFunction_2D:
  This is a master class typedef for 2D conditional functions.
  */
  template<typename T, typename U, typename Evaluator>
  using IvyConditionalFunction_2D = IvyRegularFunction_2D<T, U, Evaluator, bool, real_domain_tag, undefined_domain_tag>;

}


#endif
