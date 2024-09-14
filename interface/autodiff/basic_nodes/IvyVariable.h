#ifndef IVYVARIABLE_H
#define IVYVARIABLE_H


#include "config/IvyCompilerConfig.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/basic_nodes/IvyConstant.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyVariable;
  template<typename T> struct IvyNodeSelfRelations<IvyVariable<T>>;

  template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyVariable final :
    public IvyBaseNode,
    public real_domain_tag,
    public variable_value_tag
  {
  public:
    using dtype_t = T;
    using value_t = T;

  protected:
    value_t value_;
    value_t infinitesimal_;

  public:
    // Empty default constructor
    __HOST_DEVICE__ IvyVariable() : value_(0), infinitesimal_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyVariable(U const& value) : value_(__STATIC_CAST__(T, value)), infinitesimal_(0){}
    __HOST_DEVICE__ IvyVariable(T const& value) : value_(value), infinitesimal_(0){}
    __HOST_DEVICE__ IvyVariable(T&& value) : value_(std_util::move(value)), infinitesimal_(0){}
    __HOST_DEVICE__ IvyVariable(T const& value, T const& infinitesimal) : value_(value), infinitesimal_(infinitesimal){}
    __HOST_DEVICE__ IvyVariable(T&& value, T&& infinitesimal) : value_(std_util::move(value)), infinitesimal_(std_util::move(infinitesimal)){}
    template<typename U> __HOST_DEVICE__ IvyVariable(IvyVariable<U> const& other) : value_(__STATIC_CAST__(T, other.value())), infinitesimal_(0){}
    __HOST_DEVICE__ IvyVariable(IvyVariable<T> const& other) : value_(other.value_), infinitesimal_(other.infinitesimal_){}
    __HOST_DEVICE__ IvyVariable(IvyVariable<T>&& other) : value_(std_util::move(other.value_)), infinitesimal_(std_util::move(other.infinitesimal_)){}
    template<typename U> __HOST_DEVICE__ IvyVariable(IvyConstant<U> const& value) : value_(__STATIC_CAST__(T, value.value())), infinitesimal_(0){}
    __HOST_DEVICE__ IvyVariable(IvyConstant<T> const& value) : value_(value.value()), infinitesimal_(0){}
    __HOST_DEVICE__ IvyVariable(IvyConstant<T>&& value) : value_(value.value()), infinitesimal_(0){}
    __HOST_DEVICE__ ~IvyVariable(){}

    // Assignment operators
    template<typename U> __HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<U> const& other){ this->value_ = __STATIC_CAST__(T, other.value()); return *this; }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<T> const& other){ this->value_ = other.value_; this->infinitesimal_ = other.infinitesimal_; return *this; }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<T>&& other){ this->value_ = std_util::move(other.value_); this->infinitesimal_ = std_util::move(other.infinitesimal_); return *this; }

    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyVariable<T>& operator=(U const& value){ this->value_ = __STATIC_CAST__(T, value); return *this; }
    __HOST_DEVICE__ IvyVariable<T>& operator=(T const& value){ this->value_ = value; return *this; }
    __HOST_DEVICE__ IvyVariable<T>& operator=(T&& value){ this->value_ = std_util::move(value); return *this; }

    template<typename U> __HOST_DEVICE__ IvyVariable<T>& operator=(IvyConstant<U> const& value){ this->value_ = __STATIC_CAST__(T, value.value()); return *this; }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyConstant<T> const& value){ this->value_ = value.value(); return *this; }
    __HOST_DEVICE__ IvyVariable<T>& operator=(IvyConstant<T>&& value){ this->value_ = value.value(); return *this; }

    // Set functions
    __HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; }
    __HOST_DEVICE__ void set_infinitesimal(T const& infinitesimal){ this->infinitesimal_ = infinitesimal; }

    // Get functions
    __HOST_DEVICE__ value_t& value(){ return this->value_; }
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }
    __HOST_DEVICE__ value_t const& infinitesimal() const{ return this->infinitesimal_; }

    // IvyVariables are differentiable objects.
    __HOST_DEVICE__ bool is_differentiable() const{ return true; }

    friend struct IvyNodeSelfRelations<IvyVariable<T>>;
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyVariable<T>>{
    using type = IvyMath::IvyVariable<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct IvyNodeSelfRelations<IvyVariable<T>>{
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyVariable<T> const& x){ return true; }
    static __HOST_DEVICE__ void conjugate(IvyVariable<T>& x){}
    static constexpr bool is_conjugatable = false;
  };

  template<typename T> struct minimal_domain_type<T, real_domain_tag, variable_value_tag>{ using type = IvyVariable<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyVariablePtr_t = IvyThreadSafePtr_t< IvyVariable<T> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyVariablePtr_t<T> Variable(Args&&... args){ return make_IvyThreadSafePtr< IvyVariable<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyVariable<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyVariable<T> const& var){
      __PRINT_INFO__("Variable(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
