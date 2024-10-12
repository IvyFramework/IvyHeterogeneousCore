#ifndef IVYCONSTANT_H
#define IVYCONSTANT_H


#include "config/IvyCompilerConfig.h"
#include "stream/IvyStream.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"


namespace IvyMath{
  template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyConstant final :
    public IvyBaseNode,
    public real_domain_tag,
    public constant_value_tag
  {
  public:
    using dtype_t = T;
    using value_t = T;

  protected:
    value_t value_;

  public:
    // Constructors
    __HOST_DEVICE__ IvyConstant() : value_(0){}
    template<typename U, ENABLE_IF_ARITHMETIC(U)> __HOST_DEVICE__ IvyConstant(U const& value) : value_(__STATIC_CAST__(T, value)){}
    __HOST_DEVICE__ IvyConstant(T const& value) : value_(value){}
    __HOST_DEVICE__ IvyConstant(T&& value) : value_(std_util::move(value)){}
    template<typename U> __HOST_DEVICE__ IvyConstant(IvyConstant<U> const& other) : value_(__STATIC_CAST__(T, other.value())){}
    __HOST_DEVICE__ IvyConstant(IvyConstant<T> const& other) : value_(other.value_){}
    __HOST_DEVICE__ IvyConstant(IvyConstant<T>&& other) : value_(std_util::move(other.value_)){}

    // Assignment operators
    template<typename U> __HOST_DEVICE__ IvyConstant<T>& operator=(IvyConstant<U> const& other){ this->value_ = __STATIC_CAST__(T, other.value()); return *this; }
    __HOST_DEVICE__ IvyConstant<T>& operator=(IvyConstant<T> const& other){ this->value_ = other.value_; return *this; }
    __HOST_DEVICE__ IvyConstant<T>& operator=(IvyConstant<T>&& other){ this->value_ = std_util::move(other.value_); return *this; }

    // Empty virtual destructor
    __HOST_DEVICE__ ~IvyConstant(){}

    // Get function
    __HOST_DEVICE__ value_t const& value() const{ return this->value_; }
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyConstant<T>>{
    using type = IvyMath::IvyConstant<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct minimal_domain_type<T, real_domain_tag, constant_value_tag>{ using type = IvyConstant<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyConstantPtr_t = IvyThreadSafePtr_t< IvyConstant<T> >;

  template<typename T, typename... Args> __HOST_DEVICE__ IvyConstantPtr_t<T> Constant(Args&&... args){ return make_IvyThreadSafePtr< IvyConstant<T> >(args...); }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyConstant<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyConstant<T> const& var){
      __PRINT_INFO__("Constant(");
      print_value(var.value(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
