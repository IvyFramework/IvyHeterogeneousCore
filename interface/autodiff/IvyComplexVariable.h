#ifndef IVYCOMPLEXVARIABLE_H
#define IVYCOMPLEXVARIABLE_H


#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyCmath.h"
#include "autodiff/IvyBaseComplexVariable.h"
#include "stream/IvyStream.h"


template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyComplexVariable;
template<typename T> struct IvyNodeSelfRelations<IvyComplexVariable<T>>;

template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyComplexVariable final : public IvyBaseComplexVariable{
public:
  using dtype_t = T;
  using value_t = IvyComplexVariable<T>;

protected:
  dtype_t re;
  dtype_t im;

public:
  // Constructors
  __CUDA_HOST_DEVICE__ IvyComplexVariable() : re(0), im(0){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(T const& re_) : re(re_), im(0){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(T const& re_, T const& im_) : re(re_), im(im_){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(IvyComplexVariable const& other) : re(other.re), im(other.im){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(IvyComplexVariable const&& other) : re(std_util::move(other.re)), im(std_util::move(other.im)){}

  // Empty destructor
  __CUDA_HOST_DEVICE__ ~IvyComplexVariable(){}

  // Assignment operator
  __CUDA_HOST_DEVICE__ IvyComplexVariable& operator=(IvyComplexVariable const& other){ re = other.re; im = other.im; return *this; }

  // Accessors
  __CUDA_HOST_DEVICE__ T& Re(){ return re; }
  __CUDA_HOST_DEVICE__ T const& Re() const{ return re; }

  __CUDA_HOST_DEVICE__ T& Im(){ return im; }
  __CUDA_HOST_DEVICE__ T const& Im() const{ return im; }

  // Ensure that a complex variable can operate in just the same way as a variable or a constant.
  __CUDA_HOST_DEVICE__ value_t const& value() const{ return *this; }

  // Set functions
  __CUDA_HOST_DEVICE__ void set_real(T const& x){ re=x; }
  __CUDA_HOST_DEVICE__ void set_imaginary(T const& x){ im=x; }

  __CUDA_HOST_DEVICE__ void set_absval_phase(T const& v, T const& phi){
    re = v*std_math::cos(phi);
    im = v*std_math::sin(phi);
  }

  friend struct IvyNodeSelfRelations<IvyComplexVariable<T>>;
};

template<typename T> struct IvyNodeSelfRelations<IvyComplexVariable<T>>{
  static __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(T const& x){ return false; }
  static __CUDA_HOST_DEVICE__ void conjugate(T& x){ x.im = -x.im; }
  static constexpr bool is_conjugatable = true;
};


template<typename T> using IvyComplexVariablePtr_t = IvyThreadSafePtr_t< IvyComplexVariable<T> >;

template<typename T, typename... Args> __CUDA_HOST_DEVICE__ IvyComplexVariablePtr_t<T> Complex(Args&&... args){ return make_IvyThreadSafePtr< IvyComplexVariable<T> >(args...); }

namespace std_ivy{
  template<typename T> struct value_printout<IvyComplexVariable<T>>{
    static __CUDA_HOST_DEVICE__ void print(IvyComplexVariable<T> const& var){
      __PRINT_INFO__("Complex(");
      print_value(var.Re(), false); __PRINT_INFO__(", "); print_value(var.Im(), false);
      __PRINT_INFO__(")");
    }
  };
}


#endif
