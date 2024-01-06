#ifndef IVYCOMPLEXVARIABLE_H
#define IVYCOMPLEXVARIABLE_H


#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyCmath.h"
#include "autodiff/IvyBaseComplexVariable.h"
#include "stream/IvyStream.h"


template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyComplexVariable final : public IvyBaseComplexVariable{
public:
  using dtype_t = T;
  using value_t = IvyComplexVariable<T>;

protected:
  dtype_t re;
  dtype_t im;

public:
  // Constructors
  __CUDA_HOST_DEVICE__ IvyComplexVariable() : IvyBaseComplexVariable(), re(0), im(0){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(T const& re_) : IvyBaseComplexVariable(), re(re_), im(0){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(T const& re_, T const& im_) : IvyBaseComplexVariable(), re(re_), im(im_){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(IvyComplexVariable const& other) : IvyBaseComplexVariable(other), re(other.re), im(other.im){}
  __CUDA_HOST_DEVICE__ IvyComplexVariable(IvyComplexVariable const&& other) : IvyBaseComplexVariable(std_util::move(other)), re(std_util::move(other.re)), im(std_util::move(other.im)){}

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

  // Perform the conjugation operation
  __CUDA_HOST_DEVICE__ __CPP_VIRTUAL_CONSTEXPR__ bool is_conjugatable() const{ return true; }
  __CUDA_HOST_DEVICE__ void conjugate(){ im = -im; }
};

template<typename T> using IvyComplexVariablePtr_t = IvyThreadSafePtr_t< IvyComplexVariable<T> >;

// Make global functions for ease of use
template<typename T> __CUDA_HOST_DEVICE__ IvyComplexVariablePtr_t<T> Complex(std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream){ return make_IvyThreadSafePtr< IvyComplexVariable<T> >(mem_type, stream); }
template<typename T> __CUDA_HOST_DEVICE__ IvyComplexVariablePtr_t<T> Complex(T const& re, std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream){ return make_IvyThreadSafePtr< IvyComplexVariable<T> >(mem_type, stream, re); }
template<typename T> __CUDA_HOST_DEVICE__ IvyComplexVariablePtr_t<T> Complex(T const& re, T const& im, std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream){ return make_IvyThreadSafePtr< IvyComplexVariable<T> >(mem_type, stream, re, im); }


#endif
