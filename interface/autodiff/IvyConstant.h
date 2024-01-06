#ifndef IVYCONSTANT_H
#define IVYCONSTANT_H


#include "std_ivy/IvyTypeTraits.h"
#include "autodiff/IvyBaseConstant.h"
#include "stream/IvyStream.h"


template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyConstant final : public IvyBaseConstant{
public:
  using dtype_t = T;
  using value_t = T;

protected:
  value_t const value_;

public:
  // Constructors
  __CUDA_HOST_DEVICE__ IvyConstant() : IvyBaseConstant(), value_(0){};
  __CUDA_HOST_DEVICE__ IvyConstant(T const& value) : IvyBaseConstant(), value_(value){}
  __CUDA_HOST_DEVICE__ IvyConstant(IvyConstant const& other) : IvyBaseConstant(other), value_(other.value_){}
  __CUDA_HOST_DEVICE__ IvyConstant(IvyConstant const&& other) : IvyBaseConstant(std_util::move(other)), value_(std_util::move(other.value_)){}

  // Empty virtual destructor
  __CUDA_HOST_DEVICE__ ~IvyConstant(){}

  // Get function
  __CUDA_HOST_DEVICE__ value_t const& value() const{ return this->value_; }
};

// Make global functions for ease of use
template<typename T> __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t< IvyConstant<T> > Constant(std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream){ return make_IvyThreadSafePtr< IvyConstant<T> >(mem_type, stream); }
template<typename T> __CUDA_HOST_DEVICE__ IvyThreadSafePtr_t< IvyConstant<T> > Constant(T const& val, std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream){ return make_IvyThreadSafePtr< IvyConstant<T> >(mem_type, stream, val); }


#endif
