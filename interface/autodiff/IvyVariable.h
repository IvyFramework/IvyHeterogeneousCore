#ifndef IVYVARIABLE_H
#define IVYVARIABLE_H


#include "std_ivy/IvyTypeTraits.h"
#include "autodiff/IvyBaseVariable.h"
#include "stream/IvyStream.h"


template<typename T, ENABLE_IF_ARITHMETIC(T)> class IvyVariable;
template<typename T> struct IvyNodeSelfRelations<IvyVariable<T>>;

template<typename T, ENABLE_IF_ARITHMETIC_IMPL(T)> class IvyVariable final : public IvyBaseVariable{
public:
  using dtype_t = T;
  using value_t = T;

protected:
  value_t value_;
  value_t infinitesimal_;

public:
  // Empty default constructor
  __CUDA_HOST_DEVICE__ IvyVariable() : value_(0), infinitesimal_(0){}
  __CUDA_HOST_DEVICE__ IvyVariable(T const& value) : value_(value), infinitesimal_(0){}
  __CUDA_HOST_DEVICE__ IvyVariable(T const& value, T const& infinitesimal) : value_(value), infinitesimal_(infinitesimal){}
  __CUDA_HOST_DEVICE__ IvyVariable(IvyVariable const& other) : value_(other.value_), infinitesimal_(other.infinitesimal_){}
  __CUDA_HOST_DEVICE__ IvyVariable(IvyVariable const&& other) : value_(std_util::move(other.value_)), infinitesimal_(std_util::move(other.infinitesimal_)){}
  __CUDA_HOST_DEVICE__ ~IvyVariable(){}

  // Assignment operators
  __CUDA_HOST_DEVICE__ IvyVariable<T>& operator=(IvyVariable<T> const& other){ this->value_ = other.value_; this->infinitesimal_ = other.infinitesimal_; return *this; }
  __CUDA_HOST_DEVICE__ IvyVariable<T>& operator=(T const& value){ this->value_ = value; return *this; }

  // Set functions
  __CUDA_HOST_DEVICE__ void set_value(T const& value){ this->value_ = value; }
  __CUDA_HOST_DEVICE__ void set_infinitesimal(T const& infinitesimal){ this->infinitesimal_ = infinitesimal; }

  // Get functions
  __CUDA_HOST_DEVICE__ value_t const& value() const{ return this->value_; }
  __CUDA_HOST_DEVICE__ value_t const& infinitesimal() const{ return this->infinitesimal_; }

  // IvyVariables are differentiable objects.
  __CUDA_HOST_DEVICE__ bool is_differentiable() const{ return true; }

  friend struct IvyNodeSelfRelations<IvyVariable<T>>;
};

template<typename T> struct IvyNodeSelfRelations<IvyVariable<T>>{
  static __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(T const& x){ return true; }
  static __CUDA_HOST_DEVICE__ void conjugate(T& x){}
  static constexpr bool is_conjugatable = false;
};

template<typename T> using IvyVariablePtr_t = IvyThreadSafePtr_t< IvyVariable<T> >;

template<typename T, typename... Args> __CUDA_HOST_DEVICE__ IvyVariablePtr_t<T> Variable(std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream, Args&&... args){ return make_IvyThreadSafePtr< IvyVariable<T> >(mem_type, stream, args...); }


#endif
