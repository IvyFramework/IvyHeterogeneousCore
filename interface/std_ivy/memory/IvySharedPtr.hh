#ifndef IVYSHAREDPTR_HH
#define IVYSHAREDPTR_HH


#ifdef __USE_CUDA__

#include "IvyCompilerFlags.h"
#include "IvyCudaFlags.h"
#include "std_ivy/IvyCstddef.h"
#include "std_ivy/memory/IvyAllocator.h"


template<typename T> class IvySharedPtr{
public:
  typedef T element_type;
  typedef T* pointer;
  typedef T& reference;
  typedef unsigned long long int counter_type;

protected:
  bool* is_on_device_;
  pointer ptr_;
  counter_type* ref_count_;
  cudaStream_t* stream_;

  __CUDA_HOST_DEVICE__ void init_members(bool is_on_device);
  __CUDA_HOST_DEVICE__ void release();
  __CUDA_HOST_DEVICE__ void dump();

public:
  __CUDA_HOST_DEVICE__ IvySharedPtr();
  __CUDA_HOST_DEVICE__ IvySharedPtr(std_cstddef::nullptr_t);
  __CUDA_HOST_DEVICE__ IvySharedPtr(element_type* ptr, bool is_on_device, cudaStream_t* stream = nullptr);
  template<typename U> explicit __CUDA_HOST_DEVICE__ IvySharedPtr(U* ptr, bool is_on_device, cudaStream_t* stream = nullptr);
  template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr(IvySharedPtr<U> const& r);
  template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr(IvySharedPtr<U>&& r);
  __CUDA_HOST_DEVICE__ ~IvySharedPtr();

};
template<typename T> using shared_ptr = IvySharedPtr<T>;

#endif


#endif
