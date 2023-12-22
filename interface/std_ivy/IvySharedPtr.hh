#ifndef IVYSHAREDPTR_H
#define IVYSHAREDPTR_H


#include "IvyCompilerFlags.h"
#include "IvyCUDAFlags.h"
#include "IvyAllocator.hh"


template<typename T, typename Allocator_t = std_ivy::allocator<T>> class IvySharedPtr{
public:
  typedef Allocator_t allocator_type;
  typedef T value_type;
  typedef T* pointer;
  typedef T& reference;
  typedef unsigned long long int counter_type;

protected:
  bool is_on_device;
  T* ptr;
  counter_type* refCount;

public:
  __CUDA_HOST_DEVICE__ IvySharedPtr();
  __CUDA_HOST_DEVICE__ ~IvySharedPtr();


};


#endif
