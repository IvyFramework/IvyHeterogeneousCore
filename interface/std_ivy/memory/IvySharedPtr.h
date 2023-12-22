#ifndef IVYSHAREDPTR_H
#define IVYSHAREDPTR_H


#include "IvySharedPtr.hh"


#ifdef __USE_CUDA__

template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr() :
  is_on_device_(nullptr),
  ptr_(nullptr),
  ref_count_(nullptr),
  stream_(nullptr)
{}
template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(std_cstddef::nullptr_t) :
  is_on_device_(nullptr),
  ptr_(nullptr),
  ref_count_(nullptr),
  stream_(nullptr)
{}
template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(element_type* ptr, bool is_on_device, cudaStream_t* stream) :
  ptr_(ptr),
  stream_(stream)
{
  if (ptr_) this->init_members(is_on_device);
}
template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(U* ptr, bool is_on_device, cudaStream_t* stream) :
  ptr_(ptr),
  stream_(stream)
{
  if (ptr_) this->init_members(is_on_device);
}
template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<U> const& r) :
  is_on_device_(r.is_on_device_),
  ptr_(r.ptr_),
  ref_count_(r.ref_count_),
  stream_(r.stream_)
{
  if (ref_count_) ++(*ref_count_);
}
template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<U>&& r) :
  is_on_device_(std_util::move(r.is_on_device_)),
  ptr_(std_util::move(r.ptr_)),
  ref_count_(std_util::move(r.ref_count_)),
  stream_(std_util::move(r.stream_))
{
  r.dump();
}
template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::~IvySharedPtr(){
  this->release();
}

template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::init_members(bool is_on_device){
  std_ivy::allocator<counter_type> alloc_ctr;
  ref_count_ = alloc_ctr.allocate(1);
  *ref_count_ = 1;
  std_ivy::allocator<bool> alloc_iod;
  is_on_device_ = alloc_iod.allocate(1);
  *is_on_device_ = is_on_device;
}
template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::release(){
  if (ref_count_){
    if (*ref_count_>0) --(*ref_count_);
    if (*ref_count_ == 0){
      cudaStream_t ref_stream = (stream_ ? *stream_ : cudaStreamLegacy);
      std_ivy::allocator<element_type> alloc_ptr;
      alloc_ptr.deallocate(ptr_, 1, *is_on_device_, ref_stream);
      std_ivy::allocator<counter_type> alloc_ctr;
      alloc_ctr.deallocate(ref_count_, 1);
      std_ivy::allocator<bool> alloc_iod;
      alloc_iod.deallocate(is_on_device_, 1);
    }
  }
}
template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::dump(){
  is_on_device_ = nullptr;
  ptr_ = nullptr;
  ref_count_ = nullptr;
  stream_ = nullptr;
}

#endif


#endif
