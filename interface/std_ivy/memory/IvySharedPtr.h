#ifndef IVYSHAREDPTR_H
#define IVYSHAREDPTR_H


#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"
#include "IvySharedPtr.hh"


#ifdef __USE_CUDA__

namespace std_ivy{
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
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(U* ptr, bool is_on_device, cudaStream_t* stream){
    ptr_ = dynamic_cast<pointer>(ptr);
    if (ptr_){
      stream_ = stream;
      this->init_members(is_on_device);
    }
  }
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<U> const& other){
    ptr_ = dynamic_cast<pointer>(other.ptr_);
    if (ptr_){
      is_on_device_ = other.is_on_device_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    else{
      printf("IvySharedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
  }
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<U>&& other) :
    is_on_device_(std_util::move(other.is_on_device_)),
    ptr_(std_util::move(other.ptr_)),
    ref_count_(std_util::move(other.ref_count_)),
    stream_(std_util::move(other.stream_))
  {
    other.dump();
  }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::~IvySharedPtr(){
    this->release();
  }

  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>& IvySharedPtr<T>::operator=(IvySharedPtr<U> const& other){
    if (*this != other){
      this->release();
      is_on_device_ = other.is_on_device_;
      ptr_ = other.ptr_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
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

  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::pointer IvySharedPtr<T>::get() const noexcept{ return this->ptr_; }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::reference IvySharedPtr<T>::operator*() const noexcept{ return *(this->ptr_); }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::pointer IvySharedPtr<T>::operator->() const noexcept{ return this->ptr_; }

  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::reset(){ this->release(); }
  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::reset(std_cstddef::nullptr_t){ this->release(); }
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::reset(U* ptr, bool is_on_device, cudaStream_t* newstream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->release();
      if (newstream) stream_ = newstream;
      ptr_ = dynamic_cast<pointer>(ptr);
      if (ptr_) this->init_members(is_on_device);
    }
    else{
      if (newstream) stream_ = newstream;
      if (*is_on_device_ != is_on_device){
        printf("IvySharedPtr::reset() failed: Incompatible is_on_device flags.\n");
        assert(false);
      }
    }
  }

  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::swap(IvySharedPtr<U>& other) noexcept{
    bool inull = (ptr_==nullptr), onull = (other.ptr_==nullptr);
    pointer tmp_ptr = ptr_; ptr_ = dynamic_cast<pointer>(other.ptr_); other.ptr_ = dynamic_cast<typename IvySharedPtr<U>::pointer>(tmp_ptr);
    if ((inull != (other.ptr_==nullptr)) || (onull != (ptr_==nullptr))){
      printf("IvySharedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
    std_util::swap(ref_count_, other.ref_count_);
    std_util::swap(is_on_device_, other.is_on_device_);
    std_util::swap(stream_, other.stream_);
  }

  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::counter_type IvySharedPtr<T>::use_count() const noexcept{ return (ref_count_ ? *ref_count_ : static_cast<counter_type>(0)); }
  template<typename T> __CUDA_HOST_DEVICE__ bool IvySharedPtr<T>::unique() const noexcept{ return this->use_count() == static_cast<counter_type>(1); }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::operator bool() const noexcept{ return ptr_ != nullptr; }


  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool operator==(IvySharedPtr<T> const& a, IvySharedPtr<U> const& b) noexcept{ return (a.get()==b.get()); }
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool operator!=(IvySharedPtr<T> const& a, IvySharedPtr<U> const& b) noexcept{ return !(a==b); }

  template<typename T> __CUDA_HOST_DEVICE__ bool operator==(IvySharedPtr<T> const& a, std_cstddef::nullptr_t) noexcept{ return (a.get()==nullptr); }
  template<typename T> __CUDA_HOST_DEVICE__ bool operator!=(IvySharedPtr<T> const& a, std_cstddef::nullptr_t) noexcept{ return !(a==nullptr); }

  template<typename T> __CUDA_HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvySharedPtr<T> const& a) noexcept{ return (nullptr==a.get()); }
  template<typename T> __CUDA_HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvySharedPtr<T> const& a) noexcept{ return !(nullptr==a); }

  template<typename T, typename U> __CUDA_HOST_DEVICE__ void swap(IvySharedPtr<T> const& a, IvySharedPtr<U> const& b) noexcept{ a.swap(b); }

  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ IvySharedPtr<T> make_shared(bool is_on_device, cudaStream_t* stream, Args&&... args){
    cudaStream_t ref_stream = (stream ? *stream : cudaStreamLegacy);
    std_ivy::allocator<T> a;

    typename IvySharedPtr<T>::pointer ptr = nullptr;
    typename IvySharedPtr<T>::pointer h_ptr = a.allocate(1, false, ref_stream);
    *h_ptr = std_util::move(T(std_util::forward<Args>(args)...));
#ifndef __CUDA_DEVICE_CODE__
    if (is_on_device){
      typename IvySharedPtr<T>::pointer d_ptr = a.allocate(1, is_on_device, ref_stream);
      a.transfer(d_ptr, h_ptr, 1, false, ref_stream);
      ptr = d_ptr;
    }
    else
#endif
    {
      ptr = h_ptr;
    }

    return IvySharedPtr<T>(ptr, is_on_device, stream);
  }

}

#endif


#endif
