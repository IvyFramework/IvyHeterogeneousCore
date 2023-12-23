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
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_){
      stream_ = stream;
      this->init_members(is_on_device);
    }
  }
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<U> const& other){
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    if (ptr_){
      is_on_device_ = other.use_gpu();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) ++(*ref_count_);
    }
    else{
      printf("IvySharedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
  }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<T> const& other){
    ptr_ = other.ptr_;
    if (ptr_){
      is_on_device_ = other.is_on_device_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    else{
      printf("IvySharedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
  }
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr<U>&& other) :
    is_on_device_(std_util::move(other.use_gpu())),
    ptr_(std_util::move(other.get())),
    ref_count_(std_util::move(other.counter())),
    stream_(std_util::move(other.gpu_stream()))
  {
    other.dump();
  }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::IvySharedPtr(IvySharedPtr&& other) :
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
      is_on_device_ = other.use_gpu();
      ptr_ = __DYNAMIC_CAST__(pointer, other.get());
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) ++(*ref_count_);
    }
    return *this;
  }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>& IvySharedPtr<T>::operator=(IvySharedPtr const& other){
    if (*this != other){
      this->release();
      is_on_device_ = other.is_on_device_;
      ptr_ = other.ptr_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    return *this;
  }

  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::init_members(bool is_on_device){
    cudaStream_t ref_stream = IvyCudaConfig::get_gpu_stream_from_pointer(stream_);
    std_ivy::allocator<counter_type> alloc_ctr;
    ref_count_ = alloc_ctr.allocate(1, false, ref_stream);
    *ref_count_ = 1;
    std_ivy::allocator<bool> alloc_iod;
    is_on_device_ = alloc_iod.allocate(1, false, ref_stream);
    *is_on_device_ = is_on_device;
  }
  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::release(){
    if (ref_count_){
      if (*ref_count_>0) --(*ref_count_);
      if (*ref_count_ == 0){
        cudaStream_t ref_stream = IvyCudaConfig::get_gpu_stream_from_pointer(stream_);
        std_ivy::allocator<element_type> alloc_ptr;
        alloc_ptr.deallocate(ptr_, 1, *is_on_device_, ref_stream);
        std_ivy::allocator<counter_type> alloc_ctr;
        alloc_ctr.deallocate(ref_count_, 1, false, ref_stream);
        std_ivy::allocator<bool> alloc_iod;
        alloc_iod.deallocate(is_on_device_, 1, false, ref_stream);
      }
    }
  }
  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::dump(){
    is_on_device_ = nullptr;
    ptr_ = nullptr;
    ref_count_ = nullptr;
    stream_ = nullptr;
  }

  template<typename T> __CUDA_HOST_DEVICE__ bool* IvySharedPtr<T>::use_gpu() const noexcept{ return this->is_on_device_; }
  template<typename T> __CUDA_HOST_DEVICE__ cudaStream_t* IvySharedPtr<T>::gpu_stream() const noexcept{ return this->stream_; }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::counter_type* IvySharedPtr<T>::counter() const noexcept{ return this->ref_count_; }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::pointer IvySharedPtr<T>::get() const noexcept{ return this->ptr_; }

  template<typename T> __CUDA_HOST_DEVICE__ bool*& IvySharedPtr<T>::use_gpu() noexcept{ return this->is_on_device_; }
  template<typename T> __CUDA_HOST_DEVICE__ cudaStream_t*& IvySharedPtr<T>::gpu_stream() noexcept{ return this->stream_; }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::counter_type*& IvySharedPtr<T>::counter() noexcept{ return this->ref_count_; }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::pointer& IvySharedPtr<T>::get() noexcept{ return this->ptr_; }

  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::reference IvySharedPtr<T>::operator*() const noexcept{ return *(this->ptr_); }
  template<typename T> __CUDA_HOST_DEVICE__ IvySharedPtr<T>::pointer IvySharedPtr<T>::operator->() const noexcept{ return this->ptr_; }

  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::reset(){ this->release(); this->dump(); }
  template<typename T> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::reset(std_cstddef::nullptr_t){ this->release(); this->dump(); }
  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::reset(U* ptr, bool is_on_device, cudaStream_t* stream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->release();
      this->dump();
      stream_ = stream;
      ptr_ = __DYNAMIC_CAST__(pointer, ptr);
      if (ptr_) this->init_members(is_on_device);
    }
    else{
      if (stream) stream_ = stream;
      if (*is_on_device_ != is_on_device){
        printf("IvySharedPtr::reset() failed: Incompatible is_on_device flags.\n");
        assert(false);
      }
    }
  }

  template<typename T> template<typename U> __CUDA_HOST_DEVICE__ void IvySharedPtr<T>::swap(IvySharedPtr<U>& other) noexcept{
    bool inull = (ptr_==nullptr), onull = (other.get()==nullptr);
    pointer tmp_ptr = ptr_;
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    other.get() = __DYNAMIC_CAST__(typename IvySharedPtr<U>::pointer, tmp_ptr);
    if ((inull != (other.ptr_==nullptr)) || (onull != (ptr_==nullptr))){
      printf("IvySharedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
    std_util::swap(ref_count_, other.counter());
    std_util::swap(is_on_device_, other.use_gpu());
    std_util::swap(stream_, other.gpu_stream());
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
    cudaStream_t ref_stream = IvyCudaConfig::get_gpu_stream_from_pointer(stream);
    typename IvySharedPtr<T>::pointer ptr = std_ivy::allocator<T>::allocate(1, is_on_device, ref_stream, args...);
    return IvySharedPtr<T>(ptr, is_on_device, stream);
  }

}

#endif


#endif
