#ifndef IVYUNIFIEDPTR_H
#define IVYUNIFIEDPTR_H


#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"
#include "IvyUnifiedPtr.hh"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr() :
    is_on_device_(nullptr),
    ptr_(nullptr),
    ref_count_(nullptr),
    stream_(nullptr)
  {}
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(std_cstddef::nullptr_t) :
    is_on_device_(nullptr),
    ptr_(nullptr),
    ref_count_(nullptr),
    stream_(nullptr)
  {}
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, bool is_on_device, cudaStream_t* stream){
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_){
      stream_ = stream;
      this->init_members(is_on_device);
    }
  }
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPT> const& other){
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    if (ptr_){
      is_on_device_ = other.use_gpu();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) ++(*ref_count_);
    }
    else{
      printf("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    if (IPT==IvyPointerType::unique){
      typedef IvyUnifiedPtr<U, IPT>& uptr_t;
      __CONST_CAST__(uptr_t, other).reset();
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other){
    ptr_ = other.ptr_;
    if (ptr_){
      is_on_device_ = other.is_on_device_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    else{
      printf("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    if (IPT==IvyPointerType::unique){
      typedef IvyUnifiedPtr<T, IPT>& uptr_t;
      __CONST_CAST__(uptr_t, other).reset();
    }
  }
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPT>&& other) :
    is_on_device_(std_util::move(other.use_gpu())),
    ptr_(std_util::move(other.get())),
    ref_count_(std_util::move(other.counter())),
    stream_(std_util::move(other.gpu_stream()))
  {
    other.dump();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr&& other) :
    is_on_device_(std_util::move(other.is_on_device_)),
    ptr_(std_util::move(other.ptr_)),
    ref_count_(std_util::move(other.ref_count_)),
    stream_(std_util::move(other.stream_))
  {
    other.dump();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::~IvyUnifiedPtr(){
    this->release();
  }

  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr<U, IPT> const& other){
    if (*this != other){
      this->release();
      is_on_device_ = other.use_gpu();
      ptr_ = __DYNAMIC_CAST__(pointer, other.get());
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) ++(*ref_count_);
    }
    if (IPT==IvyPointerType::unique){
      typedef IvyUnifiedPtr<U, IPT>& uptr_t;
      __CONST_CAST__(uptr_t, other).reset();
    }
    return *this;
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr const& other){
    if (*this != other){
      this->release();
      is_on_device_ = other.is_on_device_;
      ptr_ = other.ptr_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    if (IPT==IvyPointerType::unique){
      typedef IvyUnifiedPtr<T, IPT>& uptr_t;
      __CONST_CAST__(uptr_t, other).reset();
    }
    return *this;
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::init_members(bool is_on_device){
    cudaStream_t ref_stream = IvyCudaConfig::get_gpu_stream_from_pointer(stream_);
    std_ivy::allocator<counter_type> alloc_ctr;
    ref_count_ = alloc_ctr.allocate(1, false, ref_stream);
    *ref_count_ = 1;
    std_ivy::allocator<bool> alloc_iod;
    is_on_device_ = alloc_iod.allocate(1, false, ref_stream);
    *is_on_device_ = is_on_device;
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::release(){
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
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::dump(){
    is_on_device_ = nullptr;
    ptr_ = nullptr;
    ref_count_ = nullptr;
    stream_ = nullptr;
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool* IvyUnifiedPtr<T, IPT>::use_gpu() const noexcept{ return this->is_on_device_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ cudaStream_t* IvyUnifiedPtr<T, IPT>::gpu_stream() const noexcept{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type* IvyUnifiedPtr<T, IPT>::counter() const noexcept{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::get() const noexcept{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool*& IvyUnifiedPtr<T, IPT>::use_gpu() noexcept{ return this->is_on_device_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ cudaStream_t*& IvyUnifiedPtr<T, IPT>::gpu_stream() noexcept{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type*& IvyUnifiedPtr<T, IPT>::counter() noexcept{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer& IvyUnifiedPtr<T, IPT>::get() noexcept{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::reference IvyUnifiedPtr<T, IPT>::operator*() const noexcept{ return *(this->ptr_); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::operator->() const noexcept{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(){ this->release(); this->dump(); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(std_cstddef::nullptr_t){ this->release(); this->dump(); }
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, bool is_on_device, cudaStream_t* stream){
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
        printf("IvyUnifiedPtr::reset() failed: Incompatible is_on_device flags.\n");
        assert(false);
      }
    }
  }

  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::swap(IvyUnifiedPtr<U, IPT>& other) noexcept{
    typedef typename IvyUnifiedPtr<U, IPT>::pointer uptr_t;
    bool inull = (ptr_==nullptr), onull = (other.get()==nullptr);
    pointer tmp_ptr = ptr_;
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    other.get() = __DYNAMIC_CAST__(uptr_t, tmp_ptr);
    if ((inull != (other.ptr_==nullptr)) || (onull != (ptr_==nullptr))){
      printf("IvyUnifiedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
    std_util::swap(ref_count_, other.counter());
    std_util::swap(is_on_device_, other.use_gpu());
    std_util::swap(stream_, other.gpu_stream());
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::use_count() const noexcept{ return (ref_count_ ? *ref_count_ : static_cast<counter_type>(0)); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::unique() const noexcept{ return this->use_count() == static_cast<counter_type>(1); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::operator bool() const noexcept{ return ptr_ != nullptr; }


  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) noexcept{ return (a.get()==b.get()); }
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) noexcept{ return !(a==b); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, T* ptr) noexcept{ return (a.get()==ptr); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, T* ptr) noexcept{ return !(a==ptr); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(T* ptr, IvyUnifiedPtr<T, IPT> const& a) noexcept{ return (ptr==a.get()); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(T* ptr, IvyUnifiedPtr<T, IPT> const& a) noexcept{ return !(ptr==a); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) noexcept{ return (a.get()==nullptr); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) noexcept{ return !(a==nullptr); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) noexcept{ return (nullptr==a.get()); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) noexcept{ return !(nullptr==a); }

  template<typename T, typename U, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void swap(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPT> const& b) noexcept{ a.swap(b); }

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> allocate_unified(Allocator_t const& a, bool is_on_device, cudaStream_t* stream, Args&&... args){
    cudaStream_t ref_stream = IvyCudaConfig::get_gpu_stream_from_pointer(stream);
    typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.allocate(1, is_on_device, ref_stream, args...);
    return IvyUnifiedPtr<T, IPT>(ptr, is_on_device, stream);
  }
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> allocate_shared(Allocator_t const& a, bool is_on_device, cudaStream_t* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::shared>(a, is_on_device, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> allocate_unique(Allocator_t const& a, bool is_on_device, cudaStream_t* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::unique>(a, is_on_device, stream, args...); }

  template<typename T, IvyPointerType IPT, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(bool is_on_device, cudaStream_t* stream, Args&&... args){
    return allocate_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), is_on_device, stream, args...);
  }
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(bool is_on_device, cudaStream_t* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(is_on_device, stream, args...); }
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(bool is_on_device, cudaStream_t* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(is_on_device, stream, args...); }

}

#endif


#endif
