#ifndef IVYUNIFIEDPTR_H
#define IVYUNIFIEDPTR_H


#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"
#include "std_ivy/memory/IvyUnifiedPtr.hh"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr() :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    mem_type_(nullptr),
    ptr_(nullptr),
    size_(nullptr),
    ref_count_(nullptr),
    stream_(nullptr)
  {}
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(std_cstddef::nullptr_t) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    mem_type_(nullptr),
    ptr_(nullptr),
    size_(nullptr),
    ref_count_(nullptr),
    stream_(nullptr)
  {}
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_){
      stream_ = stream;
      this->init_members(mem_type, 1);
    }
  }
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory())
  {
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_){
      stream_ = stream;
      this->init_members(mem_type, n);
    }
  }
  template<typename T, IvyPointerType IPT>
  template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other){
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    if (ptr_){
      exec_mem_type_ = other.get_exec_memory_type();
      mem_type_ = other.get_memory_type();
      size_ = other.size();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) ++(*ref_count_);
    }
    else{
      __PRINT_ERROR__("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    if (IPU==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<U, IPU>&), other).reset();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other){
    ptr_ = other.ptr_;
    if (ptr_){
      exec_mem_type_ = other.exec_mem_type_;
      mem_type_ = other.mem_type_;
      size_ = other.size_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    else{
      __PRINT_ERROR__("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    if (IPT==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
  }
  template<typename T, IvyPointerType IPT> template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other) :
    exec_mem_type_(std_util::move(other.get_exec_memory_type())),
    mem_type_(std_util::move(other.get_memory_type())),
    ptr_(std_util::move(other.get())),
    size_(std_util::move(other.size())),
    ref_count_(std_util::move(other.counter())),
    stream_(std_util::move(other.gpu_stream()))
  {
    other.dump();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr&& other) :
    exec_mem_type_(std_util::move(other.exec_mem_type_)),
    mem_type_(std_util::move(other.mem_type_)),
    ptr_(std_util::move(other.ptr_)),
    size_(std_util::move(other.size_)),
    ref_count_(std_util::move(other.ref_count_)),
    stream_(std_util::move(other.stream_))
  {
    other.dump();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::~IvyUnifiedPtr(){
    this->release();
  }

  template<typename T, IvyPointerType IPT> template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr<U, IPU> const& other){
    if (*this != other){
      this->release();
      exec_mem_type_ = other.get_exec_memory_type();
      mem_type_ = other.get_memory_type();
      ptr_ = __DYNAMIC_CAST__(pointer, other.get());
      size_ = other.size();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) ++(*ref_count_);
    }
    if (IPU==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<U, IPU>&), other).reset();
    return *this;
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr const& other){
    if (*this != other){
      this->release();
      exec_mem_type_ = other.exec_mem_type_;
      mem_type_ = other.mem_type_;
      ptr_ = other.ptr_;
      size_ = other.size_;
      ref_count_ = other.ref_count_;
      stream_ = other.stream_;
      if (ref_count_) ++(*ref_count_);
    }
    if (IPT==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
    return *this;
  }
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(U* ptr){ this->reset(ptr); return *this; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(std_cstddef::nullptr_t){ this->reset(nullptr); return *this; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::init_members(IvyMemoryType mem_type, size_type n){
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        size_ = size_allocator_traits::allocate(1, exec_mem_type_, ref_stream, n);
        ref_count_ = counter_allocator_traits::allocate(1, exec_mem_type_, ref_stream, __STATIC_CAST__(counter_type, 1));
        mem_type_ = mem_type_allocator_traits::allocate(1, exec_mem_type_, ref_stream, mem_type);
      )
    );
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::release(){
    if (ref_count_){
      if (*ref_count_>0) --(*ref_count_);
      if (*ref_count_ == 0){
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            element_allocator_traits::deallocate(ptr_, *size_, *mem_type_, ref_stream);
            size_allocator_traits::deallocate(size_, 1, exec_mem_type_, ref_stream);
            counter_allocator_traits::deallocate(ref_count_, 1, exec_mem_type_, ref_stream);
            mem_type_allocator_traits::deallocate(mem_type_, 1, exec_mem_type_, ref_stream);
          )
        );
      }
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::dump(){
    mem_type_ = nullptr;
    ptr_ = nullptr;
    size_ = nullptr;
    ref_count_ = nullptr;
    stream_ = nullptr;
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() const noexcept{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType* IvyUnifiedPtr<T, IPT>::get_memory_type() const noexcept{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyGPUStream* IvyUnifiedPtr<T, IPT>::gpu_stream() const noexcept{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::size() const noexcept{ return this->size_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type* IvyUnifiedPtr<T, IPT>::counter() const noexcept{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::get() const noexcept{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() noexcept{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType*& IvyUnifiedPtr<T, IPT>::get_memory_type() noexcept{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyGPUStream*& IvyUnifiedPtr<T, IPT>::gpu_stream() noexcept{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type*& IvyUnifiedPtr<T, IPT>::size() noexcept{ return this->size_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type*& IvyUnifiedPtr<T, IPT>::counter() noexcept{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer& IvyUnifiedPtr<T, IPT>::get() noexcept{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::reference IvyUnifiedPtr<T, IPT>::operator*() const noexcept{ return *(this->ptr_); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::operator->() const noexcept{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(){ this->release(); this->dump(); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(std_cstddef::nullptr_t){ this->release(); this->dump(); }
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->release();
      this->dump();
      stream_ = stream;
      ptr_ = __DYNAMIC_CAST__(pointer, ptr);
      if (ptr_) this->init_members(mem_type, n);
    }
    else{
      if (stream) stream_ = stream;
      if (*mem_type_ != mem_type){
        __PRINT_ERROR__("IvyUnifiedPtr::reset() failed: Incompatible mem_type flags.\n");
        assert(false);
      }
    }
  }
  template<typename T, IvyPointerType IPT>
  template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, __STATIC_CAST__(size_type, 1), mem_type, stream);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST__ void IvyUnifiedPtr<T, IPT>::transfer(IvyMemoryType new_mem_type, bool transfer_all){
    if (ptr_){
      if (*mem_type_ != new_mem_type){
        IvyMemoryType misc_mem_type = (transfer_all ? new_mem_type : exec_mem_type_);
        pointer new_ptr_ = nullptr;
        size_type* new_size_ = nullptr;
        counter_type*  new_ref_count_ = nullptr;
        IvyMemoryType* new_mem_type_ = nullptr;

        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            new_ptr_ = element_allocator_traits::allocate(1, new_mem_type, ref_stream, *ptr_);
            new_size_ = size_allocator_traits::allocate(1, misc_mem_type, ref_stream, *size_);
            new_ref_count_ = counter_allocator_traits::allocate(1, misc_mem_type, ref_stream, __STATIC_CAST__(counter_type, 1));
            new_mem_type_ = mem_type_allocator_traits::allocate(1, misc_mem_type, ref_stream, new_mem_type);
          )
        );

        this->release();

        exec_mem_type_ = misc_mem_type;
        ptr_ = new_ptr_;
        size_ = new_size_;
        ref_count_ = new_ref_count_;
        mem_type_ = new_mem_type_;
      }
    }
  }

  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::swap(IvyUnifiedPtr<U, IPT>& other) noexcept{
    bool inull = (ptr_==nullptr), onull = (other.get()==nullptr);
    pointer tmp_ptr = ptr_;
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    other.get() = __DYNAMIC_CAST__(__ENCAPSULATE__(typename IvyUnifiedPtr<U, IPT>::pointer), tmp_ptr);
    if ((inull != (other.ptr_==nullptr)) || (onull != (ptr_==nullptr))){
      __PRINT_ERROR__("IvyUnifiedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
    std_util::swap(size_, other.size());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(mem_type_, other.get_memory_type());
    std_util::swap(ref_count_, other.counter());
    std_util::swap(stream_, other.stream_);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::use_count() const noexcept{ return (ref_count_ ? *ref_count_ : static_cast<counter_type>(0)); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::unique() const noexcept{ return this->use_count() == static_cast<counter_type>(1); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::operator bool() const noexcept{ return ptr_ != nullptr; }

  // Non-member helper functions
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

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> allocate_unified(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.allocate(1, mem_type, ref_stream, args...);
      )
    );
    return IvyUnifiedPtr<T, IPT>(ptr, mem_type, stream);
  }
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> allocate_shared(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::shared>(a, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> allocate_unique(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::unique>(a, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return allocate_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), mem_type, stream, args...);
  }
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(mem_type, stream, args...); }
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(mem_type, stream, args...); }

}

#endif


#endif
