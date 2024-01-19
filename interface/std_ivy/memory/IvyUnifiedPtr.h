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
      mem_type_ = other.get_memory_type_ptr();
      size_ = other.size_ptr();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) this->inc_dec_counter(true);
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
      if (ref_count_) this->inc_dec_counter(true);
    }
    else{
      __PRINT_ERROR__("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    if (IPT==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
  }
  template<typename T, IvyPointerType IPT> template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other) :
    exec_mem_type_(std_util::move(other.get_exec_memory_type())),
    mem_type_(std_util::move(other.get_memory_type_ptr())),
    ptr_(std_util::move(other.get())),
    size_(std_util::move(other.size_ptr())),
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

  template<typename T, IvyPointerType IPT>
  template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(IvyUnifiedPtr<U, IPU> const& other){
    if (*this != other){
      this->release();
      exec_mem_type_ = other.get_exec_memory_type();
      mem_type_ = other.get_memory_type_ptr();
      ptr_ = __DYNAMIC_CAST__(pointer, other.get());
      size_ = other.size_ptr();
      ref_count_ = other.counter();
      stream_ = other.gpu_stream();
      if (ref_count_) this->inc_dec_counter(true);
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
      if (ref_count_) this->inc_dec_counter(true);
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
      auto const current_count = this->use_count();
      if (current_count>0) this->inc_dec_counter(false);
      if (current_count==1){
        auto const current_mem_type = this->get_memory_type();
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            element_allocator_traits::deallocate(ptr_, this->size(), current_mem_type, ref_stream);
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

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() const __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType* IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() const __NOEXCEPT__{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyGPUStream* IvyUnifiedPtr<T, IPT>::gpu_stream() const __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::size_ptr() const __NOEXCEPT__{ return this->size_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type* IvyUnifiedPtr<T, IPT>::counter() const __NOEXCEPT__{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::get() const __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType*& IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() __NOEXCEPT__{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyGPUStream*& IvyUnifiedPtr<T, IPT>::gpu_stream() __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type*& IvyUnifiedPtr<T, IPT>::size_ptr() __NOEXCEPT__{ return this->size_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type*& IvyUnifiedPtr<T, IPT>::counter() __NOEXCEPT__{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer& IvyUnifiedPtr<T, IPT>::get() __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::reference IvyUnifiedPtr<T, IPT>::operator*() const __NOEXCEPT__{ return *(this->ptr_); }
  template<typename T, IvyPointerType IPT>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::reference IvyUnifiedPtr<T, IPT>::operator[](size_type k) const{
    if (k >= this->size()){
      __PRINT_ERROR__("IvyUnifiedPtr::operator[] failed: Index out of bounds.\n");
      assert(false);
    }
    return *(this->ptr_ + k);
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::operator->() const __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type IvyUnifiedPtr<T, IPT>::size() const __NOEXCEPT__{
    if (!size_) return 0;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type) return *size_;
    else{
      size_type* p_size_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_size_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
            size_allocator_traits::transfer(p_size_, size_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      size_type ret = *p_size_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            size_allocator_traits::deallocate(p_size_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType IvyUnifiedPtr<T, IPT>::get_memory_type() const __NOEXCEPT__{
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (!mem_type_) return def_mem_type;
    if (exec_mem_type_ == def_mem_type) return *mem_type_;
    else{
      IvyMemoryType* p_mem_type_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_mem_type_ = mem_type_allocator_traits::allocate(1, def_mem_type, ref_stream);
            mem_type_allocator_traits::transfer(p_mem_type_, mem_type_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      IvyMemoryType ret = *p_mem_type_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            mem_type_allocator_traits::deallocate(p_mem_type_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
  }

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
      if (this->get_memory_type() != mem_type){
        __PRINT_ERROR__("IvyUnifiedPtr::reset() failed: Incompatible mem_type flags.\n");
        assert(false);
      }
    }
  }
  template<typename T, IvyPointerType IPT> template<typename U>
  __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, __STATIC_CAST__(size_type, 1), mem_type, stream);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::transfer_internal_memory(IvyMemoryType const& new_mem_type){
#ifndef __CUDA_DEVICE_CODE__
    return this->transfer_impl(new_mem_type, true, true);
#else
    return true;
#endif
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST__ bool IvyUnifiedPtr<T, IPT>::transfer_impl(IvyMemoryType const& new_mem_type, bool transfer_all, bool copy_ptr){
    bool res = true;
    if (ptr_){
      auto const current_mem_type = this->get_memory_type();
      if (copy_ptr || current_mem_type != new_mem_type){
        IvyMemoryType misc_mem_type = (transfer_all ? new_mem_type : exec_mem_type_);
        pointer new_ptr_ = nullptr;
        size_type* new_size_ = nullptr;
        counter_type* new_ref_count_ = nullptr;
        IvyMemoryType* new_mem_type_ = nullptr;
        size_type const size_val = this->size();

        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            res &= element_allocator_traits::allocate(new_ptr_, size_val, new_mem_type, ref_stream);
            res &= element_allocator_traits::transfer(new_ptr_, ptr_, size_val, new_mem_type, current_mem_type, ref_stream);
            res &= size_allocator_traits::allocate(new_size_, 1, misc_mem_type, ref_stream, size_val);
            res &= counter_allocator_traits::allocate(new_ref_count_, 1, misc_mem_type, ref_stream, __STATIC_CAST__(counter_type, 1));
            res &= mem_type_allocator_traits::allocate(new_mem_type_, 1, misc_mem_type, ref_stream, new_mem_type);
          )
        );

        if (!copy_ptr) this->release();

        exec_mem_type_ = misc_mem_type;
        ptr_ = new_ptr_;
        size_ = new_size_;
        ref_count_ = new_ref_count_;
        mem_type_ = new_mem_type_;
      }
    }
    return res;
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST__ bool IvyUnifiedPtr<T, IPT>::transfer(IvyMemoryType const& new_mem_type, bool transfer_all){
    return this->transfer_impl(new_mem_type, transfer_all, false);
  }

  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::swap(IvyUnifiedPtr<U, IPT>& other) __NOEXCEPT__{
    bool inull = (ptr_==nullptr), onull = (other.get()==nullptr);
    pointer tmp_ptr = ptr_;
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    other.get() = __DYNAMIC_CAST__(__ENCAPSULATE__(typename IvyUnifiedPtr<U, IPT>::pointer), tmp_ptr);
    if ((inull != (other.ptr_==nullptr)) || (onull != (ptr_==nullptr))){
      __PRINT_ERROR__("IvyUnifiedPtr::swap() failed: Incompatible types\n");
      assert(false);
    }
    std_util::swap(size_, other.size_ptr());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(mem_type_, other.get_memory_type_ptr());
    std_util::swap(ref_count_, other.counter());
    std_util::swap(stream_, other.stream_);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type IvyUnifiedPtr<T, IPT>::use_count() const{
    if (!ref_count_) return 0;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type) return *ref_count_;
    else{
      counter_type* p_ref_count_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_ref_count_ = counter_allocator_traits::allocate(1, def_mem_type, ref_stream);
            counter_allocator_traits::transfer(p_ref_count_, ref_count_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      counter_type ret = *p_ref_count_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            counter_allocator_traits::deallocate(p_ref_count_, 1, def_mem_type, ref_stream);
          )
        );
      }
      return ret;
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_counter(bool do_inc){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      if (do_inc) ++(*ref_count_);
      else --(*ref_count_);
    }
    else{
      counter_type* p_ref_count_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          p_ref_count_ = counter_allocator_traits::allocate(1, def_mem_type, ref_stream);
          counter_allocator_traits::transfer(p_ref_count_, ref_count_, 1, def_mem_type, exec_mem_type_, ref_stream);
          if (do_inc) ++(*p_ref_count_);
          else --(*p_ref_count_);
          counter_allocator_traits::transfer(ref_count_, p_ref_count_, 1, exec_mem_type_, def_mem_type, ref_stream);
          counter_allocator_traits::deallocate(p_ref_count_, 1, def_mem_type, ref_stream);
        )
      );
    }
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::unique() const{ return this->use_count() == __STATIC_CAST__(counter_type, 1); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::operator bool() const __NOEXCEPT__{ return ptr_ != nullptr; }

  // Non-member helper functions
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__{ return (a.get()==b.get()); }
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__{ return !(a==b); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__{ return (a.get()==ptr); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__{ return !(a==ptr); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return (ptr==a.get()); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return !(ptr==a); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__{ return (a.get()==nullptr); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__{ return !(a==nullptr); }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return (nullptr==a.get()); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__{ return !(nullptr==a); }

  template<typename T, typename U, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void swap(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPT> const& b) __NOEXCEPT__{ a.swap(b); }

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> allocate_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.allocate(n, mem_type, ref_stream, args...);
      )
    );
    return IvyUnifiedPtr<T, IPT>(ptr, n, mem_type, stream);
  }
  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> allocate_unified(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return allocate_unified<T, IPT, Allocator_t, Args...>(a, __STATIC_CAST__(__ENCAPSULATE__(typename IvyUnifiedPtr<T, IPT>::size_type), 1), mem_type, stream, args...);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> allocate_shared(Allocator_t const& a, IvyMemoryType mem_type, typename shared_ptr<T>::size_type n, IvyGPUStream* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::shared>(a, n, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> allocate_unique(Allocator_t const& a, IvyMemoryType mem_type, typename unique_ptr<T>::size_type n, IvyGPUStream* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::unique>(a, n, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> allocate_shared(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::shared>(a, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> allocate_unique(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return allocate_unified<T, Allocator_t, IvyPointerType::unique>(a, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return allocate_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), n, mem_type, stream, args...);
  }
  template<typename T, IvyPointerType IPT, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return allocate_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), mem_type, stream, args...);
  }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(n, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(n, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(mem_type, stream, args...); }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT> __CUDA_GLOBAL__ void kernel_IvyUnifiedPtr_transfer_internal_memory(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t n, IvyMemoryType mem_type);
  template<typename T, IvyPointerType IPT> __CUDA_GLOBAL__ void kernel_IvyUnifiedPtr_dump(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t n);
  template<typename T, IvyPointerType IPT> __CUDA_GLOBAL__ void kernel_IvyUnifiedPtr_reset(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t n);
  template<typename T, IvyPointerType IPT> class transfer_memory_primitive<IvyUnifiedPtr<T, IPT>>{
  public:
    using base_t = allocation_type_properties<IvyUnifiedPtr<T, IPT>>;
    using value_type = typename base_t::value_type;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool call_IvyUnifiedPtr_transfer_internal_memory(IvyUnifiedPtr<T, IPT>*& ptr, IvyMemoryType const& mem_type){
      return ptr->transfer_internal_memory(mem_type);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void call_IvyUnifiedPtr_dump(IvyUnifiedPtr<T, IPT>*& ptr){
      return ptr->dump();
    }

  protected:
    static __CUDA_HOST_DEVICE__ bool transfer_internal_memory(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t const& n, IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream){
      bool res = true;
#ifndef __CUDA_DEVICE_CODE__
      if (IvyMemoryHelpers::is_device_memory(ptr_mem_type)){
        IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
        if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
          kernel_IvyUnifiedPtr_transfer_internal_memory<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(ptr, n, mem_type);
        }
        else{
          __PRINT_ERROR__("transfer_memory_primitive::transfer_internal_memory: Unable to call the GPU kernel...\n");
          res = false;
        }
      }
      else{
#else
      {
#endif
        IvyUnifiedPtr<T, IPT>* pr = ptr;
        for (size_type i=0; i<n; i++){
          res &= call_IvyUnifiedPtr_transfer_internal_memory(pr, mem_type);
          ++pr;
        }
      }
      return res;
    }
    static __CUDA_HOST_DEVICE__ void dump(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t const& n, IvyMemoryType const& mem_type, IvyGPUStream& stream){
#ifndef __CUDA_DEVICE_CODE__
      if (IvyMemoryHelpers::is_device_memory(mem_type)){
        IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
        if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
          kernel_IvyUnifiedPtr_dump<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(ptr, n);
        }
        else{
          __PRINT_ERROR__("transfer_memory_primitive::dump: Unable to call the GPU kernel...\n");
        }
      }
      else{
#else
      {
#endif
        IvyUnifiedPtr<T, IPT>* pr = ptr;
        for (size_type i=0; i<n; i++){
          call_IvyUnifiedPtr_dump(pr);
          ++pr;
        }
      }
    }

  public:
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      if (!src) return false;
      bool res = true;
      if (type_tgt == type_src){
        res &= IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
        res &= transfer_internal_memory(tgt, n, type_tgt, type_tgt, stream);
      }
      else{
        constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
        pointer p_int = nullptr;
        res &= IvyMemoryHelpers::allocate_memory(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::transfer_memory(p_int, src, n, def_mem_type, type_src, stream);
        res &= transfer_internal_memory(p_int, n, def_mem_type, type_tgt, stream);
        res &= IvyMemoryHelpers::transfer_memory(tgt, p_int, n, type_tgt, def_mem_type, stream);
        dump(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::free_memory(p_int, n, def_mem_type, stream);
      }
      return res;
    }
  };
  template<typename T, IvyPointerType IPT> class deallocator_primitive<IvyUnifiedPtr<T, IPT>>{
  public:
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void call_IvyUnifiedPtr_reset(IvyUnifiedPtr<T, IPT>*& ptr){ ptr->reset(); }

  protected:
    static __CUDA_HOST_DEVICE__ void reset(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t const& n, IvyMemoryType const& mem_type, IvyGPUStream& stream){
#ifndef __CUDA_DEVICE_CODE__
      if (IvyMemoryHelpers::is_device_memory(mem_type)){
        IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
        if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
          kernel_IvyUnifiedPtr_reset<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(ptr, n);
        }
        else{
          __PRINT_ERROR__("deallocator_primitive::reset: Unable to call the GPU kernel...\n");
        }
      }
      else{
#else
      {
#endif
        IvyUnifiedPtr<T, IPT>* pr = ptr;
        for (size_type i=0; i<n; i++){
          call_IvyUnifiedPtr_reset(pr);
          ++pr;
        }
      }
    }

  public:
    using base_t = allocation_type_properties<IvyUnifiedPtr<T, IPT>>;
    using value_type = typename base_t::value_type;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool deallocate(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      bool res = true;
      reset(p, n, mem_type, stream);
      res &= IvyMemoryHelpers::free_memory(p, n, mem_type, stream);
      return res;
    }
  };

  template<typename T, IvyPointerType IPT> __CUDA_GLOBAL__ void kernel_IvyUnifiedPtr_transfer_internal_memory(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t n, IvyMemoryType mem_type){
    IvyTypes::size_t i = 0;
    get_kernel_call_dims_1D(i);
    if (i<n) transfer_memory_primitive<IvyUnifiedPtr<T, IPT>>::call_IvyUnifiedPtr_transfer_internal_memory(ptr+i, mem_type);
  }
  template<typename T, IvyPointerType IPT> __CUDA_GLOBAL__ void kernel_IvyUnifiedPtr_dump(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t n){
    IvyTypes::size_t i = 0;
    get_kernel_call_dims_1D(i);
    if (i<n) transfer_memory_primitive<IvyUnifiedPtr<T, IPT>>::call_IvyUnifiedPtr_dump(ptr+i);
  }
  template<typename T, IvyPointerType IPT> __CUDA_GLOBAL__ void kernel_IvyUnifiedPtr_reset(IvyUnifiedPtr<T, IPT>* ptr, IvyTypes::size_t n){
    IvyTypes::size_t i = 0;
    get_kernel_call_dims_1D(i);
    if (i<n) deallocator_primitive<IvyUnifiedPtr<T, IPT>>::call_IvyUnifiedPtr_reset(ptr+i);
  }

}

#endif


#endif
