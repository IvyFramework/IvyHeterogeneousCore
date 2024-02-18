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
    capacity_(nullptr),
    ref_count_(nullptr),
    stream_(nullptr)
  {}
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(std_cstddef::nullptr_t) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    mem_type_(nullptr),
    ptr_(nullptr),
    size_(nullptr),
    capacity_(nullptr),
    ref_count_(nullptr),
    stream_(nullptr)
  {}
  template<typename T, IvyPointerType IPT>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    ptr_(ptr),
    stream_(stream)
  {
    if (ptr_) this->init_members(mem_type, 1, 1);
  }
  template<typename T, IvyPointerType IPT>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    ptr_(ptr),
    stream_(stream)
  {
    if (ptr_) this->init_members(mem_type, n, n);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    stream_(stream)
  {
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_) this->init_members(mem_type, 1, 1);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream) :
    exec_mem_type_(IvyMemoryHelpers::get_execution_default_memory()),
    stream_(stream)
  {
    ptr_ = __DYNAMIC_CAST__(pointer, ptr);
    if (ptr_) this->init_members(mem_type, n, n);
  }
  template<typename T, IvyPointerType IPT>
  template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other) :
    stream_(other.gpu_stream())
  {
    ptr_ = __DYNAMIC_CAST__(pointer, other.get());
    if (!ptr_ && other.get()){
      __PRINT_ERROR__("IvyUnifiedPtr copy constructor failed: Incompatible types\n");
      assert(false);
    }
    else{
      exec_mem_type_ = other.get_exec_memory_type();
      mem_type_ = other.get_memory_type_ptr();
      size_ = other.size_ptr();
      capacity_ = other.capacity_ptr();
      ref_count_ = other.counter();
      if (ref_count_) this->inc_dec_counter(true);
    }
    if (IPU==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<U, IPU>&), other).reset();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other) :
    stream_(other.gpu_stream())
  {
    ptr_ = other.ptr_;
    exec_mem_type_ = other.exec_mem_type_;
    mem_type_ = other.mem_type_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    ref_count_ = other.ref_count_;
    if (ref_count_) this->inc_dec_counter(true);
    if (IPT==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
  }
  template<typename T, IvyPointerType IPT> template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool>> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other) :
    exec_mem_type_(std_util::move(other.get_exec_memory_type())),
    mem_type_(std_util::move(other.get_memory_type_ptr())),
    ptr_(std_util::move(other.get())),
    size_(std_util::move(other.size_ptr())),
    capacity_(std_util::move(other.capacity_ptr())),
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
    capacity_(std_util::move(other.capacity_)),
    ref_count_(std_util::move(other.ref_count_)),
    stream_(std_util::move(other.gpu_stream()))
  {
    other.dump();
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::~IvyUnifiedPtr(){
    this->reset();
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
      capacity_ = other.capacity_ptr();
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
      capacity_ = other.capacity_;
      ref_count_ = other.ref_count_;
      stream_ = other.gpu_stream();
      if (ref_count_) this->inc_dec_counter(true);
    }
    if (IPT==IvyPointerType::unique) __CONST_CAST__(__ENCAPSULATE__(IvyUnifiedPtr<T, IPT>&), other).reset();
    return *this;
  }
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(U* ptr){ this->reset(ptr); return *this; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& IvyUnifiedPtr<T, IPT>::operator=(std_cstddef::nullptr_t){ this->reset(nullptr); return *this; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::init_members(IvyMemoryType mem_type, size_type n_size, size_type n_capacity){
    assert(n_size<=n_capacity);
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        size_ = size_allocator_traits::build(1, exec_mem_type_, ref_stream, n_size);
        capacity_ = size_allocator_traits::build(1, exec_mem_type_, ref_stream, n_capacity);
        ref_count_ = counter_allocator_traits::build(1, exec_mem_type_, ref_stream, __STATIC_CAST__(counter_type, 1));
        mem_type_ = mem_type_allocator_traits::build(1, exec_mem_type_, ref_stream, mem_type);
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
            element_allocator_traits::destruct(ptr_, this->size(), current_mem_type, ref_stream);
            element_allocator_traits::deallocate(ptr_, this->capacity(), current_mem_type, ref_stream);
            size_allocator_traits::destroy(size_, 1, exec_mem_type_, ref_stream);
            size_allocator_traits::destroy(capacity_, 1, exec_mem_type_, ref_stream);
            counter_allocator_traits::destroy(ref_count_, 1, exec_mem_type_, ref_stream);
            mem_type_allocator_traits::destroy(mem_type_, 1, exec_mem_type_, ref_stream);
          )
        );
      }
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::dump(){
    mem_type_ = nullptr;
    ptr_ = nullptr;
    size_ = nullptr;
    capacity_ = nullptr;
    ref_count_ = nullptr;
    stream_ = nullptr;
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType const& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() const __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType* IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() const __NOEXCEPT__{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyGPUStream* IvyUnifiedPtr<T, IPT>::gpu_stream() const __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::size_ptr() const __NOEXCEPT__{ return this->size_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type* IvyUnifiedPtr<T, IPT>::capacity_ptr() const __NOEXCEPT__{ return this->capacity_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::counter_type* IvyUnifiedPtr<T, IPT>::counter() const __NOEXCEPT__{ return this->ref_count_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::pointer IvyUnifiedPtr<T, IPT>::get() const __NOEXCEPT__{ return this->ptr_; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType& IvyUnifiedPtr<T, IPT>::get_exec_memory_type() __NOEXCEPT__{ return this->exec_mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyMemoryType*& IvyUnifiedPtr<T, IPT>::get_memory_type_ptr() __NOEXCEPT__{ return this->mem_type_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyGPUStream*& IvyUnifiedPtr<T, IPT>::gpu_stream() __NOEXCEPT__{ return this->stream_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type*& IvyUnifiedPtr<T, IPT>::size_ptr() __NOEXCEPT__{ return this->size_; }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type*& IvyUnifiedPtr<T, IPT>::capacity_ptr() __NOEXCEPT__{ return this->capacity_; }
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
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::size_type IvyUnifiedPtr<T, IPT>::capacity() const __NOEXCEPT__{
    if (!capacity_) return 0;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type) return *capacity_;
    else{
      size_type* p_capacity_ = nullptr;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            p_capacity_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
            size_allocator_traits::transfer(p_capacity_, capacity_, 1, def_mem_type, exec_mem_type_, ref_stream);
          )
        );
      }
      size_type ret = *p_capacity_;
      {
        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            size_allocator_traits::deallocate(p_capacity_, 1, def_mem_type, ref_stream);
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
  template<typename T, IvyPointerType IPT> template<typename U> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->reset();
      stream_ = stream;
      ptr_ = __DYNAMIC_CAST__(pointer, ptr);
      if (ptr_) this->init_members(mem_type, n_size, n_capacity);
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
  __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, n, n, mem_type, stream);
  }
  template<typename T, IvyPointerType IPT> template<typename U>
  __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, __STATIC_CAST__(size_type, 1), mem_type, stream);
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
    bool const is_same = (ptr_ == ptr);
    if (!is_same){
      this->reset();
      stream_ = stream;
      ptr_ = ptr;
      if (ptr_) this->init_members(mem_type, n_size, n_capacity);
    }
    else{
      if (stream) stream_ = stream;
      if (this->get_memory_type() != mem_type){
        __PRINT_ERROR__("IvyUnifiedPtr::reset() failed: Incompatible mem_type flags.\n");
        assert(false);
      }
    }
  }
  template<typename T, IvyPointerType IPT>
  __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, n, n, mem_type, stream);
  }
  template<typename T, IvyPointerType IPT>
  __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reset(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->reset(ptr, __STATIC_CAST__(size_type, 1), mem_type, stream);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::transfer_internal_memory(IvyMemoryType const& new_mem_type){
    return this->transfer_impl(new_mem_type, true, true);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::transfer_impl(IvyMemoryType const& new_mem_type, bool transfer_all, bool copy_ptr){
    bool res = true;
    if (ptr_){
      auto const current_mem_type = this->get_memory_type();
      if (copy_ptr || current_mem_type != new_mem_type){
        IvyMemoryType misc_mem_type = (transfer_all ? new_mem_type : exec_mem_type_);
        pointer new_ptr_ = nullptr;
        size_type* new_size_ = nullptr;
        size_type* new_capacity_ = nullptr;
        counter_type* new_ref_count_ = nullptr;
        IvyMemoryType* new_mem_type_ = nullptr;
        size_type const n_size = this->size();
        size_type const n_capacity = this->capacity();

        operate_with_GPU_stream_from_pointer(
          stream_, ref_stream,
          __ENCAPSULATE__(
            res &= element_allocator_traits::allocate(new_ptr_, n_capacity, new_mem_type, ref_stream);
            res &= element_allocator_traits::transfer(new_ptr_, ptr_, n_capacity, new_mem_type, current_mem_type, ref_stream);
            res &= size_allocator_traits::build(new_size_, 1, misc_mem_type, ref_stream, n_size);
            res &= size_allocator_traits::build(new_capacity_, 1, misc_mem_type, ref_stream, n_capacity);
            res &= counter_allocator_traits::build(new_ref_count_, 1, misc_mem_type, ref_stream, __STATIC_CAST__(counter_type, 1));
            res &= mem_type_allocator_traits::build(new_mem_type_, 1, misc_mem_type, ref_stream, new_mem_type);
          )
        );

        if (!copy_ptr) this->release();

        exec_mem_type_ = misc_mem_type;
        ptr_ = new_ptr_;
        size_ = new_size_;
        capacity_ = new_capacity_;
        ref_count_ = new_ref_count_;
        mem_type_ = new_mem_type_;
      }
    }
    return res;
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST__ bool IvyUnifiedPtr<T, IPT>::transfer(IvyMemoryType const& new_mem_type, bool transfer_all){
    return this->transfer_impl(new_mem_type, transfer_all, false);
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::copy(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    bool res = true;
    if (ptr && n>0){
      this->reset();
      if (stream) stream_ = stream;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          res &= element_allocator_traits::allocate(ptr_, n, mem_type, ref_stream);
          res &= element_allocator_traits::transfer(ptr_, ptr, n, mem_type, mem_type, ref_stream);
          res &= size_allocator_traits::build(size_, 1, exec_mem_type_, ref_stream, n);
          res &= size_allocator_traits::build(capacity_, 1, exec_mem_type_, ref_stream, n);
          res &= counter_allocator_traits::build(ref_count_, 1, exec_mem_type_, ref_stream, __STATIC_CAST__(counter_type, 1));
          res &= mem_type_allocator_traits::build(mem_type_, 1, exec_mem_type_, ref_stream, mem_type);
        )
      );
    }
    return res;
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
    std_util::swap(capacity_, other.capacity_ptr());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(mem_type_, other.get_memory_type_ptr());
    std_util::swap(ref_count_, other.counter());
    std_util::swap(stream_, other.gpu_stream());
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::swap(IvyUnifiedPtr<T, IPT>& other) __NOEXCEPT__{
    std_util::swap(ptr_, other.get());
    std_util::swap(size_, other.size_ptr());
    std_util::swap(capacity_, other.capacity_ptr());
    std_util::swap(exec_mem_type_, other.get_exec_memory_type());
    std_util::swap(mem_type_, other.get_memory_type_ptr());
    std_util::swap(ref_count_, other.counter());
    std_util::swap(stream_, other.gpu_stream());
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
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_size(bool do_inc){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      if (do_inc) ++(*size_);
      else --(*size_);
    }
    else{
      size_type* p_size_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          p_size_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
          size_allocator_traits::transfer(p_size_, size_, 1, def_mem_type, exec_mem_type_, ref_stream);
          if (do_inc) ++(*p_size_);
          else --(*p_size_);
          size_allocator_traits::transfer(size_, p_size_, 1, exec_mem_type_, def_mem_type, ref_stream);
          size_allocator_traits::deallocate(p_size_, 1, def_mem_type, ref_stream);
        )
      );
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::inc_dec_capacity(bool do_inc, size_type inc){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    if (exec_mem_type_ == def_mem_type){
      if (do_inc) ++(*capacity_);
      else --(*capacity_);
    }
    else{
      size_type* p_capacity_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          p_capacity_ = size_allocator_traits::allocate(1, def_mem_type, ref_stream);
          size_allocator_traits::transfer(p_capacity_, capacity_, 1, def_mem_type, exec_mem_type_, ref_stream);
          if (do_inc) *p_capacity_ += inc;
          else *p_capacity_ -= inc;
          size_allocator_traits::transfer(capacity_, p_capacity_, 1, exec_mem_type_, def_mem_type, ref_stream);
          size_allocator_traits::deallocate(p_capacity_, 1, def_mem_type, ref_stream);
        )
      );
    }
  }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool IvyUnifiedPtr<T, IPT>::unique() const{ return this->use_count() == __STATIC_CAST__(counter_type, 1); }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>::operator bool() const __NOEXCEPT__{ return ptr_ != nullptr; }

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reserve(size_type const& n){
    auto n_capacity = this->capacity();
    if (n>n_capacity){
      auto mem_type = this->get_memory_type();
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n, mem_type, ref_stream);
          // Use barebones memory transfer; we do not want to call any transfer_internal_memory() function.
          IvyMemoryHelpers::transfer_memory(new_ptr_, ptr_, n_capacity, mem_type, mem_type, ref_stream);
          std_util::swap(new_ptr_, ptr_);
          element_allocator_traits::deallocate(new_ptr_, n_capacity, mem_type, ref_stream);
        )
      );
      this->inc_dec_capacity(true, n-n_capacity);
    }
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::reserve(size_type const& n, IvyMemoryType new_mem_type, IvyGPUStream* stream){
    auto n_capacity = this->capacity();
    auto mem_type = this->get_memory_type();
    if (stream) stream_ = stream;
    if (new_mem_type!=mem_type){
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n, new_mem_type, ref_stream);
        )
      );
      this->reset(new_ptr_, n, new_mem_type, stream);
    }
    else this->reserve(n);
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::shrink_to_fit(){
    auto n_capacity = this->capacity();
    auto n_size = this->size();
    if (n_size<n_capacity){
      auto mem_type = this->get_memory_type();
      pointer new_ptr_ = nullptr;
      operate_with_GPU_stream_from_pointer(
        stream_, ref_stream,
        __ENCAPSULATE__(
          element_allocator_traits::allocate(new_ptr_, n_size, mem_type, ref_stream);
          // Use barebones memory transfer; we do not want to call any transfer_internal_memory() function.
          IvyMemoryHelpers::transfer_memory(new_ptr_, ptr_, n_size, mem_type, mem_type, ref_stream);
          std_util::swap(new_ptr_, ptr_);
          element_allocator_traits::deallocate(new_ptr_, n_capacity, mem_type, ref_stream);
        )
      );
      this->inc_dec_capacity(false, n_capacity-n_size);
    }
  }
  template<typename T, IvyPointerType IPT> template<typename... Args> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::emplace_back(Args&&... args){
    auto n_size = this->size();
    auto n_capacity = this->capacity();
    if (n_size==n_capacity) this->reserve(n_capacity+2);
    auto const current_mem_type = this->get_memory_type();
    pointer ptr_loc = (ptr_+n_size);
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        element_allocator_traits::construct(ptr_loc, 1, current_mem_type, ref_stream, args...);
      )
    );
    this->inc_dec_size(true);
  }
  template<typename T, IvyPointerType IPT> template<typename... Args> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::insert(size_type const& i, Args&&... args){
    auto n_size = this->size();
    assert(i<n_size);
    auto n_capacity = this->capacity();
    auto const current_mem_type = this->get_memory_type();
    if (n_size==n_capacity) this->reserve(n_capacity+2);
    pointer ptr_here = (ptr_+i);
    pointer ptr_next = (ptr_+(i+1));
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        // Shift objects via barebones memory transfer; we do not want to call any transfer_internal_memory() function.
        IvyMemoryHelpers::transfer_memory(ptr_next, ptr_here, n_size-i, current_mem_type, current_mem_type, ref_stream);
        element_allocator_traits::construct(ptr_here, 1, current_mem_type, ref_stream, args...);
      )
    );
    this->inc_dec_size(true);
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::pop_back(){
    auto n_size = this->size();
    if (n_size==0) return;
    auto const current_mem_type = this->get_memory_type();
    pointer ptr_loc = (ptr_+(n_size-1));
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        element_allocator_traits::destruct(ptr_loc, 1, current_mem_type, ref_stream);
      )
    );
    this->inc_dec_size(false);
  }
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void IvyUnifiedPtr<T, IPT>::erase(size_type const& i){
    auto n_size = this->size();
    if (n_size==0) return;
    assert(i<n_size);
    auto n_capacity = this->capacity();
    auto const current_mem_type = this->get_memory_type();
    pointer ptr_here = (ptr_+i);
    pointer ptr_next = (ptr_+(i+1));
    operate_with_GPU_stream_from_pointer(
      stream_, ref_stream,
      __ENCAPSULATE__(
        element_allocator_traits::destruct(ptr_here, 1, current_mem_type, ref_stream);
        // Shift objects via barebones memory transfer; we do not want to call any transfer_internal_memory() function.
        IvyMemoryHelpers::transfer_memory(ptr_here, ptr_next, n_capacity-i-1, current_mem_type, current_mem_type, ref_stream);
      )
    );
    this->inc_dec_size(false);
  }


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
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.allocate(n_capacity, mem_type, ref_stream);
        a.construct(ptr, n_size, mem_type, ref_stream, args...);
      )
    );
    return IvyUnifiedPtr<T, IPT>(ptr, n_size, n_capacity, mem_type, stream);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return build_unified<T, Allocator_t, IvyPointerType::shared>(a, n_size, n_capacity, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return build_unified<T, Allocator_t, IvyPointerType::unique>(a, n_size, n_capacity, mem_type, stream, args...); }
  template<typename T, IvyPointerType IPT, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return build_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), n_size, n_capacity, mem_type, stream, args...);
  }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(n_size, n_capacity, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(n_size, n_capacity, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        typename IvyUnifiedPtr<T, IPT>::pointer ptr = a.build(n, mem_type, ref_stream, args...);
      )
    );
    return IvyUnifiedPtr<T, IPT>(ptr, n, mem_type, stream);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return build_unified<T, Allocator_t, IvyPointerType::shared>(a, n, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, typename unique_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return build_unified<T, Allocator_t, IvyPointerType::unique>(a, n, mem_type, stream, args...); }
  template<typename T, IvyPointerType IPT, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return build_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), n, mem_type, stream, args...);
  }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(n, mem_type, stream, args...); }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(n, mem_type, stream, args...); }

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return build_unified<T, IPT, Allocator_t, Args...>(a, __STATIC_CAST__(__ENCAPSULATE__(typename IvyUnifiedPtr<T, IPT>::size_type), 1), mem_type, stream, args...);
  }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return build_unified<T, Allocator_t, IvyPointerType::shared>(a, mem_type, stream, args...); }
  template<typename T, typename Allocator_t, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return build_unified<T, Allocator_t, IvyPointerType::unique>(a, mem_type, stream, args...); }
  template<typename T, IvyPointerType IPT, typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return build_unified<T, IPT, std_ivy::allocator<T>>(std_ivy::allocator<T>(), mem_type, stream, args...);
  }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::shared>(mem_type, stream, args...); }
  template<typename T, typename... Args>
  __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){ return make_unified<T, IvyPointerType::unique>(mem_type, stream, args...); }
}

#endif


#endif
