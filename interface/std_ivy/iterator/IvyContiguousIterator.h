#ifndef IVYCONTIGUOUSITERATOR_H
#define IVYCONTIGUOUSITERATOR_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/iterator/IvyIteratorPrimitives.h"
#include "std_ivy/iterator/IvyIteratorTraits.h"
#include "std_ivy/iterator/IvyIteratorImpl.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, typename ContiguousTag = contiguous_iterator_tag> class IvyContiguousIterator;
  template<typename T, typename ContiguousTag> class transfer_memory_primitive<IvyContiguousIterator<T, ContiguousTag>> : public transfer_memory_primitive_with_internal_memory<IvyContiguousIterator<T, ContiguousTag>>{};

  template<typename T, typename ContiguousTag> class IvyContiguousIterator : public iterator<ContiguousTag, T>{
  public:
    using Base_t = iterator<ContiguousTag, T>;
    using value_type = typename Base_t::value_type;
    using pointer = typename Base_t::pointer;
    using reference = typename Base_t::reference;
    using difference_type = typename Base_t::difference_type;
    using iterator_category = typename Base_t::iterator_category;
    using mem_loc_t = pointer;
    using mem_loc_container_t = std_mem::shared_ptr<mem_loc_t>;
    using pointable_t = IvyContiguousIterator<T, ContiguousTag>*;
    using const_pointable_t = IvyContiguousIterator<T, ContiguousTag> const*;

    friend class kernel_generic_transfer_internal_memory<IvyContiguousIterator<T, ContiguousTag>>;

  protected:
    mem_loc_container_t ptr_mem_loc_;
    pointable_t next_;
    pointable_t prev_;

    __CUDA_HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      bool res = true;
      IvyGPUStream* stream = ptr_mem_loc_.gpu_stream();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          res = std_mem::allocator<mem_loc_container_t>::transfer_internal_memory(&ptr_mem_loc_, 1, def_mem_type, new_mem_type, ref_stream, release_old);
      )
      );
      return res;
    }

  public:
    __CUDA_HOST_DEVICE__ void set_mem_loc(pointer const& mem_loc, IvyMemoryType mem_type, IvyGPUStream* stream){
      ptr_mem_loc_ = std_mem::make_shared<mem_loc_t>(mem_type, stream, mem_loc);
      next_ = prev_ = this;
      ++next_;
      --prev_;
    }

    __CUDA_HOST_DEVICE__ void invalidate(){
      if (ptr_mem_loc_){
        // Reset the value of the memory location pointer to null so that all related iterators that point to the same memory location are invalidated.
        // Do it using memory transfer in order to account for iterators residing on a different memory space.
        constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
        IvyMemoryType const mem_type = ptr_mem_loc_.get_memory_type();
        if (mem_type==def_mem_type) *ptr_mem_loc_ = nullptr;
        else{
          IvyGPUStream* stream = ptr_mem_loc_.gpu_stream();
          mem_loc_t*& p_mem_loc_ = ptr_mem_loc_.get();
          mem_loc_t new_mem_loc_ = nullptr;
          operate_with_GPU_stream_from_pointer(
            stream, ref_stream,
            __ENCAPSULATE__(
              std_mem::allocator<mem_loc_t>::transfer(p_mem_loc_, &new_mem_loc_, 1, mem_type, def_mem_type, ref_stream);
            )
          );
        }
      }
      ptr_mem_loc_.reset(); // This line decouples this iterator from previously related iterators.
      next_ = prev_ = nullptr;
    }

    __CUDA_HOST_DEVICE__ mem_loc_t get_mem_loc() const{
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      if (!ptr_mem_loc_) return nullptr;
      IvyMemoryType const mem_type = ptr_mem_loc_.get_memory_type();
      if (mem_type==def_mem_type) return *ptr_mem_loc_;
      IvyGPUStream* stream = ptr_mem_loc_.gpu_stream();
      mem_loc_t res = nullptr;
      mem_loc_t* p_res = &res;
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          std_mem::allocator<mem_loc_t>::transfer(p_res, ptr_mem_loc_.get(), 1, def_mem_type, mem_type, ref_stream);
        )
      );
      return res;
    }
    __CUDA_HOST_DEVICE__ mem_loc_t& get_mem_loc_fast(){ return *ptr_mem_loc_; }
    __CUDA_HOST_DEVICE__ mem_loc_t const& get_mem_loc_fast() const{ return *ptr_mem_loc_; }

    __CUDA_HOST_DEVICE__ mem_loc_container_t& get_ptr_mem_loc(){ return ptr_mem_loc_; }
    __CUDA_HOST_DEVICE__ mem_loc_container_t const& get_ptr_mem_loc() const{ return ptr_mem_loc_; }

    __CUDA_HOST_DEVICE__ IvyContiguousIterator() : next_(nullptr), prev_(nullptr){}
    __CUDA_HOST_DEVICE__ IvyContiguousIterator(IvyContiguousIterator const& other) :
      ptr_mem_loc_(other.ptr_mem_loc_), next_(other.next_), prev_(other.prev_)
    {}
    __CUDA_HOST_DEVICE__ IvyContiguousIterator(IvyContiguousIterator&& other) :
      ptr_mem_loc_(std_util::move(other.ptr_mem_loc_)), next_(std_util::move(other.next_)), prev_(std_util::move(other.prev_))
    {
      other.next_ = other.prev_ = nullptr;
    }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator(pointer const& mem_loc, IvyMemoryType mem_type, IvyGPUStream* stream) :
      next_(nullptr), prev_(nullptr)
    {
      this->set_mem_loc(mem_loc, mem_type, stream);
    }
    __CUDA_HOST_DEVICE__ ~IvyContiguousIterator(){}

    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t& next(){ return next_; }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t& prev(){ return prev_; }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t next() const{ return next_; }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t prev() const{ return prev_; }

    __CUDA_HOST_DEVICE__ IvyContiguousIterator& operator=(IvyContiguousIterator const& other){
      ptr_mem_loc_ = other.ptr_mem_loc_;
      next_ = other.next_;
      prev_ = other.prev_;
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator& operator=(IvyContiguousIterator&& other){
      ptr_mem_loc_ = std_util::move(other.ptr_mem_loc_);
      next_ = std_util::move(other.next_); other.next_ = nullptr;
      prev_ = std_util::move(other.prev_); other.prev_ = nullptr;
      return *this;
    }
    __CUDA_HOST_DEVICE__ void swap(IvyContiguousIterator& other){
      std_util::swap(ptr_mem_loc_, other.ptr_mem_loc_);
      std_util::swap(next_, other.next_);
      std_util::swap(prev_, other.prev_);
    }

    __CUDA_HOST_DEVICE__ reference operator*() const{ return *(this->get_mem_loc_fast()); }
    __CUDA_HOST_DEVICE__ pointer const& operator->() const{ return this->get_mem_loc_fast(); }
    __CUDA_HOST_DEVICE__ bool is_valid() const{ return (ptr_mem_loc_ && this->get_mem_loc()); }

    __CUDA_HOST_DEVICE__ IvyContiguousIterator& operator++(){
      *this = *(this->next());
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator operator++(int){ IvyContiguousIterator tmp(*this); operator++(); return tmp; }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator& operator--(){
      *this = *(this->prev());
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator operator--(int){ IvyContiguousIterator tmp(*this); operator--(); return tmp; }

    __CUDA_HOST_DEVICE__ IvyContiguousIterator operator+(difference_type n) const{
      if (n==0) return *this;
      IvyContiguousIterator tmp(*this);
      for (difference_type i=0; i<n; ++i) ++tmp;
      return tmp;
    }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator operator-(difference_type n) const{
      if (n==0) return *this;
      IvyContiguousIterator tmp(*this);
      for (difference_type i=0; i<n; ++i) --tmp;
      return tmp;
    }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator& operator+=(difference_type n){ *this = *this + n; return *this; }
    __CUDA_HOST_DEVICE__ IvyContiguousIterator& operator-=(difference_type n){ *this = *this - n; return *this; }

    __CUDA_HOST_DEVICE__ bool operator==(IvyContiguousIterator const& other) const{
      return
        (ptr_mem_loc_ == other.ptr_mem_loc_)
        &&
        (!ptr_mem_loc_ || this->get_mem_loc()==other.get_mem_loc());
    }
    __CUDA_HOST_DEVICE__ bool operator!=(IvyContiguousIterator const& other) const{ return !(*this==other); }

    __CUDA_HOST_DEVICE__ difference_type operator-(IvyContiguousIterator const& other) const{
      if (other == *this || (!other.is_valid() && !this->is_valid())) return 0;
      difference_type n = 0;
      if constexpr (std_util::is_base_of_v<contiguous_iterator_tag, ContiguousTag>){
        if (this->is_valid() && other.is_valid()) return (this->get_mem_loc() - other.get_mem_loc());
      }
      IvyContiguousIterator current = other;
      while (current.is_valid()){
        if (current == *this) return n;
        ++n;
        ++current;
      }
      if (!this->is_valid()) return n; // e.g., end - begin
      n = 0;
      current = *this;
      while (current.is_valid()){
        if (current == other) return n;
        --n;
        ++current;
      }
      if (!other.is_valid()) return n; // e.g., begin - end
      return 0;
    }
  };
  template<typename T, typename ContiguousTag = contiguous_iterator_tag> using IvyVectorConstIterator = IvyContiguousIterator<T const, ContiguousTag>;
}
namespace std_util{
  template<typename T, typename ContiguousTag> void swap(std_ivy::IvyContiguousIterator<T, ContiguousTag>& a, std_ivy::IvyContiguousIterator<T, ContiguousTag>& b){ a.swap(b); }
}

#endif


#endif
