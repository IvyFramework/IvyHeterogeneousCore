#ifndef IVYVECTORITERATOR_H
#define IVYVECTORITERATOR_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyIterator.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T> struct IvyVectorIterator;
  template<typename T> struct IvyVectorIteratorBuilder;
  template<typename T> class transfer_memory_primitive<IvyVectorIterator<T>> : public transfer_memory_primitive_with_internal_memory<IvyVectorIterator<T>>{};
  template<typename T> class transfer_memory_primitive<IvyVectorIteratorBuilder<T>> : public transfer_memory_primitive_with_internal_memory<IvyVectorIteratorBuilder<T>>{};

  template<typename T> class IvyVectorIterator : public iterator<std_iter::contiguous_iterator_tag, T>{
  public:
    using Base_t = iterator<std_iter::contiguous_iterator_tag, T>;
    using value_type = typename Base_t::value_type;
    using pointer = typename Base_t::pointer;
    using reference = typename Base_t::reference;
    using difference_type = typename Base_t::difference_type;
    using iterator_category = typename Base_t::iterator_category;
    using mem_loc_t = pointer;
    using mem_loc_container_t = std_mem::shared_ptr<mem_loc_t>;
    using pointable_t = IvyVectorIterator<T>*;
    using const_pointable_t = IvyVectorIterator<T> const*;

    friend struct IvyVectorIteratorBuilder<T>;
    friend class kernel_generic_transfer_internal_memory<IvyVectorIterator<T>>;

  protected:
    mem_loc_container_t ptr_mem_loc_;
    pointable_t next_;
    pointable_t prev_;

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

#ifdef __CUDA_DEBUG__
  public:
#endif
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


#ifndef __CUDA_DEBUG__
  public:
#endif
    __CUDA_HOST_DEVICE__ IvyVectorIterator() : next_(nullptr), prev_(nullptr){}
    __CUDA_HOST_DEVICE__ IvyVectorIterator(IvyVectorIterator const& other) :
      ptr_mem_loc_(other.ptr_mem_loc_), next_(other.next_), prev_(other.prev_)
    {}
    __CUDA_HOST_DEVICE__ IvyVectorIterator(IvyVectorIterator&& other) :
      ptr_mem_loc_(std_util::move(other.ptr_mem_loc_)), next_(std_util::move(other.next_)), prev_(std_util::move(other.prev_))
    {
      other.next_ = other.prev_ = nullptr;
    }
    __CUDA_HOST_DEVICE__ IvyVectorIterator(pointer const& mem_loc, IvyMemoryType mem_type, IvyGPUStream* stream) :
      next_(nullptr), prev_(nullptr)
    {
      this->set_mem_loc(mem_loc, mem_type, stream);
    }
    __CUDA_HOST_DEVICE__ ~IvyVectorIterator(){}


    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t& next(){ return next_; }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t& prev(){ return prev_; }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t next() const{ return next_; }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t prev() const{ return prev_; }

    __CUDA_HOST_DEVICE__ IvyVectorIterator& operator=(IvyVectorIterator const& other){
      ptr_mem_loc_ = other.ptr_mem_loc_;
      next_ = other.next_;
      prev_ = other.prev_;
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyVectorIterator& operator=(IvyVectorIterator&& other){
      ptr_mem_loc_ = std_util::move(other.ptr_mem_loc_);
      next_ = std_util::move(other.next_); other.next_ = nullptr;
      prev_ = std_util::move(other.prev_); other.prev_ = nullptr;
      return *this;
    }

    __CUDA_HOST_DEVICE__ reference operator*() const{ return *(this->get_mem_loc_fast()); }
    __CUDA_HOST_DEVICE__ pointer const& operator->() const{ return this->get_mem_loc_fast(); }
    __CUDA_HOST_DEVICE__ bool is_valid() const{ return (ptr_mem_loc_ && this->get_mem_loc()); }

    __CUDA_HOST_DEVICE__ IvyVectorIterator<T>& operator++(){
      *this = *(this->next());
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T> operator++(int){ IvyVectorIterator<T> tmp(*this); operator++(); return tmp; }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T>& operator--(){
      *this = *(this->prev());
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T> operator--(int){ IvyVectorIterator<T> tmp(*this); operator--(); return tmp; }

    __CUDA_HOST_DEVICE__ IvyVectorIterator<T> operator+(difference_type n) const{
      if (n==0) return *this;
      IvyVectorIterator<T> tmp(*this);
      for (difference_type i=0; i<n; ++i) ++tmp;
      return tmp;
    }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T> operator-(difference_type n) const{
      if (n==0) return *this;
      IvyVectorIterator<T> tmp(*this);
      for (difference_type i=0; i<n; ++i) --tmp;
      return tmp;
    }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T>& operator+=(difference_type n){ *this = *this + n; return *this; }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T>& operator-=(difference_type n){ *this = *this - n; return *this; }

    __CUDA_HOST_DEVICE__ bool operator==(IvyVectorIterator<T> const& other) const{
      return
        (ptr_mem_loc_ == other.ptr_mem_loc_)
        &&
        (!ptr_mem_loc_ || this->get_mem_loc()==other.get_mem_loc());
    }
    __CUDA_HOST_DEVICE__ bool operator!=(IvyVectorIterator<T> const& other) const{ return !(*this==other); }

    __CUDA_HOST_DEVICE__ difference_type operator-(IvyVectorIterator<T> const& other) const{
      if (other == *this || (!other.is_valid() && !this->is_valid())) return 0;
      difference_type n = 0;
      IvyVectorIterator<T> current = other;
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
  template<typename T> using IvyVectorConstIterator = IvyVectorIterator<T const>;

  template<typename T> class IvyVectorIteratorBuilder{
  public:
    using iterator_type = IvyVectorIterator<T>;
    using reverse_iterator_type = std_iter::reverse_iterator<iterator_type>;
    using value_type = std_ttraits::remove_cv_t<T>;
    using data_type = std_ivy::unique_ptr<value_type>;
    using pointer = typename data_type::pointer;
    using size_type = typename data_type::size_type;
    using iterator_collection_t = std_ivy::unique_ptr<iterator_type>;
    using pointable_t = iterator_type*;

    friend class kernel_generic_transfer_internal_memory<IvyVectorIteratorBuilder<T>>;

    iterator_collection_t chain;

  protected:
    static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ auto _pref(pointable_t const& ptr) -> decltype(*ptr){ return *ptr; }

    __CUDA_HOST_DEVICE__ bool fix_prev_next(size_type pos, char mem_loc_inc = 0){
      size_type const n = chain.size();
      bool res = true;
      if (n<=2 || pos<1 || pos>n-1) return res;
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      IvyMemoryType const mem_type = chain.get_memory_type();
      IvyGPUStream* stream = chain.gpu_stream();
      size_type pos_start = pos-1;
      pointable_t current = chain.get() + pos_start;
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          for (size_type i=pos_start; i<n; ++i){
            if (def_mem_type==mem_type){
              if (i<n-1) current->next() = current+1;
              if (i>0) current->prev() = current-1;
              if (
                (i>0 && i<n-1)
                &&
                ((mem_loc_inc<0 && i>=pos) || (mem_loc_inc>0 && i>pos))
                ) current->get_mem_loc_fast() += mem_loc_inc;
            }
            else{
              pointable_t tmp_ptr = nullptr;
              res &= IvyMemoryHelpers::allocate_memory(tmp_ptr, 1, def_mem_type, ref_stream);
              res &= IvyMemoryHelpers::transfer_memory(tmp_ptr, current, 1, def_mem_type, mem_type, ref_stream);
              if (i<n-1) tmp_ptr->next() = current+1;
              if (i>0) tmp_ptr->prev() = current-1;
              if (
                (i>0 && i<n-1)
                &&
                ((mem_loc_inc<0 && i>=pos) || (mem_loc_inc>0 && i>pos))
                ){
                auto& tmp_ptr_mem_loc = tmp_ptr->ptr_mem_loc_.get();
                typename iterator_type::mem_loc_t* tmp_mem_loc = nullptr;
                res &= IvyMemoryHelpers::allocate_memory(tmp_mem_loc, 1, def_mem_type, ref_stream);
                res &= IvyMemoryHelpers::transfer_memory(tmp_mem_loc, tmp_ptr_mem_loc, 1, def_mem_type, mem_type, ref_stream);
                *tmp_mem_loc = *tmp_mem_loc + mem_loc_inc;
                res &= IvyMemoryHelpers::transfer_memory(tmp_ptr_mem_loc, tmp_mem_loc, 1, mem_type, def_mem_type, ref_stream);
                res &= IvyMemoryHelpers::free_memory(tmp_mem_loc, 1, def_mem_type, ref_stream);
              }
              res &= IvyMemoryHelpers::transfer_memory(current, tmp_ptr, 1, mem_type, def_mem_type, ref_stream);
              res &= IvyMemoryHelpers::free_memory(tmp_ptr, 1, def_mem_type, ref_stream);
            }
            ++current;
          }
        )
      );
      return res;
    }

    __CUDA_HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
      bool res = true;
      if (!chain) return res;
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      IvyGPUStream* stream = chain.gpu_stream();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          res &= std_mem::allocator<iterator_collection_t>::transfer_internal_memory(&chain, 1, def_mem_type, new_mem_type, ref_stream, release_old);
          // After internal memory transfer, prev and next pointers are broken, so we need to fix them.
          res &= fix_prev_next(1);
        )
      );
      return res;
    }

  public:
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t chain_rend() const{ return chain.get(); }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t chain_front() const{
      if (!chain) return nullptr;
      size_type const n_size = chain.size();
      return std_mem::addressof(chain[1]);
    }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t chain_back() const{
      if (!chain) return nullptr;
      size_type const n_size = chain.size();
      return std_mem::addressof(chain[n_size-2]);
    }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ pointable_t chain_end() const{
      if (!chain) return nullptr;
      size_type const n_size = chain.size();
      return std_mem::addressof(chain[n_size-1]);
    }

    __CUDA_HOST_DEVICE__ void invalidate(){
      if (!chain) return;
      size_type const n = chain.size();
      pointable_t current = chain.get();
      IvyMemoryType const def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      IvyMemoryType mem_type = chain.get_memory_type();
      IvyGPUStream* stream = chain.gpu_stream();
      //char const* chain_type = __TYPE_NAME__(chain);
      /*
      __PRINT_DEBUG__("chain <%s> invalidation with mem_type = %s, def_mem_type = %s, mem_type_ptr = %p, size_ptr = %p, n = %llu, chain head = %p\n",
        chain_type,
        IvyMemoryHelpers::get_memory_type_name(mem_type), IvyMemoryHelpers::get_memory_type_name(def_mem_type),
        chain.get_memory_type_ptr(), chain.size_ptr(),
        n, chain.get()
      );
      */
      for (size_type i=0; i<n; ++i){
        if (mem_type==def_mem_type){
          //__PRINT_DEBUG__("  invalidate; Calling invalidate directly on %p\n", current);
          current->invalidate();
          //__PRINT_DEBUG__("  invalidate; Calling invalidate directly on %p is done.\n", current);
        }
        else{
          operate_with_GPU_stream_from_pointer(
            stream, ref_stream,
            __ENCAPSULATE__(
              pointable_t tmp_ptr = nullptr;
              //__PRINT_DEBUG__("  invalidate; Calling allocate_memory\n");
              IvyMemoryHelpers::allocate_memory(tmp_ptr, 1, def_mem_type, ref_stream);
              //__PRINT_DEBUG__("  invalidate; Calling transfer_memory %p -> %p\n", current, tmp_ptr);
              IvyMemoryHelpers::transfer_memory(tmp_ptr, current, 1, def_mem_type, mem_type, ref_stream);
              //__PRINT_DEBUG__("  invalidate; Calling invalidate on %p\n", tmp_ptr);
              tmp_ptr->invalidate();
              //__PRINT_DEBUG__("  invalidate; Calling transfer_memory %p -> %p\n", tmp_ptr, current);
              IvyMemoryHelpers::transfer_memory(current, tmp_ptr, 1, mem_type, def_mem_type, ref_stream);
              //__PRINT_DEBUG__("  invalidate; Calling free_memory on %p\n", tmp_ptr);
              IvyMemoryHelpers::free_memory(tmp_ptr, 1, def_mem_type, ref_stream);
            )
          );
        }
        ++current;
      }
      //__PRINT_DEBUG__("chain <%s> invalidation is done. Calling reset...\n", chain_type);
      chain.reset();
      //__PRINT_DEBUG__("chain <%s> invalidation is done. Calling reset is done.\n", chain_type);
    }

    __CUDA_HOST_DEVICE__ iterator_type begin() const{ return _pref(chain_front()); }
    __CUDA_HOST_DEVICE__ iterator_type front() const{ return _pref(chain_front()); }
    __CUDA_HOST_DEVICE__ iterator_type back() const{ return _pref(chain_back()); }
    __CUDA_HOST_DEVICE__ iterator_type end() const{ return _pref(chain_end()); }
    __CUDA_HOST_DEVICE__ reverse_iterator_type rbegin() const{ return reverse_iterator_type(_pref(chain_back())); }
    __CUDA_HOST_DEVICE__ reverse_iterator_type rend() const{ return reverse_iterator_type(_pref(chain_rend())); }

    __CUDA_HOST_DEVICE__ pointable_t find_pointable(pointer ptr) const{
      pointable_t current = chain_front();
      while (current){
        if (current->get_mem_loc() == ptr) return current;
        current = current->next();
      }
      return nullptr;
    }

    __CUDA_HOST_DEVICE__ void reset(pointer ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
      this->invalidate();
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      chain = std_mem::make_unique<iterator_type>(n_size+2, n_capacity+2, def_mem_type, stream);
      if (n_size>0){
        pointable_t current = chain.get();
        pointer ptr_data = ptr;
        for (size_type i=0; i<n_size+2; ++i){
          if (i==0) current->next() = current+1;
          else if (i==n_size+1) current->prev() = current-1;
          else{
            current->set_mem_loc(ptr_data, mem_type, stream);
            ++ptr_data;
          }
          ++current;
        }
      }
      if (mem_type!=def_mem_type){
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            std_mem::allocator<iterator_collection_t>::transfer_internal_memory(&chain, 1, def_mem_type, mem_type, ref_stream, true);
          )
        );
        fix_prev_next(1);
      }
    }
    __CUDA_HOST_DEVICE__ void reset(pointer ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(ptr, n, n, mem_type, stream);
    }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ void reset(){ this->invalidate(); }

    __CUDA_HOST_DEVICE__ void push_back(pointer ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
      size_type n_size = chain.size();
      chain.insert(n_size-1, ptr, mem_type, stream);
      fix_prev_next(n_size-1);
    }
    __CUDA_HOST_DEVICE__ void pop_back(){
      if (!chain) return;
      size_type n_size = chain.size();
      if (n_size<3) return;
      chain[n_size-2].invalidate();
      chain.erase(n_size-2);
      fix_prev_next(n_size-2);
    }

    __CUDA_HOST_DEVICE__ void insert(size_type const& pos, pointer ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
      chain.insert(pos+1, ptr, mem_type, stream);
      fix_prev_next(pos+1, +1);
    }
    __CUDA_HOST_DEVICE__ void erase(size_type const& pos){
      size_type n_size = chain.size();
      if (!chain || pos+2>=n_size) return;
      chain[pos+1].invalidate();
      chain.erase(pos+1);
      fix_prev_next(pos+1, -1);
    }

    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(){}
    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(pointer ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(ptr, n, mem_type, stream);
    }
    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(pointer ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(ptr, n_size, n_capacity, mem_type, stream);
    }
    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(IvyVectorIteratorBuilder const& other) : chain(other.chain){}
    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(IvyVectorIteratorBuilder&& other) : chain(std_util::move(other.chain)){}
    __CUDA_HOST_DEVICE__ ~IvyVectorIteratorBuilder(){ invalidate(); }
  };
}

#endif


#endif
