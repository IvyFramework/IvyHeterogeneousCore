#ifndef IVYCONTIGUOUSITERATORBUILDER_H
#define IVYCONTIGUOUSITERATORBUILDER_H


#include "std_ivy/iterator/IvyContiguousIterator.h"
#include "std_ivy/iterator/IvyReverseIterator.h"


namespace std_ivy{
  template<typename T> class IvyContiguousIteratorBuilder;
  template<typename T> class transfer_memory_primitive<IvyContiguousIteratorBuilder<T>> : public transfer_memory_primitive_with_internal_memory<IvyContiguousIteratorBuilder<T>>{};

  template<typename T> class IvyContiguousIteratorBuilder{
  public:
    using iterator_type = IvyContiguousIterator<T, contiguous_iterator_tag>;
    using reverse_iterator_type = std_ivy::reverse_iterator<iterator_type>;
    using value_type = std_ttraits::remove_cv_t<T>;
    using data_type = std_mem::unique_ptr<value_type>;
    using pointer = typename data_type::pointer;
    using size_type = typename data_type::size_type;
    using iterator_collection_t = std_mem::unique_ptr<iterator_type>;
    using pointable_t = iterator_type*;

    friend class kernel_generic_transfer_internal_memory<IvyContiguousIteratorBuilder<T>>;

    iterator_collection_t chain;

  protected:
    static __INLINE_FCN_FORCE__ __HOST_DEVICE__ auto _pref(pointable_t const& ptr) -> decltype(*ptr){ return *ptr; }

    __HOST_DEVICE__ bool fix_prev_next(size_type pos, char mem_loc_inc = 0){
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
                auto& tmp_ptr_mem_loc = tmp_ptr->get_ptr_mem_loc().get();
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

    __HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
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
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ pointable_t chain_rend() const{ return chain.get(); }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ pointable_t chain_front() const{
      if (!chain) return nullptr;
      size_type const n_size = chain.size();
      return std_mem::addressof(chain[1]);
    }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ pointable_t chain_back() const{
      if (!chain) return nullptr;
      size_type const n_size = chain.size();
      return std_mem::addressof(chain[n_size-2]);
    }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ pointable_t chain_end() const{
      if (!chain) return nullptr;
      size_type const n_size = chain.size();
      return std_mem::addressof(chain[n_size-1]);
    }

    __HOST_DEVICE__ void invalidate(){
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

    __HOST_DEVICE__ iterator_type begin() const{ return _pref(chain_front()); }
    __HOST_DEVICE__ iterator_type front() const{ return _pref(chain_front()); }
    __HOST_DEVICE__ iterator_type back() const{ return _pref(chain_back()); }
    __HOST_DEVICE__ iterator_type end() const{ return _pref(chain_end()); }
    __HOST_DEVICE__ reverse_iterator_type rbegin() const{ return reverse_iterator_type(_pref(chain_back())); }
    __HOST_DEVICE__ reverse_iterator_type rend() const{ return reverse_iterator_type(_pref(chain_rend())); }

    __HOST_DEVICE__ pointable_t find_pointable(pointer ptr) const{
      pointable_t current = chain_front();
      while (current){
        if (current->get_mem_loc() == ptr) return current;
        current = current->next();
      }
      return nullptr;
    }

    __HOST_DEVICE__ void reset(pointer ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
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
    __HOST_DEVICE__ void reset(pointer ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(ptr, n, n, mem_type, stream);
    }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ void reset(){ this->invalidate(); }

    __HOST_DEVICE__ void push_back(pointer ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
      size_type n_size = chain.size();
      chain.insert(n_size-1, ptr, mem_type, stream);
      fix_prev_next(n_size-1);
    }
    __HOST_DEVICE__ void pop_back(){
      if (!chain) return;
      size_type n_size = chain.size();
      if (n_size<3) return;
      chain[n_size-2].invalidate();
      chain.erase(n_size-2);
      fix_prev_next(n_size-2);
    }

    __HOST_DEVICE__ void insert(size_type const& pos, pointer ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
      chain.insert(pos+1, ptr, mem_type, stream);
      fix_prev_next(pos+1, +1);
    }
    __HOST_DEVICE__ void erase(size_type const& pos){
      if (!chain) return;
      size_type n_size = chain.size();
      if (pos+2>=n_size) return;
      chain[pos+1].invalidate();
      chain.erase(pos+1);
      fix_prev_next(pos+1, -1);
    }

    __HOST_DEVICE__ IvyContiguousIteratorBuilder(){}
    __HOST_DEVICE__ IvyContiguousIteratorBuilder(pointer ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(ptr, n, mem_type, stream);
    }
    __HOST_DEVICE__ IvyContiguousIteratorBuilder(pointer ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(ptr, n_size, n_capacity, mem_type, stream);
    }
    __HOST_DEVICE__ IvyContiguousIteratorBuilder(IvyContiguousIteratorBuilder const& other) : chain(other.chain){}
    __HOST_DEVICE__ IvyContiguousIteratorBuilder(IvyContiguousIteratorBuilder&& other) : chain(std_util::move(other.chain)){}
    __HOST_DEVICE__ ~IvyContiguousIteratorBuilder(){ invalidate(); }

    __HOST_DEVICE__ IvyContiguousIteratorBuilder& operator=(IvyContiguousIteratorBuilder const& other){
      chain = other.chain;
      return *this;
    }
    __HOST_DEVICE__ IvyContiguousIteratorBuilder& operator=(IvyContiguousIteratorBuilder&& other){
      chain = std_util::move(other.chain);
      return *this;
    }

    __HOST_DEVICE__ void swap(IvyContiguousIteratorBuilder& other){
      std_util::swap(chain, other.chain);
    }
  };
}
namespace std_util{
  template<typename T> __HOST_DEVICE__ void swap(std_ivy::IvyContiguousIteratorBuilder<T>& a, std_ivy::IvyContiguousIteratorBuilder<T>& b){ a.swap(b); }
}


#endif
