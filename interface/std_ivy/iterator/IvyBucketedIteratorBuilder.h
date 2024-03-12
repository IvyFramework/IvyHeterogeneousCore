#ifndef IVYBUCKETEDITERATORBUILDER_H
#define IVYBUCKETEDITERATORBUILDER_H


#include "std_ivy/IvyUtility.h"
#include "std_ivy/iterator/IvyContiguousIterator.h"
#include "std_ivy/iterator/IvyReverseIterator.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename Key, typename T, typename Hash> class IvyBucketedIteratorBuilder;
  template<typename Key, typename T, typename Hash> class transfer_memory_primitive<IvyBucketedIteratorBuilder<Key, T, Hash>> : public transfer_memory_primitive_with_internal_memory<IvyBucketedIteratorBuilder<Key, T, Hash>>{};

  template<typename Key, typename T, typename Hash> class IvyBucketedIteratorBuilder{
  public:
    using hasher = Hash;
    using value_type = std_util::pair<Key, T>;
    using const_value_type = std_util::pair<Key, T> const;
    using iterator_type = IvyContiguousIterator<value_type, partially_contiguous_iterator_tag>;
    using const_iterator_type = IvyContiguousIterator<const_value_type, partially_contiguous_iterator_tag>;
    using reverse_iterator_type = std_ivy::reverse_iterator<iterator_type>;
    using const_reverse_iterator_type = std_ivy::reverse_iterator<const_iterator_type>;

    using hash_result_type = typename hasher::result_type;
    using bucket_element_type = std_util::pair<
      hash_result_type,
      std_mem::unique_ptr<value_type>
    >;
    using allocator_bucket_element_type = std_mem::allocator<bucket_element_type>;
    using data_type = std_mem::unique_ptr<bucket_element_type>;
    using pointer = value_type*;
    using size_type = typename data_type::size_type;

    using iterator_collection_t = std_mem::unique_ptr<iterator_type>;
    using pointable_t = iterator_type*;
    using const_iterator_collection_t = std_mem::unique_ptr<const_iterator_type>;
    using const_pointable_t = const_iterator_type*;

    friend class kernel_generic_transfer_internal_memory<IvyBucketedIteratorBuilder<Key, T, Hash>>;

    iterator_collection_t chain;
    const_iterator_collection_t chain_const;

  protected:
    template<typename Iterator> static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ Iterator get_iterator(Iterator* const& ptr){ return (ptr ? *ptr : Iterator()); }

    template<typename Chain_t> __CUDA_HOST_DEVICE__ bool fix_prev_next(Chain_t& ch_obj, size_type pos){
      typedef typename Chain_t::pointer chain_pointer_t;
      size_type const n = ch_obj.size();
      bool res = true;
      if (n<=2 || pos<1 || pos>n-1) return res;
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      IvyMemoryType const mem_type = ch_obj.get_memory_type();
      IvyGPUStream* stream = ch_obj.gpu_stream();
      size_type pos_start = pos-1;
      chain_pointer_t current = ch_obj.get() + pos_start;
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          for (size_type i=pos_start; i<n; ++i){
            if (def_mem_type==mem_type){
              if (i<n-1) current->next() = current+1;
              if (i>0) current->prev() = current-1;
            }
            else{
              chain_pointer_t tmp_ptr = nullptr;
              res &= IvyMemoryHelpers::allocate_memory(tmp_ptr, 1, def_mem_type, ref_stream);
              res &= IvyMemoryHelpers::transfer_memory(tmp_ptr, current, 1, def_mem_type, mem_type, ref_stream);
              if (i<n-1) tmp_ptr->next() = current+1;
              if (i>0) tmp_ptr->prev() = current-1;
              res &= IvyMemoryHelpers::transfer_memory(current, tmp_ptr, 1, mem_type, def_mem_type, ref_stream);
              res &= IvyMemoryHelpers::free_memory(tmp_ptr, 1, def_mem_type, ref_stream);
            }
            ++current;
          }
        )
      );
      return res;
    }
    __CUDA_HOST_DEVICE__ bool fix_prev_next(size_type pos){
      return fix_prev_next(chain, pos) && fix_prev_next(chain_const, pos);
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
          res &= std_mem::allocator<const_iterator_collection_t>::transfer_internal_memory(&chain_const, 1, def_mem_type, new_mem_type, ref_stream, release_old);
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

    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ const_pointable_t chain_const_rend() const{ return chain_const.get(); }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ const_pointable_t chain_const_front() const{
      if (!chain_const) return nullptr;
      size_type const n_size = chain_const.size();
      return std_mem::addressof(chain_const[1]);
    }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ const_pointable_t chain_const_back() const{
      if (!chain_const) return nullptr;
      size_type const n_size = chain_const.size();
      return std_mem::addressof(chain_const[n_size-2]);
    }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ const_pointable_t chain_const_end() const{
      if (!chain_const) return nullptr;
      size_type const n_size = chain_const.size();
      return std_mem::addressof(chain_const[n_size-1]);
    }

    template<typename Chain_t> __CUDA_HOST_DEVICE__ void invalidate(Chain_t& ch_obj){
      typedef typename Chain_t::pointer chain_pointer_t;
      if (!ch_obj) return;
      size_type const n = ch_obj.size();
      chain_pointer_t current = ch_obj.get();
      IvyMemoryType const def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      IvyMemoryType mem_type = ch_obj.get_memory_type();
      IvyGPUStream* stream = ch_obj.gpu_stream();
      for (size_type i=0; i<n; ++i){
        if (mem_type==def_mem_type){
          current->invalidate();
        }
        else{
          operate_with_GPU_stream_from_pointer(
            stream, ref_stream,
            __ENCAPSULATE__(
              chain_pointer_t tmp_ptr = nullptr;
              IvyMemoryHelpers::allocate_memory(tmp_ptr, 1, def_mem_type, ref_stream);
              IvyMemoryHelpers::transfer_memory(tmp_ptr, current, 1, def_mem_type, mem_type, ref_stream);
              tmp_ptr->invalidate();
              IvyMemoryHelpers::transfer_memory(current, tmp_ptr, 1, mem_type, def_mem_type, ref_stream);
              IvyMemoryHelpers::free_memory(tmp_ptr, 1, def_mem_type, ref_stream);
            )
          );
        }
        ++current;
      }
      ch_obj.reset();
    }
    __CUDA_HOST_DEVICE__ void invalidate(){
      invalidate(chain);
      invalidate(chain_const);
    }

    __CUDA_HOST_DEVICE__ iterator_type begin() const{ return get_iterator(chain_front()); }
    __CUDA_HOST_DEVICE__ iterator_type front() const{ return get_iterator(chain_front()); }
    __CUDA_HOST_DEVICE__ iterator_type back() const{ return get_iterator(chain_back()); }
    __CUDA_HOST_DEVICE__ iterator_type end() const{ return get_iterator(chain_end()); }
    __CUDA_HOST_DEVICE__ reverse_iterator_type rbegin() const{ return reverse_iterator_type(get_iterator(chain_back())); }
    __CUDA_HOST_DEVICE__ reverse_iterator_type rend() const{ return reverse_iterator_type(get_iterator(chain_rend())); }
    __CUDA_HOST_DEVICE__ const_iterator_type cbegin() const{ return get_iterator(chain_const_front()); }
    __CUDA_HOST_DEVICE__ const_iterator_type cfront() const{ return get_iterator(chain_const_front()); }
    __CUDA_HOST_DEVICE__ const_iterator_type cback() const{ return get_iterator(chain_const_back()); }
    __CUDA_HOST_DEVICE__ const_iterator_type cend() const{ return get_iterator(chain_const_end()); }
    __CUDA_HOST_DEVICE__ const_reverse_iterator_type crbegin() const{ return reverse_const_iterator_type(get_iterator(chain_const_back())); }
    __CUDA_HOST_DEVICE__ const_reverse_iterator_type crend() const{ return reverse_const_iterator_type(get_iterator(chain_const_rend())); }

    __CUDA_HOST_DEVICE__ pointable_t find_pointable(pointer ptr) const{
      pointable_t current = chain_front();
      while (current){
        if (current->get_mem_loc() == ptr) return current;
        current = current->next();
      }
      return nullptr;
    }
    __CUDA_HOST_DEVICE__ const_pointable_t find_const_pointable(pointer ptr) const{
      const_pointable_t current = chain_const_front();
      while (current){
        if (current->get_mem_loc() == ptr) return current;
        current = current->next();
      }
      return nullptr;
    }

    __CUDA_HOST_DEVICE__ void reset(bucket_element_type* bucket_head, size_type n_buckets, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
      this->invalidate();

      if (n_buckets==0) return;

      // First loop to determine the size and capacity of the data
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      size_type n_size = 0;
      if (def_mem_type == mem_type){
        bucket_element_type* b_el = bucket_head;
        for (size_type ib=0; ib<n_buckets; ++ib){
          auto const& data_uptr = b_el->second;
          n_size += data_uptr.size();
          ++b_el;
        }
      }
      else{
        bucket_element_type* b_el = bucket_head;
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            for (size_type ib=0; ib<n_buckets; ++ib){
              bucket_element_type* b_cur = nullptr;
              IvyMemoryHelpers::allocate_memory(b_cur, 1, def_mem_type, ref_stream);
              IvyMemoryHelpers::transfer_memory(b_cur, b_el, 1, def_mem_type, mem_type, ref_stream);
              auto const& data_uptr = b_cur->second;
              n_size += data_uptr.size();
              IvyMemoryHelpers::free_memory(b_cur, 1, def_mem_type, ref_stream);
              ++b_el;
            }
          )
        );
      }

      if (n_size>n_capacity) n_capacity = n_size+1;

      // Make the chains
      chain = std_mem::make_unique<iterator_type>(n_size+2, n_capacity+2, def_mem_type, stream);
      chain.get()->next() = chain.get()+1;
      (chain.get()+n_size+1)->prev() = (chain.get()+n_size);

      chain_const = std_mem::make_unique<const_iterator_type>(n_size+2, n_capacity+2, def_mem_type, stream);
      chain_const.get()->next() = chain_const.get()+1;
      (chain_const.get()+n_size+1)->prev() = (chain_const.get()+n_size);

      // Second loop to link the pointers to data elements
      size_type i_ch = 1;
      if (def_mem_type == mem_type){
        bucket_element_type* b_el = bucket_head;
        for (size_type ib=0; ib<n_buckets; ++ib){
          auto& data_uptr = b_el->second;
          auto ptr_data_const = data_uptr.get();
          auto ptr_data = __CONST_CAST__(std_ttraits::remove_const_t<std_ttraits::remove_reference_t<decltype(*ptr_data_const)>>*, ptr_data_const);
          for (size_type i=0; i<data_uptr.size(); ++i){
            chain[i_ch].set_mem_loc(ptr_data, mem_type, stream);
            chain_const[i_ch].set_mem_loc(ptr_data, mem_type, stream);
            ++i_ch;
            ++ptr_data;
          }
          ++b_el;
        }
      }
      else{
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            bucket_element_type* b_el = bucket_head;
            for (size_type ib=0; ib<n_buckets; ++ib){
              bucket_element_type* b_cur = nullptr;
              IvyMemoryHelpers::allocate_memory(b_cur, 1, def_mem_type, ref_stream);
              IvyMemoryHelpers::transfer_memory(b_cur, b_el, 1, def_mem_type, mem_type, ref_stream);
              auto& data_uptr = b_cur->second;
              auto ptr_data = data_uptr.get();
              for (size_type i=0; i<data_uptr.size(); ++i){
                chain[i_ch].set_mem_loc(ptr_data, mem_type, stream);
                chain_const[i_ch].set_mem_loc(ptr_data, mem_type, stream);
                ++i_ch;
                ++ptr_data;
              }
              IvyMemoryHelpers::free_memory(b_cur, 1, def_mem_type, ref_stream);
              ++b_el;
            }
            std_mem::allocator<iterator_collection_t>::transfer_internal_memory(&chain, 1, def_mem_type, mem_type, ref_stream, true);
            std_mem::allocator<const_iterator_collection_t>::transfer_internal_memory(&chain_const, 1, def_mem_type, mem_type, ref_stream, true);
            fix_prev_next(1);
          )
        );
      }
    }
    __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ void reset(){ this->invalidate(); }

    __CUDA_HOST_DEVICE__ void push_back(pointer ptr, IvyMemoryType mem_type, IvyGPUStream* stream){
      size_type n_size = chain.size();
      chain.insert(n_size-1, ptr, mem_type, stream);
      chain_const.insert(n_size-1, ptr, mem_type, stream);
      fix_prev_next(n_size-1);
    }
    __CUDA_HOST_DEVICE__ void pop_back(){
      if (!chain) return;
      size_type n_size = chain.size();
      if (n_size<3) return;
      chain[n_size-2].invalidate();
      chain.erase(n_size-2);
      chain_const[n_size-2].invalidate();
      chain_const.erase(n_size-2);
      fix_prev_next(n_size-2);
    }

    __CUDA_HOST_DEVICE__ IvyBucketedIteratorBuilder(){}
    __CUDA_HOST_DEVICE__ IvyBucketedIteratorBuilder(bucket_element_type* bucket_head, size_type n_buckets, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream){
      reset(bucket_head, n_buckets, n_capacity, mem_type, stream);
    }
    __CUDA_HOST_DEVICE__ IvyBucketedIteratorBuilder(IvyBucketedIteratorBuilder const& other) : chain(other.chain), chain_const(other.chain_const){}
    __CUDA_HOST_DEVICE__ IvyBucketedIteratorBuilder(IvyBucketedIteratorBuilder&& other) : chain(std_util::move(other.chain)), chain_const(std_util::move(other.chain_const)){}
    __CUDA_HOST_DEVICE__ ~IvyBucketedIteratorBuilder(){ invalidate(); }

    __CUDA_HOST_DEVICE__ IvyBucketedIteratorBuilder& operator=(IvyBucketedIteratorBuilder const& other){
      chain = other.chain;
      chain_const = other.chain_const;
      return *this;
    }
    __CUDA_HOST_DEVICE__ IvyBucketedIteratorBuilder& operator=(IvyBucketedIteratorBuilder&& other){
      chain = std_util::move(other.chain);
      chain_const = std_util::move(other.chain_const);
      return *this;
    }

    __CUDA_HOST_DEVICE__ size_type n_valid_iterators() const{ return (chain ? chain.size()-2 : 0); }
    __CUDA_HOST_DEVICE__ size_type n_capacity_valid_iterators() const{ return (chain ? chain.capacity()-2 : 0); }

    __CUDA_HOST_DEVICE__ void swap(IvyBucketedIteratorBuilder& other){
      std_util::swap(chain, other.chain);
      std_util::swap(chain_const, other.chain_const);
    }
  };
}
namespace std_util{
  template<typename Key, typename T, typename Hash> __CUDA_HOST_DEVICE__ void swap(std_ivy::IvyBucketedIteratorBuilder<Key, T, Hash>& a, std_ivy::IvyBucketedIteratorBuilder<Key, T, Hash>& b){ a.swap(b); }
}

#endif


#endif
