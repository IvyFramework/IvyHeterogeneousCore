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

  template<typename T> class IvyVectorIterator : public iterator<std_iter::contiguous_iterator_tag, T>{
  public:
    using Base_t = iterator<std_iter::contiguous_iterator_tag, T>;
    using value_type = Base_t::value_type;
    using pointer = Base_t::pointer;
    using reference = Base_t::reference;
    using difference_type = Base_t::difference_type;
    using iterator_category = Base_t::iterator_category;
    using pointable_t = std_mem::shared_ptr<IvyVectorIterator<T>>;
    using mem_loc_container_t = std_mem::shared_ptr<pointer>;

    friend struct IvyVectorIteratorBuilder<T>;
    friend struct transfer_memory_primitive<IvyVectorIterator<T>>;
    friend struct deallocator_primitive<IvyVectorIterator<T>>;

  protected:
    mem_loc_container_t ptr_mem_loc_;
    pointable_t next_;
    pointable_t prev_;

  public:
    __CUDA_HOST_DEVICE__ IvyVectorIterator(){}
    __CUDA_HOST_DEVICE__ IvyVectorIterator(IvyVectorIterator const& other) : ptr_mem_loc_(other.ptr_mem_loc_), next_(other.next_), prev_(other.prev_){}
    __CUDA_HOST_DEVICE__ IvyVectorIterator(IvyVectorIterator&& other) : ptr_mem_loc_(std_util::move(other.ptr_mem_loc_)), next_(std_util::move(other.next_)), prev_(std_util::move(other.prev_)){}
    __CUDA_HOST_DEVICE__ ~IvyVectorIterator(){}

  protected:
    __CUDA_HOST_DEVICE__ void set_mem_loc(pointer const& mem_loc){
      // Iterators should only operrate in the default memory space of the context (host, GPU etc.).
      // For that reason, we can safely set the stream to nullptr even if the pointed data is operated by a particular stream.
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      constexpr IvyGPUStream* def_stream = nullptr;
      ptr_mem_loc_ = std_mem::make_shared<pointer>(def_mem_type, def_stream, mem_loc);
    }
    __CUDA_HOST_DEVICE__ pointer& get_mem_loc(){ return *ptr_mem_loc_; }
    __CUDA_HOST_DEVICE__ pointer const& get_mem_loc() const{ return *ptr_mem_loc_; }
    __CUDA_HOST_DEVICE__ void set_next(pointable_t const& next){ next_ = next; }
    __CUDA_HOST_DEVICE__ void set_prev(pointable_t const& prev){ prev_ = prev; }
    __CUDA_HOST_DEVICE__ void set_next(std_cstddef::nullptr_t){ next_.reset(); }
    __CUDA_HOST_DEVICE__ void set_prev(std_cstddef::nullptr_t){ prev_.reset(); }

  public:
    __CUDA_HOST_DEVICE__ pointable_t& next(){ return next_; }
    __CUDA_HOST_DEVICE__ pointable_t& prev(){ return prev_; }
    __CUDA_HOST_DEVICE__ pointable_t const& next() const{ return next_; }
    __CUDA_HOST_DEVICE__ pointable_t const& prev() const{ return prev_; }

    __CUDA_HOST_DEVICE__ IvyVectorIterator& operator=(IvyVectorIterator const& other){ ptr_mem_loc_=other.ptr_mem_loc_; next_=other.next_; prev_=other.prev_; return *this; }
    __CUDA_HOST_DEVICE__ IvyVectorIterator& operator=(IvyVectorIterator&& other){ ptr_mem_loc_=std_util::move(other.ptr_mem_loc_); next_=std_util::move(other.next_); prev_=std_util::move(other.prev_); return *this; }

    __CUDA_HOST_DEVICE__ reference operator*() const{ return *(this->get_mem_loc()); }
    __CUDA_HOST_DEVICE__ pointer const& operator->() const{ return this->get_mem_loc(); }

    __CUDA_HOST_DEVICE__ bool is_valid() const{ return (ptr_mem_loc_ && this->get_mem_loc()); }

  protected:
    __CUDA_HOST_DEVICE__ void invalidate(){
      if (this->is_valid()) *ptr_mem_loc_ = nullptr; // This line invalidates all related iterators.
      ptr_mem_loc_.reset(); // This line decouples this iterator from previously related iterators.
      // The following lines decouple this iterator from those that point to the next and previous memory locations.
      next_.reset();
      prev_.reset();
    }

  public:
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T>& operator++(){ *this = *next_; return *this; }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T> operator++(int){ IvyVectorIterator<T> tmp(*this); operator++(); return tmp; }
    __CUDA_HOST_DEVICE__ IvyVectorIterator<T>& operator--(){ *this = *prev_; return *this; }
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

    __CUDA_HOST_DEVICE__ bool operator==(IvyVectorIterator<T> const& other) const{ return (ptr_mem_loc_ == other.ptr_mem_loc_) && (!ptr_mem_loc_ || this->get_mem_loc()==other.get_mem_loc()); }
    __CUDA_HOST_DEVICE__ bool operator!=(IvyVectorIterator<T> const& other) const{ return !(*this==other); }

    difference_type operator-(IvyVectorIterator<T> const& other) const{
      if (other==*this || (!other.is_valid() && !this->is_valid())) return 0;
      if (other.is_valid() && !this->is_valid()) return 1; // e.g., end - begin
      if (!other.is_valid() && this->is_valid()) return -1; // e.g., begin - end
      difference_type n = 0;
      auto current = other;
      while (current){
        if (current == *this) return n;
        ++n;
        current = current.next();
      }
      current = other.prev();
      n = -1;
      while (current){
        if (current == *this) return n;
        --n;
        current = current.prev();
      }
      return 0;
    }
  };
  template<typename T> using IvyVectorConstIterator = IvyVectorIterator<T const>;

  template<typename T> struct IvyVectorIteratorBuilder{
    using iterator_type = IvyVectorIterator<T>;
    using reverse_iterator_type = std_iter::reverse_iterator<iterator_type>;
    using pointable_t = std_mem::shared_ptr<iterator_type>;
    using value_type = std_ttraits::remove_cv_t<T>;
    using data_type = std_ivy::unique_ptr<value_type>;
    using pointer = typename data_type::pointer;
    using size_type = typename data_type::size_type;

    pointable_t chain_rend;
    pointable_t chain_front;
    pointable_t chain_back;
    pointable_t chain_end;

    __CUDA_HOST_DEVICE__ void invalidate(){
      auto current = chain_rend;
      while (current){
        auto next = current->next();
        current->invalidate();
        current = next;
      }
    }

    __CUDA_HOST_DEVICE__ iterator_type begin() const{ return *chain_front; }
    __CUDA_HOST_DEVICE__ iterator_type front() const{ return *chain_front; }
    __CUDA_HOST_DEVICE__ iterator_type back() const{ return *chain_back; }
    __CUDA_HOST_DEVICE__ iterator_type end() const{ return *chain_end; }
    __CUDA_HOST_DEVICE__ reverse_iterator_type rbegin() const{ return reverse_iterator_type(*chain_back); }
    __CUDA_HOST_DEVICE__ reverse_iterator_type rend() const{ return reverse_iterator_type(*chain_rend); }

    static __CUDA_HOST_DEVICE__ pointable_t make_pointable(pointer& mem_loc, IvyMemoryType mem_type, IvyGPUStream* stream){
      auto res = make_shared<iterator_type>(mem_type, stream);
      res->set_mem_loc(mem_loc);
      return res;
    }

    __CUDA_HOST_DEVICE__ pointable_t find_pointable(pointer const& mem_loc) const{
      auto res = chain_front;
      while (res){
        if (res->get_mem_loc() == mem_loc) break;
        res = res->next();
      }
      return res;
    }

    __CUDA_HOST_DEVICE__ void reset_chain_rend(){
      if (chain_front && chain_rend && chain_front->is_valid()){
        chain_rend->set_next(chain_front);
        chain_front->set_prev(chain_rend);
      }
    }

    __CUDA_HOST_DEVICE__ void reset_chain_end(){
      if (chain_back && chain_end && chain_back->is_valid()){
        chain_back->set_next(chain_end);
        chain_end->set_prev(chain_back);
      }
    }

    __CUDA_HOST_DEVICE__ void reset(pointer ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
      this->invalidate();

      {
        pointable_t current, prev;
        for (IvyTypes::size_t i=0; i<n; ++i){
          if (i>0) prev = current;
          current = make_pointable(ptr, mem_type, stream);
          if (i==0) chain_front = current;
          if (i==n-1) chain_back = current;
          if (prev){
            prev->set_next(current);
            current->set_prev(prev);
          }
          ++ptr;
        }
      }

      if (!chain_front){
        chain_front = make_shared<IvyVectorIterator<T>>(mem_type, stream);
        chain_back = make_shared<IvyVectorIterator<T>>(mem_type, stream);
      }

      if (!chain_rend) chain_rend = make_shared<IvyVectorIterator<T>>(mem_type, stream);
      reset_chain_rend();

      if (!chain_end) chain_end = make_shared<IvyVectorIterator<T>>(mem_type, stream);
      reset_chain_end();
    }

    __CUDA_HOST_DEVICE__ void push_back(pointable_t const& it){
      if (!it || !it->is_valid()) return;
      auto const& prev = chain_back;
      it->set_prev(prev);
      if (prev) prev->set_next(it);
      chain_back = it;
    }

    // Insert iterator 'it' before position 'pos'.
    __CUDA_HOST_DEVICE__ void insert(pointable_t const& pos, pointable_t const& it){
      if (!pos || !it || !it->is_valid()) return;
      if (!pos->is_valid()){
        if (pos == chain_end) push_back(it);
        return;
      }
      if (*pos == *chain_front){
        chain_front = it;
        reset_chain_rend();
      }
      auto const& prev = pos->prev();
      auto next = pos;
      it->set_prev(prev);
      it->set_next(next);
      if (prev) prev->set_next(it);
      if (next){
        next->set_prev(it);
        while (next){
          if (next->is_valid()){
            auto& mem_loc = next->get_mem_loc();
            ++mem_loc;
          }
          next = next->next();
        }
      }
    }

    __CUDA_HOST_DEVICE__ void erase(pointable_t const& pos){
      if (!pos || !pos->is_valid()) return;
      auto const& prev = pos->prev();
      auto& next = pos->next();
      if (prev) prev->set_next(next);
      if (next) next->set_prev(prev);
      if (*pos == *chain_front){
        chain_front = next;
        reset_chain_rend();
      }
      if (*pos == *chain_back){
        chain_back = prev;
        reset_chain_end();
      }
      while (next){
        if (next->is_valid()){
          auto& mem_loc = next->get_mem_loc();
          --mem_loc;
        }
        next = next->next();
      }
      pos->invalidate();
    }

    __CUDA_HOST_DEVICE__ void pop_back(){
      if (!chain_back->is_valid()) return;
      auto const& prev = chain_back->prev();
      auto const& next = chain_back->next();
      if (prev) prev->set_next(next);
      if (next) next->set_prev(prev);
      chain_back->invalidate(); // chain_back may be shared, so invalidate living copies.
      if (prev && prev->is_valid()) chain_back = prev;
      reset_chain_end();
    }

    IvyVectorIteratorBuilder() = default;
    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(IvyVectorIteratorBuilder const& other) : chain_rend(other.chain_rend), chain_front(other.chain_front), chain_back(other.chain_back), chain_end(other.chain_end){}
    __CUDA_HOST_DEVICE__ IvyVectorIteratorBuilder(IvyVectorIteratorBuilder&& other) : chain_rend(std_util::move(other.chain_rend)), chain_front(std_util::move(other.chain_front)), chain_back(std_util::move(other.chain_back)), chain_end(std_util::move(other.chain_end)){}
    __CUDA_HOST_DEVICE__ ~IvyVectorIteratorBuilder(){ this->invalidate(); }
  };

}

#endif


#endif
