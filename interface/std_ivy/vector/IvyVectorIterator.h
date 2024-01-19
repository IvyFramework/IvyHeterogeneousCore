#ifndef IVYVECTORITERATOR_H
#define IVYVECTORITERATOR_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyIterator.h"


#ifdef __USE_CUDA__

namespace std_ivy{
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

    friend struct IvyVectorIteratorBuilder<T>;

  protected:
    pointer* mem_loc_;
    pointable_t next_;
    pointable_t prev_;

  public:
    __CUDA_HOST_DEVICE__ IvyVectorIterator() : mem_loc_(nullptr){}
    __CUDA_HOST_DEVICE__ IvyVectorIterator(IvyVectorIterator const& other) : mem_loc_(other.mem_loc_), next_(other.next_), prev_(other.prev_){}
    __CUDA_HOST_DEVICE__ IvyVectorIterator(IvyVectorIterator&& other) : mem_loc_(std_util::move(other.mem_loc_)), next_(std_util::move(other.next_)), prev_(std_util::move(other.prev_)){}
    __CUDA_HOST_DEVICE__ ~IvyVectorIterator(){}

    __CUDA_HOST_DEVICE__ void set_mem_loc(pointer& ptr){ mem_loc_ = &ptr; }
    __CUDA_HOST_DEVICE__ pointer*& get_mem_loc(){ return mem_loc_; }
    __CUDA_HOST_DEVICE__ pointer* get_mem_loc() const{ return mem_loc_; }
    __CUDA_HOST_DEVICE__ void set_next(pointable_t const& next){ next_ = next; }
    __CUDA_HOST_DEVICE__ void set_prev(pointable_t const& prev){ prev_ = prev; }
    __CUDA_HOST_DEVICE__ void set_next(std_cstddef::nullptr_t){ next_.reset(); }
    __CUDA_HOST_DEVICE__ void set_prev(std_cstddef::nullptr_t){ prev_.reset(); }
    __CUDA_HOST_DEVICE__ pointable_t const& next() const{ return next_; }
    __CUDA_HOST_DEVICE__ pointable_t const& prev() const{ return prev_; }

    __CUDA_HOST_DEVICE__ IvyVectorIterator& operator=(IvyVectorIterator const& other){ mem_loc_=other.mem_loc_; next_=other.next_; prev_=other.prev_; return *this; }
    __CUDA_HOST_DEVICE__ IvyVectorIterator& operator=(IvyVectorIterator&& other){ mem_loc_=std_util::move(other.mem_loc_); next_=std_util::move(other.next_); prev_=std_util::move(other.prev_); return *this; }

    __CUDA_HOST_DEVICE__ reference operator*() const{ return **mem_loc_; }
    __CUDA_HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __CUDA_HOST_DEVICE__ bool is_valid() const{ return (mem_loc_ && *mem_loc_); }
    __CUDA_HOST_DEVICE__ void invalidate(){ mem_loc_=nullptr; next_.reset(); prev_.reset(); }

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

    __CUDA_HOST_DEVICE__ bool operator==(IvyVectorIterator<T> const& other) const{ return (!mem_loc_ && !other.mem_loc_) || (mem_loc_ && other.mem_loc_ && *mem_loc_==*(other.mem_loc_)); }
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
  template<typename T> using VectorConstIterator = IvyVectorIterator<T const>;

  template<typename T> struct IvyVectorIteratorBuilder{
    using iterator_type = IvyVectorIterator<T>;
    using reverse_iterator_type = std_iter::reverse_iterator<iterator_type>;
    using pointable_t = std_mem::shared_ptr<iterator_type>;
    using value_type = std_ttraits::remove_cv_t<T>;
    using data_type = std_ivy::unique_ptr<value_type>;
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

    static __CUDA_HOST_DEVICE__ pointable_t make_pointable(typename data_type::pointer& mem_loc, IvyMemoryType mem_type, IvyGPUStream* stream){
      auto res = make_shared<iterator_type>(mem_type, stream);
      res->set_mem_loc(mem_loc);
      return res;
    }

    __CUDA_HOST_DEVICE__ void reset(data_type const& data, size_type n){
      this->invalidate();

      auto ptr = data.get();
      auto stream = data.gpu_stream();
      auto mem_type = data.get_memory_type();
      pointable_t current, prev;

      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
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
      )
      );

      if (!chain_front){
        chain_front = make_shared<IvyVectorIterator<T>>(mem_type, stream);
        chain_back = make_shared<IvyVectorIterator<T>>(mem_type, stream);
      }
      if (!chain_rend){
        chain_rend = make_shared<IvyVectorIterator<T>>(mem_type, stream);
        chain_end = make_shared<IvyVectorIterator<T>>(mem_type, stream);
      }

      if (chain_front->is_valid()){
        chain_rend->set_next(chain_front);
        chain_front->set_prev(chain_rend);
        chain_back->set_next(chain_end);
        chain_end->set_prev(chain_back);
      }
    }

    // Insert iterator 'it' before position 'pos'.
    __CUDA_HOST_DEVICE__ void insert(pointable_t const& pos, pointable_t const& it){
      if (!pos || !it || !pos->is_valid() || !it->is_valid()) return;
      if (*pos == *chain_front) chain_front = it;
      auto const& prev = pos->prev();
      auto next = pos;
      it->set_prev(prev);
      it->set_next(next);
      if (prev) prev->set_next(it);
      if (next){
        next->set_prev(it);
        while (next){
          auto mem_loc = next->get_mem_loc();
          *mem_loc = *mem_loc + 1;
          next = next->next();
        }
      }
    }

    __CUDA_HOST_DEVICE__ void push_back(pointable_t const& it){
      if (!it || !it->is_valid()) return;
      auto const& prev = chain_back;
      it->set_prev(prev);
      if (prev) prev->set_next(it);
      chain_back = it;
    }

    __CUDA_HOST_DEVICE__ void erase(pointable_t const& pos){
      if (!pos || !pos->is_valid()) return;
      auto const& prev = pos->prev();
      auto const& next = pos->next();
      if (prev) prev->set_next(next);
      if (next) next->set_prev(prev);
      if (*pos == *chain_front) chain_front = next;
      if (*pos == *chain_back) chain_back = prev;
      while (next){
        auto mem_loc = next->get_mem_loc();
        *mem_loc = *mem_loc - 1;
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
    }

    IvyVectorIteratorBuilder(){}
    IvyVectorIteratorBuilder(IvyVectorIteratorBuilder const& other) : chain_front(other.chain_front), chain_back(other.chain_back), chain_end(other.chain_end){}
    IvyVectorIteratorBuilder(IvyVectorIteratorBuilder&& other) : chain_front(std_util::move(other.chain_front)), chain_back(std_util::move(other.chain_back)), chain_end(std_util::move(other.chain_end)){}
    ~IvyVectorIteratorBuilder(){ this->invalidate(); }
  };
}

#endif


#endif
