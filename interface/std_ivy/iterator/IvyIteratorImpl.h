#ifndef IVYITERATORIMPL_H
#define IVYITERATORIMPL_H


#include "IvyBasicTypes.h"
#include "std_ivy/iterator/IvyIteratorTraits.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyMemory.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  // How to access the data of a container
  DEFINE_HAS_CALL(data);
  DEFINE_HAS_CALL(begin);
  template<typename T, std_ttraits::enable_if_t<has_call_data_v<T>, bool> = true> __CUDA_HOST_DEVICE__
  auto get_data_head(T& t){ return t.data(); }
  template<typename T, std_ttraits::enable_if_t<!has_call_data_v<T> && has_call_begin_v<T>, bool> = true> __CUDA_HOST_DEVICE__
  auto get_data_head(T& t){ return t.begin(); }
  template<typename T, std_ttraits::enable_if_t<!has_call_data_v<T> && !has_call_begin_v<T>, bool> = true> __CUDA_HOST_DEVICE__
  auto get_data_head(T& t){ return t; }

  // Iterators
  template<
    typename T,
    typename Distance = IvyTypes::ptrdiff_t,
    typename Pointer = T*,
    typename Reference = T&
  > class IvyInputIterator : public iterator<input_iterator_tag, T, Distance, Pointer, Reference>{
  public:
    using iterator_base = iterator<input_iterator_tag, T, Distance, Pointer, Reference>;
    using value_type = typename iterator_base::value_type;
    using pointer = typename iterator_base::pointer;
    using reference = typename iterator_base::reference;
    using difference_type = typename iterator_base::difference_type;
    using iterator_category = typename iterator_base::iterator_category;

  protected:
    pointer ptr_;
    
    __CUDA_HOST_DEVICE__ void dump() __NOEXCEPT__{ ptr_ = nullptr; }
    __CUDA_HOST_DEVICE__ pointer get() const __NOEXCEPT__{ return ptr_; }

  public:
    __CUDA_HOST_DEVICE__ IvyInputIterator() __NOEXCEPT__ : ptr_(nullptr){}
    __CUDA_HOST_DEVICE__ explicit IvyInputIterator(pointer const& ptr) __NOEXCEPT__ : ptr_(ptr){}
    __CUDA_HOST_DEVICE__ IvyInputIterator(IvyInputIterator const& it) : ptr_(it.ptr_){}
    __CUDA_HOST_DEVICE__ IvyInputIterator(IvyInputIterator&& it) : ptr_(std_util::move(it.ptr_)){ it.dump(); }
    __CUDA_HOST_DEVICE__ virtual ~IvyInputIterator(){}

    __CUDA_HOST_DEVICE__ IvyInputIterator& operator=(IvyInputIterator const& it){ ptr_ = it.ptr_; return *this; }
    __CUDA_HOST_DEVICE__ IvyInputIterator& operator=(IvyInputIterator&& it){ ptr_ = std_util::move(it.ptr_); it.dump(); return *this; }

    __CUDA_HOST_DEVICE__ reference operator*() const{ return *ptr_; }
    __CUDA_HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __CUDA_HOST_DEVICE__ IvyInputIterator& operator++(){ ++ptr_; return *this; }
    __CUDA_HOST_DEVICE__ IvyInputIterator operator++(int){ IvyInputIterator it_mp(*this); ++ptr_; return it_mp; }
    __CUDA_HOST_DEVICE__ IvyInputIterator operator+(difference_type const& n) const{ return IvyInputIterator(ptr_ + n); }
    __CUDA_HOST_DEVICE__ IvyInputIterator& operator+=(difference_type const& n){ ptr_ += n; return *this; }
    __CUDA_HOST_DEVICE__ IvyInputIterator operator-(difference_type const& n) const{ return IvyInputIterator(ptr_ - n); }
    __CUDA_HOST_DEVICE__ IvyInputIterator& operator-=(difference_type const& n){ ptr_ -= n; return *this; }

    __CUDA_HOST_DEVICE__ void swap(IvyInputIterator& it){ std_util::swap(ptr_, it.ptr_); }
  };
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ bool operator==(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return &(*x) == &(*x); }
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ bool operator!=(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return !(x==y); }
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ bool operator<(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return &(*x) < &(*y); }
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ bool operator>=(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return !(x<y); }
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ bool operator>(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return y<x; }
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ bool operator<=(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return !(y<x); }
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ void swap(IvyInputIterator<T, D, P, R>& x, IvyInputIterator<T, D, P, R>& y){ return x.swap(y); }

  template<
    typename T,
    typename Distance = IvyTypes::ptrdiff_t,
    typename Pointer = T*,
    typename Reference = T&
  > class IvyOutputIterator : public iterator<output_iterator_tag, T, Distance, Pointer, Reference>{
  public:
    using iterator_base = iterator<output_iterator_tag, T, Distance, Pointer, Reference>;
    using value_type = typename iterator_base::value_type;
    using pointer = typename iterator_base::pointer;
    using reference = typename iterator_base::reference;
    using difference_type = typename iterator_base::difference_type;
    using iterator_category = typename iterator_base::iterator_category;

  protected:
    pointer ptr_;

    __CUDA_HOST_DEVICE__ void dump() __NOEXCEPT__{ ptr_ = nullptr; }
    __CUDA_HOST_DEVICE__ pointer get() const __NOEXCEPT__{ return ptr_; }

  public:
    __CUDA_HOST_DEVICE__ IvyOutputIterator() __NOEXCEPT__ : ptr_(nullptr){}
    __CUDA_HOST_DEVICE__ explicit IvyOutputIterator(pointer const& ptr) __NOEXCEPT__ : ptr_(ptr){}
    __CUDA_HOST_DEVICE__ IvyOutputIterator(IvyOutputIterator const& it) : ptr_(it.ptr_){}
    __CUDA_HOST_DEVICE__ IvyOutputIterator(IvyOutputIterator&& it) : ptr_(std_util::move(it.ptr_)){ it.dump(); }
    __CUDA_HOST_DEVICE__ virtual ~IvyOutputIterator(){}

    __CUDA_HOST_DEVICE__ IvyOutputIterator& operator=(IvyOutputIterator const& it){ ptr_ = it.ptr_; return *this; }
    __CUDA_HOST_DEVICE__ IvyOutputIterator& operator=(IvyOutputIterator&& it){ ptr_ = std_util::move(it.ptr_); it.dump(); return *this; }
    template<typename U> __CUDA_HOST_DEVICE__ IvyOutputIterator& operator=(U val){ *ptr_ = val; return *this; }

    __CUDA_HOST_DEVICE__ reference operator*() const{ return *ptr_; }
    __CUDA_HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __CUDA_HOST_DEVICE__ IvyOutputIterator& operator++(){ ++ptr_; return *this; }
    __CUDA_HOST_DEVICE__ IvyOutputIterator operator++(int){ IvyOutputIterator it_mp(*this); ++ptr_; return it_mp; }
    __CUDA_HOST_DEVICE__ IvyOutputIterator operator+(difference_type const& n) const{ return IvyOutputIterator(ptr_ + n); }
    __CUDA_HOST_DEVICE__ IvyOutputIterator& operator+=(difference_type const& n){ ptr_ += n; return *this; }
    __CUDA_HOST_DEVICE__ IvyOutputIterator operator-(difference_type const& n) const{ return IvyOutputIterator(ptr_ - n); }
    __CUDA_HOST_DEVICE__ IvyOutputIterator& operator-=(difference_type const& n){ ptr_ -= n; return *this; }

    __CUDA_HOST_DEVICE__ void swap(IvyOutputIterator& it){ std_util::swap(ptr_, it.ptr_); }
  };
  template<typename T, typename D, typename P, typename R>
  __CUDA_HOST_DEVICE__ void swap(IvyOutputIterator<T, D, P, R>& x, IvyOutputIterator<T, D, P, R>& y){ return x.swap(y); }


  template<typename It> __CUDA_HOST_DEVICE__ constexpr typename std_ivy::iterator_traits<It>::difference_type distance(It const& first, It const& last){
    using category = typename std_ivy::iterator_traits<It>::iterator_category;
    static_assert(std_ttraits::is_base_of_v<std_ivy::input_iterator_tag, category>);
    if constexpr (std_ttraits::is_base_of_v<std_ivy::random_access_iterator_tag, category>) return last - first;
    else{
      typename std_ivy::iterator_traits<It>::difference_type result = 0;
      while (first != last){
        ++first;
        ++result;
      }
      return result;
    }
  }

  // begin functions
  template<typename T> __CUDA_HOST_DEVICE__ constexpr auto begin(T& c) -> decltype(c.begin()){ return c.begin(); }
  template<typename T> __CUDA_HOST_DEVICE__ constexpr auto begin(T const& c) -> decltype(c.begin()){ return c.begin(); }
  template<typename T, size_t N> __CUDA_HOST_DEVICE__ constexpr T* begin(T(&a)[N]) __NOEXCEPT__{ return std_mem::addressof(a[0]); }
  template<typename T> __CUDA_HOST_DEVICE__ constexpr auto cbegin(T const& c) __NOEXCEPT__ -> decltype(begin(c)){ return begin(c); }

  // end functions
  template<typename T> __CUDA_HOST_DEVICE__ constexpr auto end(T& c) -> decltype(c.end()){ return c.end(); }
  template<typename T> __CUDA_HOST_DEVICE__ constexpr auto end(T const& c) -> decltype(c.end()){ return c.end(); }
  template<typename T, size_t N> __CUDA_HOST_DEVICE__ constexpr T* end(T(&a)[N]) __NOEXCEPT__{ return (a+N); }
  template<typename T> __CUDA_HOST_DEVICE__ constexpr auto cend(T const& c) __NOEXCEPT__ -> decltype(end(c)){ return end(c); }

}

#endif


#endif
