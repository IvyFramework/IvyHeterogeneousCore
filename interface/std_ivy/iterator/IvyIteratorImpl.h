/**
 * @file IvyIteratorImpl.h
 * @brief Core iterator operations and adapter implementations.
 */
#ifndef IVYITERATORIMPL_H
#define IVYITERATORIMPL_H


#include "IvyBasicTypes.h"
#include "std_ivy/iterator/IvyIteratorTraits.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyMemory.h"


namespace std_ivy{
  // How to access the data of a container
  DEFINE_HAS_CALL(data);
  DEFINE_HAS_CALL(begin);
  /** @brief Return the data head using `data()` when available. */
  template<typename T, std_ttraits::enable_if_t<has_call_data_v<T>, bool> = true> __HOST_DEVICE__
    auto get_data_head(T& t){ return t.data(); }
  /** @brief Return the data head using `begin()` when `data()` is unavailable. */
  template<typename T, std_ttraits::enable_if_t<!has_call_data_v<T>&& has_call_begin_v<T>, bool> = true> __HOST_DEVICE__
    auto get_data_head(T& t){ return t.begin(); }
  /** @brief Return the object itself when neither `data()` nor `begin()` is available. */
  template<typename T, std_ttraits::enable_if_t<!has_call_data_v<T> && !has_call_begin_v<T>, bool> = true> __HOST_DEVICE__
    auto get_data_head(T& t){ return t; }

  // Iterators
  template<
    typename T,
    typename Distance = IvyTypes::ptrdiff_t,
    typename Pointer = T*,
    typename Reference = T&
  > class IvyInputIterator : public iterator<input_iterator_tag, T, Distance, Pointer, Reference>{
  public:
    /** @brief Iterator base type. */
    using iterator_base = iterator<input_iterator_tag, T, Distance, Pointer, Reference>;
    /** @brief Value type alias. */
    using value_type = typename iterator_base::value_type;
    /** @brief Pointer type alias. */
    using pointer = typename iterator_base::pointer;
    /** @brief Reference type alias. */
    using reference = typename iterator_base::reference;
    /** @brief Difference type alias. */
    using difference_type = typename iterator_base::difference_type;
    /** @brief Iterator category tag. */
    using iterator_category = typename iterator_base::iterator_category;

  protected:
    pointer ptr_;

    /** @brief Reset internal pointer to null. */
    __HOST_DEVICE__ void dump() __NOEXCEPT__{ ptr_ = nullptr; }
    /** @brief Return the wrapped raw pointer. */
    __HOST_DEVICE__ pointer get() const __NOEXCEPT__{ return ptr_; }

  public:
    /** @brief Default constructor. */
    __HOST_DEVICE__ IvyInputIterator() __NOEXCEPT__ : ptr_(nullptr){}
    /** @brief Construct from raw pointer. */
    __HOST_DEVICE__ explicit IvyInputIterator(pointer const& ptr) __NOEXCEPT__ : ptr_(ptr){}
    /** @brief Copy constructor. */
    __HOST_DEVICE__ IvyInputIterator(IvyInputIterator const& it) : ptr_(it.ptr_){}
    /** @brief Move constructor. */
    __HOST_DEVICE__ IvyInputIterator(IvyInputIterator&& it) : ptr_(std_util::move(it.ptr_)){ it.dump(); }
    /** @brief Virtual destructor. */
    __HOST_DEVICE__ virtual ~IvyInputIterator(){}

    /** @brief Copy assignment. */
    __HOST_DEVICE__ IvyInputIterator& operator=(IvyInputIterator const& it){ ptr_ = it.ptr_; return *this; }
    /** @brief Move assignment. */
    __HOST_DEVICE__ IvyInputIterator& operator=(IvyInputIterator&& it){ ptr_ = std_util::move(it.ptr_); it.dump(); return *this; }

    __HOST_DEVICE__ reference operator*() const{ return *ptr_; }
    __HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __HOST_DEVICE__ IvyInputIterator& operator++(){ ++ptr_; return *this; }
    __HOST_DEVICE__ IvyInputIterator operator++(int){ IvyInputIterator it_mp(*this); ++ptr_; return it_mp; }
    __HOST_DEVICE__ IvyInputIterator operator+(difference_type const& n) const{ return IvyInputIterator(ptr_ + n); }
    __HOST_DEVICE__ IvyInputIterator& operator+=(difference_type const& n){ ptr_ += n; return *this; }
    __HOST_DEVICE__ IvyInputIterator operator-(difference_type const& n) const{ return IvyInputIterator(ptr_ - n); }
    __HOST_DEVICE__ IvyInputIterator& operator-=(difference_type const& n){ ptr_ -= n; return *this; }

    __HOST_DEVICE__ void swap(IvyInputIterator& it){ std_util::swap(ptr_, it.ptr_); }
  };
  template<typename T, typename D, typename P, typename R>
  /** @brief Equality comparison between input iterators. */
  __HOST_DEVICE__ bool operator==(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return &(*x) == &(*x); }
  template<typename T, typename D, typename P, typename R>
  /** @brief Inequality comparison between input iterators. */
  __HOST_DEVICE__ bool operator!=(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return !(x==y); }
  template<typename T, typename D, typename P, typename R>
  /** @brief Strict-weak ordering comparison between input iterators. */
  __HOST_DEVICE__ bool operator<(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return &(*x) < &(*y); }
  template<typename T, typename D, typename P, typename R>
  /** @brief Greater-or-equal comparison between input iterators. */
  __HOST_DEVICE__ bool operator>=(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return !(x<y); }
  template<typename T, typename D, typename P, typename R>
  /** @brief Greater-than comparison between input iterators. */
  __HOST_DEVICE__ bool operator>(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return y<x; }
  template<typename T, typename D, typename P, typename R>
  /** @brief Less-or-equal comparison between input iterators. */
  __HOST_DEVICE__ bool operator<=(IvyInputIterator<T, D, P, R> const& x, IvyInputIterator<T, D, P, R> const& y){ return !(y<x); }
}
namespace std_util{
  template<typename T, typename D, typename P, typename R>
  /** @brief Swap overload for IvyInputIterator. */
  __HOST_DEVICE__ void swap(std_ivy::IvyInputIterator<T, D, P, R>& x, std_ivy::IvyInputIterator<T, D, P, R>& y){ return x.swap(y); }
}
namespace std_ivy{
  template<
    typename T,
    typename Distance = IvyTypes::ptrdiff_t,
    typename Pointer = T*,
    typename Reference = T&
  > class IvyOutputIterator : public iterator<output_iterator_tag, T, Distance, Pointer, Reference>{
  public:
    /** @brief Iterator base type. */
    using iterator_base = iterator<output_iterator_tag, T, Distance, Pointer, Reference>;
    /** @brief Value type alias. */
    using value_type = typename iterator_base::value_type;
    /** @brief Pointer type alias. */
    using pointer = typename iterator_base::pointer;
    /** @brief Reference type alias. */
    using reference = typename iterator_base::reference;
    /** @brief Difference type alias. */
    using difference_type = typename iterator_base::difference_type;
    /** @brief Iterator category tag. */
    using iterator_category = typename iterator_base::iterator_category;

  protected:
    pointer ptr_;

    /** @brief Reset internal pointer to null. */
    __HOST_DEVICE__ void dump() __NOEXCEPT__{ ptr_ = nullptr; }
    /** @brief Return the wrapped raw pointer. */
    __HOST_DEVICE__ pointer get() const __NOEXCEPT__{ return ptr_; }

  public:
    /** @brief Default constructor. */
    __HOST_DEVICE__ IvyOutputIterator() __NOEXCEPT__ : ptr_(nullptr){}
    /** @brief Construct from raw pointer. */
    __HOST_DEVICE__ explicit IvyOutputIterator(pointer const& ptr) __NOEXCEPT__ : ptr_(ptr){}
    /** @brief Copy constructor. */
    __HOST_DEVICE__ IvyOutputIterator(IvyOutputIterator const& it) : ptr_(it.ptr_){}
    /** @brief Move constructor. */
    __HOST_DEVICE__ IvyOutputIterator(IvyOutputIterator&& it) : ptr_(std_util::move(it.ptr_)){ it.dump(); }
    /** @brief Virtual destructor. */
    __HOST_DEVICE__ virtual ~IvyOutputIterator(){}

    __HOST_DEVICE__ IvyOutputIterator& operator=(IvyOutputIterator const& it){ ptr_ = it.ptr_; return *this; }
    __HOST_DEVICE__ IvyOutputIterator& operator=(IvyOutputIterator&& it){ ptr_ = std_util::move(it.ptr_); it.dump(); return *this; }
    template<typename U> __HOST_DEVICE__ IvyOutputIterator& operator=(U val){ *ptr_ = val; return *this; }

    __HOST_DEVICE__ reference operator*() const{ return *ptr_; }
    __HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __HOST_DEVICE__ IvyOutputIterator& operator++(){ ++ptr_; return *this; }
    __HOST_DEVICE__ IvyOutputIterator operator++(int){ IvyOutputIterator it_mp(*this); ++ptr_; return it_mp; }
    __HOST_DEVICE__ IvyOutputIterator operator+(difference_type const& n) const{ return IvyOutputIterator(ptr_ + n); }
    __HOST_DEVICE__ IvyOutputIterator& operator+=(difference_type const& n){ ptr_ += n; return *this; }
    __HOST_DEVICE__ IvyOutputIterator operator-(difference_type const& n) const{ return IvyOutputIterator(ptr_ - n); }
    __HOST_DEVICE__ IvyOutputIterator& operator-=(difference_type const& n){ ptr_ -= n; return *this; }

    __HOST_DEVICE__ void swap(IvyOutputIterator& it){ std_util::swap(ptr_, it.ptr_); }
  };
}
namespace std_util{
  template<typename T, typename D, typename P, typename R>
  /** @brief Swap overload for IvyOutputIterator. */
  __HOST_DEVICE__ void swap(std_ivy::IvyOutputIterator<T, D, P, R>& x, std_ivy::IvyOutputIterator<T, D, P, R>& y){ return x.swap(y); }
}
namespace std_ivy{
  /** @brief Compute distance between two iterators. */
  template<typename It> __HOST_DEVICE__ constexpr typename std_ivy::iterator_traits<It>::difference_type distance(It const& first, It const& last){
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
  /** @brief Return begin iterator for mutable container. */
  template<typename T> __HOST_DEVICE__ constexpr auto begin(T& c) -> decltype(c.begin()){ return c.begin(); }
  /** @brief Return begin iterator for const container. */
  template<typename T> __HOST_DEVICE__ constexpr auto begin(T const& c) -> decltype(c.begin()){ return c.begin(); }
  /** @brief Return begin pointer for native array. */
  template<typename T, size_t N> __HOST_DEVICE__ constexpr T* begin(T(&a)[N]) __NOEXCEPT__{ return std_mem::addressof(a[0]); }
  /** @brief Return const begin iterator. */
  template<typename T> __HOST_DEVICE__ constexpr auto cbegin(T const& c) -> decltype(begin(c)){ return begin(c); }

  // end functions
  /** @brief Return end iterator for mutable container. */
  template<typename T> __HOST_DEVICE__ constexpr auto end(T& c) -> decltype(c.end()){ return c.end(); }
  /** @brief Return end iterator for const container. */
  template<typename T> __HOST_DEVICE__ constexpr auto end(T const& c) -> decltype(c.end()){ return c.end(); }
  /** @brief Return end pointer for native array. */
  template<typename T, size_t N> __HOST_DEVICE__ constexpr T* end(T(&a)[N]) __NOEXCEPT__{ return (a+N); }
  /** @brief Return const end iterator. */
  template<typename T> __HOST_DEVICE__ constexpr auto cend(T const& c) -> decltype(end(c)){ return end(c); }

}


#endif
