#ifndef IVYITERATORIMPL_H
#define IVYITERATORIMPL_H


#include "IvyMemoryHelpers.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyInitializerList.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  // Iterator primitives
  struct input_iterator_tag{};
  struct output_iterator_tag{};
  struct forward_iterator_tag : public input_iterator_tag{};
  struct bidirectional_iterator_tag : public forward_iterator_tag{};
  struct random_access_iterator_tag : public bidirectional_iterator_tag{};
  struct contiguous_iterator_tag : public random_access_iterator_tag{};
  using stashing_iterator_tag = void; // Dummy tag to recognize iterators that cannot be reversed (CUDA-style solution)

  // Base class for iterators
  template <typename Category, typename T, typename Distance = IvyMemoryHelpers::ptrdiff_t, typename Pointer = T*, typename Reference = T&> struct iterator{
    using value_type = T;
    using pointer = Pointer;
    using reference = Reference;
    using difference_type = Distance;
    using iterator_category = Category;
  };

  // Iterator traits
  template<typename Iterator> struct iterator_traits{
    typedef typename Iterator::value_type value_type;
    typedef typename Iterator::pointer pointer;
    typedef typename Iterator::reference reference;
    typedef typename Iterator::difference_type difference_type;
    typedef typename Iterator::iterator_category iterator_category;
  };
  template<typename T> struct iterator_traits<T*>{
    typedef T value_type;
    typedef T* pointer;
    typedef T& reference;
    typedef IvyMemoryHelpers::ptrdiff_t difference_type;
    typedef random_access_iterator_tag iterator_category;
  };

  // Reverse iterator implementation (CUDA-style solution)
  template <typename T, typename = void> struct stashing_iterator : std_ttraits::false_type{};
  template <typename T> struct stashing_iterator<T, std_ttraits::void_t<typename T::stashing_iterator_tag>> : std_ttraits::true_type{};
  template<typename T> inline constexpr bool stashing_iterator_v = stashing_iterator<T>::value;

  template <class Iterator> class reverse_iterator : public iterator<
    typename iterator_traits<Iterator>::iterator_category,
    typename iterator_traits<Iterator>::value_type,
    typename iterator_traits<Iterator>::difference_type,
    typename iterator_traits<Iterator>::pointer,
    typename iterator_traits<Iterator>::reference
  >{
  private:
    static_assert(!stashing_iterator_v<Iterator>, "The specified iterator type cannot be used with reverse_iterator.");

  protected:
    Iterator it_;
    Iterator current_;

  public:
    using iterator_base = iterator<
      typename iterator_traits<Iterator>::iterator_category,
      typename iterator_traits<Iterator>::value_type,
      typename iterator_traits<Iterator>::difference_type,
      typename iterator_traits<Iterator>::pointer,
      typename iterator_traits<Iterator>::reference
    >;
    using iterator_type = Iterator;
    using value_type = typename iterator_base::value_type;
    using pointer = typename iterator_base::pointer;
    using reference = typename iterator_base::reference;
    using difference_type = typename iterator_base::difference_type;
    using iterator_category = typename iterator_base::iterator_category;

    __CUDA_HOST_DEVICE__ reverse_iterator() : it_(), current_(){}
    __CUDA_HOST_DEVICE__ explicit reverse_iterator(Iterator const& it) : it_(it), current_(it){}
    template <typename IterUp> __CUDA_HOST_DEVICE__
    reverse_iterator(reverse_iterator<IterUp> const& itu) : it_(itu.base()), current_(itu.base()){}

    template <typename IterUp> __CUDA_HOST_DEVICE__ reverse_iterator& operator=(reverse_iterator<IterUp> const& itu){
      it_ = current_ = itu.base(); return *this;
    }

    __CUDA_HOST_DEVICE__ Iterator base() const{ return current_; }

    __CUDA_HOST_DEVICE__ reference operator*() const{ Iterator it_mp = current_; return *(--it_mp); }
    __CUDA_HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __CUDA_HOST_DEVICE__ reverse_iterator& operator++(){ --current_; return *this; }
    __CUDA_HOST_DEVICE__ reverse_iterator operator++(int){ reverse_iterator it_mp(*this); --current_; return it_mp; }
    __CUDA_HOST_DEVICE__ reverse_iterator& operator--(){ ++current_; return *this; }
    __CUDA_HOST_DEVICE__ reverse_iterator operator--(int){ reverse_iterator it_mp(*this); ++current_; return it_mp; }
    __CUDA_HOST_DEVICE__ reverse_iterator operator+(difference_type const& n) const{ return reverse_iterator(current_ - n); }
    __CUDA_HOST_DEVICE__ reverse_iterator& operator+=(difference_type const& n){ current_ -= n; return *this; }
    __CUDA_HOST_DEVICE__ reverse_iterator operator-(difference_type const& n) const{ return reverse_iterator(current_ + n); }
    __CUDA_HOST_DEVICE__ reverse_iterator& operator-=(difference_type const& n){ current_ += n; return *this; }
    __CUDA_HOST_DEVICE__ reference operator[](difference_type const& n) const{ return *(*this + n); }
  };
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline bool operator==(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() == y.base(); }
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline bool operator<(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() > y.base(); }
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline bool operator!=(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() != y.base(); }
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline bool operator>(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() < y.base(); }
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline  bool operator>=(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() <= y.base(); }
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline bool operator<=(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() >= y.base(); }
  template <typename Iterator1, typename Iterator2> __CUDA_HOST_DEVICE__
  inline auto operator-(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y) -> decltype(y.base() - x.base()){ return y.base() - x.base(); }
  template <typename Iterator> __CUDA_HOST_DEVICE__
  inline  reverse_iterator<Iterator> operator+(typename reverse_iterator<Iterator>::difference_type const& n, reverse_iterator<Iterator> const& it){ return reverse_iterator<Iterator>(it.base() - n); }
  template <typename Iterator> __CUDA_HOST_DEVICE__
  inline reverse_iterator<Iterator> make_reverse_iterator(Iterator const& it){ return reverse_iterator<Iterator>(it); }

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
    typename Distance = IvyMemoryHelpers::ptrdiff_t,
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
    typename Distance = IvyMemoryHelpers::ptrdiff_t,
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






}

#endif


#endif
