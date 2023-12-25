#ifndef IVYITERATORIMPL_H
#define IVYITERATORIMPL_H


#include "IvyMemoryHelpers.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyMemory.h"


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

  template <typename Category, typename T, typename Distance = IvyMemoryHelpers::ptrdiff_t, typename Pointer = T*, typename Reference = T&> struct iterator{
    typedef T value_type;
    typedef Pointer pointer;
    typedef Reference reference;
    typedef Distance difference_type;
    typedef Category iterator_category;
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
  >
  {
  private:
    static_assert(!stashing_iterator_v<Iterator>, "The specified iterator type cannot be used with reverse_iterator.");

  protected:
    Iterator it_;
    Iterator current_;

  public:
    typedef Iterator iterator_type;

    __CUDA_HOST_DEVICE__ reverse_iterator() : it_(), current_(){}
    __CUDA_HOST_DEVICE__ explicit reverse_iterator(Iterator const& it) : it_(it), current_(it){}
    template <typename IterUp> __CUDA_HOST_DEVICE__
    reverse_iterator(reverse_iterator<IterUp> const& itu) : it_(itu.base()), current_(itu.base()){}

    template <typename IterUp> __CUDA_HOST_DEVICE__ reverse_iterator& operator=(reverse_iterator<IterUp> const& itu){
      it_ = current_ = itu.base(); return *this;
    }

    __CUDA_HOST_DEVICE__  Iterator base() const{ return current_; }

    __CUDA_HOST_DEVICE__ reference operator*() const{ Iterator it_mp = current_; return *(--it_mp); }
    __CUDA_HOST_DEVICE__ pointer  operator->() const{ return std_mem::addressof(operator*()); }

    __CUDA_HOST_DEVICE__ reverse_iterator& operator++(){ --current_; return *this; }
    __CUDA_HOST_DEVICE__ reverse_iterator  operator++(int){ reverse_iterator it_mp(*this); --current_; return it_mp; }
    __CUDA_HOST_DEVICE__ reverse_iterator& operator--(){ ++current_; return *this; }
    __CUDA_HOST_DEVICE__ reverse_iterator  operator--(int){ reverse_iterator it_mp(*this); ++current_; return it_mp; }
    __CUDA_HOST_DEVICE__ reverse_iterator  operator+ (difference_type const& n) const{ return reverse_iterator(current_ - n); }
    __CUDA_HOST_DEVICE__ reverse_iterator& operator+=(difference_type const& n){ current_ -= n; return *this; }
    __CUDA_HOST_DEVICE__ reverse_iterator  operator- (difference_type const& n) const{ return reverse_iterator(current_ + n); }
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




}

#endif


#endif
