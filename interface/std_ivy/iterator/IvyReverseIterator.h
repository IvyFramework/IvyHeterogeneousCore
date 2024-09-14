#ifndef IVYREVERSEITERATOR_H
#define IVYREVERSEITERATOR_H


#include "std_ivy/iterator/IvyIteratorTraits.h"
#include "std_ivy/iterator/IvyIteratorImpl.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  // Reverse iterator implementation (CUDA-style solution)
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

    __HOST_DEVICE__ reverse_iterator() : it_(), current_(){}
    __HOST_DEVICE__ explicit reverse_iterator(Iterator const& it) : it_(it), current_(it){}
    template <typename IterUp> __HOST_DEVICE__
      reverse_iterator(reverse_iterator<IterUp> const& itu) : it_(itu.base()), current_(itu.base()){}

    template <typename IterUp> __HOST_DEVICE__ reverse_iterator& operator=(reverse_iterator<IterUp> const& itu){
      it_ = current_ = itu.base(); return *this;
    }
    __HOST_DEVICE__ void swap(reverse_iterator& other){ std_util::swap(it_, other.it_); std_util::swap(current_, other.current_); }

    __HOST_DEVICE__ Iterator base() const{ return current_; }

    __HOST_DEVICE__ reference operator*() const{ Iterator it_mp = current_; return *(--it_mp); }
    __HOST_DEVICE__ pointer operator->() const{ return std_mem::addressof(operator*()); }

    __HOST_DEVICE__ reverse_iterator& operator++(){ --current_; return *this; }
    __HOST_DEVICE__ reverse_iterator operator++(int){ reverse_iterator it_mp(*this); --current_; return it_mp; }
    __HOST_DEVICE__ reverse_iterator& operator--(){ ++current_; return *this; }
    __HOST_DEVICE__ reverse_iterator operator--(int){ reverse_iterator it_mp(*this); ++current_; return it_mp; }
    __HOST_DEVICE__ reverse_iterator operator+(difference_type const& n) const{ return reverse_iterator(current_ - n); }
    __HOST_DEVICE__ reverse_iterator& operator+=(difference_type const& n){ current_ -= n; return *this; }
    __HOST_DEVICE__ reverse_iterator operator-(difference_type const& n) const{ return reverse_iterator(current_ + n); }
    __HOST_DEVICE__ reverse_iterator& operator-=(difference_type const& n){ current_ += n; return *this; }
    __HOST_DEVICE__ reference operator[](difference_type const& n) const{ return *(*this + n); }
  };
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline bool operator==(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() == y.base(); }
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline bool operator<(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() > y.base(); }
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline bool operator!=(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() != y.base(); }
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline bool operator>(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() < y.base(); }
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline  bool operator>=(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() <= y.base(); }
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline bool operator<=(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y){ return x.base() >= y.base(); }
  template <typename Iterator1, typename Iterator2> __HOST_DEVICE__
    inline auto operator-(reverse_iterator<Iterator1> const& x, reverse_iterator<Iterator2> const& y) -> decltype(y.base() - x.base()){ return y.base() - x.base(); }
  template <typename Iterator> __HOST_DEVICE__
    inline  reverse_iterator<Iterator> operator+(typename reverse_iterator<Iterator>::difference_type const& n, reverse_iterator<Iterator> const& it){ return reverse_iterator<Iterator>(it.base() - n); }
  template <typename Iterator> __HOST_DEVICE__
    inline reverse_iterator<Iterator> make_reverse_iterator(Iterator const& it){ return reverse_iterator<Iterator>(it); }
}
namespace std_util{
  template <typename Iterator> __HOST_DEVICE__ void swap(std_ivy::reverse_iterator<Iterator>& x, std_ivy::reverse_iterator<Iterator>& y){ x.swap(y); }
}

#endif


#endif
