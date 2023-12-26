#ifndef IVYVECTORIMPL_HH
#define IVYVECTORIMPL_HH


#include "std_ivy/memory/IvyAllocator.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  class IvyVector;
  typedef IvyVector vector;

  template<typename T, typename Allocator> class IvyVector{
  public:
    typedef T value_type;
    typedef Allocator allocator_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef std_ivy::allocator_traits<Allocator>::pointer pointer;
    typedef std_ivy::allocator_traits<Allocator>::const_pointer const_pointer;
    typedef std_ivy::allocator_traits<Allocator>::size_type size_type;
    typedef std_ivy::allocator_traits<Allocator>::difference_type difference_type;



  };

}

#endif


#endif
