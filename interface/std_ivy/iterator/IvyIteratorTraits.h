#ifndef IVYITERATORTRAITS_H
#define IVYITERATORTRAITS_H


#include "IvyBasicTypes.h"
#include "std_ivy/iterator/IvyIteratorPrimitives.h"


namespace std_ivy{
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
    typedef IvyTypes::ptrdiff_t difference_type;
    typedef random_access_iterator_tag iterator_category;
  };
}


#endif
