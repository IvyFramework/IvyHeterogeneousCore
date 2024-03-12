#ifndef IVYTHREADSAFEPTR_H
#define IVYTHREADSAFEPTR_H


#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyMemory.h"


// Macros for thread-safe pointer handling
#define IvyThreadSafePtr_t std_mem::shared_ptr
#define make_IvyThreadSafePtr std_mem::make_shared

namespace IvyMath{
  template<typename T> struct is_pointer : std_ttraits::false_type{};
  template<typename T> struct is_pointer< IvyThreadSafePtr_t<T> > : std_ttraits::true_type{};
  template<typename T> inline constexpr bool is_pointer_v = is_pointer<T>::value;
  template<typename T> using pointer_t = typename T::element_type;
}


#endif
