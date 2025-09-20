#ifndef IVYBASEMODIFIABLE_H
#define IVYBASEMODIFIABLE_H


#include "config/IvyConfig.h"
#include "std_ivy/IvyAtomic.h"
#include "std_ivy/IvyTypeTraits.h"


namespace IvyMath{
  class IvyBaseModifiable{
    protected:
      std_atomic::atomic<bool> is_modified_;

    public:
      __HOST_DEVICE__ IvyBaseModifiable() : is_modified_(true) {}
      __HOST_DEVICE__ bool is_modified() const{ return is_modified_.load(); }
      __HOST_DEVICE__ void set_modified(bool flag){ is_modified_.store(flag); }
  };

  template<
    typename T,
    ENABLE_IF_BASE_OF(IvyBaseModifiable, T)
  > __HOST_DEVICE__ bool is_modified(T const& obj){ return obj.is_modified(); }
  template<
    typename T,
    ENABLE_IF_NOT_BASE_OF(IvyBaseModifiable, T)
  > __HOST_DEVICE__ constexpr bool is_modified(T const& obj){ return true; }


}


#endif
