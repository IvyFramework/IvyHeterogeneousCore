#ifndef IVYADDRESSOF_H
#define IVYADDRESSOF_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyTypeTraits.h"


#ifdef __USE_CUDA__

namespace std_ivy{
#if __check_builtin(builtin_addressof) \
 || ((COMPILER == COMPILER_GCC) && (COMPILER_VERSION >= 70000)) \
 || (COMPILER == COMPILER_MSVC) \
 || (COMPILER == COMPILER_NVHPC)
  template<typename T>
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ __CPP_CONSTEXPR__ T* addressof(T& v) __NOEXCEPT__{ return __builtin_addressof(v); }
#else
  template<typename T>
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ std_ttraits::enable_if_t<std_ttraits::is_object_v<std_ttraits::remove_reference_t<T>>, T*>
    addressof(T& v) __NOEXCEPT__{ return __REINTERPRET_CAST__(T*, &__CONST_CAST__(char&, __REINTERPRET_CAST__(const volatile char&, v))); }
  template<typename T>
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ std_ttraits::enable_if_t<!std_ttraits::is_object_v<std_ttraits::remove_reference_t<T>>, T*>
    addressof(T& v) __NOEXCEPT__{ return &v; }
#endif

#define __OBJC_POINTER_CMD__(__OBJ_PTR_TYPE__) \
  template<typename T> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ __OBJ_PTR_TYPE__ T* addressof(__OBJ_PTR_TYPE__ T& v) __NOEXCEPT__{ return &v; }
  __OBJC_POINTER_CMDS__;
#undef __OBJC_POINTER_CMD__

  template<typename T> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ T* addressof(T const&&) = delete;
  template<typename T> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ T const* addressof(T const&&) = delete;
}

#endif


#endif
