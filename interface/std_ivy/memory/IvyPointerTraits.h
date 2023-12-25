#ifndef IVYPOINTERTRAITS_H
#define IVYPOINTERTRAITS_H


#include "IvyCompilerFlags.h"
#include "IvyCudaFlags.h"
#include "IvyMemoryHelpers.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/memory/IvyAddressof.h"


#ifdef __USE_CUDA__

namespace std_ivy{
#define POINTER_TRAIT_CMDS \
POINTER_TRAIT_CMD(element_type, T) \
POINTER_TRAIT_CMD(difference_type, IvyMemoryHelpers::ptrdiff_t)
#define POINTER_TRAIT_CMD(TRAIT, DEFTYPE) \
  template <typename T, typename = void> struct __has_##TRAIT : std_ttraits::false_type{}; \
  template <typename T> struct __has_##TRAIT<T, std_ttraits::void_t<typename T::TRAIT>> : std_ttraits::true_type{}; \
  template <typename T> inline constexpr bool __has_##TRAIT##_v = __has_##TRAIT<T>::value; \
  template <typename T, bool = __has_##TRAIT##_v<T>> struct __pointer_traits_##TRAIT{ typedef DEFTYPE type; }; \
  template <typename T> struct __pointer_traits_##TRAIT<T, true>{ typedef typename T::TRAIT type; }; \
  template <template <typename, typename...> typename S, typename T, typename ...Args> \
  struct __pointer_traits_##TRAIT<S<T, Args...>, true>{ typedef typename S<T, Args...>::TRAIT type; }; \
  template <template <typename, typename...> typename S, typename T, typename ...Args> \
  struct __pointer_traits_##TRAIT<S<T, Args...>, false>{ typedef DEFTYPE type; }; \
  template <typename T> using __pointer_traits_##TRAIT##_t = typename __pointer_traits_##TRAIT<T>::type;
  POINTER_TRAIT_CMDS;
#undef POINTER_TRAIT_CMD
#undef POINTER_TRAIT_CMDS

  template <typename T, typename U> struct __has_rebind{
  private:
    template <typename R> static constexpr auto test(...) -> std_ttraits::false_type;
    template <typename R> static constexpr auto test(typename R::template rebind<U>* = 0) -> std_ttraits::true_type;
  public:
    static constexpr bool value = decltype(__has_rebind::test<T>(0))::value;
  };
  template <typename T, typename U> inline constexpr bool __has_rebind_v = __has_rebind<T, U>::value;
  template <typename T, typename U, bool = __has_rebind_v<T, U>> struct __pointer_traits_rebind{ typedef typename T::template rebind<U> type; };
  template <typename T, typename U> using __pointer_traits_rebind_t = typename __pointer_traits_rebind<T, U>::type;
  template <template <typename, typename...> typename S, typename T, typename ...Args, typename U>
  struct __pointer_traits_rebind<S<T, Args...>, U, true>{ typedef typename S<T, Args...>::template rebind<U> type; };
  template <template <typename, typename...> typename S, typename T, typename ...Args, typename U>
  struct __pointer_traits_rebind<S<T, Args...>, U, false>{ typedef S<U, Args...> type; };


  template <typename T> struct pointer_traits{
    typedef T pointer;
    typedef __pointer_traits_element_type_t<pointer> element_type;
    typedef __pointer_traits_difference_type_t<pointer> difference_type;
    template <typename U> using rebind = __pointer_traits_rebind_t<pointer, U>;
  private:
    struct nat{};
  public:
    static __CUDA_HOST_DEVICE__ __CPP_CONSTEXPR__ pointer pointer_to(
      std_ttraits::conditional_t<std_ttraits::is_void_v<element_type>, nat, element_type>& x
    ){ return pointer::pointer_to(x); }
  };
  template<typename T> struct pointer_traits<T*>{
  protected:
    typedef T* pointer;
    typedef T element_type;
    typedef IvyMemoryHelpers::ptrdiff_t difference_type;
    template<typename U> using rebind = U*;
  private:
    struct nat{};
  public:
    static __CUDA_HOST_DEVICE__ __CPP_CONSTEXPR__ pointer pointer_to(
      std_ttraits::conditional_t<std_ttraits::is_void_v<element_type>, nat, element_type>& x
    ) __NOEXCEPT__{ return std_ivy::addressof(x); }
  };

}

#endif


#endif
