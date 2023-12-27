#ifndef IVYPOINTERTRAITS_H
#define IVYPOINTERTRAITS_H


#include "config/IvyCompilerConfig.h"
#include "IvyMemoryHelpers.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/memory/IvyAddressof.h"


#ifdef __USE_CUDA__

namespace std_ivy{
#define POINTER_TRAIT_CMDS \
POINTER_TRAIT_CMD(element_type, T) \
POINTER_TRAIT_CMD(difference_type, IvyMemoryHelpers::ptrdiff_t)
#define POINTER_TRAIT_CMD(TRAIT, DEFTYPE) \
  DEFINE_HAS_TRAIT(TRAIT); \
  template <typename T, bool = has_##TRAIT##_v<T>> struct pointer_traits_##TRAIT{ typedef DEFTYPE type; }; \
  template <typename T> struct pointer_traits_##TRAIT<T, true>{ typedef typename T::TRAIT type; }; \
  template <template <typename, typename...> typename S, typename T, typename ...Args> \
  struct pointer_traits_##TRAIT<S<T, Args...>, true>{ typedef typename S<T, Args...>::TRAIT type; }; \
  template <template <typename, typename...> typename S, typename T, typename ...Args> \
  struct pointer_traits_##TRAIT<S<T, Args...>, false>{ typedef DEFTYPE type; }; \
  template <typename T> using pointer_traits_##TRAIT##_t = typename pointer_traits_##TRAIT<T>::type;
  POINTER_TRAIT_CMDS;
#undef POINTER_TRAIT_CMD
#undef POINTER_TRAIT_CMDS

  template <typename T, typename U> struct has_rebind{
  private:
    template <typename R> static constexpr auto test(typename R::template rebind<U>* = 0) -> std_ttraits::true_type;
    template <typename R> static constexpr auto test(...) -> std_ttraits::false_type;
  public:
    static constexpr bool value = decltype(has_rebind::test<T>(0))::value;
  };
  template <typename T, typename U> inline constexpr bool has_rebind_v = has_rebind<T, U>::value;
  template <typename T, typename U, bool = has_rebind_v<T, U>> struct pointer_traits_rebind{ typedef typename T::template rebind<U> type; };
  template <typename T, typename U> using pointer_traits_rebind_t = typename pointer_traits_rebind<T, U>::type;
  template <template <typename, typename...> typename S, typename T, typename ...Args, typename U>
  struct pointer_traits_rebind<S<T, Args...>, U, true>{ typedef typename S<T, Args...>::template rebind<U> type; };
  template <template <typename, typename...> typename S, typename T, typename ...Args, typename U>
  struct pointer_traits_rebind<S<T, Args...>, U, false>{ typedef S<U, Args...> type; };


  template <typename T> struct pointer_traits{
    typedef T pointer;
    typedef pointer_traits_element_type_t<pointer> element_type;
    typedef pointer_traits_difference_type_t<pointer> difference_type;
    template <typename U> using rebind = pointer_traits_rebind_t<pointer, U>;
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
