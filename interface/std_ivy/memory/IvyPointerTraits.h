/**
 * @file IvyPointerTraits.h
 * @brief Pointer traits and detection helpers for custom pointer abstractions.
 */
#ifndef IVYPOINTERTRAITS_H
#define IVYPOINTERTRAITS_H


#include "config/IvyCompilerConfig.h"
#include "IvyBasicTypes.h"
#include "std_ivy/memory/IvyAddressof.h"


namespace std_ivy{
  /** @brief Macro bundle declaring default pointer trait helpers. */
#define POINTER_TRAIT_CMDS \
POINTER_TRAIT_CMD(element_type, T) \
POINTER_TRAIT_CMD(difference_type, IvyTypes::ptrdiff_t)
  /** @brief Macro template that declares one pointer-trait detector and resolver chain. */
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

  /** @brief Detect whether a pointer-like type exposes a `rebind<U>` alias template. */
  template <typename T, typename U> struct has_rebind{
  private:
    /** @brief Preferred overload when `R::rebind<U>` is available. */
    template <typename R> static constexpr auto test(typename R::template rebind<U>* = 0) -> std_ttraits::true_type;
    /** @brief Fallback overload when `R::rebind<U>` is unavailable. */
    template <typename R> static constexpr auto test(...) -> std_ttraits::false_type;
  public:
    /** @brief Compile-time boolean result for the rebind detection trait. */
    static constexpr bool value = decltype(has_rebind::test<T>(0))::value;
  };
  /** @brief Convenience variable template for `has_rebind<T, U>::value`. */
  template <typename T, typename U> __INLINE_FCN_RELAXED__ constexpr bool has_rebind_v = has_rebind<T, U>::value;
  /** @brief Rebind resolver for pointer-like types that define `rebind<U>`. */
  template <typename T, typename U, bool = has_rebind_v<T, U>> struct pointer_traits_rebind{ typedef typename T::template rebind<U> type; };
  /** @brief Convenience alias for the result type of `pointer_traits_rebind`. */
  template <typename T, typename U> using pointer_traits_rebind_t = typename pointer_traits_rebind<T, U>::type;
  /** @brief Rebind specialization for variadic pointer templates when `rebind<U>` exists. */
  template <template <typename, typename...> typename S, typename T, typename ...Args, typename U>
  struct pointer_traits_rebind<S<T, Args...>, U, true>{ typedef typename S<T, Args...>::template rebind<U> type; };
  /** @brief Rebind specialization that reconstructs variadic pointer templates when `rebind<U>` is absent. */
  template <template <typename, typename...> typename S, typename T, typename ...Args, typename U>
  struct pointer_traits_rebind<S<T, Args...>, U, false>{ typedef S<U, Args...> type; };


  /** @brief General pointer_traits implementation for pointer-like classes. */
  template <typename T> class pointer_traits{
  public:
    /** @brief Pointer type alias. */
    typedef T pointer;
    /** @brief Element type alias. */
    typedef pointer_traits_element_type_t<pointer> element_type;
    /** @brief Difference type alias. */
    typedef pointer_traits_difference_type_t<pointer> difference_type;
    /** @brief Rebind alias preserving pointer template semantics. */
    template <typename U> using rebind = pointer_traits_rebind_t<pointer, U>;
  private:
    /** @brief Placeholder type used when `element_type` is void. */
    struct nat{};
  public:
    /** @brief Obtain a pointer-like object addressing `x`. */
    static __HOST_DEVICE__ __CPP_CONSTEXPR__ pointer pointer_to(
      std_ttraits::conditional_t<std_ttraits::is_void_v<element_type>, nat, element_type>& x
    ){ return pointer::pointer_to(x); }
  };
  /** @brief Raw-pointer specialization of pointer_traits. */
  template<typename T> class pointer_traits<T*>{
  public:
    /** @brief Pointer type alias. */
    typedef T* pointer;
    /** @brief Element type alias. */
    typedef T element_type;
    /** @brief Difference type alias. */
    typedef IvyTypes::ptrdiff_t difference_type;
    /** @brief Rebind alias for raw pointers. */
    template<typename U> using rebind = U*;
  private:
    /** @brief Placeholder type used when `element_type` is void. */
    struct nat{};
  public:
    /** @brief Return the address of `x` as a raw pointer. */
    static __HOST_DEVICE__ __CPP_CONSTEXPR__ pointer pointer_to(
      std_ttraits::conditional_t<std_ttraits::is_void_v<element_type>, nat, element_type>& x
    ) __NOEXCEPT__{ return std_ivy::addressof(x); }
  };

}


#endif
