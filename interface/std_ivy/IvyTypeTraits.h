#ifndef IVYTYPETRAITS_H
#define IVYTYPETRAITS_H


#ifdef __USE_CUDA__

#include <cuda/std/type_traits>
#ifndef std_ttraits
#define std_ttraits cuda::std
#endif

#else

#include <type_traits>
#ifndef std_ttraits
#define std_ttraits std
#endif

#endif

// Define shorthands for common type trait checks
#define ENABLE_IF_BASE_OF(BASE, DERIVED) std_ttraits::enable_if_t<std_ttraits::is_base_of_v<BASE, DERIVED>, bool> = true
#define DEFINE_HAS_TRAIT(TRAIT) \
  template <typename T, typename = void> struct has_##TRAIT : std_ttraits::false_type{}; \
  template <typename T> struct has_##TRAIT<T, std_ttraits::void_t<typename T::TRAIT>> : std_ttraits::true_type{}; \
  template <typename T> inline constexpr bool has_##TRAIT##_v = has_##TRAIT<T>::value;
#define DEFINE_HAS_CALL(FCN) \
  template<typename T> struct has_call_##FCN{ \
    template <typename U> static constexpr auto test(int) -> decltype(&U::FCN); \
    template <typename U> static constexpr auto test(...) -> void; \
    static constexpr bool value = !std_ttraits::is_void_v<decltype(test<T>(0))>; \
  }; \
  template<typename T> inline constexpr bool has_call_##FCN##_v = has_call_##FCN<T>::value;
#define DEFINE_HAS_MEMBER(MEMBER) \
  template<typename T> class has_member_##MEMBER{ \
  private: \
    template <typename U> static constexpr auto test(int) -> decltype(U::MEMBER); \
    template <typename U> static constexpr auto test(...) -> void; \
  public: \
    static constexpr bool value = !std_ttraits::is_void_v<decltype(test<T>(0))>; \
  }; \
  template<typename T> inline constexpr bool has_member_##MEMBER##_v = has_member_##MEMBER<T>::value;


#endif
