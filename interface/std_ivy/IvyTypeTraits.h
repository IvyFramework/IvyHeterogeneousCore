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
#define ENABLE_IF_BOOL_IMPL(...) std_ttraits::enable_if_t<__VA_ARGS__, bool>
#define ENABLE_IF_BOOL(...) ENABLE_IF_BOOL_IMPL(__VA_ARGS__) = true
#define IS_BASE_OF(BASE, DERIVED) std_ttraits::is_base_of_v<BASE, DERIVED>
#define ENABLE_IF_BASE_OF_IMPL(BASE, DERIVED) ENABLE_IF_BOOL_IMPL(IS_BASE_OF(BASE, DERIVED))
#define ENABLE_IF_BASE_OF(BASE, DERIVED) ENABLE_IF_BOOL(IS_BASE_OF(BASE, DERIVED))
#define ENABLE_IF_ARITHMETIC_IMPL(...) ENABLE_IF_BOOL_IMPL(std_ttraits::is_arithmetic_v<__VA_ARGS__>)
#define ENABLE_IF_ARITHMETIC(...) ENABLE_IF_BOOL(std_ttraits::is_arithmetic_v<__VA_ARGS__>)
#define DEFINE_HAS_TRAIT(TRAIT) \
  template <typename T, typename = void> struct has_##TRAIT : std_ttraits::false_type{}; \
  template <typename T> struct has_##TRAIT<T, std_ttraits::void_t<typename T::TRAIT>> : std_ttraits::true_type{}; \
  template <typename T> inline constexpr bool has_##TRAIT##_v = has_##TRAIT<T>::value;
#define DEFINE_HAS_CALL(FCN) \
  template<typename T> struct has_call_##FCN{ \
    struct invalid_call_type{}; \
    template <typename U> static constexpr auto test(int) -> decltype(&U::FCN); \
    template <typename U> static constexpr auto test(...) -> invalid_call_type; \
    static constexpr bool value = !std_ttraits::is_same_v<invalid_call_type, decltype(test<T>(0))>; \
  }; \
  template<typename T> inline constexpr bool has_call_##FCN##_v = has_call_##FCN<T>::value;
#define DEFINE_INHERITED_ACCESSOR_CALL(FCN) \
  template<typename T, ENABLE_IF_BOOL(std_ttraits::is_class_v<T> && !std_ttraits::is_final_v<T>)> struct inherited_accessor_call_##FCN : public T{ \
    template<typename... Args> auto test(Args... args) -> decltype(this->FCN(args...)); \
  };
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
