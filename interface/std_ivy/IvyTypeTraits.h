/**
 * @file IvyTypeTraits.h
 * @brief Type-trait wrapper and macro helpers for SFINAE and compile-time reflection checks.
 */
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

/** @brief SFINAE helper alias producing @p TYPE when the boolean expression is true. */
#define ENABLE_IF_TYPE_IMPL(TYPE, ...) std_ttraits::enable_if_t<__VA_ARGS__, TYPE>
/** @brief SFINAE helper alias producing bool when the boolean expression is true. */
#define ENABLE_IF_BOOL_IMPL(...) std_ttraits::enable_if_t<__VA_ARGS__, bool>
/** @brief Convenience defaulted template parameter for boolean SFINAE checks. */
#define ENABLE_IF_BOOL(...) ENABLE_IF_BOOL_IMPL(__VA_ARGS__) = true
/** @brief Alias for std::is_base_of_v. */
#define IS_BASE_OF(BASE, DERIVED) std_ttraits::is_base_of_v<BASE, DERIVED>
/** @brief Alias for std::is_same_v. */
#define IS_SAME(A, B) std_ttraits::is_same_v<A, B>
/** @brief Enable @p TYPE if @p DERIVED inherits from @p BASE. */
#define ENABLE_IF_TYPED_BASE_OF_IMPL(TYPE, BASE, DERIVED) ENABLE_IF_TYPE_IMPL(TYPE, IS_BASE_OF(BASE, DERIVED))
/** @brief Enable bool if @p DERIVED inherits from @p BASE. */
#define ENABLE_IF_BASE_OF_IMPL(BASE, DERIVED) ENABLE_IF_BOOL_IMPL(IS_BASE_OF(BASE, DERIVED))
/** @brief Convenience defaulted template parameter when @p DERIVED inherits from @p BASE. */
#define ENABLE_IF_BASE_OF(BASE, DERIVED) ENABLE_IF_BOOL(IS_BASE_OF(BASE, DERIVED))
/** @brief Enable bool if @p DERIVED does not inherit from @p BASE. */
#define ENABLE_IF_NOT_BASE_OF_IMPL(BASE, DERIVED) ENABLE_IF_BOOL_IMPL(!IS_BASE_OF(BASE, DERIVED))
/** @brief Convenience defaulted template parameter when @p DERIVED does not inherit from @p BASE. */
#define ENABLE_IF_NOT_BASE_OF(BASE, DERIVED) ENABLE_IF_BOOL(!IS_BASE_OF(BASE, DERIVED))
/** @brief Enable bool if two types are identical. */
#define ENABLE_IF_SAME_IMPL(A, B) ENABLE_IF_BOOL_IMPL(IS_SAME(A, B))
/** @brief Convenience defaulted template parameter when two types are identical. */
#define ENABLE_IF_SAME(A, B) ENABLE_IF_BOOL(IS_SAME(A, B))
/** @brief Enable bool if two types are different. */
#define ENABLE_IF_NOT_SAME_IMPL(A, B) ENABLE_IF_BOOL_IMPL(!IS_SAME(A, B))
/** @brief Convenience defaulted template parameter when two types are different. */
#define ENABLE_IF_NOT_SAME(A, B) ENABLE_IF_BOOL(!IS_SAME(A, B))
/** @brief Enable bool if the type expression is arithmetic. */
#define ENABLE_IF_ARITHMETIC_IMPL(...) ENABLE_IF_BOOL_IMPL(std_ttraits::is_arithmetic_v<__VA_ARGS__>)
/** @brief Convenience defaulted template parameter when the type expression is arithmetic. */
#define ENABLE_IF_ARITHMETIC(...) ENABLE_IF_BOOL(std_ttraits::is_arithmetic_v<__VA_ARGS__>)
/** @brief Generate has_TRAIT detector templates for nested type aliases or members. */
#define DEFINE_HAS_TRAIT(TRAIT) \
  template <typename T, typename = void> struct has_##TRAIT : std_ttraits::false_type{}; \
  template <typename T> struct has_##TRAIT<T, std_ttraits::void_t<typename T::TRAIT>> : std_ttraits::true_type{}; \
  template <typename T> inline constexpr bool has_##TRAIT##_v = has_##TRAIT<T>::value;
/** @brief Generate has_call_FCN detector templates for member function lookup. */
#define DEFINE_HAS_CALL(FCN) \
  template<typename T> struct has_call_##FCN{ \
    struct invalid_call_type{}; \
    template <typename U> static constexpr auto test(int) -> decltype(&U::FCN); \
    template <typename U> static constexpr auto test(...) -> invalid_call_type; \
    static constexpr bool value = !std_ttraits::is_same_v<invalid_call_type, decltype(test<T>(0))>; \
  }; \
  template<typename T> inline constexpr bool has_call_##FCN##_v = has_call_##FCN<T>::value;
/** @brief Generate inherited accessor wrappers that expose dependent member-call expressions. */
#define DEFINE_INHERITED_ACCESSOR_CALL(FCN) \
  template<typename T, ENABLE_IF_BOOL(std_ttraits::is_class_v<T> && !std_ttraits::is_final_v<T>)> struct inherited_accessor_call_##FCN : public T{ \
    template<typename... Args> auto test(Args... args) -> decltype(this->FCN(args...)); \
  };
/** @brief Generate has_member_MEMBER detector templates for static and non-static member lookup. */
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
