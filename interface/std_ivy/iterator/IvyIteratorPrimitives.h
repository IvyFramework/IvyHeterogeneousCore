/**
 * @file IvyIteratorPrimitives.h
 * @brief Primitive iterator tags and base category types used by std_ivy iterators.
 */
#ifndef IVYITERATORPRIMITIVES_H
#define IVYITERATORPRIMITIVES_H


#include "IvyBasicTypes.h"
#include "std_ivy/IvyTypeTraits.h"


namespace std_ivy{
  /** @brief Input-iterator category tag. */
  struct input_iterator_tag{};
  /** @brief Output-iterator category tag. */
  struct output_iterator_tag{};
  /** @brief Forward-iterator category tag. */
  struct forward_iterator_tag : public input_iterator_tag{};
  /** @brief Bidirectional-iterator category tag. */
  struct bidirectional_iterator_tag : public forward_iterator_tag{};
  /** @brief Random-access-iterator category tag. */
  struct random_access_iterator_tag : public bidirectional_iterator_tag{};
  /** @brief Partially contiguous iterator category tag. */
  struct partially_contiguous_iterator_tag : public random_access_iterator_tag{};
  /** @brief Fully contiguous iterator category tag. */
  struct contiguous_iterator_tag : public partially_contiguous_iterator_tag{};
  /** @brief Marker tag type for iterators that cannot be safely reversed. */
  using stashing_iterator_tag = void; // Dummy tag to recognize iterators that cannot be reversed (CUDA-style solution)

  /**
   * @brief Base iterator facade exposing canonical nested typedefs.
   * @tparam Category Iterator category tag.
   * @tparam T Value type.
   * @tparam Distance Difference type.
   * @tparam Pointer Pointer type.
   * @tparam Reference Reference type.
   */
  template<typename Category, typename T, typename Distance = IvyTypes::ptrdiff_t, typename Pointer = T*, typename Reference = T&> struct iterator{
    /** @brief Iterator value type. */
    using value_type = T;
    /** @brief Iterator pointer type. */
    using pointer = Pointer;
    /** @brief Iterator reference type. */
    using reference = Reference;
    /** @brief Iterator distance type. */
    using difference_type = Distance;
    /** @brief Iterator category tag. */
    using iterator_category = Category;
  };

  /** @brief Trait detecting presence of stashing_iterator_tag marker. */
  template<typename T, typename = void> struct stashing_iterator : std_ttraits::false_type{};
  /** @brief Positive specialization of stashing_iterator when marker tag exists. */
  template<typename T> struct stashing_iterator<T, std_ttraits::void_t<typename T::stashing_iterator_tag>> : std_ttraits::true_type{};
  /** @brief Convenience boolean for stashing_iterator trait. */
  template<typename T> inline constexpr bool stashing_iterator_v = stashing_iterator<T>::value;

  /** @brief Internal helpers for iterator trait detection. */
  namespace _iterator_details{
    DEFINE_HAS_TRAIT(iterator_category);
  }
  /** @brief Trait detecting whether a type is a contiguous iterator category. */
  template<typename T, typename = void> struct is_contiguous_iterator : std_ttraits::false_type{};
  /** @brief Positive contiguous-iterator trait specialization. */
  template<typename T> struct is_contiguous_iterator<
    T,
    std_ttraits::enable_if_t<
      _iterator_details::has_iterator_category_v<T>
      &&
      std_ttraits::is_base_of_v<contiguous_iterator_tag, typename T::iterator_category>
    >
  > : std_ttraits::true_type{};
  /** @brief Convenience boolean for contiguous-iterator trait. */
  template<typename T> inline constexpr bool is_contiguous_iterator_v = is_contiguous_iterator<T>::value;


}


#endif
