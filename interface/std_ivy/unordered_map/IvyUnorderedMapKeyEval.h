/**
 * @file IvyUnorderedMapKeyEval.h
 * @brief Key extraction and equality/hash evaluation helpers for unordered_map nodes.
 */
#ifndef IVYUNORDEREDMAPKEYEVAL_H
#define IVYUNORDEREDMAPKEYEVAL_H


/**
  IvyUnorderedMapKeyEval.h:
  A set of structs are provided to evaluate the equality of two keys for the IvyUnorderedMap.
  The structs generalize the KeyEval concept of std::unordered_map to also receive the total present data size and capacity before insertion.
  In this way, information relevant to the stored data can be used in a more optimal manner than simply relying on key equality algorithms.
*/


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/IvyCmath.h"
#include "IvyBasicTypes.h"


namespace std_ivy{
  /**
   * @brief Generic key-equality evaluator parameterized by a binary predicate.
   * @tparam Key Key type.
   * @tparam BinaryPredicate Binary predicate used to compare keys.
   */
  template<typename Key, typename BinaryPredicate> struct IvyKeyEqualBinaryEval{
    /** @brief Key type alias. */
    using key_type = Key;
    /** @brief Predicate type alias. */
    using binary_predicate = BinaryPredicate;
    /** @brief Compute predicted bucket count from current logical size/capacity. */
    __HOST_DEVICE__ static IvyTypes::size_t bucket_size(IvyTypes::size_t const& /*n_size*/, IvyTypes::size_t const& n_capacity){ return (n_capacity<2 ? 2 : n_capacity); }
    /** @brief Compute preferred data capacity from predicted bucket count. */
    __HOST_DEVICE__ static IvyTypes::size_t preferred_data_capacity(IvyTypes::size_t const& n_capacity_buckets){ return n_capacity_buckets; }
    /** @brief Evaluate key equality with the configured binary predicate. */
    __HOST_DEVICE__ static bool eval(IvyTypes::size_t const& /*n_size*/, IvyTypes::size_t const& /*n_capacity*/, Key const& a, Key const& b){
      return BinaryPredicate()(a, b);
    }
  };
  /** @brief Key-equality evaluator using std_fcnal::equal_to. */
  template<typename Key> using IvyKeyEqualEval = IvyKeyEqualBinaryEval<Key, std_fcnal::equal_to<Key>>;

  /**
   * @brief Key-equality evaluator using modulo over sqrt(size)+1 bucketing strategy.
   * @tparam Key Key type supporting modulo operation.
   */
  template<typename Key> struct IvyKeyEqualBySqrtNSizeEval{
    /** @brief Key type alias. */
    using key_type = Key;
    /** @brief Compute predicted bucket count from current logical size. */
    __HOST_DEVICE__ static IvyTypes::size_t bucket_size(IvyTypes::size_t const& n_size, IvyTypes::size_t const& /*n_capacity*/){ return std_math::sqrt(__STATIC_CAST__(double, n_size)) + 1; }
    /** @brief Compute preferred data capacity from bucket count. */
    __HOST_DEVICE__ static IvyTypes::size_t preferred_data_capacity(IvyTypes::size_t const& n_capacity_buckets){
      if (n_capacity_buckets<2) return 1;
      return n_capacity_buckets * n_capacity_buckets-1;
    }
    /** @brief Evaluate key equivalence under sqrt(size)-based bucket mapping. */
    __HOST_DEVICE__ static bool eval(IvyTypes::size_t const& n_size, IvyTypes::size_t const& n_capacity, Key const& a, Key const& b){
      IvyTypes::size_t ns = bucket_size(n_size, n_capacity);
      return (a % ns) == (b % ns);
    }
  };

  /**
   * @brief Key-equality evaluator using modulo over sqrt(capacity)+1 bucketing strategy.
   * @tparam Key Key type supporting modulo operation.
   */
  template<typename Key> struct IvyKeyEqualBySqrtNCapacityEval{
    /** @brief Key type alias. */
    using key_type = Key;
    /** @brief Compute predicted bucket count from current capacity. */
    __HOST_DEVICE__ static IvyTypes::size_t bucket_size(IvyTypes::size_t const& /*n_size*/, IvyTypes::size_t const& n_capacity){ return std_math::sqrt(__STATIC_CAST__(double, n_capacity)) + 1; }
    /** @brief Compute preferred data capacity from bucket count. */
    __HOST_DEVICE__ static IvyTypes::size_t preferred_data_capacity(IvyTypes::size_t const& n_capacity_buckets){
      if (n_capacity_buckets<2) return 1;
      return n_capacity_buckets * n_capacity_buckets-1;
    }
    /** @brief Evaluate key equivalence under sqrt(capacity)-based bucket mapping. */
    __HOST_DEVICE__ static bool eval(IvyTypes::size_t const& n_size, IvyTypes::size_t const& n_capacity, Key const& a, Key const& b){
      IvyTypes::size_t ns = bucket_size(n_size, n_capacity);
      return (a % ns) == (b % ns);
    }
  };

  /** @brief Default key-equality evaluator alias. */
  template<typename Key> using IvyKeyEqualEvalDefault = IvyKeyEqualEval<Key>;

  /** @brief Default hash-equality evaluator alias. */
  template<typename Key> using IvyHashEqualEvalDefault = IvyKeyEqualBySqrtNSizeEval<Key>;
}


#endif
