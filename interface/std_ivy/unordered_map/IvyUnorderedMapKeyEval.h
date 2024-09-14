#ifndef IVYUNORDEREDMAPKEYEVAL_H
#define IVYUNORDEREDMAPKEYEVAL_H


/*
  IvyUnorderedMapKeyEval.h:
  A set of structs are provided to evaluate the equality of two keys for the IvyUnorderedMap.
  The structs generalize the KeyEval concept of std::unordered_map to also receive the total present data size and capacity before insertion.
  In this way, information relevant to the stored data can be used in a more optimal manner than simply relying on key equality algorithms.
*/


#ifdef __USE_CUDA__

#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/IvyCmath.h"
#include "IvyBasicTypes.h"


namespace std_ivy{
  /*
    IvyKeyEqualBinaryEval: A struct to evaluate the equality of two keys using std_fcnal::equal_to.
    IvyKeyEqualBinaryEval::bucket_size: Returns the data size for the predicted number of buckets.
  */
  template<typename Key, typename BinaryPredicate> struct IvyKeyEqualBinaryEval{
    using key_type = Key;
    using binary_predicate = BinaryPredicate;
    __HOST_DEVICE__ static IvyTypes::size_t bucket_size(IvyTypes::size_t const& /*n_size*/, IvyTypes::size_t const& n_capacity){ return (n_capacity<2 ? 2 : n_capacity); }
    __HOST_DEVICE__ static IvyTypes::size_t preferred_data_capacity(IvyTypes::size_t const& n_capacity_buckets){ return n_capacity_buckets; }
    __HOST_DEVICE__ static bool eval(IvyTypes::size_t const& /*n_size*/, IvyTypes::size_t const& /*n_capacity*/, Key const& a, Key const& b){
      return BinaryPredicate()(a, b);
    }
  };
  /*
    IvyKeyEqualEval: Specialization of IvyKeyEqualBinaryEval using BinaryPredicate = std_fcnal::equal_to.
  */
  template<typename Key> using IvyKeyEqualEval = IvyKeyEqualBinaryEval<Key, std_fcnal::equal_to<Key>>;

  /*
    IvyKeyEqualBySqrtNSizeEval: A struct to evaluate the equality of two keys using the modulo operation by the data size.
    IvyKeyEqualBySqrtNSizeEval::bucket_size: Returns the (square-root of the data size)+1 for the predicted number of buckets.
  */
  template<typename Key> struct IvyKeyEqualBySqrtNSizeEval{
    using key_type = Key;
    __HOST_DEVICE__ static IvyTypes::size_t bucket_size(IvyTypes::size_t const& n_size, IvyTypes::size_t const& /*n_capacity*/){ return std_math::sqrt(__STATIC_CAST__(double, n_size)) + 1; }
    __HOST_DEVICE__ static IvyTypes::size_t preferred_data_capacity(IvyTypes::size_t const& n_capacity_buckets){
      if (n_capacity_buckets<2) return 1;
      return n_capacity_buckets * n_capacity_buckets-1;
    }
    __HOST_DEVICE__ static bool eval(IvyTypes::size_t const& n_size, IvyTypes::size_t const& n_capacity, Key const& a, Key const& b){
      IvyTypes::size_t ns = bucket_size(n_size, n_capacity);
      return (a % ns) == (b % ns);
    }
  };

  /*
    IvyKeyEqualBySqrtNCapacityEval: A struct to evaluate the equality of two keys using the modulo operation by the data capacity.
    IvyKeyEqualBySqrtNCapacityEval::bucket_size: Returns the (square-root of the data capacity)+1 for the predicted number of buckets.
  */
  template<typename Key> struct IvyKeyEqualBySqrtNCapacityEval{
    using key_type = Key;
    __HOST_DEVICE__ static IvyTypes::size_t bucket_size(IvyTypes::size_t const& /*n_size*/, IvyTypes::size_t const& n_capacity){ return std_math::sqrt(__STATIC_CAST__(double, n_capacity)) + 1; }
    __HOST_DEVICE__ static IvyTypes::size_t preferred_data_capacity(IvyTypes::size_t const& n_capacity_buckets){
      if (n_capacity_buckets<2) return 1;
      return n_capacity_buckets * n_capacity_buckets-1;
    }
    __HOST_DEVICE__ static bool eval(IvyTypes::size_t const& n_size, IvyTypes::size_t const& n_capacity, Key const& a, Key const& b){
      IvyTypes::size_t ns = bucket_size(n_size, n_capacity);
      return (a % ns) == (b % ns);
    }
  };

  /*
    IvyKeyEqualEvalDefault: The default struct to evaluate the equality of two keys. It uses IvyKeyEqualEval.
  */
  template<typename Key> using IvyKeyEqualEvalDefault = IvyKeyEqualEval<Key>;

  /*
    IvyHashEqualEvalDefault: The default struct to evaluate the equality of two hashes. It uses IvyKeyEqualBySqrtNSizeEval.
  */
  template<typename Key> using IvyHashEqualEvalDefault = IvyKeyEqualBySqrtNSizeEval<Key>;
}

#endif


#endif
