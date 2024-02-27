#ifndef IVYHASH_H
#define IVYHASH_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyTypeTraits.h"
#include "IvyBasicTypes.h"


namespace ivy_hash_impl{
  template<typename T> struct IvyHash{
    using result_type = IvyTypes::size_t;
    using argument_type = std_ttraits::remove_cv_t<T>;

    __CUDA_HOST_DEVICE__ constexpr result_type operator()(argument_type const& v) const{
      constexpr result_type nb_T = sizeof(argument_type);
      constexpr result_type size_partition = sizeof(result_type);
      constexpr result_type nbits_partition = size_partition*8;
      constexpr result_type nparts_full = nb_T / size_partition;
      constexpr result_type remainder = nb_T % size_partition;
      constexpr result_type part_full_shift_offset = (remainder==0 ? 0 : 1);

      result_type res = 0;
      result_type const* prc = __REINTERPRET_CAST__(result_type const*, &__CONST_CAST__(char&, __REINTERPRET_CAST__(char const volatile&, v)));
      if constexpr (nparts_full>0){
        for (result_type i=0; i<nparts_full; ++i){
          result_type const& pv = prc[i];
          res ^= (pv<<((i+part_full_shift_offset)%nbits_partition));
        }
      }
      if constexpr (remainder>0){
        result_type pv = 0;
        char const* prch = __REINTERPRET_CAST__(char const*, &__CONST_CAST__(char&, __REINTERPRET_CAST__(const volatile char&, prc[nparts_full])));
        for (result_type i=0; i<remainder; ++i) pv |= (prch[i]<<(i*8));
        res ^= pv;
      }
      return res;
    }
  };
  template<> struct IvyHash<char const*>{
    using result_type = IvyTypes::size_t;
    using argument_type = char const*;

    __CUDA_HOST_DEVICE__ constexpr result_type operator()(argument_type const& v) const{
      constexpr result_type size_partition = sizeof(result_type);
      constexpr result_type nbits_partition = size_partition*8;
      constexpr result_type nbits_arg_el = sizeof(char)*8;
      result_type res = 0;
      {
        argument_type vv = v;
        for (result_type i=0; *vv; ++i){ res ^= ((*vv)<<((i*8)%(nbits_partition-nbits_arg_el+1))); ++vv; }
      }
      return res;
    }
  };
  template<> struct IvyHash<char*>{
    using result_type = IvyTypes::size_t;
    using argument_type = char*;

    __CUDA_HOST_DEVICE__ constexpr result_type operator()(argument_type const& v) const{
      constexpr result_type size_partition = sizeof(result_type);
      constexpr result_type nbits_partition = size_partition*8;
      constexpr result_type nbits_arg_el = sizeof(char)*8;
      result_type res = 0;
      {
        argument_type vv = v;
        for (result_type i=0; *vv; ++i){ res ^= ((*vv)<<((i*8)%(nbits_partition-nbits_arg_el+1))); ++vv; }
      }
      return res;
    }
  };
}
namespace std_ivy{
  template<typename T> using hash = ivy_hash_impl::IvyHash<T>;
}


#endif
