#ifndef IVYNODERELATIONS_H
#define IVYNODERELATIONS_H


#include "config/IvyCompilerConfig.h"
#include "autodiff/base_types/IvyThreadSafePtr.h"
#include "autodiff/base_types/IvyBaseNode.h"


namespace IvyMath{
  /*
  IvyNodeSelfRelations:
  Any node has dependence and differentiability properties.
  Because CUDA does not support RTTI, we need to implement these properties in a separate class and check them through static 'polymorphism'.
  The idea is to provide partial specializations when needed in order to implement these properties.

  IvyNodeSelfRelations::is_differentiable: Check if this node is differentiable. By default, no node is differentiable.
  IvyNodeSelfRelations::is_conjugatable: By default, no node is conjugatable.
  */
  template<typename T> struct IvyNodeSelfRelations{
    static constexpr bool is_conjugatable = false;
    static __CUDA_HOST_DEVICE__ void conjugate(T&){}
    static __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(T const&){ return false; }
  };
  /*
  is_conjugatable: Convenience function to check if a node is conjugatable.
  */
  template<typename T> inline constexpr bool is_conjugatable = IvyNodeSelfRelations<T>::is_conjugatable;
  /*
  conjugate: Convenience function to conjugate a node.
  */
  template<typename T> __CUDA_HOST_DEVICE__ void conjugate(T& x){ IvyNodeSelfRelations<T>::conjugate(x); }
  /*
  is_differentiable: Convenience function to check if a node is differentiable.
  */
  template<typename T> __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(T const& x){ return IvyNodeSelfRelations<T>::is_differentiable(x); }
  // Partial specializations
  template<typename T> struct IvyNodeSelfRelations<T*>{
    static constexpr bool is_conjugatable = is_conjugatable<T>;
    static __CUDA_HOST_DEVICE__ void conjugate(T*& x){ conjugate(*x); }
    static __CUDA_HOST_DEVICE__ bool is_differentiable(T* const& x){ return is_differentiable(*x); }
  };
  template<typename T> struct IvyNodeSelfRelations<IvyThreadSafePtr_t<T>>{
    static constexpr bool is_conjugatable = is_conjugatable<T>;
    static __CUDA_HOST_DEVICE__ void conjugate(IvyThreadSafePtr_t<T>& x){ conjugate(*x); }
    static __CUDA_HOST_DEVICE__ bool is_differentiable(IvyThreadSafePtr_t<T> const& x){ return is_differentiable(*x); }
  };

  /*
  IvyNodeBinaryRelations:
  Any node has binary relations with other nodes.
  Because CUDA does not support RTTI, we need to implement these properties in a separate class and check them through static 'polymorphism'.
  The idea is to provide partial specializations when needed in order to implement these properties.

  IvyNodeBinaryRelations::depends_on: Check if a function depends on a variable. By default, no function depends on any variable.
  */
  template<typename T, typename U> struct IvyNodeBinaryRelations{
    static __CUDA_HOST_DEVICE__ bool depends_on(T const& fcn, T const& var){ return (std_mem::addressof(fcn) == std_mem::addressof(var)); }
  };
  /*
  depends_on: Convenience function to check if a function depends on a variable.
  */
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool depends_on(T const& fcn, U const& var){
    return IvyNodeBinaryRelations<T, U>::depends_on(fcn, var);
  }
  // Partial specializations
  template<typename T, typename U> struct IvyNodeBinaryRelations<T, U*>{
    static __CUDA_HOST_DEVICE__ bool depends_on(T const& fcn, U* const& var){ return depends_on(fcn, *var); }
  };
  template<typename T, typename U> struct IvyNodeBinaryRelations<T*, U>{
    static __CUDA_HOST_DEVICE__ bool depends_on(T* const& fcn, U const& var){ return depends_on(*fcn, var); }
  };
  template<typename T, typename U> struct IvyNodeBinaryRelations<T*, U*>{
    static __CUDA_HOST_DEVICE__ bool depends_on(T* const& fcn, U* const& var){ return depends_on(*fcn, *var); }
  };
  template<typename T, typename U> struct IvyNodeBinaryRelations<T, IvyThreadSafePtr_t<U>>{
    static __CUDA_HOST_DEVICE__ bool depends_on(T const& fcn, U* const& var){ return depends_on(fcn, *var); }
  };
  template<typename T, typename U> struct IvyNodeBinaryRelations<IvyThreadSafePtr_t<T>, U>{
    static __CUDA_HOST_DEVICE__ bool depends_on(T* const& fcn, U const& var){ return depends_on(*fcn, var); }
  };
  template<typename T, typename U> struct IvyNodeBinaryRelations<IvyThreadSafePtr_t<T>, IvyThreadSafePtr_t<U>>{
    static __CUDA_HOST_DEVICE__ bool depends_on(T* const& fcn, U* const& var){ return depends_on(*fcn, *var); }
  };
}


#endif
