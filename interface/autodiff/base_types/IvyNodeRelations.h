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
    static __HOST_DEVICE__ void conjugate(T&){}
    static __HOST_DEVICE__ constexpr bool is_differentiable(T const&){ return false; }
  };
  /*
  is_conjugatable: Convenience function to check if a node is conjugatable.
  */
  template<typename T> inline constexpr bool is_conjugatable = IvyNodeSelfRelations<T>::is_conjugatable;
  /*
  conjugate: Convenience function to conjugate a node.
  */
  template<typename T> __HOST_DEVICE__ void conjugate(T& x){ IvyNodeSelfRelations<T>::conjugate(x); }
  /*
  is_differentiable: Convenience function to check if a node is differentiable.
  */
  template<typename T> __HOST_DEVICE__ constexpr bool is_differentiable(T const& x){ return IvyNodeSelfRelations<T>::is_differentiable(x); }
  // Partial specializations
  template<typename T> struct IvyNodeSelfRelations<T*>{
    static constexpr bool is_conjugatable = is_conjugatable<T>;
    static __HOST_DEVICE__ void conjugate(T*& x){ conjugate(*x); }
    static __HOST_DEVICE__ bool is_differentiable(T* const& x){ return is_differentiable(*x); }
  };
  template<typename T, std_mem::IvyPointerType IPT> struct IvyNodeSelfRelations<std_mem::IvyUnifiedPtr<T, IPT>>{
    static constexpr bool is_conjugatable = is_conjugatable<T>;
    static __HOST_DEVICE__ void conjugate(std_mem::IvyUnifiedPtr<T, IPT>& x){ conjugate(*x); }
    static __HOST_DEVICE__ bool is_differentiable(std_mem::IvyUnifiedPtr<T, IPT> const& x){ return is_differentiable(*x); }
  };

  /*
  IvyNodeBinaryRelations:
  Any node has binary relations with other nodes.
  Because CUDA does not support RTTI, we need to implement these properties in a separate class and check them through static 'polymorphism'.
  The idea is to provide partial specializations when needed in order to implement these properties.

  IvyNodeBinaryRelations::depends_on: Check if a function depends on a variable. By default, no function depends on any variable.
  */
  template<typename T, typename U> struct IvyNodeBinaryRelations{
    static __HOST_DEVICE__ bool depends_on(T const& fcn, U* var){ return (std_mem::addressof(fcn) == var); }
  };
  // Partial specializations for IvyNodeBinaryRelations
  template<typename T, typename U> struct IvyNodeBinaryRelations<T*, U>{
    static __HOST_DEVICE__ bool depends_on(T* const& fcn, U* var){
      return fcn && IvyNodeBinaryRelations<T, U>::depends_on(*fcn, var);
    }
  };
  template<typename T, typename U, std_mem::IvyPointerType IPU> struct IvyNodeBinaryRelations<T, std_mem::IvyUnifiedPtr<U, IPU>>{
    static __HOST_DEVICE__ bool depends_on(T const& fcn, std_mem::IvyUnifiedPtr<U, IPU> const& var){
      return var && IvyNodeBinaryRelations<T, U>::depends_on(fcn, var.get());
    }
  };
  template<typename T, typename U, std_mem::IvyPointerType IPT> struct IvyNodeBinaryRelations<std_mem::IvyUnifiedPtr<T, IPT>, U>{
    static __HOST_DEVICE__ bool depends_on(std_mem::IvyUnifiedPtr<T, IPT> const& fcn, U* var){
      return fcn && IvyNodeBinaryRelations<T, U>::depends_on(*fcn, var);
    }
  };
  template<typename T, typename U, std_mem::IvyPointerType IPT, std_mem::IvyPointerType IPU>
  struct IvyNodeBinaryRelations<std_mem::IvyUnifiedPtr<T, IPT>, std_mem::IvyUnifiedPtr<U, IPU>>{
    static __HOST_DEVICE__ bool depends_on(std_mem::IvyUnifiedPtr<T, IPT> const& fcn, std_mem::IvyUnifiedPtr<U, IPU> const& var){
      return fcn && var && IvyNodeBinaryRelations<T, U>::depends_on(*fcn, var.get());
    }
  };

  /*
  depends_on: Convenience function to check if a function depends on a variable.
  */
  template<typename T, typename U, ENABLE_IF_BOOL(std_ttraits::is_pointer_v<U>)>
  __HOST_DEVICE__ bool depends_on(T const& fcn, U const& var){
    return IvyNodeBinaryRelations<T, std_ttraits::remove_reference_t<decltype(*var)>>::depends_on(fcn, var);
  }
  template<typename T, typename U, ENABLE_IF_BOOL(!std_ttraits::is_pointer_v<U>)>
  __HOST_DEVICE__ bool depends_on(T const& fcn, U const& var){
    return IvyNodeBinaryRelations<T, U>::depends_on(fcn, var);
  }
}


#endif
