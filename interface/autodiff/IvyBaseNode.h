#ifndef IVYBASENODE_H
#define IVYBASENODE_H


/*
This is the base class of a node in the computation tree.
A node could be any variable, function, tensor etc.
*/


#include "config/IvyCompilerConfig.h"
#include "autodiff/IvyThreadSafePtr.h"


/*
IvyNodeSelfRelations:
Any node has dependence and differentiability properties.
Because CUDA does not support RTTI, we need to implement these properties in a separate class and check them through static 'polymorphism'.
The idea is to provide partial specializations when needed in order to implement these properties.

IvyNodeSelfRelations::is_differentiable: Check if this node is differentiable. By default, no node is differentiable.
IvyNodeSelfRelations::is_conjugatable: By default, no node is conjugatable.
*/
template<typename T> struct IvyNodeSelfRelations{
  static __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(T const& x){ return false; }
  static __CUDA_HOST_DEVICE__ void conjugate(T& x){}
  static constexpr bool is_conjugatable = false;
};
/*
is_differentiable: Convenience function to check if a node is differentiable.
*/
template<typename T> __CUDA_HOST_DEVICE__ constexpr bool is_differentiable(T const& x){ return IvyNodeSelfRelations<T>::is_differentiable(x); }
/*
conjugate: Convenience function to conjugate a node.
*/
template<typename T> __CUDA_HOST_DEVICE__ void conjugate(T const& x){ IvyNodeSelfRelations<T>::conjugate(x); }
/*
is_conjugatable: Convenience function to check if a node is conjugatable.
*/
template<typename T> inline constexpr bool is_conjugatable = IvyNodeSelfRelations<T>::is_conjugatable;

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
template<typename T, typename U> struct IvyNodeBinaryRelations<T, U*>{
  static __CUDA_HOST_DEVICE__ bool depends_on(T const& fcn, U* const& var){ return (std_mem::addressof(fcn) == var); }
};
template<typename T, typename U> struct IvyNodeBinaryRelations<T*, U>{
  static __CUDA_HOST_DEVICE__ bool depends_on(T* const& fcn, U const& var){ return (fcn == std_mem::addressof(var)); }
};
template<typename T, typename U> struct IvyNodeBinaryRelations<T*, U*>{
  static __CUDA_HOST_DEVICE__ bool depends_on(T* const& fcn, U* const& var){ return (fcn == var); }
};
/*
depends_on: Convenience function to check if a function depends on a variable.
*/
template<typename T, typename U> __CUDA_HOST_DEVICE__ bool depends_on(T const& fcn, U const& var){ return IvyNodeBinaryRelations<T, U>::depends_on(fcn, var); }

/*
IvyBaseNode:
Base class of a node in the computation tree.
This is an empty class just so that we can detect the object to be a node by simply inheriting from this class.
*/
struct IvyBaseNode{};

using IvyBaseNodePtr_t = IvyThreadSafePtr_t<IvyBaseNode>;


#endif
