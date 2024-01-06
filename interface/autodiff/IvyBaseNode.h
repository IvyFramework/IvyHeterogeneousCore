#ifndef IVYBASENODE_H
#define IVYBASENODE_H


/*
This is the base class of a node in the computation tree.
A node could be any variable, function, tensor etc.
*/


#include "config/IvyCompilerConfig.h"
#include "autodiff/IvyThreadSafePtr.h"


class IvyBaseNode{
public:
  /*
  Empty default constructor
  */
  __CUDA_HOST_DEVICE__ IvyBaseNode(){}

  /*
  Empty virtual destructor
  */
  virtual __CUDA_HOST_DEVICE__ ~IvyBaseNode(){}

  /*
  Check if this node is differentiable.
  By default, no node is differentiable.
  */
  virtual __CUDA_HOST_DEVICE__ bool is_differentiable() const{ return false; }

  /*
  Check if this node depends on another node.
  Note that in order to have this base class as lightweight in memory as possible, we do not have a vector of dependents as a class mamber.
  */
  virtual __CUDA_HOST_DEVICE__ bool depends_on(IvyBaseNode const& node) const{ return (this == &node); }

  /*
  Check if this node is conjugatable.
  */
  virtual __CUDA_HOST_DEVICE__ __CPP_VIRTUAL_CONSTEXPR__ bool is_conjugatable() const{ return false; }
  virtual __CUDA_HOST_DEVICE__ void conjugate(){} // Conjugate this node
};

using IvyBaseNodePtr_t = IvyThreadSafePtr_t<IvyBaseNode>;


#endif
