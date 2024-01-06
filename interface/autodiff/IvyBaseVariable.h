#ifndef IVYBASEVARIABLE_H
#define IVYBASEVARIABLE_H


#include "std_ivy/IvyUtility.h"
#include "autodiff/IvyBaseNode.h"


class IvyBaseVariable : public IvyBaseNode{
public:
  __CUDA_HOST_DEVICE__ IvyBaseVariable(){}
  __CUDA_HOST_DEVICE__ IvyBaseVariable(IvyBaseVariable const& other) : IvyBaseNode(other){}
  __CUDA_HOST_DEVICE__ IvyBaseVariable(IvyBaseVariable const&& other) : IvyBaseNode(std_util::move(other)){}
  virtual __CUDA_HOST_DEVICE__ ~IvyBaseVariable(){}
};


#endif
