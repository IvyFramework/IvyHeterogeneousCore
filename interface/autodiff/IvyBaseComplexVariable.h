#ifndef IVYBASECOMPLEXVARIABLE_H
#define IVYBASECOMPLEXVARIABLE_H


#include "std_ivy/IvyUtility.h"
#include "autodiff/IvyBaseNode.h"


class IvyBaseComplexVariable : public IvyBaseNode{
public:
  __CUDA_HOST_DEVICE__ IvyBaseComplexVariable(){}
  __CUDA_HOST_DEVICE__ IvyBaseComplexVariable(IvyBaseComplexVariable const& other) : IvyBaseNode(other){}
  __CUDA_HOST_DEVICE__ IvyBaseComplexVariable(IvyBaseComplexVariable const&& other) : IvyBaseNode(std_util::move(other)){}
  virtual __CUDA_HOST_DEVICE__ ~IvyBaseComplexVariable(){}
};


#endif
