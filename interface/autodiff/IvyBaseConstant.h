#ifndef IVYBASECONSTANT_H
#define IVYBASECONSTANT_H


#include "std_ivy/IvyUtility.h"
#include "autodiff/IvyBaseNode.h"


class IvyBaseConstant : public IvyBaseNode{
public:
  __CUDA_HOST_DEVICE__ IvyBaseConstant(){}
  __CUDA_HOST_DEVICE__ IvyBaseConstant(IvyBaseConstant const& other) : IvyBaseNode(other){}
  __CUDA_HOST_DEVICE__ IvyBaseConstant(IvyBaseConstant const&& other) : IvyBaseNode(std_util::move(other)){}
  virtual __CUDA_HOST_DEVICE__ ~IvyBaseConstant(){}
};


#endif
