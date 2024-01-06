#ifndef IVYBASETENSOR_H
#define IVYBASETENSOR_H


#include "std_ivy/IvyUtility.h"
#include "autodiff/IvyBaseNode.h"


class IvyBaseTensor : public IvyBaseNode{
public:
  __CUDA_HOST_DEVICE__ IvyBaseTensor(){}
  __CUDA_HOST_DEVICE__ IvyBaseTensor(IvyBaseTensor const& other) : IvyBaseNode(other){}
  __CUDA_HOST_DEVICE__ IvyBaseTensor(IvyBaseTensor const&& other) : IvyBaseNode(std_util::move(other)){}
  virtual __CUDA_HOST_DEVICE__ ~IvyBaseTensor(){}
};


#endif
