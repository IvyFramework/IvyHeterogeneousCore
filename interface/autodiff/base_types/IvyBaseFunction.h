#ifndef IVYBASEFUNCTION_H
#define IVYBASEFUNCTION_H


#include "std_ivy/IvyUtility.h"
#include "autodiff/IvyBaseNode.h"


class IvyBaseFunction : public IvyBaseNode{
protected:
  // Register outputs when you construct this function.
  virtual __CUDA_HOST__ void register_outputs() = 0;

  // Run any inputs before running this function.
  virtual __CUDA_HOST__ void run_components() = 0;

  // The run_impl function does the computation.
  virtual __CUDA_HOST__ void run_impl() = 0;

  virtual __CUDA_HOST__ void run_outputs() = 0;

public:
  __CUDA_HOST__ IvyBaseFunction(){}
  __CUDA_HOST__ IvyBaseFunction(IvyBaseFunction const& other) : IvyBaseNode(other){}
  __CUDA_HOST__ IvyBaseFunction(IvyBaseFunction const&& other) : IvyBaseNode(std_util::move(other)){}
  virtual __CUDA_HOST__ ~IvyBaseFunction(){}

  // The run function prepares to execute run_impl.
  // The distinction is to allow for generalization to block values and gradients from being called later on.
  void __CUDA_HOST__ run(){
    this->run_impl();
    this->run_outputs();
  }
};

using IvyBaseFcnPtr_t = IvyThreadSafePtr_t<IvyBaseFunction>;


#endif
