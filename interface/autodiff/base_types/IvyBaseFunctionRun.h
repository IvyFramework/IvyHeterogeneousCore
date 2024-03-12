#ifndef IVYBASEFUNCTIONRUN_H
#define IVYBASEFUNCTIONRUN_H


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyLimits.h"
#include "std_ivy/IvyTypeTraits.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/IvyTensor.h"


namespace primary_fcn_run{
  template<typename T, ENABLE_IF_BOOL(std_ttraits::is_arithmetic_v<T> || IvyMath::is_real_valued_v<T> || IvyMath::is_complex_valued_v<T>)>
  __CUDA_HOST_DEVICE__ void run(T& fcn);
  template<typename T, ENABLE_IF_BOOL(IvyMath::is_function_valued_v<T>)> __CUDA_HOST__ void run(T& fcn);
  template<typename T, ENABLE_IF_BOOL(IvyMath::is_tensor_valued_v<T>)> __CUDA_HOST__ void run(T& fcn);
  template<typename T, ENABLE_IF_BOOL(IvyMath::is_pointer_v<T>)> __CUDA_HOST__ void run(T& fcn);

  template<typename T, ENABLE_IF_BOOL_IMPL(std_ttraits::is_arithmetic_v<T> || IvyMath::is_real_valued_v<T> || IvyMath::is_complex_valued_v<T>)>
  __CUDA_HOST__ void run(T& fcn){}
  template<typename T, ENABLE_IF_BOOL_IMPL(IvyMath::is_function_valued_v<T>)> __CUDA_HOST__ void run(T& fcn){ fcn.run(); }
  template<typename T, ENABLE_IF_BOOL_IMPL(IvyMath::is_tensor_valued_v<T>)> __CUDA_HOST__ void run(T& fcn){
    for (IvyTensorDim_t i=0; i<fcn.num_elements(); ++i) primary_fcn_run::run(fcn.data()[i]);
  }
  template<typename T, ENABLE_IF_BOOL_IMPL(IvyMath::is_pointer_v<T>)> __CUDA_HOST__ void run(T& fcn){ primary_fcn_run::run(*fcn); }
};

// Shorthand to run functions and other objects
#define run_fcn(fcn) primary_fcn_run::run(fcn)


#endif
