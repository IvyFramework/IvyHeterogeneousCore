#ifndef IVYCALLEXCEPTION_H
#define IVYCALLEXCEPTION_H


#include "config/IvyCudaFlags.h"


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"


#define __CUDA_CHECK_OR_EXIT_WITH_ERROR__(CALL) \
{ \
  auto __cuda_error_code__ = CALL; \
  if (__cuda_error_code__ != cudaSuccess){ \
    __PRINT_ERROR__("Call '%s' at %s::%d failed with error code %s.\n", #CALL, __FILE__, __LINE__, cudaGetErrorString(__cuda_error_code__)); \
    assert(false); \
  } \
}

#define __CUDA_CHECK_AND_WARN_WITH_ERROR__(CALL) \
{ \
  auto __cuda_error_code__ = CALL; \
  if (__cuda_error_code__ != cudaSuccess) \
    __PRINT_ERROR__("Call '%s' at %s::%d failed with error code %s.\n", #CALL, __FILE__, __LINE__, cudaGetErrorString(__cuda_error_code__)); \
}

template<typename... Args> __CUDA_HOST_DEVICE__ bool cuda_check(cudaError_t(*call)(Args...), Args... args){
  return (call(args...) == cudaSuccess);
}
template<typename... Args> __CUDA_HOST_DEVICE__ bool cuda_check_and_warn(const char* fn, unsigned int fl, cudaError_t(*call)(Args...), Args... args){
  auto __cuda_error_code__ = call(args...);
  if (__cuda_error_code__ != cudaSuccess){
    __PRINT_ERROR__("CUDA call at %s::%d failed with error code %s.\n", fn, fl, cudaGetErrorString(__cuda_error_code__));
    return false;
  }
  return true;
}
template<typename... Args> __CUDA_HOST_DEVICE__ bool cuda_check_or_exit(const char* fn, unsigned int fl, cudaError_t (*call)(Args...), Args... args){
  bool res = cuda_check_and_warn(fn, fl, call, args...);
  assert(res);
  return res;
}

template<typename Fcn, typename DG, typename DB, typename SMem, typename Stream, typename... Args>
__CUDA_HOST_DEVICE__ bool cuda_check_kernel(
  Fcn fcn, DG dg, DB db, SMem sm, Stream stream, Args... args
){
  fcn<<<dg, db, sm, stream>>>(args...);
  auto const err_code = cudaPeekAtLastError();
#ifdef __CUDA_DEBUG__
  cudaDeviceSynchronize();
#endif
  return (err_code == cudaSuccess);
}

#endif


#endif
