#ifndef IVYCUDAEXCEPTION_H
#define IVYCUDAEXCEPTION_H


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif

#include "config/IvyCudaFlags.h"
#include "IvyErrorTypes.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"


#define __CHECK_OR_EXIT_WITH_ERROR__(CALL) \
{ \
  auto __error_code__ = CALL; \
  if (__error_code__ != IvySuccess){ \
    __PRINT_ERROR__("*** ERROR: Call ***\n*** '%s' ***\n*** at ***\n*** %s::%d ***\n*** failed with error '%s'. ***\n", #CALL, __FILE__, __LINE__, IvyGetErrorString(__error_code__)); \
    assert(false); \
  } \
}

#define __CHECK_AND_WARN_WITH_ERROR__(CALL) \
{ \
  auto __error_code__ = CALL; \
  if (__error_code__ != IvySuccess) \
    __PRINT_ERROR__("*** ERROR: Call ***\n*** '%s' ***\n*** at ***\n*** %s::%d ***\n*** failed with error '%s'. ***\n", #CALL, __FILE__, __LINE__, IvyGetErrorString(__error_code__)); \
}

template<typename... Args> __HOST_DEVICE__ bool cuda_check(IvyError_t(*call)(Args...), Args&&... args){
  return (call(args...) == IvySuccess);
}
template<typename... Args> __HOST_DEVICE__ bool cuda_check_and_warn(const char* fn, unsigned int fl, IvyError_t(*call)(Args...), Args&&... args){
  auto __error_code__ = call(args...);
  if (__error_code__ != IvySuccess){
    __PRINT_ERROR__("*** ERROR: Call at %s::%d failed with error '%s'. ***\n", fn, fl, IvyGetErrorString(__error_code__));
    return false;
  }
  return true;
}
template<typename... Args> __HOST_DEVICE__ bool cuda_check_or_exit(const char* fn, unsigned int fl, IvyError_t(*call)(Args...), Args&&... args){
  bool res = cuda_check_and_warn(fn, fl, call, args...);
  assert(res);
  return res;
}

#if defined(__USE_CUDA__) && defined(__CUDA_DEBUG__) && DEVICE_CODE==DEVICE_CODE_HOST
#define __CHECK_KERNEL_AND_WARN_WITH_ERROR__(CALL) \
{ \
  CALL; \
  /*auto __error_code__ = cudaPeekAtLastError();*/ \
  auto __error_code__ = cudaGetLastError(); \
  cudaDeviceSynchronize(); \
  if (__error_code__ != IvySuccess) \
    __PRINT_ERROR__("*** ERROR: Kernel call ***\n*** '%s' ***\n*** at ***\n*** %s::%d ***\n*** failed with error '%s'. ***\n", #CALL, __FILE__, __LINE__, IvyGetErrorString(__error_code__)); \
}
#else
#define __CHECK_KERNEL_AND_WARN_WITH_ERROR__(CALL) CALL;
#endif


#endif
