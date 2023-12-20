#ifndef IVYMEMORYHELPERS_H
#define IVYMEMORYHELPERS_H

#include "IvyCompilerFlags.h"
#include "IvyCudaConfig.h"
#include "IvyException.h"
#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif


namespace IvyMemoryHelpers{
  typedef unsigned long long int size_t;
  typedef long long int ptrdiff_t;

  template<typename T> __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    IvyMemoryHelpers::size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem = false
#endif
  ){
    if (n==0 || data) return false;
#ifdef __USE_CUDA__
    if (use_cuda_device_mem){
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc((void**) &data, n*sizeof(T)));
      return true;
    }
#endif
    if (n==1) data = new T;
    else data = new T[n];
    return true;
  }
  template<typename T> __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    IvyMemoryHelpers::size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem = false
#endif
  ){
    if (!data) return true;
    if (n==0) return false;
#ifdef __USE_CUDA__
    if (use_cuda_device_mem){
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(data));
      return true;
    }
#endif
    if (n==1) delete data;
    else delete[] data;
    data = nullptr;
    return true;
  }

#ifdef __USE_CUDA__
  template<typename T> __CUDA_HOST__ bool transfer_memory(T*& tgt, T*& src, IvyMemoryHelpers::size_t n, bool device_to_host){
    if (!tgt || !src) return false;
    if (device_to_host){
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(tgt, src, n*sizeof(T), cudaMemcpyDeviceToHost));
    }
    else{
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(tgt, src, n*sizeof(T), cudaMemcpyHostToDevice));
    }
    return true;
  }

  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, IvyMemoryHelpers::size_t n){
    IvyMemoryHelpers::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n) target[i] = source[i];
  }
#endif
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data(T*& target, U* const& source, IvyMemoryHelpers::size_t n_tgt_init, IvyMemoryHelpers::size_t n_src){
    bool res = true;
    res &= free_memory(target, n_tgt_init);
    res &= allocate_memory(target, n_src);
    if (res){
#ifdef __USE_CUDA__
      unsigned int nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        U* d_source = nullptr;
        T* d_target = nullptr;
        res &= allocate_memory(d_source, n_src, true);
        res &= allocate_memory(d_target, n_src, true);
        res &= transfer_memory(d_source, source, n_src, false);
        copy_data_kernel<<<nreq_blocks, nreq_threads_per_block>>>(d_target, d_source, n_src);
        res &= transfer_memory(target, d_target, n_src, true);
        free_memory(d_target, n_src, true);
        free_memory(d_source, n_src, true);
      }
      else{
#else
      {
#endif
        for (IvyMemoryHelpers::size_t i=0; i<n_src; i++) target[i] = source[i];
      }
    }
    return res;
  }

}


#endif
