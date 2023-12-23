#ifndef IVYMEMORYHELPERS_H
#define IVYMEMORYHELPERS_H

#include "IvyCompilerFlags.h"
#include "IvyCudaConfig.h"
#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif
#include "IvyException.h"


namespace IvyMemoryHelpers{
  typedef unsigned long long int size_t;
  typedef long long int ptrdiff_t;

  template<typename T> __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    IvyMemoryHelpers::size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem = false
    , cudaStream_t stream = cudaStreamLegacy
#endif
  );

  template<typename T> __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    IvyMemoryHelpers::size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem = false
    , cudaStream_t stream = cudaStreamLegacy
#endif
  );

#ifdef __USE_CUDA__
  template<typename T> __CUDA_HOST__ bool transfer_memory(T*& tgt, T* const& src, IvyMemoryHelpers::size_t n, bool device_to_host, cudaStream_t stream = cudaStreamLegacy);

  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, IvyMemoryHelpers::size_t n);
#endif

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    IvyMemoryHelpers::size_t n_tgt_init, IvyMemoryHelpers::size_t n_src
#ifdef __USE_CUDA__
    , cudaStream_t stream = cudaStreamLegacy
#endif
  );
}


namespace IvyMemoryHelpers{
  template<typename T> __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    IvyMemoryHelpers::size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem
    , cudaStream_t stream
#endif
  ){
    if (n==0 || data) return false;
#ifdef __USE_CUDA__
    if (use_cuda_device_mem){
#ifndef __CUDA_DEVICE_CODE__
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc((void**) &data, n*sizeof(T)));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocAsync((void**) &data, n*sizeof(T), stream));
      }
#else
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc((void**) &data, n*sizeof(T)));
#endif
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
    , bool use_cuda_device_mem
    , cudaStream_t stream
#endif
  ){
    if (!data) return true;
    if (n==0) return false;
#ifdef __USE_CUDA__
    if (use_cuda_device_mem){
#ifndef __CUDA_DEVICE_CODE__
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(data));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFreeAsync(data, stream));
      }
#else
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(data));
#endif
    }
    else
#endif
    {
      if (n==1) delete data;
      else delete[] data;
    }
    data = nullptr;
    return true;
  }

#ifdef __USE_CUDA__
  template<typename T> __CUDA_HOST__ bool transfer_memory(T*& tgt, T* const& src, IvyMemoryHelpers::size_t n, bool device_to_host, cudaStream_t stream){
    if (!tgt || !src) return false;
    if (device_to_host){
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(tgt, src, n*sizeof(T), cudaMemcpyDeviceToHost));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpyAsync(tgt, src, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
      }
    }
    else{
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(tgt, src, n*sizeof(T), cudaMemcpyHostToDevice));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpyAsync(tgt, src, n*sizeof(T), cudaMemcpyHostToDevice, stream));
      }
    }
    return true;
  }

  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, IvyMemoryHelpers::size_t n){
    IvyMemoryHelpers::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<n) target[i] = source[i];
  }
#endif

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    IvyMemoryHelpers::size_t n_tgt_init, IvyMemoryHelpers::size_t n_src
#ifdef __USE_CUDA__
    , cudaStream_t stream
#endif
  ){
    bool res = true;
    res &= free_memory(
      target, n_tgt_init
#ifdef __USE_CUDA__
      , false, stream
#endif
    );
    res &= allocate_memory(
      target, n_src
#ifdef __USE_CUDA__
      , false, stream
#endif
    );
    if (res){
#ifdef __USE_CUDA__
      unsigned int nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        U* d_source = nullptr;
        T* d_target = nullptr;
        res &= allocate_memory(d_source, n_src, true, stream);
        res &= allocate_memory(d_target, n_src, true, stream);
        res &= transfer_memory(d_source, source, n_src, false, stream);
        copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_src);
        res &= transfer_memory(target, d_target, n_src, true, stream);
        free_memory(d_target, n_src, true, stream);
        free_memory(d_source, n_src, true, stream);
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
