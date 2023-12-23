#ifndef IVYCUDACONFIG_H
#define IVYCUDACONFIG_H


#include "IvyCudaFlags.h"


#ifdef __USE_CUDA__

namespace IvyCudaConfig{
  typedef unsigned long long int IvyBlockThread_t;

  __CUDA_HOST__ void set_max_num_GPU_blocks(unsigned int n);
  __CUDA_HOST__ void set_max_num_GPU_threads_per_block(unsigned int n);

  __CUDA_HOST_DEVICE__ unsigned int const& get_max_num_GPU_blocks();
  __CUDA_HOST_DEVICE__ unsigned int const& get_max_num_GPU_threads_per_block();

  __CUDA_HOST_DEVICE__ bool check_GPU_usable(unsigned int& nreq_blocks, unsigned int& nreq_threads_per_block, unsigned int n);

  __CUDA_HOST_DEVICE__ cudaStream_t get_gpu_stream_from_pointer(cudaStream_t* ptr){
    return (ptr ? *ptr : cudaStreamLegacy);
  }

}

// Shorthand usage for IvyBlockThread_t outside of the IvyCudaConfig namespace
using IvyBlockThread_t = IvyCudaConfig::IvyBlockThread_t;

#endif


#endif
