#ifndef IVYCUDACONFIG_H
#define IVYCUDACONFIG_H


#include "config/IvyCudaFlags.h"


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/algorithm/IvyMinMax.h"


namespace IvyCudaConfig{
  using IvyBlockThread_t = unsigned int;
  using IvyBlockThread_signed_t = int;

  __CUDA_MANAGED__ IvyBlockThread_t MAX_NUM_BLOCKS = 1;
  __CUDA_MANAGED__ IvyBlockThread_t MAX_NUM_THREADS_PER_BLOCK = 256;

  __CUDA_HOST__ void set_max_num_GPU_blocks(IvyBlockThread_t n){ MAX_NUM_BLOCKS = n; }
  __CUDA_HOST__ void set_max_num_GPU_threads_per_block(IvyBlockThread_t n){ MAX_NUM_THREADS_PER_BLOCK = n; }

  __CUDA_HOST_DEVICE__ IvyBlockThread_t const& get_max_num_GPU_blocks(){ return MAX_NUM_BLOCKS; }
  __CUDA_HOST_DEVICE__ IvyBlockThread_t const& get_max_num_GPU_threads_per_block(){ return MAX_NUM_THREADS_PER_BLOCK; }

  __CUDA_HOST_DEVICE__ bool check_GPU_usable(IvyBlockThread_t& nreq_blocks, IvyBlockThread_t& nreq_threads_per_block, unsigned long long int n){
    nreq_blocks = 0;
    nreq_threads_per_block = 0;

    IvyBlockThread_signed_t max_threads_per_block_x_=0, max_threads_per_block_y_=0, max_threads_per_block_z_=0;
    IvyBlockThread_signed_t max_blocks_x_=0, max_blocks_y_=0, max_blocks_z_=0;
    if (
      __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_x_, cudaDevAttrMaxBlockDimX, 0))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_y_, cudaDevAttrMaxBlockDimY, 0))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_z_, cudaDevAttrMaxBlockDimZ, 0))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_x_, cudaDevAttrMaxGridDimX, 0))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_y_, cudaDevAttrMaxGridDimY, 0))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_z_, cudaDevAttrMaxGridDimZ, 0))
      ){
      IvyBlockThread_signed_t max_threads_per_block_ = max_threads_per_block_x_ * max_threads_per_block_y_ * max_threads_per_block_z_;
      IvyBlockThread_signed_t max_blocks_ = max_blocks_x_ * max_blocks_y_ * max_blocks_z_;
      if (max_blocks_>-1) nreq_blocks = std_ivy::min(get_max_num_GPU_blocks(), static_cast<IvyBlockThread_t>(max_blocks_));
      if (max_threads_per_block_>-1) nreq_threads_per_block = std_ivy::min(get_max_num_GPU_threads_per_block(), static_cast<IvyBlockThread_t>(max_threads_per_block_));
    }
    if (nreq_blocks<=1 || nreq_threads_per_block<=1){
      nreq_blocks = nreq_threads_per_block = 0;
      return false;
    }
    if (n<nreq_blocks*nreq_threads_per_block){
      if (n<=nreq_threads_per_block){
        nreq_blocks = 1;
        nreq_threads_per_block = n;
      }
      else{
        nreq_blocks = n / nreq_threads_per_block + 1;
        nreq_threads_per_block = n / nreq_blocks + 1;
      }
    }
    return true;
  }

  __CUDA_HOST_DEVICE__ cudaStream_t get_GPU_stream_from_pointer(cudaStream_t* ptr){
    return (ptr ? *ptr : cudaStreamLegacy);
  }

}

// Shorthand usage for IvyBlockThread_t outside of the IvyCudaConfig namespace
using IvyBlockThread_t = IvyCudaConfig::IvyBlockThread_t;

#endif


#endif
