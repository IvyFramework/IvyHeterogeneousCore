#ifndef IVYCUDACONFIG_H
#define IVYCUDACONFIG_H


#include "config/IvyCudaFlags.h"

#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/algorithm/IvyMinMax.h"
#include "std_ivy/IvyLimits.h"
#include "std_ivy/IvyCstdio.h"
#include "stream/IvyStream.h"


namespace IvyCudaConfig{
  using IvyDeviceNum_t = int;
  using IvyBlockThread_t = unsigned int;
  using IvyBlockThread_signed_t = int;
  using IvyBlockThreadDim_t = dim3;

  __INLINE_VAR__ __CUDA_MANAGED__ IvyBlockThread_t MAX_NUM_BLOCKS = 1;
  __INLINE_VAR__ __CUDA_MANAGED__ IvyBlockThread_t MAX_NUM_THREADS_PER_BLOCK = 256;

  __CUDA_HOST__ void set_max_num_GPU_blocks(IvyBlockThread_t n){ MAX_NUM_BLOCKS = n; }
  __CUDA_HOST__ void set_max_num_GPU_threads_per_block(IvyBlockThread_t n){ MAX_NUM_THREADS_PER_BLOCK = n; }

  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ IvyBlockThread_t const& get_max_num_GPU_blocks(){ return MAX_NUM_BLOCKS; }
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ IvyBlockThread_t const& get_max_num_GPU_threads_per_block(){ return MAX_NUM_THREADS_PER_BLOCK; }

  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool check_GPU_usable(IvyBlockThreadDim_t& nreq_blocks, IvyBlockThreadDim_t& nreq_threads_per_block, unsigned long long int n){
    nreq_blocks = 0;
    nreq_threads_per_block = 0;

    IvyDeviceNum_t device_num = 0;
    if (!__CUDA_CHECK_SUCCESS__(cudaGetDevice(&device_num))) return false;

    bool res = true;
    IvyBlockThread_signed_t max_threads_per_block_x_=0, max_threads_per_block_y_=0, max_threads_per_block_z_=0;
    IvyBlockThread_signed_t max_blocks_x_=0, max_blocks_y_=0, max_blocks_z_=0;
    IvyBlockThread_signed_t max_threads_per_block_sc_=0;
    IvyBlockThread_signed_t warp_size_ = 0;
    if (
      __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_sc_, cudaDevAttrMaxThreadsPerBlock, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_x_, cudaDevAttrMaxBlockDimX, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_y_, cudaDevAttrMaxBlockDimY, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_z_, cudaDevAttrMaxBlockDimZ, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_x_, cudaDevAttrMaxGridDimX, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_y_, cudaDevAttrMaxGridDimY, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_z_, cudaDevAttrMaxGridDimZ, device_num))
      && __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&warp_size_, cudaDevAttrWarpSize, device_num))
      && max_threads_per_block_sc_>0
      && max_threads_per_block_x_>0 && max_threads_per_block_y_>0 && max_threads_per_block_z_>0
      && max_blocks_x_>0 && max_blocks_y_>0 && max_blocks_z_>0
      && warp_size_>0
      ){
      nreq_threads_per_block.x = std_ivy::min(
        max_threads_per_block_sc_,
        static_cast<IvyBlockThread_signed_t>((n+(warp_size_+1)/2)/warp_size_)
      );
      nreq_threads_per_block.y = nreq_threads_per_block.z = 1;
      unsigned long long int nblocks_needed = (n + (nreq_threads_per_block.x+1)/2)/nreq_threads_per_block.x;
      unsigned long long int const max_blocks_ = max_blocks_x_ * max_blocks_y_ * max_blocks_z_;
      if (nblocks_needed<=max_blocks_x_){
        nreq_blocks.x = nblocks_needed;
        nreq_blocks.y = nreq_blocks.z = 1;
      }
      else if (nblocks_needed<=max_blocks_x_ * max_blocks_y_){
        nreq_blocks.x = max_blocks_x_;
        nreq_blocks.y = std_ivy::min(
          static_cast<unsigned long long int>((nblocks_needed + (max_blocks_x_+1)/2)/max_blocks_x_),
          static_cast<unsigned long long int>(max_blocks_y_)
          );
      }
      else if (nblocks_needed<=max_blocks_){
        nreq_blocks.x = max_blocks_x_;
        nreq_blocks.y = max_blocks_y_;
        nreq_blocks.z = std_ivy::min(
          static_cast<unsigned long long int>((nblocks_needed + (max_blocks_x_*max_blocks_y_+1)/2)/(max_blocks_x_*max_blocks_y_)),
          static_cast<unsigned long long int>(max_blocks_z_)
          );
      }
      else res = false;
      if (
        nreq_blocks.x<=1 && nreq_blocks.y<=1 && nreq_blocks.z<=1
        &&
        nreq_threads_per_block.x<=1 && nreq_threads_per_block.y<=1 && nreq_threads_per_block.z<=1
        ) res = false;
    }
    else res = false;
    if (!res){
      nreq_blocks.x = nreq_blocks.y = nreq_blocks.z = 0;
      nreq_threads_per_block.x = nreq_threads_per_block.y = nreq_threads_per_block.z = 0;
    }
    //__PRINT_INFO__("check_GPU_usable: n=%llu, nreq_blocks=(%u,%u,%u), nreq_threads_per_block=(%u,%u,%u)\n", n, nreq_blocks.x, nreq_blocks.y, nreq_blocks.z, nreq_threads_per_block.x, nreq_threads_per_block.y, nreq_threads_per_block.z);
    return res;
  }

  __CUDA_HOST_DEVICE__ cudaStream_t get_GPU_stream_from_pointer(cudaStream_t* ptr){
    return (ptr ? *ptr : GlobalGPUStreamRaw);
  }

}

// Shorthand usage for IvyBlockThread_t outside of the IvyCudaConfig namespace
using IvyBlockThread_t = IvyCudaConfig::IvyBlockThread_t;
using IvyBlockThread_signed_t = IvyCudaConfig::IvyBlockThread_signed_t;
using IvyBlockThreadDim_t = IvyCudaConfig::IvyBlockThreadDim_t;

#endif


#endif
