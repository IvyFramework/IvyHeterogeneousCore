#ifndef IVYCUDACONFIG_H
#define IVYCUDACONFIG_H

/**
 * @file IvyCudaConfig.h
 * @brief CUDA runtime capability helpers and launch-dimension planning utilities.
 */


#include "config/IvyCudaFlags.h"

#ifdef __USE_CUDA__

#include "std_ivy/algorithm/IvyMinMax.h"
#include "std_ivy/IvyLimits.h"
#include "std_ivy/IvyCstdio.h"
#include "stream/IvyStream.h"


namespace IvyCudaConfig{
  /** @brief CUDA device ordinal type. */
  using IvyDeviceNum_t = int;
  /** @brief Unsigned thread-count type used for launch dimensions. */
  using IvyBlockThread_t = unsigned int;
  /** @brief Signed thread-count type used for launch-dimension calculations. */
  using IvyBlockThread_signed_t = int;
  /** @brief CUDA launch-dimension type (`dim3`). */
  using IvyBlockThreadDim_t = dim3;

  /**
   * @brief Determine whether the GPU can be used for a 1D workload and propose launch dimensions.
   * @param nreq_blocks Output block dimensions.
   * @param nreq_threads_per_block Output thread dimensions per block.
   * @param n Number of work items.
   * @return True when a valid launch configuration is found.
   */
  __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool check_GPU_usable(IvyBlockThreadDim_t& nreq_blocks, IvyBlockThreadDim_t& nreq_threads_per_block, unsigned long long int n){
    nreq_blocks.x = nreq_blocks.y = nreq_blocks.z = 0;
    nreq_threads_per_block.x = nreq_threads_per_block.y = nreq_threads_per_block.z = 0;
    if (n==0) return false;

    IvyDeviceNum_t device_num = 0;
    if (!__CHECK_SUCCESS__(cudaGetDevice(&device_num))) return false;

    bool res = true;
    IvyBlockThread_signed_t max_threads_per_block_x_=0, max_threads_per_block_y_=0, max_threads_per_block_z_=0;
    IvyBlockThread_signed_t max_blocks_x_=0, max_blocks_y_=0, max_blocks_z_=0;
    IvyBlockThread_signed_t max_threads_per_block_sc_=0;
    IvyBlockThread_signed_t warp_size_ = 0;
    if (
      __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_sc_, cudaDevAttrMaxThreadsPerBlock, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_x_, cudaDevAttrMaxBlockDimX, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_y_, cudaDevAttrMaxBlockDimY, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_threads_per_block_z_, cudaDevAttrMaxBlockDimZ, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_x_, cudaDevAttrMaxGridDimX, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_y_, cudaDevAttrMaxGridDimY, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&max_blocks_z_, cudaDevAttrMaxGridDimZ, device_num))
      && __CHECK_SUCCESS__(cudaDeviceGetAttribute(&warp_size_, cudaDevAttrWarpSize, device_num))
      && max_threads_per_block_sc_>0
      && max_threads_per_block_x_>0 && max_threads_per_block_y_>0 && max_threads_per_block_z_>0
      && max_blocks_x_>0 && max_blocks_y_>0 && max_blocks_z_>0
      && warp_size_>0
      ){
      nreq_threads_per_block.x = std_ivy::min(
        max_threads_per_block_sc_,
        __STATIC_CAST__(IvyBlockThread_signed_t, ((n - 1 + warp_size_)/warp_size_)*warp_size_)
      );
      nreq_threads_per_block.y = nreq_threads_per_block.z = 1;
      unsigned long long int nblocks_needed = __STATIC_CAST__(
        unsigned long long int,
        (n - 1 + nreq_threads_per_block.x)/nreq_threads_per_block.x
      );
      unsigned long long int const max_blocks_ = max_blocks_x_ * max_blocks_y_ * max_blocks_z_;
      if (nblocks_needed<=max_blocks_x_){
        nreq_blocks.x = nblocks_needed;
        nreq_blocks.y = nreq_blocks.z = 1;
      }
      else if (nblocks_needed<=max_blocks_x_ * max_blocks_y_){
        nreq_blocks.x = max_blocks_x_;
        nreq_blocks.y = std_ivy::min(
          __STATIC_CAST__(unsigned long long int, (nblocks_needed - 1 + max_blocks_x_)/max_blocks_x_),
          __STATIC_CAST__(unsigned long long int, max_blocks_y_)
        );
      }
      else if (nblocks_needed<=max_blocks_){
        nreq_blocks.x = max_blocks_x_;
        nreq_blocks.y = max_blocks_y_;
        nreq_blocks.z = std_ivy::min(
          __STATIC_CAST__(unsigned long long int, (nblocks_needed - 1 + max_blocks_x_*max_blocks_y_)/(max_blocks_x_*max_blocks_y_)),
          __STATIC_CAST__(unsigned long long int, max_blocks_z_)
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

  /**
   * @brief Resolve a raw CUDA stream from an optional pointer.
   * @param ptr Optional pointer to stream handle.
   * @return Dereferenced stream if present; otherwise the global default stream.
   */
  __HOST_DEVICE__ cudaStream_t get_GPU_stream_from_pointer(cudaStream_t* ptr){
    return (ptr ? *ptr : GlobalGPUStreamRaw);
  }

}

// Shorthand usage for IvyBlockThread_t outside of the IvyCudaConfig namespace
using IvyBlockThread_t = IvyCudaConfig::IvyBlockThread_t;
using IvyBlockThread_signed_t = IvyCudaConfig::IvyBlockThread_signed_t;
using IvyBlockThreadDim_t = IvyCudaConfig::IvyBlockThreadDim_t;

#endif


#endif
