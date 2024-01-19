#ifndef IVYKERNELRUN_H
#define IVYKERNELRUN_H


#include "IvyBasicTypes.h"
#include "config/IvyConfig.h"
#include "std_ivy/IvyTypeTraits.h"


#ifdef __USE_CUDA__

/*
get_kernel_call_dims_1D/2D/3D: Gets the dimensions of the kernel call
corresponding to the blockIdx and threadIdx dimensions of the current thread:
- 1D is fully flattened.
- In 2D, the z dimension is folded into the y direction, and the x dimension is taken as is.
- The x, y, and z dimensions are taken as they are in 3D.
*/
__INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_kernel_call_dims_1D(IvyTypes::size_t& i){
  IvyTypes::size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
  IvyTypes::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  IvyTypes::size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  i = ix + iy * blockDim.x * gridDim.x + iz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
}
__INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_kernel_call_dims_2D(IvyTypes::size_t& i, IvyTypes::size_t& j){
  i = blockIdx.x * blockDim.x + threadIdx.x;
  IvyTypes::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
  IvyTypes::size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
  j = iy + iz * blockDim.y * gridDim.y;
}
__INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_kernel_call_dims_3D(IvyTypes::size_t& i, IvyTypes::size_t& j, IvyTypes::size_t& k){
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  k = blockIdx.z * blockDim.z + threadIdx.z;
}


/*
Calls could be structured such that Kernel_t::kernel is either a nonfactorizable function,
or it calls Kernel_t::kernel_unified_unit with a check that looks like 'if (i<n) kernel_unified_unit(i, n, args...);' and does nothing else.
While we do not check whether argument memory locations are arranged properly, in case they are and kernel_unified_unit calls are factorizable,
one can use the run_kernel struct below to run the kernel in parallel or in a loop, depending on whether the GPU is usable.
*/
DEFINE_HAS_CALL(kernel_unified_unit);
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_1D(Args... args){
  IvyTypes::size_t i = 0;
  get_kernel_call_dims_1D(i);
  Kernel_t::kernel(i, args...);
}
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_2D(Args... args){
  IvyTypes::size_t i = 0, j = 0;
  get_kernel_call_dims_2D(i, j);
  Kernel_t::kernel(i, j, args...);
}
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_3D(Args... args){
  IvyTypes::size_t i = 0, j = 0, k = 0;
  get_kernel_call_dims_3D(i, j, k);
  Kernel_t::kernel(i, j, k, args...);
}
template<typename Kernel_t, bool = has_call_kernel_unified_unit_v<Kernel_t>> struct run_kernel{
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t const& shared_mem_size, IvyGPUStream& stream, IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...);
      return true;
    }
    else return false;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t const& shared_mem_size, IvyGPUStream& stream, IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...);
      return true;
    }
    else return false;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t const& shared_mem_size, IvyGPUStream& stream, IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...);
      return true;
    }
    else return false;
  }
};
template<typename Kernel_t> struct run_kernel<Kernel_t, true>{
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t const& shared_mem_size, IvyGPUStream& stream, IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...);
    }
    else{
      for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel_unified_unit(i, n, args...);
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t const& shared_mem_size, IvyGPUStream& stream, IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...);
    }
    else{
      for (IvyTypes::size_t i = 0; i < nx; ++i){
        for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel_unified_unit(i, j, nx, ny, args...);
      }
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t const& shared_mem_size, IvyGPUStream& stream, IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...);
    }
    else{
      for (IvyTypes::size_t i = 0; i < nx; ++i){
        for (IvyTypes::size_t j = 0; j < ny; ++j){
          for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel_unified_unit(i, j, k, nx, ny, nz, args...);
        }
      }
    }
    return true;
  }
};



#endif


#endif
