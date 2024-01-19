#ifndef IVYKERNELRUN_H
#define IVYKERNELRUN_H


/*
Kernel calls could be structured such that call to static Kernel_t member function Kernel_t::kernel is either a nonfactorizable function,
or it calls static Kernel_t::kernel_unit_unified with a check that looks like 'if (i<n) kernel_unit_unified(i, n, args...);' and does nothing else.
While we do not check whether argument memory locations are arranged properly, in case they are, and if kernel_unit_unified calls are factorizable,
one can use the run_kernel struct below to run the kernel in parallel or in a loop, depending on whether the GPU is usable.

The ultimate use case is as follows:
- The user defines the kernel struct with a static kernel function with input as i for the index of the thread, n for the total number of threads, and 'args...' for the optional arguments:

  struct example_kernel{
    template<typename... Args> __CUDA_HOST_DEVICE__ static void kernel(IvyTypes::size_t i, IvyTypes::size_t n, Args... args){
      // Do something with i, n, and args...
    }
  };

or

  struct example_kernel{
    template<typename... Args> __CUDA_HOST_DEVICE__ static void kernel_unit_unified(IvyTypes::size_t i, IvyTypes::size_t n, Args... args){
      // Do something with i, n, and args...
    }
    template<typename... Args> __CUDA_HOST_DEVICE__ static void kernel(IvyTypes::size_t i, IvyTypes::size_t n, Args... args){
      if (i < n) kernel_unit_unified(i, n, args...);
    }
  };

- The user can then call the kernel as follows:

  run_kernel<example_kernel>(shared_mem_size, stream).parallel_1D(n, args...);

or

  run_kernel<example_kernel>(shared_mem_size, stream).parallel_2D(nx, ny, args...);

or

  run_kernel<example_kernel>(shared_mem_size, stream).parallel_3D(nx, ny, nz, args...);
*/


#include "IvyBasicTypes.h"
#include "config/IvyConfig.h"
#include "std_ivy/IvyTypeTraits.h"


struct run_kernel_base{
  IvyTypes::size_t shared_mem_size;
  IvyGPUStream& stream;

  __CUDA_HOST_DEVICE__ run_kernel_base(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : shared_mem_size(shared_mem_size_), stream(stream_){}
};

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

DEFINE_HAS_CALL(kernel_unit_unified);

template<typename Kernel_t, bool = has_call_kernel_unit_unified_v<Kernel_t>> struct run_kernel : run_kernel_base{
  __CUDA_HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...);
      return true;
    }
    else return false;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...);
      return true;
    }
    else return false;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...);
      return true;
    }
    else return false;
  }
};
template<typename Kernel_t> struct run_kernel<Kernel_t, true> : run_kernel_base{
  __CUDA_HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...);
    }
    else{
      for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel_unit_unified(i, n, args...);
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...);
    }
    else{
      for (IvyTypes::size_t i = 0; i < nx; ++i){
        for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel_unit_unified(i, j, nx, ny, args...);
      }
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...);
    }
    else{
      for (IvyTypes::size_t i = 0; i < nx; ++i){
        for (IvyTypes::size_t j = 0; j < ny; ++j){
          for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel_unit_unified(i, j, k, nx, ny, nz, args...);
        }
      }
    }
    return true;
  }
};

#else

template<typename Kernel_t> struct run_kernel : run_kernel_base{
  __CUDA_HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel(i, n, args...);
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    for (IvyTypes::size_t i = 0; i < nx; ++i){
      for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel(i, j, nx, ny, args...);
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    for (IvyTypes::size_t i = 0; i < nx; ++i){
      for (IvyTypes::size_t j = 0; j < ny; ++j){
        for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel(i, j, k, nx, ny, nz, args...);
      }
    }
    return true;
  }
};

#endif


#endif
