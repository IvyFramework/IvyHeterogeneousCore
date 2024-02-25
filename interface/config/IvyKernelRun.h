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
struct kernel_base_noprep_nofin{
  static __CUDA_HOST_DEVICE__ void prepare(...){}
  static __CUDA_HOST_DEVICE__ void finalize(...){}
};

/*
kernel_check_dims: A struct to check the dimensions of the kernel call.
The user may define specializations on their user-defined kernels to customize the behavior of the run_kernel struct and the generic_kernel_[N]D kernel functions,
hopefully without having to specialize them in most of their use cases.

kernel_check_dims::check_dims: Checks if index i for the current thread is within the range of the dimension n of the input data.
- In the CPU, the check is always true because dimensions are always organized to be within the range of the input data.
- In the GPU, an explicit check for i<n is performed to avoid out-of-bounds memory access
since the number of allocated threads, based on the warp size, may exceed the dimensions of the input data.

The way to use this default implementation would look like the following:

template<typename... Args> struct example_kernel_implementation{
  static __CUDA_HOST_DEVICE__ void kernel(size_t const& i, size_t const& n, Args&&... args){
    if (kernel_check_dims<example_kernel_implementation>::check_dims(i, n)){
      // Do something with i, n, and args...
    }
  }
};

*/
template<typename Kernel_t> struct kernel_check_dims{
#if DEVICE_CODE == DEVICE_CODE_GPU
  static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& n){
    return (i<n);
  }
  static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& j, IvyTypes::size_t const& nx, IvyTypes::size_t const& ny){
    return (i<nx && j<ny);
  }
  static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& j, IvyTypes::size_t const& k, IvyTypes::size_t const& nx, IvyTypes::size_t const& ny, IvyTypes::size_t const& nz){
    return (i<nx && j<ny && k<nz);
  }
#else
  static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& n){ return true; }
  static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& j, IvyTypes::size_t const& nx, IvyTypes::size_t const& ny){ return true; }
  static __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr bool check_dims(IvyTypes::size_t const& i, IvyTypes::size_t const& j, IvyTypes::size_t const& k, IvyTypes::size_t const& nx, IvyTypes::size_t const& ny, IvyTypes::size_t const& nz){ return true; }
#endif
};

#ifdef __USE_CUDA__

/*
kernel_call_dims: A struct to get the dimensions of the kernel call corresponding to the blockIdx and threadIdx dimensions of the current thread.
We provide a struct configuration rather than functions so that if needed, the functions can be specialized/partially specialized for different kernels.

kernel_call_dims::get_dims: Gets the dimensions of the kernel call corresponding to the blockIdx and threadIdx dimensions of the current thread.
- 1D is fully flattened.
- In 2D, the z dimension is folded into the y direction, and the x dimension is taken as is.
- The x, y, and z dimensions are taken as they are in 3D.
*/
template<typename Kernel_t> struct kernel_call_dims{
  static __INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_dims(IvyTypes::size_t& i){
    IvyTypes::size_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    IvyTypes::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    IvyTypes::size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    i = ix + iy * blockDim.x * gridDim.x + iz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
  }
  static __INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_dims(IvyTypes::size_t& i, IvyTypes::size_t& j){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    IvyTypes::size_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    IvyTypes::size_t iz = blockIdx.z * blockDim.z + threadIdx.z;
    j = iy + iz * blockDim.y * gridDim.y;
  }
  static __INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_dims(IvyTypes::size_t& i, IvyTypes::size_t& j, IvyTypes::size_t& k){
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;
  }
};


template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_1D(Args... args){
  IvyTypes::size_t i = 0;
  kernel_call_dims<Kernel_t>::get_dims(i);
  Kernel_t::kernel(i, args...);
}
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_2D(Args... args){
  IvyTypes::size_t i = 0, j = 0;
  kernel_call_dims<Kernel_t>::get_dims(i, j);
  Kernel_t::kernel(i, j, args...);
}
template<typename Kernel_t, typename... Args> __CUDA_GLOBAL__ void generic_kernel_3D(Args... args){
  IvyTypes::size_t i = 0, j = 0, k = 0;
  kernel_call_dims<Kernel_t>::get_dims(i, j, k);
  Kernel_t::kernel(i, j, k, args...);
}

DEFINE_HAS_CALL(kernel_unit_unified);

template<typename Kernel_t, bool = has_call_kernel_unit_unified_v<Kernel_t>> struct run_kernel : run_kernel_base{
  __CUDA_HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n)){
      Kernel_t::prepare(true, n, args...);
      __CUDA_CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...)));
      Kernel_t::finalize(true, n, args...);
      return true;
    }
    else return false;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      Kernel_t::prepare(true, nx, ny, args...);
      __CUDA_CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...)));
      Kernel_t::finalize(true, nx, ny, args...);
      return true;
    }
    else return false;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      Kernel_t::prepare(true, nx, ny, nz, args...);
      __CUDA_CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...)));
      Kernel_t::finalize(true, nx, ny, nz, args...);
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
      Kernel_t::prepare(true, n, args...);
      __CUDA_CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_1D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(n, args...)));
      Kernel_t::finalize(true, n, args...);
    }
    else{
      Kernel_t::prepare(false, n, args...);
      for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel_unit_unified(i, n, args...);
      Kernel_t::finalize(false, n, args...);
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny)){
      Kernel_t::prepare(true, nx, ny, args...);
      __CUDA_CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_2D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, args...)));
      Kernel_t::finalize(true, nx, ny, args...);
    }
    else{
      Kernel_t::prepare(false, nx, ny, args...);
      for (IvyTypes::size_t i = 0; i < nx; ++i){
        for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel_unit_unified(i, j, nx, ny, args...);
      }
      Kernel_t::finalize(false, nx, ny, args...);
    }
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nx*ny*nz)){
      Kernel_t::prepare(true, nx, ny, nz, args...);
      __CUDA_CHECK_KERNEL_AND_WARN_WITH_ERROR__(__ENCAPSULATE__(generic_kernel_3D<Kernel_t, IvyTypes::size_t, Args...><<<nreq_blocks, nreq_threads_per_block, shared_mem_size, stream>>>(nx, ny, nz, args...)));
      Kernel_t::finalize(true, nx, ny, nz, args...);
    }
    else{
      Kernel_t::prepare(false, nx, ny, nz, args...);
      for (IvyTypes::size_t i = 0; i < nx; ++i){
        for (IvyTypes::size_t j = 0; j < ny; ++j){
          for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel_unit_unified(i, j, k, nx, ny, nz, args...);
        }
      }
      Kernel_t::finalize(false, nx, ny, nz, args...);
    }
    return true;
  }
};

#else

template<typename Kernel_t> struct run_kernel : run_kernel_base{
  __CUDA_HOST_DEVICE__ run_kernel(IvyTypes::size_t const& shared_mem_size_, IvyGPUStream& stream_) : run_kernel_base(shared_mem_size_, stream_){}

  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_1D(IvyTypes::size_t n, Args... args){
    Kernel_t::prepare(false, n, args...);
    for (IvyTypes::size_t i = 0; i < n; ++i) Kernel_t::kernel(i, n, args...);
    Kernel_t::finalize(false, n, args...);
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_2D(IvyTypes::size_t nx, IvyTypes::size_t ny, Args... args){
    Kernel_t::prepare(false, nx, ny, args...);
    for (IvyTypes::size_t i = 0; i < nx; ++i){
      for (IvyTypes::size_t j = 0; j < ny; ++j) Kernel_t::kernel(i, j, nx, ny, args...);
    }
    Kernel_t::finalize(false, nx, ny, args...);
    return true;
  }
  template<typename... Args> __CUDA_HOST_DEVICE__ bool parallel_3D(IvyTypes::size_t nx, IvyTypes::size_t ny, IvyTypes::size_t nz, Args... args){
    Kernel_t::prepare(false, nx, ny, nz, args...);
    for (IvyTypes::size_t i = 0; i < nx; ++i){
      for (IvyTypes::size_t j = 0; j < ny; ++j){
        for (IvyTypes::size_t k = 0; k < nz; ++k) Kernel_t::kernel(i, j, k, nx, ny, nz, args...);
      }
    }
    Kernel_t::finalize(false, nx, ny, nz, args...);
    return true;
  }
};

#endif


#endif
