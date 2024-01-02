#ifndef IVYPARALLELOP_H
#define IVYPARALLELOP_H


/*
Header file for parallel operations on arrays.

The main function is op_parallel. The arguments are as follows:
- h_vals: Pointer to the input array of values
- n: Number of values
- n_serial: Number of elements to group into a serial operation
- mem_type_vals: Memory location of h_vals
- stream: GPU stream for the parallelization
- dyn_shared_mem: Amount of dynamic memory shared between threads of a GPU block

The template arguments for op_parallel are as follows:
- C: Class with a static function op, which takes as arguments:
  - res: Reference to the result of the operation
  - vals: Pointer to the arrays of values for a single parallel operation
  - n_serial: Number of elements to serialize within a single parallel operation
- T: Type of the values

The following parallel operations are defined:
- add_parallel: Parallel addition
- multiply_parallel: Parallel multiplication
- subtract_parallel: Parallel subtraction
- divide_parallel: Parallel division
Equivalent operations *_serial serial over the host thread, i.e., a CPU or GPU thread, are also defined.
*/


#include "config/IvyCompilerConfig.h"
#include "IvyBasicTypes.h"
#include "IvyMemoryHelpers.h"
#include "stream/IvyStream.h"
#include "config/IvyCudaConfig.h"
#include "std_ivy/IvyMemory.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename C, typename T>
  __CUDA_GLOBAL__ void kernel_op_parallel(T* vals, IvyTypes::size_t n, IvyTypes::size_t n_serial){
    IvyTypes::size_t i = 0;
    IvyMemoryHelpers::get_kernel_call_dims_1D(i);
    IvyTypes::size_t n_ops = (n-1+n_serial)/n_serial;
    if (i < n_ops){
      IvyTypes::size_t k = n_serial;
      if (i*n_serial + k>n) k = n - i*n_serial;
      C::op(vals[i+n], (vals+(i*n_serial)), k);
    }
  }
  template<typename C, typename T>
  __CUDA_HOST_DEVICE__ void op_parallel_core(T* vals, IvyTypes::size_t n, IvyTypes::size_t n_serial, IvyGPUStream& stream, int dyn_shared_mem = 0){
    if (n==1) return;
    IvyTypes::size_t n_ops = (n-1+n_serial)/n_serial;
    IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
    if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_ops)){
      kernel_op_parallel<C, T><<<nreq_blocks, nreq_threads_per_block, dyn_shared_mem, stream>>>(vals, n, n_serial);
    }
    op_parallel_core<C, T>((vals+n), n_ops, n_serial, stream);
  }

  __CUDA_HOST_DEVICE__ void parallel_op_n_mem(IvyTypes::size_t n, IvyTypes::size_t n_serial, IvyTypes::size_t& m){
    if (n==1) m+=1;
    else{
      m+=n;
      parallel_op_n_mem((n-1+n_serial)/n_serial, n_serial, m);
    }
  }
  template<typename C, typename T>
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T op_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    auto obj_allocator = std_mem::allocator<T>();

    IvyTypes::size_t neff = 0;
    parallel_op_n_mem(n, n_serial, neff);

    auto h_res = obj_allocator.allocate(1, mem_type_vals, stream);
    auto d_vals = obj_allocator.allocate(neff, IvyMemoryType::Device, stream);

    obj_allocator.transfer(d_vals, h_vals, n, IvyMemoryType::Device, mem_type_vals, stream);
    op_parallel_core<C, T>(d_vals, n, n_serial, stream, dyn_shared_mem);
    obj_allocator.transfer(h_res, (d_vals+(neff-1)), 1, mem_type_vals, IvyMemoryType::Device, stream);

#ifndef __CUDA_DEVICE_CODE__
    stream.synchronize();
#endif

    obj_allocator.deallocate(d_vals, neff, IvyMemoryType::Device, stream);

    T res = *h_res;
    obj_allocator.deallocate(h_res, 1, mem_type_vals, stream);
    return res;
  }

  template<typename T> struct add_parallel_op{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void op(T& res, T* const& vals, IvyTypes::size_t n_serial){
      for (IvyTypes::size_t j = 0; j < n_serial; j++){
        if (j==0) res = vals[j];
        else res = res + vals[j];
      }
    }
  };
  template<typename T> struct multiply_parallel_op{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void op(T& res, T* const& vals, IvyTypes::size_t n_serial){
      for (IvyTypes::size_t j = 0; j < n_serial; j++){
        if (j==0) res = vals[j];
        else res = res * vals[j];
      }
    }
  };

  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T add_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    return op_parallel<add_parallel_op<T>, T>(h_vals, n, n_serial, mem_type_vals, stream, dyn_shared_mem);
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T multiply_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    return op_parallel<multiply_parallel_op<T>, T>(h_vals, n, n_serial, mem_type_vals, stream, dyn_shared_mem);
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T subtract_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    if (n==1) return h_vals[0];
    else return h_vals[0] - op_parallel<add_parallel_op<T>, T>((h_vals+1), n-1, n_serial, mem_type_vals, stream, dyn_shared_mem);
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T divide_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    if (n==1) return h_vals[0];
    else return h_vals[0] / op_parallel<multiply_parallel_op<T>, T>((h_vals+1), n-1, n_serial, mem_type_vals, stream, dyn_shared_mem);
  }
}

#endif

namespace std_ivy{
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T add_serial(T* vals, IvyTypes::size_t n){
    T res = vals[0];
    for (IvyTypes::size_t i = 1; i < n; i++) res = res + vals[i];
    return res;
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T multiply_serial(T* vals, IvyTypes::size_t n){
    T res = vals[0];
    for (IvyTypes::size_t i = 1; i < n; i++) res = res * vals[i];
    return res;
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T subtract_serial(T* vals, IvyTypes::size_t n){
    T res = vals[0];
    for (IvyTypes::size_t i = 1; i < n; i++) res = res - vals[i];
    return res;
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T divide_serial(T* vals, IvyTypes::size_t n){
    T res = vals[0];
    for (IvyTypes::size_t i = 1; i < n; i++) res = res / vals[i];
    return res;
  }
}


#endif
