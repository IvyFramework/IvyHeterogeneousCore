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


#include "IvyBasicTypes.h"
#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyMemory.h"
#include "IvyMemoryHelpers.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename C, typename T> struct kernel_op_parallel : public kernel_base_noprep_nofin{
    static __CUDA_HOST_DEVICE__ void kernel_unit_unified(IvyTypes::size_t const& i, IvyTypes::size_t const& n, IvyTypes::size_t const& n_serial, T* const& vals){
      IvyTypes::size_t k = n_serial;
      if (i*n_serial + k>n) k = n - i*n_serial;
      C::op(vals[i+n], (vals+(i*n_serial)), k);
    }
    static __CUDA_HOST_DEVICE__ void kernel(IvyTypes::size_t const& i, IvyTypes::size_t const& n, IvyTypes::size_t const& n_serial, T* const& vals){
      IvyTypes::size_t n_ops = C::n_ops(n, n_serial);
      if (kernel_check_dims<kernel_op_parallel<C, T>>::check_dims(i, n_ops)) kernel_unit_unified(i, n, n_serial, vals);
    }
  };
  template<typename C, typename T>
  __CUDA_HOST_DEVICE__ void op_parallel_core(IvyTypes::size_t n, IvyTypes::size_t n_serial, T* vals, IvyGPUStream& stream, int dyn_shared_mem = 0){
    if (n==1) return;
    IvyTypes::size_t n_ops = C::n_ops(n, n_serial);
    run_kernel<kernel_op_parallel<C, T>>(dyn_shared_mem, stream).parallel_1D(n, n_serial, vals);
    op_parallel_core<C, T>(n_ops, n_serial, (vals+n), stream, dyn_shared_mem);
  }
  template<typename C, typename T>
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T op_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    auto obj_allocator = std_mem::allocator<T>();

    IvyTypes::size_t neff = 0;
    C::parallel_op_n_mem(n, n_serial, neff);

    auto h_res = obj_allocator.allocate(1, mem_type_vals, stream);
    auto d_vals = obj_allocator.allocate(neff, IvyMemoryType::GPU, stream);

    obj_allocator.transfer(d_vals, h_vals, n, IvyMemoryType::GPU, mem_type_vals, stream);
    op_parallel_core<C, T>(n, n_serial, d_vals, stream, dyn_shared_mem);
    obj_allocator.transfer(h_res, (d_vals+(neff-1)), 1, mem_type_vals, IvyMemoryType::GPU, stream);

    // Stream synchronization is needed because h_res in host code might otherwise be deallocated before the transfer finishes.
#ifndef __CUDA_DEVICE_CODE__
    stream.synchronize();
#endif

    obj_allocator.deallocate(d_vals, neff, IvyMemoryType::GPU, stream);

    T res = *h_res;
    obj_allocator.deallocate(h_res, 1, mem_type_vals, stream);
    return res;
  }

  template<typename Base, typename T> struct parallel_op_base{
    /*
    n_ops: Number of parallel operations given the total number of elements and the serialization block size
    - n: Total number of elements
    - n_serial: Serialization block size
    */
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyTypes::size_t n_ops(IvyTypes::size_t const& n, IvyTypes::size_t const& n_serial){
      return (n-1+n_serial)/n_serial;
    }
    static __CUDA_HOST_DEVICE__ void parallel_op_n_mem(IvyTypes::size_t n, IvyTypes::size_t n_serial, IvyTypes::size_t& m){
      if (n==1) m+=1;
      else{
        m+=n;
        parallel_op_base<Base, T>::parallel_op_n_mem((n-1+n_serial)/n_serial, n_serial, m);
      }
    }
  };
  template<typename T> struct add_parallel_op : public parallel_op_base<add_parallel_op<T>, T>{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void op(T& res, T* const& vals, IvyTypes::size_t n_serial){
      res = vals[0];
      for (IvyTypes::size_t j = 1; j < n_serial; ++j) res = res + vals[j];
    }
  };
  template<typename T> struct multiply_parallel_op : public parallel_op_base<add_parallel_op<T>, T>{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void op(T& res, T* const& vals, IvyTypes::size_t n_serial){
      res = vals[0];
      for (IvyTypes::size_t j = 1; j < n_serial; ++j) res = res * vals[j];
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
    else return h_vals[0] - add_parallel<T>((h_vals+1), n-1, n_serial, mem_type_vals, stream, dyn_shared_mem);
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T divide_parallel(
    T* h_vals, IvyTypes::size_t n, IvyTypes::size_t n_serial,
    IvyMemoryType mem_type_vals, IvyGPUStream& stream, int dyn_shared_mem = 0
  ){
    if (n==1) return h_vals[0];
    else return h_vals[0] / multiply_parallel<T>((h_vals+1), n-1, n_serial, mem_type_vals, stream, dyn_shared_mem);
  }
}

#endif

namespace std_ivy{
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T add_serial(T* vals, IvyTypes::size_t n){
    T res(vals[0]);
    for (IvyTypes::size_t i = 1; i < n; ++i) res = res + vals[i];
    return res;
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T multiply_serial(T* vals, IvyTypes::size_t n){
    T res(vals[0]);
    for (IvyTypes::size_t i = 1; i < n; ++i) res = res * vals[i];
    return res;
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T subtract_serial(T* vals, IvyTypes::size_t n){
    T res(vals[0]);
    for (IvyTypes::size_t i = 1; i < n; ++i) res = res - vals[i];
    return res;
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ T divide_serial(T* vals, IvyTypes::size_t n){
    T res(vals[0]);
    for (IvyTypes::size_t i = 1; i < n; ++i) res = res / vals[i];
    return res;
  }
}


#endif
