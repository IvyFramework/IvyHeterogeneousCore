#ifndef IVYMEMORYHELPERS_H
#define IVYMEMORYHELPERS_H


/*
IvyMemoryHelpers: A collection of functions for allocating, freeing, and copying memory.
The functions are overloaded for both host and device code when CUDA is enabled.
If CUDA is disabled, allocation and freeing are done with new and delete.
Otherwise, allocation and freeing conventions are as follows:
- Host code, native host memory: new/delete
- Host code, page-locked host memory: cudaMallocHost/cudaFreeHost
- Host code, device memory: cudaMallocAsync/cudaFreeAsync
- Host code, unified memory: cudaMallocManaged/cudaFreeAsync
  For unified memory, one can also specify a variation of the flag to enable prefetching the data.
  In that case, prefetching is done to both the CPU and GPU.
- GPU device code, device memory: new/delete
- GPU device code, host/page-locked host/unified memory: Disabled
Copy operations running on the device may call kernel functions to parallelize further.
For that reason, the -rdc=true flag is required when compiling device code.
*/


#include "IvyBasicTypes.h"
#include "IvyMemoryTypes.h"
#include "config/IvyConfig.h"
#include "config/IvyKernelRun.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyCstdio.h"
#include "stream/IvyStream.h"


// Declarations and enum definitions
namespace IvyMemoryHelpers{
  using size_t = IvyTypes::size_t;
  using ptrdiff_t = IvyTypes::ptrdiff_t;

  /*
  allocate_memory: Allocates memory for an array of type T of size n. Constructors are called for the arguments args.
  - data: Pointer to the target data.
  - n: Number of elements.
  - args: Arguments for the constructors of the elements.
  When using CUDA, the following additional arguments are required:
  - type: In host code, this flag determines whether to allocate the data in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always allocated on the device.
  - stream: In host code, this is the CUDA stream to use for the allocation.
    In device code, any allocation and object construction operations are always synchronous with the running thread.
  */
  template<typename T, typename... Args> struct allocate_memory_fcnal{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate_memory(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  template<typename T, typename... Args> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return allocate_memory_fcnal<T, Args...>::allocate_memory(
      data, n
      , type, stream
    );
  }

  /*
  construct: Allocates memory for an array of type T of size n and constructs the objects for the arguments args.
  - data: Pointer to the target data.
  - n: Number of elements.
  - args: Arguments for the constructors of the elements.
  When using CUDA, the following additional arguments are required:
  - type: In host code, this flag determines whether to allocate the data in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always allocated on the device.
  - stream: In host code, this is the CUDA stream to use for the allocation.
    In device code, any allocation and object construction operations are always synchronous with the running thread.
  */
  template<typename T, typename... Args> struct construct_fcnal{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool construct(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
      , Args&&... args
    );
  };
  template<typename T, typename... Args> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool construct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
    , Args&&... args
  ){
    return construct_fcnal<T, Args...>::construct(
      data, n
      , type, stream
      , args...
    );
  }

  /*
  free_memory: Frees the memory storing an array of type T of size n in a way consistent with allocate_memory.
  - data: Pointer to the data.
  - n: Number of elements.
  When using CUDA, the following additional arguments are required:
  - type: In host code, this flag determines whether the data resides in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always freed from the device.
  - stream: In host code, this is the CUDA stream to use for the deallocation.
    In device code, any deallocation operations are always synchronous with the running thread.
  */
  template<typename T> struct free_memory_fcnal{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool free_memory(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return free_memory_fcnal<T>::free_memory(
      data, n
      , type, stream
    );
  }

  /*
  destroy: Calls the destructors of each element of an array of type T of size n, and frees the memory of the pointer in a way consistent with allocate_memory.
  - data: Pointer to the data.
  - n: Number of elements.
  When using CUDA, the following additional arguments are required:
  - type: In host code, this flag determines whether the data resides in device, host, unified, or page-locked host memory.
    In device code, this flag is ignored, and the memory is always freed from the device.
  - stream: In host code, this is the CUDA stream to use for the deallocation.
    In device code, any deallocation operations are always synchronous with the running thread.
  */
  template<typename T> struct destroy_fcnal{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destroy(
      T*& data,
      size_t n
      , IvyMemoryType type
      , IvyGPUStream& stream
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool destroy(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    return destroy_fcnal<T>::destroy(
      data, n
      , type, stream
    );
  }

  /*
  destroy_data_kernel: Kernel function for destroying object of type T from pointer.
  - n: Number of elements.
  - data: Pointer to the data array.
  */
  template<typename T> struct destroy_data_kernel : public kernel_base_noprep_nofin{
    static __CUDA_HOST_DEVICE__ void kernel(size_t const& i, size_t const& n, T* data);
  };

  /*
  copy_data_kernel: Kernel function for copying data from a pointer of type U to a pointer of type T.
  - n_tgt: Number of elements in the target array.
  - n_src: Number of elements in the source array with the constraint (n_src==n_tgt || n_src==1).
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  */
  template<typename T, typename U> struct copy_data_kernel : public kernel_base_noprep_nofin{
    static __CUDA_HOST_DEVICE__ void kernel(size_t const& i, size_t const& n_tgt, size_t const& n_src, T* target, U* source);
  };

#ifdef __USE_CUDA__
  /*
  get_cuda_transfer_direction: Translates the target and source memory locations to the corresponding cudaMemcpyKind transfer type.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr cudaMemcpyKind get_cuda_transfer_direction(IvyMemoryType tgt, IvyMemoryType src);
#endif

  /*
  transfer_memory: Runs the transfer operation between two pointers of type T.
  - tgt: Pointer to the target data.
  - src: Pointer to the source data.
  - n: Number of elements.
  - type_tgt: Location of the target data in memory.
  - type_src: Location of the source data in memory.
  - stream: CUDA stream to use for the transfer.
    In device code, since cudaMemcpy is not available, we always use cudaMemcpyAsync instead.
  */
  template<typename T> struct transfer_memory_fcnal{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_memory(
      T*& tgt, T* const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    IvyGPUStream& stream
  ){
    return transfer_memory_fcnal<T>::transfer_memory(
      tgt, src, n,
      type_tgt, type_src,
      stream
    );
  }

  /*
  Overloads to allow passing raw cudaStream_t objects.
  */
#ifdef __USE_CUDA__
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return allocate_memory(
      data, n
      , type
      , sr
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool construct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
    , Args&&... args
  ){
    IvyGPUStream sr(stream, false);
    return construct(
      data, n
      , type
      , sr
      , args...
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return free_memory(
      data, n
      , type
      , sr
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destroy(
    T*& data,
    size_t n
    , IvyMemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return destroy(
      data, n
      , type
      , sr
    );
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return transfer_memory(
      tgt, src, n,
      type_tgt, type_src,
      sr
    );
  }
#endif
}


// Definitions
namespace IvyMemoryHelpers{
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ bool allocate_memory_fcnal<T, Args...>::allocate_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    if (n==0 || data) return false;
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    bool const is_gpu = is_gpu_memory(type);
    bool const is_uni = is_unified_memory(type);
    if (is_gpu || is_uni || is_pl){
      if (is_gpu){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocAsync((void**) &data, n*sizeof(T), stream));
        //printf("allocate_memory_fcnal::allocate_memory: Ran cudaMallocAsync on %p\n", data);
      }
      else if (is_pl){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaHostAlloc((void**) &data, n*sizeof(T), cudaHostAllocDefault));
        //printf("allocate_memory_fcnal::allocate_memory: Ran cudaHostAlloc on %p\n", data);
      }
      else if (is_uni){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocManaged((void**) &data, n*sizeof(T), cudaMemAttachGlobal));
        //printf("allocate_memory_fcnal::allocate_memory: Ran cudaMallocManaged on %p\n", data);
        if (is_prefetched(type)){
          IvyCudaConfig::IvyDeviceNum_t dev_gpu = 0;
          int supports_prefetch = 0;
          if (
            __CUDA_CHECK_SUCCESS__(cudaGetDevice(&dev_gpu))
            &&
            __CUDA_CHECK_SUCCESS__(cudaDeviceGetAttribute(&supports_prefetch, cudaDevAttrConcurrentManagedAccess, dev_gpu))
            &&
            supports_prefetch==1
            ){
            __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemPrefetchAsync(data, n*sizeof(T), cudaCpuDeviceId, stream));
            __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemPrefetchAsync(data, n*sizeof(T), dev_gpu, stream));
          }
        }
      }
    }
    else
#endif
    {
      data = (T*) malloc(n*sizeof(T));
      //printf("allocate_memory_fcnal::allocate_memory: Ran malloc on %p\n", data);
    }
    return true;
  }


  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ bool construct_fcnal<T, Args...>::construct(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
    , Args&&... args
  ){
    bool res = allocate_memory_fcnal<T, Args...>::allocate_memory(data, n, type, stream);
    if (res){
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
      static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
      bool const is_pl = is_pagelocked(type);
      bool const is_acc = use_device_acc(type);
      if (is_acc || is_pl){
        T* temp = nullptr;
        res &= construct(temp, n, IvyMemoryType::Host, stream, args...);
        res &= transfer_memory(data, temp, n, type, IvyMemoryType::Host, stream);
        stream.synchronize();
        res &= free_memory(temp, n, IvyMemoryType::Host, stream); // Important! Destroying can also invalidate the internal components of the data pointer on the device!
      }
      else
#endif
      {
        for (size_t i=0; i<n; i++) new (data+i) T(std_util::forward<Args>(args)...);
      }
    }
    return res;
  }

  template<typename T> __CUDA_HOST_DEVICE__ bool free_memory_fcnal<T>::free_memory(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    if (!data) return true;
    if (n==0) return false;
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    if (use_device_GPU(type) || is_pl){
      bool const is_uni = is_unified_memory(type);
      if (!is_pl && !is_uni){
        //printf("free_memory_fcnal::free_memory: Will run cudaFreeAsync on %p\n", data);
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFreeAsync(data, stream));
      }
      else if (is_uni){
        //printf("free_memory_fcnal::free_memory: Will run cudaFree on %p\n", data);
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(data));
      }
      else{
        //printf("free_memory_fcnal::free_memory: Will run cudaFreeHost on %p\n", data);
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFreeHost(data));
      }
    }
    else
#endif
    {
      //printf("free_memory_fcnal::free_memory: Will run free on %p\n", data);
      free(data);
    }
    data = nullptr;
    return true;
  }

  template<typename T> __CUDA_HOST_DEVICE__ bool destroy_fcnal<T>::destroy(
    T*& data,
    size_t n
    , IvyMemoryType type
    , IvyGPUStream& stream
  ){
    if (!data) return true;
    if (n==0) return false;
    bool res = true;
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    bool const is_pl = is_pagelocked(type);
    bool const is_acc = use_device_acc(type);
    if (is_acc || is_pl){
      T* temp = nullptr;
      res &= allocate_memory(temp, n, IvyMemoryType::Host, stream);
      res &= transfer_memory(temp, data, n, IvyMemoryType::Host, type, stream);
      stream.synchronize();
      {
        T* ptr = temp;
        for (size_t i=0; i<n; i++){
          ptr->~T();
          ++ptr;
        }
      }
      res &= free_memory(temp, n, IvyMemoryType::Host, stream);
    }
    else
#endif
    {
      T* ptr = data;
      for (size_t i=0; i<n; i++){
        ptr->~T();
        ++ptr;
      }
    }
    res &= free_memory(data, n, type, stream);
    return res;
  }

  template<typename T> __CUDA_HOST_DEVICE__ void destroy_data_kernel<T>::kernel(size_t const& i, size_t const& n, T* data){
    if (i<n) (data+i)->~T();
  }

  template<typename T, typename U> __CUDA_HOST_DEVICE__ void copy_data_kernel<T, U>::kernel(size_t const& i, size_t const& n_tgt, size_t const& n_src, T* target, U* source){
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_kernel::kernel: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_kernel::kernel: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (i<n_tgt) *(target+i) = *(source + (n_src==1 ? 0 : i));
  }

#ifdef __USE_CUDA__
  __CUDA_HOST_DEVICE__ constexpr cudaMemcpyKind get_cuda_transfer_direction(IvyMemoryType tgt, IvyMemoryType src){
#if (DEVICE_CODE == DEVICE_CODE_HOST)
    bool const tgt_on_device = is_gpu_memory(tgt);
    bool const tgt_on_host = is_host_memory(tgt);
    bool const tgt_unified = is_unified_memory(tgt);
    bool const src_on_device = is_gpu_memory(src);
    bool const src_on_host = is_host_memory(src);
    bool const src_unified = is_unified_memory(src);
    if (tgt_on_host && src_on_host) return cudaMemcpyHostToHost;
    else if (tgt_on_device && src_on_host) return cudaMemcpyHostToDevice;
    else if (tgt_on_host && src_on_device) return cudaMemcpyDeviceToHost;
    else if (tgt_on_device && src_on_device) return cudaMemcpyDeviceToDevice;
    else if (tgt_unified || src_unified) return cudaMemcpyDefault;
    else{
      __PRINT_ERROR__("IvyMemoryHelpers::get_cuda_transfer_direction: Unknown transfer direction.\n");
      assert(0);
      return cudaMemcpyDefault;
    }
#else
    return cudaMemcpyDeviceToDevice;
#endif
  }
#endif

  template<typename T> __CUDA_HOST_DEVICE__ bool transfer_memory_fcnal<T>::transfer_memory(
    T*& tgt, T* const& src, size_t n,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    IvyGPUStream& stream
  ){
    if (!tgt || !src) return false;
#if DEVICE_CODE == DEVICE_CODE_HOST && defined(__USE_CUDA__)
    auto dir = get_cuda_transfer_direction(type_tgt, type_src);
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpyAsync(tgt, src, n*sizeof(T), dir, stream));
#else
    memcpy(tgt, src, n*sizeof(T));
#endif
    return true;
  }
}


#endif
