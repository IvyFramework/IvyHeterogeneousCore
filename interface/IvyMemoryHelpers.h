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
- Device code, device memory: new/delete
- Device code, host/page-locked host/unified memory: Disabled
Copy operations running on the device may call kernel functions to parallelize further.
For that reason, the -rdc=true flag is required when compiling device code.
*/


#include "IvyBasicTypes.h"
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

  enum class MemoryType{
    Host,
    Device,
    PageLocked,
    Unified,
    UnifiedPrefetched
  };

  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool requires_malloc_free(MemoryType type);

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
#ifdef __USE_CUDA__
      , MemoryType type
      , IvyGPUStream& stream
#endif
      , Args&&... args
    );
  };
  template<typename T, typename... Args> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , MemoryType type
    , IvyGPUStream& stream
#endif
    , Args&&... args
  ){
    return allocate_memory_fcnal<T, Args...>::allocate_memory(
      data, n
#ifdef __USE_CUDA__
      , type, stream
#endif
      , args...
    );
  }

  /*
  free_memory: Frees memory for an array of type T of size n in a way consistent with allocate_memory.
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
#ifdef __USE_CUDA__
      , MemoryType type
      , IvyGPUStream& stream
#endif
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , MemoryType type
    , IvyGPUStream& stream
#endif
  ){
    return free_memory_fcnal<T>::free_memory(
      data, n
#ifdef __USE_CUDA__
      , type, stream
#endif
    );
  }

  /*
  is_host_memory: Returns true if the memory type is host memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool is_host_memory(MemoryType type);

  /*
  is_device_memory: Returns true if the memory type is device memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool is_device_memory(MemoryType type);

  /*
  is_unified_memory: Returns true if the memory type is unified memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool is_unified_memory(MemoryType type);

  /*
  is_pagelocked: Returns true if the memory type is page-locked host memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool is_pagelocked(MemoryType type);

  /*
  is_prefetched: Returns true if the memory type is unified memory that has been prefetched.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool is_prefetched(MemoryType type);

#ifdef __USE_CUDA__
  /*
  get_cuda_transfer_direction: Translates the target and source memory locations to the corresponding cudaMemcpyKind transfer type.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ cudaMemcpyKind get_cuda_transfer_direction(MemoryType tgt, MemoryType src);

  /*
  copy_data_kernel: Kernel function for copying data from a pointer of type U to a pointer of type T.
  - n_tgt: Number of elements in the target array.
  - n_src: Number of elements in the source array with the constraint (n_src==n_tgt || n_src==1).
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  */
  template<typename T, typename U> struct copy_data_kernel{
    static __CUDA_HOST_DEVICE__ void kernel(size_t const& i, size_t const& n_tgt, size_t const& n_src, T* target, U* source);
  };

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
      MemoryType type_tgt, MemoryType type_src,
      IvyGPUStream& stream
    );
  };
  template<typename T> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    MemoryType type_tgt, MemoryType type_src,
    IvyGPUStream& stream
  ){
    return transfer_memory_fcnal<T>::transfer_memory(
      tgt, src, n,
      type_tgt, type_src,
      stream
    );
  }
#endif

  /*
  Overloads to allow passing raw cudaStream_t objects.
  */
#ifdef __USE_CUDA__
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
    , MemoryType type
    , cudaStream_t stream
    , Args&&... args
  ){
    IvyGPUStream sr(stream, false);
    return allocate_memory(
      data, n
      , type
      , sr
      , args...
    );
  }
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
    , MemoryType type
    , cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return free_memory(
      data, n
      , type
      , sr
    );
  }
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    MemoryType type_tgt, MemoryType type_src,
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

  /*
  get_execution_default_memory: Returns the default memory type for the current execution environment.
  For host code or if not using CUDA, this is MemoryType::Host.
  Otherwise, for device code, this is MemoryType::Device.
  */
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ __CPP_CONSTEXPR__ MemoryType get_execution_default_memory();
}


// Definitions
namespace IvyMemoryHelpers{
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool requires_malloc_free(MemoryType type){
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const is_pl = is_pagelocked(type);
    bool const is_dev = is_device_memory(type);
    bool const is_uni = is_unified_memory(type);
    return (is_dev || is_uni || is_pl);
#endif
#else
    return false;
#endif
  }

  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ bool allocate_memory_fcnal<T, Args...>::allocate_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , MemoryType type
    , IvyGPUStream& stream
#endif
    , Args&&... args
  ){
    if (n==0 || data) return false;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const is_pl = is_pagelocked(type);
    bool const is_dev = is_device_memory(type);
    bool const is_uni = is_unified_memory(type);
    if (is_dev || is_uni || is_pl){
      bool res = true;
      if (is_dev){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocAsync((void**) &data, n*sizeof(T), stream));
      }
      else if (is_pl){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaHostAlloc((void**) &data, n*sizeof(T), cudaHostAllocDefault));
      }
      else if (is_uni){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocManaged((void**) &data, n*sizeof(T), cudaMemAttachGlobal));
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
      if (sizeof...(Args)>0){
        T* temp = nullptr;
        res &= allocate_memory(temp, n, MemoryType::Host, stream, args...);
        res &= transfer_memory(data, temp, n, type, MemoryType::Host, stream);
        stream.synchronize();
        res &= free_memory(temp, n, MemoryType::Host, stream);
      }
      return res;
    }
    else
#endif
#endif
    {
      if (n==1) data = new T(std_util::forward<Args>(args)...);
      else data = new T[n]{ std_util::forward<Args>(args)... };
      return true;
    }
  }

  template<typename T> __CUDA_HOST_DEVICE__ bool free_memory_fcnal<T>::free_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , MemoryType type
    , IvyGPUStream& stream
#endif
  ){
    if (!data) return true;
    if (n==0) return false;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const is_pl = is_pagelocked(type);
    if (is_device_memory(type) || is_unified_memory(type) || is_pl){
      if (!is_pl){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFreeAsync(data, stream));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFreeHost(data));
      }
    }
    else
#endif
#endif
    {
      if (n==1) delete data;
      else delete[] data;
    }
    data = nullptr;
    return true;
  }

  __CUDA_HOST_DEVICE__ bool is_host_memory(MemoryType type){
    return type==MemoryType::Host || type==MemoryType::PageLocked;
  }

  __CUDA_HOST_DEVICE__ bool is_device_memory(MemoryType type){
    return type==MemoryType::Device;
  }

  __CUDA_HOST_DEVICE__ bool is_unified_memory(MemoryType type){
    return type==MemoryType::Unified || type==MemoryType::UnifiedPrefetched;
  }

  __CUDA_HOST_DEVICE__ bool is_pagelocked(MemoryType type){
    return type==MemoryType::PageLocked;
  }

  __CUDA_HOST_DEVICE__ bool is_prefetched(MemoryType type){
    return type==MemoryType::UnifiedPrefetched;
  }

#ifdef __USE_CUDA__
  __CUDA_HOST_DEVICE__ inline cudaMemcpyKind get_cuda_transfer_direction(MemoryType tgt, MemoryType src){
#ifndef __CUDA_DEVICE_CODE__
    bool const tgt_on_device = is_device_memory(tgt);
    bool const tgt_on_host = is_host_memory(tgt);
    bool const tgt_unified = is_unified_memory(tgt);
    bool const src_on_device = is_device_memory(src);
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

  template<typename T> __CUDA_HOST_DEVICE__ bool transfer_memory_fcnal<T>::transfer_memory(
    T*& tgt, T* const& src, size_t n,
    MemoryType type_tgt, MemoryType type_src,
    IvyGPUStream& stream
  ){
    if (!tgt || !src) return false;
    auto dir = get_cuda_transfer_direction(type_tgt, type_src);
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpyAsync(tgt, src, n*sizeof(T), dir, stream));
    return true;
  }
#endif

#ifdef __CUDA_DEVICE_CODE__
  __CUDA_HOST_DEVICE__ __CPP_CONSTEXPR__ MemoryType get_execution_default_memory(){ return MemoryType::Device; }
#else
  __CUDA_HOST_DEVICE__ __CPP_CONSTEXPR__ MemoryType get_execution_default_memory(){ return MemoryType::Host; }
#endif
}


// Aliases for std_ivy namespace
namespace std_ivy{
  using IvyMemoryType = IvyMemoryHelpers::MemoryType;
}


#endif
