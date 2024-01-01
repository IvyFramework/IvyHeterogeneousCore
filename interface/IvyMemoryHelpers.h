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
  template<typename T, typename... Args> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , MemoryType type
    , IvyGPUStream& stream
#endif
    , Args&&... args
  );

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
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , MemoryType type
    , IvyGPUStream& stream
#endif
  );

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
  get_kernel_call_dims_1D/2D/3D: Gets the dimensions of the kernel call
  corresponding to the blockIdx and threadIdx dimensions of the current thread:
  - 1D is fully flattened.
  - In 2D, the z dimension is folded into the y direction, and the x dimension is taken as is.
  - The x, y, and z dimensions are taken as they are in 3D.
  */
  __INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_kernel_call_dims_1D(IvyTypes::size_t& i);
  __INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_kernel_call_dims_2D(IvyTypes::size_t& i, IvyTypes::size_t& j);
  __INLINE_FCN_RELAXED__ __CUDA_DEVICE__ void get_kernel_call_dims_3D(IvyTypes::size_t& i, IvyTypes::size_t& j, IvyTypes::size_t& k);

  /*
  copy_data_kernel: Kernel function for copying data from a pointer of type U to a pointer of type T.
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  - n_tgt: Number of elements in the target array.
  - n_src: Number of elements in the source array with the constraint (n_src==n_tgt || n_src==1).
  */
  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, size_t n_tgt, size_t n_src);

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
  template<typename T> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    MemoryType type_tgt, MemoryType type_src,
    IvyGPUStream& stream
  );
#endif

  /*
  copy_data_multistream: Copies data from a pointer of type U to a pointer of type T.
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  - n_tgt_init: Number of elements in the target array before the copy. If the target array is not null, it is freed.
  - n_tgt: Number of elements in the target array after the copy.
  - n_src: Number of elements in the source array before the copy. It has to satisfy the constraint (n_src==n_tgt || n_src==1).
  When using CUDA, the following additional arguments are required:
  - type_tgt: Location of the target data in memory.
  - type_src: Location of the source data in memory.
  - stream: CUDA stream to use for the copy.
    If stream is anything other than cudaStreamLegacy, the copy is asynchronous, even in device code.
  */
  template<typename T, typename U> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool copy_data_multistream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
#endif
  );

  /*
  copy_data_single_stream: Same as copy_data_multistream,
  but all operations are done over a single stream without creating additional streams or events
  if using CUDA.
  */
  template<typename T, typename U> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool copy_data_single_stream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
#endif
  );

  /*
  copy_data_multistream_ext: Same as copy_data_multistream, but with all streams and events passed as arguments.
  */
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_multistream_ext(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
    , IvyGPUStream& sr_tgt_al
    , IvyGPUStream& sr_src_altr
    , IvyGPUStream& sr_tgt_free
    , IvyGPUStream& sr_src_free
    , IvyGPUEvent& ev_begin
    , IvyGPUEvent& ev_src_altr_tgt_al
    , IvyGPUEvent& ev_copy
#endif
  );

  /*
  copy_data: Alias to choose between copy_data_multistream and copy_data_single_stream.
  It turns out that copy_data_single_stream is faster even when operating over >1M elements
  because the creation of additional streams and events is very expensive.
  Waiting for pre-made streams/events is also expensive, but not as much as creating them in the first place.
  */
  template<typename T, typename U> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
#endif
  ){
    return copy_data_single_stream(
      target, source,
      n_tgt_init, n_tgt, n_src
#ifdef __USE_CUDA__
      , type_tgt, type_src, stream
#endif
    );
  }

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
  template<typename T, typename U> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src,
    MemoryType type_tgt, MemoryType type_src,
    cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return copy_data(
      target, source,
      n_tgt_init, n_tgt, n_src,
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
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ bool allocate_memory(
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

  template<typename T> __CUDA_HOST_DEVICE__ bool free_memory(
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

#ifdef __USE_CUDA__
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

  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, size_t n_tgt, size_t n_src){
    IvyTypes::size_t i = 0;
    IvyMemoryHelpers::get_kernel_call_dims_1D(i);
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_kernel: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_kernel: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (i<n_tgt) *(target+i) = *(source + (n_src==1 ? 0 : i));
  }

  template<typename T> __CUDA_HOST_DEVICE__ bool transfer_memory(
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

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_multistream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
#endif
  ){
    bool res = true;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const tgt_on_device = is_device_memory(type_tgt) || is_unified_memory(type_tgt);
    bool const src_on_device = is_device_memory(type_src) || is_unified_memory(type_src);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_multistream: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_multistream: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
      res &= free_memory(
        target, n_tgt_init
#ifdef __USE_CUDA__
        , type_tgt, stream
#endif
      );
      res &= allocate_memory(
        target, n_tgt
#ifdef __USE_CUDA__
        , type_tgt, stream
#endif
      );
    }
    if (res){
#ifdef __USE_CUDA__
      IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        //__PRINT_INFO__("IvyMemoryHelpers::copy_data_multistream: Running parallel copy.\n");
#ifndef __CUDA_DEVICE_CODE__
        IvyGPUEvent ev_begin, ev_src_altr_tgt_al, ev_copy;
        ev_begin.record(stream);
#endif
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream sr_src_altr(IvyGPUStream::StreamFlags::NonBlocking);
          IvyGPUStream* stream_ptr = &sr_src_altr;
#else
          IvyGPUStream* stream_ptr = &stream;
#endif
          res &= allocate_memory(d_source, n_src, MemoryType::Device, *stream_ptr);
#ifndef __CUDA_DEVICE_CODE__
          sr_src_altr.wait(ev_begin);
#endif
          res &= transfer_memory(d_source, source, n_src, MemoryType::Device, type_src, *stream_ptr);
#ifndef __CUDA_DEVICE_CODE__
          ev_src_altr_tgt_al.record(sr_src_altr);
          stream.wait(ev_src_altr_tgt_al);
#endif
        }
        if (!tgt_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream sr_tgt_al(IvyGPUStream::StreamFlags::NonBlocking);
          IvyGPUStream* stream_tgt_al_ptr = &sr_tgt_al;
#else
          IvyGPUStream* stream_tgt_al_ptr = &stream;
#endif
          T* d_target = nullptr;
          res &= allocate_memory(d_target, n_tgt, MemoryType::Device, *stream_tgt_al_ptr);
#ifndef __CUDA_DEVICE_CODE__
          ev_src_altr_tgt_al.record(sr_tgt_al);
          stream.wait(ev_src_altr_tgt_al);
#endif
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_tgt, n_src);
          res &= transfer_memory(target, d_target, n_tgt, type_tgt, MemoryType::Device, stream);
#ifndef __CUDA_DEVICE_CODE__
          ev_copy.record(stream);

          IvyGPUStream sr_tgt_free(IvyGPUStream::StreamFlags::NonBlocking);
          IvyGPUStream* stream_tgt_free_ptr = &sr_tgt_free;
          sr_tgt_free.wait(ev_copy);
#else
          IvyGPUStream* stream_tgt_free_ptr = &stream;
#endif
          free_memory(d_target, n_tgt, MemoryType::Device, *stream_tgt_free_ptr);
        }
        else{
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(target, d_source, n_tgt, n_src);
#ifndef __CUDA_DEVICE_CODE__
          ev_copy.record(stream);
#endif
        }
        if (!src_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream sr_src_free(IvyGPUStream::StreamFlags::NonBlocking);
          IvyGPUStream* stream_src_free_ptr = &sr_src_free;
          sr_src_free.wait(ev_copy);
#else
          IvyGPUStream* stream_src_free_ptr = &stream;
#endif
          free_memory(d_source, n_src, MemoryType::Device, *stream_src_free_ptr);
        }
      }
      else{
        if (tgt_on_device!=src_on_device){
          __PRINT_ERROR__("IvyMemoryHelpers::copy_data_multistream: Failed to copy data between host and device.\n");
          assert(0);
        }
        //__PRINT_INFO__("IvyMemoryHelpers::copy_data_multistream: Running serial copy.\n");
#else
      {
#endif
        for (size_t i=0; i<n_tgt; i++) target[i] = source[(n_src==1 ? 0 : i)];
      }
    }
    return res;
  }

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_multistream_ext(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
    , IvyGPUStream& sr_tgt_al
    , IvyGPUStream& sr_src_altr
    , IvyGPUStream& sr_tgt_free
    , IvyGPUStream& sr_src_free
    , IvyGPUEvent& ev_begin
    , IvyGPUEvent& ev_src_altr_tgt_al
    , IvyGPUEvent& ev_copy
#endif
  ){
    bool res = true;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const tgt_on_device = is_device_memory(type_tgt) || is_unified_memory(type_tgt);
    bool const src_on_device = is_device_memory(type_src) || is_unified_memory(type_src);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_multistream_ext: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_multistream_ext: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
      res &= free_memory(
        target, n_tgt_init
#ifdef __USE_CUDA__
        , type_tgt, stream
#endif
      );
      res &= allocate_memory(
        target, n_tgt
#ifdef __USE_CUDA__
        , type_tgt, stream
#endif
      );
    }
    if (res){
#ifdef __USE_CUDA__
      IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        //__PRINT_INFO__("IvyMemoryHelpers::copy_data_multistream_ext: Running parallel copy.\n");
#ifndef __CUDA_DEVICE_CODE__
        ev_begin.record(stream);
#endif
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream* stream_ptr = &sr_src_altr;
#else
          IvyGPUStream* stream_ptr = &stream;
#endif
          res &= allocate_memory(d_source, n_src, MemoryType::Device, *stream_ptr);
#ifndef __CUDA_DEVICE_CODE__
          sr_src_altr.wait(ev_begin);
#endif
          res &= transfer_memory(d_source, source, n_src, MemoryType::Device, type_src, *stream_ptr);
#ifndef __CUDA_DEVICE_CODE__
          ev_src_altr_tgt_al.record(sr_src_altr);
          stream.wait(ev_src_altr_tgt_al);
#endif
        }
        if (!tgt_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream* stream_tgt_al_ptr = &sr_tgt_al;
#else
          IvyGPUStream* stream_tgt_al_ptr = &stream;
#endif
          T* d_target = nullptr;
          res &= allocate_memory(d_target, n_tgt, MemoryType::Device, *stream_tgt_al_ptr);
#ifndef __CUDA_DEVICE_CODE__
          ev_src_altr_tgt_al.record(sr_tgt_al);
          stream.wait(ev_src_altr_tgt_al);
#endif
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_tgt, n_src);
          res &= transfer_memory(target, d_target, n_tgt, type_tgt, MemoryType::Device, stream);
#ifndef __CUDA_DEVICE_CODE__
          ev_copy.record(stream);

          IvyGPUStream* stream_tgt_free_ptr = &sr_tgt_free;
          sr_tgt_free.wait(ev_copy);
#else
          IvyGPUStream* stream_tgt_free_ptr = &stream;
#endif
          free_memory(d_target, n_tgt, MemoryType::Device, *stream_tgt_free_ptr);
        }
        else{
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(target, d_source, n_tgt, n_src);
#ifndef __CUDA_DEVICE_CODE__
          ev_copy.record(stream);
#endif
        }
        if (!src_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream* stream_src_free_ptr = &sr_src_free;
          sr_src_free.wait(ev_copy);
#else
          IvyGPUStream* stream_src_free_ptr = &stream;
#endif
          free_memory(d_source, n_src, MemoryType::Device, *stream_src_free_ptr);
        }
      }
      else{
        if (tgt_on_device!=src_on_device){
          __PRINT_ERROR__("IvyMemoryHelpers::copy_data_multistream_ext: Failed to copy data between host and device.\n");
          assert(0);
        }
        //__PRINT_INFO__("IvyMemoryHelpers::copy_data_multistream_ext: Running serial copy.\n");
#else
      {
#endif
        for (size_t i=0; i<n_tgt; i++) target[i] = source[(n_src==1 ? 0 : i)];
      }
    }
    return res;
  }

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_single_stream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , MemoryType type_tgt, MemoryType type_src
    , IvyGPUStream& stream
#endif
  ){
    bool res = true;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const tgt_on_device = is_device_memory(type_tgt) || is_unified_memory(type_tgt);
    bool const src_on_device = is_device_memory(type_src) || is_unified_memory(type_src);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_single_stream: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data_single_stream: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
      res &= free_memory(
        target, n_tgt_init
#ifdef __USE_CUDA__
        , type_tgt, stream
#endif
      );
      res &= allocate_memory(
        target, n_tgt
#ifdef __USE_CUDA__
        , type_tgt, stream
#endif
      );
    }
    if (res){
#ifdef __USE_CUDA__
      IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        //__PRINT_INFO__("IvyMemoryHelpers::copy_data_single_stream: Running parallel copy.\n");
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
          res &= allocate_memory(d_source, n_src, MemoryType::Device, stream);
          res &= transfer_memory(d_source, source, n_src, MemoryType::Device, type_src, stream);
        }
        if (!tgt_on_device){
          T* d_target = nullptr;
          res &= allocate_memory(d_target, n_tgt, MemoryType::Device, stream);
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_tgt, n_src);
          res &= transfer_memory(target, d_target, n_tgt, type_tgt, MemoryType::Device, stream);
          free_memory(d_target, n_tgt, MemoryType::Device, stream);
        }
        else{
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(target, d_source, n_tgt, n_src);
        }
        if (!src_on_device){
          free_memory(d_source, n_src, MemoryType::Device, stream);
        }
      }
      else{
        if (tgt_on_device!=src_on_device){
          __PRINT_ERROR__("IvyMemoryHelpers::copy_data_single_stream: Failed to copy data between host and device.\n");
          assert(0);
        }
        //__PRINT_INFO__("IvyMemoryHelpers::copy_data_single_stream: Running serial copy.\n");
#else
      {
#endif
        for (size_t i=0; i<n_tgt; i++) target[i] = source[(n_src==1 ? 0 : i)];
      }
    }
    return res;
  }

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
