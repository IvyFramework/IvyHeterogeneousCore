#ifndef IVYMEMORYHELPERS_H
#define IVYMEMORYHELPERS_H


/*
IvyMemoryHelpers: A collection of functions for allocating, freeing, and copying memory.
The functions are overloaded for both host and device code when CUDA is enabled.
If CUDA is disabled, allocation and freeing are done with new and delete.
Otherwise, allocation and freeing conventions are as follows:
- Host code, host memory: new/delete
- Device code, device memory: new/delete
- Host code, device memory: cudaMalloc/cudaFree
If the stream argument is anything other than cudaStreamLegacy, allocation and freeing are asynchronous.
- Device code, host memory: Disallowed
*/


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif
#include "config/IvyConfig.h"
#include "IvyBasicTypes.h"
#include "IvyException.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyCstdio.h"
#include "stream/IvyStream.h"


namespace IvyMemoryHelpers{
  using size_t = IvyTypes::size_t;
  using ptrdiff_t = IvyTypes::ptrdiff_t;

  enum class TransferDirection{
    HostToDevice,
    DeviceToHost,
    HostToHost,
    DeviceToDevice
  };

  /*
  allocate_memory: Allocates memory for an array of type T of size n. Constructors are called for the arguments args.
  - data: Pointer to the target data.
  - n: Number of elements.
  - args: Arguments for the constructors of the elements.
  When using CUDA, the following additional arguments are required:
  - use_cuda_device_mem: In host code, this flag determines whether to use device or host memory.
  In device code, this flag is ignored, and the memory is always allocated on the device.
  - stream: In host code, this is the CUDA stream to use for the allocation.
  If stream is anything other than cudaStreamLegacy, the allocation is asynchronous.
  In device code, any allocation and object construction operations are always synchronous with the running thread.
  */
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem
    , cudaStream_t stream
#endif
    , Args&&... args
  );

  /*
  free_memory: Frees memory for an array of type T of size n in a way consistent with allocate_memory.
  - data: Pointer to the data.
  - n: Number of elements.
  When using CUDA, the following additional arguments are required:
  - use_cuda_device_mem: In host code, this flag determines whether to the data resides in device or host memory.
  In device code, this flag is ignored, and the memory is always freed from the device.
  - stream: In host code, this is the CUDA stream to use for the deallocation.
  If stream is anything other than cudaStreamLegacy, the deallocation is asynchronous.
  In device code, any deallocation operations are always synchronous with the running thread.
  */
  template<typename T> __CUDA_HOST_DEVICE__ bool free_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem
    , cudaStream_t stream
#endif
  );

#ifdef __USE_CUDA__
  /*
  get_cuda_transfer_direction: Translates the TransferDirection enum to the corresponding cudaMemcpyKind type.
  */
  __CUDA_HOST_DEVICE__ cudaMemcpyKind get_cuda_transfer_direction(TransferDirection direction);

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
  - direction: Direction of the transfer. In device code, this is ignored, and the transfer is always device-to-device.
  - stream: CUDA stream to use for the transfer.
  In device code, since cudaMemcpy is not available, we always use cudaMemcpyAsync instead.
  */
  template<typename T> __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    TransferDirection direction,
    cudaStream_t stream
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
  - direction: Direction of the transfer. In device code, this is ignored, and the copy is always device-to-device.
  - stream: CUDA stream to use for the copy.
  If stream is anything other than cudaStreamLegacy, the copy is asynchronous, even in device code.
  */
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_multistream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , TransferDirection direction
    , cudaStream_t stream
#endif
  );

  /*
  copy_data_single_stream: Same as copy_data_multistream,
  but all operations are done over a single stream without creating additional streams or events
  if using CUDA.
  */
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_single_stream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , TransferDirection direction
    , cudaStream_t stream
#endif
  );

  /*
  copy_data: Alias to choose between copy_data_multistream and copy_data_single_stream.
  It turns out that copy_data_single_stream is faster even when operating over >1M elements
  because the creation of additional streams and events is very expensive.
  */
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , TransferDirection direction
    , cudaStream_t stream
#endif
  ){
    return copy_data_single_stream(
      target, source,
      n_tgt_init, n_tgt, n_src
#ifdef __USE_CUDA__
      , direction, stream
#endif
    );
  }
}


namespace IvyMemoryHelpers{
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ bool allocate_memory(
    T*& data,
    size_t n
#ifdef __USE_CUDA__
    , bool use_cuda_device_mem
    , cudaStream_t stream
#endif
    , Args&&... args
  ){
    if (n==0 || data) return false;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    if (use_cuda_device_mem){
      bool res = true;
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc((void**) &data, n*sizeof(T)));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocAsync((void**) &data, n*sizeof(T), stream));
      }
      if (sizeof...(Args)>0){
        T* temp = nullptr;
        res &= allocate_memory(temp, n, false, stream, args...);
        res &= transfer_memory(data, temp, n, TransferDirection::HostToDevice, stream);
        res &= free_memory(temp, n, false, stream);
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
    , bool use_cuda_device_mem
    , cudaStream_t stream
#endif
  ){
    if (!data) return true;
    if (n==0) return false;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    if (use_cuda_device_mem){
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(data));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFreeAsync(data, stream));
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
  __CUDA_HOST_DEVICE__ inline cudaMemcpyKind get_cuda_transfer_direction(TransferDirection direction){
#ifndef __CUDA_DEVICE_CODE__
    switch (direction){
      case TransferDirection::HostToDevice:
        return cudaMemcpyHostToDevice;
      case TransferDirection::DeviceToHost:
        return cudaMemcpyDeviceToHost;
      case TransferDirection::HostToHost:
        return cudaMemcpyHostToHost;
      case TransferDirection::DeviceToDevice:
        return cudaMemcpyDeviceToDevice;
      default:
        printf("IvyMemoryHelpers::get_cuda_transfer_direction: Unknown transfer direction.\n");
        assert(0);
    }
#else
    return cudaMemcpyDeviceToDevice;
#endif
  }

  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, size_t n_tgt, size_t n_src){
    IvyBlockThread_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      printf("IvyMemoryHelpers::copy_data_kernel: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      printf("IvyMemoryHelpers::copy_data_kernel: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (i<n_tgt) target[i] = source[(n_src==1 ? 0 : i)];
  }

  template<typename T> __CUDA_HOST_DEVICE__ bool transfer_memory(
    T*& tgt, T* const& src, size_t n,
    TransferDirection direction,
    cudaStream_t stream
  ){
    if (!tgt || !src) return false;
    auto dir = get_cuda_transfer_direction(direction);
#ifndef __CUDA_DEVICE_CODE__
    if (stream==cudaStreamLegacy){
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpy(tgt, src, n*sizeof(T), dir));
    }
    else
#endif
    {
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMemcpyAsync(tgt, src, n*sizeof(T), dir, stream));
    }
    return true;
  }
#endif

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_multistream(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
#ifdef __USE_CUDA__
    , TransferDirection direction
    , cudaStream_t stream
#endif
  ){
    bool res = true;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const tgt_on_device = (direction==TransferDirection::HostToDevice || direction==TransferDirection::DeviceToDevice);
    bool const src_on_device = (direction==TransferDirection::DeviceToHost || direction==TransferDirection::DeviceToDevice);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      printf("IvyMemoryHelpers::copy_data_multistream: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      printf("IvyMemoryHelpers::copy_data_multistream: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
      res &= free_memory(
        target, n_tgt_init
#ifdef __USE_CUDA__
        , tgt_on_device, stream
#endif
      );
      res &= allocate_memory(
        target, n_tgt
#ifdef __USE_CUDA__
        , tgt_on_device, stream
#endif
      );
    }
    if (res){
#ifdef __USE_CUDA__
      IvyBlockThread_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        //printf("IvyMemoryHelpers::copy_data_multistream: Running parallel copy.\n");
#ifndef __CUDA_DEVICE_CODE__
        IvyGPUEvent ev_begin, ev_src_altr_tgt_al, ev_copy;
        ev_begin.record(stream);
#endif
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream sr_src_altr(IvyGPUStream::StreamFlags::NonBlocking);
          cudaStream_t* stream_ptr = &(sr_src_altr.stream());
#else
          cudaStream_t* stream_ptr = &stream;
#endif
          res &= allocate_memory(d_source, n_src, true, *stream_ptr);
#ifndef __CUDA_DEVICE_CODE__
          sr_src_altr.wait(ev_begin);
#endif
          res &= transfer_memory(d_source, source, n_src, TransferDirection::HostToDevice, *stream_ptr);
#ifndef __CUDA_DEVICE_CODE__
          ev_src_altr_tgt_al.record(sr_src_altr);
          __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamWaitEvent(stream, ev_src_altr_tgt_al.event(), cudaEventWaitDefault));
#endif
        }
        if (!tgt_on_device){
#ifndef __CUDA_DEVICE_CODE__
          IvyGPUStream sr_tgt_al(IvyGPUStream::StreamFlags::NonBlocking);
          cudaStream_t* stream_tgt_al_ptr = &(sr_tgt_al.stream());
#else
          cudaStream_t* stream_tgt_al_ptr = &stream;
#endif
          T* d_target = nullptr;
          res &= allocate_memory(d_target, n_tgt, true, *stream_tgt_al_ptr);
#ifndef __CUDA_DEVICE_CODE__
          ev_src_altr_tgt_al.record(sr_tgt_al);
          __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamWaitEvent(stream, ev_src_altr_tgt_al.event(), cudaEventWaitDefault));
#endif
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_tgt, n_src);
          res &= transfer_memory(target, d_target, n_tgt, TransferDirection::DeviceToHost, stream);
#ifndef __CUDA_DEVICE_CODE__
          ev_copy.record(stream);

          IvyGPUStream sr_tgt_free(IvyGPUStream::StreamFlags::NonBlocking);
          cudaStream_t* stream_tgt_free_ptr = &(sr_tgt_free.stream());
          sr_tgt_free.wait(ev_copy);
#else
          cudaStream_t* stream_tgt_free_ptr = &stream;
#endif
          free_memory(d_target, n_tgt, true, *stream_tgt_free_ptr);
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
          cudaStream_t* stream_src_free_ptr = &(sr_src_free.stream());
          sr_src_free.wait(ev_copy);
#else
          cudaStream_t* stream_src_free_ptr = &stream;
#endif
          free_memory(d_source, n_src, true, *stream_src_free_ptr);
        }
      }
      else{
        if (tgt_on_device!=src_on_device){
          printf("IvyMemoryHelpers::copy_data_multistream: Failed to copy data between host and device.\n");
          assert(0);
        }
        //printf("IvyMemoryHelpers::copy_data_multistream: Running serial copy.\n");
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
    , TransferDirection direction
    , cudaStream_t stream
#endif
  ){
    bool res = true;
#ifdef __USE_CUDA__
#ifndef __CUDA_DEVICE_CODE__
    bool const tgt_on_device = (direction==TransferDirection::HostToDevice || direction==TransferDirection::DeviceToDevice);
    bool const src_on_device = (direction==TransferDirection::DeviceToHost || direction==TransferDirection::DeviceToDevice);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      printf("IvyMemoryHelpers::copy_data_single_stream: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      printf("IvyMemoryHelpers::copy_data_single_stream: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
      res &= free_memory(
        target, n_tgt_init
#ifdef __USE_CUDA__
        , tgt_on_device, stream
#endif
      );
      res &= allocate_memory(
        target, n_tgt
#ifdef __USE_CUDA__
        , tgt_on_device, stream
#endif
      );
    }
    if (res){
#ifdef __USE_CUDA__
      IvyBlockThread_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        //printf("IvyMemoryHelpers::copy_data_single_stream: Running parallel copy.\n");
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
          res &= allocate_memory(d_source, n_src, true, stream);
          res &= transfer_memory(d_source, source, n_src, TransferDirection::HostToDevice, stream);
        }
        if (!tgt_on_device){
          T* d_target = nullptr;
          res &= allocate_memory(d_target, n_tgt, true, stream);
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_tgt, n_src);
          res &= transfer_memory(target, d_target, n_tgt, TransferDirection::DeviceToHost, stream);
          free_memory(d_target, n_tgt, true, stream);
        }
        else{
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(target, d_source, n_tgt, n_src);
        }
        if (!src_on_device){
          free_memory(d_source, n_src, true, stream);
        }
      }
      else{
        if (tgt_on_device!=src_on_device){
          printf("IvyMemoryHelpers::copy_data_single_stream: Failed to copy data between host and device.\n");
          assert(0);
        }
        printf("IvyMemoryHelpers::copy_data_single_stream: Running serial copy.\n");
#else
      {
#endif
        for (size_t i=0; i<n_tgt; i++) target[i] = source[(n_src==1 ? 0 : i)];
      }
    }
    return res;
  }
}


#endif
