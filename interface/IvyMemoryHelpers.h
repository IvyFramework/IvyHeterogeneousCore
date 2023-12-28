#ifndef IVYMEMORYHELPERS_H
#define IVYMEMORYHELPERS_H

#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif
#include "config/IvyConfig.h"
#include "IvyBasicTypes.h"
#include "IvyException.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyCstdio.h"


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

  /*
  copy_data_kernel: Kernel function for copying data from a pointer of type U to a pointer of type T.
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  - n_tgt: Number of elements in the target array.
  - n_src: Number of elements in the source array with the constraint (n_src==n_tgt || n_src==1).
  */
  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, size_t n_tgt, size_t n_src);
#endif

  /*
  copy_data: Copies data from a pointer of type U to a pointer of type T.
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  - n_tgt_init: Number of elements in the target array before the copy. If the target array is not null, it is freed.
  - n_src: Number of elements in the source array and the target array after the copy.
  When using CUDA, the following additional arguments are required:
  - direction: Direction of the transfer. In device code, this is ignored, and the copy is always device-to-device.
  - stream: CUDA stream to use for the copy.
  If stream is anything other than cudaStreamLegacy, the copy is asynchronous, even in device code.
  */
  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_src
#ifdef __USE_CUDA__
    , TransferDirection direction
    , cudaStream_t stream
#endif
  );
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
      if (stream==cudaStreamLegacy){
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMalloc((void**) &data, n*sizeof(T)));
      }
      else{
        __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocAsync((void**) &data, n*sizeof(T), stream));
      }
      if (sizeof...(Args)>0){
        T* temp = nullptr;
        allocate_memory(temp, n, false, stream, args...);
        transfer_memory(data, temp, n, TransferDirection::HostToDevice, stream);
        free_memory(temp, n, false, stream);
      }
      return true;
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

  template<typename T, typename U> __CUDA_GLOBAL__ void copy_data_kernel(T* target, U* source, size_t n_tgt, size_t n_src){
    IvyBlockThread_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      printf("IvyMemoryHelpers::copy_data_kernel: Invalid values for n_src=%Iu, n_tgt=%Iu\n", n_src, n_tgt);
#else
      printf("IvyMemoryHelpers::copy_data_kernel: Invalid values for n_src=%zu, n_tgt=%zu\n", n_src, n_tgt);
#endif
      assert(0);
    }
    if (i<n_tgt) target[i] = source[(n_src==1 ? 0 : i)];
  }
#endif

  template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_src
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
    if (n_src==0 || !source) return false;
    if (n_tgt_init!=n_src){
      res &= free_memory(
        target, n_tgt_init
#ifdef __USE_CUDA__
        , tgt_on_device, stream
#endif
      );
      res &= allocate_memory(
        target, n_src
#ifdef __USE_CUDA__
        , tgt_on_device, stream
#endif
      );
    }
    if (res){
#ifdef __USE_CUDA__
      IvyBlockThread_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, n_src)){
        U* d_source = (src_on_device ? source : nullptr);
        if (!src_on_device){
          res &= allocate_memory(d_source, n_src, true, stream);
          res &= transfer_memory(d_source, source, n_src, TransferDirection::HostToDevice, stream);
        }
        if (!tgt_on_device){
          T* d_target = nullptr;
          res &= allocate_memory(d_target, n_src, true, stream);
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(d_target, d_source, n_src, n_src);
          res &= transfer_memory(target, d_target, n_src, TransferDirection::DeviceToHost, stream);
          free_memory(d_target, n_src, true, stream);
        }
        else{
          copy_data_kernel<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(target, d_source, n_src, n_src);
        }
        if (!src_on_device) free_memory(d_source, n_src, true, stream);
      }
      else{
        if (tgt_on_device!=src_on_device){
          printf("IvyMemoryHelpers::copy_data: Failed to copy data between host and device.\n");
          assert(0);
        }
#else
      {
#endif
        for (size_t i=0; i<n_src; i++) target[i] = source[i];
      }
    }
    return res;
  }
}


#endif
