#ifndef IVYMEMORYTYPES_H
#define IVYMEMORYTYPES_H


#include "config/IvyCompilerConfig.h"


namespace IvyMemoryHelpers{
  enum class IvyMemoryType : unsigned char{
    Host,
    GPU,
    PageLocked,
    Unified,
    UnifiedPrefetched,
    nMemoryTypes
  };

  /*
  get_mem_type_name: Returns the C-string name of the memory type.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr const char* get_memory_type_name(IvyMemoryType type);

  /*
  is_host_memory: Returns true if the memory type is host memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool is_host_memory(IvyMemoryType type);

  /*
  is_gpu_memory: Returns true if the memory type is GPU device memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool is_gpu_memory(IvyMemoryType type);

  /*
  is_unified_memory: Returns true if the memory type is unified memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool is_unified_memory(IvyMemoryType type);

  /*
  is_pagelocked: Returns true if the memory type is page-locked host memory.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool is_pagelocked(IvyMemoryType type);

  /*
  is_prefetched: Returns true if the memory type is unified memory that has been prefetched.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool is_prefetched(IvyMemoryType type);

  /*
  use_device_GPU: Returns true if the memory type is associated with the GPU
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool use_device_GPU(IvyMemoryType type);

  /*
  use_device_acc: Returns true if the memory type is associated with hardware accelerators.
  */
  __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr bool use_device_acc(IvyMemoryType type);

  /*
  use_device_host: Simply the same as !use_device_acc.
  */
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr bool use_device_host(IvyMemoryType type);

  /*
  run_acc_on_host: Check if we are in the host context and the specified memory type needs to run hardware acceleration.
  */
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr bool run_acc_on_host(IvyMemoryType type);

  /*
  get_execution_default_memory: Returns the default memory type for the current execution environment.
  For host code or if not using CUDA, this is IvyMemoryType::Host.
  Otherwise, for device code, this is IvyMemoryType::GPU.
  */
  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory();
}

namespace IvyMemoryHelpers{
  __CUDA_HOST_DEVICE__ constexpr const char* get_memory_type_name(IvyMemoryType type){
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    switch(type){
      case IvyMemoryType::Host: return "CPU";
      case IvyMemoryType::GPU: return "GPU";
      case IvyMemoryType::PageLocked: return "PageLocked";
      case IvyMemoryType::Unified: return "Unified";
      case IvyMemoryType::UnifiedPrefetched: return "UnifiedPrefetched";
      default: return "Unknown";
    }
  }

  __CUDA_HOST_DEVICE__ constexpr bool is_host_memory(IvyMemoryType type){
    return type==IvyMemoryType::Host || type==IvyMemoryType::PageLocked;
  }

  __CUDA_HOST_DEVICE__ constexpr bool is_gpu_memory(IvyMemoryType type){
    return type==IvyMemoryType::GPU;
  }

  __CUDA_HOST_DEVICE__ constexpr bool is_unified_memory(IvyMemoryType type){
    return type==IvyMemoryType::Unified || type==IvyMemoryType::UnifiedPrefetched;
  }

  __CUDA_HOST_DEVICE__ constexpr bool is_pagelocked(IvyMemoryType type){
    return type==IvyMemoryType::PageLocked;
  }

  __CUDA_HOST_DEVICE__ constexpr bool is_prefetched(IvyMemoryType type){
    return type==IvyMemoryType::UnifiedPrefetched;
  }

  __CUDA_HOST_DEVICE__ constexpr bool use_device_GPU(IvyMemoryType type){
    return
#if defined(__USE_CUDA__)
      true
#else
      false
#endif
      && (is_gpu_memory(type) || is_unified_memory(type));
  }

  __CUDA_HOST_DEVICE__ constexpr bool use_device_host(IvyMemoryType type){ return !use_device_acc(type); }

  // For now, we can only test for GPUs. Once FPGAs or other devices are added, this function would need to be modified.
  __CUDA_HOST_DEVICE__ constexpr bool use_device_acc(IvyMemoryType type){
    static_assert(__STATIC_CAST__(unsigned char, IvyMemoryType::nMemoryTypes)==5);
    return use_device_GPU(type);
  }

#if (DEVICE_CODE == DEVICE_CODE_GPU)
  __CUDA_HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory(){ return IvyMemoryType::GPU; }
#elif (DEVICE_CODE == DEVICE_CODE_HOST)
  __CUDA_HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory(){ return IvyMemoryType::Host; }
#else
  __CUDA_HOST_DEVICE__ constexpr IvyMemoryType get_execution_default_memory(){
    static_assert(0, "IvyMemoryHelpers::get_execution_default_memory: Unknown device type.");
    return IvyMemoryType::nMemoryTypes;
  }
#endif

  __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ constexpr bool run_acc_on_host(IvyMemoryType type){
    constexpr IvyMemoryType def_mem_type = get_execution_default_memory();
    return (use_device_host(def_mem_type) && use_device_acc(type));
  }
}


// Aliases for std_ivy namespace
namespace std_ivy{
  using IvyMemoryType = IvyMemoryHelpers::IvyMemoryType;
}


#endif
