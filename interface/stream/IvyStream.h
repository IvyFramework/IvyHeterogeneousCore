#ifndef IVYSTREAM_H
#define IVYSTREAM_H

/**
 * @file IvyStream.h
 * @brief Unified stream/event aliases bridging CUDA and non-CUDA backends.
 */


#include "stream/IvyCudaStream.h"
#include "stream/IvyCudaEvent.h"
#include "config/IvyCudaFlags.h"
#include "stream/IvyBlankStream.h"
#include "stream/IvyBlankStreamEvent.h"


/**
These are master typedefs to interface all types of streams and events.
*/
#ifdef __USE_CUDA__

namespace IvyStreamUtils{
  /** @brief Active backend event type in CUDA builds. */
  typedef IvyCudaEvent IvyGPUEvent;
  /** @brief Active backend stream type in CUDA builds. */
  typedef IvyCudaStream IvyGPUStream;
  /** @brief Callback function type used by GPU stream callbacks. */
  typedef IvyGPUStream::fcn_callback_t IvyGPUStream_CBFcn_t;
}
using IvyStreamUtils::IvyGPUEvent;
using IvyStreamUtils::IvyGPUStream;
using IvyStreamUtils::IvyGPUStream_CBFcn_t;

#define GlobalGPUStreamRaw GlobalCudaStreamRaw

#else

namespace IvyStreamUtils{
  /** @brief Active backend event type in non-CUDA builds. */
  typedef IvyBlankStreamEvent IvyGPUEvent;
  /** @brief Active backend stream type in non-CUDA builds. */
  typedef IvyBlankStream IvyGPUStream;
}
using IvyStreamUtils::IvyGPUEvent;
using IvyStreamUtils::IvyGPUStream;

#define GlobalGPUStreamRaw IvyStreamUtils::GlobalBlankStreamRaw

#endif


#define GlobalGPUStream IvyGPUStream(GlobalGPUStreamRaw, false)
namespace IvyStreamUtils{
  /** @brief Build a non-owning wrapper to the global backend stream. */
  __HOST_DEVICE__ IvyGPUStream* make_global_gpu_stream(){ return make_stream<IvyGPUStream>(GlobalGPUStreamRaw, false); }
}

#define build_GPU_stream_reference_from_pointer(ptr, ref) \
IvyGPUStream* new_##ptr = nullptr; \
if (!ptr) new_##ptr = IvyStreamUtils::make_global_gpu_stream(); \
IvyGPUStream& ref = (ptr ? *ptr : *new_##ptr);
#define destroy_GPU_stream_reference_from_pointer(ptr) \
IvyStreamUtils::destroy_stream(new_##ptr);
#define operate_with_GPU_stream_from_pointer(ptr, ref, CALL) \
build_GPU_stream_reference_from_pointer(ptr, ref) \
CALL \
destroy_GPU_stream_reference_from_pointer(ptr)


#endif
