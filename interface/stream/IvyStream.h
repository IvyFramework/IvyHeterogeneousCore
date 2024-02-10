#ifndef IVYSTREAM_H
#define IVYSTREAM_H


#include "stream/IvyCudaStream.h"
#include "stream/IvyCudaEvent.h"
#include "config/IvyCudaFlags.h"
#include "stream/IvyBlankStream.h"
#include "stream/IvyBlankStreamEvent.h"


/*
These are master typedefs to interface all types of streams and events.
*/
#ifdef __USE_CUDA__

namespace IvyStreamUtils{
  typedef IvyCudaEvent IvyGPUEvent;
  typedef IvyCudaStream IvyGPUStream;
  typedef IvyGPUStream::fcn_callback_t IvyGPUStream_CBFcn_t;
}
using IvyStreamUtils::IvyGPUEvent;
using IvyStreamUtils::IvyGPUStream;
using IvyStreamUtils::IvyGPUStream_CBFcn_t;

#define GlobalGPUStreamRaw GlobalCudaStreamRaw

#else

namespace IvyStreamUtils{
  typedef IvyBlankStreamEvent IvyGPUEvent;
  typedef IvyBlankStream IvyGPUStream;
}
using IvyStreamUtils::IvyGPUEvent;
using IvyStreamUtils::IvyGPUStream;

#define GlobalGPUStreamRaw IvyStreamUtils::GlobalBlankStreamRaw

#endif


#define GlobalGPUStream IvyGPUStream(GlobalGPUStreamRaw, false)
namespace IvyStreamUtils{
  __CUDA_HOST_DEVICE__ IvyGPUStream* make_global_gpu_stream(){ return make_stream<IvyGPUStream>(GlobalGPUStreamRaw, false); }
}
#define operate_with_GPU_stream_from_pointer(ptr, ref, CALL) \
IvyGPUStream* new_##ptr = nullptr; \
if (!ptr) new_##ptr = IvyStreamUtils::make_global_gpu_stream(); \
IvyGPUStream& ref = (ptr ? *ptr : *new_##ptr); \
CALL \
IvyStreamUtils::destroy_stream(new_##ptr);


#endif
