#ifndef IVYSTREAM_H
#define IVYSTREAM_H


#include "IvyCudaStream.h"
#include "IvyCudaEvent.h"
#include "config/IvyCudaFlags.h"


/*
These are master typedefs to interface all types of streams and events.
*/
#ifdef __USE_CUDA__

#define GlobalGPUStreamRaw GlobalCudaStreamRaw
namespace IvyGlobal{
  typedef IvyCudaEvent IvyGPUEvent;
  typedef IvyCudaStream IvyGPUStream;
  typedef IvyGPUStream::fcn_callback_t IvyGPUStream_CBFcn_t;
}
using IvyGlobal::IvyGPUEvent;
using IvyGlobal::IvyGPUStream;
using IvyGlobal::IvyGPUStream_CBFcn_t;

#define GlobalGPUStreamRaw GlobalCudaStreamRaw
#define GlobalGPUStream IvyGPUStream(GlobalGPUStreamRaw, false)
#define operate_with_GPU_stream_from_pointer(ptr, ref, CALL) \
IvyGPUStream* new_##ptr = nullptr; \
if (!ptr) new_##ptr = new GlobalGPUStream; \
IvyGPUStream& ref = (ptr ? *ptr : *new_##ptr); \
CALL \
if (new_##ptr) delete new_##ptr;


#else

typedef void* IvyGPUStream;
typedef void* IvyGPUEvent;

#define GlobalGPUStreamRaw nullptr
#define GlobalGPUStream nullptr

#endif


#endif
