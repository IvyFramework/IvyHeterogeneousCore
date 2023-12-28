#ifndef IVYSTREAM_H
#define IVYSTREAM_H


#include "IvyCudaStream.h"
#include "IvyCudaEvent.h"


/*
These are master typedefs to interface all types of streams and events.
*/
#ifdef __USE_CUDA__

typedef IvyCudaStream IvyGPUStream;
typedef IvyCudaEvent IvyGPUEvent;

#define GlobalGPUStreamRaw cudaStreamLegacy
#define GlobalGPUStream IvyGPUStream(cudaStreamLegacy, false)

#else

typedef void* IvyGPUStream;
typedef void* IvyGPUEvent;

#define GlobalGPUStreamRaw nullptr
#define GlobalGPUStream nullptr

#endif


#endif
