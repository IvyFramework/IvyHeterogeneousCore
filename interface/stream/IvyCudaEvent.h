#ifndef IVYCUDAEVENT_H
#define IVYCUDAEVENT_H


#include "config/IvyCudaException.h"
#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"
#include "stream/IvyCudaEvent.hh"


#ifdef __USE_CUDA__
namespace IvyStreamUtils{
  template<> __HOST_DEVICE__ void createStreamEvent(cudaEvent_t& ev){
#ifndef __CUDA_DEVICE_CODE__
    __CHECK_OR_EXIT_WITH_ERROR__(cudaEventCreate(&ev));
#endif
  }
  template<> __HOST_DEVICE__ void destroyStreamEvent(cudaEvent_t& ev){
#ifndef __CUDA_DEVICE_CODE__
    __CHECK_OR_EXIT_WITH_ERROR__(cudaEventDestroy(ev));
#endif
  }
}

__HOST__ IvyCudaEvent::IvyCudaEvent(EventFlags flags) : IvyBaseStreamEvent<cudaStream_t>()
{
  is_owned_ = true;
  flags_ = get_event_flags(flags);

  IvyStreamUtils::createStreamEvent(event_);
}

__HOST__ void IvyCudaEvent::record(cudaStream_t& stream, unsigned int rcd_flags){
  __CHECK_OR_EXIT_WITH_ERROR__(cudaEventRecordWithFlags(event_, stream, rcd_flags));
}
__HOST__ void IvyCudaEvent::synchronize(){
  __CHECK_OR_EXIT_WITH_ERROR__(cudaEventSynchronize(event_));
}

__HOST__ void IvyCudaEvent::record(IvyCudaStream& stream, RecordFlags rcd_flags){
  this->record(stream.stream(), get_record_flags(rcd_flags));
}
__HOST__ void IvyCudaEvent::record(cudaStream_t& stream, RecordFlags rcd_flags){
  this->record(stream, get_record_flags(rcd_flags));
}

__HOST__ float IvyCudaEvent::elapsed_time(IvyCudaEvent const& start) const{
  float res = -1;
  if (flags_ != cudaEventDisableTiming && start.flags() != cudaEventDisableTiming){
    __CHECK_OR_EXIT_WITH_ERROR__(cudaEventElapsedTime(&res, start.event(), event_));
  }
  return res;
}
__HOST__ float IvyCudaEvent::elapsed_time(IvyCudaEvent const& start, IvyCudaEvent const& end){
  float res = -1;
  if (start.flags() != cudaEventDisableTiming && end.flags() != cudaEventDisableTiming){
    __CHECK_OR_EXIT_WITH_ERROR__(cudaEventElapsedTime(&res, start.event(), end.event()));
  }
  return res;
}

__HOST_DEVICE__ unsigned int IvyCudaEvent::get_event_flags(EventFlags const& flags){
  switch (flags){
  case EventFlags::Default:
    return cudaEventDefault;
  case EventFlags::BlockingSync:
    return cudaEventBlockingSync;
  case EventFlags::DisableTiming:
    return cudaEventDisableTiming;
  case EventFlags::Interprocess:
    return cudaEventInterprocess;
  default:
    __PRINT_ERROR__("IvyCudaEvent::get_event_flags: Unknown flag option...\n");
    assert(0);
  }
  return cudaEventDisableTiming;
}
__HOST_DEVICE__ unsigned int IvyCudaEvent::get_record_flags(RecordFlags const& flags){
  switch (flags){
  case RecordFlags::Default:
    return cudaEventRecordDefault;
  case RecordFlags::External:
    return cudaEventRecordExternal;
  default:
    __PRINT_ERROR__("IvyCudaEvent::get_record_flags: Unknown flag option...\n");
    assert(0);
  }
  return cudaEventRecordDefault;
}
__HOST_DEVICE__ unsigned int IvyCudaEvent::get_wait_flags(WaitFlags const& flags){
  switch (flags){
  case WaitFlags::Default:
    return cudaEventWaitDefault;
  case WaitFlags::External:
    return cudaEventWaitExternal;
  default:
    __PRINT_ERROR__("IvyCudaEvent::get_wait_flags: Unknown flag option...\n");
    assert(0);
  }
  return cudaEventWaitDefault;
}

namespace std_util{
  __HOST_DEVICE__ void swap(IvyCudaEvent& a, IvyCudaEvent& b){ a.swap(b); }
}

#endif


#endif
