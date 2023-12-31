#ifndef IVYCUDAEVENT_H
#define IVYCUDAEVENT_H


#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"
#include "stream/IvyCudaEvent.hh"

#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "IvyException.h"


__CUDA_HOST__ IvyCudaEvent::IvyCudaEvent(EventFlags flags) :
  flags_(get_event_flags(flags))
{
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventCreate(&event_, flags_));
}
__CUDA_HOST__ IvyCudaEvent::~IvyCudaEvent(){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventDestroy(event_));
}

__CUDA_HOST_DEVICE__ unsigned int const& IvyCudaEvent::flags() const{ return flags_; }
__CUDA_HOST_DEVICE__ cudaEvent_t const& IvyCudaEvent::event() const{ return event_; }
__CUDA_HOST_DEVICE__ IvyCudaEvent::operator cudaEvent_t const& () const{ return event_; }

__CUDA_HOST_DEVICE__ unsigned int& IvyCudaEvent::flags(){ return flags_; }
__CUDA_HOST_DEVICE__ cudaEvent_t& IvyCudaEvent::event(){ return event_; }
__CUDA_HOST_DEVICE__ IvyCudaEvent::operator cudaEvent_t& (){ return event_; }

__CUDA_HOST__ void IvyCudaEvent::record(IvyCudaStream& stream, RecordFlags rcd_flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventRecordWithFlags(event_, stream.stream(), get_record_flags(rcd_flags)));
}
__CUDA_HOST__ void IvyCudaEvent::record(cudaStream_t& stream, RecordFlags rcd_flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventRecordWithFlags(event_, stream, get_record_flags(rcd_flags)));
}
__CUDA_HOST__ void IvyCudaEvent::synchronize(){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventSynchronize(event_));
}
__CUDA_HOST__ void IvyCudaEvent::wait(IvyCudaStream& stream, WaitFlags wait_flags){
  stream.wait(*this, wait_flags);
}
__CUDA_HOST__ float IvyCudaEvent::elapsed_time(IvyCudaEvent const& start) const{
  float res = -1;
  if (flags_ != cudaEventDisableTiming && start.flags() != cudaEventDisableTiming){
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventElapsedTime(&res, start.event(), event_));
  }
  return res;
}
__CUDA_HOST__ float IvyCudaEvent::elapsed_time(IvyCudaEvent const& start, IvyCudaEvent const& end){
  float res = -1;
  if (start.flags() != cudaEventDisableTiming && end.flags() != cudaEventDisableTiming){
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventElapsedTime(&res, start.event(), end.event()));
  }
  return res;
}

__CUDA_HOST_DEVICE__ unsigned int IvyCudaEvent::get_event_flags(EventFlags const& flags){
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
    __PRINT_ERROR__("IvyCudaEvent::get_event_flags: Unknown flag option...");
    assert(0);
  }
  return cudaEventDisableTiming;
}
__CUDA_HOST_DEVICE__ unsigned int IvyCudaEvent::get_record_flags(RecordFlags const& flags){
  switch (flags){
  case RecordFlags::Default:
    return cudaEventRecordDefault;
  case RecordFlags::External:
    return cudaEventRecordExternal;
  default:
    __PRINT_ERROR__("IvyCudaEvent::get_record_flags: Unknown flag option...");
    assert(0);
  }
  return cudaEventRecordDefault;
}
__CUDA_HOST_DEVICE__ unsigned int IvyCudaEvent::get_wait_flags(WaitFlags const& flags){
  switch (flags){
  case WaitFlags::Default:
    return cudaEventWaitDefault;
  case WaitFlags::External:
    return cudaEventWaitExternal;
  default:
    __PRINT_ERROR__("IvyCudaEvent::get_wait_flags: Unknown flag option...");
    assert(0);
  }
  return cudaEventWaitDefault;
}

__CUDA_HOST_DEVICE__ void IvyCudaEvent::swap(IvyCudaEvent& other){
  std_util::swap(flags_, other.flags_);
  std_util::swap(event_, other.event_);
}

#endif


#endif
