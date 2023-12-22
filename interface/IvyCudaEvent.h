#ifndef IVYCUDAEVENT_H
#define IVYCUDAEVENT_H


#include "IvyCudaEvent.hh"

#ifdef __USE_CUDA__

#include <cuda_runtime.h>
#include "IvyException.h"


__CUDA_HOST__ IvyCudaEvent::IvyCudaEvent(unsigned int flags) : flags_(flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventCreate(&event_, flags_));
}
__CUDA_HOST__ IvyCudaEvent::~IvyCudaEvent(){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventDestroy(event_));
}

__CUDA_HOST__ unsigned int const& IvyCudaEvent::flags() const{ return flags_; }
__CUDA_HOST__ cudaEvent_t const& IvyCudaEvent::event() const{ return event_; }
__CUDA_HOST__ IvyCudaEvent::operator cudaEvent_t const& () const{ return event_; }

__CUDA_HOST__ unsigned int& IvyCudaEvent::flags(){ return flags_; }
__CUDA_HOST__ cudaEvent_t& IvyCudaEvent::event(){ return event_; }
__CUDA_HOST__ IvyCudaEvent::operator cudaEvent_t& (){ return event_; }

__CUDA_HOST__ void IvyCudaEvent::record(IvyCudaStream& stream, unsigned int rcd_flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventRecordWithFlags(event_, stream.stream(), rcd_flags));
}
__CUDA_HOST__ void IvyCudaEvent::synchronize(){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaEventSynchronize(event_));
}
__CUDA_HOST__ void IvyCudaEvent::wait(IvyCudaStream& stream, unsigned int wait_flags){
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


#endif

#endif
