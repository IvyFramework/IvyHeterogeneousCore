#ifndef IVYCUDAEVENT_HH
#define IVYCUDAEVENT_HH


#ifdef __USE_CUDA__

#include <cuda_runtime.h>
#include "IvyException.h"


/*
  IvyCudaEvent:
  This class is a wrapper around cudaEvent_t. It is used to record events and synchronize streams.
  It is also used to measure elapsed time between two events.

  Data members:
  - The flags_ member is used to specify the behavior of the event as in CUDA definitions.
    It is set to cudaEventDisableTiming by default, which is the fastest, but it disables timing information.
    This is different from the default CUDA choice, which is cudaEventDefault.
    The available choices in CUDA are cudaEventDefault, cudaEventBlockingSync, cudaEventDisableTiming, and cudaEventInterprocess.
  - The event_ member is the actual cudaEvent_t object wrapped.

  Member functions:
  - flags() returns the flags_ member.
  - event() returns the event_ member.
  - record() records the event on the specified stream.
    The rcd_flags argument could be cudaEventRecordDefault or cudaEventRecordExternal.
    The stream argument defines the stream to record this event and is defaulted to cudaStreamLegacy.
  - synchronize() synchronizes the calling thread with the event.
  - wait() waits for the event to complete on the specified stream.
    The wait_flags argument could be cudaEventWaitDefault or cudaEventWaitExternal.
    The stream argument instructs the passed stream to wait for this event.
  - elapsed_time() returns the elapsed time between the calling event and the passed event.
    The start argument marks the beginning of the time interval and needs to be another event.
*/


class IvyCudaStream;

class IvyCudaEvent{
protected:
  unsigned int flags_;
  cudaEvent_t event_;

public:
  __CUDA_HOST__ IvyCudaEvent(unsigned int flags = cudaEventDisableTiming);
  __CUDA_HOST__ IvyCudaEvent(IvyCudaEvent const&) = delete;
  __CUDA_HOST__ IvyCudaEvent(IvyCudaEvent const&&) = delete;
  __CUDA_HOST__ ~IvyCudaEvent();

  __CUDA_HOST__ unsigned int const& flags() const;
  __CUDA_HOST__ cudaEvent_t const& event() const;
  __CUDA_HOST__ operator cudaEvent_t const& () const;

  __CUDA_HOST__ unsigned int& flags();
  __CUDA_HOST__ cudaEvent_t& event();
  __CUDA_HOST__ operator cudaEvent_t& ();

  // rcd_flags could be cudaEventRecordDefault or cudaEventRecordExternal.
  __CUDA_HOST__ void record(IvyCudaStream& stream, unsigned int rcd_flags = cudaEventRecordDefault);

  // wait_flags could be cudaEventWaitDefault or cudaEventWaitExternal.
  __CUDA_HOST__ void wait(IvyCudaStream& stream, unsigned int wait_flags = cudaEventWaitDefault);

  __CUDA_HOST__ void synchronize();

  __CUDA_HOST__ float elapsed_time(IvyCudaEvent const& start) const;
  static __CUDA_HOST__ float elapsed_time(IvyCudaEvent const& start, IvyCudaEvent const& end);

};


#endif

#endif
