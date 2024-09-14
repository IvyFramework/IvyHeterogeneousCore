#ifndef IVYCUDAEVENT_HH
#define IVYCUDAEVENT_HH


#ifdef __USE_CUDA__

#include "config/IvyCudaException.h"
#include "stream/IvyBaseStreamEvent.h"


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
  - elapsed_time() returns the elapsed time between the calling event and the passed event.
    The start argument marks the beginning of the time interval and needs to be another event.
*/


class IvyCudaStream;

namespace IvyStreamUtils{
  template<> __HOST_DEVICE__ void createStreamEvent(cudaEvent_t& ev);
  template<> __HOST_DEVICE__ void destroyStreamEvent(cudaEvent_t& ev);

  template<> struct StreamEvent<cudaStream_t>{ typedef cudaEvent_t type; };
}

class IvyCudaEvent final : public IvyBaseStreamEvent<cudaStream_t>{
public:
  typedef IvyBaseStreamEvent<cudaStream_t> Base_t;

  enum class EventFlags : unsigned char{
    Default,
    BlockingSync,
    DisableTiming,
    Interprocess
  };
  enum class RecordFlags : unsigned char{
    Default,
    External
  };
  enum class WaitFlags : unsigned char{
    Default,
    External
  };

  __HOST__ IvyCudaEvent(EventFlags flags = EventFlags::DisableTiming);
  __HOST_DEVICE__ IvyCudaEvent(IvyCudaEvent const&) = delete;
  __HOST_DEVICE__ IvyCudaEvent(IvyCudaEvent const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyCudaEvent(){}

  // rcd_flags could be cudaEventRecordDefault or cudaEventRecordExternal.
  __HOST__ void record(cudaStream_t& stream, unsigned int rcd_flags);
  __HOST__ void record(IvyCudaStream& stream, RecordFlags rcd_flags = RecordFlags::Default);
  __HOST__ void record(cudaStream_t& stream, RecordFlags rcd_flags = RecordFlags::Default);

  __HOST__ void synchronize();

  __HOST__ float elapsed_time(IvyCudaEvent const& start) const;
  static __HOST__ float elapsed_time(IvyCudaEvent const& start, IvyCudaEvent const& end);

  __HOST_DEVICE__ void swap(IvyCudaEvent& other){ Base_t::swap(other); }

  static __HOST_DEVICE__ unsigned int get_event_flags(EventFlags const& flags);
  static __HOST_DEVICE__ unsigned int get_record_flags(RecordFlags const& flags);
  static __HOST_DEVICE__ unsigned int get_wait_flags(WaitFlags const& flags);
};
namespace std_util{
  __HOST_DEVICE__ void swap(IvyCudaEvent& a, IvyCudaEvent& b);
}

#endif

#endif
