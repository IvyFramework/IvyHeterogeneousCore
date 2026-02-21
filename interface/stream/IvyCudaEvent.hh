#ifndef IVYCUDAEVENT_HH
#define IVYCUDAEVENT_HH

/**
 * @file IvyCudaEvent.hh
 * @brief Public CUDA-event wrapper interface.
 */


#ifdef __USE_CUDA__

#include "config/IvyCudaException.h"
#include "stream/IvyBaseStreamEvent.h"


/**
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
  /** @brief CUDA specialization for creating a raw CUDA event. */
  template<> __HOST_DEVICE__ void createStreamEvent(cudaEvent_t& ev);
  /** @brief CUDA specialization for destroying a raw CUDA event. */
  template<> __HOST_DEVICE__ void destroyStreamEvent(cudaEvent_t& ev);

  /** @brief Event-type mapping for CUDA raw streams. */
  template<> struct StreamEvent<cudaStream_t>{ typedef cudaEvent_t type; };
}

class IvyCudaEvent final : public IvyBaseStreamEvent<cudaStream_t>{
public:
  typedef IvyBaseStreamEvent<cudaStream_t> Base_t;

  /** @brief Event-creation flags mirrored from CUDA event options. */
  enum class EventFlags : unsigned char{
    Default,
    BlockingSync,
    DisableTiming,
    Interprocess
  };
  /** @brief Event-record flags mirrored from CUDA record options. */
  enum class RecordFlags : unsigned char{
    Default,
    External
  };
  /** @brief Stream-wait flags mirrored from CUDA wait options. */
  enum class WaitFlags : unsigned char{
    Default,
    External
  };

  /** @brief Construct an owned CUDA event with selected flags. */
  __HOST__ IvyCudaEvent(EventFlags flags = EventFlags::DisableTiming);
  __HOST_DEVICE__ IvyCudaEvent(IvyCudaEvent const&) = delete;
  __HOST_DEVICE__ IvyCudaEvent(IvyCudaEvent const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyCudaEvent(){}

  // rcd_flags could be cudaEventRecordDefault or cudaEventRecordExternal.
  /** @brief Record this event on a raw CUDA stream with raw record flags. */
  __HOST__ void record(cudaStream_t& stream, unsigned int rcd_flags);
  /** @brief Record this event on an IvyCudaStream. */
  __HOST__ void record(IvyCudaStream& stream, RecordFlags rcd_flags = RecordFlags::Default);
  /** @brief Record this event on a raw CUDA stream with typed record flags. */
  __HOST__ void record(cudaStream_t& stream, RecordFlags rcd_flags = RecordFlags::Default);

  /** @brief Synchronize host execution on this event. */
  __HOST__ void synchronize();

  /** @brief Elapsed time in milliseconds from @p start to this event. */
  __HOST__ float elapsed_time(IvyCudaEvent const& start) const;
  /** @brief Elapsed time in milliseconds between two events. */
  static __HOST__ float elapsed_time(IvyCudaEvent const& start, IvyCudaEvent const& end);

  __HOST_DEVICE__ void swap(IvyCudaEvent& other){ Base_t::swap(other); }

  /** @brief Convert typed event flags to backend numeric flags. */
  static __HOST_DEVICE__ unsigned int get_event_flags(EventFlags const& flags);
  /** @brief Convert typed record flags to backend numeric flags. */
  static __HOST_DEVICE__ unsigned int get_record_flags(RecordFlags const& flags);
  /** @brief Convert typed wait flags to backend numeric flags. */
  static __HOST_DEVICE__ unsigned int get_wait_flags(WaitFlags const& flags);
};
namespace std_util{
  /** @brief Swap two CUDA event wrappers. */
  __HOST_DEVICE__ void swap(IvyCudaEvent& a, IvyCudaEvent& b);
}

#endif

#endif
