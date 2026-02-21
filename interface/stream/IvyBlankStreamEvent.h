#ifndef IVYBLANKSTREAMEVENT_H
#define IVYBLANKSTREAMEVENT_H

/**
 * @file IvyBlankStreamEvent.h
 * @brief CPU/no-op stream-event implementation used in non-CUDA builds.
 */


#include "stream/IvyBaseStreamEvent.h"
#ifndef __USE_CUDA__
#include "std_ivy/IvyChrono.h"
#endif


namespace IvyStreamUtils{
  /** @brief Marker raw stream type for blank stream backend. */
  struct BlankStream{};
  /** @brief Marker raw event type for blank stream backend. */
  struct BlankStreamEvent{};

  /** @brief Blank-event specialization: no-op event creation. */
  template<> __HOST_DEVICE__ void createStreamEvent(BlankStreamEvent& ev){}
  /** @brief Blank-event specialization: no-op event destruction. */
  template<> __HOST_DEVICE__ void destroyStreamEvent(BlankStreamEvent& ev){}
  /** @brief Event-type mapping for blank raw streams. */
  template<> struct StreamEvent<BlankStream>{ typedef BlankStreamEvent type; };

#ifndef __USE_CUDA__
  /** @brief Global raw blank stream marker used in non-CUDA builds. */
  constexpr BlankStream GlobalBlankStreamRaw;
#endif
}

class IvyBlankStream;

/** @brief Blank-stream event implementation for non-CUDA execution. */
class IvyBlankStreamEvent final : public IvyBaseStreamEvent<IvyStreamUtils::BlankStream>{
protected:
#ifndef __USE_CUDA__
  std_chrono::time_point<std_chrono::high_resolution_clock> rcd_time;
#endif

public:
  typedef IvyBaseStreamEvent<IvyStreamUtils::BlankStream> Base_t;

  /** @brief Event creation policy flags for blank events. */
  enum class EventFlags : unsigned char{
    Default
  };
  /** @brief Event record policy flags for blank events. */
  enum class RecordFlags : unsigned char{
    Default
  };
  /** @brief Event wait policy flags for blank events. */
  enum class WaitFlags : unsigned char{
    Default
  };

  /** @brief Construct a blank stream event wrapper. */
  __HOST__ IvyBlankStreamEvent(EventFlags flags = EventFlags::Default) : Base_t(){}
  /** @brief Copy constructor is disabled to avoid ownership ambiguity. */
  __HOST_DEVICE__ IvyBlankStreamEvent(IvyBlankStreamEvent const&) = delete;
  /** @brief Move-copy constructor syntax variant is disabled. */
  __HOST_DEVICE__ IvyBlankStreamEvent(IvyBlankStreamEvent const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBlankStreamEvent(){}

  /** @brief Record a blank event timestamp. */
  __HOST__ void record(Base_t::RawStream_t& stream, unsigned int rcd_flags){
#ifndef __USE_CUDA__
    rcd_time = std_chrono::high_resolution_clock::now();
#endif
  }
  /** @brief Record using typed blank-stream flags. */
  __HOST__ void record(Base_t::RawStream_t& stream, RecordFlags rcd_flags = RecordFlags::Default){
#ifndef __USE_CUDA__
    rcd_time = std_chrono::high_resolution_clock::now();
#endif
  }
  /** @brief Record against a typed IvyBlankStream object. */
  __HOST__ void record(IvyBlankStream& stream, RecordFlags rcd_flags = RecordFlags::Default){
#ifndef __USE_CUDA__
    rcd_time = std_chrono::high_resolution_clock::now();
#endif
  }

  /** @brief No-op synchronization for blank events. */
  __HOST__ void synchronize(){}

  /** @brief Return elapsed milliseconds from @p start to this event. */
  __HOST__ float elapsed_time(IvyBlankStreamEvent const& start) const{
#ifndef __USE_CUDA__
    return std_chrono::duration_cast<std_chrono::duration<float, std_ratio::milli>>(rcd_time - start.rcd_time).count();
#else
    return -1;
#endif
  }
  /** @brief Return elapsed milliseconds between two blank events. */
  static __HOST__ float elapsed_time(IvyBlankStreamEvent const& start, IvyBlankStreamEvent const& end){
#ifndef __USE_CUDA__
    return std_chrono::duration_cast<std_chrono::duration<float, std_ratio::milli>>(end.rcd_time - start.rcd_time).count();
#else
    return -1;
#endif
  }

  __HOST_DEVICE__ void swap(IvyBlankStreamEvent& other){
    Base_t::swap(other);
#ifndef __USE_CUDA__
    std_util::swap(rcd_time, other.rcd_time);
#endif
  }
};
namespace std_util{
  /** @brief Swap two blank stream events. */
  __HOST_DEVICE__ void swap(IvyBlankStreamEvent& a, IvyBlankStreamEvent& b){ a.swap(b); }
}


#endif
