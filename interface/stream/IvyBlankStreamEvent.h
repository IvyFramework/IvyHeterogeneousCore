#ifndef IVYBLANKSTREAMEVENT_H
#define IVYBLANKSTREAMEVENT_H


#include "stream/IvyBaseStreamEvent.h"


namespace IvyStreamUtils{
  struct BlankStream{};
  struct BlankStreamEvent{};

  template<> __HOST_DEVICE__ void createStreamEvent(BlankStreamEvent& ev){}
  template<> __HOST_DEVICE__ void destroyStreamEvent(BlankStreamEvent& ev){}
  template<> struct StreamEvent<BlankStream>{ typedef BlankStreamEvent type; };

#ifndef __USE_CUDA__
  constexpr BlankStream GlobalBlankStreamRaw;
#endif
}

class IvyBlankStream;

class IvyBlankStreamEvent final : public IvyBaseStreamEvent<IvyStreamUtils::BlankStream>{
public:
  typedef IvyBaseStreamEvent<IvyStreamUtils::BlankStream> Base_t;

  enum class EventFlags : unsigned char{
    Default
  };
  enum class RecordFlags : unsigned char{
    Default
  };
  enum class WaitFlags : unsigned char{
    Default
  };

  __HOST__ IvyBlankStreamEvent(EventFlags flags = EventFlags::Default) : Base_t(){}
  __HOST_DEVICE__ IvyBlankStreamEvent(IvyBlankStreamEvent const&) = delete;
  __HOST_DEVICE__ IvyBlankStreamEvent(IvyBlankStreamEvent const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBlankStreamEvent(){}

  __HOST__ void record(Base_t::RawStream_t& stream, unsigned int rcd_flags){}
  __HOST__ void record(Base_t::RawStream_t& stream, RecordFlags rcd_flags = RecordFlags::Default){}
  __HOST__ void record(IvyBlankStream& stream, RecordFlags rcd_flags = RecordFlags::Default){}

  __HOST__ void synchronize(){}

  __HOST__ float elapsed_time(IvyBlankStreamEvent const& start) const{ return -1; }
  static __HOST__ float elapsed_time(IvyBlankStreamEvent const& start, IvyBlankStreamEvent const& end){ return -1; }

  __HOST_DEVICE__ void swap(IvyBlankStreamEvent& other){ Base_t::swap(other); }
};
namespace std_util{
  __HOST_DEVICE__ void swap(IvyBlankStreamEvent& a, IvyBlankStreamEvent& b){ a.swap(b); }
}


#endif
