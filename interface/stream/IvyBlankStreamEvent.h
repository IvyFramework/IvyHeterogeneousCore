#ifndef IVYBLANKSTREAMEVENT_H
#define IVYBLANKSTREAMEVENT_H


#include "stream/IvyBaseStreamEvent.h"


namespace IvyStreamUtils{
  struct BlankStream{};
  struct BlankStreamEvent{};

  template<> __CUDA_HOST_DEVICE__ void createStreamEvent(BlankStreamEvent& ev){}
  template<> __CUDA_HOST_DEVICE__ void destroyStreamEvent(BlankStreamEvent& ev){}
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

  __CUDA_HOST__ IvyBlankStreamEvent(EventFlags flags = EventFlags::Default) : Base_t(){}
  __CUDA_HOST_DEVICE__ IvyBlankStreamEvent(IvyBlankStreamEvent const&) = delete;
  __CUDA_HOST_DEVICE__ IvyBlankStreamEvent(IvyBlankStreamEvent const&&) = delete;
  virtual __CUDA_HOST_DEVICE__ ~IvyBlankStreamEvent(){}

  __CUDA_HOST__ void record(Base_t::RawStream_t& stream, unsigned int rcd_flags){}
  __CUDA_HOST__ void record(Base_t::RawStream_t& stream, RecordFlags rcd_flags = RecordFlags::Default){}
  __CUDA_HOST__ void record(IvyBlankStream& stream, RecordFlags rcd_flags = RecordFlags::Default){}

  __CUDA_HOST__ void synchronize(){}

  __CUDA_HOST_DEVICE__ void swap(IvyBlankStreamEvent& other){ Base_t::swap(other); }
};
namespace std_util{
  __CUDA_HOST_DEVICE__ void swap(IvyBlankStreamEvent& a, IvyBlankStreamEvent& b){ a.swap(b); }
}


#endif
