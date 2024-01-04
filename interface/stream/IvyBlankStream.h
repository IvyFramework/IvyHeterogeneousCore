#ifndef IVYBLANKSTREAM_H
#define IVYBLANKSTREAM_H


#include "stream/IvyBaseStream.h"
#include "stream/IvyBlankStreamEvent.h"


namespace IvyStreamUtils{
  template<> __CUDA_HOST_DEVICE__ void createStream(BlankStream& st, unsigned int flags, unsigned int priority){}
  template<> __CUDA_HOST_DEVICE__ void destroyStream(BlankStream& st){}
}

class IvyBlankStream final : public IvyBaseStream<IvyStreamUtils::BlankStream>{
public:
  typedef IvyBaseStream<IvyStreamUtils::BlankStream> Base_t;

  enum class StreamFlags : unsigned char{
    Default
  };

  __CUDA_HOST__ IvyBlankStream(StreamFlags flags = StreamFlags::Default, int priority = 0) : Base_t(){}
  __CUDA_HOST_DEVICE__ IvyBlankStream(IvyStreamUtils::BlankStream st, bool do_own) : Base_t(){}
  __CUDA_HOST_DEVICE__ IvyBlankStream(IvyBlankStream const&) = delete;
  __CUDA_HOST_DEVICE__ IvyBlankStream(IvyBlankStream const&&) = delete;
  virtual __CUDA_HOST_DEVICE__ ~IvyBlankStream(){}

  __CUDA_HOST__ void synchronize(){}

  __CUDA_HOST__ void wait(Base_t::RawEvent_t& event, unsigned int wait_flags){}
  __CUDA_HOST__ void wait(IvyBlankStreamEvent& event, IvyBlankStreamEvent::WaitFlags wait_flags = IvyBlankStreamEvent::WaitFlags::Default){}

  __CUDA_HOST_DEVICE__ void swap(IvyBlankStream& other){ Base_t::swap(other); }
};


#endif
