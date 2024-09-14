#ifndef IVYBLANKSTREAM_H
#define IVYBLANKSTREAM_H


#include "config/IvyCudaException.h"
#include "stream/IvyBaseStream.h"
#include "stream/IvyBlankStreamEvent.h"


namespace IvyStreamUtils{
  template<> __HOST_DEVICE__ void buildRawStream(BlankStream& st, unsigned int flags, unsigned int priority){}
  template<> __HOST_DEVICE__ void destroyRawStream(BlankStream& st){}
}

class IvyBlankStream final : public IvyBaseStream<IvyStreamUtils::BlankStream>{
public:
  using Base_t = IvyBaseStream<IvyStreamUtils::BlankStream>;
  using RawStream_t = typename Base_t::RawStream_t;

  enum class StreamFlags : unsigned char{
    Default
  };

  __HOST__ IvyBlankStream(StreamFlags flags = StreamFlags::Default, int priority = 0) : Base_t(){}
  __HOST_DEVICE__ IvyBlankStream(IvyStreamUtils::BlankStream st, bool do_own) : Base_t(){}
  __HOST_DEVICE__ IvyBlankStream(IvyBlankStream const&) = delete;
  __HOST_DEVICE__ IvyBlankStream(IvyBlankStream const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBlankStream(){}

  __HOST__ void synchronize(){}

  __HOST__ void wait(Base_t::RawEvent_t& event, unsigned int wait_flags){}
  __HOST__ void wait(IvyBlankStreamEvent& event, IvyBlankStreamEvent::WaitFlags wait_flags = IvyBlankStreamEvent::WaitFlags::Default){}

  __HOST_DEVICE__ void swap(IvyBlankStream& other){ Base_t::swap(other); }

  static __HOST_DEVICE__ unsigned int get_stream_flags(StreamFlags const& flags){
    switch (flags){
    case StreamFlags::Default:
      return 0;
    default:
      __PRINT_ERROR__("IvyBlankStream::get_stream_flags: Unknown flag option...\n");
      assert(0);
    }
    return 0;
  }
  static __HOST_DEVICE__ StreamFlags get_stream_flags_reverse(unsigned int const& flags){
    switch (flags){
    case 0:
      return StreamFlags::Default;
    default:
      __PRINT_ERROR__("IvyBlankStream::get_stream_flags_reverse: Unknown flag option...\n");
      assert(0);
    }
    return StreamFlags::Default;
  }
};

namespace IvyStreamUtils{
  template<> __HOST_DEVICE__ void destroy_stream(IvyBlankStream*& stream){
    delete stream;
    stream = nullptr;
  }
  template<> __HOST__ void make_stream(IvyBlankStream*& stream, IvyBlankStream::StreamFlags flags, unsigned int priority){
    destroy_stream(stream);
    stream = new IvyBlankStream(flags, priority);
  }
  template<> __HOST__ void make_stream(IvyBlankStream*& stream, unsigned int flags, unsigned int priority){ make_stream(stream, IvyBlankStream::get_stream_flags_reverse(flags), priority); }
  template<> __HOST_DEVICE__ void make_stream(IvyBlankStream*& stream, IvyBlankStream::RawStream_t st, bool is_owned){
    destroy_stream(stream);
    stream = new IvyBlankStream(st, is_owned);
  }
}
namespace std_util{
  __HOST_DEVICE__ void swap(IvyBlankStream& a, IvyBlankStream& b){ a.swap(b); }
}


#endif
