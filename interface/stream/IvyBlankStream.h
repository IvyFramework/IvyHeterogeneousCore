#ifndef IVYBLANKSTREAM_H
#define IVYBLANKSTREAM_H

/**
 * @file IvyBlankStream.h
 * @brief CPU/no-op stream implementation used in non-CUDA builds.
 */


#include "config/IvyCudaException.h"
#include "stream/IvyBaseStream.h"
#include "stream/IvyBlankStreamEvent.h"


namespace IvyStreamUtils{
  /** @brief Blank-stream specialization: no-op raw stream creation. */
  template<> __HOST_DEVICE__ void buildRawStream(BlankStream& st, unsigned int flags, unsigned int priority){}
  /** @brief Blank-stream specialization: no-op raw stream destruction. */
  template<> __HOST_DEVICE__ void destroyRawStream(BlankStream& st){}
}

class IvyBlankStream final : public IvyBaseStream<IvyStreamUtils::BlankStream>{
public:
  using Base_t = IvyBaseStream<IvyStreamUtils::BlankStream>;
  using RawStream_t = typename Base_t::RawStream_t;

  /** @brief Stream creation flags for blank streams. */
  enum class StreamFlags : unsigned char{
    Default,
    NonBlocking
  };

  /** @brief Construct a blank stream wrapper. */
  __HOST__ IvyBlankStream(StreamFlags flags = StreamFlags::Default, int priority = 0) : Base_t(){}
  /** @brief Construct from an existing blank raw stream marker. */
  __HOST_DEVICE__ IvyBlankStream(IvyStreamUtils::BlankStream st, bool do_own) : Base_t(){}
  /** @brief Copy constructor is disabled to avoid ownership ambiguity. */
  __HOST_DEVICE__ IvyBlankStream(IvyBlankStream const&) = delete;
  /** @brief Move-copy constructor syntax variant is disabled. */
  __HOST_DEVICE__ IvyBlankStream(IvyBlankStream const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBlankStream(){}

  /** @brief No-op synchronization for blank streams. */
  __HOST__ void synchronize(){}

  /** @brief No-op wait for raw blank events. */
  __HOST__ void wait(Base_t::RawEvent_t& event, unsigned int wait_flags){}
  /** @brief No-op wait for typed blank events. */
  __HOST__ void wait(IvyBlankStreamEvent& event, IvyBlankStreamEvent::WaitFlags wait_flags = IvyBlankStreamEvent::WaitFlags::Default){}

  __HOST_DEVICE__ void swap(IvyBlankStream& other){ Base_t::swap(other); }

  /** @brief Map typed stream flags to backend unsigned flags. */
  static __HOST_DEVICE__ unsigned int get_stream_flags(StreamFlags const& flags){
    switch (flags){
    case StreamFlags::Default:
    case StreamFlags::NonBlocking:
      return 0;
    default:
      __PRINT_ERROR__("IvyBlankStream::get_stream_flags: Unknown flag option...\n");
      assert(0);
    }
    return 0;
  }
  /** @brief Map backend unsigned flags to typed stream flags. */
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
  /** @brief Destroy a blank stream wrapper instance. */
  template<> __HOST_DEVICE__ void destroy_stream(IvyBlankStream*& stream){
    delete stream;
    stream = nullptr;
  }
  /** @brief Create a blank stream wrapper from typed flags. */
  template<> __HOST__ void make_stream(IvyBlankStream*& stream, IvyBlankStream::StreamFlags flags, unsigned int priority){
    destroy_stream(stream);
    stream = new IvyBlankStream(flags, priority);
  }
  /** @brief Create a blank stream wrapper from raw flags. */
  template<> __HOST__ void make_stream(IvyBlankStream*& stream, unsigned int flags, unsigned int priority){ make_stream(stream, IvyBlankStream::get_stream_flags_reverse(flags), priority); }
  /** @brief Wrap a raw blank stream marker in a blank stream wrapper. */
  template<> __HOST_DEVICE__ void make_stream(IvyBlankStream*& stream, IvyBlankStream::RawStream_t st, bool is_owned){
    destroy_stream(stream);
    stream = new IvyBlankStream(st, is_owned);
  }
}
namespace std_util{
  /** @brief Swap two blank stream wrappers. */
  __HOST_DEVICE__ void swap(IvyBlankStream& a, IvyBlankStream& b){ a.swap(b); }
}


#endif
