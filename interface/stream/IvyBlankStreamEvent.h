#ifndef IVYBLANKSTREAMEVENT_H
#define IVYBLANKSTREAMEVENT_H


#include "stream/IvyBaseStreamEvent.h"
#ifndef __USE_CUDA__
#include "std_ivy/IvyChrono.h"
#endif


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
protected:
#ifndef __USE_CUDA__
  std_chrono::time_point<std_chrono::high_resolution_clock> rcd_time;
#endif

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

  __HOST__ void record(Base_t::RawStream_t& stream, unsigned int rcd_flags){
#ifndef __USE_CUDA__
    rcd_time = std_chrono::high_resolution_clock::now();
#endif
  }
  __HOST__ void record(Base_t::RawStream_t& stream, RecordFlags rcd_flags = RecordFlags::Default){
#ifndef __USE_CUDA__
    rcd_time = std_chrono::high_resolution_clock::now();
#endif
  }
  __HOST__ void record(IvyBlankStream& stream, RecordFlags rcd_flags = RecordFlags::Default){
#ifndef __USE_CUDA__
    rcd_time = std_chrono::high_resolution_clock::now();
#endif
  }

  __HOST__ void synchronize(){}

  __HOST__ float elapsed_time(IvyBlankStreamEvent const& start) const{
#ifndef __USE_CUDA__
    return std_chrono::duration_cast<std_chrono::duration<float, std_ratio::milli>>(rcd_time - start.rcd_time).count();
#else
    return -1;
#endif
  }
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
  __HOST_DEVICE__ void swap(IvyBlankStreamEvent& a, IvyBlankStreamEvent& b){ a.swap(b); }
}


#endif
