#ifndef IVYBASESTREAMEVENT_H
#define IVYBASESTREAMEVENT_H


#include "std_ivy/IvyUtility.h"


/*
  IvyBaseStreamEvent:
  This class is a base class for stream event records.
*/


namespace IvyStreamUtils{
  template<typename RawEvent_t> __HOST_DEVICE__ void createStreamEvent(RawEvent_t& ev);
  template<typename RawEvent_t> __HOST_DEVICE__ void destroyStreamEvent(RawEvent_t& ev);

  template<typename RawStream_t> struct StreamEvent{};
  template<typename RawStream_t> using StreamEvent_t = typename StreamEvent<RawStream_t>::type;
}

template<typename S> class IvyBaseStreamEvent{
public:
  typedef S RawStream_t;
  typedef IvyStreamUtils::StreamEvent_t<RawStream_t> RawEvent_t;

protected:
  bool is_owned_;
  unsigned int flags_;
  RawEvent_t event_;

public:
  IvyBaseStreamEvent() = default;
  __HOST__ IvyBaseStreamEvent(bool const& is_owned, unsigned int const& flags, RawEvent_t const& ev) :
    is_owned_(is_owned), flags_(flags), event_(ev)
  {}
  __HOST_DEVICE__ IvyBaseStreamEvent(IvyBaseStreamEvent const&) = delete;
  __HOST_DEVICE__ IvyBaseStreamEvent(IvyBaseStreamEvent const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBaseStreamEvent(){ if (this->is_owned_) IvyStreamUtils::destroyStreamEvent(event_); }

  __HOST_DEVICE__ unsigned int const& flags() const{ return this->flags_; }
  __HOST_DEVICE__ RawEvent_t const& event() const{ return this->event_; }
  __HOST_DEVICE__ operator RawEvent_t const& () const{ return this->event_; }

  __HOST_DEVICE__ unsigned int& flags(){ return this->flags_; }
  __HOST_DEVICE__ RawEvent_t& event(){ return this->event_; }
  __HOST_DEVICE__ operator RawEvent_t& (){ return this->event_; }

  virtual __HOST__ void record(RawStream_t& stream, unsigned int rcd_flags) = 0;
  virtual __HOST__ void synchronize() = 0;

  __HOST_DEVICE__ void swap(IvyBaseStreamEvent& other){
    std_util::swap(this->is_owned_, other.is_owned_);
    std_util::swap(this->flags_, other.flags_);
    std_util::swap(this->event_, other.event_);
  }
};


#endif
