#ifndef IVYBASESTREAMEVENT_H
#define IVYBASESTREAMEVENT_H

/**
 * @file IvyBaseStreamEvent.h
 * @brief Base abstraction for backend stream-event wrappers.
 */


#include "std_ivy/IvyUtility.h"


/**
  IvyBaseStreamEvent:
  This class is a base class for stream event records.
*/


namespace IvyStreamUtils{
  /** @brief Create a backend-native stream event object. */
  template<typename RawEvent_t> __HOST_DEVICE__ void createStreamEvent(RawEvent_t& ev);
  /** @brief Destroy a backend-native stream event object. */
  template<typename RawEvent_t> __HOST_DEVICE__ void destroyStreamEvent(RawEvent_t& ev);

  /** @brief Map a raw stream type to its raw event type. */
  template<typename RawStream_t> struct StreamEvent{};
  /** @brief Convenience alias for backend event type corresponding to a raw stream type. */
  template<typename RawStream_t> using StreamEvent_t = typename StreamEvent<RawStream_t>::type;
}

/**
 * @brief Polymorphic base class for stream-event wrappers.
 * @tparam S Backend raw stream type.
 */
template<typename S> class IvyBaseStreamEvent{
public:
  /** @brief Backend raw stream type. */
  typedef S RawStream_t;
  /** @brief Backend raw event type associated with RawStream_t. */
  typedef IvyStreamUtils::StreamEvent_t<RawStream_t> RawEvent_t;

protected:
  bool is_owned_;
  unsigned int flags_;
  RawEvent_t event_;

public:
  /** @brief Default constructor. */
  IvyBaseStreamEvent() = default;
  /** @brief Construct from explicit ownership, flags, and raw event handle. */
  __HOST__ IvyBaseStreamEvent(bool const& is_owned, unsigned int const& flags, RawEvent_t const& ev) :
    is_owned_(is_owned), flags_(flags), event_(ev)
  {}
  /** @brief Copy constructor is disabled to avoid ambiguous ownership semantics. */
  __HOST_DEVICE__ IvyBaseStreamEvent(IvyBaseStreamEvent const&) = delete;
  /** @brief Move-copy constructor syntax variant is disabled. */
  __HOST_DEVICE__ IvyBaseStreamEvent(IvyBaseStreamEvent const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyBaseStreamEvent(){ if (this->is_owned_) IvyStreamUtils::destroyStreamEvent(event_); }

  /** @brief Access immutable event flags. */
  __HOST_DEVICE__ unsigned int const& flags() const{ return this->flags_; }
  /** @brief Access immutable backend event object. */
  __HOST_DEVICE__ RawEvent_t const& event() const{ return this->event_; }
  /** @brief Implicit conversion to immutable backend event object. */
  __HOST_DEVICE__ operator RawEvent_t const& () const{ return this->event_; }

  /** @brief Access mutable event flags. */
  __HOST_DEVICE__ unsigned int& flags(){ return this->flags_; }
  /** @brief Access mutable backend event object. */
  __HOST_DEVICE__ RawEvent_t& event(){ return this->event_; }
  /** @brief Implicit conversion to mutable backend event object. */
  __HOST_DEVICE__ operator RawEvent_t& (){ return this->event_; }

  /** @brief Record this event on a backend stream. */
  virtual __HOST__ void record(RawStream_t& stream, unsigned int rcd_flags) = 0;
  /** @brief Synchronize host execution with this event. */
  virtual __HOST__ void synchronize() = 0;

  __HOST_DEVICE__ void swap(IvyBaseStreamEvent& other){
    std_util::swap(this->is_owned_, other.is_owned_);
    std_util::swap(this->flags_, other.flags_);
    std_util::swap(this->event_, other.event_);
  }
};


#endif
