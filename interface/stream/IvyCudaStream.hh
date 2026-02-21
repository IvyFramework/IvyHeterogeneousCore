#ifndef IVYCUDASTREAM_HH
#define IVYCUDASTREAM_HH

/**
 * @file IvyCudaStream.hh
 * @brief Public CUDA-stream wrapper interface.
 */


/**
IvyCudaStream:
This class is a wrapper around cudaStream_t. It is used to record streams and synchronize them.

Data members:
- The is_owned_ member is used to specify whether the stream is owned by the class or not.
  If it is owned, the stream will be destroyed when the class is destroyed.
  If it is not owned, the stream will persist outside of the class.
- The flags_ member is used to specify the behavior of the stream as in CUDA definitions.
  It is set to cudaStreamDefault by default, which is the same as the default CUDA choice.
  The available choices in CUDA are cudaStreamDefault and cudaStreamNonBlocking.
- The priority_ member is used to specify the priority of the stream. 0 is the default priority, which is the default parameter.
- The stream_ member is the actual cudaStream_t object wrapped.

Member functions:
- flags() returns the flags_ member.
- stream() returns the stream_ member.
- priority() returns the priority_ member.
- synchronize() synchronizes the calling thread with the stream.
- wait() waits for the stream to complete on the specified event.
  The wait_flags argument could be cudaEventWaitDefault (default usage) or cudaEventWaitExternal.
  The event argument instructs the stream to wait for the passed event.
- add_callback() adds a callback to the stream.
  The callback is a function pointer of type cudaStreamCallback_t.
  The user_data argument is a pointer to the data that will be passed to the callback.
  The cb_flags argument should be kept at 0 for now (per note on CUDA documentation).
  The callback function fcn will be called when the stream is complete.
*/


#ifdef __USE_CUDA__

#include "config/IvyCudaException.h"
#include "stream/IvyBaseStream.h"
#include "stream/IvyCudaEvent.hh"


namespace IvyStreamUtils{
  /** @brief CUDA specialization for creating a raw CUDA stream. */
  template<> __HOST_DEVICE__ void buildRawStream(cudaStream_t& st, unsigned int flags, unsigned int priority);
  /** @brief CUDA specialization for destroying a raw CUDA stream. */
  template<> __HOST_DEVICE__ void destroyRawStream(cudaStream_t& st);
}

class IvyCudaStream final : public IvyBaseStream<cudaStream_t>{
public:
  /** @brief Host callback signature used by CUDA stream host callbacks. */
  using fcn_callback_t = cudaHostFn_t;
  using Base_t = IvyBaseStream<cudaStream_t>;
  using RawStream_t = typename Base_t::RawStream_t;

  /** @brief Stream-creation flags mirrored from CUDA stream options. */
  enum class StreamFlags : unsigned char{
    Default,
    NonBlocking
  };

  /** @brief Construct an owned CUDA stream. */
  __HOST__ IvyCudaStream(StreamFlags flags = StreamFlags::Default, int priority = 0);
  /** @brief Construct a wrapper around an existing CUDA stream handle. */
  __HOST_DEVICE__ IvyCudaStream(cudaStream_t st, bool do_own);
  __HOST_DEVICE__ IvyCudaStream(IvyCudaStream const&) = delete;
  __HOST_DEVICE__ IvyCudaStream(IvyCudaStream const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyCudaStream(){}

  /** @brief Synchronize host execution on this stream. */
  __HOST__ void synchronize();

  // wait_flags could be cudaEventWaitDefault or cudaEventWaitExternal.
  /** @brief Wait for a raw backend event. */
  __HOST__ void wait(Base_t::RawEvent_t& event, unsigned int wait_flags);
  /** @brief Wait for a typed IvyCudaEvent. */
  __HOST__ void wait(IvyCudaEvent& event, IvyCudaEvent::WaitFlags wait_flags = IvyCudaEvent::WaitFlags::Default);

  /** @brief Enqueue a host callback to execute after prior stream work completes. */
  __HOST__ void add_callback(fcn_callback_t fcn, void* user_data);

  __HOST_DEVICE__ void swap(IvyCudaStream& other){ Base_t::swap(other); }

  /** @brief Convert typed stream flags to backend numeric flags. */
  static __HOST_DEVICE__ unsigned int get_stream_flags(StreamFlags const& flags);
  /** @brief Convert backend numeric flags to typed stream flags. */
  static __HOST_DEVICE__ StreamFlags get_stream_flags_reverse(unsigned int const& flags);
};

namespace IvyStreamUtils{
  /** @brief Destroy a CUDA stream wrapper instance. */
  template<> __HOST_DEVICE__ void destroy_stream(IvyCudaStream*& stream);
  /** @brief Create a CUDA stream wrapper from typed flags. */
  template<> __HOST__ void make_stream(IvyCudaStream*& stream, IvyCudaStream::StreamFlags flags, unsigned int priority);
  /** @brief Create a CUDA stream wrapper from raw flags. */
  template<> __HOST__ void make_stream(IvyCudaStream*& stream, unsigned int flags, unsigned int priority);
  /** @brief Wrap an existing CUDA raw stream handle in a CUDA stream wrapper. */
  template<> __HOST_DEVICE__ void make_stream(IvyCudaStream*& stream, IvyCudaStream::RawStream_t st, bool is_owned);
}

namespace std_util{
  /** @brief Swap two CUDA stream wrappers. */
  __HOST_DEVICE__ void swap(IvyCudaStream& a, IvyCudaStream& b);
}

#endif


#endif
