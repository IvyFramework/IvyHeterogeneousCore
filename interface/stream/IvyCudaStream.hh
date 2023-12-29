#ifndef IVYCUDASTREAM_HH
#define IVYCUDASTREAM_HH


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "IvyException.h"
#include "stream/IvyCudaEvent.hh"


/*
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


class IvyCudaStream{
public:
  typedef cudaHostFn_t fcn_callback_t;

  enum class StreamFlags{
    Default,
    NonBlocking
  };

  static __CUDA_HOST_DEVICE__ unsigned int get_stream_flags(StreamFlags const& flags);

protected:
  bool is_owned_;
  unsigned int flags_;
  int priority_;
  cudaStream_t stream_;

public:
  __CUDA_HOST__ IvyCudaStream(StreamFlags flags = StreamFlags::Default, int priority = 0);
  __CUDA_HOST__ IvyCudaStream(cudaStream_t st, bool do_own);
  __CUDA_HOST__ IvyCudaStream(IvyCudaStream const&) = delete;
  __CUDA_HOST__ IvyCudaStream(IvyCudaStream const&&) = delete;
  __CUDA_HOST__ ~IvyCudaStream();

  __CUDA_HOST__ unsigned int const& flags() const;
  __CUDA_HOST__ int const& priority() const;
  __CUDA_HOST__ cudaStream_t const& stream() const;
  __CUDA_HOST__ operator cudaStream_t const& () const;

  __CUDA_HOST__ unsigned int& flags();
  __CUDA_HOST__ int& priority();
  __CUDA_HOST__ cudaStream_t& stream();
  __CUDA_HOST__ operator cudaStream_t& ();

  // wait_flags could be cudaEventWaitDefault or cudaEventWaitExternal.
  __CUDA_HOST__ void wait(IvyCudaEvent& event, IvyCudaEvent::WaitFlags wait_flags = IvyCudaEvent::WaitFlags::Default);

  __CUDA_HOST__ void synchronize();

  __CUDA_HOST__ void add_callback(fcn_callback_t fcn, void* user_data);

};


#endif

#endif
