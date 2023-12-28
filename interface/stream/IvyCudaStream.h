#ifndef IVYCUDASTREAM_H
#define IVYCUDASTREAM_H


#include "stream/IvyCudaStream.hh"
#include "stream/IvyCudaEvent.h"


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "IvyException.h"


__CUDA_HOST__ IvyCudaStream::IvyCudaStream(StreamFlags flags, int priority) :
  is_owned_(true),
  flags_(get_stream_flags(flags)),
  priority_(priority)
{
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamCreateWithPriority(&stream_, flags_, priority_));
}
__CUDA_HOST__ IvyCudaStream::IvyCudaStream(cudaStream_t st, bool do_own) : is_owned_(do_own), stream_(st){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamGetFlags(stream_, &flags_));
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamGetPriority(stream_, &priority_));
}
__CUDA_HOST__ IvyCudaStream::~IvyCudaStream(){
  if (is_owned_){
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamDestroy(stream_));
  }
}

__CUDA_HOST__ unsigned int const& IvyCudaStream::flags() const{ return flags_; }
__CUDA_HOST__ int const& IvyCudaStream::priority() const{ return priority_; }
__CUDA_HOST__ cudaStream_t const& IvyCudaStream::stream() const{ return stream_; }
__CUDA_HOST__ IvyCudaStream::operator cudaStream_t const& () const{ return stream_; }

__CUDA_HOST__ unsigned int& IvyCudaStream::flags(){ return flags_; }
__CUDA_HOST__ int& IvyCudaStream::priority(){ return priority_; }
__CUDA_HOST__ cudaStream_t& IvyCudaStream::stream(){ return stream_; }
__CUDA_HOST__ IvyCudaStream::operator cudaStream_t& (){ return stream_; }

__CUDA_HOST__ void IvyCudaStream::synchronize(){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamSynchronize(stream_));
}
__CUDA_HOST__ void IvyCudaStream::wait(IvyCudaEvent& event, IvyCudaEvent::WaitFlags wait_flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamWaitEvent(stream_, event.event(), IvyCudaEvent::get_wait_flags(wait_flags)));
}

__CUDA_HOST__ void IvyCudaStream::add_callback(fcn_callback_t fcn, void* user_data, unsigned int cb_flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamAddCallback(stream_, fcn, user_data, cb_flags));
}

__CUDA_HOST_DEVICE__ unsigned int IvyCudaStream::get_stream_flags(StreamFlags const& flags){
  switch (flags){
  case StreamFlags::Default:
    return cudaStreamDefault;
  case StreamFlags::NonBlocking:
    return cudaStreamNonBlocking;
  default:
    printf("IvyCudaStream::get_stream_flags: Unknown flag option...");
    assert(0);
  }
  return cudaStreamDefault;
}


#endif

#endif
