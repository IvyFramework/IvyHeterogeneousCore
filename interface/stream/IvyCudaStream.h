#ifndef IVYCUDASTREAM_H
#define IVYCUDASTREAM_H


#include "stream/IvyCudaStream.hh"
#include "stream/IvyCudaEvent.h"

#ifdef __USE_CUDA__

namespace IvyStreamUtils{
  template<> __CUDA_HOST_DEVICE__ void createStream(cudaStream_t& st, unsigned int flags, unsigned int priority){
#ifndef __CUDA_DEVICE_CODE__
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamCreateWithPriority(&st, flags, priority));
#endif
  }
  template<> __CUDA_HOST_DEVICE__ void destroyStream(cudaStream_t& st){
#ifndef __CUDA_DEVICE_CODE__
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamDestroy(st));
#endif
  }
}

__CUDA_HOST__ IvyCudaStream::IvyCudaStream(StreamFlags flags, int priority) : IvyBaseStream<cudaStream_t>()
{
  is_owned_ = true;
  flags_ = get_stream_flags(flags);
  priority_ = priority;

  IvyStreamUtils::createStream(stream_, flags_, priority_);
}
__CUDA_HOST_DEVICE__ IvyCudaStream::IvyCudaStream(
  cudaStream_t st,
#ifndef __CUDA_DEVICE_CODE__
  bool do_own
#else
  bool
#endif
) : IvyBaseStream<cudaStream_t>()
{
  stream_ = st;
#ifndef __CUDA_DEVICE_CODE__
  is_owned_ = do_own;
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamGetFlags(stream_, &flags_));
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamGetPriority(stream_, &priority_));
#else
  is_owned_ = false;
  flags_ = 0;
  priority_ = 0;
#endif
}

__CUDA_HOST__ void IvyCudaStream::synchronize(){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamSynchronize(stream_));
}
__CUDA_HOST__ void IvyCudaStream::wait(IvyCudaStream::Base_t::RawEvent_t& event, unsigned int wait_flags){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamWaitEvent(stream_, event, wait_flags));
}
__CUDA_HOST__ void IvyCudaStream::wait(IvyCudaEvent& event, IvyCudaEvent::WaitFlags wait_flags){
  this->wait(event, IvyCudaEvent::get_wait_flags(wait_flags));
}

__CUDA_HOST__ void IvyCudaStream::add_callback(fcn_callback_t fcn, void* user_data){
  __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaLaunchHostFunc(stream_, fcn, user_data));
}

__CUDA_HOST_DEVICE__ unsigned int IvyCudaStream::get_stream_flags(StreamFlags const& flags){
  switch (flags){
  case StreamFlags::Default:
    return cudaStreamDefault;
  case StreamFlags::NonBlocking:
    return cudaStreamNonBlocking;
  default:
    __PRINT_ERROR__("IvyCudaStream::get_stream_flags: Unknown flag option...");
    assert(0);
  }
  return cudaStreamDefault;
}

#endif


#endif
