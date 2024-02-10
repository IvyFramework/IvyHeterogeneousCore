#ifndef IVYCUDASTREAM_H
#define IVYCUDASTREAM_H


#include "stream/IvyCudaStream.hh"
#include "stream/IvyCudaEvent.h"

#ifdef __USE_CUDA__

namespace IvyStreamUtils{
  template<> __CUDA_HOST_DEVICE__ void createRawStream(cudaStream_t& st, unsigned int flags, unsigned int priority){
#if (DEVICE_CODE == DEVICE_CODE_HOST)
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamCreateWithPriority(&st, flags, priority));
#endif
  }
  template<> __CUDA_HOST_DEVICE__ void destroyRawStream(cudaStream_t& st){
#if (DEVICE_CODE == DEVICE_CODE_HOST)
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamDestroy(st));
#endif
  }

  template<> __CUDA_HOST_DEVICE__ void destroy_stream(IvyCudaStream*& stream){
    if (stream){
#if (DEVICE_CODE == DEVICE_CODE_HOST)
      __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaFree(stream));
#else
      delete stream;
#endif
      stream = nullptr;
    }
  }
  template<> __CUDA_HOST__ void make_stream(IvyCudaStream*& stream, IvyCudaStream::StreamFlags flags, unsigned int priority){
    destroy_stream(stream);
    unsigned int iflags = IvyCudaStream::get_stream_flags(flags);
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocManaged((void**) &stream, sizeof(IvyCudaStream), cudaMemAttachGlobal));
    createRawStream(stream->stream(), iflags, priority);
    stream->is_owned() = true;
    stream->flags() = iflags;
    stream->priority() = priority;
  }
  template<> __CUDA_HOST__ void make_stream(IvyCudaStream*& stream, unsigned int flags, unsigned int priority){ make_stream(stream, IvyCudaStream::get_stream_flags_reverse(flags), priority); }
  template<> __CUDA_HOST_DEVICE__ void make_stream(IvyCudaStream*& stream, IvyCudaStream::RawStream_t st, bool is_owned){
    destroy_stream(stream);
#if (DEVICE_CODE == DEVICE_CODE_HOST)
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaMallocManaged((void**) &stream, sizeof(IvyCudaStream), cudaMemAttachGlobal));
    stream->stream() = st;
    stream->is_owned() = is_owned;
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamGetFlags(st, &(stream->flags())));
    __CUDA_CHECK_OR_EXIT_WITH_ERROR__(cudaStreamGetPriority(st, &(stream->priority())));
#else
    stream = new IvyCudaStream(st, is_owned);
#endif
  }
}

__CUDA_HOST__ IvyCudaStream::IvyCudaStream(StreamFlags flags, int priority) : IvyBaseStream<cudaStream_t>()
{
  is_owned_ = true;
  flags_ = get_stream_flags(flags);
  priority_ = priority;

  IvyStreamUtils::createRawStream(stream_, flags_, priority_);
}
__CUDA_HOST_DEVICE__ IvyCudaStream::IvyCudaStream(
  cudaStream_t st,
#if (DEVICE_CODE == DEVICE_CODE_HOST)
  bool do_own
#else
  bool
#endif
) : IvyBaseStream<cudaStream_t>()
{
  stream_ = st;
#if (DEVICE_CODE == DEVICE_CODE_HOST)
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
    __PRINT_ERROR__("IvyCudaStream::get_stream_flags: Unknown flag option...\n");
    assert(0);
  }
  return cudaStreamDefault;
}
__CUDA_HOST_DEVICE__ IvyCudaStream::StreamFlags IvyCudaStream::get_stream_flags_reverse(unsigned int const& flags){
  switch (flags){
  case cudaStreamDefault:
    return StreamFlags::Default;
  case cudaStreamNonBlocking:
    return StreamFlags::NonBlocking;
  default:
    __PRINT_ERROR__("IvyCudaStream::get_stream_flags_reverse: Unknown flag option...\n");
    assert(0);
  }
  return StreamFlags::Default;
}


#endif


#endif
