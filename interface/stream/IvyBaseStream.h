#ifndef IVYBASESTREAM_HH
#define IVYBASESTREAM_HH


/*
  IvyBaseStream:
  This class is a base class for any stream, requiring certain operations to be present.
*/


#include "config/IvyCompilerConfig.h"
#include "stream/IvyBaseStreamEvent.h"


namespace IvyStreamUtils{
  template<typename RawStream_t> __CUDA_HOST_DEVICE__ void createStream(RawStream_t& st, unsigned int flags, unsigned int priority);
  template<typename RawStream_t> __CUDA_HOST_DEVICE__ void destroyStream(RawStream_t& st);
}

template<typename S> class IvyBaseStream{
public:
  typedef S RawStream_t;
  typedef IvyStreamUtils::StreamEvent_t<RawStream_t> RawEvent_t;

protected:
  bool is_owned_;
  unsigned int flags_;
  int priority_;
  RawStream_t stream_;

public:
  IvyBaseStream() = default;
  __CUDA_HOST_DEVICE__ IvyBaseStream(bool const& is_owned, unsigned int const& flags, int const& priority, RawStream_t const& st) :
    is_owned_(is_owned), flags_(flags), priority_(priority), stream_(st)
  {}
  __CUDA_HOST_DEVICE__ IvyBaseStream(IvyBaseStream const&) = delete;
  __CUDA_HOST_DEVICE__ IvyBaseStream(IvyBaseStream const&&) = delete;
  virtual __CUDA_HOST_DEVICE__ ~IvyBaseStream(){ if (is_owned_) IvyStreamUtils::destroyStream(this->stream_); }

  __CUDA_HOST_DEVICE__ unsigned int const& flags() const{ return this->flags_; }
  __CUDA_HOST_DEVICE__ int const& priority() const{ return this->priority_; }
  __CUDA_HOST_DEVICE__ RawStream_t const& stream() const{ return this->stream_; }
  __CUDA_HOST_DEVICE__ operator RawStream_t const& () const{ return this->stream_; }

  __CUDA_HOST_DEVICE__ unsigned int& flags(){ return this->flags_; }
  __CUDA_HOST_DEVICE__ int& priority(){ return this->priority_; }
  __CUDA_HOST_DEVICE__ RawStream_t& stream(){ return this->stream_; }
  __CUDA_HOST_DEVICE__ operator RawStream_t& (){ return this->stream_; }

  virtual __CUDA_HOST__ void synchronize() = 0;
  virtual __CUDA_HOST__ void wait(RawEvent_t& event, unsigned int wait_flags) = 0;

  __CUDA_HOST_DEVICE__ void swap(IvyBaseStream& other){
    std_util::swap(this->is_owned_, other.is_owned_);
    std_util::swap(this->flags_, other.flags_);
    std_util::swap(this->priority_, other.priority_);
    std_util::swap(this->stream_, other.stream_);
  }
};


#endif
