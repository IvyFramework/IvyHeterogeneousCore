#ifndef IVYCPUSTREAM_H
#define IVYCPUSTREAM_H

/**
 * @file IvyCPUStream.h
 * @brief CPU execution-stream abstraction replacing the no-op IvyBlankStream in non-CUDA builds.
 *
 * IvyCPUStream presents the same interface as IvyCudaStream so that all call sites that
 * accept an IvyGPUStream* work unmodified on CPU-only builds.  Unlike IvyBlankStream,
 * IvyCPUStream provides a real @c submit() entry point that can route work through
 * OpenMP when available, and a @c synchronize() that inserts an OMP barrier inside a
 * parallel region.
 */

#include "config/IvyCudaException.h"
#include "config/IvyOpenMPConfig.h"
#include "stream/IvyBaseStream.h"
#include "stream/IvyBlankStreamEvent.h"

// IvyBlankStream.h defines the buildRawStream / destroyRawStream specialisations
// for IvyStreamUtils::BlankStream that IvyCPUStream also inherits.
#include "stream/IvyBlankStream.h"

/**
 * @brief CPU-side stream wrapper that replaces the no-op IvyBlankStream.
 *
 * Inherits from IvyBaseStream<IvyStreamUtils::BlankStream> so that all stream-
 * utility templates (make_stream, destroy_stream, …) that are specialised for
 * BlankStream work without modification.  The class additionally exposes:
 *  - submit()      — dispatches a callable, optionally in an OpenMP region.
 *  - synchronize() — inserts an OMP barrier when inside a parallel region.
 */
class IvyCPUStream final : public IvyBaseStream<IvyStreamUtils::BlankStream>{
public:
  using Base_t = IvyBaseStream<IvyStreamUtils::BlankStream>;
  using RawStream_t = typename Base_t::RawStream_t;

  /** @brief Stream creation flags (mirrors IvyBlankStream for API compatibility). */
  enum class StreamFlags : unsigned char{
    Default,
    NonBlocking
  };

  /** @brief Construct a CPU stream with the given flags and priority. */
  __HOST__ IvyCPUStream(StreamFlags flags = StreamFlags::Default, int priority = 0) : Base_t(){}
  /** @brief Construct a non-owning wrapper around an existing raw blank stream marker. */
  __HOST_DEVICE__ IvyCPUStream(IvyStreamUtils::BlankStream st, bool do_own) : Base_t(){}
  /** @brief Copy construction is disabled; streams model unique ownership. */
  __HOST_DEVICE__ IvyCPUStream(IvyCPUStream const&) = delete;
  /** @brief Move-copy construction is disabled. */
  __HOST_DEVICE__ IvyCPUStream(IvyCPUStream const&&) = delete;
  virtual __HOST_DEVICE__ ~IvyCPUStream(){}

  /**
   * @brief Synchronize the CPU stream.
   *
   * When OpenMP is enabled and code is executing inside a parallel region this
   * inserts a barrier, ensuring all threads have completed any work submitted
   * to the stream before execution continues.  Outside a parallel region this
   * is a no-op (CPU execution is inherently sequential at that level).
   */
  __HOST__ void synchronize(){
#if defined(OPENMP_ENABLED) && defined(_OPENMP)
    if (omp_in_parallel()) {
#pragma omp barrier
    }
#endif
  }

  /** @brief No-op wait on a raw blank event. */
  __HOST__ void wait(Base_t::RawEvent_t& event, unsigned int wait_flags){}
  /** @brief No-op wait on a typed blank-stream event. */
  __HOST__ void wait(IvyBlankStreamEvent& event, IvyBlankStreamEvent::WaitFlags wait_flags = IvyBlankStreamEvent::WaitFlags::Default){}

  /**
   * @brief Submit a callable for execution on the CPU.
   *
   * When OpenMP is available the callable is invoked in the current (possibly
   * parallel) execution context.  Otherwise it is called synchronously.
   *
   * @tparam Callable  Any callable type.
   * @tparam Args      Argument pack forwarded to the callable.
   * @param f    The callable to invoke.
   * @param args Arguments to forward.
   */
  template<typename Callable, typename... Args>
  __HOST__ void submit(Callable&& f, Args&&... args){
    f(std_util::forward<Args>(args)...);
  }

  __HOST_DEVICE__ void swap(IvyCPUStream& other){ Base_t::swap(other); }

  /** @brief Map typed flags to the raw unsigned representation. */
  static __HOST_DEVICE__ unsigned int get_stream_flags(StreamFlags const& flags){
    switch (flags){
    case StreamFlags::Default:
    case StreamFlags::NonBlocking:
      return 0;
    default:
      __PRINT_ERROR__("IvyCPUStream::get_stream_flags: Unknown flag option...\n");
      assert(0);
    }
    return 0;
  }
  /** @brief Reverse-map raw unsigned flags to the typed representation. */
  static __HOST_DEVICE__ StreamFlags get_stream_flags_reverse(unsigned int const& flags){
    switch (flags){
    case 0:
      return StreamFlags::Default;
    default:
      __PRINT_ERROR__("IvyCPUStream::get_stream_flags_reverse: Unknown flag option...\n");
      assert(0);
    }
    return StreamFlags::Default;
  }
};

namespace IvyStreamUtils{
  /** @brief Destroy a CPU stream wrapper instance. */
  template<> __HOST_DEVICE__ void destroy_stream(IvyCPUStream*& stream){
    delete stream;
    stream = nullptr;
  }
  /** @brief Create a CPU stream wrapper from typed flags. */
  template<> __HOST__ void make_stream(IvyCPUStream*& stream, IvyCPUStream::StreamFlags flags, unsigned int priority){
    destroy_stream(stream);
    stream = new IvyCPUStream(flags, priority);
  }
  /** @brief Create a CPU stream wrapper from raw unsigned flags. */
  template<> __HOST__ void make_stream(IvyCPUStream*& stream, unsigned int flags, unsigned int priority){
    make_stream(stream, IvyCPUStream::get_stream_flags_reverse(flags), priority);
  }
  /** @brief Wrap an existing raw blank-stream marker in a CPU stream wrapper. */
  template<> __HOST_DEVICE__ void make_stream(IvyCPUStream*& stream, IvyCPUStream::RawStream_t st, bool is_owned){
    destroy_stream(stream);
    stream = new IvyCPUStream(st, is_owned);
  }
}

namespace std_util{
  /** @brief Swap two CPU stream wrappers. */
  __HOST_DEVICE__ void swap(IvyCPUStream& a, IvyCPUStream& b){ a.swap(b); }
}


#endif
