#ifndef IVYUNIFIEDPTR_HH
#define IVYUNIFIEDPTR_HH


#ifdef __USE_CUDA__

#include "IvyCompilerFlags.h"
#include "IvyCudaFlags.h"
#include "std_ivy/IvyCstddef.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/memory/IvyAllocator.h"
#include "std_ivy/IvyFunctional.h"


namespace std_ivy{
  enum class IvyPointerType{
    shared,
    unique
  };


  template<typename T, IvyPointerType IPT> class IvyUnifiedPtr{
  public:
    typedef T element_type;
    typedef T* pointer;
    typedef T& reference;
    typedef unsigned long long int counter_type;

  protected:
    bool* is_on_device_;
    pointer ptr_;
    counter_type* ref_count_;
    cudaStream_t* stream_;

    __CUDA_HOST_DEVICE__ void init_members(bool is_on_device);
    __CUDA_HOST_DEVICE__ void release();
    __CUDA_HOST_DEVICE__ void dump();

  public:
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr();
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(std_cstddef::nullptr_t);
    template<typename U> explicit __CUDA_HOST_DEVICE__ IvyUnifiedPtr(U* ptr, bool is_on_device, cudaStream_t* stream = nullptr);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true> __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true> __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr&& other);
    __CUDA_HOST_DEVICE__ ~IvyUnifiedPtr();

    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr<U, IPU> const& other);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr const& other);
    template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(U* ptr);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(std_cstddef::nullptr_t);

    __CUDA_HOST_DEVICE__ bool* use_gpu() const noexcept;
    __CUDA_HOST_DEVICE__ cudaStream_t* gpu_stream() const noexcept;
    __CUDA_HOST_DEVICE__ counter_type* counter() const noexcept;
    __CUDA_HOST_DEVICE__ pointer get() const noexcept;

    __CUDA_HOST_DEVICE__ bool*& use_gpu() noexcept;
    __CUDA_HOST_DEVICE__ cudaStream_t*& gpu_stream() noexcept;
    __CUDA_HOST_DEVICE__ counter_type*& counter() noexcept;
    __CUDA_HOST_DEVICE__ pointer& get() noexcept;

    __CUDA_HOST_DEVICE__ reference operator*() const noexcept;
    __CUDA_HOST_DEVICE__ pointer operator->() const noexcept;

    __CUDA_HOST_DEVICE__ void reset();
    __CUDA_HOST_DEVICE__ void reset(std_cstddef::nullptr_t);
    template<typename U> __CUDA_HOST_DEVICE__ void reset(U* ptr, bool is_on_device, cudaStream_t* stream = nullptr);

    template<typename U> __CUDA_HOST_DEVICE__ void swap(IvyUnifiedPtr<U, IPT>& other) noexcept;

    __CUDA_HOST_DEVICE__ counter_type use_count() const noexcept;
    __CUDA_HOST_DEVICE__ bool unique() const noexcept;
    __CUDA_HOST_DEVICE__ explicit operator bool() const noexcept;
  };

  template<typename T> using shared_ptr = IvyUnifiedPtr<T, IvyPointerType::shared>;
  template<typename T> using unique_ptr = IvyUnifiedPtr<T, IvyPointerType::unique>;

  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) noexcept;
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) noexcept;

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, T* ptr) noexcept;
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, T* ptr) noexcept;

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(T* ptr, IvyUnifiedPtr<T, IPT> const& a) noexcept;
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(T* ptr, IvyUnifiedPtr<T, IPT> const& a) noexcept;

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) noexcept;
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) noexcept;

  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) noexcept;
  template<typename T, IvyPointerType IPT> __CUDA_HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) noexcept;

  template<typename T, typename U, IvyPointerType IPT> __CUDA_HOST_DEVICE__ void swap(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPT> const& b) noexcept;

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> allocate_unified(Allocator_t const& a, bool is_on_device, cudaStream_t* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> allocate_shared(Allocator_t const& a, bool is_on_device, cudaStream_t* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> allocate_unique(Allocator_t const& a, bool is_on_device, cudaStream_t* stream, Args&&... args);

  template<typename T, IvyPointerType IPT, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(bool is_on_device, cudaStream_t* stream, Args&&... args);
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(bool is_on_device, cudaStream_t* stream, Args&&... args);
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(bool is_on_device, cudaStream_t* stream, Args&&... args);

}

// Extension of std_fcnal::hash
// Current CUDA C++ library omits hashes, so we defer it as well.
/*
namespace std_fcnal{
  template<typename T, std_ivy::IvyPointerType IPT> struct hash<std_ivy::IvyUnifiedPtr<T, IPT>>{
    using argument_type = std_ivy::IvyUnifiedPtr<T, IPT>;
    using arg_ptr_t = typename std_ivy::IvyUnifiedPtr<T, IPT>::pointer;
    using result_type = typename hash<arg_ptr_t>::result_type;

    __CUDA_HOST_DEVICE__ result_type operator()(argument_type const& arg) const{ return hash<arg_ptr_t>{}(arg.get()); }
  };
}
*/

#endif


#endif
