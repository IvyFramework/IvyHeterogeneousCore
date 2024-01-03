#ifndef IVYALLOCATOR_H
#define IVYALLOCATOR_H


#include "config/IvyCompilerConfig.h"
#include "IvyMemoryHelpers.h"

#ifdef __USE_CUDA__

#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyLimits.h"


namespace std_ivy{
  /*
  allocator
  */
  template<typename T> class allocator{
  public:
    typedef T value_type;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef IvyTypes::size_t size_type;
    typedef IvyTypes::ptrdiff_t difference_type;

    allocator() noexcept = default;
    __CUDA_HOST_DEVICE__ allocator(allocator const& other) noexcept{}
    template<typename U> __CUDA_HOST_DEVICE__ allocator(allocator<U> const& other) noexcept{}
    /*__CUDA_HOST_DEVICE__*/ ~allocator() = default;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer address(reference x) const{ return &x; }
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ const_pointer address(const_reference x) const{ return &x; }

    template<typename... Args> static __CUDA_HOST_DEVICE__ pointer allocate(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      pointer ret = nullptr;
      IvyMemoryHelpers::allocate_memory(ret, n, mem_type, stream, args...);
      return ret;
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void deallocate(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      IvyMemoryHelpers::free_memory(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type max_size() noexcept{
      return std_limits::numeric_limits<size_type>::max() / sizeof(T);
    }
  };
  template<typename T, typename U> bool operator==(std_ivy::allocator<T> const&, std_ivy::allocator<U> const&) noexcept{ return true; }
  template<typename T, typename U> bool operator!=(std_ivy::allocator<T> const& a1, std_ivy::allocator<U> const& a2) noexcept{ return !(a1==a2); }

  /*
  allocator_traits
  */
  template<typename Allocator_t> class allocator_traits{
  public:
    typedef Allocator_t allocator_type;
    typedef typename allocator_type::value_type value_type;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef typename allocator_type::size_type size_type;
    typedef typename allocator_type::difference_type difference_type;

    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer allocate(allocator_type const& a, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.allocate(n, mem_type, stream, args...);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void deallocate(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      a.deallocate(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer(
      allocator_type const& a,
      pointer& tgt, pointer const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return a.transfer(tgt, src, n, type_tgt, type_src, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type max_size(allocator_type const& a) noexcept{
      return a.max_size();
    }

    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer allocate(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::allocate(n, mem_type, stream, args...);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void deallocate(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      allocator_type::deallocate(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_t n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return allocator_type::transfer(tgt, src, n, type_tgt, type_src, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type max_size() noexcept{
      return allocator_type::max_size();
    }

  };

  /*
  allocator_arg_t
  */
  struct allocator_arg_t { explicit /*__CUDA_HOST_DEVICE__*/ allocator_arg_t() = default; };
  inline constexpr allocator_arg_t allocator_arg;

  /*
  
  */

}


#endif


#endif
