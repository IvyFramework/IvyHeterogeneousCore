#ifndef IVYALLOCATOR_HH
#define IVYALLOCATOR_HH


#include "IvyCompilerFlags.h"
#include "IvyCudaFlags.h"


#ifdef __USE_CUDA__

#include "IvyUtility.h"
#include "IvyLimits.h"
#include "IvyMemoryHelpers.h"


namespace std_ivy{
  /*
  allocator
  */
  template<typename T> class allocator{
  public:
    typedef T value_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef IvyMemoryHelpers::size_t size_type;
    typedef IvyMemoryHelpers::ptrdiff_t difference_type;

    __CUDA_HOST_DEVICE__ allocator() noexcept{}
    __CUDA_HOST_DEVICE__ allocator(allocator const& other) noexcept{}
    template<typename U> __CUDA_HOST_DEVICE__ allocator(allocator<U> const& other) noexcept{}
    /*__CUDA_HOST_DEVICE__*/ ~allocator() = default;
    __CUDA_HOST_DEVICE__ pointer address(reference x) const{ return &x; }
    __CUDA_HOST_DEVICE__ const_pointer address(const_reference x) const{ return &x; }

    __CUDA_HOST_DEVICE__ pointer allocate(size_type n, bool use_cuda_device_mem = false){
      pointer ret;
      IvyMemoryHelpers::allocate_memory(ret, n, use_cuda_device_mem);
      return ret;
    }
    __CUDA_HOST_DEVICE__ void deallocate(pointer p, size_type n, bool use_cuda_device_mem = false){
      IvyMemoryHelpers::free_memory(p, n, use_cuda_device_mem);
    }

    __CUDA_HOST__ bool transfer(pointer tgt, pointer src, size_t n, bool device_to_host){
      return IvyMemoryHelpers::transfer_memory(tgt, src, n, device_to_host);
    }

    static __CUDA_HOST_DEVICE__ size_type max_size() noexcept{
      return std_limits::numeric_limits<size_type>::max() / sizeof(T);
    }
  };
  template<typename T, typename U> bool operator==(std_ivy::allocator<T> const&, std_ivy::allocator<U> const&) noexcept{ return true; }
  template<typename T, typename U> bool operator!=(std_ivy::allocator<T> const& a1, std_ivy::allocator<U> const& a2) noexcept{ return !(a1==a2); }

  /*
  allocator_traits
  */
  // LEFT HERE: implement allocator_traits static equivalents in CUDA
  template<typename Allocator_t> class allocator_traits{
  public:
    typedef Allocator_t allocator_type;
    typedef typename allocator_type::value_type value_type;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef typename allocator_type::difference_type difference_type;
    typedef typename allocator_type::size_type size_type;

    static pointer allocate(allocator_type& a, size_type n, bool use_cuda_device_mem = false){
      return a.allocate(n, use_cuda_device_mem);
    }
    static void deallocate(allocator_type& a, pointer p, size_type n, bool use_cuda_device_mem = false){
      a.deallocate(p, n, use_cuda_device_mem);
    }

    static size_type max_size(const allocator_type& a) noexcept{
      return a.max_size();
    }

  };

  /*
  allocator_arg_t
  */
  struct allocator_arg_t { explicit /*__CUDA_HOST_DEVICE__*/ allocator_arg_t() = default; };
#define allocator_arg std_ivy::allocator_arg_t();

  /*
  
  */

}


#endif


#endif
