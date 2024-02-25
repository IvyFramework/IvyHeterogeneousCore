#ifndef IVYALLOCATOR_H
#define IVYALLOCATOR_H


#include "config/IvyCompilerConfig.h"
#include "IvyMemoryHelpers.h"

#ifdef __USE_CUDA__

#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyLimits.h"


namespace std_ivy{
  /*
  Base class of allocator primitives
  */
  template<typename T> class allocation_type_properties{
  public:
    typedef T value_type;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef IvyTypes::size_t size_type;
    typedef IvyTypes::ptrdiff_t difference_type;

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer address(reference x){ return &x; }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ const_pointer address(const_reference x){ return &x; }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type max_size() noexcept{
      return std_limits::numeric_limits<size_type>::max() / sizeof(T);
    }
  };

  /*
  allocator primitives
  */
  template<typename T> class allocator_primitive{
  public:
    using base_t = allocation_type_properties<T>;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate(
      pointer& tgt, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::allocate_memory(tgt, n, mem_type, stream);
    }
    static __CUDA_HOST_DEVICE__ pointer allocate(
      size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      pointer res = nullptr;
      IvyMemoryHelpers::allocate_memory(res, n, mem_type, stream);
      return res;
    }

    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool construct(
      pointer& tgt, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args
    ){
      return IvyMemoryHelpers::construct(tgt, n, mem_type, stream, args...);
    }

    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool build(
      pointer& tgt, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args
    ){
      return IvyMemoryHelpers::build(tgt, n, mem_type, stream, args...);
    }
    template<typename... Args> static __CUDA_HOST_DEVICE__ pointer build(
      size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args
    ){
      pointer res = nullptr;
      IvyMemoryHelpers::build(res, n, mem_type, stream, args...);
      return res;
    }
  };
  template<typename T> class deallocator_primitive{
  public:
    using base_t = allocation_type_properties<T>;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool deallocate(
      pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::free_memory(p, n, mem_type, stream);
    }

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destruct(
      pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::destruct(p, n, mem_type, stream);
    }

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destroy(
      pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::destroy(p, n, mem_type, stream);
    }
  };

  /*
  We hold the convention to codfy all classes with internal data using a public or protected
  'bool transfer_internal_memory(IvyMemoryType const& new_mem_type)' member function.
  */
  template<typename T> class kernel_generic_transfer_internal_memory;
  template<typename T> class transfer_memory_primitive_without_internal_memory;
  template<typename T> class transfer_memory_primitive_with_internal_memory;
  template<typename T> class transfer_memory_primitive;

  template<typename T> class kernel_generic_transfer_internal_memory final : public kernel_base_noprep_nofin{
  protected:
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_internal_memory(T* const& ptr, IvyMemoryType const& mem_type, bool release_old){
      return ptr->transfer_internal_memory(mem_type, release_old);
    }

  public:
    static __CUDA_HOST_DEVICE__ void kernel(
      IvyTypes::size_t const& i, IvyTypes::size_t const& n, T* const& ptr,
      IvyMemoryType const& mem_type, bool const& release_old
    ){
      if (kernel_check_dims<kernel_generic_transfer_internal_memory<T>>::check_dims(i, n)) transfer_internal_memory(ptr+i, mem_type, release_old);
    }

    friend class transfer_memory_primitive_with_internal_memory<T>;
  };

  template<typename T> class transfer_memory_primitive_without_internal_memory{
  public:
    using base_t = allocation_type_properties<T>;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      return IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
    }
  };
  template<typename T> class transfer_memory_primitive_with_internal_memory{
  public:
    using base_t = allocation_type_properties<T>;
    using value_type = typename base_t::value_type;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;
    using kernel_type = kernel_generic_transfer_internal_memory<value_type>;

    static __CUDA_HOST_DEVICE__ bool transfer_internal_memory(pointer ptr, IvyTypes::size_t const& n, IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream, bool release_old){
      bool res = true;
      if (IvyMemoryHelpers::run_acc_on_host(ptr_mem_type)){
        if (!run_kernel<kernel_type>(0, stream).parallel_1D(n, ptr, mem_type, release_old)){
          __PRINT_ERROR__("transfer_memory_primitive::transfer_internal_memory: Unable to call the acc. hardware kernel...\n");
          res = false;
        }
      }
      else{
        pointer pr = ptr;
        for (size_type i=0; i<n; ++i){
          res &= kernel_type::transfer_internal_memory(pr, mem_type, release_old);
          ++pr;
        }
      }
      return res;
    }

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      if (!src) return false;
      bool res = true;
      /*
#if DEVICE_CODE == DEVICE_CODE_HOST
      printf("transfer_memory_primitive_with_internal_memory::transfer: type = %s | n=%llu | src = %p (%d), tgt = %p (%d)\n", typeid(T).name(), n, src, int(type_src), tgt, int(type_tgt));
#endif
      */
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      constexpr bool release_old = false; // We do not release existing memory from internal memory transfers in order to preserve src pointer.
      if (def_mem_type==type_tgt && def_mem_type==type_src){
        res &= IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
        res &= transfer_internal_memory(tgt, n, type_tgt, type_tgt, stream, release_old);
      }
      else{
        pointer p_int = nullptr;
        res &= IvyMemoryHelpers::allocate_memory(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::transfer_memory(p_int, src, n, def_mem_type, type_src, stream);
        res &= transfer_internal_memory(p_int, n, def_mem_type, type_tgt, stream, release_old);
        res &= IvyMemoryHelpers::transfer_memory(tgt, p_int, n, type_tgt, def_mem_type, stream);
        res &= IvyMemoryHelpers::free_memory(p_int, n, def_mem_type, stream);
      }
      return res;
    }
  };
  // By default, we assume that the class has no internal memory to transfer.
  template<typename T> class transfer_memory_primitive : public transfer_memory_primitive_without_internal_memory<T>{};

  /*
  allocator
  */
  template<typename T> class allocator : public allocation_type_properties<T>, public allocator_primitive<T>, public deallocator_primitive<T>, public transfer_memory_primitive<T>{
  public:
    using base_t = allocation_type_properties<T>;
    using value_type = typename base_t::value_type;
    using pointer = typename base_t::pointer;
    using const_pointer = typename base_t::const_pointer;
    using reference = typename base_t::reference;
    using const_reference = typename base_t::const_reference;
    using size_type = typename base_t::size_type;
    using difference_type = typename base_t::difference_type;

    allocator() noexcept = default;
    __CUDA_HOST_DEVICE__ allocator(allocator const& other) noexcept{}
    template<typename U> __CUDA_HOST_DEVICE__ allocator(allocator<U> const& other) noexcept{}
    /*__CUDA_HOST_DEVICE__*/ ~allocator() noexcept = default;
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

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer allocate(allocator_type const& a, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.allocate(n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate(allocator_type const& a, pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.allocate(ptr, n, mem_type, stream);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool construct(allocator_type const& a, pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.construct(ptr, n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer build(allocator_type const& a, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.build(n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool build(allocator_type const& a, pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return a.build(ptr, n, mem_type, stream, args...);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool deallocate(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.deallocate(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destruct(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.destruct(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destroy(allocator_type const& a, pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return a.destroy(p, n, mem_type, stream);
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

    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer allocate(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::allocate(n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool allocate(pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::allocate(ptr, n, mem_type, stream);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer construct(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::construct(n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool construct(pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::construct(ptr, n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer build(size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::build(n, mem_type, stream, args...);
    }
    template<typename... Args> static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool build(pointer& ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream, Args&&... args){
      return allocator_type::build(ptr, n, mem_type, stream, args...);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool deallocate(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::deallocate(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destruct(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::destruct(p, n, mem_type, stream);
    }
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool destroy(pointer& p, size_type n, IvyMemoryType mem_type, IvyGPUStream& stream){
      return allocator_type::destroy(p, n, mem_type, stream);
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


// IvyMemoryHelpers copy_data functionality should rely on allocators so that complex data structures can be copied correctly.
namespace IvyMemoryHelpers{
  /*
  copy_data: Copies data from a pointer of type U to a pointer of type T.
  - target: Pointer to the target data.
  - source: Pointer to the source data.
  - n_tgt_init: Number of elements in the target array before the copy. If the target array is not null, it is freed.
  - n_tgt: Number of elements in the target array after the copy.
  - n_src: Number of elements in the source array before the copy. It has to satisfy the constraint (n_src==n_tgt || n_src==1).
  When using CUDA, the following additional arguments are required:
  - type_tgt: Location of the target data in memory.
  - type_src: Location of the source data in memory.
  - stream: CUDA stream to use for the copy.
    If stream is anything other than cudaStreamLegacy, the copy is asynchronous, even in device code.
  */
  template<typename T, typename U> struct copy_data_fcnal{
    static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool copy_data(
      T*& target, U* const& source,
      size_t n_tgt_init, size_t n_tgt, size_t n_src
      , IvyMemoryType type_tgt, IvyMemoryType type_src
      , IvyGPUStream& stream
    );
  };
  template<typename T, typename U> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
    , IvyMemoryType type_tgt, IvyMemoryType type_src
    , IvyGPUStream& stream
  ){
    return copy_data_fcnal<T, U>::copy_data(
      target, source,
      n_tgt_init, n_tgt, n_src
      , type_tgt, type_src
      , stream
    );
  }

  /*
  Overload to allow passing raw cudaStream_t objects.
  */
#ifdef __USE_CUDA__
  template<typename T, typename U> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src,
    IvyMemoryType type_tgt, IvyMemoryType type_src,
    cudaStream_t stream
  ){
    IvyGPUStream sr(stream, false);
    return copy_data(
      target, source,
      n_tgt_init, n_tgt, n_src,
      type_tgt, type_src,
      sr
    );
  }
#endif
}

namespace IvyMemoryHelpers{
    template<typename T, typename U> __CUDA_HOST_DEVICE__ bool copy_data_fcnal<T, U>::copy_data(
    T*& target, U* const& source,
    size_t n_tgt_init, size_t n_tgt, size_t n_src
    , IvyMemoryType type_tgt, IvyMemoryType type_src
    , IvyGPUStream& stream
  ){
    bool res = true;
#if (DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__)
    bool const tgt_on_device = use_device_acc(type_tgt);
    bool const src_on_device = use_device_acc(type_src);
#else
    constexpr bool tgt_on_device = true;
    constexpr bool src_on_device = true;
#endif
    if (n_tgt==0 || n_src==0 || !source) return false;
    if (!(n_src==n_tgt || n_src==1)){
#if COMPILER == COMPILER_MSVC
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data: Invalid values for n_tgt=%Iu, n_src=%Iu\n", n_tgt, n_src);
#else
      __PRINT_ERROR__("IvyMemoryHelpers::copy_data: Invalid values for n_tgt=%zu, n_src=%zu\n", n_tgt, n_src);
#endif
      assert(0);
    }
    if (n_tgt_init!=n_tgt){
#ifdef __USE_CUDA__
      res &= std_ivy::deallocator_primitive<T>::destroy(target, n_tgt_init, type_tgt, stream);
      res &= std_ivy::allocator_primitive<T>::allocate(target, n_tgt, type_tgt, stream);
#else
      res &= destroy(target, n_tgt_init, type_tgt, stream);
      res &= allocate(target, n_tgt, type_tgt, stream);
#endif
    }
    if (res){
#ifdef __USE_CUDA__
      U* d_source = (src_on_device ? source : nullptr);
      if (!src_on_device){
        res &= std_ivy::allocator_primitive<U>::allocate(d_source, n_src, IvyMemoryType::GPU, stream);
        res &= std_ivy::transfer_memory_primitive<U>::transfer(d_source, source, n_src, IvyMemoryType::GPU, type_src, stream);
      }
      T* d_target = nullptr;
      if (!tgt_on_device) res &= std_ivy::allocator_primitive<T>::allocate(d_target, n_tgt, IvyMemoryType::GPU, stream);
      else d_target = target;
      res &= run_kernel<copy_data_kernel<T, U>>(0, stream).parallel_1D(n_tgt, n_src, d_target, d_source);
      if (!tgt_on_device){
        res &= std_ivy::transfer_memory_primitive<T>::transfer(target, d_target, n_tgt, type_tgt, IvyMemoryType::GPU, stream);
        res &= std_ivy::deallocator_primitive<T>::deallocate(d_target, n_tgt, IvyMemoryType::GPU, stream);
      }
      if (!src_on_device) res &= std_ivy::deallocator_primitive<U>::deallocate(d_source, n_src, IvyMemoryType::GPU, stream);
      if (!res){
        if (tgt_on_device!=src_on_device){
          __PRINT_ERROR__("IvyMemoryHelpers::copy_data: Failed to copy data between host and device.\n");
          assert(0);
        }
#else
      {
#endif
        for (size_t i=0; i<n_tgt; ++i) target[i] = source[(n_src==1 ? 0 : i)];
      }
    }
    return res;
  }
}


#endif
