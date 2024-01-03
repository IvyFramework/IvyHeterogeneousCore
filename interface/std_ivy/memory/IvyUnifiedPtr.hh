#ifndef IVYUNIFIEDPTR_HH
#define IVYUNIFIEDPTR_HH


#ifdef __USE_CUDA__

#include "config/IvyCompilerConfig.h"
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
    typedef IvyTypes::size_t size_type;
    typedef IvyTypes::size_t counter_type;
    typedef std_ivy::allocator<element_type> element_allocator_type;
    typedef std_ivy::allocator<size_type> size_allocator_type;
    typedef std_ivy::allocator<counter_type> counter_allocator_type;
    typedef std_ivy::allocator<IvyMemoryType> mem_type_allocator_type;
    typedef std_ivy::allocator_traits<element_allocator_type> element_allocator_traits;
    typedef std_ivy::allocator_traits<size_allocator_type> size_allocator_traits;
    typedef std_ivy::allocator_traits<counter_allocator_type> counter_allocator_traits;
    typedef std_ivy::allocator_traits<mem_type_allocator_type> mem_type_allocator_traits;

  protected:
    IvyMemoryType exec_mem_type_;
    IvyMemoryType* mem_type_;
    pointer ptr_;
    size_type* size_;
    counter_type* ref_count_;
    IvyGPUStream* stream_;

    __CUDA_HOST_DEVICE__ void init_members(IvyMemoryType mem_type, size_type n);
    __CUDA_HOST_DEVICE__ void release();
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void dump();

  public:
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr();
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(std_cstddef::nullptr_t);
    template<typename U>
    explicit __CUDA_HOST_DEVICE__ IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    template<typename U>
    explicit __CUDA_HOST_DEVICE__ IvyUnifiedPtr(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true>
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true>
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr&& other);
    __CUDA_HOST_DEVICE__ ~IvyUnifiedPtr();

    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr<U, IPU> const& other);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr const& other);
    template<typename U> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(U* ptr);
    __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(std_cstddef::nullptr_t);

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyMemoryType const& get_exec_memory_type() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyMemoryType* get_memory_type() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyGPUStream* gpu_stream() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type* size_ptr() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ counter_type* counter() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer get() const noexcept;

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyMemoryType& get_exec_memory_type() noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyMemoryType*& get_memory_type() noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ IvyGPUStream*& gpu_stream() noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type*& size_ptr() noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ counter_type*& counter() noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ pointer& get() noexcept;

    __CUDA_HOST_DEVICE__ size_type size() const noexcept;

    __CUDA_HOST_DEVICE__ reference operator*() const noexcept;
    __CUDA_HOST_DEVICE__ pointer operator->() const noexcept;

    __CUDA_HOST_DEVICE__ void reset();
    __CUDA_HOST_DEVICE__ void reset(std_cstddef::nullptr_t);
    template<typename U> __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    template<typename U> __CUDA_HOST_DEVICE__ void reset(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);

    template<typename U> __CUDA_HOST_DEVICE__ void swap(IvyUnifiedPtr<U, IPT>& other) noexcept;

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ counter_type use_count() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool unique() const noexcept;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ explicit operator bool() const noexcept;

    /*
    transfer: Transfers the memory type of the pointer to the new memory type.
    If transfer_all is true, pointers ref_count_ and mem_type_ are also transferred.
    Otherwise, these two pointers are created in the default memory location of the execution space.
    */
    __CUDA_HOST__ void transfer(IvyMemoryType new_mem_type, bool transfer_all);
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

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> allocate_unified(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> allocate_shared(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> allocate_unique(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  template<typename T, IvyPointerType IPT, typename... Args> __CUDA_HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args> __CUDA_HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

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
