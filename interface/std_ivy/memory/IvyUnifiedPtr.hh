#ifndef IVYUNIFIEDPTR_HH
#define IVYUNIFIEDPTR_HH


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyCstddef.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/memory/IvyAllocator.h"
#include "std_ivy/memory/IvyPointerTraits.h"
//#include "std_ivy/IvyFunctional.h"
#include "IvyPrintout.h"


namespace std_ivy{
  enum class IvyPointerType{
    shared,
    unique
  };

  template<typename T, IvyPointerType IPT> class IvyUnifiedPtr;
  template<typename T, IvyPointerType IPT> class transfer_memory_primitive<IvyUnifiedPtr<T, IPT>> : public transfer_memory_primitive_with_internal_memory<IvyUnifiedPtr<T, IPT>>{};

  template<typename T, IvyPointerType IPT> class IvyUnifiedPtr{
  public:
    typedef T element_type;
    typedef T* pointer;
    typedef T& reference;
    typedef IvyTypes::size_t size_type;
    typedef IvyTypes::size_t counter_type;
    typedef size_type difference_type;
    typedef std_ivy::allocator<element_type> element_allocator_type;
    typedef std_ivy::allocator<size_type> size_allocator_type;
    typedef std_ivy::allocator<counter_type> counter_allocator_type;
    typedef std_ivy::allocator<IvyMemoryType> mem_type_allocator_type;
    typedef std_ivy::allocator_traits<element_allocator_type> element_allocator_traits;
    typedef std_ivy::allocator_traits<size_allocator_type> size_allocator_traits;
    typedef std_ivy::allocator_traits<counter_allocator_type> counter_allocator_traits;
    typedef std_ivy::allocator_traits<mem_type_allocator_type> mem_type_allocator_traits;

    template<typename U> using rebind = IvyUnifiedPtr<U, IPT>;

    friend class kernel_generic_transfer_internal_memory<IvyUnifiedPtr<T, IPT>>;

  protected:
    pointer ptr_;
    IvyMemoryType* mem_type_;
    size_type* size_;
    size_type* capacity_;
    counter_type* ref_count_;
    IvyGPUStream* stream_;
    IvyMemoryType exec_mem_type_;
    IvyMemoryType progenitor_mem_type_;

    __HOST_DEVICE__ void init_members(IvyMemoryType mem_type, size_type n_size, size_type n_capacity);
    __HOST_DEVICE__ void release();
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void dump();

    __HOST_DEVICE__ void inc_dec_counter(bool do_inc);
    __HOST_DEVICE__ void inc_dec_size(bool do_inc);
    __HOST_DEVICE__ void inc_dec_capacity(bool do_inc, size_type inc=1);

    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old);

    /*
    transfer_impl: Implementation for transferring the memory type of the pointer to the new memory type.
    If transfer_all is true, pointers ref_count_ and mem_type_ are also transferred.
    Otherwise, these two pointers are created in the default memory location of the execution space.
    If copy_ptr is true, a new pointer is created.
    If release_old is true, the old pointers are released.
    */
    __HOST_DEVICE__ bool transfer_impl(IvyMemoryType const& new_mem_type, bool transfer_all, bool copy_ptr, bool release_old);

  public:
    __HOST_DEVICE__ IvyUnifiedPtr();
    __HOST_DEVICE__ IvyUnifiedPtr(std_cstddef::nullptr_t);
    explicit __HOST_DEVICE__ IvyUnifiedPtr(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    explicit __HOST_DEVICE__ IvyUnifiedPtr(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    explicit __HOST_DEVICE__ IvyUnifiedPtr(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U>
    explicit __HOST_DEVICE__ IvyUnifiedPtr(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U>
    explicit __HOST_DEVICE__ IvyUnifiedPtr(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U>
    explicit __HOST_DEVICE__ IvyUnifiedPtr(U* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true>
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU> const& other);
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<T, IPT> const& other);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true>
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr<U, IPU>&& other);
    __HOST_DEVICE__ IvyUnifiedPtr(IvyUnifiedPtr&& other);
    __HOST_DEVICE__ ~IvyUnifiedPtr();

    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true>
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr<U, IPU> const& other);
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr const& other);
    template<typename U, IvyPointerType IPU, std_ttraits::enable_if_t<IPU==IPT || IPU==IvyPointerType::unique, bool> = true>
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr<U, IPU>&& other);
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(IvyUnifiedPtr&& other);
    __HOST_DEVICE__ IvyUnifiedPtr<T, IPT>& operator=(std_cstddef::nullptr_t);

    // Copy n values from external pointer via memory transfer
    __HOST_DEVICE__ bool copy(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);

    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType const& get_progenitor_memory_type() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType const& get_exec_memory_type() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType* get_memory_type_ptr() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyGPUStream* gpu_stream() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type* size_ptr() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type* capacity_ptr() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ counter_type* counter() const __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer get() const __NOEXCEPT__;

    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType& get_progenitor_memory_type() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType& get_exec_memory_type() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyMemoryType*& get_memory_type_ptr() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ IvyGPUStream*& gpu_stream() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type*& size_ptr() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type*& capacity_ptr() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ counter_type*& counter() __NOEXCEPT__;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ pointer& get() __NOEXCEPT__;

    __HOST_DEVICE__ size_type size() const __NOEXCEPT__;
    __HOST_DEVICE__ size_type capacity() const __NOEXCEPT__;
    __HOST_DEVICE__ IvyMemoryType get_memory_type() const __NOEXCEPT__;

    __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool empty() const __NOEXCEPT__;

    __HOST_DEVICE__ reference operator*() const __NOEXCEPT__;
    __HOST_DEVICE__ reference operator[](size_type k) const;
    __HOST_DEVICE__ pointer operator->() const __NOEXCEPT__;

    __HOST_DEVICE__ void reset();
    __HOST_DEVICE__ void reset(std_cstddef::nullptr_t);
    template<typename U> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void reset(U* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U> __HOST_DEVICE__ void reset(U* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename U> __HOST_DEVICE__ void reset(U* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void reset(T* ptr, IvyMemoryType mem_type, IvyGPUStream* stream);
    __HOST_DEVICE__ void reset(T* ptr, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    __HOST_DEVICE__ void reset(T* ptr, size_type n_size, size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream);

    template<typename U> __HOST_DEVICE__ void swap(IvyUnifiedPtr<U, IPT>& other) __NOEXCEPT__;
    __HOST_DEVICE__ void swap(IvyUnifiedPtr<T, IPT>& other) __NOEXCEPT__;

    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ counter_type use_count() const;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool unique() const;
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ explicit operator bool() const __NOEXCEPT__;

    /*
    transfer: Transfers the memory type of the pointer to the new memory type.
    If transfer_all is true, pointers ref_count_ and mem_type_ are also transferred.
    Otherwise, these two pointers are created in the default memory location of the execution space.

    See also transfer_impl for the internal implementation, allowing the user to also make a new copy of the pointer.
    */
    __HOST__ bool transfer(IvyMemoryType const& new_mem_type, bool transfer_all, bool release_old);

    __HOST_DEVICE__ void reserve(size_type const& n);
    __HOST_DEVICE__ void reserve(size_type const& n, IvyMemoryType new_mem_type, IvyGPUStream* stream);
    template<typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void emplace_back(Args&&... args);
    template<typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void push_back(Args&&... args);
    template<typename... Args> __HOST_DEVICE__ void insert(size_type const& i, Args&&... args);
    __HOST_DEVICE__ void pop_back();
    __HOST_DEVICE__ void erase(size_type const& i);
    __HOST_DEVICE__ void shrink_to_fit();

    // Write access checks using progenitor_mem_type_ that do not result in aborting the program
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_write_access() const;
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ bool check_write_access(IvyMemoryType const& mem_type) const;
    // Write access checks using progenitor_mem_type_ that do result in aborting the program
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ void check_write_access_or_die() const;
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ void check_write_access_or_die(IvyMemoryType const& mem_type) const;
  };

  template<typename T> using shared_ptr = IvyUnifiedPtr<T, IvyPointerType::shared>;
  template<typename T> using unique_ptr = IvyUnifiedPtr<T, IvyPointerType::unique>;

  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__;
  template<typename T, typename U, IvyPointerType IPT, IvyPointerType IPU> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, IvyUnifiedPtr<U, IPU> const& b) __NOEXCEPT__;

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__;
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, T* ptr) __NOEXCEPT__;

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(T* ptr, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__;
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(IvyUnifiedPtr<T, IPT> const& a, std_cstddef::nullptr_t) __NOEXCEPT__;

  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator==(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;
  template<typename T, IvyPointerType IPT> __HOST_DEVICE__ bool operator!=(std_cstddef::nullptr_t, IvyUnifiedPtr<T, IPT> const& a) __NOEXCEPT__;

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n_size, typename IvyUnifiedPtr<T, IPT>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n_size, typename shared_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(typename unique_ptr<T>::size_type n_size, typename unique_ptr<T>::size_type n_capacity, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, typename unique_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(typename IvyUnifiedPtr<T, IPT>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(typename shared_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(typename unique_ptr<T>::size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  template<typename T, IvyPointerType IPT, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> build_unified(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> build_shared(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename Allocator_t, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> build_unique(Allocator_t const& a, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, IvyPointerType IPT, typename... Args>
  __HOST_DEVICE__ IvyUnifiedPtr<T, IPT> make_unified(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args>
  __HOST_DEVICE__ shared_ptr<T> make_shared(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
  template<typename T, typename... Args>
  __HOST_DEVICE__ unique_ptr<T> make_unique(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

  /*
  Specialization of pointer_traits:
  There is no pointer_to for IvyUnifiedPtr since this class is for shared/unique pointers with ownership.
  */
  template<typename T, IvyPointerType IPT> class pointer_traits<IvyUnifiedPtr<T, IPT>>{
  public:
    static constexpr IvyPointerType ivy_ptr_type = IPT;
    typedef IvyUnifiedPtr<T, IPT> pointer;
    typedef typename pointer::element_type element_type;
    typedef typename pointer::difference_type difference_type;
    template <typename U> using rebind = pointer_traits_rebind_t<pointer, U>;
  };

  /*
  Specialization of the value printout routine
  */
  template<typename T, IvyPointerType IPT> struct value_printout<IvyUnifiedPtr<T, IPT>>;
}
namespace IvyTypes{
  template<typename T, std_ivy::IvyPointerType IPT> struct convert_to_floating_point<std_ivy::IvyUnifiedPtr<T, IPT>>{
    using type = std_ivy::IvyUnifiedPtr<convert_to_floating_point_t<T>, IPT>;
  };
}

// Extension of std_fcnal::hash
// Current CUDA C++ library omits hashes, so we defer it as well.
/*
namespace std_fcnal{
  template<typename T, std_ivy::IvyPointerType IPT> struct hash<std_ivy::IvyUnifiedPtr<T, IPT>>{
    using argument_type = std_ivy::IvyUnifiedPtr<T, IPT>;
    using arg_ptr_t = typename std_ivy::IvyUnifiedPtr<T, IPT>::pointer;
    using result_type = typename hash<arg_ptr_t>::result_type;

    __HOST_DEVICE__ result_type operator()(argument_type const& arg) const{ return hash<arg_ptr_t>{}(arg.get()); }
  };
}
*/

/*
Extension of std_util::swap
*/
namespace std_util{
  template<typename T, typename U, std_ivy::IvyPointerType IPT> __HOST_DEVICE__ void swap(std_ivy::IvyUnifiedPtr<T, IPT>& a, std_ivy::IvyUnifiedPtr<U, IPT>& b) __NOEXCEPT__;
  template<typename T, std_ivy::IvyPointerType IPT> __HOST_DEVICE__ void swap(std_ivy::IvyUnifiedPtr<T, IPT>& a, std_ivy::IvyUnifiedPtr<T, IPT>& b) __NOEXCEPT__;
}


#endif
