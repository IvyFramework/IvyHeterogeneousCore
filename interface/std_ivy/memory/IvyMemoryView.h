/**
 * @file IvyMemoryView.h
 * @brief Non-owning view types over contiguous memory with memory-domain metadata.
 */
#ifndef IVYMEMORYVIEW_H
#define IVYMEMORYVIEW_H


#include "std_ivy/memory/IvyAllocator.h"


namespace std_ivy{
  /**
  class memview:

  A memory view class that provides a view of an object in a different memory space.

  When the recursive flag is false, only the bytes in memory are copied. The contents of the object remain as in the original object.
  Otherwise, internal objects are copied as well.

  The memory view does not own the memory if the memory type of the original object is the same as the default execution memory type.
  In that case, the data pointer is simply set to the original pointer.

  If the memory types are different, a new memory is allocated in the default execution memory type, and the data is copied over.
  In that case, the memory view owns the memory, and it is freed upon destruction of the memory view object.
  */
  template<typename T, typename Allocator = allocator<T>> class memview{
  public:
    /** @brief Allocator type used for recursive/semantic copies. */
    typedef Allocator allocator_type;
    /** @brief Traits facade over allocator operations. */
    typedef allocator_traits<allocator_type> allocator_traits_t;
    /** @brief Element value type. */
    typedef typename allocator_traits_t::value_type value_type;
    /** @brief Element reference type. */
    typedef typename allocator_traits_t::reference reference;
    /** @brief Element pointer type. */
    typedef typename allocator_traits_t::pointer pointer;
    /** @brief Size type. */
    typedef typename allocator_traits_t::size_type size_type;

  protected:
    /** @brief Pointer to viewed data (owned or borrowed depending on transfer rules). */
    pointer data;
    /** @brief Logical size requested for view semantics. */
    size_type const s;
    /** @brief Physical allocation size when ownership transfer occurs. */
    size_type const n;
    /** @brief Optional stream pointer used for async-aware operations. */
    IvyGPUStream* stream;
    /** @brief Whether deep transfer of internal memory is requested. */
    bool recursive;
    /** @brief Whether this view owns and must release `data`. */
    bool do_own;

    /**
     * @brief Release owned backing memory according to recursive vs raw-byte semantics.
     * @note This routine mirrors constructor allocation/transfer strategy.
     */
    __HOST_DEVICE__ void destroy(){
      if (do_own && data){
        constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
        build_GPU_stream_reference_from_pointer(stream, ref_stream);
        if (!recursive) IvyMemoryHelpers::free_memory(data, n, def_mem_type, ref_stream);
        else{
          if (s==n) allocator_traits_t::destroy(data, n, def_mem_type, ref_stream);
          else{
            allocator_traits_t::destruct(data, s, def_mem_type, ref_stream);
            allocator_traits_t::deallocate(data, n, def_mem_type, ref_stream);
          }
        }
        destroy_GPU_stream_reference_from_pointer(stream);
      }
    }

  public:
    /**
     * @brief Construct a single-element view from a pointer source.
     * @param ptr Source pointer.
     * @param owning_mem_type_ Source memory-domain tag.
     * @param stream_ Optional stream used for transfer operations.
     * @param recursive_ If true, deep/internal-memory transfer is requested.
     * @param force_transfer_ Force copy even when source and execution domains match.
     */
    __HOST_DEVICE__ memview(pointer const& ptr, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_, bool force_transfer_=false) :
      data(nullptr), s(1), n(1), stream(stream_), recursive(recursive_)
    {
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_ || force_transfer_);
      if (!do_own){
        data = ptr;
        return;
      }
      build_GPU_stream_reference_from_pointer(stream, ref_stream);
      if (!recursive){
        // In the non-recursive case, only the bytes in memory are copied. The contents of the object remain as in the original object.
        // For that reason, the accompanying statement to free the data pointer would be simply IvyMemoryHelpers::free_memory.
        IvyMemoryHelpers::allocate_memory(data, n, def_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(data, ptr, s, def_mem_type, owning_mem_type_, ref_stream);
      }
      else{
        // In the recursive case, the object is copied as well.
        // The accompanying statement to free the data pointer would be allocator_traits_t::destroy
        // because new internal objects are created and need to be destroyed as well.
        allocator_traits_t::allocate(data, n, def_mem_type, ref_stream);
        allocator_traits_t::transfer(data, ptr, s, def_mem_type, owning_mem_type_, ref_stream);
      }
      destroy_GPU_stream_reference_from_pointer(stream);
    }
    /**
     * @brief Construct a single-element view from a reference source.
     * @param ref Source reference.
     * @param owning_mem_type_ Source memory-domain tag.
     * @param stream_ Optional stream used for transfer operations.
     * @param recursive_ If true, deep/internal-memory transfer is requested.
     * @param force_transfer_ Force copy even when source and execution domains match.
     */
    __HOST_DEVICE__ memview(reference ref, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_, bool force_transfer_=false) :
      data(nullptr), s(1), n(1), stream(stream_), recursive(recursive_)
    {
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_ || force_transfer_);
      if (!do_own){
        data = addressof(ref);
        return;
      }
      build_GPU_stream_reference_from_pointer(stream, ref_stream);
      if (!recursive){
        // In the non-recursive case, only the bytes in memory are copied. The contents of the object remain as in the original object.
        // For that reason, the accompanying statement to free the data pointer would be simply IvyMemoryHelpers::free_memory.
        IvyMemoryHelpers::allocate_memory(data, n, def_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(data, &ref, s, def_mem_type, def_mem_type, ref_stream);
      }
      else{
        // In the recursive case, the object is copied as well.
        // The accompanying statement to free the data pointer would be allocator_traits_t::destroy
        // because new internal objects are created and need to be destroyed as well.
        allocator_traits_t::allocate(data, n, def_mem_type, ref_stream);
        allocator_traits_t::transfer(data, &ref, s, def_mem_type, def_mem_type, ref_stream);
      }
      destroy_GPU_stream_reference_from_pointer(stream);
    }

    /**
     * @brief Construct an N-element view from pointer source.
     * @param ptr Source pointer.
     * @param n_ Number of elements.
     * @param owning_mem_type_ Source memory-domain tag.
     * @param stream_ Optional stream used for transfer operations.
     * @param recursive_ If true, deep/internal-memory transfer is requested.
     * @param force_transfer_ Force copy even when source and execution domains match.
     */
    __HOST_DEVICE__ memview(pointer const& ptr, size_type const& n_, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_, bool force_transfer_=false) :
      data(nullptr), s(n_), n(n_), stream(stream_), recursive(recursive_)
    {
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_ || force_transfer_);
      if (!do_own){
        data = ptr;
        return;
      }
      build_GPU_stream_reference_from_pointer(stream, ref_stream);
      if (!recursive){
        // In the non-recursive case, only the bytes in memory are copied. The contents of the object remain as in the original object.
        // For that reason, the accompanying statement to free the data pointer would be simply IvyMemoryHelpers::free_memory.
        IvyMemoryHelpers::allocate_memory(data, n, def_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(data, ptr, s, def_mem_type, owning_mem_type_, ref_stream);
      }
      else{
        // In the recursive case, the object is copied as well.
        // The accompanying statement to free the data pointer would be allocator_traits_t::destroy
        // because new internal objects are created and need to be destroyed as well.
        allocator_traits_t::allocate(data, n, def_mem_type, ref_stream);
        allocator_traits_t::transfer(data, ptr, s, def_mem_type, owning_mem_type_, ref_stream);
      }
      destroy_GPU_stream_reference_from_pointer(stream);
    }
    /**
     * @brief Construct a view with logical size `s_` and storage size `n_`.
     * @param ptr Source pointer.
     * @param s_ Logical number of initialized elements.
     * @param n_ Allocation size.
     * @param owning_mem_type_ Source memory-domain tag.
     * @param stream_ Optional stream used for transfer operations.
     * @param recursive_ If true, deep/internal-memory transfer is requested.
     * @param force_transfer_ Force copy even when source and execution domains match.
     */
    __HOST_DEVICE__ memview(pointer const& ptr, size_type const& s_, size_type const& n_, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_, bool force_transfer_=false) :
      data(nullptr), s(s_), n(n_), stream(stream_), recursive(recursive_)
    {
      assert(s<=n);
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_ || force_transfer_);
      if (!do_own){
        data = ptr;
        return;
      }
      build_GPU_stream_reference_from_pointer(stream, ref_stream);
      if (!recursive){
        // In the non-recursive case, only the bytes in memory are copied. The contents of the object remain as in the original object.
        // For that reason, the accompanying statement to free the data pointer would be simply IvyMemoryHelpers::free_memory.
        IvyMemoryHelpers::allocate_memory(data, n, def_mem_type, ref_stream);
        IvyMemoryHelpers::transfer_memory(data, ptr, s, def_mem_type, owning_mem_type_, ref_stream);
      }
      else{
        // In the recursive case, the object is copied as well.
        // The accompanying statement to free the data pointer would be allocator_traits_t::destroy
        // because new internal objects are created and need to be destroyed as well.
        allocator_traits_t::allocate(data, n, def_mem_type, ref_stream);
        allocator_traits_t::transfer(data, ptr, s, def_mem_type, owning_mem_type_, ref_stream);
      }
      destroy_GPU_stream_reference_from_pointer(stream);
    }

    memview(memview const&) = delete;
    /** @brief Move constructor transferring view ownership state. */
    __HOST_DEVICE__ memview(memview&& other) :
      data(std_util::move(other.data)),
      s(std_util::move(other.s)),
      n(std_util::move(other.n)),
      stream(std_util::move(other.stream)),
      recursive(std_util::move(other.recursive)),
      do_own(std_util::move(other.do_own))
    {
      other.data = nullptr;
      other.stream = nullptr;
      other.do_own = other.recursive = false;
    }
    memview& operator=(memview const&) = delete;
    /** @brief Move assignment releasing current ownership then stealing state. */
    __HOST_DEVICE__ memview& operator=(memview&& other){
      if (this != &other){
        this->destroy();

        data = std_util::move(other.data);
        s = std_util::move(other.s);
        n = std_util::move(other.n);
        stream = std_util::move(other.stream);
        recursive = std_util::move(other.recursive);
        do_own = std_util::move(other.do_own);

        other.data = nullptr;
        other.stream = nullptr;
        other.do_own = other.recursive = false;
      }
      return *this;
    }

    /** @brief Destructor releasing owned resources when applicable. */
    __HOST_DEVICE__ ~memview(){ destroy(); }

    /** @brief Get raw pointer to viewed data. */
    __HOST_DEVICE__ pointer const& get() const{ return data; }
    /** @brief Get allocation-size view extent. */
    __HOST_DEVICE__ size_type const& size() const{ return n; }
    /** @brief Indexed element access. */
    __HOST_DEVICE__ reference operator[](size_type const& i) const{ return data[i]; }
    /** @brief Implicit conversion to raw pointer. */
    __HOST_DEVICE__ operator pointer() const{ return data; }
    /** @brief Dereference first element. */
    __HOST_DEVICE__ reference operator*() const{ return *data; }
    /** @brief Member access through underlying pointer. */
    __HOST_DEVICE__ pointer operator->() const{ return data; }
  };
}


#endif
