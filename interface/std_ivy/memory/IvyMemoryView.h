#ifndef IVYMEMORYVIEW_H
#define IVYMEMORYVIEW_H


#include "std_ivy/memory/IvyAllocator.h"


namespace std_ivy{
  /*
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
    typedef Allocator allocator_type;
    typedef allocator_traits<allocator_type> allocator_traits_t;
    typedef typename allocator_traits_t::value_type value_type;
    typedef typename allocator_traits_t::reference reference;
    typedef typename allocator_traits_t::pointer pointer;
    typedef typename allocator_traits_t::size_type size_type;

  protected:
    pointer data;
    size_type const s;
    size_type const n;
    IvyGPUStream* stream;
    bool recursive;
    bool do_own;

  public:
    __HOST_DEVICE__ memview(pointer const& ptr, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_) :
      data(nullptr), s(1), n(1), stream(stream_), recursive(recursive_)
    {
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_);
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
    __HOST_DEVICE__ memview(reference ref, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_) :
      data(nullptr), s(1), n(1), stream(stream_), recursive(recursive_)
    {
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_);
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

    __HOST_DEVICE__ memview(pointer const& ptr, size_type const& n_, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_) :
      data(nullptr), s(n_), n(n_), stream(stream_), recursive(recursive_)
    {
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_);
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
    __HOST_DEVICE__ memview(pointer const& ptr, size_type const& s_, size_type const& n_, IvyMemoryType owning_mem_type_, IvyGPUStream* stream_, bool recursive_) :
      data(nullptr), s(s_), n(n_), stream(stream_), recursive(recursive_)
    {
      assert(s<=n);
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      do_own = (def_mem_type != owning_mem_type_);
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
    memview(memview&&) = delete;
    memview& operator=(memview const&) = delete;
    memview& operator=(memview&&) = delete;

    __HOST_DEVICE__ ~memview(){
      if (do_own){
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

    __HOST_DEVICE__ pointer const& get() const{ return data; }
    __HOST_DEVICE__ size_type const& size() const{ return n; }
    __HOST_DEVICE__ reference operator[](size_type const& i) const{ return data[i]; }
    __HOST_DEVICE__ operator pointer() const{ return data; }
    __HOST_DEVICE__ reference operator*() const{ return *data; }
    __HOST_DEVICE__ pointer operator->() const{ return data; }
  };
}


#endif
