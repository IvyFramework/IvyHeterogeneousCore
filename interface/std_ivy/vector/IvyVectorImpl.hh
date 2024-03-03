#ifndef IVYVECTORIMPL_HH
#define IVYVECTORIMPL_HH


#include "std_ivy/iterator/IvyContiguousIteratorBuilder.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, typename Allocator=std_mem::allocator<T>> class IvyVector;
  template<typename T, typename Allocator> class transfer_memory_primitive<IvyVector<T, Allocator>> : public transfer_memory_primitive_with_internal_memory<IvyVector<T, Allocator>>{};

  template<typename T, typename Allocator> class IvyVector{
  public:
    typedef T value_type;
    typedef Allocator allocator_type;
    typedef std_mem::unique_ptr<value_type> data_container;
    typedef std_mem::allocator_traits<Allocator> allocator_type_traits;
    typedef std_mem::allocator<data_container> allocator_data_container;
    typedef std_mem::allocator_traits<allocator_data_container> allocator_data_container_traits;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef IvyContiguousIteratorBuilder<value_type> iterator_builder_t;
    typedef IvyContiguousIteratorBuilder<value_type const> const_iterator_builder_t;
    typedef std_mem::allocator<iterator_builder_t> allocator_iterator_builder_t;
    typedef std_mem::allocator<const_iterator_builder_t> allocator_const_iterator_builder_t;
    typedef std_mem::allocator_traits<allocator_iterator_builder_t> allocator_iterator_builder_traits_t;
    typedef std_mem::allocator_traits<allocator_const_iterator_builder_t> allocator_const_iterator_builder_traits_t;
    typedef std_mem::allocator_traits<allocator_type>::pointer pointer;
    typedef std_mem::allocator_traits<allocator_type>::const_pointer const_pointer;
    typedef std_mem::allocator_traits<allocator_type>::size_type size_type;
    typedef std_mem::allocator_traits<allocator_type>::difference_type difference_type;
    typedef iterator_builder_t::iterator_type iterator;
    typedef const_iterator_builder_t::iterator_type const_iterator;
    typedef std_ivy::reverse_iterator<iterator> reverse_iterator;
    typedef std_ivy::reverse_iterator<const_iterator> const_reverse_iterator;

    friend class kernel_generic_transfer_internal_memory<IvyVector<T, Allocator>>;

  protected:
    data_container _data;
    iterator_builder_t _iterator_builder;
    const_iterator_builder_t _const_iterator_builder;

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old);

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void destroy_iterator_builders();
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void reset_iterator_builders();

  public:
    __CUDA_HOST_DEVICE__ IvyVector();
    __CUDA_HOST_DEVICE__ IvyVector(IvyVector const& v);
    __CUDA_HOST_DEVICE__ IvyVector(IvyVector&& v);
    template<typename... Args> __CUDA_HOST_DEVICE__ IvyVector(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    template<typename InputIterator> __CUDA_HOST_DEVICE__ IvyVector(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    __CUDA_HOST_DEVICE__ IvyVector(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);
    __CUDA_HOST_DEVICE__ ~IvyVector();

    __CUDA_HOST_DEVICE__ IvyVector& operator=(IvyVector const& v);
    __CUDA_HOST_DEVICE__ IvyVector& operator=(IvyVector&& v);

    template<typename... Args> __CUDA_HOST_DEVICE__ void assign(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    template<typename InputIterator> __CUDA_HOST_DEVICE__ void assign(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    __CUDA_HOST_DEVICE__ void assign(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

    __CUDA_HOST_DEVICE__ reference at(size_type n);
    __CUDA_HOST_DEVICE__ const_reference at(size_type n) const;
    __CUDA_HOST_DEVICE__ reference operator[](size_type n);
    __CUDA_HOST_DEVICE__ const_reference operator[](size_type n) const;
    __CUDA_HOST_DEVICE__ reference front();
    __CUDA_HOST_DEVICE__ const_reference front() const;
    __CUDA_HOST_DEVICE__ reference back();
    __CUDA_HOST_DEVICE__ const_reference back() const;
    __CUDA_HOST_DEVICE__ pointer data();
    __CUDA_HOST_DEVICE__ const_pointer data() const;

    __CUDA_HOST_DEVICE__ iterator begin();
    __CUDA_HOST_DEVICE__ const_iterator begin() const;
    __CUDA_HOST_DEVICE__ const_iterator cbegin() const;
    __CUDA_HOST_DEVICE__ iterator end();
    __CUDA_HOST_DEVICE__ const_iterator end() const;
    __CUDA_HOST_DEVICE__ const_iterator cend() const;
    __CUDA_HOST_DEVICE__ reverse_iterator rbegin();
    __CUDA_HOST_DEVICE__ const_reverse_iterator rbegin() const;
    __CUDA_HOST_DEVICE__ const_reverse_iterator crbegin() const;
    __CUDA_HOST_DEVICE__ reverse_iterator rend();
    __CUDA_HOST_DEVICE__ const_reverse_iterator rend() const;
    __CUDA_HOST_DEVICE__ const_reverse_iterator crend() const;

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool empty() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type size() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ constexpr size_type max_size() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type capacity() const;

    __CUDA_HOST_DEVICE__ IvyMemoryType get_memory_type() const;
    __CUDA_HOST_DEVICE__ IvyGPUStream* gpu_stream() const;

    __CUDA_HOST_DEVICE__ void reserve(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    __CUDA_HOST_DEVICE__ void reserve(size_type n);
    __CUDA_HOST_DEVICE__ void shrink_to_fit();

    __CUDA_HOST_DEVICE__ void clear();

    template<typename PosIterator, typename... Args> __CUDA_HOST_DEVICE__ iterator insert(PosIterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    template<typename PosIterator, typename... Args> __CUDA_HOST_DEVICE__ iterator insert(PosIterator pos, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    template<typename PosIterator, typename InputIterator> __CUDA_HOST_DEVICE__ iterator insert(PosIterator pos, InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    template<typename PosIterator> __CUDA_HOST_DEVICE__ iterator insert(PosIterator pos, std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

    template<typename PosIterator> __CUDA_HOST_DEVICE__ iterator erase(PosIterator pos);
    template<typename PosIterator> __CUDA_HOST_DEVICE__ iterator erase(PosIterator first, PosIterator last);

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void push_back(value_type const& val);
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void push_back(IvyMemoryType mem_type, IvyGPUStream* stream, value_type const& val);
    __CUDA_HOST_DEVICE__ void pop_back();

    template<typename PosIterator, typename... Args> __CUDA_HOST_DEVICE__ iterator emplace(PosIterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    template<typename... Args> __CUDA_HOST_DEVICE__ void emplace_back(Args&&... args);
    template<typename... Args> __CUDA_HOST_DEVICE__ void emplace_back(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

    template<typename... Args> __CUDA_HOST_DEVICE__ void resize(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

    __CUDA_HOST_DEVICE__ void swap(IvyVector& v);

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ data_container const& get_data_container() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ iterator_builder_t const& get_iterator_builder() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ const_iterator_builder_t const& get_const_iterator_builder() const;
  };
  template<typename T, typename Allocator=std_mem::allocator<T>> using vector = IvyVector<T, Allocator>;
}
namespace std_util{
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void swap(std_ivy::IvyVector<T, Allocator>& a, std_ivy::IvyVector<T, Allocator>& b);
}

#endif


#endif
