#ifndef IVYVECTORIMPL_HH
#define IVYVECTORIMPL_HH


#include "config/IvyCompilerConfig.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyIterator.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, typename Allocator=std_mem::allocator<T>>
  class IvyVector{
  public:
    typedef T value_type;
    typedef Allocator allocator_type;
    typedef std_mem::allocator_traits<Allocator> allocator_type_traits;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef IvyVectorIteratorBuilder<value_type> iterator_builder_t;
    typedef IvyVectorIteratorBuilder<value_type const> const_iterator_builder_t;
    typedef std_mem::allocator_traits<allocator_type>::pointer pointer;
    typedef std_mem::allocator_traits<allocator_type>::const_pointer const_pointer;
    typedef std_mem::allocator_traits<allocator_type>::size_type size_type;
    typedef std_mem::allocator_traits<allocator_type>::difference_type difference_type;
    typedef iterator_builder_t::iterator_type iterator;
    typedef const_iterator_builder_t::iterator_type const_iterator;
    typedef std_iter::reverse_iterator<iterator> reverse_iterator;
    typedef std_iter::reverse_iterator<const_iterator> const_reverse_iterator;

  protected:
    std_ivy::unique_ptr<value_type> _data;
    iterator_builder_t _iterator_builder;
    const_iterator_builder_t _const_iterator_builder;

  public:
    __CUDA_HOST_DEVICE__ IvyVector();
    __CUDA_HOST_DEVICE__ IvyVector(IvyVector const& v);
    __CUDA_HOST_DEVICE__ IvyVector(IvyVector&& v);
    __CUDA_HOST_DEVICE__ IvyVector(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ IvyVector(size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    template<typename InputIterator> __CUDA_HOST_DEVICE__ IvyVector(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ IvyVector(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ ~IvyVector();

    __CUDA_HOST_DEVICE__ IvyVector& operator=(IvyVector const& v);
    __CUDA_HOST_DEVICE__ IvyVector& operator=(IvyVector&& v);

    __CUDA_HOST_DEVICE__ void assign(size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    template<typename InputIterator> __CUDA_HOST_DEVICE__ void assign(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ void assign(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);

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

    __CUDA_HOST_DEVICE__ bool empty() const;
    __CUDA_HOST_DEVICE__ size_type size() const;
    __CUDA_HOST_DEVICE__ size_type max_size() const;
    __CUDA_HOST_DEVICE__ void reserve(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ size_type capacity() const;
    __CUDA_HOST_DEVICE__ void shrink_to_fit();

    __CUDA_HOST_DEVICE__ void clear();
    __CUDA_HOST_DEVICE__ iterator insert(const_iterator pos, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ iterator insert(const_iterator pos, size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    template<typename InputIterator> __CUDA_HOST_DEVICE__ iterator insert(const_iterator pos, InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ iterator insert(const_iterator pos, std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);

    __CUDA_HOST_DEVICE__ iterator erase(const_iterator pos);
    __CUDA_HOST_DEVICE__ iterator erase(const_iterator first, const_iterator last);

    __CUDA_HOST_DEVICE__ void push_back(value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ void pop_back();

    __CUDA_HOST_DEVICE__ iterator emplace(const_iterator pos, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ void emplace_back(value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);

    __CUDA_HOST_DEVICE__ void resize(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);
    __CUDA_HOST_DEVICE__ void resize(size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr);

    __CUDA_HOST_DEVICE__ void swap(IvyVector& v);
  };

  template<typename T, typename Allocator=std_mem::allocator<T>> using vector = IvyVector<T, Allocator>;


}

#endif


#endif
