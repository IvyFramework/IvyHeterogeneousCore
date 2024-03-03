#ifndef IVYUNORDEREDMAPIMPL_HH
#define IVYUNORDEREDMAPIMPL_HH


#include "std_ivy/iterator/IvyBucketedIteratorBuilder.h"
#include "std_ivy/unordered_map/IvyUnorderedMapKeyEval.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyUtility.h"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<
    typename Key,
    typename T,
    typename Hash = std_ivy::hash<Key const>,
    typename KeyEqual = std_ivy::IvyKeyEqualEvalDefault<Key const>,
    typename HashEqual = std_ivy::IvyHashEqualEvalDefault<typename Hash::result_type>,
    typename Allocator = std_mem::allocator<std_util::pair<Key const, T>>
  > class IvyUnorderedMap;

  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator>
  class transfer_memory_primitive<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>> : public transfer_memory_primitive_with_internal_memory<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>>{};

  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator> class IvyUnorderedMap{
  public:
    typedef Key const key_type;
    typedef T mapped_type;
    typedef std_ivy::hash<key_type> hasher;
    typedef typename hasher::result_type hash_result_type;
    typedef KeyEqual key_equal;
    typedef HashEqual hash_equal;
    typedef std_util::pair<key_type, mapped_type> value_type;
    typedef Allocator allocator_type;
    typedef std_mem::unique_ptr<value_type> bucket_data_type;
    typedef std_util::pair<hash_result_type, bucket_data_type> bucket_element;
    typedef std_mem::unique_ptr<bucket_element> data_container;
    typedef std_mem::allocator_traits<Allocator> allocator_type_traits;
    typedef std_mem::allocator<data_container> allocator_data_container;
    typedef std_mem::allocator_traits<allocator_data_container> allocator_data_container_traits;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef IvyBucketedIteratorBuilder<key_type, mapped_type, hasher> iterator_builder_t;
    typedef std_mem::allocator<iterator_builder_t> allocator_iterator_builder_t;
    typedef std_mem::allocator_traits<allocator_iterator_builder_t> allocator_iterator_builder_traits_t;
    typedef std_mem::allocator_traits<allocator_type>::pointer pointer;
    typedef std_mem::allocator_traits<allocator_type>::const_pointer const_pointer;
    typedef std_mem::allocator_traits<allocator_type>::size_type size_type;
    typedef std_mem::allocator_traits<allocator_type>::difference_type difference_type;
    typedef iterator_builder_t::iterator_type iterator;
    typedef iterator_builder_t::const_iterator_type const_iterator;
    typedef std_ivy::reverse_iterator<iterator> reverse_iterator;
    typedef std_ivy::reverse_iterator<const_iterator> const_reverse_iterator;

    friend class kernel_generic_transfer_internal_memory<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>>;

  protected:
    IvyMemoryType const progenitor_mem_type;
    data_container _data;
    iterator_builder_t _iterator_builder;

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old);

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void destroy_iterator_builder();
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void reset_iterator_builder();

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool check_write_access() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ bool check_write_access(IvyMemoryType const& mem_type) const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void check_write_access_or_die() const;
    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void check_write_access_or_die(IvyMemoryType const& mem_type) const;

    __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ size_type get_predicted_bucket_count() const;

    __CUDA_HOST_DEVICE__ data_container get_rehashed_data(size_type new_n_buckets) const;

    template<typename... Args> __CUDA_HOST_DEVICE__ void insert_impl(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);
    __CUDA_HOST_DEVICE__ void erase_impl(Key const& key, size_type& n_erased);

    /*
    IvyUnorderedMap::calculate_data_size_capacity: Brute-force calculation of the actual size and capacity of the data container.
    */
    __CUDA_HOST_DEVICE__ void calculate_data_size_capacity(size_type& n_size, size_type& n_capacity) const;

    __CUDA_HOST_DEVICE__ iterator find_iterator(Key const& key) const;
    __CUDA_HOST_DEVICE__ const_iterator find_const_iterator(Key const& key) const;

  public:
    __CUDA_HOST_DEVICE__ IvyUnorderedMap();
    __CUDA_HOST_DEVICE__ IvyUnorderedMap(IvyUnorderedMap const& v);
    __CUDA_HOST_DEVICE__ IvyUnorderedMap(IvyUnorderedMap&& v);
    __CUDA_HOST_DEVICE__ ~IvyUnorderedMap();

    __CUDA_HOST_DEVICE__ IvyUnorderedMap& operator=(IvyUnorderedMap const& v);
    __CUDA_HOST_DEVICE__ IvyUnorderedMap& operator=(IvyUnorderedMap&& v);

    __CUDA_HOST_DEVICE__ void swap(IvyUnorderedMap& v);

    __CUDA_HOST_DEVICE__ reference front();
    __CUDA_HOST_DEVICE__ const_reference front() const;
    __CUDA_HOST_DEVICE__ reference back();
    __CUDA_HOST_DEVICE__ const_reference back() const;

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

    __CUDA_HOST_DEVICE__ void clear();

    template<typename... Args> __CUDA_HOST_DEVICE__ iterator insert(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);
    template<typename InputIterator> __CUDA_HOST_DEVICE__ iterator insert(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    __CUDA_HOST_DEVICE__ iterator insert(std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

    template<typename... Args> __INLINE_FCN_FORCE__ __CUDA_HOST_DEVICE__ iterator emplace(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);

    __CUDA_HOST_DEVICE__ size_type erase(Key const& key);
    template<typename PosIterator> __CUDA_HOST_DEVICE__ size_type erase(PosIterator pos);
    template<typename PosIterator> __CUDA_HOST_DEVICE__ size_type erase(PosIterator first, PosIterator last);

    __CUDA_HOST_DEVICE__ size_type bucket_count() const;
    __CUDA_HOST_DEVICE__ size_type bucket_capacity() const;
    __CUDA_HOST_DEVICE__ constexpr size_type max_bucket_count() const;

    __CUDA_HOST_DEVICE__ void rehash(size_type new_n_buckets);

    __CUDA_HOST_DEVICE__ mapped_type const& operator[](Key const& key) const;
    __CUDA_HOST_DEVICE__ mapped_type& operator[](Key const& key);
    template<typename... Args> __CUDA_HOST_DEVICE__ mapped_type& operator()(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);
    template<typename... Args> __CUDA_HOST_DEVICE__ mapped_type& operator()(Key const& key, Args&&... args);

    __CUDA_HOST_DEVICE__ iterator find(Key const& key);
    __CUDA_HOST_DEVICE__ const_iterator find(Key const& key) const;
  };
  template<
    typename Key,
    typename T,
    typename Hash = std_ivy::hash<Key const>,
    typename KeyEqual = std_ivy::IvyKeyEqualEvalDefault<Key const>,
    typename HashEqual = std_ivy::IvyHashEqualEvalDefault<typename Hash::result_type>,
    typename Allocator = std_mem::allocator<std_util::pair<Key const, T>>
  > using unordered_map = IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>;

  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator>
  __CUDA_HOST_DEVICE__ void swap(IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>& a, IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>& b);
}

#endif


#endif
