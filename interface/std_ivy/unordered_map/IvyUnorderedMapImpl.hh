/**
 * @file IvyUnorderedMapImpl.hh
 * @brief Declarations for std_ivy unordered_map internals and public API.
 */
#ifndef IVYUNORDEREDMAPIMPL_HH
#define IVYUNORDEREDMAPIMPL_HH


#include "std_ivy/iterator/IvyBucketedIteratorBuilder.h"
#include "std_ivy/unordered_map/IvyUnorderedMapKeyEval.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyUtility.h"
#include "IvyPrintout.h"


namespace std_ivy{
  /**
   * @brief Forward declaration of IvyUnorderedMap.
   */
  template<
    typename Key,
    typename T,
    typename Hash = std_ivy::hash<Key const>,
    typename KeyEqual = std_ivy::IvyKeyEqualEvalDefault<Key const>,
    typename HashEqual = std_ivy::IvyHashEqualEvalDefault<typename Hash::result_type>,
    typename Allocator = std_mem::allocator<std_util::pair<Key const, T>>
  > class IvyUnorderedMap;

  /** @brief Transfer-memory primitive specialization for IvyUnorderedMap. */
  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator>
  class transfer_memory_primitive<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>> : public transfer_memory_primitive_with_internal_memory<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>>{};

  /**
   * @brief Hash-map container with customizable key/hash equality evaluation and host/device-aware storage.
   */
  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator> class IvyUnorderedMap{
  public:
    /** @brief Key type. */
    typedef Key const key_type;
    /** @brief Mapped value type. */
    typedef T mapped_type;
    /** @brief Hash functor type. */
    typedef std_ivy::hash<key_type> hasher;
    /** @brief Hash-result type. */
    typedef typename hasher::result_type hash_result_type;
    /** @brief Key equality policy type. */
    typedef KeyEqual key_equal;
    /** @brief Hash-equality (bucket comparison) policy type. */
    typedef HashEqual hash_equal;

    /** @brief Stored key/value pair type. */
    typedef std_util::pair<key_type, mapped_type> value_type;
    /** @brief Allocator type for stored values. */
    typedef Allocator allocator_type;
    /** @brief Allocator traits for @ref allocator_type. */
    typedef std_mem::allocator_traits<Allocator> allocator_traits;

    /** @brief Bucket-local contiguous storage type. */
    typedef std_mem::unique_ptr<value_type> bucket_data_type;
    /** @brief Bucket node type: hash plus bucket storage. */
    typedef std_util::pair<hash_result_type, bucket_data_type> bucket_element;
    /** @brief Owning storage over all bucket nodes. */
    typedef std_mem::unique_ptr<bucket_element> data_container;
    /** @brief Allocator type for bucket elements. */
    typedef typename data_container::element_allocator_type allocator_bucket_element;
    /** @brief Allocator traits for @ref allocator_bucket_element. */
    typedef typename data_container::element_allocator_traits allocator_bucket_element_traits;

    /** @brief Allocator for @ref data_container wrapper objects. */
    typedef std_mem::allocator<data_container> allocator_data_container;
    /** @brief Allocator traits for @ref allocator_data_container. */
    typedef std_mem::allocator_traits<allocator_data_container> allocator_data_container_traits;

    /** @brief Mutable key/value reference type. */
    typedef value_type& reference;
    /** @brief Const key/value reference type. */
    typedef value_type const& const_reference;
    /** @brief Iterator builder over bucketed storage. */
    typedef IvyBucketedIteratorBuilder<key_type, mapped_type, hasher> iterator_builder_t;
    /** @brief Allocator for iterator-builder state. */
    typedef std_mem::allocator<iterator_builder_t> allocator_iterator_builder_t;
    /** @brief Allocator traits for @ref allocator_iterator_builder_t. */
    typedef std_mem::allocator_traits<allocator_iterator_builder_t> allocator_iterator_builder_traits_t;
    /** @brief Mutable pointer type. */
    typedef std_mem::allocator_traits<allocator_type>::pointer pointer;
    /** @brief Const pointer type. */
    typedef std_mem::allocator_traits<allocator_type>::const_pointer const_pointer;
    /** @brief Unsigned size/index type. */
    typedef std_mem::allocator_traits<allocator_type>::size_type size_type;
    /** @brief Signed difference type. */
    typedef std_mem::allocator_traits<allocator_type>::difference_type difference_type;
    /** @brief Mutable forward iterator type. */
    typedef iterator_builder_t::iterator_type iterator;
    /** @brief Const forward iterator type. */
    typedef iterator_builder_t::const_iterator_type const_iterator;
    /** @brief Mutable reverse iterator type. */
    typedef std_ivy::reverse_iterator<iterator> reverse_iterator;
    /** @brief Const reverse iterator type. */
    typedef std_ivy::reverse_iterator<const_iterator> const_reverse_iterator;

    friend class kernel_generic_transfer_internal_memory<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>>;

  protected:
    /** @brief Bucket data container. */
    data_container _data;
    /** @brief Iterator builder over bucketed data. */
    iterator_builder_t _iterator_builder;

    /** @brief Transfer internal memory for container and iterator state. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old);

    /** @brief Destroy iterator builder resources. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void destroy_iterator_builder();
    /** @brief Rebuild iterator builder state from current data. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void reset_iterator_builder();

    /** @brief Predict bucket count from current storage state. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type get_predicted_bucket_count() const;

    /** @brief Build a rehashed data container with a new bucket count. */
    __HOST_DEVICE__ data_container get_rehashed_data(size_type new_n_buckets) const;

    /** @brief Insert implementation used by overload set. */
    template<typename... Args> __HOST_DEVICE__ void insert_impl(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);
    /** @brief Erase implementation used by overload set. */
    __HOST_DEVICE__ void erase_impl(Key const& key, size_type& n_erased);

    /**
    IvyUnorderedMap::calculate_data_size_capacity: Brute-force calculation of the actual size and capacity of the data container.
    */
    __HOST_DEVICE__ void calculate_data_size_capacity(size_type& n_size, size_type& n_capacity) const;

    /** @brief Locate mutable iterator for a key. */
    __HOST_DEVICE__ iterator find_iterator(Key const& key) const;
    /** @brief Locate const iterator for a key. */
    __HOST_DEVICE__ const_iterator find_const_iterator(Key const& key) const;

  public:
    /** @brief Default constructor. */
    __HOST_DEVICE__ IvyUnorderedMap();
    /** @brief Copy constructor. */
    __HOST_DEVICE__ IvyUnorderedMap(IvyUnorderedMap const& v);
    /** @brief Move constructor. */
    __HOST_DEVICE__ IvyUnorderedMap(IvyUnorderedMap&& v);
    /** @brief Destructor. */
    __HOST_DEVICE__ ~IvyUnorderedMap();

    /** @brief Copy assignment operator. */
    __HOST_DEVICE__ IvyUnorderedMap& operator=(IvyUnorderedMap const& v);
    /** @brief Move assignment operator. */
    __HOST_DEVICE__ IvyUnorderedMap& operator=(IvyUnorderedMap&& v);

    /** @brief Swap all state with another map. */
    __HOST_DEVICE__ void swap(IvyUnorderedMap& v);

    /** @brief Access first valid key/value pair. */
    __HOST_DEVICE__ reference front();
    /** @brief Access first valid key/value pair (const overload). */
    __HOST_DEVICE__ const_reference front() const;
    /** @brief Access last valid key/value pair. */
    __HOST_DEVICE__ reference back();
    /** @brief Access last valid key/value pair (const overload). */
    __HOST_DEVICE__ const_reference back() const;

    /** @brief Return mutable iterator to first valid element. */
    __HOST_DEVICE__ iterator begin();
    /** @brief Return const iterator to first valid element. */
    __HOST_DEVICE__ const_iterator begin() const;
    /** @brief Return const iterator to first valid element. */
    __HOST_DEVICE__ const_iterator cbegin() const;
    /** @brief Return mutable iterator to end sentinel. */
    __HOST_DEVICE__ iterator end();
    /** @brief Return const iterator to end sentinel. */
    __HOST_DEVICE__ const_iterator end() const;
    /** @brief Return const iterator to end sentinel. */
    __HOST_DEVICE__ const_iterator cend() const;
    /** @brief Return mutable reverse iterator to reverse beginning. */
    __HOST_DEVICE__ reverse_iterator rbegin();
    /** @brief Return const reverse iterator to reverse beginning. */
    __HOST_DEVICE__ const_reverse_iterator rbegin() const;
    /** @brief Return const reverse iterator to reverse beginning. */
    __HOST_DEVICE__ const_reverse_iterator crbegin() const;
    /** @brief Return mutable reverse iterator to reverse end sentinel. */
    __HOST_DEVICE__ reverse_iterator rend();
    /** @brief Return const reverse iterator to reverse end sentinel. */
    __HOST_DEVICE__ const_reverse_iterator rend() const;
    /** @brief Return const reverse iterator to reverse end sentinel. */
    __HOST_DEVICE__ const_reverse_iterator crend() const;

    /** @brief Check if the map has no stored elements. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool empty() const;
    /** @brief Number of stored key/value pairs. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type size() const;
    /** @brief Maximum representable number of key/value pairs. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr size_type max_size() const;
    /** @brief Current storage capacity in value slots. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type capacity() const;

    /** @brief Get const reference to the underlying bucket container. */
    __HOST_DEVICE__ data_container const& get_data_container() const;
    /** @brief Get memory type used by the bucket container. */
    __HOST_DEVICE__ IvyMemoryType get_memory_type() const;
    /** @brief Get associated GPU stream pointer. */
    __HOST_DEVICE__ IvyGPUStream* gpu_stream() const;

    /** @brief Remove all elements. */
    __HOST_DEVICE__ void clear();

    /**
     * @brief Insert or construct a value for @p key.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param key Key to insert.
     * @param args Constructor arguments forwarded to mapped value construction.
     * @return Iterator to inserted or existing element for @p key.
     */
    template<typename... Args> __HOST_DEVICE__ iterator insert(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);
    /**
     * @brief Insert elements from range @p [first,last).
     * @tparam InputIterator Input iterator type.
     * @param first Range start.
     * @param last Range end.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @return Iterator to first inserted element when insertion occurs, or end iterator if no insert occurred.
     */
    template<typename InputIterator> __HOST_DEVICE__ iterator insert(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    /**
     * @brief Insert elements from initializer list.
     * @param ilist Source list of key/value pairs.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @return Iterator to first inserted element when insertion occurs, or end iterator if no insert occurred.
     */
    __HOST_DEVICE__ iterator insert(std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

    /**
     * @brief In-place insertion forwarding to the same insertion semantics as @ref insert.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param key Key to insert.
     * @param args Constructor arguments forwarded to mapped value construction.
     * @return Iterator to inserted or existing element.
     */
    template<typename... Args> __INLINE_FCN_FORCE__ __HOST_DEVICE__ iterator emplace(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);

    /**
     * @brief Erase element by key.
     * @param key Key to erase.
     * @return Number of erased elements.
     */
    __HOST_DEVICE__ size_type erase(Key const& key);
    /**
     * @brief Erase element pointed to by iterator.
     * @tparam PosIterator Iterator type.
     * @param pos Iterator to erase.
     * @return Number of erased elements.
     */
    template<typename PosIterator> __HOST_DEVICE__ size_type erase(PosIterator pos);
    /**
     * @brief Erase range @p [first,last).
     * @tparam PosIterator Iterator type.
     * @param first Range start.
     * @param last Range end.
     * @return Number of erased elements.
     */
    template<typename PosIterator> __HOST_DEVICE__ size_type erase(PosIterator first, PosIterator last);

    /** @brief Number of active buckets. */
    __HOST_DEVICE__ size_type bucket_count() const;
    /** @brief Bucket storage capacity. */
    __HOST_DEVICE__ size_type bucket_capacity() const;
    /** @brief Maximum representable bucket count. */
    __HOST_DEVICE__ constexpr size_type max_bucket_count() const;

    /** @brief Rehash container to a new number of buckets. */
    __HOST_DEVICE__ void rehash(size_type new_n_buckets);

    /**
     * @brief Access mapped value by key (const overload).
     * @param key Lookup key.
     * @return Const reference to mapped value.
     */
    __HOST_DEVICE__ mapped_type const& operator[](Key const& key) const;
    /**
     * @brief Access mapped value by key (mutable overload).
     * @param key Lookup key.
     * @return Mutable reference to mapped value.
     */
    __HOST_DEVICE__ mapped_type& operator[](Key const& key);
    /**
     * @brief Get or create mapped value with explicit memory target/stream and forwarded constructor args.
     * @param mem_type Target memory type when insertion is required.
     * @param stream Optional GPU stream.
     * @param key Lookup/insert key.
     * @param args Constructor arguments forwarded to mapped value construction.
     * @return Mutable reference to mapped value for @p key.
     */
    template<typename... Args> __HOST_DEVICE__ mapped_type& operator()(IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args);
    /**
     * @brief Get or create mapped value in default execution memory with forwarded constructor args.
     * @param key Lookup/insert key.
     * @param args Constructor arguments forwarded to mapped value construction.
     * @return Mutable reference to mapped value for @p key.
     */
    template<typename... Args> __HOST_DEVICE__ mapped_type& operator()(Key const& key, Args&&... args);

    /** @brief Find element by key (mutable iterator). */
    __HOST_DEVICE__ iterator find(Key const& key);
    /** @brief Find element by key (const iterator). */
    __HOST_DEVICE__ const_iterator find(Key const& key) const;
  };
  /** @brief STL-style alias for IvyUnorderedMap. */
  template<
    typename Key,
    typename T,
    typename Hash = std_ivy::hash<Key const>,
    typename KeyEqual = std_ivy::IvyKeyEqualEvalDefault<Key const>,
    typename HashEqual = std_ivy::IvyHashEqualEvalDefault<typename Hash::result_type>,
    typename Allocator = std_mem::allocator<std_util::pair<Key const, T>>
  > using unordered_map = IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>;

  /**
  Specialization of the value printout routine
  */
  /** @brief value_printout specialization declaration for IvyUnorderedMap. */
  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator>
  struct value_printout<IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>>;
}
namespace std_util{
  /** @brief swap overload for IvyUnorderedMap. */
  template<typename Key, typename T, typename Hash, typename KeyEqual, typename HashEqual, typename Allocator>
  __HOST_DEVICE__ void swap(std_ivy::IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>& a, std_ivy::IvyUnorderedMap<Key, T, Hash, KeyEqual, HashEqual, Allocator>& b);
}


#endif
