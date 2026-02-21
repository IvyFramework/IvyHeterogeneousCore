/**
 * @file IvyVectorImpl.hh
 * @brief Declarations for the std_ivy vector container and related iterators.
 */
#ifndef IVYVECTORIMPL_HH
#define IVYVECTORIMPL_HH


#include "std_ivy/iterator/IvyContiguousIteratorBuilder.h"
#include "IvyPrintout.h"


namespace std_ivy{
  /** @brief Forward declaration of IvyVector. */
  template<typename T, typename Allocator=std_mem::allocator<T>> class IvyVector;
  /** @brief Transfer-memory primitive specialization for IvyVector. */
  template<typename T, typename Allocator> class transfer_memory_primitive<IvyVector<T, Allocator>> : public transfer_memory_primitive_with_internal_memory<IvyVector<T, Allocator>>{};

  /**
   * @brief Vector container with host/device-aware memory ownership and iterator builders.
   * @tparam T Element type.
   * @tparam Allocator Allocator type for elements.
   */
  template<typename T, typename Allocator> class IvyVector{
  public:
    /** @brief Element type. */
    typedef T value_type;
    /** @brief Allocator type. */
    typedef Allocator allocator_type;
    /** @brief Owning storage container for contiguous data. */
    typedef std_mem::unique_ptr<value_type> data_container;
    /** @brief Allocator traits for @ref allocator_type. */
    typedef std_mem::allocator_traits<Allocator> allocator_type_traits;
    /** @brief Allocator for the internal data container wrapper. */
    typedef std_mem::allocator<data_container> allocator_data_container;
    /** @brief Traits for @ref allocator_data_container. */
    typedef std_mem::allocator_traits<allocator_data_container> allocator_data_container_traits;
    /** @brief Mutable element reference type. */
    typedef value_type& reference;
    /** @brief Const element reference type. */
    typedef value_type const& const_reference;
    /** @brief Iterator builder for mutable iterators. */
    typedef IvyContiguousIteratorBuilder<value_type> iterator_builder_t;
    /** @brief Iterator builder for const iterators. */
    typedef IvyContiguousIteratorBuilder<value_type const> const_iterator_builder_t;
    /** @brief Allocator for mutable iterator builder state. */
    typedef std_mem::allocator<iterator_builder_t> allocator_iterator_builder_t;
    /** @brief Allocator for const iterator builder state. */
    typedef std_mem::allocator<const_iterator_builder_t> allocator_const_iterator_builder_t;
    /** @brief Traits for @ref allocator_iterator_builder_t. */
    typedef std_mem::allocator_traits<allocator_iterator_builder_t> allocator_iterator_builder_traits_t;
    /** @brief Traits for @ref allocator_const_iterator_builder_t. */
    typedef std_mem::allocator_traits<allocator_const_iterator_builder_t> allocator_const_iterator_builder_traits_t;
    /** @brief Mutable pointer type. */
    typedef std_mem::allocator_traits<allocator_type>::pointer pointer;
    /** @brief Const pointer type. */
    typedef std_mem::allocator_traits<allocator_type>::const_pointer const_pointer;
    /** @brief Unsigned size/index type. */
    typedef std_mem::allocator_traits<allocator_type>::size_type size_type;
    /** @brief Signed difference type for iterators and indices. */
    typedef std_mem::allocator_traits<allocator_type>::difference_type difference_type;
    /** @brief Mutable forward iterator type. */
    typedef iterator_builder_t::iterator_type iterator;
    /** @brief Const forward iterator type. */
    typedef const_iterator_builder_t::iterator_type const_iterator;
    /** @brief Mutable reverse iterator type. */
    typedef std_ivy::reverse_iterator<iterator> reverse_iterator;
    /** @brief Const reverse iterator type. */
    typedef std_ivy::reverse_iterator<const_iterator> const_reverse_iterator;

    friend class kernel_generic_transfer_internal_memory<IvyVector<T, Allocator>>;

  protected:
    /** @brief Underlying managed data container. */
    data_container _data;
    /** @brief Mutable iterator builder. */
    iterator_builder_t _iterator_builder;
    /** @brief Const iterator builder. */
    const_iterator_builder_t _const_iterator_builder;

    /** @brief Transfer internal memory for data and iterator-builder state. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old);

    /** @brief Destroy iterator builders before rebuild or destruction. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void destroy_iterator_builders();
    /** @brief Recreate iterator builders to match current storage. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void reset_iterator_builders();

  public:
    /** @brief Default constructor. */
    __HOST_DEVICE__ IvyVector();
    /** @brief Copy constructor. */
    __HOST_DEVICE__ IvyVector(IvyVector const& v);
    /** @brief Move constructor. */
    __HOST_DEVICE__ IvyVector(IvyVector&& v);
    /**
     * @brief Construct with @p n elements initialized from forwarded arguments.
     * @param n Number of elements.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param args Constructor arguments forwarded to element initialization.
     */
    template<typename... Args> __HOST_DEVICE__ IvyVector(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    /**
     * @brief Construct from an iterator range.
     * @tparam InputIterator Input iterator type.
     * @param first Start iterator.
     * @param last End iterator.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     */
    template<typename InputIterator> __HOST_DEVICE__ IvyVector(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    /**
     * @brief Construct from an initializer list.
     * @param ilist Source initializer list.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     */
    __HOST_DEVICE__ IvyVector(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);
    /** @brief Destructor. */
    __HOST_DEVICE__ ~IvyVector();

    /** @brief Copy assignment operator. */
    __HOST_DEVICE__ IvyVector& operator=(IvyVector const& v);
    /** @brief Move assignment operator. */
    __HOST_DEVICE__ IvyVector& operator=(IvyVector&& v);

    /**
     * @brief Replace contents with @p n elements initialized from forwarded arguments.
     * @param n Number of elements.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param args Constructor arguments forwarded to element initialization.
     */
    template<typename... Args> __HOST_DEVICE__ void assign(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    /**
     * @brief Replace contents from an iterator range.
     * @tparam InputIterator Input iterator type.
     * @param first Start iterator.
     * @param last End iterator.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     */
    template<typename InputIterator> __HOST_DEVICE__ void assign(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    /**
     * @brief Replace contents from an initializer list.
     * @param ilist Source initializer list.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     */
    __HOST_DEVICE__ void assign(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

    /** @brief Bounds-checked mutable element access. */
    __HOST_DEVICE__ reference at(size_type n);
    /** @brief Bounds-checked const element access. */
    __HOST_DEVICE__ const_reference at(size_type n) const;
    /** @brief Unchecked mutable element access. */
    __HOST_DEVICE__ reference operator[](size_type n);
    /** @brief Unchecked const element access. */
    __HOST_DEVICE__ const_reference operator[](size_type n) const;
    /** @brief Access first element. */
    __HOST_DEVICE__ reference front();
    /** @brief Access first element (const overload). */
    __HOST_DEVICE__ const_reference front() const;
    /** @brief Access last element. */
    __HOST_DEVICE__ reference back();
    /** @brief Access last element (const overload). */
    __HOST_DEVICE__ const_reference back() const;
    /** @brief Access raw mutable pointer to contiguous storage. */
    __HOST_DEVICE__ pointer data();
    /** @brief Access raw const pointer to contiguous storage. */
    __HOST_DEVICE__ const_pointer data() const;

    /** @brief Return mutable iterator to beginning. */
    __HOST_DEVICE__ iterator begin();
    /** @brief Return const iterator to beginning. */
    __HOST_DEVICE__ const_iterator begin() const;
    /** @brief Return const iterator to beginning. */
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

    /** @brief Check if container is empty. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool empty() const;
    /** @brief Number of stored elements. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type size() const;
    /** @brief Maximum representable element count. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ constexpr size_type max_size() const;
    /** @brief Allocated storage capacity. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ size_type capacity() const;

    /** @brief Get memory type used by the underlying storage. */
    __HOST_DEVICE__ IvyMemoryType get_memory_type() const;
    /** @brief Get associated GPU stream pointer. */
    __HOST_DEVICE__ IvyGPUStream* gpu_stream() const;

    /**
     * @brief Ensure capacity is at least @p n using explicit memory target and stream.
     * @param n Required minimum capacity.
     * @param mem_type Target memory type for reallocation.
     * @param stream Optional GPU stream.
     */
    __HOST_DEVICE__ void reserve(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
    /**
     * @brief Ensure capacity is at least @p n.
     * @param n Required minimum capacity.
     */
    __HOST_DEVICE__ void reserve(size_type n);
    /** @brief Reduce capacity to fit current size where possible. */
    __HOST_DEVICE__ void shrink_to_fit();

    /** @brief Remove all elements and reset storage state. */
    __HOST_DEVICE__ void clear();

    /**
     * @brief Insert one element at position @p pos.
     * @tparam PosIterator Position iterator type.
     * @param pos Insertion position.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param args Arguments forwarded to element construction.
     * @return Iterator to inserted element.
     */
    template<typename PosIterator, typename... Args> __HOST_DEVICE__ iterator insert(PosIterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    /**
     * @brief Insert @p n copies/constructed elements at @p pos.
     * @tparam PosIterator Position iterator type.
     * @param pos Insertion position.
     * @param n Number of elements to insert.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param args Arguments forwarded to element construction.
     * @return Iterator to first inserted element.
     */
    template<typename PosIterator, typename... Args> __HOST_DEVICE__ iterator insert(PosIterator pos, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    /**
     * @brief Insert range @p [first,last) at @p pos.
     * @tparam PosIterator Position iterator type.
     * @tparam InputIterator Input iterator type.
     * @param pos Insertion position.
     * @param first Range start.
     * @param last Range end.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @return Iterator to first inserted element.
     */
    template<typename PosIterator, typename InputIterator> __HOST_DEVICE__ iterator insert(PosIterator pos, InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
    /**
     * @brief Insert initializer-list contents at @p pos.
     * @tparam PosIterator Position iterator type.
     * @param pos Insertion position.
     * @param ilist Source initializer list.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @return Iterator to first inserted element.
     */
    template<typename PosIterator> __HOST_DEVICE__ iterator insert(PosIterator pos, std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

    /**
     * @brief Erase element at @p pos.
     * @tparam PosIterator Position iterator type.
     * @param pos Iterator to element to erase.
     * @return Iterator following erased element.
     */
    template<typename PosIterator> __HOST_DEVICE__ iterator erase(PosIterator pos);
    /**
     * @brief Erase range @p [first,last).
     * @tparam PosIterator Position iterator type.
     * @param first Start iterator.
     * @param last End iterator.
     * @return Iterator following last removed element.
     */
    template<typename PosIterator> __HOST_DEVICE__ iterator erase(PosIterator first, PosIterator last);

    /** @brief Append an element using current memory context. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void push_back(value_type const& val);
    /**
     * @brief Append an element using explicit memory target and stream.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param val Value to append.
     */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void push_back(IvyMemoryType mem_type, IvyGPUStream* stream, value_type const& val);
    /** @brief Remove the last element. */
    __HOST_DEVICE__ void pop_back();

    /**
     * @brief Construct and insert an element in-place at @p pos.
     * @tparam PosIterator Position iterator type.
     * @param pos Insertion position.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param args Arguments forwarded to element construction.
     * @return Iterator to the inserted element.
     */
    template<typename PosIterator, typename... Args> __HOST_DEVICE__ iterator emplace(PosIterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);
    /**
     * @brief Construct and append an element in-place.
     * @param args Arguments forwarded to element construction.
     */
    template<typename... Args> __HOST_DEVICE__ void emplace_back(Args&&... args);
    /**
     * @brief Construct and append an element in-place using explicit memory target and stream.
     * @param mem_type Target memory type.
     * @param stream Optional GPU stream.
     * @param args Arguments forwarded to element construction.
     */
    template<typename... Args> __HOST_DEVICE__ void emplace_back(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

    /**
     * @brief Resize the container to @p n elements.
     * @param n Target size.
     * @param mem_type Target memory type for any required allocation.
     * @param stream Optional GPU stream.
     * @param args Constructor arguments used for appended elements when growing.
     */
    template<typename... Args> __HOST_DEVICE__ void resize(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args);

    /** @brief Swap all state with another vector. */
    __HOST_DEVICE__ void swap(IvyVector& v);

    /** @brief Get const reference to underlying data container. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ data_container const& get_data_container() const;
    /** @brief Get const reference to mutable-iterator builder state. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ iterator_builder_t const& get_iterator_builder() const;
    /** @brief Get const reference to const-iterator builder state. */
    __INLINE_FCN_RELAXED__ __HOST_DEVICE__ const_iterator_builder_t const& get_const_iterator_builder() const;
  };
  /** @brief STL-style alias for IvyVector. */
  template<typename T, typename Allocator=std_mem::allocator<T>> using vector = IvyVector<T, Allocator>;

  /**
  Specialization of the value printout routine
  */
  /** @brief value_printout specialization declaration for IvyVector. */
  template<typename T, typename Allocator> struct value_printout<IvyVector<T, Allocator>>;

}
namespace std_util{
  /** @brief swap overload for IvyVector. */
  template<typename T, typename Allocator> __HOST_DEVICE__ void swap(std_ivy::IvyVector<T, Allocator>& a, std_ivy::IvyVector<T, Allocator>& b);
}
namespace std_mem{
  /** @brief Non-owning memview alias over IvyVector objects. */
  template<typename T, typename Allocator=std_mem::allocator<T>> using vector_view = std_mem::memview<std_ivy::IvyVector<T, Allocator>>;
  template<typename T, typename Allocator=std_mem::allocator<T>>
  /** @brief Build a memview over an IvyVector object. */
  __HOST_DEVICE__ vector_view<T, Allocator> view(std_ivy::IvyVector<T, Allocator> const& vec);
}


#endif
