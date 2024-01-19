#ifndef IVYVECTORIMPL_H
#define IVYVECTORIMPL_H


#include "std_ivy/vector/IvyVectorImpl.hh"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(){}
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(IvyVector const& v){
    auto c = v.capacity();
    auto n = v.size();
    auto mem_type = v._data.get_memory_type();
    auto stream = v._data.get_gpu_stream();

    allocator_type a;
    auto _data = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator>(a, c, mem_type, stream);
    auto& data_first = v._data.get();
    auto& data_new_first = _data.get();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_type_traits::transfer(a, data_new_first, data_first, n, mem_type, mem_type, ref_stream);
      )
    );

    _iterator_builder.reset(_data, n);
    _const_iterator_builder.reset(_data, n);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(IvyVector&& v) :
    _data(std_util::move(v._data)),
    _iterator_builder(std_util::move(v._iterator_builder)),
    _const_iterator_builder(std_util::move(v._const_iterator_builder))
  {}
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->assign(n, value_type(), mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->assign(n, val, mem_type, stream);
  }
  template<typename T, typename Allocator> template<typename InputIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->assign(first, last, mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream){
    this->assign(ilist, mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::~IvyVector(){}

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector& operator=(IvyVector const& v){
    auto c = v.capacity();
    auto n = v.size();
    auto mem_type = v._data.get_memory_type();
    auto stream = v._data.get_gpu_stream();

    allocator_type a;
    auto _data = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator>(a, c, mem_type, stream);
    auto& data_first = v._data.get();
    auto& data_new_first = _data.get();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_type_traits::transfer(a, data_new_first, data_first, n, mem_type, mem_type, ref_stream);
      )
    );

    _iterator_builder.reset(_data, n);
    _const_iterator_builder.reset(_data, n);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector& operator=(IvyVector&& v){
    _data = std_util::move(v._data);
    _iterator_builder = std_util::move(v._iterator_builder);
    _const_iterator_builder = std_util::move(v._const_iterator_builder);
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream){
    allocator_type a;
    _data = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator, value_type const&>(a, n, mem_type, stream, val);
    _iterator_builder.reset(_data, n);
    _const_iterator_builder.reset(_data, n);
  }
  template<typename T, typename Allocator> template<typename InputIterator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream){
    using category = typename std_ivy::iterator_traits<InputIterator>::iterator_category;
    static_assert(std_ttraits::is_base_of_v<std_ivy::input_iterator_tag, category>);
    allocator_type a;
    InputIterator::difference_type n = std_ivy::distance(first, last);
    if (n<0){
      n = -n;
      std_util::swap(first, last);
    }
    _data = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator, value_type const&>(a, __STATIC_CAST__(size_type, n), mem_type, stream);
    if constexpr(std_ttraits::is_base_of_v<std_ivy::contiguous_iterator_tag, category>){
      auto ptr_first = std_mem::addressof(*first);
      {
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            allocator_type_traits::transfer(a, _data.get(), ptr_first, __STATIC_CAST__(size_type, n), mem_type, mem_type, ref_stream);
          )
        );
      }
    }
    else{
      auto& data_first = _data.get();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          while (first!=last){
            auto ptr_first = std_mem::addressof(*first);
            allocator_type_traits::transfer(a, data_first, ptr_first, 1, mem_type, mem_type, ref_stream);
            ++first;
            ++data_first;
          }
        )
      );
    }
    _iterator_builder.reset(_data, __STATIC_CAST__(size_type, n));
    _const_iterator_builder.reset(_data, __STATIC_CAST__(size_type, n));
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream){
    if (!ilist.empty()){
      allocator_type a;
      size_type n = ilist.size();
      _data = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator, value_type const&>(a, n, mem_type, stream);
      auto& data_first = _data.get();
      auto ptr_first = ilist.begin();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          allocator_type_traits::transfer(a, data_first, ptr_first, n, mem_type, IvyMemoryHelpers::get_execution_default_memory(), ref_stream);
        )
      );
    }
    else this->clear();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::reference IvyVector<T, Allocator>::at(size_type n){
    return _data[n];
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reference IvyVector<T, Allocator>::at(size_type n) const{
    return _data[n];
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::reference IvyVector<T, Allocator>::operator[](size_type n){
    return this->at(n);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reference IvyVector<T, Allocator>::operator[](size_type n) const{
    return this->at(n);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::reference IvyVector<T, Allocator>::front(){
    return *(_iterator_builder.front());
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reference IvyVector<T, Allocator>::front() const{
    return *(_const_iterator_builder.front());
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::reference IvyVector<T, Allocator>::back(){
    return *(_iterator_builder.back());
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reference IvyVector<T, Allocator>::back() const{
    return *(_const_iterator_builder.back());
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::pointer IvyVector<T, Allocator>::data(){
    return _data.get();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_pointer IvyVector<T, Allocator>::data() const{
    return _data.get();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::begin(){
    return _iterator_builder.begin();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::end(){
    return _iterator_builder.end();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::reverse_iterator IvyVector<T, Allocator>::rbegin(){
    return _iterator_builder.rbegin();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::reverse_iterator IvyVector<T, Allocator>::rend(){
    return _iterator_builder.rend();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_iterator IvyVector<T, Allocator>::begin() const{
    return _const_iterator_builder.begin();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_iterator IvyVector<T, Allocator>::cbegin() const{
    return _const_iterator_builder.begin();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_iterator IvyVector<T, Allocator>::end() const{
    return _const_iterator_builder.end();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_iterator IvyVector<T, Allocator>::cend() const{
    return _const_iterator_builder.end();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reverse_iterator IvyVector<T, Allocator>::rbegin() const{
    return _const_iterator_builder.rbegin();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reverse_iterator IvyVector<T, Allocator>::crbegin() const{
    return _const_iterator_builder.rbegin();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reverse_iterator IvyVector<T, Allocator>::rend() const{
    return _const_iterator_builder.rend();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::const_reverse_iterator IvyVector<T, Allocator>::crend() const{
    return _const_iterator_builder.rend();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::empty() const{ return this->size()==0; }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::size() const{

  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::max_size() const{
    return std_limits::numeric_limits<size_type>::max();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::reserve(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream = nullptr){
    if (n<=this->capacity()) return;

    auto const sn = this->size();

    allocator_type a;
    auto _data_new = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator>(a, n, mem_type, stream);
    auto& data_first = _data.get();
    auto& data_new_first = _data_new.get();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_type_traits::transfer(a, data_new_first, data_first, sn, mem_type, mem_type, ref_stream);
      )
    );

    _data = std_util::move(_data_new);
    _iterator_builder.reset(_data, sn);
    _const_iterator_builder.reset(_data, sn);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::capacity() const{ return _data.size(); }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::shrink_to_fit(){
    auto const sn = this->size();
    if (sn==this->capacity()) return;

    auto mem_type = _data.get_memory_type();
    auto stream = _data.get_gpu_stream();

    allocator_type a;
    auto _data_new = std_ivy::allocate_unified<T, IvyPointerType::unique, Allocator>(a, sn, mem_type, stream);
    auto& data_first = _data.get();
    auto& data_new_first = _data_new.get();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_type_traits::transfer(a, data_new_first, data_first, sn, mem_type, mem_type, ref_stream);
      )
    );

    _data = std_util::move(_data_new);
    _iterator_builder.reset(_data, sn);
    _const_iterator_builder.reset(_data, sn);
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::clear(){
    _data.reset();
    _iterator_builder.reset(_data, 0);
    _const_iterator_builder.reset(_data, 0);
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(
    const_iterator pos, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const ipos = pos-this->cbegin();
    auto const sn = this->size();
    auto const cn = this->capacity();

    // The call below might invalidate pos; that is why we determined ipos in advance.
    bool const call_reserve = (cn==sn);
    if (call_reserve) this->reserve(cn+2, mem_type, stream);

    value_type cpv = val;

    allocator_type a;
    auto& data_ptr = std_ivy::addressof(_data[ipos]);
    auto& data_ptr_next = data_ptr+1;
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        allocator_type_traits::transfer(a, data_ptr_next, data_ptr, sn, mem_type, mem_type, ref_stream);
        allocator_type_traits::transfer(a, data_ptr, &cpv, 1, mem_type, def_mem_type, ref_stream);
      )
    );

    // If we have not called reserve, we need to update the iterators.
    // Otherwise, iterators are already updated.
    if (!call_reserve){
      auto it = iterator_builder_t::make_pointable(data_ptr, mem_type, stream);
      auto ncpos = iterator_builder_t::make_pointable(data_ptr, mem_type, stream);
      _iterator_builder.insert(ncpos, it);

      auto c_it = const_iterator_builder_t::make_pointable(data_ptr, mem_type, stream);
      auto cpos = const_iterator_builder_t::make_pointable(data_ptr, mem_type, stream);
      _const_iterator_builder.insert(cpos, c_it);
    }
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(const_iterator pos, size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream){

  }
  template<typename T, typename Allocator> template<typename InputIterator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(const_iterator pos, InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream);
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(const_iterator pos, std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream);

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::erase(const_iterator pos, IvyMemoryType mem_type, IvyGPUStream* stream);
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::erase(const_iterator first, const_iterator last, IvyMemoryType mem_type, IvyGPUStream* stream);

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::push_back(value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream);
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::pop_back();

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::emplace(const_iterator pos, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream);
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::emplace_back(value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream);

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::resize(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream);
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::resize(size_type n, value_type const& val, IvyMemoryType mem_type, IvyGPUStream* stream);

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::swap(IvyVector& v);
}

#endif


#endif
