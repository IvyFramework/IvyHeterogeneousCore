#ifndef IVYVECTORIMPL_H
#define IVYVECTORIMPL_H


#include "std_ivy/vector/IvyVectorImpl.hh"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector() : _iterator_builder(nullptr), _const_iterator_builder(nullptr){}
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(IvyVector const& v) : _iterator_builder(nullptr), _const_iterator_builder(nullptr){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    allocator_data_container_traits::transfer(&_data, &(v._data), 1, def_mem_type, def_mem_type, stream);
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(IvyVector&& v) :
    _data(std_util::move(v._data))
  {
    v.reset_iterator_builders();
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> template<typename... Args> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args) : _iterator_builder(nullptr), _const_iterator_builder(nullptr){
    this->assign(n, mem_type, stream, args);
  }
  template<typename T, typename Allocator> template<typename InputIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream) : _iterator_builder(nullptr), _const_iterator_builder(nullptr){
    this->assign(first, last, mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream) : _iterator_builder(nullptr), _const_iterator_builder(nullptr){
    this->assign(ilist, mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::~IvyVector(){ this->destroy_iterator_builders(); }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector& operator=(IvyVector const& v){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    _data.reset();
    allocator_data_container_traits::transfer(&_data, &(v._data), 1, def_mem_type, def_mem_type, stream);
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector& operator=(IvyVector&& v){
    _data = std_util::move(v._data);
    v.reset_iterator_builders();
    this->reset_iterator_builders();
  }

  template<typename T, typename Allocator> template<typename... Args> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    allocator_type a;
    _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, n, mem_type, stream, args...);
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> template<typename InputIterator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream){
    using category = typename std_ivy::iterator_traits<InputIterator>::iterator_category;
    static_assert(std_ttraits::is_base_of_v<std_ivy::input_iterator_tag, category>);
    allocator_type a;
    InputIterator::difference_type n = std_iter::distance(first, last);
    if (n<0){
      n = -n;
      std_util::swap(first, last);
    }
    _data.reset();
    if constexpr(std_ttraits::is_base_of_v<std_iter::contiguous_iterator_tag, category>){
      auto ptr_first = std_mem::addressof(*first);
      _data.copy(ptr_first, __STATIC_CAST__(size_type, n), mem_type, stream);
    }
    else{
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          typename data_container_type::pointer new_ptr = a.allocate(__STATIC_CAST__(size_type, n), mem_type, ref_stream);
          typename data_container_type::pointer data_first = new_ptr;
          while (first!=last){
            auto ptr_first = std_mem::addressof(*first);
            allocator_type_traits::transfer(a, data_first, ptr_first, 1, mem_type, mem_type, ref_stream);
            ++first;
            ++data_first;
          }
          _data.reset(new_ptr, __STATIC_CAST__(size_type, n), mem_type, stream);
        )
      );
    }
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream){
    if (!ilist.empty()){
      allocator_type a;
      size_type n = ilist.size();
      _data.reset();
      auto ptr_first = ilist.begin();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          typename data_container_type::pointer new_ptr = a.allocate(__STATIC_CAST__(size_type, n), mem_type, ref_stream);
          allocator_type_traits::transfer(a, new_ptr, ptr_first, n, mem_type, IvyMemoryHelpers::get_execution_default_memory(), ref_stream);
          _data.reset(new_ptr, __STATIC_CAST__(size_type, n), mem_type, stream);
        )
      );
    }
    else this->clear();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::transfer_internal_memory(IvyMemoryType const& new_mem_type){
    this->destroy_iterator_builders();
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    allocator_data_container::transfer_internal_memory(&_data, 1, def_mem_type, new_mem_type, _data.gpu_stream());
    this->reset_iterator_builders();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::destroy_iterator_builders(){
    if (_iterator_builder){
      auto const mem_type = _data.get_memory_type();
      auto const& stream = _data.gpu_stream();
      if (IvyMemoryHelpers::run_acc_on_host(mem_type)){
        if (
          !run_kernel<destruct_data_kernel<iterator_builder_t>>(0, stream).parallel_1D(1, _iterator_builder, mem_type)
          ||
          !run_kernel<destruct_data_kernel<const_iterator_builder_t>>(0, stream).parallel_1D(1, _const_iterator_builder, mem_type)
          ){
          __PRINT_ERROR__("IvyVector::destroy_iterator_builders: Unable to call the acc. hardware kernel...\n");
          return false;
        }
      }
      else{
        _iterator_builder->~iterator_builder_t();
        _const_iterator_builder->~const_iterator_builder_t();
      }
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          IvyMemoryHelpers::free_memory(_iterator_builder, 1, mem_type, ref_stream);
          IvyMemoryHelpers::free_memory(_const_iterator_builder, 1, mem_type, ref_stream);
        )
      );
    }
    return true;
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::reset_iterator_builders(){
    this->destroy_iterator_builders();

    auto const mem_type = _data.get_memory_type();
    auto const& stream = _data.gpu_stream();
    auto const n = _data.size();
    auto ptr = _data.get();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        IvyMemoryHelpers::allocate_memory(_iterator_builder, 1, mem_type, ref_stream);
        IvyMemoryHelpers::allocate_memory(_const_iterator_builder, 1, mem_type, ref_stream);
        if (IvyMemoryHelpers::run_acc_on_host(mem_type)){
          if (
            !run_kernel<construct_data_kernel<iterator_builder_t>>(0, stream).parallel_1D(1, _iterator_builder, mem_type)
            ||
            !run_kernel<construct_data_kernel<const_iterator_builder_t>>(0, stream).parallel_1D(1, _const_iterator_builder, mem_type)
            ||
            !run_kernel<kernel_reset_iterator<iterator_builder_t>>(0, stream).parallel_1D(1, _iterator_builder, ptr, n, mem_type, stream)
            ||
            !run_kernel<kernel_reset_iterator<const_iterator_builder_t>>(0, stream).parallel_1D(1, _const_iterator_builder, ptr, n, mem_type, stream)
            ){
            __PRINT_ERROR__("IvyVector::reset_iterator_builders: Unable to call the acc. hardware kernel...\n");
            return false;
          }
        }
        else{
          new(_iterator_builder) iterator_builder_t(); _iterator_builder->reset(ptr, n, mem_type, stream);
          new(_const_iterator_builder) const_iterator_builder_t(); _const_iterator_builder->reset(ptr, n, mem_type, stream);
        }
      )
    );
    return true;
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
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::size() const{ return _data.size(); }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::max_size() const{
    return std_limits::numeric_limits<size_type>::max();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::reserve(size_type n){
    _data.reserve(n);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::reserve(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    _data.reserve(n, mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::capacity() const{ return _data.capacity(); }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::shrink_to_fit(){ _data.shrink_to_fit(); }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::clear(){
    _data.reset();
    this->reset_iterator_builders();
  }

  template<typename T, typename Allocator> template<typename... Args> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(
    const_iterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args
  ){
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, n, mem_type, stream, args...);
      this->reset_iterator_builders();
      return this->begin();
    }
    else{
      if (!pos) return this->end();
      auto mem_loc_pos = std_mem::addressof(*pos);
      iterator it_eff = _iterator_builder->find_pointable(mem_loc_pos);
      const_iterator cit_eff = _const_iterator_builder->find_pointable(mem_loc_pos);
      if (it_eff==this->end()) return it_eff;
      size_type iloc = _data.size();
      {
        auto ptr = _data.get();
        for (size_type i=0; i<_data.size(); ++i){
          if (ptr==mem_loc_pos){
            iloc = i;
            break;
          }
          ++ptr;
        }
      }
      if (this->size()==iloc) return this->end();
      _data.insert(iloc, args...);
      iterator it_ins = _iterator_builder->make_pointable(mem_loc_pos, mem_type, stream);
      _iterator_builder->insert(it_eff, it_ins);
      _const_iterator_builder->insert(cit_eff, _const_iterator_builder->make_pointable(mem_loc_pos, mem_type, stream));
      return it_ins;
    }
  }
  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(const_iterator pos, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, n, mem_type, stream, args...);
      this->reset_iterator_builders();
      return this->begin();
    }
    else{
      IvyVector<T, Allocator>::iterator res = this->end();
      for (size_type i=0; i<n; ++i){
        res = this->insert(pos, mem_type, stream, args...);
        if (!res.is_valid()) break;
      }
      return res;
    }
  }
  template<typename T, typename Allocator> template<typename InputIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(
    const_iterator pos, InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    IvyVector<T, Allocator>::iterator res = this->end();
    while (first!=last){
      res = this->insert(pos, mem_type, stream, *first);
      if (!res.is_valid()) break;
      ++first;
    }
    return res;
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(const_iterator pos, std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream){
    IvyVector<T, Allocator>::iterator res = this->end();
    for (auto const& v:ilist){
      res = this->insert(pos, mem_type, stream, v);
      if (!res.is_valid()) break;
    }
    return res;
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::erase(const_iterator pos){
    if (!pos || !_data) return iterator();

    auto mem_loc_pos = std_mem::addressof(*pos);
    iterator it_eff = _iterator_builder->find_pointable(mem_loc_pos);
    const_iterator cit_eff = _const_iterator_builder->find_pointable(mem_loc_pos);
    _iterator_builder->erase(it_eff);
    _const_iterator_builder->erase(cit_eff);
    size_type iloc = _data.size();
    {
      auto ptr = _data.get();
      for (size_type i=0; i<_data.size(); ++i){
        if (ptr==mem_loc_pos){
          iloc = i;
          break;
        }
        ++ptr;
      }
    }
    if (this->size()==iloc) return iterator();
    _data.erase(iloc);

    if (!_data) return iterator();
    else if (this->size()<=iloc) return this->end();
    auto mem_loc_pos_next = std_mem::addressof(*(_data[iloc]));
    return _iterator_builder->find_pointable(mem_loc_pos_next);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::erase(const_iterator first, const_iterator last){
    IvyVector<T, Allocator>::iterator res = this->end();
    while (first!=last){
      res = this->erase(*first);
      if (!res.is_valid()) break;
      ++first;
    }
    return res;
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::push_back(IvyMemoryType mem_type, IvyGPUStream* stream, value_type const& val){
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type>(a, 1, 2, mem_type, stream, val);
      this->reset_iterator_builders();
    }
    else{
      _data.emplace_back(val);
      auto mem_loc_pos = _data.get()+(_data.size()-1);
      iterator it_eff = _iterator_builder->make_pointable(mem_loc_pos, mem_type, stream);
      const_iterator cit_eff = _const_iterator_builder->make_pointable(mem_loc_pos, mem_type, stream);
      _iterator_builder->push_back(it_eff);
      _const_iterator_builder->push_back(cit_eff);
    }
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::pop_back(){
    if (!_data) return;
    _iterator_builder->pop_back();
    _const_iterator_builder->pop_back();
    _data.pop_back();
  }

  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::emplace(const_iterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    return this->insert(pos, mem_type, stream, args...);
  }
  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::emplace_back(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, 1, 2, mem_type, stream, args...);
      this->reset_iterator_builders();
    }
    else{
      _data.emplace_back(args...);
      auto mem_loc_pos = _data.get()+(_data.size()-1);
      iterator it_eff = _iterator_builder->make_pointable(mem_loc_pos, mem_type, stream);
      const_iterator cit_eff = _const_iterator_builder->make_pointable(mem_loc_pos, mem_type, stream);
      _iterator_builder->push_back(it_eff);
      _const_iterator_builder->push_back(cit_eff);
    }
  }

  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::resize(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, n, mem_type, stream, args...);
      this->reset_iterator_builders();
    }
    auto s = this->size();
    if (s==n) return;
    else if (s<n){
      this->reserve(n);
      for (size_type i=s; i<n; ++i) this->emplace_back(mem_type, stream, args...);
    }
    else{
      for (size_type i=n; i<s; ++i) this->pop_back();
      this->shrink_to_fit();
    }
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::swap(IvyVector& v){
    std_mem::swap(_data, v._data);
    std_util::swap(_iterator_builder, v._iterator_builder);
    std_util::swap(_const_iterator_builder, v._const_iterator_builder);
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void swap(IvyVector<T, Allocator>& a, IvyVector<T, Allocator>& b){ a.swap(b); }


}

#endif


#endif
