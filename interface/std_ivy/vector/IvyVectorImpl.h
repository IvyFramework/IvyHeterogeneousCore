#ifndef IVYVECTORIMPL_H
#define IVYVECTORIMPL_H


#include "std_ivy/vector/IvyVectorImpl.hh"


#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector() :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {}
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(IvyVector const& v) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    allocator_data_container_traits::transfer(std_mem::addressof(_data), std_mem::addressof(v._data), 1, def_mem_type, def_mem_type, stream);
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(IvyVector&& v) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory()),
    _data(std_util::move(v._data))
  {
    check_write_access_or_die(v.progenitor_mem_type);
  }
  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {
    this->assign(n, mem_type, stream, args...);
  }
  template<typename T, typename Allocator> template<typename InputIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {
    this->assign(first, last, mem_type, stream);
  }
  template<typename T, typename Allocator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::IvyVector(std_ilist::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {
    this->assign(ilist, mem_type, stream);
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::~IvyVector(){
    this->destroy_iterator_builders();
    _data.reset();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::check_write_access() const{ return (progenitor_mem_type==IvyMemoryHelpers::get_execution_default_memory()); }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::check_write_access(IvyMemoryType const& mem_type) const{ return (progenitor_mem_type==mem_type); }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::check_write_access_or_die() const{
    if (!this->check_write_access()){
      __PRINT_ERROR__("IvyVector: Write access denied.\n");
      assert(false);
    }
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::check_write_access_or_die(IvyMemoryType const& mem_type) const{
    if (!this->check_write_access(mem_type)){
      __PRINT_ERROR__("IvyVector: Write access denied.\n");
      assert(false);
    }
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>& IvyVector<T, Allocator>::operator=(IvyVector<T, Allocator> const& v){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    _data.reset();
    allocator_data_container_traits::transfer(std_mem::addressof(_data), std_mem::addressof(v._data), 1, def_mem_type, def_mem_type, stream);
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>& IvyVector<T, Allocator>::operator=(IvyVector<T, Allocator>&& v){
    check_write_access_or_die(v.progenitor_mem_type);
    _data = std_util::move(v._data);
    _iterator_builder = std_util::move(v._iterator_builder);
    _const_iterator_builder = std_util::move(v._const_iterator_builder);
  }

  template<typename T, typename Allocator> template<typename... Args> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    check_write_access_or_die();
    allocator_type a;
    _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, n, mem_type, stream, args...);
    this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> template<typename InputIterator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::assign(InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream){
    check_write_access_or_die();
    using category = typename std_ivy::iterator_traits<InputIterator>::iterator_category;
    static_assert(std_ttraits::is_base_of_v<std_ivy::input_iterator_tag, category>);
    allocator_type a;
    typename InputIterator::difference_type n = std_iter::distance(first, last);
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
    check_write_access_or_die();
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
    this->reset_iterator_builders();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ bool IvyVector<T, Allocator>::transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
    bool res = true;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = _data.gpu_stream();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        //printf("IvyVector::transfer_internal_memory: Transferring data.\n");
        res &= allocator_data_container::transfer_internal_memory(&_data, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        //printf("IvyVector::transfer_internal_memory: Transferring _iterator_builder.\n");
        res &= allocator_iterator_builder_t::transfer_internal_memory(&_iterator_builder, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        //printf("IvyVector::transfer_internal_memory: Transferring _const_iterator_builder.\n");
        res &= allocator_const_iterator_builder_t::transfer_internal_memory(&_const_iterator_builder, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        //printf("IvyVector::transfer_internal_memory: Reset its.\n");
        this->reset_iterator_builders();
      )
    );
    return res;
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::destroy_iterator_builders(){
    check_write_access_or_die();
    _iterator_builder.reset();
    _const_iterator_builder.reset();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::reset_iterator_builders(){
    check_write_access_or_die();
    auto const mem_type = _data.get_memory_type();
    auto const& stream = _data.gpu_stream();
    auto const s = _data.size();
    auto const c = _data.capacity();
    auto ptr = _data.get();
    _iterator_builder.reset(ptr, s, c, mem_type, stream);
    _const_iterator_builder.reset(ptr, s, c, mem_type, stream);
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
    check_write_access_or_die();
    auto const current_capacity = this->capacity();
    _data.reserve(n);
    if (current_capacity!=this->capacity()) this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::reserve(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream){
    check_write_access_or_die();
    auto const current_capacity = this->capacity();
    auto const current_mem_type = _data.get_memory_type();
    _data.reserve(n, mem_type, stream);
    if (current_capacity!=this->capacity() || mem_type!=current_mem_type) this->reset_iterator_builders();
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::size_type IvyVector<T, Allocator>::capacity() const{ return _data.capacity(); }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::shrink_to_fit(){
    check_write_access_or_die();
    auto const current_capacity = this->capacity();
    _data.shrink_to_fit();
    if (current_capacity!=this->capacity()) this->reset_iterator_builders();
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::clear(){
    check_write_access_or_die();
    _data.reset();
    this->reset_iterator_builders();
  }

  // insert functions
  template<typename T, typename Allocator> template<typename PosIterator, typename... Args>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(
    PosIterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args
  ){
    check_write_access_or_die();
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, 1, 2, mem_type, stream, args...);
      this->reset_iterator_builders();
      return this->begin();
    }
    else{
      auto const current_capacity = this->capacity();
      typename PosIterator::pointer mem_loc_pos_const = std_mem::addressof(*pos);
      auto mem_loc_pos = __CONST_CAST__(std_ttraits::remove_const_t<std_ttraits::remove_reference_t<decltype(*mem_loc_pos_const)>>*, mem_loc_pos_const);
      size_type const n_size = this->size();
      size_type iloc = n_size;
      {
        auto ptr = _data.get();
        for (size_type i=0; i<n_size; ++i){
          if (ptr==mem_loc_pos){
            iloc = i;
            break;
          }
          ++ptr;
        }
      }
      if (iloc==n_size){
        __PRINT_ERROR__("IvyVector::insert: Invalid data position.\n");
        return this->end();
      }
      _data.insert(iloc, args...);
      mem_loc_pos = std_mem::addressof(_data[iloc]);
      if (current_capacity!=this->capacity()){
        this->reset_iterator_builders();
      }
      else{
        _iterator_builder.insert(iloc, mem_loc_pos, mem_type, stream);
        _const_iterator_builder.insert(iloc, mem_loc_pos, mem_type, stream);
      }
      auto it_ins = _iterator_builder.find_pointable(mem_loc_pos);
      return *it_ins;
    }
  }
  template<typename T, typename Allocator> template<typename PosIterator, typename... Args>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(PosIterator pos, size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    check_write_access_or_die();
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, n, mem_type, stream, args...);
      this->reset_iterator_builders();
      return this->begin();
    }
    else{
      IvyVector<T, Allocator>::iterator res = this->end();
      bool first_time = true;
      for (size_type i=0; i<n; ++i){
        if (first_time) res = this->insert(pos, mem_type, stream, args...);
        else res = this->insert(res+1, mem_type, stream, args...);
        first_time = false;
        if (!res.is_valid()) return this->end();
      }
      if (n>1) res -= (n-1);
      return res;
    }
  }
  template<typename T, typename Allocator> template<typename PosIterator, typename InputIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(
    PosIterator pos, InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    check_write_access_or_die();
    IvyVector<T, Allocator>::iterator res = this->end();
    bool first_time = true;
    size_type n = 0;
    while (first!=last){
      auto const& v = *first;
      if (first_time) res = this->insert(pos, mem_type, stream, v);
      else res = this->insert(res+1, mem_type, stream, v);
      first_time = false;
      if (!res.is_valid()) return this->end();
      ++first;
      ++n;
    }
    if (n>1) res -= (n-1);
    return res;
  }
  template<typename T, typename Allocator> template<typename PosIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::insert(
    PosIterator pos, std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    check_write_access_or_die();
    IvyVector<T, Allocator>::iterator res = this->end();
    bool first_time = true;
    size_type const n = ilist.size();
    for (auto const& v:ilist){
      if (first_time) res = this->insert(pos, mem_type, stream, v);
      else res = this->insert(res+1, mem_type, stream, v);
      first_time = false;
      if (!res.is_valid()) return this->end();
    }
    if (n>1) res -= (n-1);
    return res;
  }

  // erase functions
  template<typename T, typename Allocator> template<typename PosIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::erase(PosIterator pos){
    check_write_access_or_die();

    if (!_data) return iterator();

    typename PosIterator::pointer mem_loc_pos_const = std_mem::addressof(*pos);
    auto mem_loc_pos = __CONST_CAST__(std_ttraits::remove_const_t<std_ttraits::remove_reference_t<decltype(*mem_loc_pos_const)>>*, mem_loc_pos_const);
    size_type const n_size = this->size();
    size_type iloc = n_size;
    {
      auto ptr = _data.get();
      for (size_type i=0; i<n_size; ++i){
        if (ptr==mem_loc_pos){
          iloc = i;
          break;
        }
        ++ptr;
      }
    }
    if (iloc==n_size){
      __PRINT_ERROR__("IvyVector::erase: Invalid data position.\n");
      return this->end();
    }

    _data.erase(iloc);
    _iterator_builder.erase(iloc);
    _const_iterator_builder.erase(iloc);

    if (!_data) return iterator();
    else if (this->size()<=iloc) return this->end();
    auto mem_loc_pos_next = std_mem::addressof(_data[iloc]);
    return *(_iterator_builder.find_pointable(mem_loc_pos_next));
  }
  template<typename T, typename Allocator> template<typename PosIterator>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::erase(PosIterator first, PosIterator last){
    check_write_access_or_die();

    IvyVector<T, Allocator>::iterator res = this->end();
    while (first!=last){
      res = this->erase(first);
      if (!res.is_valid()) break;
      ++first;
    }
    return res;
  }

  // push_back and pop_back functions
  template<typename T, typename Allocator>
  __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::push_back(IvyMemoryType mem_type, IvyGPUStream* stream, value_type const& val){
    check_write_access_or_die();

    auto const current_capacity = this->capacity();
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type>(a, 1, 2, mem_type, stream, val);
      this->reset_iterator_builders();
    }
    else{
      _data.emplace_back(val);
      if (current_capacity!=this->capacity()) this->reset_iterator_builders();
      else{
        auto const& mem_loc = _data.get()+(_data.size()-1);
        _iterator_builder.push_back(mem_loc, mem_type, stream);
        _const_iterator_builder.push_back(mem_loc, mem_type, stream);
      }
    }
  }
  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::pop_back(){
    check_write_access_or_die();

    if (!_data) return;
    _iterator_builder.pop_back();
    _const_iterator_builder.pop_back();
    _data.pop_back();
  }

  // emplace and emplace_back
  template<typename T, typename Allocator> template<typename PosIterator, typename... Args>
  __CUDA_HOST_DEVICE__ IvyVector<T, Allocator>::iterator IvyVector<T, Allocator>::emplace(PosIterator pos, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    check_write_access_or_die();
    return this->insert(pos, mem_type, stream, args...);
  }
  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::emplace_back(IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    check_write_access_or_die();
    if (!_data || _data.get_memory_type()!=mem_type){
      allocator_type a;
      _data = std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, Args...>(a, 1, 2, mem_type, stream, args...);
      this->reset_iterator_builders();
    }
    else{
      auto const current_capacity = this->capacity();
      _data.emplace_back(args...);
      if (current_capacity!=this->capacity()) this->reset_iterator_builders();
      else{
        auto mem_loc = _data.get()+(_data.size()-1);
        _iterator_builder.push_back(mem_loc, mem_type, stream);
        _const_iterator_builder.push_back(mem_loc, mem_type, stream);
      }
    }
  }

  template<typename T, typename Allocator> template<typename... Args>
  __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::resize(size_type n, IvyMemoryType mem_type, IvyGPUStream* stream, Args&&... args){
    check_write_access_or_die();
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

  template<typename T, typename Allocator>
  __CUDA_HOST_DEVICE__ typename IvyVector<T, Allocator>::data_container const& IvyVector<T, Allocator>::get_data_container() const{
    return _data;
  }
  template<typename T, typename Allocator>
  __CUDA_HOST_DEVICE__ typename IvyVector<T, Allocator>::iterator_builder_t const& IvyVector<T, Allocator>::get_iterator_builder() const{
    return _iterator_builder;
  }
  template<typename T, typename Allocator>
  __CUDA_HOST_DEVICE__ typename IvyVector<T, Allocator>::const_iterator_builder_t const& IvyVector<T, Allocator>::get_const_iterator_builder() const{
    return _const_iterator_builder;
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void IvyVector<T, Allocator>::swap(IvyVector& v){
    check_write_access_or_die(v.progenitor_mem_type);
    std_mem::swap(_data, v._data);
    std_util::swap(_iterator_builder, v._iterator_builder);
    std_util::swap(_const_iterator_builder, v._const_iterator_builder);
  }

  template<typename T, typename Allocator> __CUDA_HOST_DEVICE__ void swap(IvyVector<T, Allocator>& a, IvyVector<T, Allocator>& b){ a.swap(b); }


}

#endif


#endif
