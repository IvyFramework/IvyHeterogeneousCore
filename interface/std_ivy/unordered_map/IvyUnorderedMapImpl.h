#ifndef IVYUNORDEREDMAPIMPL_H
#define IVYUNORDEREDMAPIMPL_H


#include "std_ivy/unordered_map/IvyUnorderedMapImpl.hh"


#ifdef __USE_CUDA__

#define __UMAPTPLARGSINIT__ <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
#define __UMAPTPLARGS__ <Key, T, Hash, KeyEqual, Allocator>

namespace std_ivy{
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::IvyUnorderedMap() :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {}
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::IvyUnorderedMap(IvyUnorderedMap const& v) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory())
  {
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    allocator_data_container_traits::transfer(std_mem::addressof(_data), std_mem::addressof(v._data), 1, def_mem_type, def_mem_type, stream);
    this->reset_iterator_builder();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::IvyUnorderedMap(IvyUnorderedMap&& v) :
    progenitor_mem_type(IvyMemoryHelpers::get_execution_default_memory()),
    _data(std_util::move(v._data)),
    _iterator_builder(std_util::move(v._iterator_builder))
  {
    check_write_access_or_die(v.progenitor_mem_type);
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::~IvyUnorderedMap(){
    this->destroy_iterator_builder();
    _data.reset();
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__& IvyUnorderedMap __UMAPTPLARGS__::operator=(IvyUnorderedMap __UMAPTPLARGS__ const& v){
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = v._data.get_gpu_stream();
    _data.reset();
    allocator_data_container_traits::transfer(std_mem::addressof(_data), std_mem::addressof(v._data), 1, def_mem_type, def_mem_type, stream);
    this->reset_iterator_builder();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__& IvyUnorderedMap __UMAPTPLARGS__::operator=(IvyUnorderedMap __UMAPTPLARGS__&& v){
    check_write_access_or_die(v.progenitor_mem_type);
    _data = std_util::move(v._data);
    _iterator_builder = std_util::move(v._iterator_builder);
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ bool IvyUnorderedMap __UMAPTPLARGS__::check_write_access() const{ return (progenitor_mem_type==IvyMemoryHelpers::get_execution_default_memory()); }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ bool IvyUnorderedMap __UMAPTPLARGS__::check_write_access(IvyMemoryType const& mem_type) const{ return (progenitor_mem_type==mem_type); }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::check_write_access_or_die() const{
    if (!this->check_write_access()){
      __PRINT_ERROR__("IvyUnorderedMap: Write access denied.\n");
      assert(false);
    }
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::check_write_access_or_die(IvyMemoryType const& mem_type) const{
    if (!this->check_write_access(mem_type)){
      __PRINT_ERROR__("IvyUnorderedMap: Write access denied.\n");
      assert(false);
    }
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::destroy_iterator_builder(){
    check_write_access_or_die();
    _iterator_builder.reset();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::reset_iterator_builder(){
    check_write_access_or_die();
    auto const mem_type = _data.get_memory_type();
    auto const& stream = _data.gpu_stream();
    auto const s = _data.size();
    auto ptr = _data.get();
    _iterator_builder.reset(ptr, s, mem_type, stream);
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ bool IvyUnorderedMap __UMAPTPLARGS__::transfer_internal_memory(IvyMemoryType const& new_mem_type, bool release_old){
    bool res = true;
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = _data.gpu_stream();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        res &= allocator_data_container::transfer_internal_memory(&_data, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        res &= allocator_iterator_builder_t::transfer_internal_memory(&_iterator_builder, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        this->reset_iterator_builder();
      )
    );
    return res;
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reference IvyUnorderedMap __UMAPTPLARGS__::front(){
    return *(_iterator_builder.front());
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reference IvyUnorderedMap __UMAPTPLARGS__::front() const{
    return *(_iterator_builder.cfront());
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reference IvyUnorderedMap __UMAPTPLARGS__::back(){
    return *(_iterator_builder.back());
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reference IvyUnorderedMap __UMAPTPLARGS__::back() const{
    return *(_iterator_builder.cback());
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::begin(){
    return _iterator_builder.begin();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::end(){
    return _iterator_builder.end();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rbegin(){
    return _iterator_builder.rbegin();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rend(){
    return _iterator_builder.rend();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::begin() const{
    return _iterator_builder.cbegin();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::cbegin() const{
    return _iterator_builder.cbegin();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::end() const{
    return _iterator_builder.cend();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_iterator IvyUnorderedMap __UMAPTPLARGS__::cend() const{
    return _iterator_builder.cend();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rbegin() const{
    return _iterator_builder.crbegin();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::crbegin() const{
    return _iterator_builder.crbegin();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::rend() const{
    return _iterator_builder.crend();
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::const_reverse_iterator IvyUnorderedMap __UMAPTPLARGS__::crend() const{
    return _iterator_builder.crend();
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ bool IvyUnorderedMap __UMAPTPLARGS__::empty() const{ return this->size()==0; }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::size() const{ return _iterator_builder.n_valid_iterators(); }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ constexpr IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::max_size() const{ return std_limits::numeric_limits<size_type>::max(); }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::capacity() const{ return _iterator_builder.n_capacity_valid_iterators(); }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::clear(){
    check_write_access_or_die();
    _data.reset();
    this->reset_iterator_builder();
  }

  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::size_type IvyUnorderedMap __UMAPTPLARGS__::get_predicted_bucket_count() const{
    return KeyEqual::bucket_size(this->size(), this->capacity());
  }

  template __UMAPTPLARGSINIT__
    __CUDA_HOST_DEVICE__ void IvyUnorderedMap __UMAPTPLARGS__::swap(IvyUnorderedMap __UMAPTPLARGS__& v){
    check_write_access_or_die(v.progenitor_mem_type);
    std_mem::swap(_data, v._data);
    std_ivy::swap(_iterator_builder, v._iterator_builder);
  }
  template __UMAPTPLARGSINIT__
  __CUDA_HOST_DEVICE__ void swap(IvyUnorderedMap __UMAPTPLARGS__& a, IvyUnorderedMap __UMAPTPLARGS__& b);

  // insert functions
  template __UMAPTPLARGSINIT__ template<typename... Args>
  __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::insert(
    IvyMemoryType mem_type, IvyGPUStream* stream, Key const& key, Args&&... args
  ){
    check_write_access_or_die();
    constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    size_type const current_capacity = this->capacity();
    size_type const current_size = this->size();
    size_type const new_size = current_size + 1;
    size_type const new_capacity = (new_size>current_capacity ? new_size + 1 : current_capacity);
    size_type const current_bucket_capacity = _data.capacity();
    size_type const current_bucket_size = _data.size();
    size_type const new_bucket_capacity = key_equal::bucket_size(new_size, new_capacity);
    hash_result_type const hash = hasher()(key);
    allocator_type a;
    if (!_data || _data.get_memory_type()!=mem_type){
      _data = std_mem::make_unique<bucket_element>(
        a, 1, new_bucket_capacity,
        mem_type, stream,
        std_util::make_pair<hash_result_type, bucket_data_type>(
          hash,
          std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, bucket_data_type, key_type, mapped_type>(
            a, 1, 2,
            mem_type, stream, key, mapped_type(std_util::forward<Args>(args)...)
          )
        )
      );
      this->reset_iterator_builder();
      return this->begin();
    }
    else{
      bool is_found = false;
      value_type* mem_loc_pos = nullptr;
      value_type* mem_loc_pos_final = nullptr;
      if (mem_type==def_mem_type){
        auto& data_ptr = _data.get();
        for (size_type ib=0; ib<current_bucket_capacity; ++ib){
          auto& bucket_element = *data_ptr;
          auto const& bucket_hash = bucket_element.first;
          if (key_equal::eval(current_size, current_capacity, hash, bucket_hash)){
            auto& data_bucket = bucket_element.second;
            for (size_t jd=0; jd<data_bucket.size(); ++jd){
              if (data_bucket[jd].first == key){
                data_bucket[jd].second = mapped_type(std_util::forward<Args>(args)...);
                mem_loc_pos_final = data_bucket.get() + jd;
                is_found = true;
                break;
              }
            }
            if (!is_found){
              data_bucket.emplace_back(key, mapped_type(std_util::forward<Args>(args)...));
              mem_loc_pos_final = mem_loc_pos = data_bucket.get() + data_bucket.size() - 1;
              is_found = true;
              break;
            }
          }
          if (is_found) break;
          ++data_ptr;
        }
        if (!is_found){
          // If we stil have not found the bucket, we need to create a new one.
          _data.emplace_back(
            hash,
            std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, bucket_data_type, key_type, mapped_type>(
              a, 1, 2,
              mem_type, stream, key, mapped_type(std_util::forward<Args>(args)...)
            )
          );
          mem_loc_pos_final = mem_loc_pos = _data[_data.size()-1].second.get();
        }
      }
      else{
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            bucket_element* data_ptr = _data.get();
            bucket_element* tmp_data_ptr = nullptr;
            IvyMemoryHelpers::allocate_memory(tmp_data_ptr, current_bucket_size+1, def_mem_type, ref_stream);
            IvyMemoryHelpers::transfer_memory(tmp_data_ptr, data_ptr, current_bucket_size, def_mem_type, mem_type, ref_stream);
            bucket_element* tr_tmp_data_ptr = tmp_data_ptr;
            for (size_type ib=0; ib<current_bucket_size; ++ib){
              auto& bucket_element = *tr_tmp_data_ptr;
              auto const& bucket_hash = bucket_element.first;
              if (key_equal::eval(current_size, current_capacity, hash, bucket_hash)){
                auto& data_bucket = bucket_element.second;
                size_type const n_size_data_bucket = data_bucket.size();
                value_type* bucket_data_ptr = data_bucket.get();
                value_type* tmp_bucket_data_ptr = nullptr;
                IvyMemoryHelpers::allocate_memory(tmp_bucket_data_ptr, n_size_data_bucket, def_mem_type, ref_stream);
                IvyMemoryHelpers::transfer_memory(tmp_bucket_data_ptr, bucket_data_ptr, n_size_data_bucket, def_mem_type, mem_type, ref_stream);
                value_type* tr_tmp_bucket_data_ptr = tmp_bucket_data_ptr;
                for (size_t jd=0; jd<n_size_data_bucket; ++jd){
                  value_type& data_value = *tr_tmp_bucket_data_ptr;
                  if (data_value.first == key){
                    data_value.second = mapped_type(std_util::forward<Args>(args)...);
                    mem_loc_pos_final = bucket_data_ptr + jd;
                    is_found = true;
                    break;
                  }
                  ++tr_tmp_bucket_data_ptr;
                }
                if (!is_found){
                  data_bucket.emplace_back(key, mapped_type(std_util::forward<Args>(args)...));
                  mem_loc_pos_final = mem_loc_pos = data_bucket.get() + data_bucket.size() - 1;
                  is_found = true;
                  break;
                }
                else IvyMemoryHelpers::transfer_memory(bucket_data_ptr, tmp_bucket_data_ptr, n_size_data_bucket, mem_type, def_mem_type, ref_stream);
                IvyMemoryHelpers::free_memory(tmp_bucket_data_ptr, n_size_data_bucket, def_mem_type, ref_stream);
              }
              if (is_found) break;
              ++tr_tmp_data_ptr;
            }
            if (!is_found){
              // If we stil have not found the bucket, we need to create a new one.
              _data.emplace_back(
                hash,
                std_mem::build_unified<value_type, IvyPointerType::unique, allocator_type, bucket_data_type, key_type, mapped_type>(
                  a, 1, 2,
                  mem_type, stream, key, mapped_type(std_util::forward<Args>(args)...)
                )
              );
              IvyMemoryHelpers::transfer_memory(tmp_data_ptr+current_bucket_size, _data.get()+current_bucket_size, 1, def_mem_type, mem_type, ref_stream);
              bucket_element* tr_tmp_data_ptr = tmp_data_ptr+current_bucket_size;
              auto& data_bucket = (*tr_tmp_data_ptr).second;
              mem_loc_pos_final = mem_loc_pos = data_bucket.get();
            }
            else IvyMemoryHelpers::transfer_memory(data_ptr, tmp_data_ptr, current_bucket_size, mem_type, def_mem_type, ref_stream);
            IvyMemoryHelpers::free_memory(tmp_data_ptr, current_bucket_size+1, def_mem_type, ref_stream);
          )
        );
      }

      if (mem_loc_pos){
        // We need to update the iterators if a new element is inserted.
        // If bucket capacity needs to be increased, we just rehash, which will update the iterator builder in the process.
        // Otherwise, we need to insert a new iterator into the existing iterator builder construct.
        if (new_bucket_capacity!=current_bucket_capacity) this->rehash(new_bucket_capacity, mem_type, stream);
        else _iterator_builder.push_back(mem_loc_pos, mem_type, stream);
      }

      return _iterator_builder.find_pointable(mem_loc_pos_final);
    }
  }
  template __UMAPTPLARGSINIT__ template<typename InputIterator>
  __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::insert(
    InputIterator first, InputIterator last, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    check_write_access_or_die();
    IvyUnorderedMap __UMAPTPLARGS__::iterator res = this->end();
    bool first_time = true;
    while (first!=last){
      auto const& v = *first;
      if (first_time) res = this->insert(mem_type, stream, v.first, v.second);
      else this->insert(mem_type, stream, v.first, v.second);
      first_time = false;
      ++first;
    }
    return res;
  }
  template __UMAPTPLARGSINIT__ __CUDA_HOST_DEVICE__ IvyUnorderedMap __UMAPTPLARGS__::iterator IvyUnorderedMap __UMAPTPLARGS__::insert(
    std::initializer_list<value_type> ilist, IvyMemoryType mem_type, IvyGPUStream* stream
  ){
    check_write_access_or_die();
    IvyUnorderedMap __UMAPTPLARGS__::iterator res = this->end();
    bool first_time = true;
    for (auto const& v:ilist){
      if (first_time) res = this->insert(mem_type, stream, v.first, v.second);
      else this->insert(mem_type, stream, v.first, v.second);
      first_time = false;
    }
    return res;
  }
}

#undef __UMAPTPLARGS__
#undef __UMAPTPLARGSINIT__

#endif


#endif
