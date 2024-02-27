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

}

#undef __UMAPTPLARGS__
#undef __UMAPTPLARGSINIT__

#endif


#endif
