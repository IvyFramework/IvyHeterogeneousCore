#ifndef IVYTENSOR_H
#define IVYTENSOR_H


#include "config/IvyCompilerConfig.h"
#include "stream/IvyStream.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyMemory.h"
#include "autodiff/IvyBaseMathTypes.h"
#include "autodiff/base_types/IvyNodeRelations.h"
#include "autodiff/base_types/IvyBaseModifiable.h"
#include "autodiff/base_types/IvyClientManager.h"
#include "autodiff/basic_nodes/IvyTensorShape.h"
#include "IvyPrintout.h"


namespace IvyMath{
  template<typename T> class IvyTensor;
  template<typename T> struct IvyNodeSelfRelations<IvyTensor<T>>;
  template<typename T, typename U> struct IvyNodeBinaryRelations<IvyTensor<T>, U>;
  template<typename T, bool is_arithmetic = is_arithmetic_v<T>> struct tensor_data_client_updator;

  template<typename T> class IvyTensorBase : 
    public IvyBaseNode,
    public IvyBaseModifiable,
    public IvyClientManager
  {
    public:
      __HOST_DEVICE__ IvyTensorBase() : IvyBaseModifiable(), IvyClientManager(){}
      __HOST_DEVICE__ IvyTensorBase(IvyTensorBase const& other) : IvyBaseModifiable(), IvyClientManager(other){}
      __HOST_DEVICE__ IvyTensorBase(IvyTensorBase&& other) : IvyBaseModifiable(), IvyClientManager(std_util::move(other)){}
      __HOST_DEVICE__ ~IvyTensorBase() = default;

    protected:
      __HOST_DEVICE__ IvyTensorBase& operator=(IvyTensorBase const& other){
        if (this == &other) return *this;
        IvyClientManager::operator=(other);
        return *this;
      }
      __HOST_DEVICE__ IvyTensorBase& operator=(IvyTensorBase&& other){
        if (this == &other) return *this;
        IvyClientManager::operator=(std_util::move(other));
        return *this;
      }
  };

  template<typename T> class IvyTensor :
    public IvyTensorBase<T>,
    public tensor_domain_tag,
    public get_operability_t<T>
  {
  public:
    using dtype_t = T;
    using value_t = IvyTensor<T>;
    using data_container = std_mem::unique_ptr<dtype_t>;
    using allocator_data_container = std_mem::allocator<data_container>;
    using allocator_data_container_traits = std_mem::allocator_traits<allocator_data_container>;
    using allocator_tensor_shape = std_mem::allocator<IvyTensorShape>;
    using allocator_tensor_shape_traits = std_mem::allocator_traits<allocator_tensor_shape>;

  protected:
    IvyTensorShape shape_;
    data_container data_;

    __HOST_DEVICE__ bool transfer_internal_memory(std_ivy::IvyMemoryType const& new_mem_type, bool release_old){
      bool res = true;
      constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      auto stream = this->gpu_stream();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          res &= allocator_tensor_shape::transfer_internal_memory(&shape_, 1, def_mem_type, new_mem_type, ref_stream, release_old);
          res &= allocator_data_container::transfer_internal_memory(&data_, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        )
      );
      return res;
    }

  public:
    // Convenience accessors from IvyTensorShape
    __HOST_DEVICE__ IvyTensorShape const& shape() const{ return shape_; }
    __HOST_DEVICE__ IvyTensorRank_t const& rank() const{ return shape_.rank(); }
    __HOST_DEVICE__ IvyTensorDim_t const& num_elements() const{ return shape_.num_elements(); }
    __HOST_DEVICE__ std_vec::vector<IvyTensorDim_t> const& get_dimensions() const{ return shape_.get_dimensions(); }
    __HOST_DEVICE__ IvyTensorDim_t const& get_dimension(IvyTensorRank_t const& iaxis) const{ return shape_.get_dimension(iaxis); }

    // Get (number of bytes) = (number of elements)*(size of the data type in bytes).
    __HOST_DEVICE__ IvyTensorDim_t num_bytes() const{ return num_elements()*sizeof(T); }

    // Get the memory type and stream
    __HOST_DEVICE__ std_ivy::IvyMemoryType get_memory_type() const{ return shape_.get_memory_type(); }
    __HOST_DEVICE__ IvyGPUStream* gpu_stream() const{ return shape_.gpu_stream(); }

    // Get container to data
    __HOST_DEVICE__ data_container const& get_data_container() const{ return data_; }
    //__HOST_DEVICE__ data_container& get_data_container(){ return data_; }

  protected:
    template<typename... Args> __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void build_data(Args&&... args){
      data_ = std_mem::make_unique<dtype_t>(this->num_elements(), this->get_memory_type(), this->gpu_stream(), args...);
    }
    __HOST_DEVICE__ void copy_data(IvyTensor const& other){
      constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      auto stream = other.gpu_stream();
      data_container* ptr_data = std_mem::addressof(data_);
      data_container* ptr_v_data = __CONST_CAST__(data_container*, std_mem::addressof(other.data_));
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          allocator_data_container_traits::transfer(ptr_data, ptr_v_data, 1, def_mem_type, def_mem_type, ref_stream);
        )
      );
      this->set_modified(other.is_modified());
      IvyTensorBase<T>::operator=(other);
    }

  public:
    __HOST_DEVICE__ IvyTensor() : IvyTensorBase<T>(){}
    template<typename... Args> __HOST_DEVICE__ IvyTensor(IvyTensorShape const& shape, Args&&... args) :
      IvyTensorBase<T>(),
      shape_(shape)
    {
      build_data(args...);
    }
    __HOST_DEVICE__ IvyTensor(IvyTensor const& other) :
      IvyTensorBase<T>(other),
      shape_(other.shape_)
    {
      copy_data(other);
    }
    __HOST_DEVICE__ IvyTensor(IvyTensor&& other) :
      IvyTensorBase<T>(std_util::move(other)),
      shape_(std_util::move(other.shape_)),
      data_(std_util::move(other.data_))
    {}
    __HOST_DEVICE__ ~IvyTensor(){}

    // Assignment operator
    __HOST_DEVICE__ IvyTensor& operator=(IvyTensor const& other){
      this->shape_ = other.shape_;
      copy_data(other);
      return *this;
    }
    __HOST_DEVICE__ IvyTensor& operator=(IvyTensor&& other){
      this->shape_ = std_util::move(other.shape_);
      this->data_ = std_util::move(other.data_);
      this->set_modified(other.is_modified());
      IvyTensorBase<T>::operator=(other);
      return *this;
    }
    __HOST_DEVICE__ IvyTensor& operator=(T const& val){
      data_ = std_mem::make_unique<dtype_t>(this->num_elements(), this->get_memory_type(), this->gpu_stream(), val);
      this->set_modified(true);
      return *this;
    }
    __HOST_DEVICE__ IvyTensor& operator=(T&& val){
      data_ = std_mem::make_unique<dtype_t>(this->num_elements(), this->get_memory_type(), this->gpu_stream(), std_util::move(val));
      this->set_modified(true);
      return *this;
    }

    // Get a pointer to the data
    //__HOST_DEVICE__ T* data() const{ return data_.get(); }
    __HOST_DEVICE__ T* const& data(){ return data_.get(); }

    // Access operator (for nonconst tensor)
    __HOST_DEVICE__ T& operator[](std_ilist::initializer_list<IvyTensorDim_t> const& indices){
      this->set_modified(true);
      return data_[shape_.get_abs_index(indices)];
    }
    __HOST_DEVICE__ T& operator[](std_vec::vector<IvyTensorDim_t> const& indices){
      this->set_modified(true);
      return data_[shape_.get_abs_index(indices)];
    }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ T& operator[](IvyTensorDim_t const& idx){
      this->set_modified(true);
      return data_[idx];
    }
    __HOST_DEVICE__ T& at(std_ilist::initializer_list<IvyTensorDim_t> const& indices){
      this->set_modified(true);
      return data_[shape_.get_abs_index(indices)];
    }
    __HOST_DEVICE__ T& at(std_vec::vector<IvyTensorDim_t> const& indices){
      this->set_modified(true);
      return data_[shape_.get_abs_index(indices)];
    }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ T& at(IvyTensorDim_t const& idx){
      this->set_modified(true);
      return data_[idx];
    }

    // Access operator (for const tensor)
    __HOST_DEVICE__ T const& operator[](std_ilist::initializer_list<IvyTensorDim_t> const& indices) const{ return data_[shape_.get_abs_index(indices)]; }
    __HOST_DEVICE__ T const& operator[](std_vec::vector<IvyTensorDim_t> const& indices) const{ return data_[shape_.get_abs_index(indices)]; }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ T const& operator[](IvyTensorDim_t const& idx) const{ return data_[idx]; }
    __HOST_DEVICE__ T const& at(std_ilist::initializer_list<IvyTensorDim_t> const& indices) const{ return data_[shape_.get_abs_index(indices)]; }
    __HOST_DEVICE__ T const& at(std_vec::vector<IvyTensorDim_t> const& indices) const{ return data_[shape_.get_abs_index(indices)]; }
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ T const& at(IvyTensorDim_t const& idx) const{ return data_[idx]; }

    // Ensure that a tensor can operate in just the same way as a variable or a constant.
    __INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t const& value() const{ return *this; }
    //__INLINE_FCN_FORCE__ __HOST_DEVICE__ value_t& value(){ return *this; }

    // friend classes
    friend class std_mem::kernel_generic_transfer_internal_memory<IvyTensor<T>>;
    friend struct IvyNodeSelfRelations<IvyTensor<T>>;
    friend struct tensor_data_client_updator<T>;
  };

  template<typename T> struct IvyNodeSelfRelations<IvyTensor<T>>{
    static constexpr bool is_conjugatable = is_conjugatable<T>;
    static __HOST_DEVICE__ constexpr bool is_differentiable(IvyTensor<T> const& x){
      if constexpr (is_constant_v<IvyTensor<T>>) return false;
      bool res = false;
      for (IvyTensorDim_t i=0; i<x.num_elements(); ++i){ res |= is_differentiable(x.data_[i]); if (res) break; }
      return res;
    }
    static __HOST_DEVICE__ void conjugate(IvyTensor<T>& x){
      if (!is_conjugatable) return;
      IvyTensorDim_t const n = x.num_elements();
      #define _CMD \
      for (IvyTensorDim_t i=0; i<n; ++i){ conjugate(x.data_[i]); }
#if defined(OPENMP_ENABLED)
      if (n>=NUM_CPU_THREADS_THRESHOLD){
        #pragma omp parallel for schedule(static)
        _CMD
      }
      else
#endif
      {
        _CMD
      }
      #undef _CMD
    }
  };
  template<typename T, typename U> struct IvyNodeBinaryRelations<IvyTensor<T>, U>{
    static __HOST_DEVICE__ bool depends_on(IvyTensor<T> const& fcn, U* var){
      if (!var) return false;
      bool res = std_mem::addressof(fcn) == var;
      if (!res){
        for (IvyTensorDim_t i=0; i<fcn.num_elements(); ++i){ res |= IvyMath::depends_on(fcn.at(i), var); if (res) break; }
      }
      return res;
    }
  };
}
namespace IvyTypes{
  template<typename T> struct convert_to_floating_point<IvyMath::IvyTensor<T>>{
    using type = IvyMath::IvyTensor<convert_to_floating_point_t<T>>;
  };
}
namespace IvyMath{
  template<typename T> struct fundamental_data_type<IvyTensor<T>>{
    using type = fundamental_data_t<reduced_data_t<IvyTensor<T>>>;
  };
  template<typename T> struct convert_to_floating_point_if_complex<IvyTensor<T>>{
    using type = IvyTensor<convert_to_floating_point_if_complex_t<T>>;
  };
  template<typename T> struct convert_to_real_type<IvyTensor<T>>{
    using type = IvyTensor<convert_to_real_t<T>>;
  };
  template<typename T> struct convert_to_complex_type<IvyTensor<T>>{
    using type = IvyTensor<convert_to_complex_t<T>>;
  };
  template<typename T> struct minimal_domain_type<T, tensor_domain_tag, get_operability_t<T>>{ using type = IvyTensor<std_ttraits::remove_cv_t<T>>; };

  template<typename T> using IvyTensorPtr_t = IvyThreadSafePtr_t< IvyTensor<T> >;

  template<typename T, bool is_arithmetic>
  struct tensor_data_client_updator{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void update(IvyTensorPtr_t<T>& tensor){
      for (IvyTensorDim_t i=0; i<tensor->num_elements(); ++i) tensor->data_[i].add_client(tensor);
    }
  };
  template<typename T>
  struct tensor_data_client_updator<IvyThreadSafePtr_t<T>, false>{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void update(IvyTensorPtr_t<IvyThreadSafePtr_t<T>>& tensor){
      for (IvyTensorDim_t i=0; i<tensor->num_elements(); ++i) tensor->data_[i]->add_client(tensor);
    }
  };
  template<typename T>
  struct tensor_data_client_updator<T, true>{
    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ void update(IvyTensorPtr_t<T>& tensor){}
  };


  template<typename T, typename... Args> __HOST_DEVICE__ IvyTensorPtr_t<T> Tensor(Args&&... args){
    auto res = make_IvyThreadSafePtr< IvyTensor<T> >(args...);
    tensor_data_client_updator<T>::update(res);
    return res;
  }
}
namespace std_ivy{
  template<typename T> struct value_printout<IvyMath::IvyTensor<T>>{
    static __HOST_DEVICE__ void print(IvyMath::IvyTensor<T> const& var){
      auto const& data = var.get_data_container();
      auto const n = var.num_elements();
      __PRINT_INFO__("Tensor[%llu]", n);
      print_value(data, false);
    }
  };
}


#endif
