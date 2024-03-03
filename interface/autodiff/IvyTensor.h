#ifndef IVYTENSOR_H
#define IVYTENSOR_H

#include "stream/IvyStream.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyMemory.h"
#include "autodiff/IvyBaseTensor.h"
#include "autodiff/IvyTensorShape.h"


template<typename T> class IvyTensor;
template<typename T> struct IvyNodeSelfRelations<IvyTensor<T>>;
template<typename T, typename U> struct IvyNodeBinaryRelations<IvyTensor<T>, U>;

template<typename T> class IvyTensor : public IvyBaseTensor{
public:
  using dtype_t = T;
  using value_t = IvyTensor<T>;

protected:
  IvyTensorShape def_dims;
  IvyThreadSafePtr_t<dtype_t> vals;

  __CUDA_HOST_DEVICE__ bool transfer_internal_memory(std_ivy::IvyMemoryType const& new_mem_type, bool release_old);

public:
  // Convenience accessors from IvyTensorShape
  __CUDA_HOST_DEVICE__ IvyTensorShape const& shape() const{ return def_dims; }
  __CUDA_HOST_DEVICE__ IvyTensorRank_t const& rank() const{ return def_dims.rank(); }
  __CUDA_HOST_DEVICE__ IvyTensorDim_t const& num_elements() const{ return def_dims.num_elements(); }
  __CUDA_HOST_DEVICE__ std_vec::vector<IvyTensorDim_t> const& get_dimensions() const{ return def_dims.get_dimensions(); }
  __CUDA_HOST_DEVICE__ IvyTensorDim_t const& get_dimension(IvyTensorRank_t const& iaxis) const{ return def_dims.get_dimension(iaxis); }

  // Get (number of bytes) = (number of elements)*(size of the data type in bytes).
  __CUDA_HOST_DEVICE__ IvyTensorDim_t num_bytes() const{ return num_elements()*sizeof(T); }

protected:
  __CUDA_HOST_DEVICE__ void initialize_data(T const& defval){ if (vals){ for (IvyTensorDim_t i=0; i<this->num_elements(); ++i) vals[i] = defval; } }
  __CUDA_HOST_DEVICE__ void allocate_data(){ IvyMemoryHelpers::allocate_memory(vals, num_elements()); }
  __CUDA_HOST_DEVICE__ void copy_data(IvyTensor const& other){ IvyMemoryHelpers::copy_data(vals, other.vals, this->num_elements(), other.num_elements()); }

public:
  __CUDA_HOST_DEVICE__ IvyTensor(){}
  __CUDA_HOST_DEVICE__ IvyTensor(IvyTensorShape const& shape) : def_dims(shape){ allocate_data(); }
  __CUDA_HOST_DEVICE__ IvyTensor(IvyTensorShape const& shape, T const& defval) : def_dims(shape){ allocate_data(); initialize_data(defval); }
  __CUDA_HOST_DEVICE__ IvyTensor(IvyTensor const& other) : def_dims(other.def_dims){ copy_data(other); }
  __CUDA_HOST_DEVICE__ IvyTensor(IvyTensor&& other) : def_dims(std_util::move(other.def_dims)), vals(std_util::move(other.vals)){}
  __CUDA_HOST_DEVICE__ ~IvyTensor(){}

  // Assignment operator
  __CUDA_HOST_DEVICE__ IvyTensor& operator=(IvyTensor const& other){
    copy_data(other);
    this->def_dims = other.def_dims;
    return *this;
  }

  // Get memory type and stream
  __CUDA_HOST_DEVICE__ std_ivy::IvyMemoryType get_memory_type() const{ return def_dims.get_memory_type(); }
  __CUDA_HOST_DEVICE__ IvyGPUStream* gpu_stream() const{ return def_dims.gpu_stream(); }

  // Get a pointer to the data
  __CUDA_HOST_DEVICE__ T* const& data() const{ return vals.get(); }

  // Access operator (for nonconst tensor)
  __CUDA_HOST_DEVICE__ T& operator[](std_vec::vector<IvyTensorDim_t> const& indices){ return vals[def_dims.get_abs_index(indices)]; }
  __CUDA_HOST_DEVICE__ T& operator[](IvyTensorDim_t const& idx){ return (*this)[{idx}]; }

  // Access operator (for const tensor)
  __CUDA_HOST_DEVICE__ T const& operator[](std_vec::vector<IvyTensorDim_t> const& indices) const{ return vals[def_dims.get_abs_index(indices)]; }
  __CUDA_HOST_DEVICE__ T const& operator[](IvyTensorDim_t const& idx) const{ return (*this)[{idx}]; }

  // Ensure that a tensor can operate in just the same way as a variable or a constant.
  __CUDA_HOST_DEVICE__ value_t const& value() const{ return *this; }

  friend class std_mem::kernel_generic_transfer_internal_memory<IvyTensor<T>>;

  friend struct IvyNodeSelfRelations<IvyTensor<T>>;
  template<typename U> friend struct IvyNodeBinaryRelations<IvyTensor<T>, U>;
};

template<typename T> struct IvyNodeSelfRelations<IvyTensor<T>>{
  static __CUDA_HOST_DEVICE__ bool is_differentiable(IvyTensor<T> const& x){
    bool res = false;
    for (IvyTensorDim_t i=0; i<x.num_elements(); ++i){ res |= is_differentiable(x.vals[i]); if (res) break; }
    return res;
  }
  static __CUDA_HOST_DEVICE__ void conjugate(IvyTensor<T>& x){
    if (!is_conjugatable<T>) return;
    for (IvyTensorDim_t i=0; i<x.num_elements(); ++i){ conjugate(x.vals[i]); }
  }
  static constexpr bool is_conjugatable = is_conjugatable<T>;
};
template<typename T, typename U> struct IvyNodeBinaryRelations<IvyTensor<T>, U>{
  static __CUDA_HOST_DEVICE__ bool depends_on(IvyTensor<T> const& fcn, U const& var){
    bool res = std_mem::addressof(fcn) == std_mem::addressof(var);
    if (!res){
      for (IvyTensorDim_t i=0; i<fcn.num_elements(); ++i){ res |= depends_on(fcn.vals[i], var); if (res) break; }
    }
    return res;
  }
};

template<typename T> using IvyTensorPtr_t = IvyThreadSafePtr_t< IvyTensor<T> >;

template<typename T, typename... Args> __CUDA_HOST_DEVICE__ IvyTensorPtr_t<T> Tensor(std_ivy::IvyMemoryType const& mem_type, IvyGPUStream* stream, Args&&... args){ return make_IvyThreadSafePtr< IvyTensor<T> >(mem_type, stream, args...); }


#endif
