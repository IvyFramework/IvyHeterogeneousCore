#ifndef IVYTENSORSHAPE_HH
#define IVYTENSORSHAPE_HH


#include "config/IvyCompilerConfig.h"
#include "stream/IvyStream.h"
#include "std_ivy/IvyInitializerList.h"
#include "std_ivy/IvyUtility.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyMemory.h"
#include "IvyBasicTypes.h"


namespace IvyMath{
  typedef unsigned short IvyTensorRank_t;
  typedef IvyTypes::size_t IvyTensorDim_t;

  /*
  IvyTensorShape: This is a class to define the shape of any tensor.
  The 'rank_' variable is the programmer's definition of rank, i.e., number of axes of the tensor.
  The 'dims' variable holds the number of bins over each axis.

  Use the various 'get_slice_shape' functions to get the shape of a slice of the tensor.
  Use the 'get_contraction_shape' function to get the dimensions of a contraction of two tensors.
  - In this context, note that a matrix multiplication is simply the contraction of two rank=2 tensors with over a common axis.
  */
  class IvyTensorShape{
  public:
    typedef std_vec::vector<IvyTensorDim_t> data_container;
    typedef std_mem::allocator<data_container> allocator_data_container;
    typedef std_mem::allocator_traits<allocator_data_container> allocator_data_container_traits;
    typedef std_vec::vector<IvyTensorRank_t> rank_container;
    typedef std_mem::allocator<rank_container> allocator_rank_container;
    typedef std_mem::allocator_traits<allocator_rank_container> allocator_rank_container_traits;

  protected:
    IvyTensorRank_t rank_; // Number of dimension indices per dimension
    data_container dims; // Range of indices per dimension
    IvyTensorDim_t nel; // Cached number of elements

    // Calculate the number of elements
    __CUDA_HOST_DEVICE__ IvyTensorDim_t calc_num_elements() const;

    __CUDA_HOST_DEVICE__ bool transfer_internal_memory(std_ivy::IvyMemoryType const& new_mem_type, bool release_old);

  public:
    __CUDA_HOST_DEVICE__ IvyTensorShape() : rank_(0), nel(0){}
    __CUDA_HOST_DEVICE__ IvyTensorShape(std_vec::vector<IvyTensorDim_t> const& dims_) : rank_(dims_.size()), dims(dims_), nel(this->calc_num_elements()){}
    __CUDA_HOST_DEVICE__ IvyTensorShape(std_ilist::initializer_list<IvyTensorDim_t> const& dims_, std_ivy::IvyMemoryType mem_type, IvyGPUStream* stream) : rank_(dims_.size()), dims(dims_, mem_type, stream), nel(this->calc_num_elements()){}
    __CUDA_HOST_DEVICE__ IvyTensorShape(std_ilist::initializer_list<IvyTensorDim_t> const& dims_) : rank_(dims_.size()), dims(dims_, IvyMemoryHelpers::get_execution_default_memory(), nullptr), nel(this->calc_num_elements()){}
    __CUDA_HOST_DEVICE__ IvyTensorShape(IvyTensorShape const& other) : rank_(other.rank_), dims(other.dims), nel(other.nel){}
    __CUDA_HOST_DEVICE__ IvyTensorShape(IvyTensorShape&& other) : rank_(std_util::move(other.rank_)), dims(std_util::move(other.dims)), nel(std_util::move(other.nel)){}
    __CUDA_HOST_DEVICE__ ~IvyTensorShape(){}

    // Assignment operator
    __CUDA_HOST_DEVICE__ IvyTensorShape& operator=(IvyTensorShape const& other);
    __CUDA_HOST_DEVICE__ IvyTensorShape& operator=(IvyTensorShape&& other);

    // Swap operation
    __CUDA_HOST_DEVICE__ void swap(IvyTensorShape& other);

    // Get memory type and stream
    __CUDA_HOST_DEVICE__ std_ivy::IvyMemoryType get_memory_type() const{ return dims.get_memory_type(); }
    __CUDA_HOST_DEVICE__ IvyGPUStream* gpu_stream() const{ return dims.gpu_stream(); }

    // Get the total number of elements, i.e., product of the elements of the 'dims' vector
    __CUDA_HOST_DEVICE__ IvyTensorDim_t const& num_elements() const{ return nel; }

    // Various get functions
    __CUDA_HOST_DEVICE__ IvyTensorRank_t const& rank() const{ return rank_; }
    __CUDA_HOST_DEVICE__ std_vec::vector<IvyTensorDim_t> const& get_dimensions() const{ return dims; }
    __CUDA_HOST_DEVICE__ IvyTensorDim_t const& get_dimension(IvyTensorRank_t const& iaxis) const;

    // Get absolute index given an ordered set of indices for each axis
    __CUDA_HOST_DEVICE__ IvyTensorDim_t get_abs_index(std_vec::vector<IvyTensorDim_t> const& indices) const;
    __CUDA_HOST_DEVICE__ IvyTensorDim_t get_abs_index(std_ilist::initializer_list<IvyTensorDim_t> const& indices) const;

    // Get map of indices after reordering axes
    __CUDA_HOST_DEVICE__ std_vec::vector<IvyTensorDim_t> get_reordered_index_map(std_vec::vector<IvyTensorRank_t> const& reord_ax) const;

    // Get sliced shape along the specified axes at particular index positions.
    __CUDA_HOST_DEVICE__ IvyTensorShape get_slice_shape(IvyTensorRank_t const& axis) const;
    __CUDA_HOST_DEVICE__ IvyTensorShape get_slice_shape(std_vec::vector<IvyTensorRank_t> const& axes) const;
    __CUDA_HOST_DEVICE__ IvyTensorShape get_slice_shape(std_ilist::initializer_list<IvyTensorRank_t> const& axes) const;

    // Convenience functions calling the versions above
    __CUDA_HOST_DEVICE__ IvyTensorShape get_slice_shape(std_util::pair<IvyTensorRank_t, IvyTensorDim_t> const& sp) const;
    __CUDA_HOST_DEVICE__ IvyTensorShape get_slice_shape(std_ilist::initializer_list<std_util::pair<IvyTensorRank_t, IvyTensorDim_t>> const& sps) const;
    __CUDA_HOST_DEVICE__ IvyTensorShape get_slice_shape(std_vec::vector<std_util::pair<IvyTensorRank_t, IvyTensorDim_t>> const& sps) const;

    // Get the shape of a tensor resulting from the contraction of two tensors
    // Notice that the order of uncontracted axes are preserved, and the new axes are ordered as those from s1 first, and then those from s2.
    static __CUDA_HOST_DEVICE__ IvyTensorShape get_contraction_shape(
      IvyTensorShape const& s1, IvyTensorShape const& s2,
      std_vec::vector<std_util::pair<IvyTensorRank_t, IvyTensorRank_t>> const& contraction_axis_pairs,
      IvyTensorDim_t* uncontracted_nel_s1 = nullptr, IvyTensorDim_t* uncontracted_nel_s2 = nullptr,
      std_vec::vector<IvyTensorDim_t>* idx_reord_s1 = nullptr, std_vec::vector<IvyTensorDim_t>* idx_reord_s2 = nullptr
    );
    static __CUDA_HOST_DEVICE__ IvyTensorShape get_contraction_shape(
      IvyTensorShape const& s1, IvyTensorShape const& s2,
      std_util::pair<IvyTensorRank_t, IvyTensorRank_t> const& contraction_axes,
      IvyTensorDim_t* uncontracted_nel_s1 = nullptr, IvyTensorDim_t* uncontracted_nel_s2 = nullptr,
      std_vec::vector<IvyTensorDim_t>* idx_reord_s1 = nullptr, std_vec::vector<IvyTensorDim_t>* idx_reord_s2 = nullptr
    ){
      return get_contraction_shape(s1, s2, { contraction_axes }, uncontracted_nel_s1, uncontracted_nel_s2, idx_reord_s1, idx_reord_s2);
    }

    // Print the shape
    void __CUDA_HOST_DEVICE__ print() const;

    friend class std_mem::kernel_generic_transfer_internal_memory<IvyTensorShape>;
  };
}
namespace std_util{
  void swap(IvyMath::IvyTensorShape& a, IvyMath::IvyTensorShape& b);
}


#endif
