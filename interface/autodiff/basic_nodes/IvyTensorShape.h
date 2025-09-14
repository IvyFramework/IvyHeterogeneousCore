#ifndef IVYTENSORSHAPE_H
#define IVYTENSORSHAPE_H


#include "config/IvyCudaException.h"
#include "std_ivy/IvyCstdio.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyAlgorithm.h"
#include "std_ivy/IvyNumeric.h"
#include "std_ivy/IvyIterator.h"
#include "autodiff/basic_nodes/IvyTensorShape.hh"


namespace IvyMath{
  __HOST_DEVICE__ IvyTensorShape& IvyTensorShape::operator=(IvyTensorShape const& other){
    this->rank_ = other.rank_;
    this->dims = other.dims;
    this->nel = other.nel;
    return *this;
  }
  __HOST_DEVICE__ IvyTensorShape& IvyTensorShape::operator=(IvyTensorShape&& other){
    this->rank_ = std_util::move(other.rank_);
    this->dims = std_util::move(other.dims);
    this->nel = std_util::move(other.nel);
    return *this;
  }
  __HOST_DEVICE__ void IvyTensorShape::swap(IvyTensorShape& other){
    std_util::swap(this->rank_, other.rank_);
    std_util::swap(this->dims, other.dims);
    std_util::swap(this->nel, other.nel);
  }

  __HOST_DEVICE__ bool IvyTensorShape::transfer_internal_memory(std_ivy::IvyMemoryType const& new_mem_type, bool release_old){
    bool res = true;
    constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto stream = dims.gpu_stream();
    operate_with_GPU_stream_from_pointer(
      stream, ref_stream,
      __ENCAPSULATE__(
        res &= allocator_data_container::transfer_internal_memory(&dims, 1, def_mem_type, new_mem_type, ref_stream, release_old);
    )
    );
    return res;
  }


  __HOST_DEVICE__ IvyTensorDim_t const& IvyTensorShape::get_dimension(IvyTensorRank_t const& iaxis) const{
    if (iaxis>=rank_) __PRINT_ERROR__("IvyTensorShape::get_dimension: Axis index %hu exceeds rank = %hu.\n", iaxis, rank_);
    return dims.at(iaxis);
  }

  __HOST_DEVICE__ IvyTensorDim_t IvyTensorShape::calc_num_elements() const{
    if (dims.empty()) return 0;
    constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = this->get_memory_type();
    auto stream = dims.gpu_stream();

    IvyTensorDim_t res = 0;

    if (mem_type==def_mem_type) res = std_numeric::accumulate(
      std_iter::begin(dims),
      std_iter::end(dims),
      __STATIC_CAST__(IvyTensorDim_t, 1),
      std_fcnal::multiplies<IvyTensorDim_t>()
    );
    else{
      data_container h_dims = dims;
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          allocator_data_container::transfer_internal_memory(&h_dims, 1, def_mem_type, def_mem_type, ref_stream, true);
          res = std_numeric::accumulate(
            std_iter::begin(h_dims),
            std_iter::end(h_dims),
            __STATIC_CAST__(IvyTensorDim_t, 1),
            std_fcnal::multiplies<IvyTensorDim_t>()
          );
        )
      );
    }

    return res;
  }

  __HOST_DEVICE__ IvyTensorDim_t IvyTensorShape::get_abs_index(std_vec::vector<IvyTensorDim_t> const& indices) const{
    if (indices.size()>rank_) __PRINT_ERROR__("IvyTensorShape::get_abs_index: Number of axes = %llu exceeds rank = %hu.\n", indices.size(), rank_);
    constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = this->get_memory_type();
    auto stream = dims.gpu_stream();

    build_GPU_stream_reference_from_pointer(stream, ref_stream);

    data_container ref_dims = dims;
    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&ref_dims, 1, def_mem_type, def_mem_type, ref_stream, true);

    data_container* h_indices = nullptr;
    if (indices.get_memory_type()!=def_mem_type){
      allocator_data_container_traits::allocate(h_indices, 1, def_mem_type, ref_stream);
      auto tmp_indices = indices;
      allocator_data_container_traits::transfer(h_indices, &tmp_indices, 1, def_mem_type, def_mem_type, ref_stream);
    }
    data_container const& ref_indices = (h_indices ? *h_indices : indices);

    IvyTensorDim_t res = 0;
    IvyTensorRank_t iaxis = 0;
    auto nel_tmp = nel;
    auto it_index = std_iter::begin(ref_indices);
    auto it_end_index = std_iter::end(ref_indices);
    auto it_dim = std_iter::begin(ref_dims);
    while (it_index != it_end_index){
      if (*it_index >= *it_dim) __PRINT_ERROR__("IvyTensorShape::get_abs_index: Index = %llu for axis %hu exceeds number of dimensions = %llu.\n", *it_index, iaxis, *it_dim);
      nel_tmp /= *it_dim;
      res += nel_tmp*(*it_index);
      ++iaxis;
      ++it_index;
      ++it_dim;
    }

    if (h_indices) allocator_data_container_traits::destroy(h_indices, 1, def_mem_type, ref_stream);

    destroy_GPU_stream_reference_from_pointer(stream);

    return res;
  }
  __HOST_DEVICE__ IvyTensorDim_t IvyTensorShape::get_abs_index(std_ilist::initializer_list<IvyTensorDim_t> const& indices) const{
    if (indices.size()>rank_) __PRINT_ERROR__("IvyTensorShape::get_abs_index: Number of axes = %llu exceeds rank = %hu.\n", indices.size(), rank_);

    constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = this->get_memory_type();
    auto stream = dims.gpu_stream();

    build_GPU_stream_reference_from_pointer(stream, ref_stream);

    data_container ref_dims = dims;
    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&ref_dims, 1, def_mem_type, def_mem_type, ref_stream, true);

    IvyTensorDim_t res = 0;
    IvyTensorRank_t iaxis = 0;
    auto nel_tmp = nel;
    auto it_index = std_iter::begin(indices);
    auto it_end_index = std_iter::end(indices);
    auto it_dim = std_iter::begin(ref_dims);
    while (it_index != it_end_index){
      if (*it_index >= *it_dim) __PRINT_ERROR__("IvyTensorShape::get_abs_index: Index = %llu for axis %hu exceeds number of dimensions = %llu.\n", *it_index, iaxis, *it_dim);
      nel_tmp /= *it_dim;
      res += nel_tmp*(*it_index);
      ++iaxis;
      ++it_index;
      ++it_dim;
    }

    destroy_GPU_stream_reference_from_pointer(stream);

    return res;
  }

  __HOST_DEVICE__ std_vec::vector<IvyTensorDim_t> IvyTensorShape::get_reordered_index_map(std_vec::vector<IvyTensorRank_t> const& reord_ax) const{
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = this->get_memory_type();
    auto stream = this->gpu_stream();

    if (reord_ax.size()!=rank_) __PRINT_ERROR__("IvyTensorShape::get_reordered_index_map: Axis indices after reordering should be a complete list.\n");

    build_GPU_stream_reference_from_pointer(stream, ref_stream);

    data_container ref_dims = dims;
    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&ref_dims, 1, def_mem_type, def_mem_type, ref_stream, true);

    rank_container* h_reord_ax = nullptr;
    if (reord_ax.get_memory_type()!=def_mem_type){
      allocator_rank_container_traits::allocate(h_reord_ax, 1, def_mem_type, ref_stream);
      auto tmp_reord_ax = reord_ax;
      allocator_rank_container_traits::transfer(h_reord_ax, &tmp_reord_ax, 1, def_mem_type, def_mem_type, ref_stream);
    }
    auto const& ref_reord_ax = (h_reord_ax ? *h_reord_ax : reord_ax);

    std_vec::vector<IvyTensorDim_t> res; res.reserve(nel, def_mem_type, stream);
    {
      std_vec::vector<IvyTensorDim_t> dims_reord; dims_reord.reserve(rank_, def_mem_type, stream);
      for (auto const& iax:ref_reord_ax) dims_reord.emplace_back(ref_dims.at(iax));
      std_vec::vector<IvyTensorDim_t> nels_axs; nels_axs.reserve(rank_, def_mem_type, stream);
      for (IvyTensorRank_t iax=0; iax<rank_; ++iax){
        if (iax==0) nels_axs.push_back(nel/dims_reord.at(iax));
        else nels_axs.push_back(nels_axs.at(iax-1)/dims_reord.at(iax));
      }
      for (IvyTensorDim_t i=0; i<nel; ++i){
        std_vec::vector<IvyTensorDim_t> idxs(rank_, def_mem_type, stream, 0);
        #define _CMD \
        for (IvyTensorRank_t iax=0; iax<rank_; ++iax){ \
          auto const& iax_revord = ref_reord_ax.at(iax); \
          idxs.at(iax_revord) = (i/nels_axs.at(iax)) % dims_reord.at(iax); \
        }
#if defined(OPENMP_ENABLED)
        if (rank_>=NUM_CPU_THREADS_THRESHOLD){
          #pragma omp parallel for schedule(static)
          _CMD
        }
        else
#endif
        {
          _CMD
        }
        #undef _CMD
        res.emplace_back(get_abs_index(idxs));
      }
    }

    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&res, 1, def_mem_type, mem_type, ref_stream, true);
    if (h_reord_ax) allocator_rank_container_traits::destroy(h_reord_ax, 1, def_mem_type, ref_stream);

    destroy_GPU_stream_reference_from_pointer(stream);

    return res;
  }

  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(IvyTensorRank_t const& iaxis) const{ return get_slice_shape({ iaxis }); }
  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_vec::vector<IvyTensorRank_t> const& axes) const{
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = this->get_memory_type();
    auto stream = this->gpu_stream();

    build_GPU_stream_reference_from_pointer(stream, ref_stream);

    data_container ref_dims = dims;
    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&ref_dims, 1, def_mem_type, def_mem_type, ref_stream, true);

    rank_container* h_axes = nullptr;
    if (axes.get_memory_type()!=def_mem_type){
      allocator_rank_container_traits::allocate(h_axes, 1, def_mem_type, ref_stream);
      auto tmp_axes = axes;
      allocator_rank_container_traits::transfer(h_axes, &tmp_axes, 1, def_mem_type, def_mem_type, ref_stream);
    }
    auto const& ref_axes = (h_axes ? *h_axes : axes);

    for (auto const& ax:ref_axes){
      if (ax>=rank_) __PRINT_ERROR__("Axis index = %hu exceeds rank %hu.\n", ax, rank_);
    }

    std_vec::vector<IvyTensorDim_t> dims_new; dims_new.reserve(ref_dims.size(), def_mem_type, stream);
    auto it_dims = std_iter::begin(ref_dims);
    for (IvyTensorRank_t iaxis=0; iaxis<rank_; ++iaxis){
      if (std_algo::find(std_iter::begin(ref_axes), std_iter::end(ref_axes), iaxis)==std_iter::end(ref_axes)) dims_new.emplace_back(*it_dims);
      ++it_dims;
    }

    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&dims_new, 1, def_mem_type, mem_type, ref_stream, true);
    if (h_axes) allocator_rank_container_traits::destroy(h_axes, 1, def_mem_type, ref_stream);

    destroy_GPU_stream_reference_from_pointer(stream);

    return IvyTensorShape(dims_new);
  }
  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_ilist::initializer_list<IvyTensorRank_t> const& axes) const{
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = this->get_memory_type();
    auto stream = this->gpu_stream();

    build_GPU_stream_reference_from_pointer(stream, ref_stream);

    data_container ref_dims = dims;
    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&ref_dims, 1, def_mem_type, def_mem_type, ref_stream, true);

    for (auto const& ax:axes){
      if (ax>=rank_) __PRINT_ERROR__("Axis index = %hu exceeds rank %hu.\n", ax, rank_);
    }

    std_vec::vector<IvyTensorDim_t> dims_new; dims_new.reserve(ref_dims.size(), def_mem_type, stream);
    auto it_dims = std_iter::begin(ref_dims);
    for (IvyTensorRank_t iaxis=0; iaxis<rank_; ++iaxis){
      if (std_algo::find(std_iter::begin(axes), std_iter::end(axes), iaxis)==std_iter::end(axes)) dims_new.emplace_back(*it_dims);
      ++it_dims;
    }

    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&dims_new, 1, def_mem_type, mem_type, ref_stream, true);

    destroy_GPU_stream_reference_from_pointer(stream);

    return IvyTensorShape(dims_new);
  }

  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_util::pair<IvyTensorRank_t, IvyTensorDim_t> const& sp) const{ return get_slice_shape({ sp.first }); }
  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_ilist::initializer_list<std_util::pair<IvyTensorRank_t, IvyTensorDim_t>> const& sps) const{
    std_vec::vector<IvyTensorRank_t> axes; axes.reserve(sps.size(), IvyMemoryHelpers::get_execution_default_memory(), this->gpu_stream());
    for (auto const& sp:sps) axes.emplace_back(sp.first);
    return get_slice_shape(axes);
  }
  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_vec::vector<std_util::pair<IvyTensorRank_t, IvyTensorDim_t>> const& sps) const{
    std_vec::vector<IvyTensorRank_t> axes; axes.reserve(sps.size(), IvyMemoryHelpers::get_execution_default_memory(), this->gpu_stream());
    for (auto const& sp:sps) axes.emplace_back(sp.first);
    return get_slice_shape(axes);
  }

  __HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_contraction_shape(
    IvyTensorShape const& s1, IvyTensorShape const& s2,
    std_vec::vector<std_util::pair<IvyTensorRank_t, IvyTensorRank_t>> const& contraction_axis_pairs,
    IvyTensorDim_t* uncontracted_nel_s1, IvyTensorDim_t* uncontracted_nel_s2,
    std_vec::vector<IvyTensorDim_t>* idx_reord_s1, std_vec::vector<IvyTensorDim_t>* idx_reord_s2
  ){
    constexpr std_ivy::IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
    auto const mem_type = s1.get_memory_type();
    auto stream = s1.gpu_stream();
    if (!stream) stream = s2.gpu_stream();

    // Do some checks
    if (mem_type!=s2.get_memory_type()) __PRINT_ERROR__("IvyTensorShape::get_contraction_shape: Memory type of the two shapes are not the same.\n");

    build_GPU_stream_reference_from_pointer(stream, ref_stream);

    auto s1_dims = s1.get_dimensions();
    auto s2_dims = s2.get_dimensions();
    if (mem_type!=def_mem_type){
      allocator_data_container::transfer_internal_memory(&s1_dims, 1, def_mem_type, def_mem_type, ref_stream, true);
      allocator_data_container::transfer_internal_memory(&s2_dims, 1, def_mem_type, def_mem_type, ref_stream, true);
    }

    for (auto const& cax:contraction_axis_pairs){
      if (cax.first>=s1_dims.size()) __PRINT_ERROR__(
        "IvyTensorShape::get_contraction_shape: Axis index = %hu of tensor 1 exceeds rank = %llu.\n", cax.first, s1_dims.size()
      );
      if (cax.second>=s2_dims.size()) __PRINT_ERROR__(
        "IvyTensorShape::get_contraction_shape: Axis index = %hu of tensor 2 exceeds rank = %llu.\n", cax.second, s2_dims.size()
      );
      if (s1_dims.at(cax.first)!=s2_dims.at(cax.second)) __PRINT_ERROR__(
        "IvyTensorShape::get_contraction_shape: Number of dimensions are not equal for axes [%hu, %hu], i.e., %llu != %llu.\n",
        cax.first, cax.second, s1_dims.at(cax.first), s2_dims.at(cax.second)
      );
    }

    if (uncontracted_nel_s1) *uncontracted_nel_s1 = 1;
    if (uncontracted_nel_s2) *uncontracted_nel_s2 = 1;
    if (idx_reord_s1) idx_reord_s1->clear();
    if (idx_reord_s2) idx_reord_s2->clear();

    // Get the new dimensions
    std_vec::vector<IvyTensorRank_t> aord_s1, aord_s2; aord_s1.reserve(s1.rank(), def_mem_type, stream); aord_s2.reserve(s2.rank(), def_mem_type, stream);
    std_vec::vector<IvyTensorDim_t> dims; dims.reserve(s1_dims.size() + s2_dims.size(), def_mem_type, stream);
    IvyTensorRank_t iaxis = 0;
    for (auto const& dim:s1_dims){
      bool included = true;
      for (auto const& cax:contraction_axis_pairs){
        if (iaxis == cax.first){
          included = false;
          break;
        }
      }
      if (included){
        dims.emplace_back(dim);
        aord_s1.push_back(iaxis);
        if (uncontracted_nel_s1) *uncontracted_nel_s1 *= dim;
      }
      ++iaxis;
    }
    iaxis = 0;
    for (auto const& dim:s2_dims){
      bool included = true;
      for (auto const& cax:contraction_axis_pairs){
        if (iaxis == cax.second){
          included = false;
          break;
        }
      }
      if (included){
        dims.emplace_back(dim);
        aord_s2.push_back(iaxis);
        if (uncontracted_nel_s2) *uncontracted_nel_s2 *= dim;
      }
      ++iaxis;
    }
    for (auto const& cax:contraction_axis_pairs){
      aord_s1.push_back(cax.first);
      aord_s2.push_back(cax.second);
    }

    if (idx_reord_s1) *idx_reord_s1 = s1.get_reordered_index_map(aord_s1);
    if (idx_reord_s2) *idx_reord_s2 = s2.get_reordered_index_map(aord_s2);

    if (mem_type!=def_mem_type) allocator_data_container::transfer_internal_memory(&dims, 1, def_mem_type, mem_type, ref_stream, true);

    destroy_GPU_stream_reference_from_pointer(stream);

    return IvyTensorShape(dims);
  }

  __HOST_DEVICE__ void IvyTensorShape::print() const{
#ifndef __CUDA_DEVICE_CODE__
    __PRINT_INFO__("IvyTensorShape rank = %hu with dimensions {", rank_);
    bool first = true;
    for (auto const& dim:dims){
      if (!first) __PRINT_INFO__(",");
      __PRINT_INFO__(" %llu", dim);
      first = false;
    }
    __PRINT_INFO__((rank_>0 ? " " : ""));
    __PRINT_INFO__("}\n");
#endif
  }
}

namespace std_util{
  void swap(IvyMath::IvyTensorShape& a, IvyMath::IvyTensorShape& b){ a.swap(b); }
}


#endif
