#ifndef IVYTENSORSHAPE_H
#define IVYTENSORSHAPE_H


#include "std_ivy/IvyCstdio.h"
#include "std_ivy/IvyFunctional.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyAlgorithm.h"
#include "std_ivy/IvyNumeric.h"
#include "std_ivy/IvyIterator.h"
#include "config/IvyCudaException.h"
#include "autodiff/IvyTensorShape.hh"


__CUDA_HOST_DEVICE__ IvyTensorShape& IvyTensorShape::operator=(IvyTensorShape const& other){
  this->rank_ = other.rank_;
  this->dims = other.dims;
  this->nel = other.nel;
  return *this;
}

__CUDA_HOST_DEVICE__ IvyTensorDim_t const& IvyTensorShape::get_dimension(IvyTensorRank_t const& iaxis) const{
  if (iaxis>=rank_) __PRINT_ERROR__("IvyTensorShape::get_dimension: Axis index %llu exceeds rank = %hu.\n", iaxis, rank_);
  return dims.at(iaxis);
}

__CUDA_HOST_DEVICE__ IvyTensorDim_t IvyTensorShape::calc_num_elements() const{
  return std_numeric::accumulate(
    std_iter::begin(dims),
    std_iter::end(dims),
    static_cast<IvyTensorDim_t>(1),
    std_fcnal::multiplies<IvyTensorDim_t>()
  );
}

__CUDA_HOST_DEVICE__ IvyTensorDim_t IvyTensorShape::get_abs_index(std_vec::vector<IvyTensorDim_t> const& indices) const{
  if (indices.size()>rank_) __PRINT_ERROR__("IvyTensorShape::get_abs_index: Number of axes = %llu exceeds rank = %hu.\n", indices.size(), rank_);
  IvyTensorDim_t res = 0;
  auto nel_tmp = nel;
  auto it_index = std_iter::begin(indices);
  auto it_end_index = std_iter::end(indices);
  auto it_dim = std_iter::begin(dims);
  IvyTensorRank_t iaxis = 0;
  while (it_index != it_end_index){
    if (*it_index >= *it_dim) __PRINT_ERROR__("IvyTensorShape::get_abs_index: Index = %llu for axis %hu exceeds number of dimensions = %llu.\n", *it_index, iaxis, *it_dim);
    nel_tmp /= *it_dim;
    res += nel_tmp*(*it_index);
    ++iaxis;
    ++it_index;
    ++it_dim;
  }
  return res;
}

__CUDA_HOST_DEVICE__ std_vec::vector<IvyTensorDim_t> IvyTensorShape::get_reordered_index_map(std_vec::vector<IvyTensorRank_t> const& reord_ax) const{
  if (reord_ax.size()!=rank_) __PRINT_ERROR__("IvyTensorShape::get_reordered_index_map: Axis indices after reordering should be a complete list.\n");

  std_vec::vector<IvyTensorDim_t> res; res.reserve(nel);
  std_vec::vector<IvyTensorDim_t> dims_reord; dims_reord.reserve(rank_);
  for (auto const& iax:reord_ax) dims_reord.emplace_back(dims.at(iax));
  std_vec::vector<IvyTensorDim_t> nels_axs; nels_axs.reserve(rank_);
  for (IvyTensorRank_t iax=0; iax<rank_; ++iax){
    if (iax==0) nels_axs.push_back(nel/dims_reord.at(iax));
    else nels_axs.push_back(nels_axs.at(iax-1)/dims_reord.at(iax));
  }
  for (IvyTensorDim_t i=0; i<nel; ++i){
    std_vec::vector<IvyTensorDim_t> idxs(rank_, 0);
    for (IvyTensorRank_t iax=0; iax<rank_; ++iax){
      auto const& iax_revord = reord_ax.at(iax);
      idxs.at(iax_revord) = (i/nels_axs.at(iax)) % dims_reord.at(iax);
    }
    res.emplace_back(get_abs_index(idxs));
  }
  return res;
}

__CUDA_HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(IvyTensorRank_t const& iaxis) const{ return get_slice_shape({ iaxis }); }
__CUDA_HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_vec::vector<IvyTensorRank_t> const& axes) const{
  for (auto const& ax:axes){
    if (ax>=rank_) __PRINT_ERROR__("Axis index = %hu exceeds rank %hu.\n", ax, rank_);
  }

  std_vec::vector<IvyTensorDim_t> dims_new; dims_new.reserve(dims.size());
  auto it_dims = std_iter::begin(dims);
  for (IvyTensorRank_t iaxis=0; iaxis<rank_; ++iaxis){
    if (std_algo::find(std_iter::begin(axes), std_iter::end(axes), iaxis)==std_iter::end(axes)) dims_new.emplace_back(*it_dims);
  }

  return IvyTensorShape(dims_new);
}

__CUDA_HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_util::pair<IvyTensorRank_t, IvyTensorDim_t> const& sp) const{ return get_slice_shape({ sp.first }); }
__CUDA_HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_slice_shape(std_vec::vector<std_util::pair<IvyTensorRank_t, IvyTensorDim_t>> const& sps) const{
  std_vec::vector<IvyTensorRank_t> axes; axes.reserve(sps.size());
  for (auto const& sp:sps) axes.emplace_back(sp.first);
  return get_slice_shape(axes);
}

__CUDA_HOST_DEVICE__ IvyTensorShape IvyTensorShape::get_contraction_shape(
  IvyTensorShape const& s1, IvyTensorShape const& s2,
  std_vec::vector<std_util::pair<IvyTensorRank_t, IvyTensorRank_t>> const& contraction_axis_pairs,
  IvyTensorDim_t* uncontracted_nel_s1, IvyTensorDim_t* uncontracted_nel_s2,
  std_vec::vector<IvyTensorDim_t>* idx_reord_s1, std_vec::vector<IvyTensorDim_t>* idx_reord_s2
){
  auto const& s1_dims = s1.get_dimensions();
  auto const& s2_dims = s2.get_dimensions();

  // Do some checks
  for (auto const& cax:contraction_axis_pairs){
    if (cax.first>=s1_dims.size()) __PRINT_ERROR__(
      "IvyTensorShape::get_contraction_shape: Axis index = %llu of tensor 1 exceeds rank = %hu.\n", cax.first, s1_dims.size()
    );
    if (cax.second>=s2_dims.size()) __PRINT_ERROR__(
      "IvyTensorShape::get_contraction_shape: Axis index = %llu of tensor 2 exceeds rank = %hu.\n", cax.second, s2_dims.size()
    );
    if (s1_dims.at(cax.first)!=s2_dims.at(cax.second)) __PRINT_ERROR__(
      "IvyTensorShape::get_contraction_shape: Number of dimensions are not equal for axes [%llu, %llu], i.e., %llu != %llu.",
      cax.first, cax.second, s1_dims.at(cax.first), s2_dims.at(cax.second)
    );
  }

  if (uncontracted_nel_s1) *uncontracted_nel_s1 = 1;
  if (uncontracted_nel_s2) *uncontracted_nel_s2 = 1;
  if (idx_reord_s1) idx_reord_s1->clear();
  if (idx_reord_s2) idx_reord_s2->clear();

  // Get the new dimensions
  std_vec::vector<IvyTensorRank_t> aord_s1, aord_s2; aord_s1.reserve(s1.rank()); aord_s2.reserve(s2.rank());
  std_vec::vector<IvyTensorDim_t> dims; dims.reserve(s1_dims.size() + s2_dims.size());
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

  return IvyTensorShape(dims);
}

__CUDA_HOST_DEVICE__ void IvyTensorShape::print() const{
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


#endif
