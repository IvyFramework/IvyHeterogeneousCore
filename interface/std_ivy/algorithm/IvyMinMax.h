#ifndef IVYMINMAX_H
#define IVYMINMAX_H


#include "config/IvyCompilerConfig.h"

#ifdef __USE_CUDA__

namespace std_ivy{
  template<typename T> __CUDA_HOST_DEVICE__ T const& min(T const& x, T const& y){ return (x>y ? y : x); }
  template<typename T> __CUDA_HOST_DEVICE__ T const& max(T const& x, T const& y){ return (x>y ? x : y); }
}

#endif


#endif
