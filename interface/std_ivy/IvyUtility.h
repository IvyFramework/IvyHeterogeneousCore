#ifndef IVYUTILITY_H
#define IVYUTILITY_H


#ifdef __USE_CUDA__

#ifndef CUDA_RT_INCLUDED
#define CUDA_RT_INCLUDED
#include "cuda_runtime.h"
#endif
#include <cuda/std/utility>
#ifndef std_util
#define std_util cuda::std
#endif

#else

#include <utility>
#ifndef std_util
#define std_util std
#endif

#endif


#endif
