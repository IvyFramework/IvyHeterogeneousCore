#ifndef IVYLIMITS_H
#define IVYLIMITS_H


#ifdef __USE_CUDA__

#ifndef CUDA_RT_INCLUDED
#define CUDA_RT_INCLUDED
#include "cuda_runtime.h"
#endif
#include <cuda/std/limits>
#ifndef std_limits
#define std_limits cuda::std
#endif

#else

#include <limits>
#ifndef std_limits
#define std_limits std
#endif

#endif


#endif
