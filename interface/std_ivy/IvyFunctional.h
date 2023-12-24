#ifndef IVYFUNCTIONAL_H
#define IVYFUNCTIONAL_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/std/functional>

#ifndef std_fcnal
#define std_fcnal cuda::std
#endif

#else

#include <functional>

#ifndef std_fcnal
#define std_fcnal std
#endif

#endif


#endif
