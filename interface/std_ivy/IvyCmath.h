#ifndef IVYCMATH_H
#define IVYCMATH_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/std/cmath>

#ifndef std_math
#define std_math cuda::std
#endif

#else

#include <cmath>

#ifndef std_math
#define std_math std
#endif

#endif


#endif
