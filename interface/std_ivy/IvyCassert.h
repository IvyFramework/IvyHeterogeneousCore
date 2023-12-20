#ifndef IVYCASSERT_H
#define IVYCASSERT_H


#ifdef __USE_CUDA__

#ifndef CUDA_RT_INCLUDED
#define CUDA_RT_INCLUDED
#include "cuda_runtime.h"
#endif

#include <cuda/std/cassert>

#else

#include <cassert>

#endif


#endif
