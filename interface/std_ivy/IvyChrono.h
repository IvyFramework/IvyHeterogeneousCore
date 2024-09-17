#ifndef IVYCHRONO_H
#define IVYCHRONO_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/std/chrono>

#ifndef std_chrono
#define std_chrono cuda::std::chrono
#endif

#else

#include <chrono>

#ifndef std_chrono
#define std_chrono std::chrono
#define std_ratio std
#endif

#endif


#endif
