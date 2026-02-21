/**
 * @file IvyAtomic.h
 * @brief Backend-selecting atomic wrapper that aliases CUDA or standard atomic facilities.
 */
#ifndef IVYATOMIC_H
#define IVYATOMIC_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/atomic>

#ifndef std_atomic
#define std_atomic cuda
#endif

#else

#include <atomic>

#ifndef std_atomic
#define std_atomic std
#endif

#endif


#endif
