/**
 * @file IvyLimits.h
 * @brief Backend-selecting numeric limits wrapper.
 */
#ifndef IVYLIMITS_H
#define IVYLIMITS_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
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
