/**
 * @file IvyUtility.h
 * @brief Backend-selecting utility wrapper for pair and move-related utilities.
 */
#ifndef IVYUTILITY_H
#define IVYUTILITY_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
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
