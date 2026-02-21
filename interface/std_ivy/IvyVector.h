/**
 * @file IvyVector.h
 * @brief Umbrella header exposing the std_ivy vector implementation.
 */
#ifndef IVYVECTOR_H
#define IVYVECTOR_H


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif

#include "std_ivy/vector/IvyVectorImpl.h"

#ifndef std_vec
#define std_vec std_ivy
#endif


#endif
