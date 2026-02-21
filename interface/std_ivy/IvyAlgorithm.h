/**
 * @file IvyAlgorithm.h
 * @brief Umbrella header exposing algorithm utilities under the std_ivy namespace.
 */
#ifndef IVYALGORITHM_H
#define IVYALGORITHM_H


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif

#include "std_ivy/algorithm/IvyMinMax.h"
#include "std_ivy/algorithm/IvyParallelOp.h"
#include "std_ivy/algorithm/IvyFind.h"

#ifndef std_algo
#define std_algo std_ivy
#endif


#endif
