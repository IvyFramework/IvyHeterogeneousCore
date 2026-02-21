/**
 * @file IvyUnorderedMap.h
 * @brief Umbrella header exposing the std_ivy unordered_map implementation.
 */
#ifndef IVYUNORDEREDMAP_H
#define IVYUNORDEREDMAP_H


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif

#include "std_ivy/unordered_map/IvyUnorderedMapImpl.h"

#ifndef std_umap
#define std_umap std_ivy
#endif


#endif
