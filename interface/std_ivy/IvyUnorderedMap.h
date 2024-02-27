#ifndef IVYUNORDEREDMAP_H
#define IVYUNORDEREDMAP_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/unordered_map/IvyUnorderedMapImpl.h"

#ifndef std_umap
#define std_umap std_ivy
#endif

#else

#include <unordered_map>

#ifndef std_umap
#define std_umap std
#endif

#endif


#endif
