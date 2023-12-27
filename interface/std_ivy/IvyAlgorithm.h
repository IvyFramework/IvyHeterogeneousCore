#ifndef IVYALGORITHM_H
#define IVYALGORITHM_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/algorthm/IvyMinMax.h"

#ifndef std_algo
#define std_algo std_ivy
#endif

#else

#include <algorithm>
#ifndef std_algo
#define std_algo std
#endif

#endif


#endif
