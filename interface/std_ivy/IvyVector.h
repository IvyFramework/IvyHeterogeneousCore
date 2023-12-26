#ifndef IVYVECTOR_H
#define IVYVECTOR_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/vector/IvyVectorImpl.h"

#ifndef std_vec
#define std_vec std_ivy
#endif

#else

#include <vector>

#ifndef std_vec
#define std_vec std
#endif

#endif


#endif
