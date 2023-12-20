#ifndef IVYMEMORY_H
#define IVYMEMORY_H


#ifdef __USE_CUDA__

#ifndef CUDA_RT_INCLUDED
#define CUDA_RT_INCLUDED
#include "cuda_runtime.h"
#endif

#include "IvyAllocator.hh"

#ifndef std_mem
#define std_mem std_ivy
#endif

#else

#include <memory>
#ifndef std_mem
#define std_mem std
#endif

#endif


#endif
