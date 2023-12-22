#ifndef IVYMEMORY_H
#define IVYMEMORY_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"

#include "std_ivy/memory/IvyAllocator.h"
#include "std_ivy/memory/IvySharedPtr.h"

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
