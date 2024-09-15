#ifndef IVYMEMORY_H
#define IVYMEMORY_H


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif

#include "std_ivy/memory/IvyAddressof.h"
#include "std_ivy/memory/IvyPointerTraits.h"
#include "std_ivy/memory/IvyAllocator.h"
#include "std_ivy/memory/IvyMemoryView.h"
#include "std_ivy/memory/IvyUnifiedPtr.h"

#ifndef std_mem
#define std_mem std_ivy
#endif


#endif
