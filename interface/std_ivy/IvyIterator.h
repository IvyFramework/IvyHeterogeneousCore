#ifndef IVYITERATOR_H
#define IVYITERATOR_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include "std_ivy/iterator/IvyIteratorImpl.h"

#ifndef std_iter
#define std_iter std_ivy
#endif

#else

#include <iterator>

#ifndef std_iter
#define std_iter std
#endif

#endif


#endif
