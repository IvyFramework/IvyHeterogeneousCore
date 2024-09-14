#ifndef IVYERRORTYPES_H
#define IVYERRORTYPES_H


#include "std_ivy/IvyCstring.h"


#ifdef __USE_CUDA__

#define IvyError_t cudaError_t
#define IvySuccess cudaSuccess
#define IvyGetErrorString cudaGetErrorString

#else

#define IvyError_t int
#define IvySuccess 0

#define IvyGetErrorString std::strerror

#endif


#endif
