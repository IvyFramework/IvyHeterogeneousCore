/**
 * @file IvyCstddef.h
 * @brief Backend-selecting cstddef wrapper for size and pointer-related utilities.
 */
#ifndef IVYCSTDDEF_H
#define IVYCSTDDEF_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/std/cstddef>
#ifndef std_cstddef
#define std_cstddef cuda::std
#endif

#else

#include <cstddef>
#ifndef std_cstddef
#define std_cstddef std
#endif

#endif


#endif
