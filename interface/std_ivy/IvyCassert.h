/**
 * @file IvyCassert.h
 * @brief Backend-selecting assertion wrapper for host and CUDA builds.
 */
#ifndef IVYCASSERT_H
#define IVYCASSERT_H


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/std/cassert>

#else

#include <cassert>

#endif


#endif
