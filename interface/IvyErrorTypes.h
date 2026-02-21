#ifndef IVYERRORTYPES_H
#define IVYERRORTYPES_H

/**
 * @file IvyErrorTypes.h
 * @brief Error-type aliases and helpers abstracting CUDA vs non-CUDA builds.
 */


#include "std_ivy/IvyCstring.h"


#ifdef __USE_CUDA__

/** @brief Backend error-code type when CUDA is enabled. */
#define IvyError_t cudaError_t
/** @brief Backend success-code constant when CUDA is enabled. */
#define IvySuccess cudaSuccess
/** @brief Backend error-string accessor when CUDA is enabled. */
#define IvyGetErrorString cudaGetErrorString

#else

/** @brief Backend error-code type when CUDA is disabled. */
#define IvyError_t int
/** @brief Backend success-code constant when CUDA is disabled. */
#define IvySuccess 0

/** @brief Backend error-string accessor when CUDA is disabled. */
#define IvyGetErrorString std::strerror

#endif


#endif
