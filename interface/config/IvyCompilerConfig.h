#ifndef IVYCOMPILERCONFIG_H
#define IVYCOMPILERCONFIG_H

/**
 * @file IvyCompilerConfig.h
 * @brief Compile-time execution-context selectors for host vs device builds.
 */


#include "config/IvyCudaFlags.h"


/** @brief Constant identifying host compilation/execution context. */
#define DEVICE_CODE_HOST 0
/** @brief Constant identifying GPU compilation/execution context. */
#define DEVICE_CODE_GPU 1
#ifdef __CUDA_DEVICE_CODE__
/** @brief Active execution-context marker resolved at compile time. */
#define DEVICE_CODE DEVICE_CODE_GPU
#else
/** @brief Active execution-context marker resolved at compile time. */
#define DEVICE_CODE DEVICE_CODE_HOST
#endif


#endif
