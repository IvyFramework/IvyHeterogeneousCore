#ifndef IVYOPENMPCONFIG_H
#define IVYOPENMPCONFIG_H

/**
 * @file IvyOpenMPConfig.h
 * @brief OpenMP feature toggles for host-side parallel fallbacks.
 */


#include "IvyCudaFlags.h"


#if defined(_OPENMP) && !defined(__CUDA_DEVICE_CODE__)
  #include <omp.h>

  /** @brief Defined when host OpenMP support is available for this build. */
  #define OPENMP_ENABLED
  /** @brief Minimum loop size before enabling OpenMP parallel loops. */
  #define NUM_CPU_THREADS_THRESHOLD 8
#endif


#endif
