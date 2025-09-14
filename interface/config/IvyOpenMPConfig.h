#ifndef IVYOPENMPCONFIG_H
#define IVYOPENMPCONFIG_H


#include "IvyCudaFlags.h"


#if defined(_OPENMP) && !defined(__CUDA_DEVICE_CODE__)
  #include <omp.h>

  #define OPENMP_ENABLED
  #define NUM_CPU_THREADS_THRESHOLD 8
#endif
  

#endif
