#ifndef IVYCOMPILERCONFIG_H
#define IVYCOMPILERCONFIG_H


#include "config/IvyCompilerFlags.h"
#include "config/IvyCudaFlags.h"


#define DEVICE_CODE_HOST 0
#define DEVICE_CODE_CUDA 1
#ifdef __CUDA_DEVICE_CODE__
#define DEVICE_CODE DEVICE_CODE_CUDA
#else
#define DEVICE_CODE DEVICE_CODE_HOST
#endif


#endif
