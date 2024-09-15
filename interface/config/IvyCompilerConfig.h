#ifndef IVYCOMPILERCONFIG_H
#define IVYCOMPILERCONFIG_H


#include "config/IvyCudaFlags.h"


#define DEVICE_CODE_HOST 0
#define DEVICE_CODE_GPU 1
#ifdef __CUDA_DEVICE_CODE__
#define DEVICE_CODE DEVICE_CODE_GPU
#else
#define DEVICE_CODE DEVICE_CODE_HOST
#endif


#endif
