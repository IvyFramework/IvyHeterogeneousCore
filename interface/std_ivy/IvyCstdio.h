#ifndef IVYCSTDIO_H
#define IVYCSTDIO_H


#include <cstdio>
#ifndef std_cstdio
#define std_cstdio std
#endif

#ifdef __USE_CUDA__

#ifndef __CUDA_DEVICE_CODE__
#define __PRINT_INFO__(...) fprintf(stdout, __VA_ARGS__)
#define __PRINT_ERROR__(...) fprintf(stderr, __VA_ARGS__)
#else
#define __PRINT_INFO__(...) {}
#define __PRINT_ERROR__(...) {}
#endif

#else

#define __PRINT_INFO__(...) fprintf(stdout, __VA_ARGS__)
#define __PRINT_ERROR__(...) fprintf(stderr, __VA_ARGS__)

#endif


#endif
