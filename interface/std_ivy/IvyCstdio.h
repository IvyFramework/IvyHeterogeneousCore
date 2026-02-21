/**
 * @file IvyCstdio.h
 * @brief C stdio wrapper defining common print macros for host and device contexts.
 */
#ifndef IVYCSTDIO_H
#define IVYCSTDIO_H


#include <cstdio>
#ifndef std_cstdio
#define std_cstdio std
#endif

#ifdef __USE_CUDA__

#if (DEVICE_CODE==DEVICE_CODE_HOST)
#define __PRINT_INFO__(...) printf(__VA_ARGS__)
#define __PRINT_ERROR__(...) printf(__VA_ARGS__)
#else
#define __PRINT_INFO__(...) {}
#define __PRINT_ERROR__(...) {}
#endif
#define __PRINT_DEBUG__(...) printf(__VA_ARGS__)

#else

#define __PRINT_INFO__(...) fprintf(stdout, __VA_ARGS__)
#define __PRINT_ERROR__(...) fprintf(stderr, __VA_ARGS__)
#define __PRINT_DEBUG__(...) fprintf(stdout, __VA_ARGS__)

#endif


#endif
