#ifndef IVYCUDAFLAGS_H
#define IVYCUDAFLAGS_H


#ifdef __USE_CUDA__

#define __CUDA_HOST__ __host__
#define __CUDA_DEVICE__ __device__
#define __CUDA_GLOBAL__ __global__
#define __CUDA_HOST_DEVICE__ __host__ __device__
#define __CUDA_DEVICE_HOST__ __device__ __host__
#define __CUDA_CHECK_SUCCESS__(CALL) (CALL == cudaSuccess)

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
#define __CUDA_DEVICE_CODE__
#endif

#ifdef __CUDA_DEVICE_CODE__
#define __STATIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#define __DYNAMIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#else
#define __STATIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#define __DYNAMIC_CAST__(TYPE, PTR) dynamic_cast<TYPE>(PTR)
#endif

#else

#define __CUDA_HOST__
#define __CUDA_DEVICE_
#define __CUDA_GLOBAL__
#define __CUDA_HOST_DEVICE__ __CUDA_HOST__ __CUDA_DEVICE_
#define __CUDA_DEVICE_HOST__ __CUDA_DEVICE_ __CUDA_HOST__

#define __STATIC_CAST__(TYPE, PTR) static_cast<TYPE>(PTR)
#define __DYNAMIC_CAST__(TYPE, PTR) dynamic_cast<TYPE>(PTR)

#endif


#endif
