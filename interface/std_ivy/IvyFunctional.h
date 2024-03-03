#ifndef IVYFUNCTIONAL_H
#define IVYFUNCTIONAL_H


// CUDA functional is incomplete and may remain so in the future, so we need to define our own hash class.
// The defined hash class lives within the std_ivy namespace.
// Note that std_fcnal does NOT point to std_ivy in any case.
// That means we can have
// std_fcnal::hash (cuda::std::hash, if CUDA provides it later, or std::hash in C++ STL when CUDA is disabled)
// and std_ivy::hash at the same time.
#include "std_ivy/functional/IvyHash.h"

// In case we need to define our own set of classes, we can also define them in the std_ivy namespace.
#include "std_ivy/functional/IvyMultiplies.h"


#ifdef __USE_CUDA__

#include "cuda_runtime.h"
#include <cuda/std/functional>

#ifndef std_fcnal
#define std_fcnal cuda::std
#endif

#else

#include <functional>

#ifndef std_fcnal
#define std_fcnal std
#endif

#endif


#endif
