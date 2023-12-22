#ifndef IVYTYPETRAITS_H
#define IVYTYPETRAITS_H


#ifdef __USE_CUDA__

#include <cuda/std/type_traits>
#ifndef std_ttraits
#define std_ttraits cuda::std
#endif

#else

#include <type_traits>
#ifndef std_ttraits
#define std_ttraits std
#endif

#endif

// Define shorthands for common type trait checks
#define ENABLE_IF_BASE_OF(BASE, DERIVED) std_ttraits::enable_if_t<std_ttraits::is_base_of_v<BASE, DERIVED>::value>, bool> = true


#endif
