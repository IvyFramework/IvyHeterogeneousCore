#ifndef IVYNUMERIC_H
#define IVYNUMERIC_H


#ifdef __USE_CUDA__

#include "std_ivy/numeric/IvyAccumulate.h"

#ifndef std_numeric
#define std_numeric std_ivy
#endif

#else

#include <numeric>
#ifndef std_numeric
#define std_numeric std
#endif

#endif


#endif
