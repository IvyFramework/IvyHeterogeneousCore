#ifndef IVYIOSTREAM_H
#define IVYIOSTREAM_H


#ifdef __USE_CUDA__

#include "std_ivy/IvyCstdio.h"
#ifndef std_ios
#define std_ios std_cstdio
#endif

#else

#include <iostream>
#ifndef std_ios
#define std_ios std
#endif

#endif


#endif
