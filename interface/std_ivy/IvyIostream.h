#ifndef IVYIOSTREAM_H
#define IVYIOSTREAM_H

#ifdef __USE_CUDA__

#include "IvyCstdio.h"
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
