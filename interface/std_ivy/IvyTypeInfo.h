#ifndef IVYTYPEINFO_H
#define IVYTYPEINFO_H


#include "config/IvyCompilerConfig.h"

#if !defined(__USE_CUDA__) || ((DEVICE_CODE == DEVICE_CODE_HOST) && defined(__USE_CUDA__))

#include <typeinfo>
#ifndef std_tinfo
#define std_tinfo std
#endif

#define __TYPE_NAME__(TYPE_OR_OBJ) typeid(TYPE_OR_OBJ).name()

#else

#define __TYPE_NAME__(...) "Undetermined type info. in device codes"

#endif


#endif
