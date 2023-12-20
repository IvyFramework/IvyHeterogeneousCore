#ifndef IVYCOMPILERFLAGS_H
#define IVYCOMPILERFLAGS_h


#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define __HAS_CPP17_FEATURES__
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || __cplusplus >= 202002L)
#define __HAS_CPP20_FEATURES__
#endif


#endif
