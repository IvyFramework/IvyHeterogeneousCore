/**
 * @file IvyCstring.h
 * @brief Conditional cstring include wrapper for non-CUDA translation units.
 */
#ifndef IVYCSTRING_H
#define IVYCSTRING_H


#ifndef __USE_CUDA__
#include <cstring>
#endif


#endif
