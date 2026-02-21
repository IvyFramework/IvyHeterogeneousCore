/**
 * @file IvyIterator.h
 * @brief Umbrella header exposing iterator primitives, traits, adapters, and builders.
 */
#ifndef IVYITERATOR_H
#define IVYITERATOR_H


#ifdef __USE_CUDA__
#include "cuda_runtime.h"
#endif

#include "std_ivy/iterator/IvyIteratorPrimitives.h"
#include "std_ivy/iterator/IvyIteratorTraits.h"
#include "std_ivy/iterator/IvyIteratorImpl.h"
#include "std_ivy/iterator/IvyReverseIterator.h"
#include "std_ivy/iterator/IvyContiguousIterator.h"
#include "std_ivy/iterator/IvyContiguousIteratorBuilder.h"
#include "std_ivy/iterator/IvyBucketedIteratorBuilder.h"

#ifndef std_iter
#define std_iter std_ivy
#endif


#endif
