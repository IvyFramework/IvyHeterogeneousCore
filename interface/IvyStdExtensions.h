// No ifndef/define blocks for this file

#include "IvyCompilerFlags.h"


#ifdef __USE_CUDA__

#ifndef CUDA_RT_INCLUDED
#define CUDA_RT_INCLUDED
#include "cuda_runtime.h"
#endif

#ifdef INCLUDE_CMATH
#include <cuda/std/cmath>
#ifndef std_math
#define std_math cuda::std
#endif
#endif
#ifdef INCLUDE_UTILITY
#include <cuda/std/utility>
#ifndef std_util
#define std_util cuda::std
#endif
#endif
#ifdef INCLUDE_LIMITS
#include <cuda/std/limits>
#ifndef std_limits
#define std_limits cuda::std
#endif
#endif
#ifdef INCLUDE_TYPE_TRAITS
#include <cuda/std/type_traits>
#ifndef std_ttraits
#define std_ttraits cuda::std
#endif
#endif
#ifdef INCLUDE_FUNCTIONAL
#include <cuda/std/functional>
#ifndef std_functional
#define std_functional cuda::std
#endif
#endif

#ifdef INCLUDE_IOSTREAM
#include "cuda/std/detail/libcxx/include/iostream"
#ifndef std_iostream
#define std_iostream cuda::std
#endif
#endif
#ifdef INCLUDE_CSTDLIB
#include "cuda/std/detail/libcxx/include/cstdlib"
#ifndef std_cstdlib
#define std_cstdlib cuda::std
#endif
#endif
#ifdef INCLUDE_NUMERIC
#include "cuda/std/detail/libcxx/include/numeric"
#ifndef std_numeric
#define std_numeric cuda::std
#endif
#endif
#ifdef INCLUDE_ITERATOR
#include "cuda/std/detail/libcxx/include/iterator"
#ifndef std_iterator
#define std_iterator cuda::std
#endif
#endif

#ifdef INCLUDE_ALGORITHM
#include "IvyStd/IvyAlgorithm.h"
#ifndef std_algo
#define std_algo std_ivy
#endif
#endif
#ifdef INCLUDE_VECTOR
#include "cuda/std/detail/libcxx/include/vector"
#ifndef std_vec
#define std_vec cuda::std
#endif
#endif
#ifdef INCLUDE_UNORDERED_MAP
#include "cuda/std/detail/libcxx/include/unordered_map"
#ifndef std_umap
#define std_umap cuda::std
#endif
#endif
#ifdef INCLUDE_SET
#include "cuda/std/detail/libcxx/include/set"
#ifndef std_set
#define std_set cuda::std
#endif
#endif

#ifdef INCLUDE_INITIALIZER_LIST
#include <initializer_list>
#ifndef std_ilist
#define std_ilist std
#endif
#endif
#ifdef INCLUDE_CSTDIO
#include <cstdio>
#ifndef std_cstdio
#define std_cstdio std
#endif
#endif


#ifdef INCLUDE_MEMORY
#include "cuda/std/detail/libcxx/include/memory"
#ifndef std_mem
#define std_mem cuda::std
#endif
#endif
#ifdef INCLUDE_STRING
#include "cuda/std/detail/libcxx/include/string"
#ifndef std_str
#define std_str cuda::std
#endif
#endif
#ifdef INCLUDE_STRING_VIEW
#include "cuda/std/detail/libcxx/include/string_view"
#ifndef std_strview
#define std_strview cuda::std
#endif
#endif

#ifdef __HAS_CPP20_FEATURES__
#ifdef INCLUDE_SOURCE_LOCATION
#include <source_location>
#ifndef std_srcloc
#define std_srcloc std
#endif
#endif
#endif

#else

#ifdef INCLUDE_CMATH
#include <cmath>
#ifndef std_math
#define std_math std
#endif
#endif
#ifdef INCLUDE_UTILITY
#include <utility>
#ifndef std_util
#define std_util std
#endif
#endif
#ifdef INCLUDE_LIMITS
#include <limits>
#ifndef std_limits
#define std_limits std
#endif
#endif
#ifdef INCLUDE_TYPE_TRAITS
#include <type_traits>
#ifndef std_ttraits
#define std_ttraits std
#endif
#endif
#ifdef INCLUDE_FUNCTIONAL
#include <functional>
#ifndef std_functional
#define std_functional std
#endif
#endif
#ifdef INCLUDE_IOSTREAM
#include <iostream>
#ifndef std_iostream
#define std_iostream std
#endif
#endif
#ifdef INCLUDE_CSTDLIB
#include <cstdlib>
#ifndef std_cstdlib
#define std_cstdlib std
#endif
#endif
#ifdef INCLUDE_NUMERIC
#include <numeric>
#ifndef std_numeric
#define std_numeric std
#endif
#endif
#ifdef INCLUDE_ITERATOR
#include <iterator>
#ifndef std_iterator
#define std_iterator std
#endif
#endif
#ifdef INCLUDE_ALGORITHM
#include <algorithm>
#ifndef std_algo
#define std_algo std
#endif
#endif
#ifdef INCLUDE_VECTOR
#include <vector>
#ifndef std_vec
#define std_vec std
#endif
#endif
#ifdef INCLUDE_UNORDERED_MAP
#include <unordered_map>
#ifndef std_umap
#define std_umap std
#endif
#endif
#ifdef INCLUDE_SET
#include <set>
#ifndef std_set
#define std_set std
#endif
#endif
#ifdef INCLUDE_INITIALIZER_LIST
#include <initializer_list>
#ifndef std_ilist
#define std_ilist std
#endif
#endif
#ifdef INCLUDE_CSTDIO
#include <cstdio>
#ifndef std_cstdio
#define std_cstdio std
#endif
#endif
#ifdef INCLUDE_MEMORY
#include <memory>
#ifndef std_mem
#define std_mem std
#endif
#endif
#ifdef INCLUDE_STRING
#include <string>
#ifndef std_str
#define std_str std
#endif
#endif
#ifdef INCLUDE_STRING_VIEW
#include <string_view>
#ifndef std_strview
#define std_strview std
#endif
#endif

#ifdef __HAS_CPP20_FEATURES__
#ifdef INCLUDE_SOURCE_LOCATION
#include <source_location>
#ifndef std_srcloc
#define std_srcloc std
#endif
#endif
#endif

#endif
