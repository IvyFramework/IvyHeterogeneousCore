#ifndef IVYCOMPILERFLAGS_H
#define IVYCOMPILERFLAGS_H

/**
 * @file IvyCompilerFlags.h
 * @brief Compiler and language-feature detection macros used by Ivy portability layers.
 */


/*
C++ standard version
*/
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define __HAS_CPP17_FEATURES__
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) || __cplusplus >= 202002L)
#define __HAS_CPP20_FEATURES__
#endif

/*
Compiler types and compiler-dependent macros
*/
#define COMPILER_UNKNOWN 0
#define COMPILER_CLANG 1
#define COMPILER_CLANG_APPLE 2
#define COMPILER_GCC 3
#define COMPILER_ILLVM 4
#define COMPILER_MSVC 5
#define COMPILER_IBM 6
#define COMPILER_NVHPC 7
#define COMPILER_NVRTC 8
#if defined(__clang__)
#ifndef __apple_build_version__
#define COMPILER COMPILER_CLANG
#define COMPILER_VERSION (__clang_major__ * 100 + __clang_minor__)
#else
#define COMPILER COMPILER_CLANG_APPLE
#define COMPILER_VERSION __apple_build_version__
#endif
#elif defined(__INTEL_LLVM_COMPILER)
#define COMPILER COMPILER_ILLVM
#define COMPILER_VERSION __VERSION__
#elif defined(__GNUC__)
#define COMPILER COMPILER_GCC
#define COMPILER_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#elif defined(_MSC_VER)
#define COMPILER COMPILER_MSVC
#define COMPILER_VERSION _MSC_VER
#elif defined(__IBMCPP__)
#define COMPILER COMPILER_IBM
#define COMPILER_VERSION __IBMCPP__
#elif defined(__NVCOMPILER)
#define COMPILER COMPILER_NVHPC
#define COMPILER_VERSION (__NVCOMPILER_MAJOR__ * 100 + __NVCOMPILER_MINOR__)
#elif defined(__CUDACC_RTC__)
#define COMPILER COMPILER_NVRTC
#define COMPILER_VERSION (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__)
#else
#define COMPILER COMPILER_UNKNOWN
#define COMPILER_VERSION 0
#endif


/*
Encapsulator for a set of arguments.
Needed to distinguish a macro argument with commas.
*/
#define __ENCAPSULATE__(...) __VA_ARGS__


/*
Common compiler flags.
These are adapted from cuda/std/detail/libcxx/include/__config.
They allow finer control over the available features in the compilers.
By default, they should be defined.
If they are not, they should simply return 'No'.
*/
#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#ifndef __has_extension
#define __has_extension(x) 0
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#ifndef __has_cpp_attribute
#define __has_cpp_attribute(x) 0
#endif

// '__is_identifier' returns '0' if 'x' is a reserved identifier provided by the compiler, and '1' otherwise.
#ifndef __is_identifier
#define __is_identifier(x) 1
#endif

#ifndef __has_keyword
#define __has_keyword(x) !(__is_identifier(x))
#endif

#ifndef __has_declspec_attribute
#define __has_declspec_attribute(x) 0
#endif

#ifndef __has_include
#define __has_include(...) 0
#endif

#ifndef __check_builtin
#define __check_builtin(x) (__has_builtin(__##x) || __has_keyword(__##x) || __has_feature(x))
#endif


/*
A few keywords in C++ depend on the language version and compilers,
so the macros below are to maintain compatibility.
*/
#if defined(__HAS_CPP17_FEATURES__) && !(__has_feature(cxx_relaxed_constexpr))
#define __CPP_CONSTEXPR__ constexpr
#else
#define __CPP_CONSTEXPR__
#endif

#if defined(__HAS_CPP20_FEATURES__)
#define __CPP_VIRTUAL_CONSTEXPR__ constexpr
#else
#define __CPP_VIRTUAL_CONSTEXPR__
#endif

#if __has_feature(cxx_noexcept)
#define __NOEXCEPT__ noexcept
#define __NOEXCEPT_COND__(x) noexcept(x)
#else
#define __NOEXCEPT__ throw()
#define __NOEXCEPT_COND__(x)
#endif


/*
Objective-C++ support
*/
#if __has_feature(objc_arc)
#define __HAS_OBJC_ARC__
#endif
#if __has_feature(objc_arc_weak)
#define __HAS_OBJC_ARC_WEAK__
#endif
#ifdef __HAS_OBJC_ARC__
#ifdef __HAS_OBJC_ARC_WEAK__
#define __OBJC_POINTER_CMDS__ \
__OBJC_POINTER_CMD__(__strong) \
__OBJC_POINTER_CMD__(__weak) \
__OBJC_POINTER_CMD__(__unsafe_unretained) \
__OBJC_POINTER_CMD__(__autoreleasing)
#else
#define __OBJC_POINTER_CMDS__ \
__OBJC_POINTER_CMD__(__strong) \
__OBJC_POINTER_CMD__(__unsafe_unretained) \
__OBJC_POINTER_CMD__(__autoreleasing)
#endif
#else
#define __OBJC_POINTER_CMDS__
#endif


/*
CUDA does not support long double, so we need to implement workarounds.
We also always forbid long int.
*/
#ifdef __USE_CUDA__
#define __LONG_DOUBLE_FORBIDDEN__
#endif
#define __LONG_INT_FORBIDDEN__


#endif
