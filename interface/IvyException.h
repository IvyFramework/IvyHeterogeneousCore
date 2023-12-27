#ifndef IVYEXCEPTION_H
#define IVYEXCEPTION_H


#include "config/IvyCudaFlags.h"


#ifndef __USE_CUDA__
#ifndef INCLUDE_STRING
#define INCLUDE_STRING
#endif
#ifndef INCLUDE_STRING_VIEW
#define INCLUDE_STRING_VIEW
#endif
#ifndef INCLUDE_SOURCE_LOCATION
#define INCLUDE_SOURCE_LOCATION
#endif

#include "IvyStdExtensions.h"


class IvyException{
public:
  __CUDA_HOST_DEVICE__ IvyException(
    std_strview::string_view msg, int const& exit_code=1
#ifdef __HAS_CPP20_FEATURES__
    , std_srcloc::source_location const& srcloc = std_srcloc::source_location::current()
#endif
  );

protected:
  __CUDA_HOST_DEVICE__ void print(
    std_strview::string_view msg
#ifdef __HAS_CPP20_FEATURES__
    , std_srcloc::source_location const& srcloc
#endif
  ) const;
};


#ifdef __USE_CUDA__
#define __CUDA_CHECK_OR_EXIT_WITH_ERROR__(CALL) \
{ \
  auto __cuda_error_code__ = CALL; \
  if (__cuda_error_code__ != cudaSuccess) \
    IvyException( \
      std_str::string("Call '") + #CALL + "' failed with error code " + std_str::to_string(static_case<int>(__cuda_error_code__)) + "." \
    ); \
}
#define __CUDA_CHECK_OR_CONTINUE_WITH_ERROR__(CALL) \
{ \
  auto __cuda_error_code__ = CALL; \
  if (__cuda_error_code__ != cudaSuccess) \
    IvyException( \
      std_str::string("Call '") + #CALL + "' failed with error code " + std_str::to_string(static_case<int>(__cuda_error_code__)) + ".", \
      0 \
    ); \
}


#endif // Temporary workaround:


#else

#include "std_ivy/IvyCassert.h"
#include "std_ivy/IvyCstdio.h"


#define __CUDA_CHECK_OR_EXIT_WITH_ERROR__(CALL) \
{ \
  auto __cuda_error_code__ = CALL; \
  if (__cuda_error_code__ != cudaSuccess){ \
    printf("Call '%s' failed with error code %d.\n", #CALL, __cuda_error_code__); \
    assert(false); \
  } \
}


#endif

#endif
