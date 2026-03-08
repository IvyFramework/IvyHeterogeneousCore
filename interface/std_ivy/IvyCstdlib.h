#ifndef IVYCSTDLIB_H
#define IVYCSTDLIB_H

/**
 * @file IvyCstdlib.h
 * @brief Host-side wrapper for the C standard library.
 *
 * Guards the inclusion of \<cstdlib\> behind the CUDA device-code sentinel so
 * that standard allocation symbols are available in host compilation units
 * without polluting device-code translation units.
 *
 * The three `using` declarations for `malloc`, `free`, and `realloc` are
 * intentionally absent: adding them would pull `std::size_t` into the global
 * namespace and cause ambiguity with `IvyTypes::size_t` when CUDA's
 * `host_runtime.h` later looks up unqualified `size_t`.  All call sites in
 * `IvyMemoryHelpers.h` use fully-qualified `std::malloc` / `std::free`.
 */

#ifndef __CUDA_DEVICE_CODE__
#include <cstdlib>
#endif

#endif
