#ifndef IVYTHREADSAFEPTR_H
#define IVYTHREADSAFEPTR_H


#include "std_ivy/IvyMemory.h"


// Macros for thread-safe pointer handling
#define IvyThreadSafePtr_t std_mem::shared_ptr
#define make_IvyThreadSafePtr std_mem::make_shared


#endif
