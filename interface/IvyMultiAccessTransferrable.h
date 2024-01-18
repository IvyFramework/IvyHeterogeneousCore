#ifndef IVYMULTIACCESSTRANSFERRABLE_H
#define IVYMULTIACCESSTRANSFERRABLE_H


/*
IvyMultiAccessTransferrable:
Abstract base class to be inherited by any class that needs to be able to transfer data between different memory types.
*/


#include "IvyMemoryHelpers.h"


struct IvyMultiAccessTransferrable{
  virtual __CUDA_HOST_DEVICE__ bool transfer_internal_memory(std_ivy::IvyMemoryType const& new_mem_type) = 0;
};


#endif
