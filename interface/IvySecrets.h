#ifndef IVYSECRETS_H
#define IVYSECRETS_H


#include "config/IvyCompilerConfig.h"


namespace IvySecrets{
  struct dump_helper{
    template<typename T> static __HOST_DEVICE__ void dump(T& obj){
      obj.dump();
    }
  };
}


#endif
