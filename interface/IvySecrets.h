#ifndef IVYSECRETS_H
#define IVYSECRETS_H

/**
 * @file IvySecrets.h
 * @brief Internal helper hooks for friend-like utility operations.
 */


#include "config/IvyCompilerConfig.h"


namespace IvySecrets{
  /** @brief Helper granting controlled access to `dump()` routines on objects. */
  struct dump_helper{
    /**
     * @brief Invoke `obj.dump()`.
     * @tparam T Object type exposing `dump()`.
     * @param obj Target object.
     */
    template<typename T> static __HOST_DEVICE__ void dump(T& obj){
      obj.dump();
    }
  };
}


#endif
