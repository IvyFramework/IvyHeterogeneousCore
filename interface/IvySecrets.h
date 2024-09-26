#ifndef IVYSECRETS_H
#define IVYSECRETS_H

namespace IvySecrets{
  struct dump_helper{
    template<typename T> static void dump(T& obj){
      obj.dump();
    }
  };
}


#endif
