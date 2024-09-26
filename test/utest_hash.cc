#include "common_test_defs.h"

#include "std_ivy/IvyAlgorithm.h"


void utest(){
  constexpr unsigned int nsum = 10;
  double sum_vals[nsum];
  for (unsigned int i = 0; i < nsum; i++) sum_vals[i] = i+1;

  typedef std_ivy::hash<int> hash_int;
  typedef std_ivy::hash<float> hash_float;
  typedef std_ivy::hash<double> hash_double;
  typedef std_ivy::hash<unsigned long long int> hash_ullint;
  typedef std_ivy::hash<double*> hash_pdouble;
  typedef std_ivy::hash<char const*> hash_cstr;
  __PRINT_INFO__("hash_int(7) = %llu\n", hash_int()(7));
  __PRINT_INFO__("hash_float(7.7e5) = %llu\n", hash_float()(7.7e5));
  __PRINT_INFO__("hash_double(7.7e5) = %llu\n", hash_double()(7.7e5));
  __PRINT_INFO__("hash_ullint(7) = %llu\n", hash_ullint()(7));
  __PRINT_INFO__("hash_ullint(1527) = %llu\n", hash_ullint()(1527));
  __PRINT_INFO__("hash_pdouble(&sum_vals[0]) = %llu\n", hash_pdouble()(&(sum_vals[0])));
  __PRINT_INFO__("hash_pdouble(&sum_vals[1]) = %llu\n", hash_pdouble()(&(sum_vals[1])));
  __PRINT_INFO__("hash_cstr(\"Hello, world!\") = %llu\n", hash_cstr()("Hello, world!"));
}


int main(){
  utest();

  return 0;
}
