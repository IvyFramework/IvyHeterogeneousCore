#include "autodiff/IvyConstant.h"
#include "autodiff/IvyVariable.h"
#include "autodiff/IvyComplexVariable.h"
#include "autodiff/IvyTensor.h"


int main(){
  using namespace std_ivy;

  auto cplx = Complex<double>(std_ivy::IvyMemoryType::Host, nullptr, 1, 2);
  auto rvar = Variable<double>(std_ivy::IvyMemoryType::Host, nullptr, 3);
  auto rconst = Constant<double>(std_ivy::IvyMemoryType::Host, nullptr, 5);

  IvyTensorShape tshape({2, 3, 4});
  tshape.print();
  auto tshape_slice1 = tshape.get_slice_shape(1);
  tshape_slice1.print();
  auto absi = tshape.get_abs_index({0, 1, 2});
  __PRINT_INFO__("Absolute index of tensor coordinates {0, 1, 2} = %llu, size = %llu\n", absi, tshape.num_elements());

  auto t3d = Tensor<double>(tshape.get_memory_type(), tshape.gpu_stream(), tshape, 5.);
  t3d->at({ 0,1,2 }) = 3.14;
  print_value(t3d);
}