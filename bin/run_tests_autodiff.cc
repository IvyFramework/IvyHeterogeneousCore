#include "autodiff/IvyConstant.h"
#include "autodiff/IvyVariable.h"
#include "autodiff/IvyComplexVariable.h"
#include "autodiff/IvyTensorShape.h"


int main(){
  auto cplx = Complex<double>(std_ivy::IvyMemoryType::Host, nullptr, 1, 2);
  auto rvar = Variable<double>(std_ivy::IvyMemoryType::Host, nullptr, 3);
  auto rconst = Constant<double>(std_ivy::IvyMemoryType::Host, nullptr, 5);

  IvyTensorShape tshape({2, 3, 4});
  tshape.print();
  auto tshape_slice1 = tshape.get_slice_shape(1);
  tshape_slice1.print();
}