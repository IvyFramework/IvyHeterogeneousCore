#include "autodiff/IvyConstant.h"
#include "autodiff/IvyVariable.h"
#include "autodiff/IvyComplexVariable.h"
#include "autodiff/IvyTensorShape.h"


int main(){
  auto cplx = Complex<double>(1, 1, std_ivy::IvyMemoryType::Host, nullptr);

  IvyTensorShape tshape({2, 3, 4});
  tshape.print();
}