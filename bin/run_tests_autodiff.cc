#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"


int main(){
  using namespace std_ivy;
  using namespace IvyMath;

  auto cplx = Complex<double>(std_ivy::IvyMemoryType::Host, nullptr, 1, 2);
  auto rvar = Variable<double>(std_ivy::IvyMemoryType::Host, nullptr, 3);
  auto rconst = Constant<double>(std_ivy::IvyMemoryType::Host, nullptr, 5);

  __PRINT_INFO__("cplx = "); print_value(cplx);
  __PRINT_INFO__("-cplx = "); print_value(-(*cplx));
  __PRINT_INFO__("exp(cplx) = "); print_value(Exp(*cplx));
  __PRINT_INFO__("ln(cplx) = "); print_value(Log(*cplx));
  __PRINT_INFO__("cplx* = "); print_value(Conjugate(*cplx));
  __PRINT_INFO__("Re(cplx) = "); print_value(Real(*cplx));
  __PRINT_INFO__("Im(cplx) = "); print_value(Imag(*cplx));

  IvyTensorShape tshape({2, 3, 4});
  tshape.print();
  auto tshape_slice1 = tshape.get_slice_shape(1);
  tshape_slice1.print();
  auto absi = tshape.get_abs_index({0, 1, 2});
  __PRINT_INFO__("Absolute index of tensor coordinates {0, 1, 2} = %llu, size = %llu\n", absi, tshape.num_elements());

  auto t3d = Tensor<double>(tshape.get_memory_type(), tshape.gpu_stream(), tshape, 5.);
  t3d->at({ 0,1,2 }) = 3.14;
  print_value(t3d);

  using NegateEvaluator = NegateFcnal<std_ttraits::remove_reference_t<decltype(*cplx)>>;
  auto grad_neg = NegateEvaluator::gradient(cplx);
  __PRINT_INFO__("grad_neg(%s) = ", typeid(grad_neg).name());
  print_value(grad_neg);

  auto grad_cplx_rvar = function_gradient<decltype(cplx)>::get(cplx, rvar);
  __PRINT_INFO__("grad_cplx_rvar(%s) = ", typeid(grad_cplx_rvar).name());
  print_value(grad_cplx_rvar);

  auto grad_cplx_cplx = function_gradient<decltype(cplx)>::get(cplx, cplx);
  __PRINT_INFO__("grad_cplx_cplx(%s) = ", typeid(grad_cplx_cplx).name());
  print_value(grad_cplx_cplx);

  auto fcn_negate_cplx = -cplx;
  __PRINT_INFO__("fcn_negate_cplx = ");
  print_value(fcn_negate_cplx);
  auto grad_fcn_negate_cplx = fcn_negate_cplx->gradient(rvar);
  __PRINT_INFO__("grad_fcn_negate_cplx(%s) = ", typeid(grad_fcn_negate_cplx).name());
  print_value(grad_fcn_negate_cplx);

  using MultiplyEvaluator = MultiplyFcnal<std_ttraits::remove_reference_t<decltype(*cplx)>, std_ttraits::remove_reference_t<decltype(*cplx)>>;
  auto prod_cplx_cplx = MultiplyEvaluator::eval(*cplx, *cplx);
  __PRINT_INFO__("prod_cplx_cplx = ");
  print_value(prod_cplx_cplx);

  auto pow_cplx_cplx = Pow(cplx, cplx);
  __PRINT_INFO__("pow_cplx_cplx = ");
  print_value(pow_cplx_cplx);
  auto grad_pow_cplx_cplx = pow_cplx_cplx->gradient(cplx);
  __PRINT_INFO__("grad_pow_cplx_cplx(%s) = ", typeid(grad_pow_cplx_cplx).name());
  print_value(grad_pow_cplx_cplx); // FIXME: The output is not correct yet.

  auto fcn_manual_cubed = cplx*cplx*cplx;
  __PRINT_INFO__("fcn_manual_cubed = ");
  print_value(fcn_manual_cubed);
  //auto grad_fcn_manual_cubed = fcn_manual_cubed->gradient(cplx);
  //__PRINT_INFO__("grad_fcn_manual_cubed(%s) = ", typeid(grad_fcn_manual_cubed).name());
  //print_value(grad_fcn_manual_cubed->value());
}