#include "autodiff/basic_nodes/IvyConstant.h"
#include "autodiff/basic_nodes/IvyVariable.h"
#include "autodiff/basic_nodes/IvyComplexVariable.h"
#include "autodiff/basic_nodes/IvyTensor.h"
#include "autodiff/arithmetic/IvyMathBaseArithmetic.h"


using namespace std_ivy;
using namespace IvyMath;


void utest(){
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

  IvyTensorShape tshape({ 2, 3, 4 });
  tshape.print();
  auto tshape_slice1 = tshape.get_slice_shape(1);
  tshape_slice1.print();
  auto absi = tshape.get_abs_index({ 0, 1, 2 });
  __PRINT_INFO__("Absolute index of tensor coordinates {0, 1, 2} = %llu, size = %llu\n", absi, tshape.num_elements());

  __PRINT_INFO__("Tensor shape copy pointer:\n");
  TensorShape(std_ivy::IvyMemoryType::Host, nullptr, tshape)->print();

  auto t3d = Tensor<double>(tshape.get_memory_type(), tshape.gpu_stream(), tshape, 5.);
  t3d->at({ 0,1,2 }) = 3.14;
  print_value(t3d);

  auto t3dp = Tensor<IvyMath::IvyConstantPtr_t<double>>(
    tshape.get_memory_type(),
    tshape.gpu_stream(),
    tshape,
    Constant<double>(
      tshape.get_memory_type(),
      tshape.gpu_stream(),
      5.
    )
  );
  print_value(t3dp);
  __PRINT_INFO__(
    "Addrs: t3dp[0,1,2] = %p, t3dp[0,0,0] = %p\n",
    t3dp->at({ 0,1,2 }).get(),
    t3dp->at({ 0,0,0 }).get()
  );
  __PRINT_INFO__(
    "t3dp[0,1,2] clients %p ?= %p\n",
    t3dp->at({ 0,1,2 })->get_clients()[0].get(),
    t3dp.get()
  );

  IvyGPUStream* stream = IvyStreamUtils::make_global_gpu_stream();
  using t3dp_allocator = std_mem::allocator<decltype(t3dp)>;
  using t3dp_allocator_traits = std_mem::allocator_traits<t3dp_allocator>;
  IvyMath::IvyTensorPtr_t<IvyMath::IvyConstantPtr_t<double>>* t3dpn = t3dp_allocator_traits::allocate(1, IvyMemoryType::Host, *stream);
  t3dp_allocator_traits::transfer(t3dpn, &t3dp, 1, IvyMemoryType::Host, IvyMemoryType::Host, *stream);
  __PRINT_INFO__("t3dpn transfer complete!\n");
  print_value(*t3dpn);
  __PRINT_INFO__(
    "Number of clients of t3dpn[0,1,2] = %llu\n",
    (*t3dpn)->at({ 0,1,2 })->get_clients().size()
  );
  __PRINT_INFO__(
    "Number of clients of t3dp[0,1,2] = %llu\n",
    t3dp->at({ 0,1,2 })->get_clients().size()
  );
  __PRINT_INFO__(
    "t3dp[0,1,2] clients reprint %p ?= %p\n",
    t3dp->at({ 0,1,2 })->get_clients()[0].get(),
    t3dp.get()
  );
  //__PRINT_INFO__(
  //  "t3dpn[0,1,2] clients %p ?= %p\n",
  //  (*t3dpn)->at({ 0,1,2 })->get_clients()[0].get(),
  //  (*t3dpn).get()
  //);
  t3dp_allocator_traits::destroy(t3dpn, 1, IvyMemoryType::Host, *stream);
  IvyStreamUtils::destroy_stream(stream);
}


int main(){
  utest();
  return 0;
}
