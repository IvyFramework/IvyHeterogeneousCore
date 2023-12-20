#include "std_ivy/IvyMemory.h"
#include <iostream>


__CUDA_GLOBAL__ void kernel_set_doubles(double* ptr, IvyBlockThread_t n){
  IvyBlockThread_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    ptr[i] = i+1;
    printf("ptr[%i] = %f\n", static_cast<int>(i), ptr[i]);
  }
}


int main(){
  auto obj_allocator = std_mem::allocator<double>();

  auto ptr_h = obj_allocator.allocate(3);
  std::cout << "ptr_h = " << ptr_h << std::endl;
  ptr_h[0] = 1.0;
  ptr_h[1] = 2.0;
  ptr_h[2] = 3.0;
  std::cout << "ptr_h values: " << ptr_h[0] << ", " << ptr_h[1] << ", " << ptr_h[2] << std::endl;
  obj_allocator.deallocate(ptr_h, 3);

  std::cout << "Trying device..." << std::endl;
  auto ptr_d = obj_allocator.allocate(3, true);
  std::cout << "ptr_d = " << ptr_d << std::endl;
  ptr_h = obj_allocator.allocate(3);
  std::cout << "ptr_h new = " << ptr_h << std::endl;
  kernel_set_doubles<<<1, 3>>>(ptr_d, 3);
  obj_allocator.transfer(ptr_h, ptr_d, 3, true);
  std::cout << "ptr_h new values: " << ptr_h[0] << ", " << ptr_h[1] << ", " << ptr_h[2] << std::endl;
  obj_allocator.deallocate(ptr_h, 3);
  obj_allocator.deallocate(ptr_d, 3, true);

  return 0;
}
