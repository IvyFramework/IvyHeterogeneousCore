#include <std_ivy/IvyMemory.h>
#include "IvyCudaStream.h"


class dummy_B{
public:
  dummy_B() = default;
};
class dummy_D : public dummy_B{
public:
  double a;
  __CUDA_HOST_DEVICE__ dummy_D() : dummy_B(){}
  __CUDA_HOST_DEVICE__ dummy_D(double a_) : dummy_B(), a(a_){}
  __CUDA_HOST_DEVICE__ dummy_D(dummy_D const& other) : a(other.a){}
};


__CUDA_GLOBAL__ void kernel_set_doubles(double* ptr, IvyBlockThread_t n, unsigned char is){
  IvyBlockThread_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    ptr[i] = (i+2)*(is+1);
    if (i<3) printf("ptr[%i] = %f in stream %i\n", static_cast<int>(i), ptr[i], static_cast<int>(is));
  }
}


__CUDA_HOST_DEVICE__ void print_dummy_D(dummy_D* ptr){
  printf("dummy_D.a = %f\n", ptr->a);
}
__CUDA_GLOBAL__ void kernel_print_dummy_D(dummy_D* ptr){
  print_dummy_D(ptr);
}

__CUDA_HOST_DEVICE__ void print_dummy_B_as_D(dummy_B* ptr){
  printf("dummy_B.a = %f\n", __STATIC_CAST__(dummy_D*, ptr)->a);
}
__CUDA_GLOBAL__ void kernel_print_dummy_B_as_D(dummy_B* ptr){
  print_dummy_B_as_D(ptr);
}


int main(){
  constexpr unsigned char nStreams = 3;
  constexpr unsigned int nvars = 1000;
  auto obj_allocator = std_mem::allocator<double>();
  IvyCudaStream streams[nStreams]{
    IvyCudaStream(cudaStreamLegacy, false),
    IvyCudaStream(),
    IvyCudaStream()
  };

  for (unsigned char i = 0; i < nStreams; i++){
    printf("Stream %i (%p) computing...\n", i, streams[i].stream());

    std_mem::shared_ptr<dummy_D> ptr_shared = std_mem::make_shared<dummy_D>(true, &(streams[i].stream()), 1.);
    std_mem::shared_ptr<dummy_B> ptr_shared_copy = ptr_shared; ptr_shared_copy.reset(); ptr_shared_copy = ptr_shared;
    printf("ptr_shared no. of copies: %i\n", ptr_shared.use_count());
    printf("ptr_shared_copy no. of copies: %i\n", ptr_shared_copy.use_count());
    kernel_print_dummy_D<<<1, 1, 0, streams[i]>>>(ptr_shared.get());
    kernel_print_dummy_B_as_D<<<1, 1, 0, streams[i]>>>(ptr_shared_copy.get());

    std_mem::unique_ptr<dummy_D> ptr_unique = std_mem::make_unique<dummy_D>(true, &(streams[i].stream()), 1.);
    std_mem::unique_ptr<dummy_B> ptr_unique_copy = ptr_unique;
    printf("ptr_unique no. of copies: %i\n", ptr_unique.use_count());
    printf("ptr_unique_copy no. of copies: %i\n", ptr_unique_copy.use_count());
    kernel_print_dummy_B_as_D<<<1, 1, 0, streams[i]>>>(ptr_shared_copy.get());
    ptr_unique_copy.reset();
    printf("ptr_unique_copy no. of copies after reset: %i\n", ptr_unique_copy.use_count());

    IvyCudaEvent ev_allocate;
    IvyCudaEvent ev_set;
    IvyCudaEvent ev_transfer;
    IvyCudaEvent ev_deallocate;

    auto ptr_h = obj_allocator.allocate(nvars, false, streams[i]);
    printf("ptr_h = %p\n", ptr_h);
    ptr_h[0] = 1.0;
    ptr_h[1] = 2.0;
    ptr_h[2] = 3.0;
    printf("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, nvars, false, streams[i]);
    printf("Trying device...\n");

    auto ptr_d = obj_allocator.allocate(nvars, true, streams[i]);
    ev_allocate.record(streams[i]);
    ev_allocate.synchronize();
    printf("ptr_d = %p\n", ptr_d);
    ptr_h = obj_allocator.allocate(nvars, false, streams[i]);
    printf("ptr_h new = %p\n", ptr_h);
    kernel_set_doubles<<<1, nvars, 0, streams[i]>>>(ptr_d, nvars, i);
    ev_set.record(streams[i]);
    streams[i].wait(ev_set);
    obj_allocator.transfer(ptr_h, ptr_d, nvars, true, streams[i]);
    ev_transfer.record(streams[i]);
    streams[i].wait(ev_transfer);
    printf("ptr_h new values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, nvars, false, streams[i]);
    obj_allocator.deallocate(ptr_d, nvars, true, streams[i]);
    ev_deallocate.record(streams[i]);
    streams[i].synchronize();
  }

  return 0;
}
