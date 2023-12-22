#include <std_ivy/IvyMemory.h>
#include "IvyCudaStream.h"


__CUDA_GLOBAL__ void kernel_set_doubles(double* ptr, IvyBlockThread_t n, unsigned char is){
  IvyBlockThread_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    ptr[i] = (i+2)*(is+1);
    if (i<3) printf("ptr[%i] = %f in stream %i\n", static_cast<int>(i), ptr[i], static_cast<int>(is));
  }
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

    IvyCudaEvent ev_allocate;
    IvyCudaEvent ev_set;
    IvyCudaEvent ev_transfer;
    IvyCudaEvent ev_deallocate;

    auto ptr_h = obj_allocator.allocate(nvars);
    printf("ptr_h = %p\n", ptr_h);
    ptr_h[0] = 1.0;
    ptr_h[1] = 2.0;
    ptr_h[2] = 3.0;
    printf("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, nvars);
    printf("Trying device...\n");

    auto ptr_d = obj_allocator.allocate(nvars, true, streams[i]);
    ev_allocate.record(streams[i]);
    ev_allocate.synchronize();
    printf("ptr_d = %p\n", ptr_d);
    ptr_h = obj_allocator.allocate(nvars);
    printf("ptr_h new = %p\n", ptr_h);
    kernel_set_doubles<<<1, nvars, 0, streams[i]>>>(ptr_d, nvars, i);
    ev_set.record(streams[i]);
    streams[i].wait(ev_set);
    obj_allocator.transfer(ptr_h, ptr_d, nvars, true, streams[i]);
    ev_transfer.record(streams[i]);
    streams[i].wait(ev_transfer);
    printf("ptr_h new values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, nvars);
    obj_allocator.deallocate(ptr_d, nvars, true, streams[i]);
    ev_deallocate.record(streams[i]);
    streams[i].synchronize();
  }

  return 0;
}
