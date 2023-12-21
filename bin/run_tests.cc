#include <std_ivy/IvyMemory.h>


__CUDA_GLOBAL__ void kernel_set_doubles(double* ptr, IvyBlockThread_t n, unsigned char is){
  IvyBlockThread_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n){
    ptr[i] = (i+2)*(is+1);
    printf("ptr[%i] = %f in stream %i\n", static_cast<int>(i), ptr[i], static_cast<int>(is));
  }
}

int main(){
  constexpr unsigned char nStreams = 3;
  auto obj_allocator = std_mem::allocator<double>();
  cudaStream_t streams[nStreams];
  streams[0] = cudaStreamLegacy;
  for (unsigned char i = 1; i < nStreams; i++) cudaStreamCreate(&streams[i]);

  for (unsigned char i = 0; i < nStreams; i++){
    printf("Stream %i (%p) computing...\n", i, streams[i]);

    auto ptr_h = obj_allocator.allocate(3);
    printf("ptr_h = %p\n", ptr_h);
    ptr_h[0] = 1.0;
    ptr_h[1] = 2.0;
    ptr_h[2] = 3.0;
    printf("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, 3);
    printf("Trying device...\n");

    auto ptr_d = obj_allocator.allocate(3, true, streams[i]);
    printf("ptr_d = %p\n", ptr_d);
    ptr_h = obj_allocator.allocate(3);
    printf("ptr_h new = %p\n", ptr_h);
    kernel_set_doubles<<<1, 3, 0, streams[i]>>>(ptr_d, 3, i);
    obj_allocator.transfer(ptr_h, ptr_d, 3, true, streams[i]);
    printf("ptr_h new values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, 3);
    obj_allocator.deallocate(ptr_d, 3, true, streams[i]);
  }

  for (unsigned char i = 1; i < nStreams; i++) cudaStreamDestroy(streams[i]);

  return 0;
}
