#include "std_ivy/IvyTypeInfo.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyIterator.h"
#include "stream/IvyStream.h"


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
  constexpr unsigned int nvars = 10000000;
  IvyCudaConfig::set_max_num_GPU_blocks(1024);
  IvyCudaConfig::set_max_num_GPU_threads_per_block(1024*16);
  auto obj_allocator = std_mem::allocator<double>();
  auto obj_allocator_i = std_mem::allocator<int>();
  IvyGPUStream streams[nStreams]{
    GlobalGPUStream,
    IvyGPUStream(IvyGPUStream::StreamFlags::NonBlocking),
    IvyGPUStream(IvyGPUStream::StreamFlags::NonBlocking)
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

    int* ptr_i = nullptr;

    auto ptr_h = obj_allocator.allocate(nvars, false, streams[i]);
    printf("ptr_h = %p\n", ptr_h);
    ptr_h[0] = 1.0;
    ptr_h[1] = 2.0;
    ptr_h[2] = 3.0;
    printf("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator.deallocate(ptr_h, nvars, false, streams[i]);

    printf("Trying device...\n");

    IvyGPUEvent ev_allocate(IvyGPUEvent::EventFlags::Default); ev_allocate.record(streams[i]);
    auto ptr_d = obj_allocator.allocate(nvars, true, streams[i]);
    IvyGPUEvent ev_allocate_end(IvyGPUEvent::EventFlags::Default); ev_allocate_end.record(streams[i]);
    ev_allocate_end.synchronize();
    auto time_allocate = ev_allocate_end.elapsed_time(ev_allocate);
    printf("Allocation time = %f ms\n", time_allocate);
    printf("ptr_d = %p\n", ptr_d);

    ptr_h = obj_allocator.allocate(nvars, false, streams[i]);
    printf("ptr_h new = %p\n", ptr_h);

    {
      IvyBlockThread_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nvars)){
        IvyGPUEvent ev_set(IvyGPUEvent::EventFlags::Default); ev_set.record(streams[i]);
        kernel_set_doubles<<<nreq_blocks, nreq_threads_per_block, 0, streams[i]>>>(ptr_d, nvars, i);
        IvyGPUEvent ev_set_end(IvyGPUEvent::EventFlags::Default); ev_set_end.record(streams[i]);
        ev_set_end.synchronize();
        auto time_set = ev_set_end.elapsed_time(ev_set);
        printf("Set time = %f ms\n", time_set);
      }
    }

    IvyGPUEvent ev_transfer(IvyGPUEvent::EventFlags::Default); ev_transfer.record(streams[i]);
    obj_allocator.transfer(ptr_h, ptr_d, nvars, IvyMemoryHelpers::TransferDirection::DeviceToHost, streams[i]);
    IvyGPUEvent ev_transfer_end(IvyGPUEvent::EventFlags::Default); ev_transfer_end.record(streams[i]);
    ev_transfer_end.synchronize();
    auto time_transfer = ev_transfer_end.elapsed_time(ev_transfer);
    printf("Transfer time = %f ms\n", time_transfer);

    IvyGPUEvent ev_cp(IvyGPUEvent::EventFlags::Default); ev_cp.record(streams[i]);
    IvyMemoryHelpers::copy_data(ptr_i, ptr_h, 0, nvars, nvars, IvyMemoryHelpers::TransferDirection::HostToHost, streams[i]);
    IvyGPUEvent ev_cp_end(IvyGPUEvent::EventFlags::Default); ev_cp_end.record(streams[i]);
    ev_cp_end.synchronize();
    auto time_cp = ev_cp_end.elapsed_time(ev_cp);
    printf("Copy time = %f ms\n", time_cp);

    printf("ptr_h new values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    printf("ptr_i new values: %d, %d, %d\n", ptr_i[0], ptr_i[1], ptr_i[2]);
    obj_allocator.deallocate(ptr_h, nvars, false, streams[i]);
    obj_allocator_i.deallocate(ptr_i, nvars, false, streams[i]);

    IvyGPUEvent ev_deallocate(IvyGPUEvent::EventFlags::Default); ev_deallocate.record(streams[i]);
    obj_allocator.deallocate(ptr_d, nvars, true, streams[i]);
    IvyGPUEvent ev_deallocate_end(IvyGPUEvent::EventFlags::Default); ev_deallocate_end.record(streams[i]);
    ev_deallocate_end.synchronize();
    auto time_deallocate = ev_deallocate_end.elapsed_time(ev_allocate);
    printf("Deallocation time = %f ms\n", time_deallocate);

    streams[i].synchronize();
  }

  return 0;
}
