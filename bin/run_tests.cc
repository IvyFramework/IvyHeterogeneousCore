#include "std_ivy/IvyTypeInfo.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyIterator.h"
#include "std_ivy/IvyChrono.h"
#include "std_ivy/IvyAlgorithm.h"
#include "stream/IvyStream.h"
#include "IvyMemoryHelpers.h"


using std_ivy::IvyMemoryType;


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
class dummy_N{
public:
  int n;
  dummy_N() = delete;
  __CUDA_HOST_DEVICE__ dummy_N(int k) : n(k){}
};


__CUDA_GLOBAL__ void kernel_set_doubles(double* ptr, IvyTypes::size_t n, unsigned char is){
  IvyTypes::size_t i = 0;
  get_kernel_call_dims_1D(i);
  if (i < n){
    ptr[i] = (i+2)*(is+1);
    if (i<3 || i==n-1) printf("ptr[%llu] = %f in stream %u\n", static_cast<unsigned long long int>(i), ptr[i], static_cast<unsigned int>(is));
  }
}


__CUDA_HOST_DEVICE__ void print_dummy_D(dummy_D* ptr){
  printf("dummy_D address = %p, a = %f\n", ptr, ptr->a);
}
__CUDA_GLOBAL__ void kernel_print_dummy_D(dummy_D* ptr){
  print_dummy_D(ptr);
}

__CUDA_HOST_DEVICE__ void print_dummy_B_as_D(dummy_B* ptr){
  printf("dummy_B address = %p, a = %f\n", ptr, __STATIC_CAST__(dummy_D*, ptr)->a);
}
__CUDA_GLOBAL__ void kernel_print_dummy_B_as_D(dummy_B* ptr){
  print_dummy_B_as_D(ptr);
}
template<typename T> __CUDA_GLOBAL__ void kernel_test_unifiedptr_ptr(T* ptr){
  printf("Calling kernel_test_unifiedptr_ptr...\n");
  printf("kernel_test_unifiedptr_ptr: Device ptr ptrs: address, ptr, mem_type, size, counter, exec_mem = %p, %p, %p, %p, %p, %u\n", ptr, ptr->get(), ptr->get_memory_type_ptr(), ptr->size_ptr(), ptr->counter(), ptr->get_exec_memory_type());
  printf("kernel_test_unifiedptr_ptr: Device ptr no. of copies, counter val., memory type, dummy_D.a: %llu, %llu, %u, %f\n", ptr->use_count(), *(ptr->counter()), ptr->get_memory_type(), (*ptr)->a);
}


int main(){
  constexpr unsigned char nStreams = 3;
  constexpr unsigned int nvars = 10000; // Can be up to ~700M
  IvyCudaConfig::set_max_num_GPU_blocks(1024);
  IvyCudaConfig::set_max_num_GPU_threads_per_block(1024);
  typedef std_mem::allocator<std_mem::unique_ptr<dummy_D>> uniqueptr_allocator;
  typedef std_mem::allocator_traits<uniqueptr_allocator> uniqueptr_allocator_traits;
  typedef std_mem::allocator<std_mem::shared_ptr<dummy_D>> sharedptr_allocator;
  typedef std_mem::allocator_traits<sharedptr_allocator> sharedptr_allocator_traits;
  typedef std_mem::allocator<double> obj_allocator;
  typedef std_mem::allocator<int> obj_allocator_i;
  IvyGPUStream* streams[nStreams]{
    new GlobalGPUStream,
    new IvyGPUStream(IvyGPUStream::StreamFlags::NonBlocking),
    new IvyGPUStream(IvyGPUStream::StreamFlags::NonBlocking)
  };

  constexpr int nsum = 1000000;
  constexpr int nsum_serial = 64;
  double sum_vals[nsum];
  for (int i = 0; i < nsum; i++) sum_vals[i] = i+1;

  for (unsigned char i = 0; i < nStreams; i++){
    __PRINT_INFO__("**********\n");

    auto& stream = *(streams[i]);
    __PRINT_INFO__("Stream %i (%p) computing...\n", i, stream.stream());

    __PRINT_INFO__("\t- Benchmarking parallel and serial summation...\n");

    IvyGPUEvent ev_sum(IvyGPUEvent::EventFlags::Default); ev_sum.record(stream);
    double sum_p = std_algo::add_parallel<double>(sum_vals, nsum, nsum_serial, IvyMemoryType::Host, stream);
    IvyGPUEvent ev_sum_end(IvyGPUEvent::EventFlags::Default); ev_sum_end.record(stream);
    ev_sum_end.synchronize();
    auto time_sum = ev_sum_end.elapsed_time(ev_sum);
    __PRINT_INFO__("Sum_parallel time = %f ms\n", time_sum);
    __PRINT_INFO__("Sum parallel = %f\n", sum_p);

    auto time_sum_s = std_chrono::high_resolution_clock::now();
    double sum_s = std_algo::add_serial<double>(sum_vals, nsum);
    auto time_sum_s_end = std_chrono::high_resolution_clock::now();
    auto time_sum_s_ms = std_chrono::duration_cast<std_chrono::microseconds>(time_sum_s_end - time_sum_s).count()/1000.;
    __PRINT_INFO__("Sum_serial time = %f ms\n", time_sum_s_ms);
    __PRINT_INFO__("Sum serial = %f\n", sum_s);

    __PRINT_INFO__("\t- Testing IvyUniquePtr...\n");

    std_mem::shared_ptr<dummy_D> ptr_shared = std_mem::make_shared<dummy_D>(IvyMemoryType::Device, &stream, 1.);
    std_mem::shared_ptr<dummy_B> ptr_shared_copy = ptr_shared; ptr_shared_copy.reset(); ptr_shared_copy = ptr_shared;
    __PRINT_INFO__("ptr_shared no. of copies: %i\n", ptr_shared.use_count());
    __PRINT_INFO__("ptr_shared_copy no. of copies: %i\n", ptr_shared_copy.use_count());
    kernel_print_dummy_D<<<1, 1, 0, stream>>>(ptr_shared.get());
    kernel_print_dummy_B_as_D<<<1, 1, 0, stream>>>(ptr_shared_copy.get());

    printf("%s\n", typeid(ptr_shared).name());
    printf("%s\n", typeid(std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::element_type).name());
    printf("%s\n", typeid(std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::pointer).name());
    printf("%s\n", typeid(std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::rebind<dummy_B>).name());

    std_mem::unique_ptr<dummy_D> ptr_unique = std_mem::make_unique<dummy_D>(IvyMemoryType::Device, &stream, 1.);
    std_mem::unique_ptr<dummy_B> ptr_unique_copy = ptr_unique;
    __PRINT_INFO__("ptr_unique no. of copies: %i\n", ptr_unique.use_count());
    __PRINT_INFO__("ptr_unique_copy no. of copies: %i\n", ptr_unique_copy.use_count());
    kernel_print_dummy_B_as_D<<<1, 1, 0, stream>>>(ptr_shared_copy.get());
    ptr_unique_copy.reset();
    __PRINT_INFO__("ptr_unique_copy no. of copies after reset: %i\n", ptr_unique_copy.use_count());

    __PRINT_INFO__("\t- Testing IvyUnifiedPtr memory allocation...\n");
    std_mem::unique_ptr<dummy_D> h_unique_transferable = std_mem::make_unique<dummy_D>(IvyMemoryType::Host, &stream, 1.);
    __PRINT_INFO__("Allocating h_ptr_unique...\n");
    std_mem::unique_ptr<dummy_D>* h_ptr_unique = uniqueptr_allocator_traits::allocate(1, IvyMemoryType::Host, stream);
    uniqueptr_allocator_traits::transfer(h_ptr_unique, &h_unique_transferable, 1, IvyMemoryType::Host, IvyMemoryType::Host, stream);
    __PRINT_INFO__("h_unique_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_unique_transferable.use_count(), h_unique_transferable.get(), h_unique_transferable->a);
    __PRINT_INFO__("h_ptr_unique no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_ptr_unique->use_count(), h_ptr_unique->get(), (*h_ptr_unique)->a);
    __PRINT_INFO__("Deallocating h_ptr_unique...\n");
    uniqueptr_allocator_traits::deallocate(h_ptr_unique, 1, IvyMemoryType::Host, stream);

    std_mem::shared_ptr<dummy_D> h_shared_transferable = std_mem::make_shared<dummy_D>(IvyMemoryType::Host, &stream, 1.);
    __PRINT_INFO__("Allocating h_ptr_shared...\n");
    std_mem::shared_ptr<dummy_D>* h_ptr_shared = sharedptr_allocator_traits::allocate(1, IvyMemoryType::Host, stream);
    sharedptr_allocator_traits::transfer(h_ptr_shared, &h_shared_transferable, 1, IvyMemoryType::Host, IvyMemoryType::Host, stream);
    __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
    __PRINT_INFO__("h_ptr_shared no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_ptr_shared->use_count(), h_ptr_shared->get(), (*h_ptr_shared)->a);
    __PRINT_INFO__("Deallocating h_ptr_shared...\n");
    sharedptr_allocator_traits::deallocate(h_ptr_shared, 1, IvyMemoryType::Host, stream);
    __PRINT_INFO__("h_shared_transferable (after dealloc.) no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
    __PRINT_INFO__("Allocating d_ptr_shared...\n");
    std_mem::shared_ptr<dummy_D>* d_ptr_shared = sharedptr_allocator_traits::allocate(1, IvyMemoryType::Device, stream);
    sharedptr_allocator_traits::transfer(d_ptr_shared, &h_shared_transferable, 1, IvyMemoryType::Device, IvyMemoryType::Host, stream);
    __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
    stream.synchronize();
    kernel_test_unifiedptr_ptr<<<1, 1, 0, stream>>>(d_ptr_shared);
    stream.synchronize();
    __PRINT_INFO__("Deallocating d_ptr_shared...\n");
    sharedptr_allocator_traits::deallocate(d_ptr_shared, 1, IvyMemoryType::Device, stream);

    __PRINT_INFO__("\t- Testing basic data allocation, transfer, and deallocation...\n");

    int* ptr_i = nullptr;

    auto ptr_h = obj_allocator::allocate(nvars, IvyMemoryType::Host, stream);
    __PRINT_INFO__("ptr_h = %p\n", ptr_h);
    ptr_h[0] = 1.0;
    ptr_h[1] = 2.0;
    ptr_h[2] = 3.0;
    __PRINT_INFO__("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
    obj_allocator::deallocate(ptr_h, nvars, IvyMemoryType::Host, stream);

    __PRINT_INFO__("Trying device...\n");

    IvyGPUEvent ev_allocate(IvyGPUEvent::EventFlags::Default); ev_allocate.record(stream);
    auto ptr_d = obj_allocator::allocate(nvars, IvyMemoryType::Device, stream);
    IvyGPUEvent ev_allocate_end(IvyGPUEvent::EventFlags::Default); ev_allocate_end.record(stream);
    ev_allocate_end.synchronize();
    auto time_allocate = ev_allocate_end.elapsed_time(ev_allocate);
    __PRINT_INFO__("Allocation time = %f ms\n", time_allocate);
    __PRINT_INFO__("ptr_d = %p\n", ptr_d);

    ptr_h = obj_allocator::allocate(nvars, IvyMemoryType::Host, stream);
    __PRINT_INFO__("ptr_h new = %p\n", ptr_h);

    {
      IvyBlockThreadDim_t nreq_blocks, nreq_threads_per_block;
      if (IvyCudaConfig::check_GPU_usable(nreq_blocks, nreq_threads_per_block, nvars)){
        IvyGPUEvent ev_set(IvyGPUEvent::EventFlags::Default); ev_set.record(stream);
        kernel_set_doubles<<<nreq_blocks, nreq_threads_per_block, 0, stream>>>(ptr_d, nvars, i);
        IvyGPUEvent ev_set_end(IvyGPUEvent::EventFlags::Default); ev_set_end.record(stream);
        ev_set_end.synchronize();
        auto time_set = ev_set_end.elapsed_time(ev_set);
        __PRINT_INFO__("Set time = %f ms\n", time_set);
      }
    }

    IvyGPUEvent ev_transfer(IvyGPUEvent::EventFlags::Default); ev_transfer.record(stream);
    obj_allocator::transfer(ptr_h, ptr_d, nvars, IvyMemoryType::Host, IvyMemoryType::Device, stream);
    IvyGPUEvent ev_transfer_end(IvyGPUEvent::EventFlags::Default); ev_transfer_end.record(stream);
    ev_transfer_end.synchronize();
    auto time_transfer = ev_transfer_end.elapsed_time(ev_transfer);
    __PRINT_INFO__("Transfer time = %f ms\n", time_transfer);

    IvyGPUEvent ev_cp(IvyGPUEvent::EventFlags::Default); ev_cp.record(stream);
    IvyMemoryHelpers::copy_data(ptr_i, ptr_h, 0, nvars, nvars, IvyMemoryType::Host, IvyMemoryType::Host, stream);
    IvyGPUEvent ev_cp_end(IvyGPUEvent::EventFlags::Default); ev_cp_end.record(stream);
    ev_cp_end.synchronize();
    auto time_cp = ev_cp_end.elapsed_time(ev_cp);
    __PRINT_INFO__("Copy time = %f ms\n", time_cp);

    __PRINT_INFO__("ptr_h new values: %f, %f, %f, ..., %f\n", ptr_h[0], ptr_h[1], ptr_h[2], ptr_h[nvars-1]);
    __PRINT_INFO__("ptr_i new values: %d, %d, %d, ..., %d\n", ptr_i[0], ptr_i[1], ptr_i[2], ptr_i[nvars-1]);
    obj_allocator::deallocate(ptr_h, nvars, IvyMemoryType::Host, stream);
    obj_allocator_i::deallocate(ptr_i, nvars, IvyMemoryType::Host, stream);

    IvyGPUEvent ev_deallocate(IvyGPUEvent::EventFlags::Default); ev_deallocate.record(stream);
    obj_allocator::deallocate(ptr_d, nvars, IvyMemoryType::Device, stream);
    IvyGPUEvent ev_deallocate_end(IvyGPUEvent::EventFlags::Default); ev_deallocate_end.record(stream);
    ev_deallocate_end.synchronize();
    auto time_deallocate = ev_deallocate_end.elapsed_time(ev_allocate);
    __PRINT_INFO__("Deallocation time = %f ms\n", time_deallocate);

    stream.synchronize();

    __PRINT_INFO__("**********\n");
  }

  // Clean up local streams
  for (unsigned char i = 0; i < nStreams; i++) delete streams[i];

  return 0;
}
