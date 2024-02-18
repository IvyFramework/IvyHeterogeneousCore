#define __CUDA_DEBUG__

#include "std_ivy/IvyTypeInfo.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyMemory.h"
#include "std_ivy/vector/IvyVectorIterator.h"
//#include "std_ivy/IvyIterator.h"
#include "std_ivy/IvyChrono.h"
#include "std_ivy/IvyAlgorithm.h"


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
  //__CUDA_HOST_DEVICE__ ~dummy_D(){ printf("dummy D destructor...\n"); }
};
class dummy_N{
public:
  int n;
  dummy_N() = delete;
  __CUDA_HOST_DEVICE__ dummy_N(int k) : n(k){}
};


typedef std_mem::allocator<std_mem::unique_ptr<dummy_D>> uniqueptr_allocator;
typedef std_mem::allocator_traits<uniqueptr_allocator> uniqueptr_allocator_traits;
typedef std_mem::allocator<std_mem::shared_ptr<dummy_D>> sharedptr_allocator;
typedef std_mem::allocator_traits<sharedptr_allocator> sharedptr_allocator_traits;


struct set_doubles : public kernel_base_noprep_nofin{
  static __INLINE_FCN_RELAXED__ __CUDA_HOST_DEVICE__ void kernel_unified_unit(IvyTypes::size_t i, IvyTypes::size_t n, double* ptr, unsigned char is){
    ptr[i] = (i+2)*(is+1);
    if (i<3 || i==n-1) printf("ptr[%llu] = %f in stream %u\n", static_cast<unsigned long long int>(i), ptr[i], static_cast<unsigned int>(is));
  }
  static __CUDA_HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, double* ptr, unsigned char is){
    if (kernel_check_dims<set_doubles>::check_dims(i, n)) kernel_unified_unit(i, n, ptr, is);
  }
};

template<typename T, std_mem::IvyPointerType IPT> struct test_unifiedptr_fcn{
  static __CUDA_HOST_DEVICE__ void prepare(...){
    printf("Preparing for test_unifiedptr_fcn...\n");
  }
  static __CUDA_HOST_DEVICE__ void finalize(...){
    printf("Finalizing test_unifiedptr_fcn...\n");
  }
  static __CUDA_HOST_DEVICE__ void kernel(IvyTypes::size_t i, IvyTypes::size_t n, std_mem::IvyUnifiedPtr<T, IPT>* ptr){
    if (kernel_check_dims<test_unifiedptr_fcn<T, IPT>>::check_dims(i, n)){
      printf("Inside test_unifiedptr_fcn now...\n");
      //ptr->reset();
      printf("test_unifiedptr_fcn: get = %p\n", ptr->get());
    }
  }
};


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
  printf("kernel_test_unifiedptr_ptr: Device ptr ptrs: address, ptr, mem_type, stream, size, counter, exec_mem = %p, %p, %p, %p, %p, %p, %u\n", ptr, ptr->get(), ptr->get_memory_type_ptr(), ptr->gpu_stream(), ptr->size_ptr(), ptr->counter(), ptr->get_exec_memory_type());
  if (ptr->get()) printf("kernel_test_unifiedptr_ptr: Device ptr no. of copies, counter val., memory type, dummy_D.a: %llu, %llu, %u, %f\n", ptr->use_count(), *(ptr->counter()), ptr->get_memory_type(), (*ptr)->a);
  else printf("kernel_test_unifiedptr_ptr: Device ptr is null.\n");
}


template<typename T> void test_placement_new_seq(T const& val){
  printf("Testing placement new and delete...\n");
  T* ptr = nullptr;
  ptr = (T*) malloc(sizeof(T));
  new(ptr) T(val);
  printf("ptr = %p, *ptr = %f\n", ptr, __STATIC_CAST__(double, *ptr));
  ptr->~T();
  printf("ptr = %p, *ptr = %f\n", ptr, __STATIC_CAST__(double, *ptr));
  free(ptr);
  printf("ptr = %p\n", ptr);
}
template<typename T> void test_placement_new_dummy(T const& val){
  printf("Testing placement new and delete...\n");
  T* ptr = nullptr;
  ptr = (T*) malloc(sizeof(T));
  new(ptr) T(val);
  printf("ptr = %p, *ptr = %f\n", ptr, __STATIC_CAST__(double, ptr->a));
  ptr->~T();
  printf("ptr = %p, *ptr = %f\n", ptr, __STATIC_CAST__(double, ptr->a));
  free(ptr);
  printf("ptr = %p\n", ptr);
}

/**************/
/* Unit tests */
/**************/
template<unsigned int nsum, unsigned int nsum_serial> void utest_sumparallel(IvyGPUStream& stream, double* sum_vals){
  __PRINT_INFO__("|*** Benchmarking parallel and serial summation... ***|\n");

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

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}
void utest_IvyUnifiedPtr_basic(IvyGPUStream& stream){
  __PRINT_INFO__("|*** Benchmarking IvyUnifiedPtr basic functionality... ***|\n");

  std_mem::shared_ptr<dummy_D> ptr_shared = std_mem::make_shared<dummy_D>(IvyMemoryType::GPU, &stream, 1.);
  std_mem::shared_ptr<dummy_B> ptr_shared_copy = ptr_shared; ptr_shared_copy.reset(); ptr_shared_copy = ptr_shared;
  __PRINT_INFO__("ptr_shared no. of copies: %i\n", ptr_shared.use_count());
  __PRINT_INFO__("ptr_shared_copy no. of copies: %i\n", ptr_shared_copy.use_count());
  kernel_print_dummy_D<<<1, 1, 0, stream>>>(ptr_shared.get());
  kernel_print_dummy_B_as_D<<<1, 1, 0, stream>>>(ptr_shared_copy.get());

  printf("%s\n", typeid(ptr_shared).name());
  printf("%s\n", typeid(std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::element_type).name());
  printf("%s\n", typeid(std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::pointer).name());
  printf("%s\n", typeid(std_mem::pointer_traits<std_mem::shared_ptr<dummy_D>>::rebind<dummy_B>).name());

  std_mem::unique_ptr<dummy_D> ptr_unique = std_mem::make_unique<dummy_D>(IvyMemoryType::GPU, &stream, 1.);
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

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}
void utest_transfer_sharedptr(IvyGPUStream& stream){
  __PRINT_INFO__("|*** Benchmarking IvyUnifiedPtr transfer functionalities... ***|\n");

  std_mem::shared_ptr<dummy_D> h_shared_transferable = std_mem::make_shared<dummy_D>(IvyMemoryType::Host, &stream, 1.);
  __PRINT_INFO__("Allocating h_ptr_shared...\n");
  std_mem::shared_ptr<dummy_D>* h_ptr_shared = sharedptr_allocator_traits::allocate(1, IvyMemoryType::Host, stream);
  __PRINT_INFO__("Transferring h_shared_transferable to h_ptr_shared...\n");
  sharedptr_allocator_traits::transfer(h_ptr_shared, &h_shared_transferable, 1, IvyMemoryType::Host, IvyMemoryType::Host, stream);
  __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("h_ptr_shared no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_ptr_shared->use_count(), h_ptr_shared->get(), (*h_ptr_shared)->a);
  __PRINT_INFO__("Destroying h_ptr_shared...\n");
  sharedptr_allocator_traits::destroy(h_ptr_shared, 1, IvyMemoryType::Host, stream);
  __PRINT_INFO__("h_shared_transferable (after destroying h_ptr_shared) no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("Allocating d_ptr_shared...\n");
  std_mem::shared_ptr<dummy_D>* d_ptr_shared = sharedptr_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Allocating d_ptr_shared_dtransfer...\n");
  std_mem::shared_ptr<dummy_D>* d_ptr_shared_dtransfer = sharedptr_allocator_traits::allocate(1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Transfering h_shared_transferable to d_ptr_shared...\n");
  sharedptr_allocator_traits::transfer(d_ptr_shared, &h_shared_transferable, 1, IvyMemoryType::GPU, IvyMemoryType::Host, stream);
  stream.synchronize();
  __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("Running kernel test on d_ptr_shared...\n");
  kernel_test_unifiedptr_ptr<<<1, 1, 0, stream>>>(d_ptr_shared);
  run_kernel<test_unifiedptr_fcn<dummy_D, std_mem::IvyPointerType::shared>>(0, stream).parallel_1D(1, d_ptr_shared);
  stream.synchronize();
  //__PRINT_INFO__("Running kernel test on d_ptr_shared_dtransfer before transfer...\n");
  //kernel_test_unifiedptr_ptr<<<1, 1, 0, stream>>>(d_ptr_shared_dtransfer);
  //stream.synchronize();

  __PRINT_INFO__("Transfering d_ptr_shared to d_ptr_shared_dtransfer...\n");
  sharedptr_allocator_traits::transfer(d_ptr_shared_dtransfer, d_ptr_shared, 1, IvyMemoryType::GPU, IvyMemoryType::GPU, stream);
  stream.synchronize();
  __PRINT_INFO__("h_shared_transferable no. of copies, dummy_D addr., dummy_D.a: %llu, %p, %f\n", h_shared_transferable.use_count(), h_shared_transferable.get(), h_shared_transferable->a);
  __PRINT_INFO__("Running kernel test on d_ptr_shared_dtransfer...\n");
  kernel_test_unifiedptr_ptr<<<1, 1, 0, stream>>>(d_ptr_shared_dtransfer);
  run_kernel<test_unifiedptr_fcn<dummy_D, std_mem::IvyPointerType::shared>>(0, stream).parallel_1D(1, d_ptr_shared_dtransfer);
  stream.synchronize();
  __PRINT_INFO__("Destroying d_ptr_shared...\n");
  sharedptr_allocator_traits::destroy(d_ptr_shared, 1, IvyMemoryType::GPU, stream);
  __PRINT_INFO__("Destroying d_ptr_shared_dtransfer...\n");
  sharedptr_allocator_traits::destroy(d_ptr_shared_dtransfer, 1, IvyMemoryType::GPU, stream);

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}
template<unsigned int nvars> void utest_basic_alloc_copy_dealloc(IvyGPUStream& stream, unsigned int i_st){
  __PRINT_INFO__("|*** Benchmarking basic data allocation, transfer, and deallocation... ***|\n");

  typedef std_mem::allocator<double> obj_allocator;
  typedef std_mem::allocator<int> obj_allocator_i;

  int* ptr_i = nullptr;

  auto ptr_h = obj_allocator::build(nvars, IvyMemoryType::Host, stream);
  __PRINT_INFO__("ptr_h = %p\n", ptr_h);
  ptr_h[0] = 1.0;
  ptr_h[1] = 2.0;
  ptr_h[2] = 3.0;
  __PRINT_INFO__("ptr_h values: %f, %f, %f\n", ptr_h[0], ptr_h[1], ptr_h[2]);
  obj_allocator::destroy(ptr_h, nvars, IvyMemoryType::Host, stream);

  __PRINT_INFO__("Trying device...\n");

  IvyGPUEvent ev_build(IvyGPUEvent::EventFlags::Default); ev_build.record(stream);
  auto ptr_d = obj_allocator::build(nvars, IvyMemoryType::GPU, stream);
  IvyGPUEvent ev_build_end(IvyGPUEvent::EventFlags::Default); ev_build_end.record(stream);
  ev_build_end.synchronize();
  auto time_build = ev_build_end.elapsed_time(ev_build);
  __PRINT_INFO__("Construction time = %f ms\n", time_build);
  __PRINT_INFO__("ptr_d = %p\n", ptr_d);

  ptr_h = obj_allocator::build(nvars, IvyMemoryType::Host, stream);
  __PRINT_INFO__("ptr_h new = %p\n", ptr_h);

  {
    run_kernel<set_doubles> set_doubles_kernel(0, stream);
    IvyGPUEvent ev_set(IvyGPUEvent::EventFlags::Default); ev_set.record(stream);
    set_doubles_kernel.parallel_1D(nvars, ptr_d, i_st);
    IvyGPUEvent ev_set_end(IvyGPUEvent::EventFlags::Default); ev_set_end.record(stream);
    ev_set_end.synchronize();
    auto time_set = ev_set_end.elapsed_time(ev_set);
    __PRINT_INFO__("Set time = %f ms\n", time_set);
  }

  IvyGPUEvent ev_transfer(IvyGPUEvent::EventFlags::Default); ev_transfer.record(stream);
  obj_allocator::transfer(ptr_h, ptr_d, nvars, IvyMemoryType::Host, IvyMemoryType::GPU, stream);
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
  obj_allocator::destroy(ptr_h, nvars, IvyMemoryType::Host, stream);
  obj_allocator_i::destroy(ptr_i, nvars, IvyMemoryType::Host, stream);

  IvyGPUEvent ev_destroy(IvyGPUEvent::EventFlags::Default); ev_destroy.record(stream);
  obj_allocator::destroy(ptr_d, nvars, IvyMemoryType::GPU, stream);
  IvyGPUEvent ev_destroy_end(IvyGPUEvent::EventFlags::Default); ev_destroy_end.record(stream);
  ev_destroy_end.synchronize();
  auto time_destroy = ev_destroy_end.elapsed_time(ev_destroy);
  __PRINT_INFO__("Destruction time = %f ms\n", time_destroy);

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}
void utest_IvyVectorIterator_basic(IvyGPUStream& stream){
  using namespace std_ivy;

  __PRINT_INFO__("|*** Benchmarking IvyVectorIterator basic functionality... ***|\n");

  constexpr unsigned int ndata = 10;
  std_mem::unique_ptr<dummy_D> ptr_unique = std_mem::make_unique<dummy_D>(ndata, IvyMemoryType::Host, &stream, 1.);
  for (size_t i=0; i<ndata; i++){
    ptr_unique[i].a += i;
    printf("ptr_unique[%llu].a = %f\n", i, ptr_unique[i].a);
  }

  IvyVectorIteratorBuilder<dummy_D> it_builder;
  IvyVectorIteratorBuilder<dummy_D const> cit_builder;
  it_builder.reset(ptr_unique.get(), ndata, ptr_unique.get_memory_type(), ptr_unique.gpu_stream());
  cit_builder.reset(ptr_unique.get(), ndata, ptr_unique.get_memory_type(), ptr_unique.gpu_stream());

  printf("Use count of it_builder.chain_rend: %llu\n", it_builder.chain_rend.use_count());
  printf("Use count of it_builder.chain_front: %llu\n", it_builder.chain_front.use_count());
  printf("Use count of it_builder.chain_front->next(): %llu\n", it_builder.chain_front->next().use_count());
  printf("Use count of it_builder.chain_back: %llu\n", it_builder.chain_back.use_count());
  printf("Use count of it_builder.chain_end: %llu\n", it_builder.chain_end.use_count());

  {
    auto it_begin = it_builder.begin();
    auto it_end = it_builder.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it.a = %f\n", it->a);
      ++it;
    }
  }

  printf("Testing range-based for-loop...\n");
  for (auto const& obj:it_builder) printf("obj.a = %f\n", obj.a);

  it_builder.pop_back();
  ptr_unique.pop_back();
  printf("After pop_back...\n");
  {
    auto it_begin = it_builder.begin();
    auto it_end = it_builder.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it.a = %f\n", it->a);
      ++it;
    }
  }

  auto it_middle = it_builder.find_pointable(ptr_unique.get()+3);
  auto cit_middle = cit_builder.find_pointable(ptr_unique.get()+3);
  it_builder.erase(it_middle);
  ptr_unique.erase(3);
  printf("After erase...\n");
  {
    auto it_begin = it_builder.begin();
    auto it_end = it_builder.end();
    auto it = it_begin;
    while (it != it_end){
      printf("it.a = %f\n", it->a);
      ++it;
    }
    if (it_middle->is_valid()) printf("it_middle.a = %f\n", (*it_middle)->a);
    else printf("it_middle is invalid.\n");
  }

  stream.synchronize();
  __PRINT_INFO__("|\\-/|/-\\|\\-/|/-\\|\\-/|/-\\|\n");
}


int main(){
  constexpr unsigned char nStreams = 3;

  IvyGPUStream* streams[nStreams]{
    IvyStreamUtils::make_global_gpu_stream(),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking)
  };

  constexpr unsigned int nvars = 10000; // Can be up to ~700M
  constexpr unsigned int nsum = 1000000;
  constexpr unsigned int nsum_serial = 64;
  double sum_vals[nsum];
  for (unsigned int i = 0; i < nsum; i++) sum_vals[i] = i+1;

  test_placement_new_seq<int>(1);
  test_placement_new_seq<int>(5);
  test_placement_new_seq<double>(7);
  test_placement_new_seq<double>(9);
  test_placement_new_dummy<dummy_D>(dummy_D(1.));
  test_placement_new_dummy<dummy_D>(dummy_D(5.));

  for (unsigned char i = 0; i < nStreams; i++){
    __PRINT_INFO__("**********\n");

    auto& stream = *(streams[i]);
    __PRINT_INFO__("Stream %i (%p, %p, size in bytes = %d) computing...\n", i, &stream, stream.stream(), sizeof(&stream));

    utest_basic_alloc_copy_dealloc<nvars>(stream, i);
    utest_sumparallel<nsum, nsum_serial>(stream, sum_vals);
    utest_IvyUnifiedPtr_basic(stream);
    utest_transfer_sharedptr(stream);
    utest_IvyVectorIterator_basic(stream);

    __PRINT_INFO__("**********\n");
  }

  // Clean up local streams
  for (unsigned char i = 0; i < nStreams; i++) IvyStreamUtils::destroy_stream(streams[i]);

  return 0;
}
