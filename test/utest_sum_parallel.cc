#include "common_test_defs.h"

#include "stream/IvyStream.h"
#include "std_ivy/IvyChrono.h"
#include "std_ivy/IvyAlgorithm.h"


using std_ivy::IvyMemoryType;


template<unsigned int nsum, unsigned int nsum_serial> void utest_sum_parallel(IvyGPUStream& stream, double* sum_vals){
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



int main(){
  constexpr unsigned char nStreams = 3;

  IvyGPUStream* streams[nStreams]{
    IvyStreamUtils::make_global_gpu_stream(),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking),
    IvyStreamUtils::make_stream<IvyGPUStream>(IvyGPUStream::StreamFlags::NonBlocking)
  };

  constexpr unsigned int nsum = 1000000;
  constexpr unsigned int nsum_serial = 64;
  double sum_vals[nsum];
  for (unsigned int i = 0; i < nsum; i++) sum_vals[i] = i+1;

  for (unsigned char i = 0; i < nStreams; i++){
    __PRINT_INFO__("**********\n");

    auto& stream = *(streams[i]);
    __PRINT_INFO__("Stream %i (%p, %p, size in bytes = %d) computing...\n", i, &stream, stream.stream(), sizeof(&stream));

    utest_sum_parallel<nsum, nsum_serial>(stream, sum_vals);

    __PRINT_INFO__("**********\n");
  }

  // Clean up local streams
  for (unsigned char i = 0; i < nStreams; i++) IvyStreamUtils::destroy_stream(streams[i]);

  return 0;
}