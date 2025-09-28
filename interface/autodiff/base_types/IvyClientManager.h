#ifndef IVYCLIENTMANAGER_H
#define IVYCLIENTMANAGER_H


#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyAlgorithm.h"
#include "autodiff/base_types/IvyBaseModifiable.h"


namespace IvyMath{
  template<typename T=void> class IvyClientManager;
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive_with_internal_memory<T, ENABLE_IF_TYPED_BASE_OF_IMPL(T, IvyClientManager<T>, T)>;
}
namespace IvyMath{
  template<typename T> class IvyClientManager{
    public:
      typedef T origin_t;
      typedef IvyBaseModifiable client_t;
      typedef std_mem::shared_ptr<client_t> client_ptr_t;
      typedef std_vec::vector<client_ptr_t> data_container;
      using allocator_data_container = std_mem::allocator<data_container>;

      friend class std_mem::kernel_generic_transfer_internal_memory<origin_t, origin_t>;

    protected:
      data_container clients_;

    __HOST_DEVICE__ bool transfer_internal_memory(std_ivy::IvyMemoryType const& new_mem_type, bool release_old){
      bool res = true;
      constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      auto stream = clients_.gpu_stream();
      operate_with_GPU_stream_from_pointer(
        stream, ref_stream,
        __ENCAPSULATE__(
          res &= allocator_data_container::transfer_internal_memory(&clients_, 1, def_mem_type, new_mem_type, ref_stream, release_old);
        )
      );
      return res;
    }

    public:
      __HOST_DEVICE__ IvyClientManager() = default;
      __HOST_DEVICE__ IvyClientManager(IvyClientManager const& other) = default;
      __HOST_DEVICE__ IvyClientManager(IvyClientManager&& other) : clients_(std_util::move(other.clients_)){}
      __HOST_DEVICE__ ~IvyClientManager() = default;

      __HOST_DEVICE__ IvyClientManager& operator=(IvyClientManager const& other) = default;
      __HOST_DEVICE__ IvyClientManager& operator=(IvyClientManager&& other){
        if (this != &other) clients_ = std_util::move(other.clients_);
        return *this;
      }

      template<typename U, ENABLE_IF_BASE_OF(client_t, U)>
      __HOST_DEVICE__ bool add_client(std_mem::shared_ptr<U> const& client){
        client_ptr_t base_ptr(client);
        auto it_end = clients_.end();
        if (std_algo::find(clients_.begin(), it_end, base_ptr) != it_end) return false;
        clients_.push_back(base_ptr);
        return true;
      }
      __HOST_DEVICE__ void update_clients_modified() const{
        for (auto const& client : clients_) client->set_modified(true);
      }

      __HOST_DEVICE__ data_container const& get_clients() const{ return clients_; }

      __HOST_DEVICE__ data_container* clone_clients_and_release(){
        data_container* tgt_bkp = nullptr;
        data_container* ptr_clients_ = &clients_;
        constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
        auto stream = clients_.gpu_stream();
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            allocator_data_container::allocate(tgt_bkp, 1, def_mem_type, ref_stream);
            data_container* new_clients_ = allocator_data_container::build(1, def_mem_type, ref_stream);
            IvyMemoryHelpers::transfer_memory(tgt_bkp, ptr_clients_, 1, def_mem_type, def_mem_type, ref_stream);
            IvyMemoryHelpers::transfer_memory(ptr_clients_, new_clients_, 1, def_mem_type, def_mem_type, ref_stream);
            allocator_data_container::deallocate(new_clients_, 1, def_mem_type, ref_stream);
          )
        );
        return tgt_bkp;
      }

      __HOST_DEVICE__ void restore_clients_and_reabsorb(data_container*& clients_bkp){
        if (!clients_bkp) return;
        data_container* ptr_clients_ = &clients_;
        constexpr auto def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
        auto stream = clients_.gpu_stream();
        operate_with_GPU_stream_from_pointer(
          stream, ref_stream,
          __ENCAPSULATE__(
            IvyMemoryHelpers::transfer_memory(ptr_clients_, clients_bkp, 1, def_mem_type, def_mem_type, ref_stream);
            allocator_data_container::deallocate(clients_bkp, 1, def_mem_type, ref_stream);
          )
        );
      }

  };

  class IvyClientlessManager{
    public:
      __HOST_DEVICE__ IvyClientlessManager() = default;
      __HOST_DEVICE__ IvyClientlessManager(IvyClientlessManager const& other) = default;
      __HOST_DEVICE__ IvyClientlessManager(IvyClientlessManager&& other) = default;
      __HOST_DEVICE__ ~IvyClientlessManager() = default;

      __HOST_DEVICE__ IvyClientlessManager& operator=(IvyClientlessManager const& other) = default;
      __HOST_DEVICE__ IvyClientlessManager& operator=(IvyClientlessManager&& other) = default;

      template<typename T>
      __HOST_DEVICE__ bool add_client(T const& client){ return false; }
      __HOST_DEVICE__ void update_clients_modified() const{}
  };
}
namespace std_ivy{
  using namespace IvyMath;
  template<typename T> class transfer_memory_primitive_with_internal_memory<T, ENABLE_IF_TYPED_BASE_OF_IMPL(T, IvyClientManager<T>, T)>{
  public:
    using base_t = allocation_type_properties<T>;
    using value_type = typename base_t::value_type;
    using pointer = typename base_t::pointer;
    using size_type = typename base_t::size_type;
    using kernel_type = kernel_generic_transfer_internal_memory<value_type, value_type>;

    static __HOST_DEVICE__ bool transfer_internal_memory(pointer ptr, IvyTypes::size_t const& n, IvyMemoryType const& ptr_mem_type, IvyMemoryType const& mem_type, IvyGPUStream& stream, bool release_old){
      bool res = true;
      if (IvyMemoryHelpers::run_acc_on_host(ptr_mem_type)){
        if (!run_kernel<kernel_type>(0, stream).parallel_1D(n, ptr, mem_type, release_old)){
          __PRINT_ERROR__("transfer_memory_primitive::transfer_internal_memory: Unable to call the acc. hardware kernel...\n");
          res = false;
        }
      }
      else{
        #define _CMD \
        for (size_type i=0; i<n; ++i) res &= kernel_type::transfer_internal_memory(ptr+i, mem_type, release_old);
#if defined(OPENMP_ENABLED)
        if (n>=NUM_CPU_THREADS_THRESHOLD){
          #pragma omp parallel for schedule(static)
          _CMD
        }
        else
#endif
        {
          _CMD
        }
        #undef _CMD
      }
      return res;
    }

    static __INLINE_FCN_RELAXED__ __HOST_DEVICE__ bool transfer(
      pointer& tgt, pointer const& src, size_type n,
      IvyMemoryType type_tgt, IvyMemoryType type_src,
      IvyGPUStream& stream
    ){
      if (!src) return false;
      bool res = true;
      /*
#if DEVICE_CODE == DEVICE_CODE_HOST
      printf("transfer_memory_primitive_with_internal_memory::transfer: type = %s | n=%llu | src = %p (%d), tgt = %p (%d)\n", typeid(T).name(), n, src, int(type_src), tgt, int(type_tgt));
#endif
      */
      constexpr IvyMemoryType def_mem_type = IvyMemoryHelpers::get_execution_default_memory();
      constexpr bool release_old = false; // We do not release existing memory from internal memory transfers in order to preserve src pointer.
      auto clients_src = src->clone_clients_and_release();
#if defined(__USE_CUDA__)
      if (def_mem_type==type_tgt && def_mem_type==type_src){
#endif
        res &= IvyMemoryHelpers::transfer_memory(tgt, src, n, type_tgt, type_src, stream);
        res &= transfer_internal_memory(tgt, n, type_tgt, type_tgt, stream, release_old);
#if defined(__USE_CUDA__)
      }
      else{
        pointer p_int = nullptr;
        res &= IvyMemoryHelpers::allocate_memory(p_int, n, def_mem_type, stream);
        res &= IvyMemoryHelpers::transfer_memory(p_int, src, n, def_mem_type, type_src, stream);
        res &= transfer_internal_memory(p_int, n, def_mem_type, type_tgt, stream, release_old);
        res &= IvyMemoryHelpers::transfer_memory(tgt, p_int, n, type_tgt, def_mem_type, stream);
        res &= IvyMemoryHelpers::free_memory(p_int, n, def_mem_type, stream);
      }
#endif
      src->restore_clients_and_reabsorb(clients_src);
      return res;
    }
  };
  template<typename T> class transfer_memory_primitive<IvyClientManager<T>> : public transfer_memory_primitive_with_internal_memory<IvyClientManager<T>, bool>{};
}


#endif
