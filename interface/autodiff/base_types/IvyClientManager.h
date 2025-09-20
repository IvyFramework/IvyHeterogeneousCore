#ifndef IVYCLIENTMANAGER_H
#define IVYCLIENTMANAGER_H


#include "std_ivy/IvyMemory.h"
#include "std_ivy/IvyVector.h"
#include "std_ivy/IvyTypeTraits.h"
#include "std_ivy/IvyAlgorithm.h"
#include "autodiff/base_types/IvyBaseModifiable.h"


namespace IvyMath{
  class IvyClientManager{
    public:
      typedef IvyBaseModifiable client_t;
      typedef std_mem::shared_ptr<client_t> client_ptr_t;
      typedef std_vec::vector<client_ptr_t> data_container;

    protected:
      data_container clients_;

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

      template<typename T, ENABLE_IF_BASE_OF(client_t, T)>
      __HOST_DEVICE__ bool add_client(std_mem::shared_ptr<T> const& client){
        client_ptr_t base_ptr(client);
        auto it_end = clients_.end();
        if (std_algo::find(clients_.begin(), it_end, base_ptr) != it_end) return false;
        clients_.push_back(base_ptr);
        return true;
      }
      __HOST_DEVICE__ void update_clients_modified() const{
        for (auto const& client : clients_) client->set_modified(true);
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


#endif
