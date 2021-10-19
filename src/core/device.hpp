#ifndef device_hpp_
#define device_hpp_
#include <string>
#include <occa.hpp>
#include <mpi.h>
#include "nrssys.hpp"

class setupAide;

class device_t {
  public:
    device_t(setupAide& options, MPI_Comm commg, MPI_Comm comm);
    MPI_Comm comm;
    occa::memory malloc(const size_t Nbytes, const void* src = nullptr, const occa::properties& properties = occa::properties());
    occa::memory malloc(const size_t Nbytes, const occa::properties& properties);
    occa::memory malloc(const hlong Nwords, const dlong wordSize, occa::memory src);
    occa::memory malloc(const hlong Nwords, const dlong wordSize);

    occa::memory mallocHost(const size_t Nbytes);

    int id() const { return _device_id; }
    const occa::device& occaDevice() const { return _device; }
    std::string mode() const { return _device.mode(); }
    occa::device& occaDevice() { return _device; }
    void finish() { _device.finish(); }
    occa::kernel buildKernel(const std::string &fullPath,
                             const occa::properties &props,
                             std::string suffix = std::string(),
                             bool buildRank0 = false) const;
    occa::kernel buildNativeKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props) const;
    occa::kernel doBuildKernel(const std::string &filename,
                             const std::string &kernelName,
                             const occa::properties &props,
                             std::string suffix = std::string()) const;
    occa::kernel doBuildKernel(const std::string &fullPath,
                             const occa::properties &props,
                             std::string suffix = std::string()) const;
  private:
    occa::device _device;
    int _device_id;
};
#endif
