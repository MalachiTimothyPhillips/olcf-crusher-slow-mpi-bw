#include "benchmarkFDM.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include "nrs.hpp"

#include "benchmarkUtils.hpp"
#include "kernelBenchmarker.hpp"
#include "omp.h"

occa::kernel benchmarkFDM(const occa::properties& baseProps, int Nelements, int Nq, int verbosity, int Ntests, double elapsedTarget)
{
  const auto N = Nq-1;
  const auto Np = Nq * Nq * Nq;

  // infer from baseProps what type to use for benchmarking
  const std::string floatingPointType = static_cast<std::string>(baseProps["defines/pfloat"]);
  const int overlap = static_cast<int>(baseProps["defines/p_overlap"]);
  const int useRAS = static_cast<int>(baseProps["defines/p_restrict"]);

  auto benchmarkFDMWithPrecision = [&](auto sampleWord){
    using FPType = decltype(sampleWord);
    const auto wordSize = sizeof(FPType);
    auto Sx   = randomVector<FPType>(Nelements * Nq * Nq);
    auto Sy   = randomVector<FPType>(Nelements * Nq * Nq);
    auto Sz   = randomVector<FPType>(Nelements * Nq * Nq);
    auto invL = randomVector<FPType>(Nelements * Np);
    auto Su   = randomVector<FPType>(Nelements * Np);
    auto u    = randomVector<FPType>(Nelements * Np);
    auto invDegree    = randomVector<FPType>(Nelements * Np);

    // elementList[e] = e
    std::vector<int> elementList(Nelements);
    std::iota(elementList.begin(), elementList.end(), 0);
    auto o_elementList = platform->device.malloc(Nelements * sizeof(int), elementList.data());

    auto o_Sx = platform->device.malloc(Nelements * Nq * Nq * wordSize, Sx.data());
    auto o_Sy = platform->device.malloc(Nelements * Nq * Nq * wordSize, Sy.data());
    auto o_Sz = platform->device.malloc(Nelements * Nq * Nq * wordSize, Sz.data());
    auto o_invL = platform->device.malloc(Nelements * Np * wordSize, invL.data());
    auto o_Su = platform->device.malloc(Nelements * Np * wordSize, Su.data());
    auto o_u = platform->device.malloc(Nelements * Np * wordSize, u.data());
    auto o_invDegree = platform->device.malloc(Nelements * Np * wordSize, invDegree.data());

    constexpr int Nkernels = 5;
    std::vector<int> kernelVariants;
    if (platform->serial) {
      kernelVariants.push_back(0);
    }
    else {
      for (int knl = 0; knl < Nkernels; ++knl) {
        kernelVariants.push_back(knl);
      }
    }

    const std::string installDir(getenv("NEKRS_HOME"));

    auto fdmKernelBuilder = [&](int kernelVariant) {
      auto newProps = baseProps;
      newProps["defines/p_knl"] = kernelVariant;

      const std::string kernelName = "fusedFDM";
      const std::string ext = platform->serial ? ".c" : ".okl";
      const std::string fileName = installDir + "/okl/elliptic/" + kernelName + ext;

      return platform->device.buildKernel(fileName, newProps, true);
    };

    auto kernelRunner = [&](occa::kernel &kernel) { 
      if(useRAS){
        if(!overlap){
          kernel(Nelements, o_Su,o_Sx,o_Sy,o_Sz,o_invL,o_invDegree,o_u);
        } else {
          kernel(Nelements, o_elementList, o_Su,o_Sx,o_Sy,o_Sz,o_invL,o_invDegree,o_u);
        }
      } else {
        if(!overlap){
          kernel(Nelements, o_Su,o_Sx,o_Sy,o_Sz,o_invL,o_u);
        } else {
          kernel(Nelements, o_elementList, o_Su,o_Sx,o_Sy,o_Sz,o_invL,o_u);
        }
      }
    };

    auto printPerformanceInfo = [&](int kernelVariant, double elapsed, int Ntests) {

      double NGlobalElements = Nelements;
      MPI_Allreduce(MPI_IN_PLACE, &NGlobalElements, 1, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm);


      // print statistics
      const double GDOFPerSecond = (NGlobalElements * (N* N * N) / elapsed) / 1.e9;

      size_t bytesPerElem = (3 * Np + 3 * Nq * Nq) * wordSize;
      const double bw = (NGlobalElements * bytesPerElem / elapsed) / 1.e9;

      double flopsPerElem = 12 * Nq * Np + Np;
      const double gflops = (NGlobalElements * flopsPerElem / elapsed) / 1.e9;
      const int Nthreads =  omp_get_max_threads();

      if (platform->comm.mpiRank == 0) {
        if(verbosity > 1){
          std::cout << "MPItasks=" << platform->comm.mpiCommSize << " OMPthreads=" << Nthreads << " NRepetitions=" << Ntests;
        }
        if(verbosity > 0){
          std::cout << " N=" << N << " Nelements=" << NGlobalElements << " elapsed time=" << elapsed
                    << " wordSize=" << 8 * wordSize << " GDOF/s=" << GDOFPerSecond << " GB/s=" << bw
                    << " GFLOPS/s=" << gflops << " kernel=" << kernelVariant << "\n";
        }
      }
    };

    auto kernelAndTime = benchmarkKernel(fdmKernelBuilder, kernelRunner, printPerformanceInfo, kernelVariants, Ntests, elapsedTarget);

    free(o_Sx);
    free(o_Sy);
    free(o_Sz);
    free(o_invL);
    free(o_Su);
    free(o_u);
    free(o_invDegree);
    free(o_elementList);

    return kernelAndTime;

  };

  occa::kernel kernel;

  if(floatingPointType.find("float") != std::string::npos){
    float p = 0.0;
    auto kernelAndTime = benchmarkFDMWithPrecision(p);
    kernel = kernelAndTime.first;
  } else {
    double p = 0.0;
    auto kernelAndTime = benchmarkFDMWithPrecision(p);
    kernel = kernelAndTime.first;
  }

  return kernel;

}