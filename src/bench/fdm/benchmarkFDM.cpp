#include "benchmarkFDM.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include "nrs.hpp"

#include "randomVector.hpp"
#include "kernelBenchmarker.hpp"
#include "omp.h"

occa::kernel benchmarkFDM(int Nelements, int Nq_e,
  int wordSize,
  bool useRAS,
  bool overlap,
  int verbosity,
  int Ntests,
  double elapsedTarget,
  bool requiresBenchmark)
{
  const auto Nq = Nq_e - 2;
  const auto N_e = Nq_e - 1;
  const auto N = Nq-1;
  const auto Np_e = Nq_e * Nq_e * Nq_e;

  occa::properties props = platform->kernelInfo + meshKernelProperties(N); // regular, non-extended mesh
  if(wordSize == 4) props["defines/pfloat"] = "float";
  else props["defines/pfloat"] = "dfloat";

  props["defines/p_Nq_e"] = Nq_e;
  props["defines/p_Np_e"] = Np_e;
  props["defines/p_overlap"] = overlap;

  if(useRAS){
    props["defines/p_restrict"] = 1;
  } else {
    props["defines/p_restrict"] = 0;
  }

  auto benchmarkFDMWithPrecision = [&](auto sampleWord){
    using FPType = decltype(sampleWord);
    const auto wordSize = sizeof(FPType);

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
    // only a single choice, no need to run benchmark
    if(kernelVariants.size() == 1 && !requiresBenchmark){
      auto newProps = props;
      newProps["defines/p_knl"] = kernelVariants.back();

      const std::string kernelName = "fusedFDM";
      const std::string ext = platform->serial ? ".c" : ".okl";
      const std::string fileName = installDir + "/okl/elliptic/" + kernelName + ext;

      return std::make_pair(platform->device.buildKernel(fileName, newProps, true), -1.0);
    }

    auto Sx   = randomVector<FPType>(Nelements * Nq_e * Nq_e);
    auto Sy   = randomVector<FPType>(Nelements * Nq_e * Nq_e);
    auto Sz   = randomVector<FPType>(Nelements * Nq_e * Nq_e);
    auto invL = randomVector<FPType>(Nelements * Np_e);
    auto Su   = randomVector<FPType>(Nelements * Np_e);
    auto u    = randomVector<FPType>(Nelements * Np_e);
    auto invDegree    = randomVector<dfloat>(Nelements * Np_e);

    // elementList[e] = e
    std::vector<int> elementList(Nelements);
    std::iota(elementList.begin(), elementList.end(), 0);
    auto o_elementList = platform->device.malloc(Nelements * sizeof(int), elementList.data());

    auto o_Sx = platform->device.malloc(Nelements * Nq_e * Nq_e * wordSize, Sx.data());
    auto o_Sy = platform->device.malloc(Nelements * Nq_e * Nq_e * wordSize, Sy.data());
    auto o_Sz = platform->device.malloc(Nelements * Nq_e * Nq_e * wordSize, Sz.data());
    auto o_invL = platform->device.malloc(Nelements * Np_e * wordSize, invL.data());
    auto o_Su = platform->device.malloc(Nelements * Np_e * wordSize, Su.data());
    auto o_u = platform->device.malloc(Nelements * Np_e * wordSize, u.data());
    auto o_invDegree = platform->device.malloc(Nelements * Np_e * sizeof(dfloat), invDegree.data());

    auto fdmKernelBuilder = [&](int kernelVariant) {
      auto newProps = props;
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

    auto printPerformanceInfo = [&](int kernelVariant, double elapsed, int Ntests, bool skipPrint) {

      double NGlobalElements = Nelements;
      MPI_Allreduce(MPI_IN_PLACE, &NGlobalElements, 1, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm);


      // print statistics
      const double GDOFPerSecond = (NGlobalElements * (N_e*N_e*N_e) / elapsed) / 1.e9;

      size_t bytesPerElem = (3 * Np_e + 3 * Nq_e * Nq_e) * wordSize;
      const double bw = (NGlobalElements * bytesPerElem / elapsed) / 1.e9;

      double flopsPerElem = 12 * Nq_e * Np_e + Np_e;
      const double gflops = (NGlobalElements * flopsPerElem / elapsed) / 1.e9;
      const int Nthreads =  omp_get_max_threads();

      if (platform->comm.mpiRank == 0 && !skipPrint) {
        if(verbosity > 1){
          std::cout << "MPItasks=" << platform->comm.mpiCommSize << " OMPthreads=" << Nthreads << " NRepetitions=" << Ntests;
        }
        if(verbosity > 0){
          std::cout << " N=" << N_e << " Nelements=" << NGlobalElements << " elapsed time=" << elapsed
                    << " wordSize=" << 8 * wordSize << " GDOF/s=" << GDOFPerSecond << " GB/s=" << bw
                    << " GFLOPS/s=" << gflops << " kernel=" << kernelVariant << "\n";
        }
      }
    };

    auto printCallBack = [&](int kernelVariant, double elapsed, int Ntests) {
      printPerformanceInfo(kernelVariant, elapsed, Ntests, verbosity < 2);
    };

    auto kernelAndTime = benchmarkKernel(fdmKernelBuilder, kernelRunner, printCallBack, kernelVariants, Ntests, elapsedTarget);

    int bestKernelVariant = static_cast<int>(kernelAndTime.first.properties()["defines/p_knl"]);
    
    // print only the fastest kernel
    if(verbosity == 1){
      printPerformanceInfo(bestKernelVariant, kernelAndTime.second, 0, false);
    }

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

  if(wordSize == sizeof(float)){
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