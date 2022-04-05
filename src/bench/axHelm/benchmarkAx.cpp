#include "benchmarkAx.hpp"
#include <vector>
#include <iostream>
#include "mesh.h"
#include "nrs.hpp"

#include "kernelBenchmarker.hpp"
#include "benchmarkUtils.hpp"
#include "omp.h"

occa::kernel benchmarkAx(const occa::properties& baseProps, const mesh_t& mesh, bool verbose, int Ntests, double elapsedTarget)
{
  const auto Nelements = mesh.Nelements;
  const auto Nq = mesh.Nq;
  const auto N = Nq-1;
  const auto Np = Nq * Nq * Nq;

  // infer from baseProps what type to use for benchmarking
  const std::string floatingPointType = static_cast<std::string>(baseProps["defines/dfloat"]);

  auto benchmarkAxWithPrecision = [&](auto sampleWord){
    using FPType = decltype(sampleWord);
    const auto wordSize = sizeof(FPType);
    const int Ndim = 1;
    const int Np_g = Np;
    const int Ng = N;
    const int Nq_g = Nq;
    constexpr int p_Nggeo {7};

    auto DrV    = randomVector<FPType>(Nq * Nq);
    auto ggeo   = randomVector<FPType>(Np_g * Nelements * p_Nggeo);
    auto q      = randomVector<FPType>((Ndim * Np) * Nelements);
    auto Aq     = randomVector<FPType>((Ndim * Np) * Nelements);
    auto exyz   = randomVector<FPType>((3 * Np_g) * Nelements);
    auto gllwz  = randomVector<FPType>(2 * Nq_g);
    auto lambda = randomVector<FPType>(2 * Np * Nelements);

    std::vector<dlong> elementList(Nelements);
    for(int e = 0; e < Nelements; ++e){
      elementList[e] = e;
    }
    auto o_elementList = platform->device.malloc(Nelements * sizeof(dlong), elementList.data());

    auto o_D = platform->device.malloc(Nq * Nq * wordSize, DrV.data());
    auto o_S = o_D;
    auto o_ggeo = platform->device.malloc(Np_g * Nelements * p_Nggeo * wordSize, ggeo.data());
    auto o_q = platform->device.malloc((Ndim * Np) * Nelements * wordSize, q.data());
    auto o_Aq = platform->device.malloc((Ndim * Np) * Nelements * wordSize, Aq.data());
    auto o_exyz = platform->device.malloc((3 * Np_g) * Nelements * wordSize, exyz.data());
    auto o_gllwz = platform->device.malloc(2 * Nq_g * wordSize, gllwz.data());

    auto o_lambda = platform->device.malloc(2 * Np * Nelements * wordSize, lambda.data());

    constexpr int Nkernels = 1;
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

    auto axKernelBuilder = [&](int kernelVariant) {
      auto newProps = baseProps;
      newProps["defines/p_knl"] = kernelVariant;

      const std::string kernelName = "ellipticPartialAxHex3D";
      const std::string ext = platform->serial ? ".c" : ".okl";
      const std::string fileName = installDir + "/okl/elliptic/" + kernelName + ext;

      return platform->device.buildKernel(fileName, newProps, true);
    };

    auto kernelRunner = [&](occa::kernel &kernel) {
      const int loffset = 0;
      const int offset = Nelements * Np;
      kernel(Nelements, offset, loffset, o_elementList, o_ggeo, o_D, o_S, o_lambda, o_q, o_Aq);
    };

    auto printPerformanceInfo = [&](int kernelVariant, double elapsed, int Ntests) {

      const bool BKmode = true;

      double NGlobalElements = Nelements;
      MPI_Allreduce(MPI_IN_PLACE, &NGlobalElements, 1, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm);

      // print statistics
      const dfloat GDOFPerSecond = (NGlobalElements * Ndim * (N * N * N) / elapsed) / 1.e9;

      size_t bytesMoved = Ndim * 2 * Np * wordSize; // x, Ax
      bytesMoved += 6 * Np_g * wordSize; // geo
      if(!BKmode) bytesMoved += 3 * Np * wordSize; // lambda1, lambda2, Jw
      const double bw = (NGlobalElements * bytesMoved / elapsed) / 1.e9;

      double flopCount = Np * 12 * Nq + 15 * Np;
      if(!BKmode) flopCount += 5 * Np;
      const double gflops = Ndim * (flopCount * NGlobalElements / elapsed) / 1.e9;
      const int Nthreads =  omp_get_max_threads();

      if(platform->comm.mpiRank == 0 && verbose){
        std::cout << "MPItasks=" << platform->comm.mpiCommSize
                  << " OMPthreads=" << Nthreads
                  << " NRepetitions=" << Ntests
                  << " Ndim=" << Ndim
                  << " N=" << N
                  << " Ng=" << Ng
                  << " Nelements=" << NGlobalElements
                  << " elapsed time=" << elapsed
                  << " bkMode=" << BKmode
                  << " wordSize=" << 8*wordSize
                  << " GDOF/s=" << GDOFPerSecond
                  << " GB/s=" << bw
                  << " GFLOPS/s=" << gflops
                  << "\n";
      }
    };

    auto kernel = benchmarkKernel(axKernelBuilder, kernelRunner, printPerformanceInfo, kernelVariants, Ntests, elapsedTarget);

    free(o_D);
    free(o_S);
    free(o_ggeo);
    free(o_q);
    free(o_Aq);
    free(o_exyz);
    free(o_gllwz);
    free(o_lambda);

    return kernel;

  };

  occa::kernel kernel;

  if(floatingPointType.find("float") != std::string::npos){
    float p = 0.0;
    auto kernelAndTime = benchmarkAxWithPrecision(p);
    kernel = kernelAndTime.first;
  } else {
    double p = 0.0;
    auto kernelAndTime = benchmarkAxWithPrecision(p);
    kernel = kernelAndTime.first;
  }

  return kernel;

}