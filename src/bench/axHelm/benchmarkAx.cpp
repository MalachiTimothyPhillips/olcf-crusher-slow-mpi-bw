#include "benchmarkAx.hpp"
#include <vector>
#include <iostream>
#include <numeric>
#include "nrs.hpp"

#include "kernelBenchmarker.hpp"
#include "randomVector.hpp"
#include "omp.h"

template <typename T>
occa::kernel benchmarkAx(int Nelements,
                         int Nq,
                         int Ng,
                         bool constCoeff,
                         bool poisson,
                         bool computeGeom,
                         int wordSize,
                         int Ndim,
                         int verbosity,
                         T NtestsOrTargetTime,
                         bool requiresBenchmark)
{
  const auto N = Nq-1;
  const auto Np = Nq * Nq * Nq;
  const auto Nq_g = Ng+1;
  const int Np_g = Nq_g * Nq_g * Nq_g;

  occa::properties props = platform->kernelInfo + meshKernelProperties(N);
  if(wordSize == 4) props["defines/dfloat"] = "float";
  if(Ng != N) {
    props["defines/p_Nq_g"] = Nq_g;
    props["defines/p_Np_g"] = Np_g;
  }
  if(poisson) props["defines/p_poisson"] = 1;

  std::string kernelName = "elliptic";
  if(Ndim > 1) kernelName += "Block";
  kernelName += "PartialAx";
  if(!constCoeff) kernelName += "Coeff";
  if(Ng != N) {
    if(computeGeom) {
      if(Ng == 1) {
        kernelName += "Trilinear";
      } else {
        printf("Unsupported g-order=%d\n", Ng);
        exit(1);
      }
    } else {
      printf("for now g-order != p-order requires --computeGeom!\n");
      exit(1);
      kernelName += "Ngeom";
    } 
  }
  kernelName += "Hex3D";
  if (Ndim > 1) kernelName += "_N" + std::to_string(Ndim);

  auto benchmarkAxWithPrecision = [&](auto sampleWord){
    using FPType = decltype(sampleWord);
    const auto wordSize = sizeof(FPType);
    constexpr int p_Nggeo {7};

    int Nkernels = 1;
    if(kernelName == "ellipticPartialAxHex3D") Nkernels = 8;
    std::vector<int> kernelVariants;
    if (platform->serial) {
      kernelVariants.push_back(0);
    }
    else {
      for (int knl = 0; knl < Nkernels; ++knl) {

        // v3 requires Nq^3 < 1024 (max threads/thread block on CUDA/HIP)
        if(knl == 3 && Np > 1024) continue;
        // v6 requires Nq % 2 == 0
        if(knl == 6 && Nq % 2 == 1) continue;
        kernelVariants.push_back(knl);
      }
    }

    const std::string installDir(getenv("NEKRS_HOME"));

    // only a single choice, no need to run benchmark
    if(kernelVariants.size() == 1 && !requiresBenchmark){
      auto newProps = props;
      newProps["defines/p_knl"] = kernelVariants.back();

      const std::string ext = platform->serial ? ".c" : ".okl";
      const std::string fileName = installDir + "/okl/elliptic/" + kernelName + ext;

      return std::make_pair(platform->device.buildKernel(fileName, newProps, true), -1.0);
    }

    auto DrV    = randomVector<FPType>(Nq * Nq);
    auto ggeo   = randomVector<FPType>(Np_g * Nelements * p_Nggeo);
    auto q      = randomVector<FPType>((Ndim * Np) * Nelements);
    auto Aq     = randomVector<FPType>((Ndim * Np) * Nelements);
    auto exyz   = randomVector<FPType>((3 * Np_g) * Nelements);
    auto gllwz  = randomVector<FPType>(2 * Nq_g);
    auto lambda = randomVector<FPType>(2 * Np * Nelements);

    // elementList[e] = e
    std::vector<dlong> elementList(Nelements);
    std::iota(elementList.begin(), elementList.end(), 0);
    auto o_elementList = platform->device.malloc(Nelements * sizeof(dlong), elementList.data());

    auto o_D = platform->device.malloc(Nq * Nq * wordSize, DrV.data());
    auto o_S = o_D;
    auto o_ggeo = platform->device.malloc(Np_g * Nelements * p_Nggeo * wordSize, ggeo.data());
    auto o_q = platform->device.malloc((Ndim * Np) * Nelements * wordSize, q.data());
    auto o_Aq = platform->device.malloc((Ndim * Np) * Nelements * wordSize, Aq.data());
    auto o_exyz = platform->device.malloc((3 * Np_g) * Nelements * wordSize, exyz.data());
    auto o_gllwz = platform->device.malloc(2 * Nq_g * wordSize, gllwz.data());

    auto o_lambda = platform->device.malloc(2 * Np * Nelements * wordSize, lambda.data());

    auto axKernelBuilder = [&](int kernelVariant) {
      auto newProps = props;
      newProps["defines/p_knl"] = kernelVariant;

      const std::string ext = platform->serial ? ".c" : ".okl";
      const std::string fileName = installDir + "/okl/elliptic/" + kernelName + ext;

      return platform->device.buildKernel(fileName, newProps, true);
    };

    auto kernelRunner = [&](occa::kernel &kernel) {
      const int loffset = 0;
      const int offset = Nelements * Np;
      if(computeGeom){
        kernel(Nelements, offset, loffset, o_elementList, o_exyz, o_gllwz, o_D, o_S, o_lambda, o_q, o_Aq);
      } else {
        kernel(Nelements, offset, loffset, o_elementList, o_ggeo, o_D, o_S, o_lambda, o_q, o_Aq);
      }
    };

    auto printPerformanceInfo = [&](int kernelVariant, double elapsed, int Ntests, bool skipPrint) {

      const bool BKmode = constCoeff && poisson;

      double NGlobalElements = Nelements;
      MPI_Allreduce(MPI_IN_PLACE, &NGlobalElements, 1, MPI_DOUBLE, MPI_SUM, platform->comm.mpiComm);

      // print statistics
      const dfloat GDOFPerSecond = (NGlobalElements * Ndim * (N * N * N) / elapsed) / 1.e9;

      size_t bytesMoved = Ndim * 2 * Np * wordSize; // x, Ax
      bytesMoved += 6 * Np_g * wordSize; // geo
      if(!constCoeff) bytesMoved += 3 * Np * wordSize; // lambda1, lambda2, Jw
      const double bw = (NGlobalElements * bytesMoved / elapsed) / 1.e9;

      double flopCount = Np * 12 * Nq + 15 * Np;
      if(!constCoeff) flopCount += 5 * Np;
      const double gflops = Ndim * (flopCount * NGlobalElements / elapsed) / 1.e9;
      const int Nthreads =  omp_get_max_threads();

      if(platform->comm.mpiRank == 0 && !skipPrint){
        if(verbosity > 0){
          std::cout << "Ax:";
        }
        if(verbosity > 1){
          std::cout << " MPItasks=" << platform->comm.mpiCommSize
                    << " OMPthreads=" << Nthreads
                    << " NRepetitions=" << Ntests;
        }
        if(verbosity > 0){
          std::cout << " Ndim=" << Ndim
                    << " N=" << N
                    << " Ng=" << Ng
                    << " Nelements=" << NGlobalElements
                    << " elapsed time=" << elapsed
                    << " bkMode=" << BKmode
                    << " wordSize=" << 8*wordSize
                    << " GDOF/s=" << GDOFPerSecond
                    << " GB/s=" << bw
                    << " GFLOPS/s=" << gflops
                    << " kernel=" << kernelVariant
                    << "\n";
        }
      }
    };

    auto printCallBack = [&](int kernelVariant, double elapsed, int Ntests) {
      printPerformanceInfo(kernelVariant, elapsed, Ntests, verbosity < 2);
    };

    auto kernelAndTime =
        benchmarkKernel(axKernelBuilder, kernelRunner, printCallBack, kernelVariants, NtestsOrTargetTime);
    int bestKernelVariant = static_cast<int>(kernelAndTime.first.properties()["defines/p_knl"]);
    
    // print only the fastest kernel
    if(verbosity == 1){
      printPerformanceInfo(bestKernelVariant, kernelAndTime.second, 0, false);
    }


    free(o_D);
    free(o_S);
    free(o_ggeo);
    free(o_q);
    free(o_Aq);
    free(o_exyz);
    free(o_gllwz);
    free(o_lambda);
    free(o_elementList);

    return kernelAndTime;

  };

  occa::kernel kernel;

  if(wordSize == sizeof(float)){
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

template occa::kernel benchmarkAx<int>(int Nelements,
                                       int Nq,
                                       int Ng,
                                       bool constCoeff,
                                       bool poisson,
                                       bool computeGeom,
                                       int wordSize,
                                       int Ndim,
                                       int verbosity,
                                       int Ntests,
                                       bool requiresBenchmark = false);

template occa::kernel benchmarkAx<double>(int Nelements,
                                          int Nq,
                                          int Ng,
                                          bool constCoeff,
                                          bool poisson,
                                          bool computeGeom,
                                          int wordSize,
                                          int Ndim,
                                          int verbosity,
                                          double targetTime,
                                          bool requiresBenchmark = false);