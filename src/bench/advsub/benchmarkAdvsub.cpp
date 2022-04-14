#include "benchmarkAdvsub.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include "nrs.hpp"

#include "randomVector.hpp"
#include "kernelBenchmarker.hpp"
#include "omp.h"

template <typename T>
occa::kernel
benchmarkAdvsub(int Nelements, int Nq, int cubNq, int nEXT, bool dealias, int verbosity, T NtestsOrTargetTime, bool requiresBenchmark)
{
  static constexpr int nFields = 3;
  const int N = Nq-1;
  const int cubN = cubNq - 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;
  int fieldOffset = Np * Nelements;
  const int pageW = ALIGN_SIZE / sizeof(dfloat);
  if (fieldOffset % pageW) fieldOffset = (fieldOffset / pageW + 1) * pageW;
  int cubatureOffset = std::max(fieldOffset, Nelements * cubNp);
  if (cubatureOffset % pageW)
    cubatureOffset = (cubatureOffset / pageW + 1) * pageW;

  occa::properties props = platform->kernelInfo + meshKernelProperties(N);
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();
  props["include_paths"].asArray();

  props["defines/p_cubNq"] = cubNq;
  props["defines/p_cubNp"] = cubNp;
  props["defines/p_nEXT"] = nEXT;
  props["defines/p_NVfields"] = nFields;
  props["defines/p_MovingMesh"] = platform->options.compareArgs("MOVING MESH", "TRUE");

  std::string installDir;
  installDir.assign(getenv("NEKRS_INSTALL_DIR"));

  std::string diffDataFile = installDir + "/okl/mesh/constantDifferentiationMatrices.h";
  std::string interpDataFile = installDir + "/okl/mesh/constantInterpolationMatrices.h";
  std::string diffInterpDataFile = installDir + "/okl/mesh/constantDifferentiationInterpolationMatrices.h";

  props["includes"] += diffDataFile.c_str();
  props["includes"] += interpDataFile.c_str();
  props["includes"] += diffInterpDataFile.c_str();

  std::string fileName = 
    installDir + "/okl/bench/advsub/readCubDMatrix.okl";
  auto readCubDMatrixKernel = platform->device.buildKernel(fileName, props, true);

  fileName = 
    installDir + "/okl/bench/advsub/readIMatrix.okl";
  auto readIMatrixKernel = platform->device.buildKernel(fileName, props, true);

  std::string kernelName;
  if(dealias){
    kernelName = "subCycleStrongCubatureVolumeHex3D";
  } else {
    kernelName = "subCycleStrongVolumeHex3D";
  }

  const std::string ext = (platform->device.mode() == "Serial") ? ".c" : ".okl";
  fileName = 
    installDir + "/okl/nrs/" + kernelName + ext;
  
  // currently lacking a native implementation of the non-dealiased kernel
  if(!dealias) fileName = installDir + "/okl/nrs/subCycleHex3D.okl";

  std::vector<int> kernelVariants = {0};
  if(!platform->serial && dealias){
    // TODO: reduce number of kernel variants
    constexpr int Nkernels = 14;
    for(int i = 1; i < Nkernels; ++i){

      // v12 requires cubNq <=13
      if(i == 11 && cubNq > 13) continue;

      kernelVariants.push_back(i);
    }
  }

  if(kernelVariants.size() == 1 && !requiresBenchmark){
    auto newProps = props;
    newProps["defines/p_knl"] = kernelVariants.back();
    return platform->device.buildKernel(fileName, newProps, true);
  }

  occa::kernel referenceKernel;
  {
    auto newProps = props;
    newProps["defines/p_knl"] = kernelVariants.front();
    referenceKernel = platform->device.buildKernel(fileName, newProps, true);
  }

  const int wordSize = sizeof(dfloat);

  auto invLMM   = randomVector<dfloat>(Nelements * Np);
  auto cubD  = randomVector<dfloat>(cubNq * cubNq);
  auto NU  = randomVector<dfloat>(nFields * fieldOffset);
  auto conv  = randomVector<dfloat>(nFields * cubatureOffset * nEXT);
  auto cubInterpT  = randomVector<dfloat>(Nq * cubNq);
  auto Ud  = randomVector<dfloat>(nFields * fieldOffset);
  occa::memory o_BdivW;

  // elementList[e] = e
  std::vector<dlong> elementList(Nelements);
  std::iota(elementList.begin(), elementList.end(), 0);
  auto o_elementList = platform->device.malloc(Nelements * sizeof(dlong), elementList.data());

  auto o_invLMM = platform->device.malloc(Nelements * Np * wordSize, invLMM.data());
  auto o_cubD = platform->device.malloc(cubNq * cubNq * wordSize, cubD.data());
  auto o_NU = platform->device.malloc(nFields * fieldOffset * wordSize, NU.data());
  auto o_conv = platform->device.malloc(nFields * cubatureOffset * nEXT * wordSize, conv.data());
  auto o_cubInterpT = platform->device.malloc(Nq * cubNq * wordSize, cubInterpT.data());
  auto o_Ud = platform->device.malloc(nFields * fieldOffset * wordSize, Ud.data());

  // popular cubD, cubInterpT with correct data
  readCubDMatrixKernel(o_cubD);
  readIMatrixKernel(o_cubInterpT);

  auto kernelRunner = [&](occa::kernel & subcyclingKernel){
    const auto c0 = 0.1;
    const auto c1 = 0.2;
    const auto c2 = 0.3;
    if(!dealias) {
      subcyclingKernel(Nelements, o_elementList, o_cubD, fieldOffset,
        0, o_invLMM, o_BdivW, c0, c1, c2, o_conv, o_Ud, o_NU);
    } else {
      subcyclingKernel(Nelements, o_elementList, o_cubD, o_cubInterpT, fieldOffset,
        cubatureOffset, 0, o_invLMM, o_BdivW, c0, c1, c2, o_conv, o_Ud, o_NU);
    }
  };

  auto advSubKernelBuilder = [&](int kernelVariant){
    auto newProps = props;
    newProps["defines/p_knl"] = kernelVariant;
    auto kernel = platform->device.buildKernel(fileName, newProps, true);

    // perform correctness check
    std::vector<dfloat> referenceResults(3*fieldOffset);
    std::vector<dfloat> results(3*fieldOffset);

    kernelRunner(referenceKernel);
    o_NU.copyTo(referenceResults.data(), referenceResults.size() * sizeof(dfloat));
    
    kernelRunner(kernel);
    o_NU.copyTo(results.data(), results.size() * sizeof(dfloat));

    double err = 0.0;
    for(auto i = 0; i < results.size(); ++i){
      err = std::max(err, std::abs(results[i] - referenceResults[i]));
    }

    if(platform->comm.mpiRank == 0 && verbosity > 1){
      std::cout << "Error in kernel " << kernelVariant << " is " << err << " compared to reference implementation.\n";
    }

    return kernel;
  };


  auto printPerformanceInfo = [&](int kernelVariant, double elapsed, int Ntests, bool skipPrint) {
    const dfloat GDOFPerSecond = nFields * ( Nelements * (N * N * N) / elapsed) / 1.e9;

    size_t bytesPerElem = 2 * nFields * Np; // Ud, NU
    bytesPerElem += Np; // inv mass matrix
    bytesPerElem += nFields * cubNp * nEXT; // U(r,s,t)

    size_t otherBytes = cubNq * cubNq; // D
    if(cubNq > Nq){
      otherBytes += Nq * cubNq; // interpolator
    }
    otherBytes   *= wordSize;
    bytesPerElem *= wordSize;
    const double bw = ( (Nelements * bytesPerElem + otherBytes) / elapsed) / 1.e9;

    double flopCount = 0.0; // per elem basis
    if(cubNq > Nq){
      flopCount += 6. * cubNp * nEXT; // extrapolate U(r,s,t) to current time
      flopCount += 18. * cubNp * cubNq; // apply Dcub
      flopCount += 9. * Np; // compute NU
      flopCount += 12. * Nq * (cubNp + cubNq * cubNq * Nq + cubNq * Nq * Nq); // interpolation
    } else {
      flopCount = Nq * Nq * Nq * (18. * Nq + 6. * nEXT + 24.);
    }
    const double gflops = ( flopCount * Nelements / elapsed) / 1.e9;
    const int Nthreads =  omp_get_max_threads();

    if(platform->comm.mpiRank == 0 && !skipPrint){

      if(verbosity > 0){
        std::cout << "advsub:";
      }

      if(verbosity > 1){
        std::cout << " MPItasks=" << platform->comm.mpiCommSize << " OMPthreads=" << Nthreads << " NRepetitions=" << Ntests;
      }
      if(verbosity > 0){
        std::cout << " N=" << N << " cubN=" << cubN << " nEXT=" << nEXT << " Nelements=" << Nelements
                  << " elapsed time=" << elapsed << " wordSize=" << 8 * wordSize << " GDOF/s=" << GDOFPerSecond
                  << " GB/s=" << bw << " GFLOPS/s=" << gflops << " kernel=" << kernelVariant << "\n";
      }
    }
  };

  auto printCallBack = [&](int kernelVariant, double elapsed, int Ntests) {
    printPerformanceInfo(kernelVariant, elapsed, Ntests, verbosity < 2);
  };

  auto kernelAndTime =
      benchmarkKernel(advSubKernelBuilder, kernelRunner, printCallBack, kernelVariants, NtestsOrTargetTime);
  int bestKernelVariant = static_cast<int>(kernelAndTime.first.properties()["defines/p_knl"]);

  // print only the fastest kernel
  if(verbosity == 1){
    printPerformanceInfo(bestKernelVariant, kernelAndTime.second, 0, false);
  }

  free(o_elementList);
  free(o_invLMM);
  free(o_cubD);
  free(o_NU);
  free(o_conv);
  free(o_cubInterpT);
  free(o_Ud);

  return kernelAndTime.first;
}

template
occa::kernel benchmarkAdvsub<int>(int Nelements,
                             int Nq,
                             int cubNq,
                             int nEXT,
                             bool dealias,
                             int verbosity,
                             int Ntests,
                             bool requiresBenchmark);

template
occa::kernel benchmarkAdvsub<double>(int Nelements,
                             int Nq,
                             int cubNq,
                             int nEXT,
                             bool dealias,
                             int verbosity,
                             double targetTime,
                             bool requiresBenchmark);