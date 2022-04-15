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

  std::string kernelName;
  if(dealias){
    kernelName = "subCycleStrongCubatureVolumeHex3D";
  } else {
    kernelName = "subCycleStrongVolumeHex3D";
  }

  const std::string ext = (platform->device.mode() == "Serial") ? ".c" : ".okl";
  std::string fileName = 
    installDir + "/okl/nrs/" + kernelName + ext;
  
  // currently lacking a native implementation of the non-dealiased kernel
  if(!dealias) fileName = installDir + "/okl/nrs/subCycleHex3D.okl";

  auto subcyclingKernel = platform->device.buildKernel(fileName, props, true);

  return occa::kernel();
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