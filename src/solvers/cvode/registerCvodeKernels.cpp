#include "compileKernels.hpp"
#include "nrs.hpp"

void registerCvodeKernels(occa::properties kernelInfoBC){
#ifdef ENABLE_CVODE
  std::string kernelName, fileName;
  std::string installDir;
  installDir.assign(getenv("NEKRS_INSTALL_DIR"));
  const std::string oklpath = installDir + "/okl/";

  kernelName = "mapLToE";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "mapEToL";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "extrapolateInPlace";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "pack";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "unpack";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);
#endif
}
