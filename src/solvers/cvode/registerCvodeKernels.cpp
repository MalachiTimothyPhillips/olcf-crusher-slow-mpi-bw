#include "compileKernels.hpp"
void registerCvodeKernels(occa::properties kernelInfoBC){

  kernelName = "mapLToE";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "mapEToL";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "mapEToL";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);

  kernelName = "extrapolateInPlace";
  fileName = oklpath + "cvode/" + kernelName + ".okl";
  platform->kernels.add(kernelName, fileName, platform->kernelInfo);
}
