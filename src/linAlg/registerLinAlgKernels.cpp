#include <nrs.hpp>
#include <compileKernels.hpp>

void registerLinAlgKernels()
{
  occa::properties kernelInfo = platform->kernelInfo;

  std::string oklDir;
  oklDir.assign(getenv("NEKRS_INSTALL_DIR"));
  oklDir += "/okl/linAlg/";
  std::string fileName;
  const bool serial = (platform->device.mode() == "Serial" ||
                       platform->device.mode() == "OpenMP");

  platform->kernels.add_kernel(
      "fill", oklDir + "linAlgFill.okl", "fill", kernelInfo);
  platform->kernels.add_kernel(
      "vabs", oklDir + "linAlgAbs.okl", "vabs", kernelInfo);
  platform->kernels.add_kernel(
      "add", oklDir + "linAlgAdd.okl", "add", kernelInfo);
  platform->kernels.add_kernel(
      "scale", oklDir + "linAlgScale.okl", "scale", kernelInfo);
  platform->kernels.add_kernel(
      "scaleMany", oklDir + "linAlgScale.okl", "scaleMany", kernelInfo);
  fileName = std::string("linAlgAXPBY") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel("axpby", oklDir + fileName, "axpby", kernelInfo);
  fileName = std::string("linAlgAXPBY") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "axpbyMany", oklDir + fileName, "axpbyMany", kernelInfo);
  platform->kernels.add_kernel(
      "axpbyz", oklDir + "linAlgAXPBY.okl", "axpbyz", kernelInfo);
  platform->kernels.add_kernel(
      "axpbyzMany", oklDir + "linAlgAXPBY.okl", "axpbyzMany", kernelInfo);
  fileName = std::string("linAlgAXMY") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel("axmy", oklDir + fileName, "axmy", kernelInfo);
  fileName = std::string("linAlgAXMY") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "axmyMany", oklDir + fileName, "axmyMany", kernelInfo);
  fileName = std::string("linAlgAXMY") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "axmyVector", oklDir + fileName, "axmyVector", kernelInfo);
  platform->kernels.add_kernel(
      "axmyz", oklDir + "linAlgAXMY.okl", "axmyz", kernelInfo);
  platform->kernels.add_kernel(
      "axmyzMany", oklDir + "linAlgAXMY.okl", "axmyzMany", kernelInfo);
  platform->kernels.add_kernel(
      "ady", oklDir + "linAlgAXDY.okl", "ady", kernelInfo);
  platform->kernels.add_kernel(
      "adyMany", oklDir + "linAlgAXDY.okl", "adyMany", kernelInfo);
  platform->kernels.add_kernel(
      "axdy", oklDir + "linAlgAXDY.okl", "axdy", kernelInfo);
  platform->kernels.add_kernel(
      "aydx", oklDir + "linAlgAXDY.okl", "aydx", kernelInfo);
  platform->kernels.add_kernel(
      "aydxMany", oklDir + "linAlgAXDY.okl", "aydxMany", kernelInfo);
  platform->kernels.add_kernel(
      "axdyz", oklDir + "linAlgAXDY.okl", "axdyz", kernelInfo);
  platform->kernels.add_kernel(
      "sum", oklDir + "linAlgSum.okl", "sum", kernelInfo);
  platform->kernels.add_kernel(
      "sumMany", oklDir + "linAlgSum.okl", "sumMany", kernelInfo);
  platform->kernels.add_kernel(
      "min", oklDir + "linAlgMin.okl", "min", kernelInfo);
  platform->kernels.add_kernel(
      "max", oklDir + "linAlgMax.okl", "max", kernelInfo);
  fileName = std::string("linAlgNorm2") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "norm2", oklDir + fileName, "norm2", kernelInfo);
  platform->kernels.add_kernel(
      "norm2Many", oklDir + fileName, "norm2Many", kernelInfo);
  fileName = std::string("linAlgNorm1") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "norm1", oklDir + fileName, "norm1", kernelInfo);
  platform->kernels.add_kernel(
      "norm1Many", oklDir + fileName, "norm1Many", kernelInfo);
  fileName = std::string("linAlgWeightedNorm1") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "weightedNorm1", oklDir + fileName, "weightedNorm1", kernelInfo);
  platform->kernels.add_kernel(
      "weightedNorm1Many", oklDir + fileName, "weightedNorm1Many", kernelInfo);
  fileName = std::string("linAlgWeightedNorm2") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "weightedNorm2", oklDir + fileName, "weightedNorm2", kernelInfo);
  fileName = std::string("linAlgWeightedNorm2") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "weightedNorm2Many", oklDir + fileName, "weightedNorm2Many", kernelInfo);
  platform->kernels.add_kernel(
      "innerProd", oklDir + "linAlgInnerProd.okl", "innerProd", kernelInfo);
  fileName = std::string("linAlgWeightedInnerProd") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel(
      "weightedInnerProd", oklDir + fileName, "weightedInnerProd", kernelInfo);
  fileName = std::string("linAlgWeightedInnerProd") +
             (serial ? std::string(".c") : std::string(".okl"));
  platform->kernels.add_kernel("weightedInnerProdMany",
      oklDir + fileName,
      "weightedInnerProdMany",
      kernelInfo);
  platform->kernels.add_kernel("weightedInnerProdMulti",
      oklDir + "linAlgWeightedInnerProd.okl",
      "weightedInnerProdMulti",
      kernelInfo);
}