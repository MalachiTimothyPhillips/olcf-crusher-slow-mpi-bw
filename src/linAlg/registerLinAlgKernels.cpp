#include <nrs.hpp>
#include <compileKernels.hpp>
#include <tuple>

void registerLinAlgKernels()
{
  occa::properties kernelInfo = platform->kernelInfo;

  std::string oklDir;
  oklDir.assign(getenv("NEKRS_INSTALL_DIR"));
  oklDir += "/okl/linAlg/";
  std::string fileName;
  const bool serial = platform->serial;

  const std::vector<std::pair<std::string, bool>> allKernels{
      {"fill", false},
      {"vabs", false},
      {"add", false},
      {"scale", false},
      {"scaleMany", false},
      {"axpby", true},
      {"axpbyMany", true},
      {"axpbyz", false},
      {"axpbyzMany", false},
      {"axmy", true},
      {"axmyMany", true},
      {"axmyVector", true},
      {"axmyz", false},
      {"axmyzMany", false},
      {"ady", false},
      {"adyMany", false},
      {"axdy", false},
      {"aydx", false},
      {"aydxMany", false},
      {"axdyz", false},
      {"sum", false},
      {"sumMany", false},
      {"min", false},
      {"max", false},
      {"norm2", true},
      {"norm2Many", true},
      {"norm1", true},
      {"norm1Many", true},
      {"weightedNorm1", true},
      {"weightedNorm1Many", true},
      {"weightedNorm2", true},
      {"weightedNorm2Many", true},
      {"innerProd", true},
      {"weightedInnerProd", true},
      {"weightedInnerProdMany", true},
      {"weightedInnerProdMulti", false},
      {"crossProduct", false},
      {"unitVector", false},
  };

  std::string kernelName;
  bool nativeSerialImplementation;
  for(auto&& nameAndSerialImpl : allKernels){
    std::tie(kernelName, nativeSerialImplementation) = nameAndSerialImpl;
    const std::string extension = (serial && nativeSerialImplementation) ? ".c" : ".okl";
    platform->kernels.add(
        kernelName, oklDir + kernelName + extension,  kernelInfo);
  }

  // FP32 kernels
  auto pfloatKernelInfo = kernelInfo;
  pfloatKernelInfo["defines/dfloat"] = pfloatString;

  const std::string extension = ".okl";
  for(const auto fp32KernelName : {"vabs", "fill", "max", "axmyzMany", "ady", "axmy", "axpbyzMany"}){
    platform->kernels.add(
        std::string("f") + fp32KernelName, oklDir + fp32KernelName + extension,  pfloatKernelInfo);
  }

}