#include <compileKernels.hpp>
#include "elliptic.h"

namespace{

void registerGMRESKernels(const std::string &section, int Nfields) {
  std::string installDir;
  installDir.assign(getenv("NEKRS_INSTALL_DIR"));
  const std::string oklpath = installDir + "/okl/elliptic/";
  std::string fileName;
  const bool serial = useSerial();

  const std::string fileNameExtension = (serial) ? ".c" : ".okl";
  const std::string sectionIdentifier = std::to_string(Nfields) + "-";

  occa::properties gmresKernelInfo = platform->kernelInfo;
  gmresKernelInfo["defines/p_Nfields"] = Nfields;

  std::string kernelName = "gramSchmidtOrthogonalization";
  fileName = oklpath + "ellipticGramSchmidtOrthogonalization" + fileNameExtension;
  platform->kernels.add(
      sectionIdentifier + kernelName, fileName, kernelName, gmresKernelInfo);

  kernelName = "updatePGMRESSolution";
  fileName = oklpath + "ellipticUpdatePGMRES" + fileNameExtension;
  platform->kernels.add(
      sectionIdentifier + kernelName, fileName, kernelName, gmresKernelInfo);

  kernelName = "fusedResidualAndNorm";
  fileName = oklpath + "ellipticFusedResidualAndNorm" + fileNameExtension;
  platform->kernels.add(
      sectionIdentifier + kernelName, fileName, kernelName, gmresKernelInfo);
}

}

void registerEllipticKernels(std::string section) {
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  const std::string optionsPrefix = createOptionsPrefix(section);

  std::string installDir;
  installDir.assign(getenv("NEKRS_INSTALL_DIR"));
  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();
  kernelInfo += meshKernelProperties(N);

  const bool blockSolver = [&section]() {
    if (section.find("velocity") == std::string::npos)
      return false;
    if (platform->options.compareArgs("STRESSFORMULATION", "TRUE"))
      return true;
    if (platform->options.compareArgs("VELOCITY BLOCK SOLVER", "TRUE"))
      return true;
    return false;
  }();
  const int Nfields = (blockSolver) ? 3 : 1;
  const bool stressForm = [&section]() {
    if (section.find("velocity") == std::string::npos)
      return false;
    if (platform->options.compareArgs("STRESSFORMULATION", "TRUE"))
      return true;
    return false;
  }();

  const bool serial = useSerial();

  const std::string fileNameExtension = (serial) ? ".c" : ".okl";

  const std::string sectionIdentifier = std::to_string(Nfields) + "-";

  if (platform->options.compareArgs(
          optionsPrefix + "KRYLOV SOLVER", "PGMRES")) {
    registerGMRESKernels(section, Nfields);
  }

  // solution projection kernels
  {
    const std::string oklpath = installDir + "/okl/elliptic/";
    std::string fileName, kernelName;

    {
      occa::properties properties = platform->kernelInfo;
      properties["defines/p_Nfields"] = Nfields;

      fileName = oklpath + "ellipticResidualProjection.okl";
      kernelName = "multiScaledAddwOffset";
      platform->kernels.add(
          sectionIdentifier + kernelName, fileName, kernelName, properties);
      kernelName = "accumulate";
      platform->kernels.add(
          sectionIdentifier + kernelName, fileName, kernelName, properties);
    }
  }

  {
    const std::string oklpath = installDir + "/okl/core/";
    std::string fileName;

    fileName = oklpath + "mask.okl";
    platform->kernels.add("mask", fileName, "mask", kernelInfo);
  }

  kernelInfo["defines/p_Nfields"] = Nfields;

  occa::properties dfloatKernelInfo = kernelInfo;
  occa::properties floatKernelInfo = kernelInfo;
  floatKernelInfo["defines/pfloat"] = pfloatString;
  floatKernelInfo["defines/dfloat"] = pfloatString;

  constexpr int elementType{HEXAHEDRA};

  const std::string suffix = "Hex3D";

  occa::properties AxKernelInfo = dfloatKernelInfo;
  const std::string oklpath = installDir + "/okl/elliptic/";
  std::string fileName;
  std::string kernelName;


  if(section == "pressure"){
    AxKernelInfo["defines/p_poisson"] = 1;
  }

  for(auto&& varCoeff : {true, false}){
    std::string kernelNamePrefix = "elliptic";
    if (blockSolver)
      kernelNamePrefix += (stressForm) ? "Stress" : "Block";

    kernelName = "Ax";
    if (varCoeff) kernelName += "Var";
    if (platform->options.compareArgs("ELEMENT MAP", "TRILINEAR")) kernelName += "Trilinear";
    kernelName += suffix; 
    if (blockSolver && !stressForm) kernelName += "_N" + std::to_string(Nfields);

    const std::string _kernelName = kernelNamePrefix + "Partial" + kernelName;
    const std::string prefix = (section == "pressure") ? "pressure-" : "";
    fileName = oklpath + _kernelName + fileNameExtension; 
    platform->kernels.add(
      prefix + _kernelName, fileName, _kernelName, AxKernelInfo);
  }


  fileName = oklpath + "ellipticBuildDiagonal" + suffix + ".okl";
  kernelName = "ellipticBlockBuildDiagonal" + suffix;
  dfloatKernelInfo["defines/dfloat"] = dfloatString;
  dfloatKernelInfo["defines/pfloat"] = pfloatString;
  platform->kernels.add(
      sectionIdentifier + kernelName, fileName, kernelName, dfloatKernelInfo);
  dfloatKernelInfo["defines/pfloat"] = dfloatString;

  // PCG update
  if (serial) {
    fileName = oklpath + "ellipticSerialUpdatePCG.c";
  } else {
    fileName = oklpath + "ellipticUpdatePCG.okl";
  }
  platform->kernels.add(sectionIdentifier + "ellipticBlockUpdatePCG",
      fileName,
      "ellipticBlockUpdatePCG",
      dfloatKernelInfo);
}