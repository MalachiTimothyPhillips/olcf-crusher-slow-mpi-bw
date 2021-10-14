#include <nrs.hpp>
#include <compileKernels.hpp>

void registerCdsKernels(occa::properties kernelInfoBC) {
  std::string installDir;
  installDir.assign(getenv("NEKRS_INSTALL_DIR"));
  occa::properties kernelInfo = platform->kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  int N, cubN;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  platform->options.getArgs("CUBATURE POLYNOMIAL DEGREE", cubN);
  const int Nq = N + 1;
  const int cubNq = cubN + 1;
  const int Np = Nq * Nq * Nq;
  const int cubNp = cubNq * cubNq * cubNq;
  constexpr int Nfaces{6};

  constexpr int NVfields{3};
  kernelInfo["defines/p_NVfields"] = NVfields;

  int Nsubsteps = 0;
  int nBDF = 0;
  int nEXT = 0;
  platform->options.getArgs("SUBCYCLING STEPS", Nsubsteps);

  if (platform->options.compareArgs("TIME INTEGRATOR", "TOMBO1")) {
    nBDF = 1;
  } else if (platform->options.compareArgs("TIME INTEGRATOR", "TOMBO2")) {
    nBDF = 2;
  } else if (platform->options.compareArgs("TIME INTEGRATOR", "TOMBO3")) {
    nBDF = 3;
  }
  nEXT = 3;
  if (Nsubsteps)
    nEXT = nBDF;


  std::string fileName, kernelName;
  const std::string suffix = "Hex3D";
  const std::string oklpath = installDir + "/okl/";
  const std::string section = "cds-";
  occa::properties meshProps = kernelInfo;
  meshProps += meshKernelProperties(N);
  {
    {
      occa::properties prop = meshProps;
      prop["defines/p_cubNq"] = cubNq;
      prop["defines/p_cubNp"] = cubNp;
      fileName = oklpath + "cds/advection" + suffix + ".okl";

      kernelName = "strongAdvectionVolume" + suffix;
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);

      kernelName = "strongAdvectionCubatureVolume" + suffix;
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);
    }

    fileName = oklpath + "cds/advectMeshVelocityHex3D.okl";
    kernelName = "advectMeshVelocityHex3D";
    platform->kernels.add(
        section + kernelName, fileName, kernelName, meshProps);

    fileName = oklpath + "core/mask.okl";
    kernelName = "maskCopy";
    platform->kernels.add(
        section + kernelName, fileName, kernelName, meshProps);

    {
      occa::properties prop = kernelInfo;
      const int movingMesh =
          platform->options.compareArgs("MOVING MESH", "TRUE");
      prop["defines/p_MovingMesh"] = movingMesh;
      prop["defines/p_nEXT"] = nEXT;
      prop["defines/p_nBDF"] = nBDF;
      if (Nsubsteps)
        prop["defines/p_SUBCYCLING"] = 1;
      else
        prop["defines/p_SUBCYCLING"] = 0;

      fileName = oklpath + "cds/sumMakef.okl";
      kernelName = "sumMakef";
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);
    }

    fileName = oklpath + "cds/helmholtzBC" + suffix + ".okl";
    kernelName = "helmholtzBC" + suffix;
    platform->kernels.add(
        section + kernelName, fileName, kernelName, kernelInfoBC);
    kernelName = "dirichletBC";
    platform->kernels.add(
        section + kernelName, fileName, kernelName, kernelInfoBC);

    fileName = oklpath + "core/setEllipticCoeff.okl";
    kernelName = "setEllipticCoeff";
    platform->kernels.add(
        section + kernelName, fileName, kernelName, kernelInfo);

    fileName = oklpath + "cds/regularization/filterRT" + suffix + ".okl";
    kernelName = "filterRT" + suffix;
    platform->kernels.add(
        section + kernelName, fileName, kernelName, meshProps);

    fileName = oklpath + "core/nStagesSum.okl";
    kernelName = "nStagesSum3";
    platform->kernels.add(
        section + kernelName, fileName, kernelName, platform->kernelInfo);

    {
      occa::properties prop = meshProps;
      const int movingMesh =
          platform->options.compareArgs("MOVING MESH", "TRUE");
      prop["defines/p_MovingMesh"] = movingMesh;
      prop["defines/p_nEXT"] = nEXT;
      prop["defines/p_nBDF"] = nBDF;
      prop["defines/p_cubNq"] = cubNq;
      prop["defines/p_cubNp"] = cubNp;

      fileName = oklpath + "cds/subCycle" + suffix + ".okl";
      occa::properties subCycleStrongCubatureProps = prop;
      const bool serial = useSerial();

      if(useSerial)
        fileName = oklpath + "cds/subCycle" + suffix + ".c";

      kernelName = "subCycleStrongCubatureVolume" + suffix;
      platform->kernels.add(section + kernelName,
          fileName,
          kernelName,
          subCycleStrongCubatureProps);
      fileName = oklpath + "cds/subCycle" + suffix + ".okl";
      kernelName = "subCycleStrongVolume" + suffix;
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);

      fileName = oklpath + "cds/subCycleRKUpdate.okl";
      kernelName = "subCycleERKUpdate";
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);
      kernelName = "subCycleRK";
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);

      kernelName = "subCycleInitU0";
      platform->kernels.add(
          section + kernelName, fileName, kernelName, prop);
    }
  }
}