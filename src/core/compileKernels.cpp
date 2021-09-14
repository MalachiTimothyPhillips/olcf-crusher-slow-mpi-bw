#include <compileKernels.hpp>
#include "elliptic.h"
#include "mesh.h"
#include "ogs.hpp"
#include "ogsKernels.hpp"
#include "udf.hpp"
#include <vector>
#include <tuple>

std::string createOptionsPrefix(std::string section) {
  std::string prefix = section + std::string(" ");
  if (section.find("temperature") != std::string::npos) {
    prefix = std::string("scalar00 ");
  }
  std::transform(
      prefix.begin(), prefix.end(), prefix.begin(), [](unsigned char c) {
        return std::toupper(c);
      });
  return prefix;
}
void constructCoarseningAndProlongationKernels(occa::properties kernelInfo)
{
  const std::string suffix = "Hex3D";
  const bool serial = platform->device.mode() == "Serial" || platform->device.mode() == "OpenMP";
  std::string install_dir;
  install_dir.assign(getenv("NEKRS_INSTALL_DIR"));
  const std::string oklpath = install_dir + "/okl/elliptic/";
  std::string filename, kernelName;
  for (int pass = 0; pass < 2; ++pass) {
    auto levels = determineMGLevels("pressure", pass);

    for (int levelIndex = 1; levelIndex < levels.size(); ++levelIndex) {

      const int Nf = levels[levelIndex - 1];
      const int Nc = levels[levelIndex];

      // sizes for the coarsen and prolongation kernels. degree NFine to degree N
      int NqFine = (Nf + 1);
      int NqCoarse = (Nc + 1);
      occa::properties coarsenProlongateKernelInfo = kernelInfo;
      coarsenProlongateKernelInfo["defines/p_NqFine"] = Nf + 1;
      coarsenProlongateKernelInfo["defines/p_NqCoarse"] = Nc + 1;

      const int NpFine = (Nf + 1) * (Nf + 1) * (Nf + 1);
      const int NpCoarse = (Nc + 1) * (Nc + 1) * (Nc + 1);
      coarsenProlongateKernelInfo["defines/p_NpFine"] = NpFine;
      coarsenProlongateKernelInfo["defines/p_NpCoarse"] = NpCoarse;

      const std::string orderSuffix =
          std::string("_") + std::to_string(Nf) + std::string("_") + std::to_string(Nc);

      if (serial) {
        filename = oklpath + "ellipticPreconCoarsen" + suffix + ".c";
        kernelName = "ellipticPreconCoarsen" + suffix;
        platform->kernels.add(kernelName + orderSuffix, filename, coarsenProlongateKernelInfo, orderSuffix);
        filename = oklpath + "ellipticPreconProlongate" + suffix + ".c";
        kernelName = "ellipticPreconProlongate" + suffix;
        platform->kernels.add(kernelName + orderSuffix, filename, coarsenProlongateKernelInfo, orderSuffix);
      }
      else {
        filename = oklpath + "ellipticPreconCoarsen" + suffix + ".okl";
        kernelName = "ellipticPreconCoarsen" + suffix;
        platform->kernels.add(kernelName + orderSuffix, filename, coarsenProlongateKernelInfo, orderSuffix);
        filename = oklpath + "ellipticPreconProlongate" + suffix + ".okl";
        kernelName = "ellipticPreconProlongate" + suffix;
        platform->kernels.add(kernelName + orderSuffix, filename, coarsenProlongateKernelInfo, orderSuffix);
      }
    }
  }
}

void compileKernels() {

  MPI_Barrier(platform->comm.mpiComm);
  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0)
    printf("loading kernels (this may take awhile) ...\n");
  fflush(stdout);

  const occa::properties kernelInfoBC = compileUDFKernels();

  registerLinAlgKernels();

  registerMeshKernels(kernelInfoBC);

  registerNrsKernels(kernelInfoBC);

  int Nscalars;
  platform->options.getArgs("NUMBER OF SCALARS", Nscalars);
  const int scalarWidth = getDigitsRepresentation(NSCALAR_MAX - 1);

  if (Nscalars) {
    registerCdsKernels(kernelInfoBC);
    for(int is = 0; is < Nscalars; is++){
      std::stringstream ss;
      ss << std::setfill('0') << std::setw(scalarWidth) << is;
      std::string sid = ss.str();
      const std::string section = "scalar" + sid;
      const int poisson = 0;

      if(!platform->options.compareArgs("SCALAR" + sid + " SOLVER", "NONE")){
        registerEllipticKernels(section, poisson);
        registerEllipticPreconditionerKernels(section, poisson);
      }
    }
  }

  // Scalar section is omitted
  // as pressure section kernels are the same.
  const std::vector<std::pair<std::string,int>> sections = {
      {"pressure", 1},
      {"velocity", 0}
  };

  std::string section;
  int poissonEquation;
  for (auto&& entry : sections) {
    std::tie(section, poissonEquation) = entry;
    registerEllipticKernels(section, poissonEquation);
    registerEllipticPreconditionerKernels(section, poissonEquation);
  }


  {
    const bool buildNodeLocal = useNodeLocalCache();
    const bool buildOnly = platform->options.compareArgs("BUILD ONLY", "TRUE");
    auto communicator = buildNodeLocal ? platform->comm.mpiCommLocal : platform->comm.mpiComm;
    oogs::compile(
        platform->device.occaDevice(), platform->device.mode(), communicator, buildOnly);
  }

  platform->kernels.compile();

  // load platform related kernels
  std::string kernelName;
  kernelName = "copyDfloatToPfloat";
  platform->copyDfloatToPfloatKernel = platform->kernels.get(kernelName);

  kernelName = "copyPfloatToDfloat";
  platform->copyPfloatToDfloatKernel = platform->kernels.get(kernelName);

  MPI_Barrier(platform->comm.mpiComm);
  const double loadTime = MPI_Wtime() - tStart;


  fflush(stdout);
  if (platform->comm.mpiRank == 0) {
    std::ofstream ofs;
    ofs.open(occa::env::OCCA_CACHE_DIR + "cache/compile.timestamp", 
	     std::ofstream::out | std::ofstream::trunc);
    ofs.close();
  }
 
  platform->timer.set("loadKernels", loadTime);
  if (platform->comm.mpiRank == 0)
    printf("done (%gs)\n\n", loadTime);
  fflush(stdout);
}
