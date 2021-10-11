#include <compileKernels.hpp>
#include "udf.hpp"

occa::properties compileUDFKernels()
{
  int buildNodeLocal = 0;
  if (getenv("NEKRS_CACHE_LOCAL"))
    buildNodeLocal = std::stoi(getenv("NEKRS_CACHE_LOCAL"));

  std::string install_dir;
  install_dir.assign(getenv("NEKRS_INSTALL_DIR"));
  int N;
  platform->options.getArgs("POLYNOMIAL DEGREE", N);
  occa::properties kernelInfo = platform->kernelInfo + meshKernelProperties(N);
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  auto rank = buildNodeLocal ? platform->comm.localRank : platform->comm.mpiRank;
  auto communicator = buildNodeLocal ? platform->comm.mpiCommLocal : platform->comm.mpiComm;

  MPI_Barrier(platform->comm.mpiComm);
  const double tStart = MPI_Wtime();
  if (platform->comm.mpiRank == 0)
    printf("loading udf kernels ... ");
  fflush(stdout);

  occa::properties kernelInfoBC;

  for(int pass = 0; pass < 2; ++pass)
  {
    bool executePass = (pass == 0) && (rank == 0);
    executePass |= (pass == 1) && (rank != 0);
    if(executePass){
      kernelInfoBC = kernelInfo;
      if (udf.loadKernels) {
        // side-effect: kernelInfoBC will include any relevant user-defined kernel props
        udf.loadKernels(kernelInfoBC);
      }
      const std::string bcDataFile = install_dir + "/include/core/bcData.h";
      kernelInfoBC["includes"] += bcDataFile.c_str();
      std::string boundaryHeaderFileName;
      platform->options.getArgs("DATA FILE", boundaryHeaderFileName);
      kernelInfoBC["includes"] += realpath(boundaryHeaderFileName.c_str(), NULL);

      kernelInfoBC += meshKernelProperties(N);
    }
    MPI_Barrier(communicator);
  }

  MPI_Barrier(platform->comm.mpiComm);
  const double loadTime = MPI_Wtime() - tStart;
  if (platform->comm.mpiRank == 0)
    printf("done (%gs)\n\n", loadTime);
  fflush(stdout);

  return kernelInfoBC;
}