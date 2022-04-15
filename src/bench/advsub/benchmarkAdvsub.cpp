#include "benchmarkAdvsub.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include "nrs.hpp"

#include "randomVector.hpp"
#include "kernelBenchmarker.hpp"
#include "omp.h"

#include "mesh.h"
#include "mesh3D.h"

namespace {
std::string dumpMatrix(const dfloat *mat, const int N, const int M, std::string matName)
{
  std::ostringstream buffer;
  buffer << "const dfloat " << matName << "[" << N << "][" << M << "] = {\n";
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      buffer << "  " << mat[i * M + j];
      // omit last
      if(i != (N-1) && j != (M-1)) buffer << ",";
    }
    buffer << "\n";
  }
  buffer << "};\n";

  return buffer.str();
}
void writeFactors(const mesh_t &mesh)
{
  std::string cache_dir;
  
  if(!getenv("NEKRS_CACHE_DIR")){
    // NEKRS_CACHE_DIR may not be set in the context of the benchmarker, so
    // let's set it here to the OCCA_CACHE_DIR
    if(getenv("OCCA_CACHE_DIR")){
      // OCCA_CACHE_DIR is set, use that
      setenv("NEKRS_CACHE_DIR", getenv("OCCA_CACHE_DIR"), 0);
    } else {
      // what else can we do at this point?
      setenv("NEKRS_CACHE_DIR", getenv("PWD"), 0);
    }
  }

  cache_dir.assign(getenv("NEKRS_CACHE_DIR"));
  std::cout << "cache_dir: " << cache_dir << std::endl;

  std::string udf_dir = cache_dir + "/udf";
  const std::string dataFile = udf_dir + "/cubatureData.okl";

  int buildRank = platform->comm.mpiRank;
  MPI_Comm comm = platform->comm.mpiComm;
  const bool buildNodeLocal = useNodeLocalCache();
  if (buildNodeLocal) {
    MPI_Comm_rank(platform->comm.mpiCommLocal, &buildRank);
    comm = platform->comm.mpiCommLocal;
  }

  if (buildRank == 0) {

    std::ofstream out;
    out.open(dataFile, std::ios::trunc);

    std::ostringstream buffer;
    buffer << "// This file automatically generated in benchmarkAdvsub.cpp\n";
    buffer << "\n\n\n";
    buffer << dumpMatrix(mesh.cubD, mesh.cubNq, mesh.cubNq, "c_D");

    out << buffer.str();
    out.close();
  }

  MPI_Barrier(comm);
}
} // namespace

template <typename T>
occa::kernel
benchmarkAdvsub(int Nelements, int Nq, int cubNq, int verbosity, T NtestsOrTargetTime, bool requiresBenchmark)
{
  mesh_t mesh;
  mesh.Nelements = Nelements;

  for(int currRank = 0; currRank < platform->comm.mpiCommSize; ++currRank){
    if(platform->comm.mpiRank == currRank) {
      printf("rank %d: pid<%d>\n", platform->comm.mpiRank, ::getpid());
    }
  }

  if (platform->comm.mpiRank == 0)
    std::cout << "Attach debugger, then press enter to continue\n";
  if (platform->comm.mpiRank == 0)
    std::cin.get();
  MPI_Barrier(platform->comm.mpiComm);

  // construct cubature derivative and interpolation operators
  meshLoadReferenceNodesHex3D(&mesh, Nq - 1, cubNq - 1);

  // dump cubD, cubInterpT to file for kernels
  writeFactors(mesh);

  meshFree(&mesh);

  return occa::kernel();
}

template occa::kernel
benchmarkAdvsub<int>(int Nelements, int Nq, int cubNq, int verbosity, int Ntests, bool requiresBenchmark);
template occa::kernel benchmarkAdvsub<double>(int Nelements,
                                              int Nq,
                                              int cubNq,
                                              int verbosity,
                                              double targetTime,
                                              bool requiresBenchmark);