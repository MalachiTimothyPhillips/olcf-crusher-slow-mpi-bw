#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"
#include <vector>
#include <algorithm>

#include "nrssys.hpp"
#include "setupAide.hpp"
#include "platform.hpp"
#include "configReader.hpp"

#include "mesh.h"
#include "benchFDM.hpp"

int main(int argc, char** argv)
{
  int rank = 0, size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  configRead(MPI_COMM_WORLD);
  std::string installDir(getenv("NEKRS_HOME"));
  setupAide options; 

  int err = 0;
  int cmdCheck = 0;

  int wordSize = 8;
  int Nelements;

  int N;
  int okl = 1;
  int Ntests = -1;

  while(1) {
    static struct option long_options[] =
    {
      {"p-order", required_argument, 0, 'p'},
      {"elements", required_argument, 0, 'e'},
      {"backend", required_argument, 0, 'b'},
      {"arch", required_argument, 0, 'a'},
      {"fp32", no_argument, 0, 'f'},
      {"help", required_argument, 0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
    };
    int option_index = 0;
    int c = getopt_long (argc, argv, "", long_options, &option_index);

    if (c == -1)
      break;

    switch(c) {
    case 'p':
      N = atoi(optarg); 
      cmdCheck++; 
      break;
    case 'e':
      Nelements = atoi(optarg);
      cmdCheck++;
      break;
    case 'b':
      options.setArgs("THREAD MODEL", std::string(optarg));
      cmdCheck++;
      break;
    case 'f':
      wordSize = 4;;
      break;
    case 'i':
      Ntests = atoi(optarg);
      break;
    case 'h':
      err = 1;
      break;
    default:
      err = 1;
    }
  }

  if(err || cmdCheck != 3) {
    if(rank == 0)
      printf("Usage: ./nekrs-fdm  --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>\n"
             "                    [--fp32] [--iterations <n>]\n"); 
    exit(1); 
  }

  if(N <= 2){
    if(rank == 0){
      printf("Error: N > 2!\n");
    }
    exit(1);
  }

  Nelements = std::max(1, Nelements/size);
  const int Nq = N + 1;
  const int Np = Nq * Nq * Nq;

  platform = platform_t::getInstance(options, MPI_COMM_WORLD, MPI_COMM_WORLD); 

  // build+load kernel
  occa::properties props = platform->kernelInfo + meshKernelProperties(N-2); // regular, non-extended mesh
  if(wordSize == 4) props["defines/pfloat"] = "float";
  else props["defines/pfloat"] = "dfloat";

  props["defines/p_Nq_e"] = Nq;
  props["defines/p_Np_e"] = Np;
  props["defines/p_overlap"] = 0;

  // always benchmark ASM
  props["defines/p_restrict"] = 0;

  // populate dummy mesh struct to pass Nelements/Nq
  mesh_t dummyMesh;
  dummyMesh.Nelements = Nelements;
  dummyMesh.Nq = Nq;
  dummyMesh.Np = Np;

  pickFDMKernel(props, dummyMesh, true, Ntests, 10.0);

  MPI_Finalize();
  exit(0);
}
