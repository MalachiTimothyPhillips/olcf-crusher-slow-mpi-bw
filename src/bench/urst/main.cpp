#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include "mpi.h"

#include "nrssys.hpp"
#include "setupAide.hpp"
#include "platform.hpp"
#include "configReader.hpp"

namespace {

occa::kernel urstKernel;
occa::memory o_D;
occa::memory o_x;
occa::memory o_y;
occa::memory o_z;
occa::memory o_cubInterpT;
occa::memory o_cubw;
occa::memory o_U;
occa::memory o_Urst;
occa::memory o_vgeo;
occa::memory o_cubvgeo;

int Nelements;
int Np;
int cubNp;
dlong fieldOffset;
dlong cubatureOffset;
bool dealias;

double run(int Ntests)
{
  occa::memory o_meshU;

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();

  for (int test = 0; test < Ntests; ++test) {
    if (!dealias) {
      urstKernel(Nelements, o_vgeo, fieldOffset, o_U, o_meshU, o_Urst);
    }
    else {
      urstKernel(Nelements, o_cubvgeo, o_cubInterpT, fieldOffset, cubatureOffset, o_U, o_meshU, o_Urst);
    }
  }

  platform->device.finish();
  MPI_Barrier(MPI_COMM_WORLD);
  return (MPI_Wtime() - start) / Ntests;
}

void *(*randAlloc)(int);

void *rand32Alloc(int N)
{
  float *v = (float *)malloc(N * sizeof(float));

  for (int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}

void *rand64Alloc(int N)
{
  double *v = (double *)malloc(N * sizeof(double));

  for (int n = 0; n < N; ++n)
    v[n] = drand48();

  return v;
}

} // namespace

int main(int argc, char **argv)
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

  std::string threadModel;
  int N;
  int cubN = -1;
  int okl = 1;
  int Ntests = -1;
  dealias = true;
  static constexpr size_t wordSize = 8;

  while (1) {
    static struct option long_options[] = {{"p-order", required_argument, 0, 'p'},
                                           {"c-order", required_argument, 0, 'c'},
                                           {"no-cubature", no_argument, 0, 'd'},
                                           {"elements", required_argument, 0, 'e'},
                                           {"backend", required_argument, 0, 'b'},
                                           {"arch", required_argument, 0, 'a'},
                                           {"help", required_argument, 0, 'h'},
                                           {"iterations", required_argument, 0, 'i'},
                                           {0, 0, 0, 0}};
    int option_index = 0;
    int c = getopt_long(argc, argv, "", long_options, &option_index);

    if (c == -1)
      break;

    switch (c) {
    case 'p':
      N = atoi(optarg);
      cmdCheck++;
      break;
    case 'c':
      cubN = atoi(optarg);
      break;
    case 'd':
      dealias = false;
      break;
    case 'e':
      Nelements = atoi(optarg);
      cmdCheck++;
      break;
    case 'b':
      options.setArgs("THREAD MODEL", std::string(optarg));
      cmdCheck++;
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

  if (err || cmdCheck != 3) {
    if (rank == 0)
      printf("Usage: ./nekrs-bench-urst  --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>\n"
             "                    [--c-order <n>] [--no-cubature] [--iterations <n>]\n");
    exit(1);
  }

  if (cubN < 0) {
    if (dealias)
      cubN = round((3. / 2) * (N + 1) - 1) - 1;
    else
      cubN = N;
  }
  if (cubN < N) {
    if (rank == 0)
      printf("Error: cubature order (%d) must be larger than or equal to the quadrature order (%d)!\n",
             cubN,
             N);
    exit(1);
  }
  Nelements = std::max(1, Nelements / size);
  const int Nq = N + 1;
  Np = Nq * Nq * Nq;
  const int cubNq = cubN + 1;
  cubNp = cubNq * cubNq * cubNq;
  fieldOffset = Np * Nelements;
  const int pageW = ALIGN_SIZE / sizeof(dfloat);
  if (fieldOffset % pageW)
    fieldOffset = (fieldOffset / pageW + 1) * pageW;
  cubatureOffset = std::max(fieldOffset, Nelements * cubNp);

  platform = platform_t::getInstance(options, MPI_COMM_WORLD, MPI_COMM_WORLD);
  const int Nthreads = omp_get_max_threads();

  // build+load kernel
  occa::properties props = platform->kernelInfo + meshKernelProperties(N);
  static constexpr int nFields = 3;
  static constexpr int Nvgeo = 12;
  props["defines/p_cubNq"] = cubNq;
  props["defines/p_cubNp"] = cubNp;
  props["defines/p_relative"] = 0;

  std::string kernelName;
  if (dealias) {
    kernelName = "UrstCubatureHex3D";
  }
  else {
    kernelName = "UrstHex3D";
  }

  const std::string ext = (platform->device.mode() == "Serial") ? ".c" : ".okl";
  std::string fileName = installDir + "/okl/nrs/" + kernelName + ext;

  // currently lacking a native implementation of the non-dealiased kernel
  if (!dealias)
    fileName = installDir + "/okl/nrs/subCycleHex3D.okl";

  urstKernel = platform->device.buildKernel(fileName, props, true);

  // populate arrays

  randAlloc = &rand64Alloc;

  void *D = randAlloc(Nq * Nq);
  void *x = randAlloc(Nelements * Np);
  void *y = randAlloc(Nelements * Np);
  void *z = randAlloc(Nelements * Np);
  void *cubInterpT = randAlloc(Nq * cubNq);
  void *cubw = randAlloc(cubNq);
  void *U = randAlloc(nFields * fieldOffset);
  void *Urst = randAlloc(nFields * cubatureOffset);
  void *vgeo = randAlloc(Nvgeo * Nelements * Np);
  void *cubvgeo = randAlloc(Nvgeo * Nelements * cubNp);

  o_D = platform->device.malloc(Nq * Nq * wordSize, D);
  free(D);
  o_x = platform->device.malloc(Nelements * Np * wordSize, x);
  free(x);
  o_y = platform->device.malloc(Nelements * Np * wordSize, y);
  free(y);
  o_z = platform->device.malloc(Nelements * Np * wordSize, z);
  free(z);
  o_cubInterpT = platform->device.malloc(Nq * cubNq * wordSize, cubInterpT);
  free(cubInterpT);
  o_cubw = platform->device.malloc(cubNq * wordSize, cubw);
  free(cubw);
  o_U = platform->device.malloc(nFields * fieldOffset * wordSize, U);
  free(U);
  o_Urst = platform->device.malloc(nFields * cubatureOffset * wordSize, Urst);
  free(Urst);
  o_vgeo = platform->device.malloc(Nvgeo * Nelements * Np * wordSize, vgeo);
  free(vgeo);
  o_cubvgeo = platform->device.malloc(Nvgeo * Nelements * cubNp * wordSize, cubvgeo);
  free(cubvgeo);

  // warm-up
  double elapsed = run(10);
  const int elapsedTarget = 10;
  if (Ntests < 0)
    Ntests = elapsedTarget / elapsed;

  // *****
  elapsed = run(Ntests);
  // *****

  if (rank == 0)
    std::cout << "MPItasks=" << size << " OMPthreads=" << Nthreads << " NRepetitions=" << Ntests << " N=" << N
              << " cubN=" << cubN << " Nelements=" << size * Nelements << " elapsed time=" << elapsed << "\n";

  MPI_Finalize();
  exit(0);
}
