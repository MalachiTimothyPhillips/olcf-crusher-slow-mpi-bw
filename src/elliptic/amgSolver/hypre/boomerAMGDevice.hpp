#ifndef BOOMERAMG_DEVICE_H
#define BOOMERAMG_DEVICE_H

#include <mpi.h>
#include "occa.hpp"

#define BOOMERAMG_NPARAM 10

int boomerAMGSetupDevice(int nrows, int nz,
                         const occa::memory& o_Ai, const occa::memory& o_Aj, const occa::memory& o_Av,
                         const int null_space, const MPI_Comm ce,
                         const int useFP32, const double *param, const int verbose);

int boomerAMGSolveDevice(const occa::memory& o_x, const occa::memory& o_b);

void boomerAMGFreeDevice();

#endif
