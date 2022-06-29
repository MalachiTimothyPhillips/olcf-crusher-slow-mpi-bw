#include <math.h>
#include <iostream>

#include "cvodeSolver.hpp"

static int ciMode = 0;

#define PASS { if (platform->comm.mpiRank == 0) printf("TESTS passed \n"); MPI_Finalize(); exit(0); }
#define FAIL { if (platform->comm.mpiRank == 0) printf("TESTS failed!\n"); MPI_Finalize(); exit(2); }

#define EPS 1e-1

using namespace cvode;

void checkRHS(nrs_t * nrs, dfloat time, int tstep)
{
  auto * cds = nrs->cds;
  auto mesh = nrs->meshV;
  if(nrs->cht)
    mesh = nrs->cds->mesh[0];

  // mark all scalar fields as CVODE
  for(int is = 0; is < nrs->Nscalar; ++is){
    cds->compute[is] = true;
    cds->cvodeSolve[is] = true;
  }

  auto cvodeSolver = cvodeSolver_t(nrs, {});

  cvodeSolver.o_U0.copyFrom(nrs->o_U, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
  
  if(platform->options.compareArgs("MOVING MESH", "TRUE")){

    cvodeSolver.o_meshU0.copyFrom(mesh->o_U, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

    cvodeSolver.o_xyz0.copyFrom(mesh->o_x,
      mesh->Nlocal * sizeof(dfloat),
      (0 * sizeof(dfloat)) * nrs->fieldOffset,
      0);

    cvodeSolver.o_xyz0.copyFrom(mesh->o_y,
      mesh->Nlocal * sizeof(dfloat),
      (1 * sizeof(dfloat)) * nrs->fieldOffset,
      0);

    cvodeSolver.o_xyz0.copyFrom(mesh->o_z,
      mesh->Nlocal * sizeof(dfloat),
      (2 * sizeof(dfloat)) * nrs->fieldOffset,
      0);
  }

  const auto LFieldOffset = cvodeSolver.LFieldOffset;

  auto o_y = platform->device.malloc((nrs->Nscalar * sizeof(dfloat)) * LFieldOffset);
  auto o_ydot = platform->device.malloc((nrs->Nscalar * sizeof(dfloat)) * LFieldOffset);

  cvodeSolver.pack(nrs, nrs->cds->o_S, o_y);

  const dfloat dt = 1e-2;

  cvodeSolver.rhs(nrs, tstep, time + dt, time, o_y, o_ydot);
  cvodeSolver.unpack(nrs, o_ydot, nrs->cds->o_FS);

  auto * expectedValues = static_cast<double*> (nek::scPtr(1));

  std::vector<double> realValues(mesh->Nlocal);
  nrs->cds->o_FS.copyTo(realValues.data(), mesh->Nlocal * sizeof(dfloat));

  double LinfError = 0.0;
  for(int i = 0; i < mesh->Nlocal; ++i){
    auto denom = std::max(std::abs(expectedValues[i]), std::abs(realValues[i]));
    denom = std::abs(denom) > 1e-16 ? denom : 1.0;
    LinfError = std::max(LinfError, std::abs(expectedValues[i] - realValues[i]) / denom);
  }

  MPI_Allreduce(MPI_IN_PLACE, &LinfError, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm);

  // free memory prior to potential FAIL
  o_y.free();
  o_ydot.free();
  
  double tol = 1e-8;
  if(LinfError > tol){
    if(platform->comm.mpiRank == 0){
      std::cout << "Linf error in ydot: " << LinfError << "\n";
    }
    FAIL;
  }

}
