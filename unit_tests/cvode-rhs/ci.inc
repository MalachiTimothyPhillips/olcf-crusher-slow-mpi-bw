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

  // mark all scalar fields as CVODE
  for(int is = 0; is < nrs->Nscalar; ++is){
    cds->compute[is] = true;
    cds->cvodeSolve[is] = true;
  }

  auto cvodeSolver = cvodeSolver_t(nrs, {});

  cvodeSolver.o_U0.copyFrom(nrs->o_U, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
  
  if(platform->options.compareArgs("MOVING MESH", "TRUE")){
    auto mesh = nrs->meshV;
    if(nrs->cht)
      mesh = nrs->cds->mesh[0];

    cvodeSolver.o_meshU0.copyFrom(mesh->o_U0, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

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
  std::cout << "time = " << time << ", dt = " << dt << "\n";

  //cvodeSolver.tprev = ;

  cvodeSolver.rhs(nrs, tstep, time + dt, time, o_y, o_ydot);
  cvodeSolver.unpack(nrs, o_ydot, nrs->cds->o_FS);

  // compute norms
  const auto Ynorm = platform->linAlg->sumMany(LFieldOffset, nrs->Nscalar, LFieldOffset, o_y, platform->comm.mpiComm);
  const auto YdotNorm = platform->linAlg->sumMany(LFieldOffset, nrs->Nscalar, LFieldOffset, o_ydot, platform->comm.mpiComm);
  const auto FSSum = platform->linAlg->sumMany(nrs->meshV->Nlocal, nrs->Nscalar, nrs->fieldOffset, nrs->cds->o_FS, platform->comm.mpiComm);


  if(platform->comm.mpiRank == 0){
    std::cout << "dp0thdt = " << nrs->dp0thdt << "\n";
    std::cout << "Y sum = " << Ynorm << "\n";
    std::cout << "Ydot sum = " << YdotNorm << std::endl;
    std::cout << "FS sum = " << FSSum << std::endl;
  }
  std::cout << std::flush;

  o_y.free();
  o_ydot.free();

}
