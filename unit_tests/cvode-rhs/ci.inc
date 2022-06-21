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

  const auto LFieldOffset = cvodeSolver.LFieldOffset;

  auto o_y = platform->device.malloc((nrs->Nscalar * sizeof(dfloat)) * LFieldOffset);
  auto o_ydot = platform->device.malloc((nrs->Nscalar * sizeof(dfloat)) * LFieldOffset);

  const dfloat dt = 1e-2;
  cvodeSolver.rhs(nrs, tstep, time, time + dt, o_y, o_ydot);

  // compute norms
  const auto Ynorm = platform->linAlg->norm2Many(LFieldOffset, nrs->Nscalar, LFieldOffset, o_y, platform->comm.mpiComm);
  const auto YdotNorm = platform->linAlg->norm2Many(LFieldOffset, nrs->Nscalar, LFieldOffset, o_ydot, platform->comm.mpiComm);

  if(platform->comm.mpiRank == 0){
    std::cout << "Ynorm = " << Ynorm << "\n";
    std::cout << "YdotNorm = " << YdotNorm << std::endl;
  }
  std::cout << std::flush;

  o_y.free();
  o_ydot.free();

}
