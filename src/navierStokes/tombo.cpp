#include "nrs.hpp"
#include "udf.hpp"
#include "linAlg.hpp"
#include <limits>

namespace tombo
{
occa::memory pressureSolve(nrs_t* nrs, dfloat time, int stage)
{
  const int verbose = platform->options.compareArgs("VERBOSE", "TRUE");
  platform->timer.tic("pressure rhs", 1);
  double flopCount = 0.0;
  mesh_t* mesh = nrs->meshV;

  nrs->curlKernel(mesh->Nelements,
                  1,
                  mesh->o_vgeo,
                  mesh->o_D,
                  nrs->fieldOffset,
                  nrs->o_Ue,
                  platform->o_mempool.slice0);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice0,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("curl norm: %.15e\n", debugNorm);
  }

  flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

  platform->o_mempool.slice3.copyFrom(platform->o_mempool.slice0,
                                      nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  ogsGatherScatterMany(platform->o_mempool.slice3,
                       nrs->NVfields,
                       nrs->fieldOffset,
                       ogsDfloat,
                       ogsAdd,
                       mesh->ogs);
  oogs::startFinish(platform->o_mempool.slice0, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice3,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("ogs curl norm: %.15e\n", debugNorm);
  }

  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice0,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("gs curl norm: %.15e\n", debugNorm);
  }
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        1,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        nrs->meshV->o_invLMM,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("invLMM norm: %.15e\n", debugNorm);
  }
  
  platform->linAlg->axmyVector(
    mesh->Nlocal,
    nrs->fieldOffset,
    0,
    1.0,
    nrs->meshV->o_invLMM,
    platform->o_mempool.slice0
  );

  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice0,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("invLMM curl norm: %.15e\n", debugNorm);
  }

  flopCount += mesh->Nlocal;

  nrs->curlKernel(
    mesh->Nelements,
    1,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    platform->o_mempool.slice0,
    platform->o_mempool.slice3);

  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice3,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("second curl norm: %.15e\n", debugNorm);
  }
  flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Np * mesh->Nq + 36 * mesh->Np);

  nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    nrs->o_div,
    platform->o_mempool.slice0);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice0,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("grad curl norm: %.15e\n", debugNorm);
  }
  flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

  if (platform->options.compareArgs("STRESSFORMULATION", "TRUE")) {
    nrs->pressureStressKernel(
         mesh->Nelements,
         mesh->o_vgeo,
         mesh->o_D,
         nrs->fieldOffset,
         nrs->o_mue,
         nrs->o_Ue,
         nrs->o_div,
         platform->o_mempool.slice3);
    flopCount += static_cast<double>(mesh->Nelements) * (18 * mesh->Nq * mesh->Np + 100 * mesh->Np);
  }

  occa::memory o_irho = nrs->o_ellipticCoeff;
  nrs->pressureRhsKernel(
    mesh->Nelements * mesh->Np,
    nrs->fieldOffset,
    nrs->o_mue,
    o_irho,
    nrs->o_BF,
    platform->o_mempool.slice3,
    platform->o_mempool.slice0,
    platform->o_mempool.slice6);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice6,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("pressure rhs (3 comp) norm: %.15e\n", debugNorm);
  }
  flopCount += 12 * static_cast<double>(mesh->Nlocal);

  oogs::startFinish(platform->o_mempool.slice6, nrs->NVfields, nrs->fieldOffset,ogsDfloat, ogsAdd, nrs->gsh);

  platform->linAlg->axmyVector(
    mesh->Nlocal,
    nrs->fieldOffset,
    0,
    1.0,
    nrs->meshV->o_invLMM,
    platform->o_mempool.slice6
  );
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        nrs->NVfields,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice6,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("inv LMM pressure rhs (3 comp) norm: %.15e\n", debugNorm);
  }

  nrs->wDivergenceVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    platform->o_mempool.slice6,
    platform->o_mempool.slice3);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        1,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice3,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("wdiv norm: %.15e\n", debugNorm);
  }
  flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

  nrs->pressureAddQtlKernel(
    mesh->Nlocal,
    mesh->o_LMM,
    nrs->g0 * nrs->idt,
    nrs->o_div,
    platform->o_mempool.slice3);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        1,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice3,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("add qtl norm: %.15e\n", debugNorm);
  }
  flopCount += 3 * mesh->Nlocal;

  nrs->divergenceSurfaceKernel(
    mesh->Nelements,
    mesh->o_sgeo,
    mesh->o_vmapM,
    nrs->o_EToB,
    nrs->g0 * nrs->idt,
    nrs->fieldOffset,
    platform->o_mempool.slice6,
    nrs->o_U,
    platform->o_mempool.slice3);
  if (verbose) {
    const dfloat debugNorm = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
        1,
        nrs->fieldOffset,
        mesh->ogs->o_invDegree,
        platform->o_mempool.slice3,
        platform->comm.mpiComm);
    if (platform->comm.mpiRank == 0)
      printf("div surf norm: %.15e\n", debugNorm);
  }
  flopCount += 25 * static_cast<double>(mesh->Nelements) * mesh->Nq * mesh->Nq;

  platform->timer.toc("pressure rhs");

  platform->o_mempool.slice1.copyFrom(nrs->o_P, mesh->Nlocal * sizeof(dfloat));
  ellipticSolve(nrs->pSolver, platform->o_mempool.slice3, platform->o_mempool.slice1);

  platform->flopCounter->add("pressure RHS", flopCount);

  return platform->o_mempool.slice1;
}

occa::memory velocitySolve(nrs_t* nrs, dfloat time, int stage)
{
  platform->timer.tic("velocity rhs", 1);
  double flopCount = 0.0;
  mesh_t* mesh = nrs->meshV;
  
  dfloat scale = -1./3;
  if(platform->options.compareArgs("STRESSFORMULATION", "TRUE")) scale = 2./3;

  platform->linAlg->axmyz(
       mesh->Nlocal,
       scale,
       nrs->o_mue,
       nrs->o_div,
       platform->o_mempool.slice3);

  nrs->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    platform->o_mempool.slice3,
    platform->o_mempool.slice0);

  flopCount += static_cast<double>(mesh->Nelements) * (6 * mesh->Np * mesh->Nq + 18 * mesh->Np);

  nrs->wgradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_D,
    nrs->fieldOffset,
    nrs->o_P,
    platform->o_mempool.slice3);

  flopCount += static_cast<double>(mesh->Nelements) * 18 * (mesh->Np * mesh->Nq + mesh->Np);

  platform->linAlg->axpby(
    nrs->NVfields*nrs->fieldOffset,
    1.0,
    platform->o_mempool.slice3,
    -1.0,
    platform->o_mempool.slice0);

  nrs->velocityNeumannBCKernel(mesh->Nelements,
                               nrs->fieldOffset,
                               mesh->o_sgeo,
                               mesh->o_vmapM,
                               mesh->o_EToB,
                               nrs->o_EToB,
                               time,
                               mesh->o_x,
                               mesh->o_y,
                               mesh->o_z,
                               nrs->o_usrwrk,
                               nrs->o_U,
                               platform->o_mempool.slice0);

  flopCount += static_cast<double>(mesh->Nelements) * (3 * mesh->Np + 36 * mesh->Nq * mesh->Nq);

  nrs->velocityRhsKernel(
    mesh->Nlocal,
    nrs->fieldOffset,
    nrs->o_BF,
    platform->o_mempool.slice0,
    nrs->o_rho,
    platform->o_mempool.slice3);

  flopCount += 6 * mesh->Nlocal;

  platform->timer.toc("velocity rhs");
  platform->o_mempool.slice0.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  if(nrs->uvwSolver) {
    ellipticSolve(nrs->uvwSolver, platform->o_mempool.slice3, platform->o_mempool.slice0);
  } else {
    ellipticSolve(nrs->uSolver, platform->o_mempool.slice3, platform->o_mempool.slice0);
    ellipticSolve(nrs->vSolver, platform->o_mempool.slice4, platform->o_mempool.slice1);
    ellipticSolve(nrs->wSolver, platform->o_mempool.slice5, platform->o_mempool.slice2);
  }

  platform->flopCounter->add("velocity RHS", flopCount);

  return platform->o_mempool.slice0;
}

} // namespace
