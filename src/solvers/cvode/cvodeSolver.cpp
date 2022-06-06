#include "cvodeSolver.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "Urst.hpp"
#include <limits>
#include <array>

#include <cvode/cvode.h>
#include "timeStepper.hpp"

namespace cvode {

void cvodeSolver_t::reallocBuffer(dlong Nbytes)
{
  if (o_wrk.size() < Nbytes) {
    if (o_wrk.size() > 0)
      o_wrk.free();
    o_wrk = platform->device.malloc(Nbytes);
  }

  if (o_coeffExt.size() == 0) {
    o_coeffExt = platform->device.malloc(maxExtrapolationOrder * sizeof(dfloat));
  }
}

void cvodeSolver_t::rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  if (time != tprev) {
    tprev = time;
    std::array<dfloat, 3> dtCvode = {0, 0, 0};
    std::array<dfloat, 3> coeffAB = {0, 0, 0};

    // TODO: need to check this???
    const auto cvodeDt = time - tf;
    dtCvode[0] = cvodeDt;
    dtCvode[1] = nrs->dt[1];
    dtCvode[2] = nrs->dt[2];

    const int extOrder = std::min(tstep, maxExtrapolationOrder);
    nek::coeffAB(coeffAB.data(), dtCvode.data(), extOrder);
    for (int i = maxExtrapolationOrder; i > extOrder; i--)
      coeffAB[i - 1] = 0.0;

    o_coeffExt.copyFrom(coeffAB.data(), maxExtrapolationOrder * sizeof(dfloat));

    extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, nrs->o_U);

    if (movingMesh) {
      mesh->coeffs(dtCvode.data(), tstep);
      mesh->move();

      extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, mesh->o_U);
    }

    computeUrst(nrs);
  }

  unpack();

  evaluateProperties(nrs, time);

  // terms to include: user source, advection, filtering, add "weak" laplacian
  platform->linAlg->fillKernel(cds->fieldOffsetSum, 0.0, cds->o_FS);
  makeqImpl(nrs);

  // TODO: will need to consolidate all cvode scalar fields into one large vector

  // dssum
  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;
    
    auto gsh = (is == 0) ? cds->gshT : cds->gsh;
    
    const dlong isOffset = cds->fieldOffsetScan[is];
    auto o_FS_i = cds->o_FS + isOffset * sizeof(dfloat);
    
    oogs::start(o_FS_i, 1, cds->fieldOffset[is], ogsDfloat, ogsAdd, gsh);
  }

  if(userLocalPointSource){
    userLocalPointSource();
  }

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;
    
    auto gsh = (is == 0) ? cds->gshT : cds->gsh;
    
    const dlong isOffset = cds->fieldOffsetScan[is];
    auto o_FS_i = cds->o_FS + isOffset * sizeof(dfloat);
    
    oogs::finish(o_FS_i, 1, cds->fieldOffset[is], ogsDfloat, ogsAdd, gsh);
  }

  pack();
}

void cvodeSolver_t::makeqImpl(nrs_t* nrs)
{

  auto * cds = nrs->cds;
  auto o_FS = nrs->cds->o_FS;

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;

    mesh_t *mesh;
    (is) ? mesh = cds->meshV : mesh = cds->mesh[0];
    const dlong isOffset = cds->fieldOffsetScan[is];


    if(cds->options[is].compareArgs("REGULARIZATION METHOD", "RELAXATION")){
      cds->filterRTKernel(
        cds->meshV->Nelements,
        is,
        cds->o_filterMT,
        cds->filterS[is],
        isOffset,
        cds->o_rho,
        cds->o_S,
        o_FS);

      double flops = 6 * mesh->Np * mesh->Nq + 4 * mesh->Np;
      flops *= static_cast<double>(mesh->Nelements);
      platform->flopCounter->add("scalarFilterRT", flops);
    }
    if (cds->options[is].compareArgs("ADVECTION", "TRUE")) {
      if (cds->options[is].compareArgs("ADVECTION TYPE", "CUBATURE")){
        cds->strongAdvectionCubatureVolumeKernel(cds->meshV->Nelements,
                                                 mesh->o_vgeo,
                                                 mesh->o_cubDiffInterpT,
                                                 mesh->o_cubInterpT,
                                                 mesh->o_cubProjectT,
                                                 cds->vFieldOffset,
                                                 isOffset,
                                                 cds->vCubatureOffset,
                                                 cds->o_S,
                                                 cds->o_Urst,
                                                 cds->o_rho,
                                                 platform->o_mempool.slice0);
      }
      else{
        cds->strongAdvectionVolumeKernel(cds->meshV->Nelements,
                                         mesh->o_vgeo,
                                         mesh->o_D,
                                         cds->vFieldOffset,
                                         isOffset,
                                         cds->o_S,
                                         cds->o_Urst,
                                         cds->o_rho,
                                         platform->o_mempool.slice0);
      }
      platform->linAlg->axpby(cds->meshV->Nelements * cds->meshV->Np,
          -1.0,
          platform->o_mempool.slice0,
          1.0,
          o_FS,
          0,
          isOffset);

      timeStepper::advectionFlops(cds->mesh[is], 1);
    }

    // weak laplcian + boundary terms
    occa::memory o_Si = cds->o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

    platform->linAlg->fill(mesh->Nlocal, platform->o_mempool.slice1, 0.0);

    // TODO: confirm sign on this kernel
    cds->helmholtzRhsBCKernel(mesh->Nelements,
                              mesh->o_sgeo,
                              mesh->o_vmapM,
                              mesh->o_EToB,
                              is,
                              time,
                              cds->fieldOffset[is],
                              mesh->o_x,
                              mesh->o_y,
                              mesh->o_z,
                              o_Si,
                              cds->o_EToB[is],
                              *(cds->o_usrwrk),
                              platform->o_mempool.slice1);

    cds->setEllipticCoeffKernel(mesh->Nlocal,
        cds->g0 * cds->idt,
        cds->fieldOffsetScan[is],
        cds->fieldOffset[is],
        cds->o_diff,
        cds->o_rho,
        cds->o_ellipticCoeff);

    if (cds->o_BFDiag.ptr())
      platform->linAlg->axpby(mesh->Nlocal,
          1.0,
          cds->o_BFDiag,
          1.0,
          cds->o_ellipticCoeff,
          cds->fieldOffsetScan[is],
          cds->fieldOffset[is]);
    
    ellipticOperator(cds->solver[is], o_Si, platform->o_mempool.slice2);

    platform->linAlg->axpby(mesh->Nlocal, 1.0, platform->o_mempool.slice1,
      -1.0, platform->o_mempool.slice2);

    platform->linAlg->axpby(mesh->Nlocal,
        1.0,
        platform->o_mempool.slice2,
        1.0,
        o_FS,
        0,
        isOffset);

    auto o_FS_i = o_FS + isOffset * sizeof(dfloat);
    auto o_rho_i = cds->o_rho + isOffset * sizeof(dfloat);
    
    // include inv LMM * LMM weighting
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_FS_i);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_FS_i);

  }
}

void cvodeSolver_t::pack()
{
  if(userPackFunction){
    userPackFunction();
  } else {
    // default
  }
}

void cvodeSolver_t::unpack()
{
  if(userUnpackFunction){
    userUnpackFunction();
  } else {
    //default
  }
}

void cvodeSolver_t::setup(nrs_t *nrs, Parameters_t params)
{
  this->initialized = true;

  dlong Nwords = nrs->NVfields * nrs->fieldOffset; // velocities
  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    Nwords += 2 * nrs->NVfields * nrs->fieldOffset; // coordinates, mesh velocities
  }

  reallocBuffer(Nwords * sizeof(dfloat));

  setupEToLMapping(nrs);
}
void cvodeSolver_t::solve(nrs_t *nrs, double t0, double t1, int tstep)
{
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  // copy current state into buffer
  auto o_U = o_wrk + 0 * nrs->fieldOffset * sizeof(dfloat);

  o_U.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  occa::memory o_x;
  occa::memory o_y;
  occa::memory o_z;
  occa::memory o_meshU;

  if (movingMesh) {
    o_x = o_wrk + (nrs->NVfields + 0) * nrs->fieldOffset * sizeof(dfloat);
    o_y = o_wrk + (nrs->NVfields + 1) * nrs->fieldOffset * sizeof(dfloat);
    o_z = o_wrk + (nrs->NVfields + 2) * nrs->fieldOffset * sizeof(dfloat);
    o_meshU = o_wrk + (nrs->NVfields + 3) * nrs->fieldOffset * sizeof(dfloat);

    o_x.copyFrom(mesh->o_x, mesh->Nlocal * sizeof(dfloat));
    o_y.copyFrom(mesh->o_y, mesh->Nlocal * sizeof(dfloat));
    o_z.copyFrom(mesh->o_z, mesh->Nlocal * sizeof(dfloat));

    o_meshU.copyFrom(mesh->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));
  }

  pack();

  // call cvode solver

  unpack();

  // restore previous state
  nrs->o_U.copyFrom(o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  if (movingMesh) {
    mesh->o_x.copyFrom(o_x, mesh->Nlocal * sizeof(dfloat));
    mesh->o_y.copyFrom(o_y, mesh->Nlocal * sizeof(dfloat));
    mesh->o_z.copyFrom(o_z, mesh->Nlocal * sizeof(dfloat));

    mesh->o_U.copyFrom(o_meshU, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

    mesh->update();

    computeUrst(nrs);
  }
}
} // namespace cvode