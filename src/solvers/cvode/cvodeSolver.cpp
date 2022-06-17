#include "cvodeSolver.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "Urst.hpp"
#include <limits>
#include <array>

#include <cvode/cvode.h>
#include "timeStepper.hpp"

// TODO: rename o_FS -> o_RHS

namespace{
void computeInvLMMLMM(mesh_t* mesh, occa::memory& o_invLMMLMM)
{
  o_invLMMLMM.copyFrom(mesh->o_LMM, mesh->Nlocal * sizeof(dfloat));
  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM);
}
}

namespace cvode {

cvodeSolver_t::cvodeSolver_t(nrs_t* nrs, const Parameters_t & params)
{
  auto cds = nrs->cds;

  o_coeffExt = platform->device.malloc(maxExtrapolationOrder * sizeof(dfloat));

  dlong Nwords = nrs->NVfields * nrs->fieldOffset; // velocities
  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    Nwords += 2 * nrs->NVfields * nrs->fieldOffset; // coordinates, mesh velocities
  }

  o_oldState = platform->device.malloc(Nwords * sizeof(dfloat));

  if(nrs->cht){
    o_invLMMLMMT = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat));
    computeInvLMMLMM(cds->mesh[0], o_invLMMLMMT);
  }

  o_invLMMLMMV = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat));
  computeInvLMMLMM(nrs->meshV, o_invLMMLMMV);

  setupEToLMapping(nrs);

  std::vector<dlong> scalarIds;
  std::vector<dlong> cvodeScalarIds(cds->NSfields, -1);

  Nscalar = 0;
  fieldOffsetScan = {0};
  fieldOffsetSum = 0;

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;
    
    cvodeScalarIds[is] = Nscalar;
    scalarIds.push_back(is);

    cvodeScalarToScalarIndex[Nscalar] = is;
    scalarToCvodeScalarIndex[is] = Nscalar;

    fieldOffset.push_back(cds->fieldOffset[is]);
    fieldOffsetScan.push_back(fieldOffsetScan.back() + fieldOffset.back());
    fieldOffsetSum += fieldOffset.back();

    Nscalar++;
  }

  o_scalarIds = platform->device.malloc(scalarIds.size() * sizeof(dlong), scalarIds.data());
  o_cvodeScalarIds = platform->device.malloc(cvodeScalarIds.size() * sizeof(dlong), cvodeScalarIds.data());

}

void cvodeSolver_t::setupEToLMapping(nrs_t *nrs)
{
  static_assert(sizeof(dlong) == sizeof(int), "dlong and int must be the same size");
  auto *mesh = nrs->meshV;
  if (mesh->cht)
    mesh = nrs->cds->mesh[0];

  auto o_Lids = platform->device.malloc(mesh->Nlocal * sizeof(dlong));
  std::vector<dlong> Eids(mesh->Nlocal);
  std::iota(Eids.begin(), Eids.end(), 0);
  o_Lids.copyFrom(Eids.data(), mesh->Nlocal * sizeof(dlong));

  {
    const auto saveNhaloGather = mesh->ogs->NhaloGather;
    mesh->ogs->NhaloGather = 0;
    ogsGatherScatter(o_Lids, "int", "ogsMin", mesh->ogs);
    mesh->ogs->NhaloGather = saveNhaloGather;
  }

  std::vector<dlong> Lids(mesh->Nlocal);
  o_Lids.copyTo(Lids.data(), mesh->Nlocal * sizeof(dlong));

  std::set<dlong> uniqueIds;
  for (auto &&id : Lids) {
    uniqueIds.insert(id);
  }

  const auto NL = uniqueIds.size();

  this->LFieldOffset = NL;

  std::vector<dlong> EToL(mesh->Nlocal);
  std::map<dlong, std::set<dlong>> LToE;
  for (auto &&Eid : Eids) {
    const auto Lid = std::distance(uniqueIds.begin(), uniqueIds.find(Eid));
    EToL[Eid] = Lid;
    LToE[Lid].insert(Eid);
  }

  std::vector<dlong> EToLUnique(mesh->Nlocal);
  std::copy(EToL.begin(), EToL.end(), EToLUnique.begin());

  // use the first Eid value for the deciding unique Lid value
  for (int eid = 0; eid < mesh->Nlocal; ++eid) {
    const auto lid = EToL[eid];
    if (*LToE[lid].begin() != eid) {
      EToLUnique[eid] = -1;
    }
  }

  this->o_EToL = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToL.data());
  this->o_EToLUnique = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToLUnique.data());

  this->mapEToLKernel = platform->kernels.get("mapEToL");
  this->mapLToEKernel = platform->kernels.get("mapLToE");
  this->packKernel = platform->kernels.get("pack");
  this->unpackKernel = platform->kernels.get("unpack");

  o_Lids.free();
}

void cvodeSolver_t::rhs(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  if (time != tprev) {
    tprev = time;
    std::array<dfloat, 3> dtCvode = {0, 0, 0};
    std::array<dfloat, 3> coeffAB = {0, 0, 0};

    const auto cvodeDt = time - t0;
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

  unpack(o_y, cds->o_S);

  evaluateProperties(nrs, time);

  // terms to include: user source, advection, filtering, add "weak" laplacian
  platform->linAlg->fillKernel(cds->fieldOffsetSum, 0.0, cds->o_FS);
  makeqImpl(nrs);

  auto o_FS = o_S;
  // map all scalars to cvode scalars
  for(int cvodeScalarId = 0; cvodeScalarId < Nscalar; ++cvodeScalarId){
    const auto scalarId = cvodeScalarToScalarIndex.at(cvodeScalarId);
    o_FS.copyFrom(nrs->o_FS,
      fieldOffset[cvodeScalarId] * sizeof(dfloat), 
      fieldOffsetScan[cvodeScalarId] * sizeof(dfloat),
      cds->fieldOffsetScan[scalarId] * sizeof(dfloat));
  }

  // make note: try to lump all cvode scalars contiguously
  // find contiguous blocks and do the gs that way

  if(nrs->cht){
    auto * gsh = cds->gsh;
    auto * gshT = cds->gshT;
    oogs::start(o_FS, 1, fieldOffset[0], ogsDfloat, ogsAdd, gshT);

    if(Nscalar > 1){
      auto o_FS_sans_T = o_FS + fieldOffset[0] * sizeof(dfloat);

      // TODO: will not work in the variable scalar field offset case
      oogs::start(o_FS_sans_T, Nscalar-1, fieldOffset[1], ogsDfloat, ogsAdd, gsh);
      
    }

  } else {
    auto * gsh = cds->gsh;
    oogs::start(o_FS, Nscalar, fieldOffset[0], ogsDfloat, ogsAdd, gsh);
  }

  // TODO: overlap...
  // unresolved issue: worth allocating additional memory?
  if(userLocalPointSource){
    userLocalPointSource(nrs, o_S, o_FS);
  }


  if(nrs->cht){
    auto * gsh = cds->gsh;
    auto * gshT = cds->gshT;
    oogs::finish(o_FS, 1, fieldOffset[0], ogsDfloat, ogsAdd, gshT);

    if(Nscalar > 1){
      auto o_FS_minus_T = o_FS + fieldOffset[0] * sizeof(dfloat);

      // TODO: will not work in the variable scalar field offset case
      oogs::finish(o_FS_minus_T, Nscalar-1, fieldOffset[1], ogsDfloat, ogsAdd, gsh);
      
    }

  } else {
    auto * gsh = cds->gsh;
    oogs::finish(o_FS, Nscalar, fieldOffset[0], ogsDfloat, ogsAdd, gsh);
  }

  pack(o_FS, o_ydot);
}

void cvodeSolver_t::makeq(nrs_t* nrs, dfloat time)
{

  auto * cds = nrs->cds;
  auto o_FS = nrs->cds->o_FS;

  if (udf.sEqnSource) {
    platform->timer.tic("udfSEqnSource", 1);
    udf.sEqnSource(nrs, time, cds->o_S, o_FS);
    platform->timer.toc("udfSEqnSource");
  }

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

    // TODO: subtract mesh velocity
    // TODO: apply operator for multiple fields

    // poor man's solution
    // 2 outer loop Nelements, Nfields
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

    // TODO: turn off diagonal -- should be no need to call kernel below
    // just populate cds->o_ellipticCoeff
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
    
    auto o_invLMMLMM = (nrs->cht && is == 0) ? o_invLMMLMMT : o_invLMMLMMV;
    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_invLMMLMM, o_FS_i);

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

void cvodeSolver_t::unpack(occa::memory & o_y, occa::memory & o_S)
{
  if(userUnpackFunction){
    userUnpackFunction(o_y, o_S);
  } else {
    // TODO: move back
    // map cvode scalars to all scalars
    for(int cvodeScalarId = 0; cvodeScalarId < Nscalar; ++cvodeScalarId){
      const auto scalarId = cvodeScalarToScalarIndex.at(cvodeScalarId);
      cds->o_S.copyFrom(o_S,
        fieldOffset[cvodeScalarId] * sizeof(dfloat), 
        cds->fieldOffsetScan[scalarId] * sizeof(dfloat),
        fieldOffsetScan[cvodeScalarId] * sizeof(dfloat));
    }
  }
}

void cvodeSolver_t::solve(nrs_t *nrs, double t0, double t1, int tstep)
{
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  // copy current state into buffer
  auto o_U = o_oldState + 0 * nrs->fieldOffset * sizeof(dfloat);

  o_U.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  occa::memory o_x;
  occa::memory o_y;
  occa::memory o_z;
  occa::memory o_meshU;

  if (movingMesh) {
    o_x = o_oldState + (nrs->NVfields + 0) * nrs->fieldOffset * sizeof(dfloat);
    o_y = o_oldState + (nrs->NVfields + 1) * nrs->fieldOffset * sizeof(dfloat);
    o_z = o_oldState + (nrs->NVfields + 2) * nrs->fieldOffset * sizeof(dfloat);
    o_meshU = o_oldState + (nrs->NVfields + 3) * nrs->fieldOffset * sizeof(dfloat);

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