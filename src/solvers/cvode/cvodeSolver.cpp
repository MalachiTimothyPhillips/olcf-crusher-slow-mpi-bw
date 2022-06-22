#include "cvodeSolver.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "Urst.hpp"
#include <limits>
#include <array>
#include <numeric>
#include "udf.hpp"

//#include <cvode/cvode.h>
#include "timeStepper.hpp"
#include "plugins/lowMach.hpp"

// TODO: rename o_FS -> o_RHS

namespace{
void computeInvLMMLMM(mesh_t* mesh, occa::memory& o_invLMMLMM)
{
  o_invLMMLMM.copyFrom(mesh->o_LMM, mesh->Nlocal * sizeof(dfloat));
  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_invLMMLMM);
}
}

namespace cvode {

cvodeSolver_t::cvodeSolver_t(nrs_t* nrs, const Parameters_t & params)
{
  auto cds = nrs->cds;

  o_coeffExt = platform->device.malloc(maxExtrapolationOrder * sizeof(dfloat));

  o_U0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    o_meshU0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
    o_xyz0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
  }

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
    if (!cds->compute[is]){
      continue;
    }
    if (!cds->cvodeSolve[is]){
      continue;
    }
    
    cvodeScalarIds[is] = Nscalar;
    scalarIds.push_back(is);

    fieldOffset.push_back(cds->fieldOffset[is]);
    fieldOffsetScan.push_back(fieldOffsetScan.back() + fieldOffset.back());
    fieldOffsetSum += fieldOffset.back();

    Nscalar++;

    // TODO: batch gather scatter operations as possible
    gatherScatterOperations.push_back(std::make_tuple(is, is+1, is == 0 ? cds->gshT : cds->gsh));
  }

  o_scalarIds = platform->device.malloc(scalarIds.size() * sizeof(dlong), scalarIds.data());
  o_cvodeScalarIds = platform->device.malloc(cvodeScalarIds.size() * sizeof(dlong), cvodeScalarIds.data());

  this->extrapolateInPlaceKernel = platform->kernels.get("extrapolateInPlace");
  this->mapEToLKernel = platform->kernels.get("mapEToL");
  this->mapLToEKernel = platform->kernels.get("mapLToE");
  this->packKernel = platform->kernels.get("pack");
  this->unpackKernel = platform->kernels.get("unpack");

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

  o_Lids.free();
}

void cvodeSolver_t::rhs(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  auto * cds = nrs->cds;

  if (time != tprev) {
    tprev = time;

    const auto cvodeDt = time - t0;
    dtCvode[0] = cvodeDt;
    dtCvode[1] = nrs->dt[1];
    dtCvode[2] = nrs->dt[2];

    if(platform->comm.mpiRank == 0){
      std::cout << "cv_dtlag = ";
      for(int i = 0; i < 3; ++i){
        std::cout << dtCvode[i] << ", ";
      }
      std::cout << std::endl;
    }

    const int bdfOrder = std::min(tstep, nrs->nBDF);
    const int extOrder = std::min(tstep, nrs->nEXT);
    nek::extCoeff(coeffEXT.data(), dtCvode.data(), extOrder, bdfOrder);
    nek::bdfCoeff(&this->g0, coeffBDF.data(), dtCvode.data(), bdfOrder);
    for (int i = nrs->nEXT; i > extOrder; i--){
      coeffEXT[i - 1] = 0.0;
    }
    for (int i = nrs->nBDF; i > bdfOrder; i--){
      coeffBDF[i - 1] = 0.0;
    }
    
    if(platform->comm.mpiRank == 0){
      std::cout << "coeffEXT = ";
      for(auto && v : coeffEXT){
        std::cout << v << ", ";
      }
      std::cout << std::endl;
    }


    o_coeffExt.copyFrom(coeffEXT.data(), maxExtrapolationOrder * sizeof(dfloat));

    extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, nrs->o_U);

    if (movingMesh) {
      mesh->coeffs(dtCvode.data(), tstep);
      mesh->move();

      extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, mesh->o_U);
    }

    if(platform->comm.mpiRank == 0){
      std::cout << "mesh coeffAB = ";
      for(int i = 0; i < 3; ++i){
        std::cout << mesh->coeffAB[i] << ", ";
      }
      std::cout << std::endl;
    }



    computeUrst(nrs);
  }

  unpack(nrs, o_y, cds->o_S);

  evaluateProperties(nrs, time);

  // terms to include: user source, advection, filtering, add "weak" laplacian
  platform->linAlg->fillKernel(cds->fieldOffsetSum, 0.0, cds->o_FS);
  makeq(nrs, time);

  {
    const auto sumTerm = platform->linAlg->sumMany(
      mesh->Nlocal,
      nrs->Nscalar,
      nrs->fieldOffset,
      cds->o_FS,
      platform->comm.mpiComm
    );
    if(platform->comm.mpiRank == 0){
      std::cout << "sum FS post makeq = " << sumTerm << std::endl;
    }
  }

  // TODO: how to overlap without requiring an allocation?
  if(userLocalPointSource){
    userLocalPointSource(nrs, cds->o_S, cds->o_FS);
  }

  auto applyOgsOperation = [&](auto ogsFunc)
  {
    dlong startId, endId;
    oogs_t * gsh;
    for(const auto p : gatherScatterOperations){
      std::tie(startId, endId, gsh) = p;
      const auto Nfields = endId - startId;
      auto o_fld = cds->o_FS + nrs->cds->fieldOffsetScan[startId] * sizeof(dfloat);
      ogsFunc(o_fld, Nfields, nrs->cds->fieldOffset[startId], ogsDfloat, ogsAdd, gsh);
    }
  };

  applyOgsOperation(oogs::start);
  applyOgsOperation(oogs::finish);

  // apply lowMach correct, if applicable
  if(platform->options.compareArgs("LOWMACH", "TRUE")){
    lowMach::cvodeArguments_t args{this->coeffBDF, this->g0, this->dtCvode[0]};
    // TODO: check condition
    platform->linAlg->fill(mesh->Nlocal, 0.0, nrs->o_div);

    lowMach::qThermalIdealGasSingleComponent(time, nrs->o_div, &args);
    const auto gamma0 = lowMach::gamma();
    platform->linAlg->add(mesh->Nlocal, nrs->dp0thdt * (gamma0 - 1.0) / gamma0, cds->o_FS);
    if(platform->comm.mpiRank == 0){
      std::cout << "nrs->dp0thdt = " << nrs->dp0thdt << "\n";
    }
  }

  pack(nrs, cds->o_FS, o_ydot);
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

    {
      const auto sumTerm = platform->linAlg->sumMany(
        mesh->Nlocal,
        nrs->Nscalar,
        nrs->fieldOffset,
        cds->o_FS,
        platform->comm.mpiComm
      );
      if(platform->comm.mpiRank == 0){
        std::cout << "sum FS post sEqnSource and filter term = " << sumTerm << std::endl;
      }
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

    {
      const auto sumTerm = platform->linAlg->sumMany(
        mesh->Nlocal,
        nrs->Nscalar,
        nrs->fieldOffset,
        cds->o_FS,
        platform->comm.mpiComm
      );
      if(platform->comm.mpiRank == 0){
        std::cout << "sum FS post advection = " << sumTerm << std::endl;
      }
    }

    // weak laplcian + boundary terms
    occa::memory o_Si = cds->o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

    platform->linAlg->fill(mesh->Nlocal, 0.0, platform->o_mempool.slice1);

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
    
    cds->o_ellipticCoeff.copyFrom(cds->o_diff, mesh->Nlocal * sizeof(dfloat),
      cds->fieldOffsetScan[is] * sizeof(dfloat), 0);
    
    // make purely Laplacian
    auto o_helmholtzPart = cds->o_ellipticCoeff + nrs->fieldOffset * sizeof(dfloat);
    platform->linAlg->fill(mesh->Nlocal, 0.0, o_helmholtzPart);

    ellipticOperator(cds->solver[is], o_Si, platform->o_mempool.slice2, dfloatString);

    platform->linAlg->axpby(mesh->Nlocal, 1.0, platform->o_mempool.slice1,
      -1.0, platform->o_mempool.slice2);

    platform->linAlg->axpby(mesh->Nlocal,
        1.0,
        platform->o_mempool.slice2,
        1.0,
        o_FS,
        0,
        isOffset);

    {
      const auto sumTerm = platform->linAlg->sumMany(
        mesh->Nlocal,
        nrs->Nscalar,
        nrs->fieldOffset,
        cds->o_FS,
        platform->comm.mpiComm
      );
      if(platform->comm.mpiRank == 0){
        std::cout << "sum FS after wlaplacian = " << sumTerm << std::endl;
      }
    }

    auto o_FS_i = o_FS + isOffset * sizeof(dfloat);
    auto o_rho_i = cds->o_rho + isOffset * sizeof(dfloat);
    
    auto o_invLMMLMM = (nrs->cht && is == 0) ? o_invLMMLMMT : o_invLMMLMMV;
    platform->linAlg->axmy(mesh->Nlocal, 1.0, o_invLMMLMM, o_FS_i);

  }
}

void cvodeSolver_t::pack(nrs_t * nrs, occa::memory o_field, occa::memory o_y)
{
  if(userPack){
    userPack(nrs, o_field, o_y);
  } else {
    packKernel(nrs->cds->mesh[0]->Nlocal,
               nrs->meshV->Nlocal,
               nrs->Nscalar,
               nrs->fieldOffset,
               this->LFieldOffset,
               this->o_cvodeScalarIds,
               this->o_EToLUnique,
               o_field,
               o_y);
  }
}

void cvodeSolver_t::unpack(nrs_t * nrs, occa::memory o_y, occa::memory o_field)
{
  if(userUnpack){
    userUnpack(nrs, o_y, o_field);
  } else {
    unpackKernel(nrs->cds->mesh[0]->Nlocal,
               nrs->meshV->Nlocal,
               this->Nscalar,
               nrs->fieldOffset,
               this->LFieldOffset,
               this->o_cvodeScalarIds,
               this->o_EToL,
               o_y,
               o_field);
  }
}

void cvodeSolver_t::solve(nrs_t *nrs, double t0, double t1, int tstep)
{
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  o_U0.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  if (movingMesh) {

    o_meshU0.copyFrom(mesh->o_U, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

    o_xyz0.copyFrom(mesh->o_x, mesh->Nlocal * sizeof(dfloat), (0 * sizeof(dfloat)) * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_y, mesh->Nlocal * sizeof(dfloat), (1 * sizeof(dfloat)) * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_z, mesh->Nlocal * sizeof(dfloat), (2 * sizeof(dfloat)) * nrs->fieldOffset, 0);

  }

  occa::memory o_cvodeY;

  pack(nrs, nrs->cds->o_S, o_cvodeY);

  // call cvode solver

  unpack(nrs, o_cvodeY, nrs->cds->o_S);

  // restore previous state
  nrs->o_U.copyFrom(o_U0, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (movingMesh) {
    mesh->o_x.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (0 * sizeof(dfloat)) * nrs->fieldOffset);
    mesh->o_y.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (1 * sizeof(dfloat)) * nrs->fieldOffset);
    mesh->o_z.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (2 * sizeof(dfloat)) * nrs->fieldOffset);

    mesh->o_U.copyFrom(o_meshU0, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

    mesh->update();

    computeUrst(nrs);
  }
}
} // namespace cvode