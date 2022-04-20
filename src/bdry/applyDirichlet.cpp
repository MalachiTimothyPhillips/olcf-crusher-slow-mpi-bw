#include "nrs.hpp"
#include "bcMap.hpp"

void createZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory& o_EToBV, occa::memory &o_mask)
{
  auto mesh = nrs->meshV;

  platform->linAlg->fill(3 * nrs->fieldOffset, 0.0, o_mask);

  nrs->initializeZeroNormalMaskKernel(mesh->Nlocal, nrs->fieldOffset, o_EToBV, o_mask);

  // normal + count (4 fields)
  auto o_avgNormal = platform->o_mempool.slice0;

  nrs->averageNormalBcTypeKernel(mesh->Nelements,
                                 nrs->fieldOffset,
                                 ZERO_NORMAL,
                                 mesh->o_sgeo,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_avgNormal);

  oogs::startFinish(o_avgNormal, 4, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  nrs->fixZeroNormalMaskKernel(mesh->Nelements,
                     nrs->fieldOffset,
                     mesh->o_sgeo,
                     mesh->o_vmapM,
                     o_EToB,
                     o_avgNormal,
                     o_mask);

  oogs::startFinish(o_mask, 3, nrs->fieldOffset, ogsDfloat, ogsMin, nrs->gsh);
}


void applyZeroNormalMask(nrs_t *nrs,
                         dlong Nelements,
                         occa::memory &o_elementList,
                         occa::memory &o_EToB,
                         occa::memory &o_mask,
                         occa::memory &o_x)
{
  if (Nelements == 0)
    return;

  auto *mesh = nrs->meshV;

  nrs->applyZeroNormalMaskKernel(Nelements,
                                 nrs->fieldOffset,
                                 o_elementList,
                                 mesh->o_sgeo,
                                 o_mask,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_x);
}

void applyZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory &o_mask, occa::memory &o_x)
{
  auto *mesh = nrs->meshV;
  nrs->applyZeroNormalMaskKernel(mesh->Nelements,
                                 nrs->fieldOffset,
                                 mesh->o_elementList,
                                 mesh->o_sgeo,
                                 o_mask,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_x);
}

void applyDirichletMeshVelocity(nrs_t *nrs, double time)
{
  mesh_t *mesh = nrs->meshV;

  occa::memory& o_U = nrs->o_U;
  occa::memory& o_Ue = nrs->o_Ue;

  platform->linAlg->fill(nrs->NVfields * nrs->fieldOffset,
                         -1.0 * std::numeric_limits<dfloat>::max(),
                         platform->o_mempool.slice3);

  for (int sweep = 0; sweep < 2; sweep++) {
    nrs->meshV->velocityDirichletKernel(mesh->Nelements,
                                        nrs->fieldOffset,
                                        time,
                                        bcMap::useDerivedMeshBoundaryConditions(),
                                        mesh->o_sgeo,
                                        nrs->o_zeroNormalMaskMeshVelocity,
                                        mesh->o_x,
                                        mesh->o_y,
                                        mesh->o_z,
                                        mesh->o_vmapM,
                                        mesh->o_EToB,
                                        nrs->o_EToBMeshVelocity,
                                        nrs->o_usrwrk,
                                        nrs->o_U,
                                        platform->o_mempool.slice3);

    if (sweep == 0)
      oogs::startFinish(platform->o_mempool.slice3,
                        nrs->NVfields,
                        nrs->fieldOffset,
                        ogsDfloat,
                        ogsMax,
                        nrs->gsh);
    if (sweep == 1)
      oogs::startFinish(platform->o_mempool.slice3,
                        nrs->NVfields,
                        nrs->fieldOffset,
                        ogsDfloat,
                        ogsMin,
                        nrs->gsh);
  }

  if (nrs->meshSolver->Nmasked) {
    nrs->maskCopyKernel(nrs->meshSolver->Nmasked,
                        0 * nrs->fieldOffset,
                        nrs->meshSolver->o_maskIds,
                        platform->o_mempool.slice3,
                        o_U);
    nrs->maskCopyKernel(nrs->meshSolver->Nmasked,
                        0 * nrs->fieldOffset,
                        nrs->meshSolver->o_maskIds,
                        platform->o_mempool.slice3,
                        o_Ue);
  }

  if (bcMap::unalignedBoundary(mesh->cht, "mesh")) {
    applyZeroNormalMask(nrs, nrs->meshSolver->o_EToB, nrs->o_zeroNormalMaskMeshVelocity, o_U);
    applyZeroNormalMask(nrs, nrs->meshSolver->o_EToB, nrs->o_zeroNormalMaskMeshVelocity, o_Ue);
  }
}

void applyDirichletScalars(nrs_t *nrs, double time)
{
  cds_t *cds = nrs->cds;
  for (int is = 0; is < cds->NSfields; is++) {
    mesh_t *mesh = cds->mesh[0];
    oogs_t *gsh = cds->gshT;
    if (is) {
      mesh = cds->meshV;
      gsh = cds->gsh;
    }

    occa::memory& o_S = cds->o_S;
    occa::memory& o_Se = cds->o_Se;

    platform->linAlg->fill(cds->fieldOffset[is],
                           -1.0 * std::numeric_limits<dfloat>::max(),
                           platform->o_mempool.slice2);

    for (int sweep = 0; sweep < 2; sweep++) {
      cds->dirichletBCKernel(mesh->Nelements,
                             cds->fieldOffset[is],
                             is,
                             time,
                             mesh->o_sgeo,
                             mesh->o_x,
                             mesh->o_y,
                             mesh->o_z,
                             mesh->o_vmapM,
                             mesh->o_EToB,
                             cds->o_EToB[is],
                             *(cds->o_usrwrk),
                             platform->o_mempool.slice2);

      if (sweep == 0)
        oogs::startFinish(platform->o_mempool.slice2, 1, cds->fieldOffset[is], ogsDfloat, ogsMax, gsh);
      if (sweep == 1)
        oogs::startFinish(platform->o_mempool.slice2, 1, cds->fieldOffset[is], ogsDfloat, ogsMin, gsh);
    }

    if (cds->solver[is]->Nmasked) {
      occa::memory o_Si =
        o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

      cds->maskCopyKernel(cds->solver[is]->Nmasked,
                          0,
                          cds->solver[is]->o_maskIds,
                          platform->o_mempool.slice2,
                          o_Si);

      occa::memory o_Si_e =
        o_Se.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

      cds->maskCopyKernel(cds->solver[is]->Nmasked,
                          0,
                          cds->solver[is]->o_maskIds,
                          platform->o_mempool.slice2,
                          o_Si_e);
    }
  }
}

void applyDirichletVelocity(nrs_t *nrs, double time)
{
  mesh_t *mesh = nrs->meshV;

  occa::memory& o_p = nrs->o_P;
  occa::memory& o_U = nrs->o_U;
  occa::memory& o_Ue = nrs->o_Ue;

  platform->linAlg->fill((1 + nrs->NVfields) * nrs->fieldOffset,
                         -1.0 * std::numeric_limits<dfloat>::max(),
                         platform->o_mempool.slice6);

  for (int sweep = 0; sweep < 2; sweep++) {
    nrs->pressureDirichletBCKernel(mesh->Nelements,
                                   time,
                                   nrs->fieldOffset,
                                   mesh->o_sgeo,
                                   mesh->o_x,
                                   mesh->o_y,
                                   mesh->o_z,
                                   mesh->o_vmapM,
                                   mesh->o_EToB,
                                   nrs->o_EToB,
                                   nrs->o_usrwrk,
                                   o_U,
                                   platform->o_mempool.slice6);

    nrs->velocityDirichletBCKernel(mesh->Nelements,
                                   nrs->fieldOffset,
                                   time,
                                   mesh->o_sgeo,
                                   nrs->o_zeroNormalMaskVelocity,
                                   mesh->o_x,
                                   mesh->o_y,
                                   mesh->o_z,
                                   mesh->o_vmapM,
                                   mesh->o_EToB,
                                   nrs->o_EToB,
                                   nrs->o_usrwrk,
                                   o_U,
                                   platform->o_mempool.slice7);

    if (sweep == 0)
      oogs::startFinish(platform->o_mempool.slice6,
                        1 + nrs->NVfields,
                        nrs->fieldOffset,
                        ogsDfloat,
                        ogsMax,
                        nrs->gsh);
    if (sweep == 1)
      oogs::startFinish(platform->o_mempool.slice6,
                        1 + nrs->NVfields,
                        nrs->fieldOffset,
                        ogsDfloat,
                        ogsMin,
                        nrs->gsh);
  }

  if (nrs->pSolver->Nmasked)
    nrs->maskCopyKernel(nrs->pSolver->Nmasked,
                        0,
                        nrs->pSolver->o_maskIds,
                        platform->o_mempool.slice6,
                        o_p);

  if (nrs->uvwSolver) {
    if (nrs->uvwSolver->Nmasked) {
      nrs->maskCopyKernel(nrs->uvwSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->uvwSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_U);
      nrs->maskCopyKernel(nrs->uvwSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->uvwSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_Ue);
    }
    if (bcMap::unalignedBoundary(mesh->cht, "velocity")) {
      applyZeroNormalMask(nrs, nrs->uvwSolver->o_EToB, nrs->o_zeroNormalMaskVelocity, o_U);
      applyZeroNormalMask(nrs, nrs->uvwSolver->o_EToB, nrs->o_zeroNormalMaskVelocity, o_Ue);
    }
  }
  else {
    if (nrs->uSolver->Nmasked) {
      nrs->maskCopyKernel(nrs->uSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->uSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_U);
      nrs->maskCopyKernel(nrs->uSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->uSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_Ue);
    }
    if (nrs->vSolver->Nmasked) {
      nrs->maskCopyKernel(nrs->vSolver->Nmasked,
                          1 * nrs->fieldOffset,
                          nrs->vSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_U);
      nrs->maskCopyKernel(nrs->vSolver->Nmasked,
                          1 * nrs->fieldOffset,
                          nrs->vSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_Ue);
    }
    if (nrs->wSolver->Nmasked) {
      nrs->maskCopyKernel(nrs->wSolver->Nmasked,
                          2 * nrs->fieldOffset,
                          nrs->wSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_U);
      nrs->maskCopyKernel(nrs->wSolver->Nmasked,
                          2 * nrs->fieldOffset,
                          nrs->wSolver->o_maskIds,
                          platform->o_mempool.slice7,
                          o_Ue);
    }
  }
}
