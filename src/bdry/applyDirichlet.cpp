#include "nrs.hpp"
#include "applyDirichlet.hpp"
#include "applyZeroNormalMask.hpp"
#include "bcMap.hpp"

void applyDirichlet(nrs_t *nrs, double time)
{
  if (nrs->Nscalar) {
    cds_t *cds = nrs->cds;
    for (int is = 0; is < cds->NSfields; is++) {
      mesh_t *mesh = cds->mesh[0];
      ;
      oogs_t *gsh = cds->gshT;
      if (is) {
        mesh = cds->meshV;
        gsh = cds->gsh;
      }

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
      occa::memory o_Si =
          cds->o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));
      if (cds->solver[is]->Nmasked)
        cds->maskCopyKernel(cds->solver[is]->Nmasked,
                            0,
                            cds->solver[is]->o_maskIds,
                            platform->o_mempool.slice2,
                            o_Si);
    }
  }

  if (nrs->flow) {
    mesh_t *mesh = nrs->meshV;

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
                                     nrs->o_U,
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
                                     nrs->o_U,
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
                          nrs->o_P);

    if (nrs->uvwSolver) {
      if (bcMap::unalignedBoundary(mesh->cht, "velocity")) {
        applyZeroNormalMask(nrs, nrs->o_EToB, nrs->o_zeroNormalMaskVelocity, nrs->o_U);
      }
      if (nrs->uvwSolver->Nmasked)
        nrs->maskCopyKernel(nrs->uvwSolver->Nmasked,
                            0 * nrs->fieldOffset,
                            nrs->uvwSolver->o_maskIds,
                            platform->o_mempool.slice7,
                            nrs->o_U);
    }
    else {
      if (nrs->uSolver->Nmasked)
        nrs->maskCopyKernel(nrs->uSolver->Nmasked,
                            0 * nrs->fieldOffset,
                            nrs->uSolver->o_maskIds,
                            platform->o_mempool.slice7,
                            nrs->o_U);
      if (nrs->vSolver->Nmasked)
        nrs->maskCopyKernel(nrs->vSolver->Nmasked,
                            1 * nrs->fieldOffset,
                            nrs->vSolver->o_maskIds,
                            platform->o_mempool.slice7,
                            nrs->o_U);
      if (nrs->wSolver->Nmasked)
        nrs->maskCopyKernel(nrs->wSolver->Nmasked,
                            2 * nrs->fieldOffset,
                            nrs->wSolver->o_maskIds,
                            platform->o_mempool.slice7,
                            nrs->o_U);
    }
  }

  if (platform->options.compareArgs("MESH SOLVER", "ELASTICITY")) {
    mesh_t *mesh = nrs->meshV;
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

    if (bcMap::unalignedBoundary(mesh->cht, "mesh")) {
      applyZeroNormalMask(nrs, nrs->o_EToBMeshVelocity, nrs->o_zeroNormalMaskMeshVelocity, mesh->o_U);
    }
    if (nrs->meshSolver->Nmasked)
      nrs->maskCopyKernel(nrs->meshSolver->Nmasked,
                          0 * nrs->fieldOffset,
                          nrs->meshSolver->o_maskIds,
                          platform->o_mempool.slice3,
                          mesh->o_U);
  }
}