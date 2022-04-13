#include "createZeroNormalMask.hpp"
#include "linAlg.hpp"
#include "nrs.hpp"
#include "nrssys.hpp"
#include "platform.hpp"

// pre: o_EToB allocated, capacity mesh->Nfaces * mesh->Nelements dfloat words
//      o_mask allocated, capacity 3 nrs->fieldOffset dfloat words
void createZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory &o_mask)
{

  auto mesh = nrs->meshV;

  auto o_avgNormal = platform->o_mempool.slice0;
  auto o_isSYM = platform->o_mempool.slice3;

  nrs->copySYMNormalKernel(mesh->Nelements,
                           nrs->fieldOffset,
                           mesh->o_sgeo,
                           mesh->o_vmapM,
                           o_EToB,
                           o_avgNormal,
                           o_isSYM);

  oogs::startFinish(o_isSYM, 1, nrs->fieldOffset, ogsDfloat, ogsMin, nrs->gsh);

  // collocate with mask
  auto o_nx = o_avgNormal + 0 * nrs->fieldOffset * sizeof(dfloat);
  auto o_ny = o_avgNormal + 1 * nrs->fieldOffset * sizeof(dfloat);
  auto o_nz = o_avgNormal + 2 * nrs->fieldOffset * sizeof(dfloat);

  platform->linAlg->axmy(mesh->Nlocal, 1.0, o_isSYM, o_nx);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, o_isSYM, o_ny);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, o_isSYM, o_nz);

  oogs::startFinish(o_avgNormal, 3, nrs->fieldOffset, ogsDfloat, ogsAdd, nrs->gsh);

  platform->linAlg->unitVector(mesh->Nlocal, nrs->fieldOffset, o_avgNormal);

  platform->linAlg->fill(3 * nrs->fieldOffset, 1.0, o_mask);

  nrs->fixMaskKernel(mesh->Nelements,
                     nrs->fieldOffset,
                     mesh->o_sgeo,
                     mesh->o_vmapM,
                     o_EToB,
                     o_avgNormal,
                     o_mask);
}