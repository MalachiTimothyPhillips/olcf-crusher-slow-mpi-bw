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

  auto o_EToBV = platform->o_mempool.slice6;

  nrs->createEToBVKernel(mesh->Nelements, o_EToB, mesh->o_faceNodes, o_EToBV);

  // TODO: implement int type reductions in oogs
  //oogs::startFinish(o_EToBV, 1, nrs->fieldOffset, ogsInt, ogsMin, nrs->gsh);
  ogsGatherScatter(o_EToBV, ogsInt, ogsMin, nrs->gsh->ogs);

  nrs->zeroLargeNumberKernel(mesh->Nlocal, o_EToBV);

  platform->linAlg->fill(3 * nrs->fieldOffset, 0.0, o_mask);

  nrs->initializeZeroNormalMaskKernel(mesh->Nlocal, nrs->fieldOffset, o_EToBV, o_mask);

  // normal + count (4 fields)
  auto o_avgNormal = platform->o_mempool.slice0;

  nrs->averageNormalBcTypeKernel(mesh->Nelements,
                                 nrs->fieldOffset,
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