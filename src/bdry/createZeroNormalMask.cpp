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

  // TODO: fill o_mask appropriately to start

  // normal + count (4 fields)
  auto o_avgNormal = platform->o_mempool.slice0;

  nrs->volumeAverageNormalKernel(mesh->Nelements,
                                 nrs->fieldOffset,
                                 mesh->o_sgeo,
                                 mesh->o_vmapM,
                                 o_EToB,
                                 o_avgNormal);

  oogs::startFinish(o_avgNormal, 4, nrs->fieldOffset, ogsDfloat, ogsMin, nrs->gsh);

  nrs->fixMaskKernel(mesh->Nelements,
                     nrs->fieldOffset,
                     mesh->o_sgeo,
                     mesh->o_vmapM,
                     o_EToB,
                     o_avgNormal,
                     o_mask);

  oogs::startFinish(o_mask, 3, nrs->fieldOffset, ogsDfloat, ogsMin, nrs->gsh);
}