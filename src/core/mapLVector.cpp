#include "nrssys.hpp"
#include "nrs.hpp"
#include "mapLVector.hpp"
#include "ogs.hpp"
#include <numeric>
#include <vector>
#include <set>
#include <map>

void setupEToLMapping(nrs_t *nrs)
{
  static_assert(sizeof(dlong) == sizeof(int), "dlong and int must be the same size");
  auto *mesh = nrs->meshV;
  if (mesh->cht)
    mesh = nrs->cds->mesh[0];

  auto *handle = oogs::setup(mesh->ogs, 1, nrs->fieldOffset, ogsDfloat, NULL, OOGS_LOCAL);

  auto o_Lids = platform->device.malloc(mesh->Nlocal * sizeof(dlong));
  std::vector<dlong> Eids(mesh->Nlocal);
  std::iota(Eids.begin(), Eids.end(), 0);
  o_Lids.copyFrom(Eids.data(), mesh->Nlocal * sizeof(dlong));

  oogs::startFinish(o_Lids, 1, nrs->fieldOffset, "int", "ogsMin", handle);

  std::vector<dlong> Lids(mesh->Nlocal);
  o_Lids.copyTo(Lids.data(), mesh->Nlocal * sizeof(dlong));

  std::set<dlong> uniqueIds;
  for (auto &&id : Lids) {
    uniqueIds.insert(id);
  }

  const auto NL = uniqueIds.size();

  // compute page-aligned NL
  nrs->LFieldOffset = NL;
  const int pageW = ALIGN_SIZE / sizeof(dfloat);
  if (nrs->LFieldOffset % pageW)
    nrs->LFieldOffset = (nrs->LFieldOffset / pageW + 1) * pageW;

  std::vector<dlong> EToL(mesh->Nlocal);
  std::map<dlong, std::set<dlong>> LToE;
  for (auto &&Eid : Eids) {
    const auto Lid = std::distance(uniqueIds.begin(), uniqueIds.find(Eid));
    EToL[Eid] = Lid;
    LToE[Lid].insert(Eid);
  }

  // use the first Eid value for the mask (arbitrary)
  std::vector<dlong> mask(mesh->Nlocal);
  for (int eid = 0; eid < mesh->Nlocal; ++eid) {
    const auto lid = EToL[eid];
    if (*LToE[lid].begin() == eid) {
      mask[eid] = 1;
    }
    else {
      mask[eid] = 0;
    }
  }

  nrs->o_Lmask = platform->device.malloc(mesh->Nlocal * sizeof(dlong), mask.data());

  nrs->o_EToL = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToL.data());

  std::vector<dlong> starts(NL + 1);
#if 0
  std::iota(starts.begin(), starts.end(), 0);
  std::partial_sum(LToE.begin(), LToE.end(), starts.begin()+1);
#endif
  starts[0] = 0;
  for (int lid = 0; lid < NL; ++lid) {
    starts[lid + 1] = starts[lid] + LToE[lid].size();
  }

  unsigned ctr = 0;
  std::vector<dlong> scatterIds(mesh->Nlocal);
  for (int lid = 0; lid < NL; ++lid) {
    for (auto &&eid : LToE[lid]) {
      scatterIds[ctr++] = eid;
    }
  }

  if (starts[NL] != mesh->Nlocal) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "Error!\n";
    }
    ABORT(1);
  }

  nrs->o_LToEStarts = platform->device.malloc(starts.size() * sizeof(dlong), starts.data());
  nrs->o_LToE = platform->device.malloc(scatterIds.size() * sizeof(dlong), scatterIds.data());

  nrs->mapEToLKernel = platform->kernels.get("mapEToL");
  nrs->mapLToEKernel = platform->kernels.get("mapLToE");

  // oogs::destroy(handle);
  o_Lids.free();
}