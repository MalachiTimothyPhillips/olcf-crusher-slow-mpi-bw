#include "nrs.hpp"
#include "cvodeSolver.hpp"
#include "nrssys.hpp"
#include "nrs.hpp"
#include "mapLVector.hpp"
#include "ogs.hpp"
#include <numeric>
#include <vector>
#include <set>
#include <map>

namespace cvode{

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

  o_Lids.free();
}

}