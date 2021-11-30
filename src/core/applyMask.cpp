#include "nrs.hpp"
#include <applyMask.hpp>
void applyMask(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  mesh_t* mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;

  const dlong Nmasked = solver->Nmasked;
  occa::memory &o_maskIds = solver->o_maskIds;
  if (Nmasked)
    maskKernel(Nmasked, o_maskIds, o_x);
}
void applyMaskInterior(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  mesh_t *mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;

  const dlong Nmasked = solver->NmaskedLocal;
  occa::memory &o_maskIds = solver->o_maskIdsLocal;
  if (Nmasked)
    maskKernel(Nmasked, o_maskIds, o_x);
}

void applyMaskExterior(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  mesh_t *mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;

  const dlong Nmasked = solver->NmaskedGlobal;
  occa::memory &o_maskIds = solver->o_maskIdsGlobal;
  if (Nmasked)
    maskKernel(Nmasked, o_maskIds, o_x);
}
void applyMaskUnaligned(nrs_t *nrs,
                        occa::memory &o_mask,
                        elliptic_t *solver,
                        occa::memory &o_x,
                        std::string precision)
{
  applyMaskUnalignedInterior(nrs, o_mask, solver, o_x, precision);
  applyMaskUnalignedExterior(nrs, o_mask, solver, o_x, precision);
}
void applyMaskUnalignedInterior(nrs_t *nrs,
                                occa::memory &o_mask,
                                elliptic_t *solver,
                                occa::memory &o_x,
                                std::string precision)
{
  const dlong Nelems = solver->mesh->NlocalGatherElements;
  if (Nelems == 0)
    return;
  occa::memory &o_elemList = solver->mesh->o_localGatherElementList;

  nrs->enforceUnKernel(Nelems,
                       nrs->fieldOffset,
                       o_elemList,
                       nrs->o_Vn,
                       nrs->o_V1,
                       nrs->o_V2,
                       o_mask,
                       solver->mesh->o_vmapM,
                       nrs->o_EToB,
                       o_x);

  applyMaskInterior(solver, o_x, precision);
}
void applyMaskUnalignedExterior(nrs_t *nrs,
                                occa::memory &o_mask,
                                elliptic_t *solver,
                                occa::memory &o_x,
                                std::string precision)
{
  const dlong Nelems = solver->mesh->NglobalGatherElements;
  if (Nelems == 0)
    return;
  occa::memory &o_elemList = solver->mesh->o_globalGatherElementList;

  nrs->enforceUnKernel(Nelems,
                       nrs->fieldOffset,
                       o_elemList,
                       nrs->o_Vn,
                       nrs->o_V1,
                       nrs->o_V2,
                       o_mask,
                       solver->mesh->o_vmapM,
                       nrs->o_EToB,
                       o_x);

  applyMaskExterior(solver, o_x, precision);
}