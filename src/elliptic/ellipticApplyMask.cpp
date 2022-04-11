#include <elliptic.h>
#include <ellipticApplyMask.hpp>
void ellipticEnforceUnZero(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  auto *mesh = solver->mesh;
  ellipticEnforceUnZero(solver, mesh->NlocalGatherElements, mesh->o_localGatherElementList, o_x, precision);
  ellipticEnforceUnZero(solver, mesh->NglobalGatherElements, mesh->o_globalGatherElementList, o_x, precision);
}

void ellipticConstructAvgNormal(elliptic_t *solver)
{
  bool bail = solver->o_avgNormal.size() != 0;
  bail &= !solver->recomputeAvgNormals;
  bail |= !solver->UNormalZero;
  if (bail) {
    return;
  }

  auto *mesh = solver->mesh;

  solver->recomputeAvgNormals = false;

  if (solver->o_avgNormal.size() == 0) {
    solver->o_avgNormal = platform->device.malloc(3 * solver->Ntotal * sizeof(dfloat));
    solver->o_avgTangential1 = platform->device.malloc(3 * solver->Ntotal * sizeof(dfloat));
    solver->o_avgTangential2 = platform->device.malloc(3 * solver->Ntotal * sizeof(dfloat));
    solver->o_isSYM = platform->device.malloc(mesh->Nlocal * sizeof(dfloat));
  }

  solver->copySYMNormalKernel(mesh->Nelements,
                              solver->Ntotal,
                              mesh->o_sgeo,
                              mesh->o_vmapM,
                              mesh->o_EToB,
                              solver->o_BCType,
                              solver->o_avgNormal,
                              solver->o_isSYM);
  oogs::startFinish(solver->o_isSYM, 1, solver->Ntotal, ogsDfloat, ogsMin, solver->oogs);

  // collocate with mask
  auto o_nx = solver->o_avgNormal + 0 * solver->Ntotal * sizeof(dfloat);
  auto o_ny = solver->o_avgNormal + 1 * solver->Ntotal * sizeof(dfloat);
  auto o_nz = solver->o_avgNormal + 2 * solver->Ntotal * sizeof(dfloat);

  platform->linAlg->axmy(mesh->Nlocal, 1.0, solver->o_isSYM, o_nx);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, solver->o_isSYM, o_ny);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, solver->o_isSYM, o_nz);

  oogs::startFinish(solver->o_avgNormal, 3, solver->Ntotal, ogsDfloat, ogsAdd, solver->oogs);

  platform->linAlg->unitVector(mesh->Nlocal, solver->Ntotal, solver->o_avgNormal);

  solver->volumetricTangentialKernel(mesh->Nlocal,
                                     solver->Ntotal,
                                     solver->o_avgNormal,
                                     solver->o_avgTangential1);

  oogs::startFinish(solver->o_avgTangential1, 3, solver->Ntotal, ogsDfloat, ogsAdd, solver->oogs);
  platform->linAlg->unitVector(mesh->Nlocal, solver->Ntotal, solver->o_avgTangential1);

  platform->linAlg->crossProduct(mesh->Nlocal,
                                 solver->Ntotal,
                                 solver->o_avgNormal,
                                 solver->o_avgTangential1,
                                 solver->o_avgTangential2);
  oogs::startFinish(solver->o_avgTangential2, 3, solver->Ntotal, ogsDfloat, ogsAdd, solver->oogs);
  platform->linAlg->unitVector(mesh->Nlocal, solver->Ntotal, solver->o_avgTangential2);
}

void ellipticEnforceUnZero(elliptic_t *solver,
                           dlong Nelements,
                           occa::memory &o_elemList,
                           occa::memory &o_x,
                           std::string precision)
{
  if (solver->o_avgNormal.size() == 0 || solver->recomputeAvgNormals) {
    ellipticConstructAvgNormal(solver);
  }

  auto *mesh = solver->mesh;

  occa::kernel &enforceUnKernel =
      (precision != dfloatString) ? solver->enforceUnPfloatKernel : solver->enforceUnKernel;
  if (Nelements > 0) {
    enforceUnKernel(Nelements,
                    solver->Ntotal,
                    o_elemList,
                    solver->o_avgNormal,
                    solver->o_avgTangential1,
                    solver->o_avgTangential2,
                    mesh->o_vmapM,
                    mesh->o_EToB,
                    solver->o_BCType,
                    o_x);
  }
}

void ellipticApplyMask(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  mesh_t *mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;

  if (solver->UNormalZero) {
    ellipticEnforceUnZero(solver, o_x, precision);
  }

  const dlong Nmasked = solver->Nmasked;
  occa::memory &o_maskIds = solver->o_maskIds;
  if (Nmasked) {
    maskKernel(Nmasked, o_maskIds, o_x);
  }
}
void ellipticApplyMaskInterior(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  mesh_t *mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;

  if (solver->UNormalZero) {
    dlong Nelems = mesh->NlocalGatherElements;
    occa::memory &o_elemList = solver->mesh->o_localGatherElementList;
    ellipticEnforceUnZero(solver, Nelems, o_elemList, o_x, precision);
  }

  const dlong Nmasked = solver->NmaskedLocal;
  occa::memory &o_maskIds = solver->o_maskIdsLocal;
  if (Nmasked) {
    maskKernel(Nmasked, o_maskIds, o_x);
  }
}

void ellipticApplyMaskExterior(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  mesh_t *mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;
  occa::kernel &enforceUnKernel =
      (precision != dfloatString) ? solver->enforceUnPfloatKernel : solver->enforceUnKernel;

  const dlong Nmasked = solver->NmaskedGlobal;
  occa::memory &o_maskIds = solver->o_maskIdsGlobal;

  if (solver->UNormalZero) {
    const dlong Nelems = mesh->NglobalGatherElements;
    occa::memory &o_elemList = solver->mesh->o_globalGatherElementList;
    ellipticEnforceUnZero(solver, Nelems, o_elemList, o_x, precision);
  }
  if (Nmasked) {
    maskKernel(Nmasked, o_maskIds, o_x);
  }
}