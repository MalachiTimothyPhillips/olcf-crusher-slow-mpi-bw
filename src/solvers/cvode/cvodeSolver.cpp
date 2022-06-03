#include "cvodeSolver.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "Urst.hpp"
#include <limits>
#include <array>

#include <cvode/cvode.h>

namespace cvode {

void cvodeSolver_t::reallocBuffer(dlong Nbytes)
{
  if (o_wrk.size() < Nbytes) {
    if (o_wrk.size() > 0)
      o_wrk.free();
    o_wrk = platform->device.malloc(Nbytes);
  }

  if (o_coeffExt.size() == 0) {
    o_coeffExt = platform->device.malloc(maxExtrapolationOrder * sizeof(dfloat));
  }
}

void cvodeSolver_t::rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  if (time != tprev) {
    tprev = time;
    std::array<dfloat, 3> dtCvode = {0, 0, 0};
    std::array<dfloat, 3> coeffAB = {0, 0, 0};

    // TODO: need to check this???
    const auto cvodeDt = time - tf;
    dtCvode[0] = cvodeDt;
    dtCvode[1] = nrs->dt[1];
    dtCvode[2] = nrs->dt[2];

    const int extOrder = std::min(tstep, maxExtrapolationOrder);
    nek::coeffAB(coeffAB.data(), dtCvode.data(), extOrder);
    for (int i = maxExtrapolationOrder; i > extOrder; i--)
      coeffAB[i - 1] = 0.0;

    o_coeffExt.copyFrom(coeffAB.data(), maxExtrapolationOrder * sizeof(dfloat));

    extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, nrs->o_U);

    if (movingMesh) {
      mesh->coeffs(dtCvode.data(), tstep);
      mesh->move();

      extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, mesh->o_U);
    }

    computeUrst(nrs);
  }

  unpack();

  // terms to include: user source, advection, filtering, add "weak" laplacian
  makeqImpl();

  // dssum

  pack();
}

void cvodeSolver_t::setup(nrs_t *nrs, Parameters_t params)
{
  this->initialized = true;

  dlong Nwords = nrs->NVfields * nrs->fieldOffset; // velocities
  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    Nwords += 2 * nrs->NVfields * nrs->fieldOffset; // coordinates, mesh velocities
  }

  reallocBuffer(Nwords * sizeof(dfloat));

  setupEToLMapping(nrs);
}
void cvodeSolver_t::solve(nrs_t *nrs, double t0, double t1, int tstep)
{
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  // copy current state into buffer
  auto o_U = o_wrk + 0 * nrs->fieldOffset * sizeof(dfloat);

  o_U.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  occa::memory o_x;
  occa::memory o_y;
  occa::memory o_z;
  occa::memory o_meshU;

  if (movingMesh) {
    o_x = o_wrk + (nrs->NVfields + 0) * nrs->fieldOffset * sizeof(dfloat);
    o_y = o_wrk + (nrs->NVfields + 1) * nrs->fieldOffset * sizeof(dfloat);
    o_z = o_wrk + (nrs->NVfields + 2) * nrs->fieldOffset * sizeof(dfloat);
    o_meshU = o_wrk + (nrs->NVfields + 3) * nrs->fieldOffset * sizeof(dfloat);

    o_x.copyFrom(mesh->o_x, mesh->Nlocal * sizeof(dfloat));
    o_y.copyFrom(mesh->o_y, mesh->Nlocal * sizeof(dfloat));
    o_z.copyFrom(mesh->o_z, mesh->Nlocal * sizeof(dfloat));

    o_meshU.copyFrom(mesh->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));
  }

  pack();

  // call cvode solver

  unpack();

  // restore previous state
  nrs->o_U.copyFrom(o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  if (movingMesh) {
    mesh->o_x.copyFrom(o_x, mesh->Nlocal * sizeof(dfloat));
    mesh->o_y.copyFrom(o_y, mesh->Nlocal * sizeof(dfloat));
    mesh->o_z.copyFrom(o_z, mesh->Nlocal * sizeof(dfloat));

    mesh->o_U.copyFrom(o_meshU, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

    mesh->update();

    computeUrst(nrs);
  }
}
} // namespace cvode