
#include <mpi.h>
#include "nrs.hpp"
#include "platform.hpp"
#include <vector>

#include "findpts.hpp"

#include "pointInterpolation.hpp"
#include <algorithm>

pointInterpolation_t::pointInterpolation_t(nrs_t *nrs_, double newton_tol_, bool profile_)
    : nrs(nrs_), newton_tol(newton_tol_), profile(profile_), nPoints(0)
{

  newton_tol = std::max(5e-13, newton_tol_);

  const int npt_max = 128;
  const dfloat bb_tol = 0.01;

  mesh_t *mesh = nrs->meshV;

  // used for # of cells in hash tables
  const dlong hash_size = mesh->Nlocal;

  MPI_Comm comm = platform_t::getInstance()->comm.mpiComm;

  findpts_ = findptsSetup(
                            comm,
                            mesh->x,
                            mesh->y,
                            mesh->z,
                            mesh->Nq,
                            mesh->Nelements,
                            2 * mesh->Nq,
                            bb_tol,
                            hash_size,
                            hash_size,
                            npt_max,
                            newton_tol,
                            platform_t::getInstance()->device.occaDevice());
}

pointInterpolation_t::~pointInterpolation_t() { findptsFree(findpts_); }

void pointInterpolation_t::find(bool printWarnings)
{
  if(profile){
    platform->timer.tic("pointInterpolation_t::find", 1);
  }

  const auto n = nPoints;

  findpts(&data_, _x, _y, _z, n, findpts_);

  if (printWarnings) {
    dlong nFail = 0;
    for (int in = 0; in < n; ++in) {
      if (data_.code_base[in] == 1) {
        if (data_.dist2_base[in] > 10 * newton_tol) {
          nFail += 1;
          if (nFail < 5) {
            std::cerr << " WARNING: point on boundary or outside the mesh xy[z]d^2: " << _x[in] << ","
                      << _y[in] << ", " << _z[in] << ", " << data_.dist2_base[in] << std::endl;
          }
        }
      }
      else if (data_.code_base[in] == 2) {
        nFail += 1;
        if (nFail < 5) {
          std::cerr << " WARNING: point not within mesh xy[z]: " << _x[in] << "," << _y[in] << ", "
                    << _z[in] << std::endl;
        }
      }
    }
    hlong counts[4] = {n, nFail, 0, 0};
    MPI_Reduce(counts, counts + 2, 2, MPI_HLONG, MPI_SUM, 0, platform_t::getInstance()->comm.mpiComm);
    if (platform_t::getInstance()->comm.mpiRank == 0 && counts[3] > 0) {
      std::cout << "interp::find - Total number of points = " << counts[2] << ", failed = " << counts[3]
                << std::endl;
    }
  }

  if(profile){
    platform->timer.toc("pointInterpolation_t::find");
  }
}

void pointInterpolation_t::eval(occa::memory o_fields,
                                const dlong nFields,
                                dfloat *out)
{
  if(profile){
    platform->timer.tic("pointInterpolation_t::eval", 1);
  }
  
  const auto n = data_.code.size();
  for (int i = 0; i < nFields; ++i) {
    findptsEval(n, o_fields + i * nrs->fieldOffset * sizeof(dfloat), findpts_, &data_, out + i * n);
  }

  if(profile){
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::evalLocalPoints(occa::memory o_fields,
                                           const dlong nFields,
                                           const dlong offset,
                                           const dlong *el,
                                           const dfloat *r,
                                           occa::memory o_out,
                                           dlong n)
{
  if (n == 0 || nFields == 0) {
    return;
  }

  // re-allocate
  if ( n * sizeof(dlong) > o_el.size()){
    o_r.free();
    o_el.free();

    o_el = platform->device.malloc(n * sizeof(dlong));
    o_r = platform->device.malloc(3 * n * sizeof(dfloat));
  }

  o_r.copyFrom(r, 3 * n * sizeof(dfloat));
  o_el.copyFrom(el, n * sizeof(dlong));

  findptsLocalEval(n, nFields, nrs->fieldOffset, offset, o_fields, o_el, o_r, findpts_, o_out);
}

void pointInterpolation_t::addPoints(int n, dfloat * x, dfloat * y, dfloat * z)
{

  if(n > nPoints){
    data_ = findpts_data_t(n);
  }

  nPoints = n;

  _x = x;
  _y = y;
  _z = z;

}