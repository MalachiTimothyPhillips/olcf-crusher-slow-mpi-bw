
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

  dlong nmsh = mesh->N;
  dlong nelm = mesh->Nelements;

  // element geometry
  dfloat *elx[3] = {mesh->x, mesh->y, mesh->z};

  // element dimensions
  dlong n1[3] = {mesh->N + 1, mesh->N + 1, mesh->N + 1};

  dlong m1[3] = {2 * n1[0], 2 * n1[1], 2 * n1[2]};

  // used for # of cells in hash tables
  const dlong hash_size = nelm * n1[0] * n1[1] * n1[2];

  MPI_Comm comm = platform_t::getInstance()->comm.mpiComm;

  findpts_ = findptsSetup(3,
                            comm,
                            elx,
                            n1,
                            nelm,
                            m1,
                            bb_tol,
                            hash_size,
                            hash_size,
                            npt_max,
                            newton_tol,
                            &platform_t::getInstance()->device.occaDevice());
}

pointInterpolation_t::~pointInterpolation_t() { findptsFree(findpts_); }

void pointInterpolation_t::find(bool printWarnings)
{
  if(profile){
    platform->timer.tic("pointInterpolation_t::find", 1);
  }

  dfloat * x[3] = {_x, _y, _z};
  const auto n = nPoints;

  dlong xStrideBytes[3] = {sizeof(dfloat),
                           sizeof(dfloat),
                           sizeof(dfloat)};

  findpts(&data_, x, xStrideBytes, n, findpts_);

  if (printWarnings) {
    dlong nFail = 0;
    for (int in = 0; in < n; ++in) {
      if (data_.code_base[in] == 1) {
        if (data_.dist2_base[in] > 10 * newton_tol) {
          nFail += 1;
          if (nFail < 5) {
            std::cerr << " WARNING: point on boundary or outside the mesh xy[z]d^2: " << x[0][in] << ","
                      << x[1][in] << ", " << x[2][in] << ", " << data_.dist2_base[in] << std::endl;
          }
        }
      }
      else if (data_.code_base[in] == 2) {
        nFail += 1;
        if (nFail < 5) {
          std::cerr << " WARNING: point not within mesh xy[z]: " << x[0][in] << "," << x[1][in] << ", "
                    << x[2][in] << std::endl;
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
    findptsEval(out + i * n, &data_, n, o_fields + i * nrs->fieldOffset * sizeof(dfloat), findpts_);
  }

  if(profile){
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::evalLocalPoints(occa::memory o_fields,
                                           const dlong nFields,
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

  for (int i = 0; i < nFields; ++i) {
    occa::memory o_out_i = o_out + i * n * sizeof(dfloat);
    findptsLocalEval(o_out_i,
                        sizeof(dfloat),
                        o_el,
                        sizeof(dlong),
                        o_r,
                        3 * sizeof(dfloat),
                        n,
                        o_fields + i * nrs->fieldOffset * sizeof(dfloat),
                        findpts_);
  }
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