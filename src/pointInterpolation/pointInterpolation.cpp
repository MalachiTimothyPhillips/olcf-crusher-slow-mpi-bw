
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

void pointInterpolation_t::eval(const dfloat *fields,
                                const dlong nFields,
                                findpts_data_t *findPtsdata_,
                                dfloat **out,
                                const dlong outStride[],
                                dlong n)
{
  if(profile){
    platform->timer.tic("pointInterpolation_t::eval", 1);
  }
  dlong fieldOffset = nrs->fieldOffset;
  for (int i = 0; i < nFields; ++i) {
    findptsEval(out[i], findPtsdata_, n, fields + i * nrs->fieldOffset, findpts_);
  }
  if(profile){
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::eval(occa::memory o_fields,
                                const dlong nFields,
                                findpts_data_t *findPtsdata_,
                                dfloat **out,
                                const dlong outStride[],
                                dlong n)
{
  if(profile){
    platform->timer.tic("pointInterpolation_t::eval", 1);
  }
  for (int i = 0; i < nFields; ++i) {
    findptsEval(out[i], findPtsdata_, n, o_fields + i * nrs->fieldOffset * sizeof(dfloat), findpts_);
  }
  if(profile){
    platform->timer.toc("pointInterpolation_t::eval");
  }
}

void pointInterpolation_t::evalLocalPoints(const dfloat *fields,
                                           const dlong nFields,
                                           const dlong *el,
                                           const dlong elStride,
                                           const dfloat *r,
                                           const dlong rStride,
                                           dfloat **out,
                                           const dlong outStride[],
                                           dlong n)
{
  if (n == 0 || nFields == 0) {
    return;
  }
  dlong fieldOffset = nrs->fieldOffset;
  for (int i = 0; i < nFields; ++i) {
    findptsLocalEval(out[i],
                        outStride[i] * sizeof(dfloat),
                        el,
                        elStride * sizeof(dlong),
                        r,
                        rStride * sizeof(dfloat),
                        n,
                        fields + i * fieldOffset,
                        findpts_);
  }
}

void pointInterpolation_t::evalLocalPoints(occa::memory o_fields,
                                           const dlong nFields,
                                           const dlong *el,
                                           const dlong elStride,
                                           const dfloat *r,
                                           const dlong rStride,
                                           dfloat **out,
                                           const dlong outStride[],
                                           dlong n)
{
  if (n == 0 || nFields == 0) {
    return;
  }

  dlong elStrideBytes = elStride * sizeof(dlong);
  dlong rStrideBytes = rStride * sizeof(dfloat);

  dlong outBytes = 0;
  bool unitOutStride = true;
  for (dlong i = 0; i < nFields; ++i) {
    unitOutStride &= outStride[i] == 1;
    outBytes += outStride[i];
  }
  outBytes *= sizeof(dfloat);

  occa::device device = *findpts_->device;
  dlong allocSize = (nFields * outBytes + rStrideBytes + elStrideBytes) * n;
  occa::memory workspace;
  occa::memory mempool = platform_t::getInstance()->o_mempool.o_ptr;
  if (allocSize < mempool.size()) {
    workspace = mempool.cast(occa::dtype::byte);
  }
  else {
    workspace = device.malloc(allocSize, occa::dtype::byte);
  }
  occa::memory o_out = workspace;
  workspace += n * nFields * sizeof(dfloat);
  occa::memory o_r = workspace;
  workspace += n * rStrideBytes;
  occa::memory o_el = workspace;
  workspace += n * elStrideBytes;
  o_r.copyFrom(r, rStrideBytes * n);
  o_el.copyFrom(el, elStrideBytes * n);

  dlong fieldOffset = nrs->fieldOffset;
  for (int i = 0; i < nFields; ++i) {
    occa::memory o_out_i = o_out.slice(i * sizeof(dfloat) * n, sizeof(dfloat) * n);
    findptsLocalEval(o_out_i,
                        sizeof(dfloat),
                        o_el,
                        elStrideBytes,
                        o_r,
                        rStrideBytes,
                        n,
                        o_fields + i * fieldOffset * sizeof(dfloat),
                        findpts_);
  }
  if (unitOutStride) {
    // combine d->h copies where able
    dlong i = 0;
    while (i < nFields) {
      dfloat *start = out[i];
      dlong j = 0;
      while (i + j < nFields && out[i + j] == start + j * n) {
        ++j;
      }
      o_out.copyTo(start, j * n * sizeof(dfloat));
      i += j;
    }
  }
  else {
    dfloat *outTemp = new dfloat[nFields * sizeof(dfloat) * n];
    o_out.copyTo(outTemp, nFields * sizeof(dfloat) * n);
    for (dlong i = 0; i < nFields; ++i) {
      for (dlong j = 0; j < n; ++j) {
        out[i][j * outStride[i]] = outTemp[i * n + j];
      }
    }
  }
}

void pointInterpolation_t::addPoints(int n, dfloat * x, dfloat * y, dfloat * z)
{

  if(n > nPoints){
    
    dist2.resize(n, 0.0);
    r.resize(3*n, 0.0);

    code.resize(n, 0);
    el.resize(n, 0);
    proc.resize(n, 0);

    data_ = findpts_data_t(code.data(), proc.data(), el.data(), r.data(), dist2.data());
  }

  nPoints = n;

  _x = x;
  _y = y;
  _z = z;

}