/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <cassert>
#include <cstdlib>
#include "ogstypes.h"
#include "findpts.hpp"
#include "legacyFindptsSetup.h"
#include "gslib.h"
#include "findptsTypes.h"
#include "findptsImpl.hpp"

extern "C" {
uint findpts_local_hash_opt_size_3(struct findpts_local_hash_data_3 *p,
                                   const struct obbox_3 *const obb,
                                   const uint nel,
                                   const uint max_size);
}

dlong getHashSize(const struct gslibFindptsData_t *fd, dlong nel, dlong max_hash_size)
{
  const findpts_local_data_3 *fd_local = &fd->local;
  auto hash_data_copy = fd_local->hd;
  return findpts_local_hash_opt_size_3(&hash_data_copy, fd_local->obb, nel, max_hash_size);
}

findpts_t *findptsSetup(MPI_Comm comm,
                        const dfloat *const x,
                        const dfloat *const y,
                        const dfloat *const z,
                        const dlong Nq,
                        const dlong Nelements,
                        const dlong m,
                        const dfloat bbox_tol,
                        const hlong local_hash_size,
                        const hlong global_hash_size,
                        const dlong npt_max,
                        const dfloat newt_tol,
                        occa::device &device)
{

  const dlong Nlocal = Nq * Nq * Nq * Nelements;

  const dfloat *elx[3] = {x, y, z};
  const int n[3] = {Nq, Nq, Nq};
  const int ms[3] = {m, m, m};

  auto findpts_data = legacyFindptsSetup(comm,
                                         elx,
                                         n,
                                         Nelements,
                                         ms,
                                         bbox_tol,
                                         local_hash_size,
                                         global_hash_size,
                                         npt_max,
                                         newt_tol);

  findpts_t *handle = new findpts_t();
  handle->D = 3;
  handle->tol = findpts_data->local.tol;
  handle->hash = &findpts_data->hash;
  handle->cr = &findpts_data->cr;

  if (x != nullptr) {
    handle->o_x = device.malloc(Nlocal * sizeof(dfloat));
    handle->o_y = device.malloc(Nlocal * sizeof(dfloat));
    handle->o_z = device.malloc(Nlocal * sizeof(dfloat));

    handle->o_x.copyFrom(x, Nlocal * sizeof(dfloat));
    handle->o_y.copyFrom(y, Nlocal * sizeof(dfloat));
    handle->o_z.copyFrom(z, Nlocal * sizeof(dfloat));
    std::vector<dfloat> c(3 * Nelements, 0.0);
    std::vector<dfloat> A(9 * Nelements, 0.0);
    std::vector<dfloat> minBound(3 * Nelements, 0.0);
    std::vector<dfloat> maxBound(3 * Nelements, 0.0);

    for (int e = 0; e < Nelements; ++e) {
      auto box = findpts_data->local.obb[e];

      c[3 * e + 0] = box.c0[0];
      c[3 * e + 1] = box.c0[1];
      c[3 * e + 2] = box.c0[2];

      minBound[3 * e + 0] = box.x[0].min;
      minBound[3 * e + 1] = box.x[1].min;
      minBound[3 * e + 2] = box.x[2].min;

      maxBound[3 * e + 0] = box.x[0].max;
      maxBound[3 * e + 1] = box.x[1].max;
      maxBound[3 * e + 2] = box.x[2].max;

      for (int i = 0; i < 9; ++i) {
        A[9 * e + i] = box.A[i];
      }
    }

    handle->o_c = device.malloc(c.size() * sizeof(dfloat));
    handle->o_A = device.malloc(A.size() * sizeof(dfloat));
    handle->o_min = device.malloc(minBound.size() * sizeof(dfloat));
    handle->o_max = device.malloc(maxBound.size() * sizeof(dfloat));

    handle->o_c.copyFrom(c.data(), c.size() * sizeof(dfloat));
    handle->o_A.copyFrom(A.data(), A.size() * sizeof(dfloat));
    handle->o_min.copyFrom(minBound.data(), minBound.size() * sizeof(dfloat));
    handle->o_max.copyFrom(maxBound.data(), maxBound.size() * sizeof(dfloat));
  }

  auto hash = findpts_data->local.hd;
  for (int d = 0; d < 3; ++d) {
    handle->hashMin[d] = hash.bnd[d].min;
    handle->hashFac[d] = hash.fac[d];
  }
  handle->hash_n = hash.hash_n;

  handle->device = device;
  auto kernels = initFindptsKernels(comm, device, 3, Nq);
  handle->local_eval_kernel = kernels.at(0);
  handle->local_eval_many_kernel = kernels.at(1);
  handle->local_kernel = kernels.at(2);

  handle->o_wtend_x = device.malloc(6 * Nq * sizeof(dfloat));
  handle->o_wtend_y = device.malloc(6 * Nq * sizeof(dfloat));
  handle->o_wtend_z = device.malloc(6 * Nq * sizeof(dfloat));
  handle->o_wtend_x.copyFrom(findpts_data->local.fed.wtend[0], 6 * Nq * sizeof(dfloat));
  handle->o_wtend_y.copyFrom(findpts_data->local.fed.wtend[1], 6 * Nq * sizeof(dfloat));
  handle->o_wtend_z.copyFrom(findpts_data->local.fed.wtend[2], 6 * Nq * sizeof(dfloat));

  const auto hd_d_size = getHashSize(findpts_data, Nelements, local_hash_size);

  std::vector<dlong> offsets(hd_d_size, 0);
  for (dlong i = 0; i < hd_d_size; ++i) {
    offsets[i] = findpts_data->local.hd.offset[i];
  }
  handle->o_offset = device.malloc(offsets.size() * sizeof(dlong));
  handle->o_offset.copyFrom(offsets.data(), offsets.size() * sizeof(dlong));

  return handle;
}

void findptsFree(findpts_t *fd)
{
  // Use OCCA's reference counting to free memory and kernel objects
  fd->local_eval_kernel = occa::kernel();
  fd->local_kernel = occa::kernel();
  delete fd;
}

#define D 3

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

static slong lfloor(double x) { return floor(x); }
static slong lceil(double x) { return ceil(x); }

static ulong hash_index_aux(double low, double fac, ulong n, double x)
{
  const slong i = lfloor((x - low) * fac);
  return i < 0 ? 0 : (n - 1 < (ulong)i ? n - 1 : (ulong)i);
}

static ulong hash_index_3(const hashData_t *p, const double x[D])
{
  const ulong n = p->hash_n;
  return (hash_index_aux(p->bnd[2].min, p->fac[2], n, x[2]) * n +
          hash_index_aux(p->bnd[1].min, p->fac[1], n, x[1])) *
             n +
         hash_index_aux(p->bnd[0].min, p->fac[0], n, x[0]);
}

struct srcPt_t {
  double x[D];
  int index, proc;
};
struct outPt_t {
  double r[D], dist2;
  int index, code, el, proc;
};

void findpts(findpts_data_t *const findPtsData,
             const dfloat *const x_base[],
             const dlong npt,
             findpts_t *const fd)
{
  int *const code_base = findPtsData->code_base;
  int *const proc_base = findPtsData->proc_base;
  int *const el_base = findPtsData->el_base;
  double *const r_base = findPtsData->r_base;
  double *const dist2_base = findPtsData->dist2_base;
  hashData_t &hash = *fd->hash;
  crystal &cr = *fd->cr;
  const void *const findptsData = fd;
  const int np = cr.comm.np, id = cr.comm.id;
  struct array hash_pt, srcPt_t, outPt_t;
  /* look locally first */
  if (npt) {
    findpts_local(code_base, el_base, r_base, dist2_base, x_base, npt, findptsData);
  }
  /* send unfound and border points to global hash cells */
  {
    int index;
    int *code = code_base, *proc = proc_base;
    const double *xp[D];
    struct srcPt_t *pt;

    for (int d = 0; d < D; ++d) {
      xp[d] = x_base[d];
    }
    array_init(struct srcPt_t, &hash_pt, npt);
    pt = (struct srcPt_t *)hash_pt.ptr;

    double x[D];

    for (index = 0; index < npt; ++index) {
      for (int d = 0; d < D; ++d) {
        x[d] = *xp[d];
      }
      *proc = id;
      if (*code != CODE_INTERNAL) {
        const int hi = hash_index_3(&hash, x);
        for (int d = 0; d < D; ++d) {
          pt->x[d] = x[d];
        }
        pt->index = index;
        pt->proc = hi % np;
        ++pt;
      }
      for (int d = 0; d < D; ++d) {
        xp[d]++;
      }
      code++;
      proc++;
    }
    hash_pt.n = pt - (struct srcPt_t *)hash_pt.ptr;
    sarray_transfer(struct srcPt_t, &hash_pt, proc, 1, &cr);
  }
  /* look up points in hash cells, route to possible procs */
  {
    const unsigned int *const hash_offset = hash.offset;
    int count = 0, *proc, *proc_p;
    const struct srcPt_t *p = (struct srcPt_t *)hash_pt.ptr, *const pe = p + hash_pt.n;
    struct srcPt_t *q;
    for (; p != pe; ++p) {
      const int hi = hash_index_3(&hash, p->x) / np;
      const int i = hash_offset[hi], ie = hash_offset[hi + 1];
      count += ie - i;
    }
    proc = tmalloc(int, count);
    proc_p = proc;
    array_init(struct srcPt_t, &srcPt_t, count), q = (struct srcPt_t *)srcPt_t.ptr;
    p = (struct srcPt_t *)hash_pt.ptr;
    for (; p != pe; ++p) {
      const int hi = hash_index_3(&hash, p->x) / np;
      int i = hash_offset[hi];
      const int ie = hash_offset[hi + 1];
      for (; i != ie; ++i) {
        const int pp = hash_offset[i];
        if (pp == p->proc)
          continue; /* don't send back to source proc */
        *proc_p++ = pp;
        *q++ = *p;
      }
    }
    array_free(&hash_pt);
    srcPt_t.n = proc_p - proc;
#ifdef DIAGNOSTICS
    printf("(proc %u) hashed; routing %u/%u\n", id, (int)srcPt_t.n, count);
#endif
    sarray_transfer_ext(struct srcPt_t, &srcPt_t, reinterpret_cast<unsigned int *>(proc), sizeof(int), &cr);
    free(proc);
  }
  /* look for other procs' points, send back */
  {
    int n = srcPt_t.n;
    const struct srcPt_t *spt;
    struct outPt_t *opt;
    array_init(struct outPt_t, &outPt_t, n);
    outPt_t.n = n;
    spt = (struct srcPt_t *)srcPt_t.ptr;
    opt = (struct outPt_t *)outPt_t.ptr;
    for (; n; --n, ++spt, ++opt) {
      opt->index = spt->index;
      opt->proc = spt->proc;
    }
    spt = (struct srcPt_t *)srcPt_t.ptr;
    opt = (struct outPt_t *)outPt_t.ptr;
    if (srcPt_t.n) {

      // result buffers
      std::vector<int> codeArr(srcPt_t.n, 0);
      std::vector<int> elArr(srcPt_t.n, 0);
      std::vector<double> rArr(3 * srcPt_t.n, 0.0);
      std::vector<double> dist2Arr(srcPt_t.n, 0.0);

      std::vector<double> x0(srcPt_t.n, 0.0);
      std::vector<double> x1(srcPt_t.n, 0.0);
      std::vector<double> x2(srcPt_t.n, 0.0);

      // pack position data into arrays
      double *spt_x_base[3] = {x0.data(), x1.data(), x2.data()};

      for (int point = 0; point < srcPt_t.n; ++point) {
        for (int d = 0; d < 3; ++d) {
          spt_x_base[d][point] = spt[point].x[d];
        }
      }

      findpts_local(codeArr.data(),
                    elArr.data(),
                    rArr.data(),
                    dist2Arr.data(),
                    spt_x_base,
                    srcPt_t.n,
                    findptsData);

      // unpack arrays into opt
      for (int point = 0; point < srcPt_t.n; point++) {
        opt[point].code = codeArr[point];
        opt[point].el = elArr[point];
        opt[point].dist2 = dist2Arr[point];
        for (int d = 0; d < D; ++d) {
          opt[point].r[d] = rArr[3 * point + d];
        }
      }
    }
    array_free(&srcPt_t);
    /* group by code to eliminate unfound points */
    sarray_sort(struct outPt_t, opt, outPt_t.n, code, 0, &cr.data);
    n = outPt_t.n;
    while (n && opt[n - 1].code == CODE_NOT_FOUND)
      --n;
    outPt_t.n = n;
#ifdef DIAGNOSTICS
    printf("(proc %u) sending back %u found points\n", id, (int)outPt_t.n);
#endif
    sarray_transfer(struct outPt_t, &outPt_t, proc, 1, &cr);
  }
  /* merge remote results with user data */
  {
    int n = outPt_t.n;
    struct outPt_t *opt = (struct outPt_t *)outPt_t.ptr;
    for (; n; --n, ++opt) {
      const int index = opt->index;
      if (code_base[index] == CODE_INTERNAL)
        continue;
      if (code_base[index] == CODE_NOT_FOUND || opt->code == CODE_INTERNAL ||
          opt->dist2 < dist2_base[index]) {
        for (int d = 0; d < D; ++d) {
          r_base[3 * index + d] = opt->r[d];
        }
        dist2_base[index] = opt->dist2;
        proc_base[index] = opt->proc;
        el_base[index] = opt->el;
        code_base[index] = opt->code;
      }
    }
    array_free(&outPt_t);
  }
}

void findptsEval(const dlong npt,
                 occa::memory o_in,
                 findpts_t *fd,
                 findpts_data_t *findPtsData,
                 dfloat *out_base)
{

  findpts_eval_impl(out_base,
                    findPtsData->code_base,
                    findPtsData->proc_base,
                    findPtsData->el_base,
                    findPtsData->r_base,
                    npt,
                    1,
                    npt,
                    npt,
                    &o_in,
                    *fd->hash,
                    *fd->cr,
                    fd);
}

void findptsEval(const dlong npt,
                 const dlong nFields,
                 const dlong inputOffset,
                 const dlong outputOffset,
                 occa::memory o_in,
                 findpts_t *fd,
                 findpts_data_t *findPtsData,
                 dfloat *out_base)
{
  if (nFields == 1) {
    findpts_eval_impl(out_base,
                      findPtsData->code_base,
                      findPtsData->proc_base,
                      findPtsData->el_base,
                      findPtsData->r_base,
                      npt,
                      nFields,
                      inputOffset,
                      npt,
                      &o_in,
                      *fd->hash,
                      *fd->cr,
                      fd);
  }
  else if (nFields == 3) {
    findpts_eval_impl<evalOutPt_t<3>>(out_base,
                                      findPtsData->code_base,
                                      findPtsData->proc_base,
                                      findPtsData->el_base,
                                      findPtsData->r_base,
                                      npt,
                                      nFields,
                                      inputOffset,
                                      npt,
                                      &o_in,
                                      *fd->hash,
                                      *fd->cr,
                                      fd);
  }

  // TODO: add other sizes, or correctly error
}

void findptsLocalEval(const dlong npt,
                      occa::memory o_in,
                      occa::memory o_el,
                      occa::memory o_r,
                      findpts_t *fd,
                      occa::memory o_out)
{

  if (npt == 0)
    return;
  fd->local_eval_kernel(npt, 1, 0, 0, o_el, o_r, o_in, o_out);
}

void findptsLocalEval(const dlong npt,
                      const dlong nFields,
                      const dlong inputOffset,
                      const dlong outputOffset,
                      occa::memory o_in,
                      occa::memory o_el,
                      occa::memory o_r,
                      findpts_t *fd,
                      occa::memory o_out)
{
  if (npt == 0)
    return;
  fd->local_eval_kernel(npt, nFields, inputOffset, npt, o_el, o_r, o_in, o_out);
}

crystal *crystalRouter(findpts_t *const fd) { return fd->cr; }
