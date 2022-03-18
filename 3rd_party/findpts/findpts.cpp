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
                               const struct obbox_3 *const obb, const uint nel,
                               const uint max_size);
}

dlong getHashSize(const struct gslibFindptsData_t *fd,
                                         dlong nel, dlong max_hash_size)
{
  const findpts_local_data_3 *fd_local = &fd->local;
  auto hash_data_copy = fd_local->hd;
  return findpts_local_hash_opt_size_3(&hash_data_copy, fd_local->obb, nel, max_hash_size);
}

findpts_t *findptsSetup(
  MPI_Comm comm,
  const dfloat* const x,
  const dfloat* const y,
  const dfloat* const z,
  const dlong Nq,
  const dlong Nelements,
  const dlong m,
  const dfloat bbox_tol,
  const hlong local_hash_size, const hlong global_hash_size,
  const dlong npt_max, const dfloat newt_tol,
  occa::device &device){

  const dlong Nlocal = Nq * Nq * Nq * Nelements;

  const dfloat* elx[3] = {x,y,z};
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

  findpts_t* handle = new findpts_t();
  handle->D = 3;
  //handle->findpts_data = findpts_data;
  handle->tol = findpts_data->local.tol;
  handle->hash = &findpts_data->hash;
  handle->cr = &findpts_data->cr;

  if(x != nullptr){
    handle->o_x = device.malloc(Nlocal * sizeof(dfloat));
    handle->o_y = device.malloc(Nlocal * sizeof(dfloat));
    handle->o_z = device.malloc(Nlocal * sizeof(dfloat));

    handle->o_x.copyFrom(x, Nlocal * sizeof(dfloat));
    handle->o_y.copyFrom(y, Nlocal * sizeof(dfloat));
    handle->o_z.copyFrom(z, Nlocal * sizeof(dfloat));
    std::vector<dfloat> c(3*Nelements, 0.0);
    std::vector<dfloat> A(9*Nelements, 0.0);
    std::vector<dfloat> minBound(3*Nelements, 0.0);
    std::vector<dfloat> maxBound(3*Nelements, 0.0);

    for(int e = 0; e < Nelements; ++e){
      auto box = findpts_data->local.obb[e];

      c[3*e + 0] = box.c0[0];
      c[3*e + 1] = box.c0[1];
      c[3*e + 2] = box.c0[2];

      minBound[3*e + 0] = box.x[0].min;
      minBound[3*e + 1] = box.x[1].min;
      minBound[3*e + 2] = box.x[2].min;

      maxBound[3*e + 0] = box.x[0].max;
      maxBound[3*e + 1] = box.x[1].max;
      maxBound[3*e + 2] = box.x[2].max;

      for(int i = 0; i < 9; ++i){
        A[9*e + i] = box.A[i];
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
  for(int d = 0; d < 3; ++d){
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

  const auto hd_d_size = getHashSize(findpts_data,
                                                Nelements,
                                                local_hash_size);
  
  std::vector<dlong> offsets(hd_d_size, 0);
  for(dlong i = 0; i < hd_d_size; ++i){
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

void findpts(findpts_data_t *const findPtsData,
                const dfloat *const x_base[],
                const dlong npt,
                findpts_t *const fd)
{
  findpts_impl(findPtsData->code_base,
                  findPtsData->proc_base,
                  findPtsData->el_base,
                  findPtsData->r_base,
                  findPtsData->dist2_base,
                  x_base,
                  npt,
                  //fd->findpts_data,
                        *fd->hash,
                        *fd->cr,
                  fd);

}

void findptsEval(const dlong npt,
  occa::memory o_in,
  findpts_t* fd,
  findpts_data_t* findPtsData,
  dfloat * out_base)
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
                      //fd->findpts_data,
                        *fd->hash,
                        *fd->cr,
                      fd);
}

void findptsEval(const dlong npt,
  const dlong nFields,
  const dlong inputOffset,
  const dlong outputOffset,
  occa::memory o_in,
  findpts_t* fd,
  findpts_data_t* findPtsData,
  dfloat * out_base)
{
  if(nFields == 1){
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
                        //fd->findpts_data,
                        *fd->hash,
                        *fd->cr,
                        fd);
  } else if (nFields == 3){
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
                        //fd->findpts_data,
                        *fd->hash,
                        *fd->cr,
                        fd);
  }

  // TODO: add other sizes, or correctly error
}

void findptsLocalEval(
  const dlong npt,
  occa::memory o_in,
  occa::memory o_el,
  occa::memory o_r,
  findpts_t * fd,
  occa::memory o_out){

  if(npt == 0) return;
  fd->local_eval_kernel(npt, 1, 0, 0, o_el, o_r, o_in, o_out);
}

void findptsLocalEval(
  const dlong npt,
  const dlong nFields,
  const dlong inputOffset,
  const dlong outputOffset,
  occa::memory o_in,
  occa::memory o_el,
  occa::memory o_r,
  findpts_t * fd,
  occa::memory o_out){
  if(npt == 0) return;
  fd->local_eval_kernel(npt, nFields, inputOffset, npt, o_el, o_r, o_in, o_out);
}

crystal *crystalRouter(findpts_t *const fd) { return fd->cr; }
