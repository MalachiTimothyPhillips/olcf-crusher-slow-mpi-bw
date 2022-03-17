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

static occa::memory findptsCopyData_3(const struct gslibFindptsData_t *fd,
                                         dlong nel, dlong max_hash_size,
                                         occa::device device)
{
  const findpts_local_data_3 *fd_local = &fd->local;

  //create device allocation
  dlong lag_size[3];
  for(dlong d=0;d<3;++d) lag_size[d] = gll_lag_size(fd_local->fed.n[d]);
  auto hash_data_copy = fd_local->hd;
  dlong hd_d_size  = findpts_local_hash_opt_size_3(&hash_data_copy, fd_local->obb, nel, max_hash_size);
  dlong elx_d_size = nel*fd_local->ntot;
  dlong alloc_size =  sizeof(findpts_local_data_3)
                    + sizeof(double)*elx_d_size*3 // elx
                    + sizeof(obbox_3)*nel
                    + sizeof(uint)*hd_d_size // hd.offset
                    + sizeof(double)*(fd_local->fed.n[0]+fd_local->fed.n[1]+fd_local->fed.n[2]) // fed.z
                    + sizeof(double)*(lag_size[0]+lag_size[1]+lag_size[2]) // fed.lag_data
                    + sizeof(double)*(6*fd_local->fed.n[0]+6*fd_local->fed.n[1]+6*fd_local->fed.n[2]) // fed.wtend
                    + 8*6; // room for alignment padding
                    // other structures are unused
  occa::memory o_dev_copy = device.malloc(alloc_size, occa::dtype::byte);
  char *o_dev_copy_ptr = o_dev_copy.ptr<char>();

  // create and fill host buffer
  char *host_copy = new char[alloc_size];
  struct findpts_local_data_3 *fd_local_copy = (struct findpts_local_data_3*)host_copy;
  *fd_local_copy = *fd_local;
  dlong working_offset = sizeof(findpts_local_data_3);

  // ensures all fields are aligned to 8 bytes
  #define SET_FIELD(field, type, size) do {\
        dlong align = 8; \
        working_offset = ((working_offset + align - 1)/align)*align; \
        fd_local_copy->field = (type*)(o_dev_copy_ptr+working_offset); \
        memcpy(host_copy+working_offset, fd_local->field, size*sizeof(type)); \
        working_offset += size*sizeof(type); \
      } while(0)

  SET_FIELD(elx[0], double, elx_d_size);
  SET_FIELD(elx[1], double, elx_d_size);
  SET_FIELD(elx[2], double, elx_d_size);
  SET_FIELD(obb, struct obbox_3, nel);
  SET_FIELD(hd.offset, uint, hd_d_size);
  SET_FIELD(fed.z[0], double, fd_local->fed.n[0]);
  SET_FIELD(fed.z[1], double, fd_local->fed.n[1]);
  SET_FIELD(fed.z[2], double, fd_local->fed.n[2]);
  SET_FIELD(fed.lag_data[0], double, lag_size[0]);
  SET_FIELD(fed.lag_data[1], double, lag_size[1]);
  SET_FIELD(fed.lag_data[2], double, lag_size[2]);
  SET_FIELD(fed.wtend[0], double, 6*fd_local->fed.n[0]);
  SET_FIELD(fed.wtend[1], double, 6*fd_local->fed.n[1]);
  SET_FIELD(fed.wtend[2], double, 6*fd_local->fed.n[2]);

  #undef SET_FIELD

  // copy buffer to device
  o_dev_copy.copyFrom(host_copy);
  delete[] host_copy;
  return o_dev_copy;
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
  handle->findpts_data = findpts_data;

  if(x != nullptr){
    handle->o_x = device.malloc(Nlocal * sizeof(dfloat));
    handle->o_y = device.malloc(Nlocal * sizeof(dfloat));
    handle->o_z = device.malloc(Nlocal * sizeof(dfloat));

    handle->o_x.copyFrom(x, Nlocal * sizeof(dfloat));
    handle->o_y.copyFrom(y, Nlocal * sizeof(dfloat));
    handle->o_z.copyFrom(z, Nlocal * sizeof(dfloat));
  }

  handle->device = device;
  auto kernels = initFindptsKernels(comm, device, 3, Nq);
  handle->local_eval_kernel = kernels.at(0);
  handle->local_eval_many_kernel = kernels.at(1);
  handle->local_kernel = kernels.at(2);

  // Need to copy findpts data to the
  handle->o_fd_local = findptsCopyData_3(handle->findpts_data,
                                                Nelements,
                                                local_hash_size,
                                                device);

  return handle;
}

void findptsFree(findpts_t *fd)
{
  // Use OCCA's reference counting to free memory and kernel objects
  fd->local_eval_kernel = occa::kernel();
  fd->local_kernel = occa::kernel();
  fd->o_fd_local = occa::memory();
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
                  fd->findpts_data,
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
                      fd->findpts_data,
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
                        fd->findpts_data,
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
                        fd->findpts_data,
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

crystal *crystalRouter(findpts_t *const fd) { return &fd->findpts_data->cr; }
