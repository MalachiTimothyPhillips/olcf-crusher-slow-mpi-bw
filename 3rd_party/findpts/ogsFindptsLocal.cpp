
#include <type_traits>
#include "ogstypes.h"
#include "ogsFindpts.hpp"
#include "platform.hpp"

extern "C" {

#include "gslib.h"
#include "ogsFindpts.h"

static_assert(std::is_same<dfloat, double>::value, "OGS dfloat is not compatible with GSLIB double");
static_assert(sizeof(dlong) == sizeof(uint), "OGS dlong is not compatible with GSLIB uint");

void ogs_findpts_local_3(    uint   *const  code_base   , const unsigned  code_stride   ,
                             uint   *const    el_base   , const unsigned    el_stride   ,
                             double *const     r_base   , const unsigned     r_stride   ,
                             double *const dist2_base   , const unsigned dist2_stride   ,
                       const double *const     x_base[3], const unsigned     x_stride[3],
                       const uint pn, const void *const ogs_fd_void)
{
  if (pn == 0) return;

  ogs_findpts_t *ogs_fd = (ogs_findpts_t*)ogs_fd_void;
  occa::device device = *ogs_fd->device;

  dlong worksize = code_stride+el_stride+r_stride+dist2_stride
                   +x_stride[0]+x_stride[1]+x_stride[2];
  dlong alloc_size = worksize*pn+3*(sizeof(dfloat*)+sizeof(dlong));
  occa::memory workspace;
  occa::memory mempool = platform_t::getInstance()->o_mempool.o_ptr;
  if(alloc_size < mempool.size()) {
    workspace = mempool.cast(occa::dtype::byte);
  } else {
    workspace = device.malloc(alloc_size, occa::dtype::byte);
  }
  occa::memory  d_code_base = workspace; workspace +=   sizeof(dlong) *pn;
  occa::memory    d_el_base = workspace; workspace +=   sizeof(dlong) *pn;
  occa::memory     d_r_base = workspace; workspace += 3*sizeof(dfloat)*pn;
  occa::memory d_dist2_base = workspace; workspace +=   sizeof(dfloat)*pn;
  occa::memory     d_x_base = workspace; workspace += 3*sizeof(dfloat*);
  occa::memory    d_x0_base = workspace; workspace +=  x_stride[0]*pn;
  occa::memory    d_x1_base = workspace; workspace +=  x_stride[1]*pn;
  occa::memory    d_x2_base = workspace; workspace +=  x_stride[2]*pn;
  occa::memory   d_x_stride = workspace; workspace += 3*sizeof(dlong);

  dfloat *x_base_d[3] = {(double*)d_x0_base.ptr(), (double*)d_x1_base.ptr(), (double*)d_x2_base.ptr()};
  d_x_base.copyFrom(x_base_d, 3*sizeof(dfloat*));
  d_x0_base.copyFrom(x_base[0], x_stride[0]*pn);
  d_x1_base.copyFrom(x_base[1], x_stride[1]*pn);
  d_x2_base.copyFrom(x_base[2], x_stride[2]*pn);
  d_x_stride.copyFrom(x_stride, 3*sizeof(dlong));

  ogs_fd->local_kernel( d_code_base, (dlong)sizeof(dlong),
                          d_el_base, (dlong)sizeof(dlong),
                           d_r_base, (dlong)(3*sizeof(dfloat)),
                       d_dist2_base, (dlong)sizeof(dfloat),
                           d_x_base, d_x_stride,
                       pn, ogs_fd->o_fd_local);

  d_code_base.copyTo(code_base, sizeof(dlong) * pn);
  d_el_base.copyTo(el_base, el_stride * pn);
  d_r_base.copyTo(r_base, r_stride * pn);
  d_dist2_base.copyTo(dist2_base, dist2_stride * pn);
}

void ogs_findpts_local_eval_internal_3(
    struct eval_out_pt_3 *opt, const struct eval_src_pt_3 *spt,
    const uint pn, const void *const in,
    struct findpts_local_data_3 *const gs_fd, const void *const ogs_fd_void)
{
  if (pn == 0) return;

  ogs_findpts_t *ogs_fd = (ogs_findpts_t*)ogs_fd_void;
  occa::device device = *ogs_fd->device;
  occa::memory d_in = *(occa::memory*)in;

  dlong alloc_size = (sizeof(struct eval_out_pt_3)+sizeof(struct eval_src_pt_3))*pn;
  occa::memory workspace;
  occa::memory mempool = platform_t::getInstance()->o_mempool.o_ptr;
  if(alloc_size < mempool.size()) {
    workspace = mempool.cast(occa::dtype::byte);
  } else {
    workspace = device.malloc(alloc_size, occa::dtype::byte);
  }
  occa::memory d_out_pt = workspace; workspace += sizeof(struct eval_out_pt_3)*pn;
  occa::memory d_src_pt = workspace;
  d_src_pt.copyFrom(spt, sizeof(struct eval_src_pt_3)*pn);

  occa::memory d_out_base   = d_out_pt + offsetof(struct eval_out_pt_3, out);
  occa::memory d_el_base    = d_src_pt + offsetof(struct eval_src_pt_3, el);
  occa::memory d_r_base     = d_src_pt + offsetof(struct eval_src_pt_3, r);

  dlong out_stride = sizeof(struct eval_out_pt_3);
  dlong src_stride = sizeof(struct eval_src_pt_3);

  ogs_fd->local_eval_kernel(d_out_base,   out_stride,
                            d_el_base,    src_stride,
                            d_r_base,     src_stride,
                            pn, d_in, gs_fd->ntot,
                            ogs_fd->o_fd_local);

  d_out_pt.copyTo(opt, sizeof(struct eval_out_pt_3)*pn);
}

void ogs_findpts_local_eval_3(
          void * const out_base, const uint out_stride,
    const void * const el_base,  const uint el_stride,
    const void * const r_base,   const uint r_stride,
    const uint pn, const void *const in,
    struct findpts_local_data_3 *const gs_fd, const void *const ogs_fd_void)
{
  if (pn == 0) return;

  ogs_findpts_t *ogs_fd = (ogs_findpts_t*)ogs_fd_void;
  occa::memory d_out_base = *(occa::memory*)out_base;
  occa::memory d_el_base  = *(occa::memory*) el_base;
  occa::memory d_r_base   = *(occa::memory*)  r_base;
  occa::memory d_in       = *(occa::memory*)in;

  ogs_fd->local_eval_kernel(d_out_base, out_stride,
                            d_el_base,  el_stride,
                            d_r_base,   r_stride,
                            pn, d_in, gs_fd->ntot,
                            ogs_fd->o_fd_local);

}

}
