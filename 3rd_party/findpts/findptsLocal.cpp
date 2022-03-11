
#include <type_traits>
#include "ogstypes.h"
#include "findpts.hpp"
#include "platform.hpp"
#include <vector>

extern "C" {

#include "gslib.h"
#include "internal_findpts.h"

static_assert(std::is_same<dfloat, double>::value, "findpts dfloat is not compatible with GSLIB double");
static_assert(sizeof(dlong) == sizeof(int), "findpts dlong is not compatible with GSLIB int");

void findpts_local(    int   *const  code_base,
                             int   *const    el_base,
                             double *const     r_base   ,
                             double *const dist2_base   ,
                       const double *const     x_base[3],
                       const int pn, const void *const findptsData_void)
{
  if (pn == 0) return;

  findpts_t *findptsData = (findpts_t*)findptsData_void;
  occa::device device = *findptsData->device;

  dlong worksize = 2*sizeof(dlong)+7*sizeof(dfloat);
  dlong alloc_size = worksize*pn+3*(sizeof(dfloat*)+sizeof(dlong));
  occa::memory workspace;
  occa::memory mempool = platform_t::getInstance()->o_mempool.o_ptr;
  if(alloc_size < mempool.size()) {
    workspace = mempool.cast(occa::dtype::byte);
  } else {
    workspace = device.malloc(alloc_size, occa::dtype::byte);
  }
  occa::memory  o_code_base = workspace; workspace +=   sizeof(dlong) *pn;
  occa::memory    o_el_base = workspace; workspace +=   sizeof(dlong) *pn;
  occa::memory     o_r_base = workspace; workspace += 3*sizeof(dfloat)*pn;
  occa::memory o_dist2_base = workspace; workspace +=   sizeof(dfloat)*pn;
  occa::memory     o_x_base = workspace; workspace += 3*sizeof(dfloat*);
  occa::memory    o_x0_base = workspace; workspace +=  sizeof(dfloat)*pn;
  occa::memory    o_x1_base = workspace; workspace +=  sizeof(dfloat)*pn;
  occa::memory    o_x2_base = workspace; workspace +=  sizeof(dfloat)*pn;

  dfloat *x_base_d[3] = {(double*)o_x0_base.ptr(), (double*)o_x1_base.ptr(), (double*)o_x2_base.ptr()};
  o_x_base.copyFrom(x_base_d, 3*sizeof(dfloat*));
  o_x0_base.copyFrom(x_base[0], sizeof(dfloat)*pn);
  o_x1_base.copyFrom(x_base[1], sizeof(dfloat)*pn);
  o_x2_base.copyFrom(x_base[2], sizeof(dfloat)*pn);

  findptsData->local_kernel( o_code_base,
                          o_el_base,
                           o_r_base,
                       o_dist2_base,
                           o_x_base,
                       pn, findptsData->o_fd_local);

  o_code_base.copyTo( code_base, sizeof(dlong) *pn);
  o_el_base.copyTo(   el_base,   sizeof(dlong)*pn);
  o_r_base.copyTo(    r_base,     3*sizeof(dfloat)*pn);
  o_dist2_base.copyTo(dist2_base, sizeof(dfloat)*pn);
}

void findpts_local_eval_internal(
    struct eval_out_pt_3 *opt, const struct eval_src_pt_3 *spt,
    const int pn, const void *const in,
    struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void)
{
  if (pn == 0) return;

  findpts_t *findptsData = (findpts_t*)findptsData_void;
  occa::device device = *findptsData->device;
  occa::memory o_in = *(occa::memory*)in;

  // pack buffers
  std::vector<dfloat> out(pn, 0.0);
  std::vector<dfloat> r(3*pn, 0.0);
  std::vector<dlong> el(pn, 0);

  for(int point = 0; point < pn; ++point){
    for(int component = 0; component < 3; ++component){
      r[3*point+component] = spt[point].r[component];
      el[point] = spt[point].el;
    }
  }

  // TODO: use pinned memory buffers + common scratch space
  auto o_out = device.malloc(pn * sizeof(dfloat));
  auto o_el  = device.malloc(pn * sizeof(dlong));
  auto o_r   = device.malloc(3 * pn * sizeof(dfloat));

  o_r.copyFrom(r.data(), 3 * pn * sizeof(dfloat));
  o_el.copyFrom(el.data(), pn * sizeof(dlong));

  findptsData->local_eval_kernel(pn, o_el, o_r, o_in, o_out);

  o_out.copyTo(out.data(), pn * sizeof(dfloat));

  // unpack buffer
  for(int point = 0; point < pn; ++point){
    opt[point].out = out[point];
  }

  o_out.free();
  o_el.free();
  o_r.free();

}

void findpts_local_eval(
          void * const out,
    const void * const el,
    const void * const r,
    const int pn, const void *const in,
    struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void)
{
  if (pn == 0) return;

  findpts_t *findptsData = (findpts_t*)findptsData_void;
  occa::memory o_out = *(occa::memory*)out;
  occa::memory o_el  = *(occa::memory*) el;
  occa::memory o_r   = *(occa::memory*)  r;
  occa::memory o_in       = *(occa::memory*)in;

  findptsData->local_eval_kernel(pn, o_el, o_r, o_in, o_out);
}

}
