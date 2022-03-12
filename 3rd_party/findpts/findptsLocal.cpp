
#include <type_traits>
#include "ogstypes.h"
#include "findpts.hpp"
#include "platform.hpp"
#include <vector>

#include "gslib.h"
#include "findptsImpl.hpp"

static occa::memory o_scratch;
static occa::memory h_scratch;
static void* scratch;
static void realloc_scratch(occa::device& device, dlong Nbytes){
  if(o_scratch.size()) o_scratch.free();
  if(h_scratch.size()) h_scratch.free();

  {
    occa::properties props;
    props["host"] = true;
  
    void* buffer = std::calloc(Nbytes, 1);
    occa::memory h_scratch = device.malloc(Nbytes, buffer, props);
    std::free(buffer);
    
    scratch = h_scratch.ptr();
  }

  {
    o_scratch = device.malloc(Nbytes);
  }
}

static_assert(std::is_same<dfloat, double>::value, "findpts dfloat is not compatible with GSLIB double");
static_assert(sizeof(dlong) == sizeof(int), "findpts dlong is not compatible with GSLIB int");

// Can't use pinned memory for D->H transfer without
// doing H->H transfer to input arrays
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
  if(alloc_size > o_scratch.size()){
    realloc_scratch(device, alloc_size);
  }

  dlong byteOffset = 0;

  occa::memory  o_code_base = o_scratch + byteOffset; byteOffset +=   sizeof(dlong) *pn;
  occa::memory    o_el_base = o_scratch + byteOffset; byteOffset +=   sizeof(dlong) *pn;
  occa::memory     o_r_base = o_scratch + byteOffset; byteOffset += 3*sizeof(dfloat)*pn;
  occa::memory o_dist2_base = o_scratch + byteOffset; byteOffset +=   sizeof(dfloat)*pn;
  occa::memory     o_x_base = o_scratch + byteOffset; byteOffset += 3*sizeof(dfloat*);
  occa::memory    o_x0_base = o_scratch + byteOffset; byteOffset +=  sizeof(dfloat)*pn;
  occa::memory    o_x1_base = o_scratch + byteOffset; byteOffset +=  sizeof(dfloat)*pn;
  occa::memory    o_x2_base = o_scratch + byteOffset; byteOffset +=  sizeof(dfloat)*pn;

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
    struct evalOutPt_t *opt, const struct evalSrcPt_t *spt,
    const int pn, const void *const in,
    struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void)
{
  if (pn == 0) return;

  findpts_t *findptsData = (findpts_t*)findptsData_void;
  occa::device device = *findptsData->device;
  occa::memory o_in = *(occa::memory*)in;

  const auto Nbytes = 4 * pn * sizeof(dfloat) + pn * sizeof(dlong);
  if(Nbytes > o_scratch.size()){
    realloc_scratch(device, Nbytes);
  }

  dlong byteOffset = 0;

  dfloat* out = (dfloat*) (static_cast<char*>(scratch) + byteOffset);
  auto o_out = o_scratch + byteOffset;
  byteOffset += pn * sizeof(dfloat);

  dfloat* r = (dfloat*) (static_cast<char*>(scratch) + byteOffset);
  auto o_r = o_scratch + byteOffset;
  byteOffset += 3 * pn * sizeof(dfloat);

  dlong* el = (dlong*) (static_cast<char*>(scratch) + byteOffset);
  auto o_el = o_scratch + byteOffset;
  byteOffset += pn * sizeof(dlong);

  // pack host buffers
  for(int point = 0; point < pn; ++point){
    for(int component = 0; component < 3; ++component){
      r[3*point+component] = spt[point].r[component];
      el[point] = spt[point].el;
    }
  }

  o_r.copyFrom(r, 3 * pn * sizeof(dfloat));
  o_el.copyFrom(el, pn * sizeof(dlong));

  findptsData->local_eval_kernel(pn, o_el, o_r, o_in, o_out);

  o_out.copyTo(out, pn * sizeof(dfloat));

  // unpack buffer
  for(int point = 0; point < pn; ++point){
    opt[point].out = out[point];
  }

  o_out.free();
  o_el.free();
  o_r.free();

}