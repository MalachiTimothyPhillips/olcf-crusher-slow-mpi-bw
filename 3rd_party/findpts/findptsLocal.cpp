
#include <type_traits>
#include "ogstypes.h"
#include "findpts.hpp"
#include "platform.hpp"
#include <vector>

#include "gslib.h"
#include "findptsImpl.hpp"

static occa::memory o_scratch;
static occa::memory h_out;
static occa::memory h_r;
static occa::memory h_el;
static dfloat* out;
static dfloat* r;
static dlong* el;

// findpts, do not allocate pinned memory
static void realloc_scratch(occa::device& device, dlong Nbytes){
  if(o_scratch.size()) o_scratch.free();
  {
    void* buffer = std::calloc(Nbytes, 1);
    o_scratch = device.malloc(Nbytes, buffer);
    std::free(buffer);
  }
}

// findpts_eval
static void realloc_scratch(occa::device& device, dlong pn, dlong nFields){

  const auto Nbytes = (3 * pn + nFields * pn) * sizeof(dfloat) + pn * sizeof(dlong);

  if(h_out.size()) h_out.free();
  if(h_r.size()) h_r.free();
  if(h_el.size()) h_el.free();

  if(Nbytes > o_scratch.size())
  {
    if(o_scratch.size()) o_scratch.free();
    void* buffer = std::calloc(Nbytes, 1);
    o_scratch = device.malloc(Nbytes, buffer);
    std::free(buffer);
  }

  occa::properties props;
  props["host"] = true;

  {
    void* buffer = std::calloc(nFields * pn * sizeof(dfloat), 1);
    h_out = device.malloc(nFields * pn * sizeof(dfloat), buffer, props);
    out = (dfloat*) h_out.ptr();
    std::free(buffer);
  }

  {
    void* buffer = std::calloc(3 * pn * sizeof(dfloat), 1);
    h_r = device.malloc(3 * pn * sizeof(dfloat), buffer, props);
    r = (dfloat*) h_r.ptr();
    std::free(buffer);
  }

  {
    void* buffer = std::calloc(pn * sizeof(dlong), 1);
    h_el = device.malloc(pn * sizeof(dlong), buffer, props);
    el = (dlong*) h_el.ptr();
    std::free(buffer);
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
                       pn, fd->local.tol, findptsData->o_fd_local);

  o_code_base.copyTo( code_base, sizeof(dlong) *pn);
  o_el_base.copyTo(   el_base,   sizeof(dlong)*pn);
  o_r_base.copyTo(    r_base,     3*sizeof(dfloat)*pn);
  o_dist2_base.copyTo(dist2_base, sizeof(dfloat)*pn);
}

template<typename OutputType>
void findpts_local_eval_internal(
    OutputType *opt, const evalSrcPt_t *spt,
    const int pn, const int nFields, const int inputOffset, const int outputOffset, const void *const in,
    const void *const findptsData_void)
{
  if (pn == 0) return;

  findpts_t *findptsData = (findpts_t*)findptsData_void;
  occa::device device = *findptsData->device;
  occa::memory o_in = *(occa::memory*)in;

  const auto Nbytes = (3 * pn + nFields * pn) * sizeof(dfloat) + pn * sizeof(dlong);
  if(Nbytes > o_scratch.size() || h_out.size() == 0){
    realloc_scratch(device, pn, nFields);
  }

  dlong byteOffset = 0;

  auto o_out = o_scratch;
  byteOffset += nFields * pn * sizeof(dfloat);

  auto o_r = o_scratch + byteOffset;
  byteOffset += 3 * pn * sizeof(dfloat);

  auto o_el = o_scratch + byteOffset;
  byteOffset += pn * sizeof(dlong);

  // pack host buffers
  for(int point = 0; point < pn; ++point){
    for(int component = 0; component < 3; ++component){
      r[3*point+component] = spt[point].r[component];
    }
    el[point] = spt[point].el;
  }

  o_r.copyFrom(r, 3 * pn * sizeof(dfloat));
  o_el.copyFrom(el, pn * sizeof(dlong));

  findptsData->local_eval_kernel(pn, nFields, inputOffset, pn, o_el, o_r, o_in, o_out);

  o_out.copyTo(out, nFields * pn * sizeof(dfloat));

  // unpack buffer
  for(int point = 0; point < pn; ++point){
    for(int field = 0; field < nFields; ++field){
      opt[point].out[field] = out[point + field * pn];
    }
  }

}

template
void findpts_local_eval_internal<evalOutPt_t<1>>(
  evalOutPt_t<1> *opt, const evalSrcPt_t *spt,
    const int pn, const int nFields, const int inputOffset, const int outputOffset, const void *const in,
  const void *const findptsData_void);

template
void findpts_local_eval_internal<evalOutPt_t<3>>(
  evalOutPt_t<3> *opt, const evalSrcPt_t *spt,
    const int pn, const int nFields, const int inputOffset, const int outputOffset, const void *const in,
  const void *const findptsData_void);