#include <stdlib.h>
#include <math.h>
#include "gslib.h"

#include "findptsTypes.h"
#include "findptsImpl.hpp"

#include <vector>

#define D 3

#define CODE_INTERNAL 0
#define CODE_BORDER 1
#define CODE_NOT_FOUND 2

template <typename OutputType>
void findpts_eval_impl(double *const out_base,
                       const int *const code_base,
                       const int *const proc_base,
                       const int *const el_base,
                       const double *const r_base,
                       const int npt,
                       const int nFields,
                       const int inputOffset,
                       const int outputOffset,
                       const void *const in,
                       hashData_t &hash,
                       crystal &cr,
                       const void *const findptsData)
{
  struct array src, outpt;
  /* copy user data, weed out unfound points, send out */
  {
    int index;
    const int *code = code_base, *proc = proc_base, *el = el_base;
    const double *r = r_base;
    evalSrcPt_t *pt;
    array_init(evalSrcPt_t, &src, npt);
    pt = (evalSrcPt_t *)src.ptr;
    for (index = 0; index < npt; ++index) {
      if (*code != CODE_NOT_FOUND) {
        for (int d = 0; d < D; ++d) {
          pt->r[d] = r[d];
        }
        pt->index = index;
        pt->proc = *proc;
        pt->el = *el;
        ++pt;
      }
      r += D;
      code++;
      proc++;
      el++;
    }
    src.n = pt - (evalSrcPt_t *)src.ptr;
    sarray_transfer(evalSrcPt_t, &src, proc, 1, &cr);
  }
  /* evaluate points, send back */
  {
    int n = src.n;
    const evalSrcPt_t *spt;
    OutputType *opt;
    /* group points by element */
    sarray_sort(evalSrcPt_t, src.ptr, n, el, 0, &cr.data);
    array_init(OutputType, &outpt, n);
    outpt.n = n;
    spt = (evalSrcPt_t *)src.ptr;
    opt = (OutputType *)outpt.ptr;
    findpts_local_eval_internal(opt, spt, src.n, nFields, inputOffset, outputOffset, in, findptsData);
    spt = (evalSrcPt_t *)src.ptr;
    opt = (OutputType *)outpt.ptr;
    for (; n; --n, ++spt, ++opt) {
      opt->index = spt->index;
      opt->proc = spt->proc;
    }
    array_free(&src);
    sarray_transfer(OutputType, &outpt, proc, 1, &cr);
  }
  /* copy results to user data */
  {
    int n = outpt.n;
    OutputType *opt = (OutputType *)outpt.ptr;
    for (; n; --n, ++opt) {
      for (int field = 0; field < nFields; ++field) {
        out_base[opt->index + field * npt] = opt->out[field];
      }
    }
    array_free(&outpt);
  }
}

// explicit instantiation
template void findpts_eval_impl<evalOutPt_t<1>>(double *const out_base,
                                                const int *const code_base,
                                                const int *const proc_base,
                                                const int *const el_base,
                                                const double *const r_base,
                                                const int npt,
                                                const int nFields,
                                                const int inputOffset,
                                                const int outputOffset,
                                                const void *const in,
                                                hashData_t &hash,
                                                crystal &cr,
                                                const void *const findptsData);
template void findpts_eval_impl<evalOutPt_t<3>>(double *const out_base,
                                                const int *const code_base,
                                                const int *const proc_base,
                                                const int *const el_base,
                                                const double *const r_base,
                                                const int npt,
                                                const int nFields,
                                                const int inputOffset,
                                                const int outputOffset,
                                                const void *const in,
                                                hashData_t &hash,
                                                crystal &cr,
                                                const void *const findptsData);