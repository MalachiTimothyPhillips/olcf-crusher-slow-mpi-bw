#ifndef FINDPTS_IMPL_HPP
#define FINDPTS_IMPL_HPP

#include "findptsTypes.h"

void findpts_local(int *const code_base,
                         int *const el_base,
                         double *const r_base,
                         double *const dist2_base,
                         const double *const x_base[3],
                         const int npt,
                         const void *const findptsData);
void findpts_impl(    int   *const  code_base,
                       int   *const  proc_base,
                       int   *const    el_base,
                       double *const     r_base,
                       double *const dist2_base,
                 const double *const     x_base[3],
                 const int npt, struct gslibFindptsData_t *const fd,
                 const void *const findptsData);

void findpts_local_eval_internal(
  struct evalOutPt_t *opt, const struct evalSrcPt_t *spt,
  const int pn, const void *const in,
  struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void);

void findpts_eval_impl(
        double *const  out_base,
  const int   *const code_base,
  const int   *const proc_base,
  const int   *const   el_base,
  const double *const    r_base,
  const int npt,
  const void *const in, struct gslibFindptsData_t *const fd,
  const void *const findptsData);

#endif