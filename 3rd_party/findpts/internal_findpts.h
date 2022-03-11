#ifndef INTERNAL_FINDPTS_H
#define INTERNAL_FINDPTS_H

#include "findptsTypes.h"

#if !defined(MEM_H) || !defined(FINDPTS_H) || !defined(FINDPTS_LOCAL_H) || !defined(FINDPTS_EL_H) || !defined(OBBOX_H)
#warning "internal_findpts.h" requires "mem.h", "findpts.h", "findpts_local.h", "findpts_el.h", "obbox.h"
#endif

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
                 const int npt, struct findpts_data_3 *const fd,
                 const void *const findptsData);

#if 0
struct eval_src_pt_3 {
  double r[3];
  int index, proc, el;
};
struct eval_out_pt_3 {
  double out;
  int index, proc;
};
#endif

void findpts_local_eval_internal(
  struct eval_out_pt_3 *opt, const struct eval_src_pt_3 *spt,
  const int pn, const void *const in,
  struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void);

void findpts_local_eval(
        void *const  out_base,
  const void *const   el_base,
  const void *const    r_base,
  const int pn, const void *const in,
  struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void);

void findpts_eval_impl(
        double *const  out_base,
  const int   *const code_base,
  const int   *const proc_base,
  const int   *const   el_base,
  const double *const    r_base,
  const int npt,
  const void *const in, struct findpts_data_3 *const fd,
  const void *const findptsData);

#endif
