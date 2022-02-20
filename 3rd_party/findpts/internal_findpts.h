#ifndef INTERNAL_FINDPTS_H
#define INTERNAL_FINDPTS_H

#if !defined(MEM_H) || !defined(FINDPTS_H) || !defined(FINDPTS_LOCAL_H) || !defined(FINDPTS_EL_H) || !defined(OBBOX_H)
#warning "findpts.h" requires "mem.h", "findpts.h", "findpts_local.h", "findpts_el.h", "obbox.h"
#endif

void findpts_local(uint *const code_base,
                         const unsigned code_stride,
                         uint *const el_base,
                         const unsigned el_stride,
                         double *const r_base,
                         const unsigned r_stride,
                         double *const dist2_base,
                         const unsigned dist2_stride,
                         const double *const x_base[3],
                         const unsigned x_stride[3],
                         const uint npt,
                         const void *const findptsData);
void findpts_impl(    uint   *const  code_base   , const unsigned  code_stride   ,
                       uint   *const  proc_base   , const unsigned  proc_stride   ,
                       uint   *const    el_base   , const unsigned    el_stride   ,
                       double *const     r_base   , const unsigned     r_stride   ,
                       double *const dist2_base   , const unsigned dist2_stride   ,
                 const double *const     x_base[3], const unsigned     x_stride[3],
                 const uint npt, struct findpts_data_3 *const fd,
                 const void *const findptsData);

struct eval_src_pt_3 {
  double r[3];
  uint index, proc, el;
};
struct eval_out_pt_3 {
  double out;
  uint index, proc;
};

void findpts_local_eval_internal(
  struct eval_out_pt_3 *opt, const struct eval_src_pt_3 *spt,
  const uint pn, const void *const in,
  struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void);

void findpts_local_eval(
        void *const  out_base, const unsigned  out_stride,
  const void *const   el_base, const unsigned   el_stride,
  const void *const    r_base, const unsigned    r_stride,
  const uint pn, const void *const in,
  struct findpts_local_data_3 *const gs_fd, const void *const findptsData_void);

void findpts_eval_impl(
        double *const  out_base, const unsigned  out_stride,
  const uint   *const code_base, const unsigned code_stride,
  const uint   *const proc_base, const unsigned proc_stride,
  const uint   *const   el_base, const unsigned   el_stride,
  const double *const    r_base, const unsigned    r_stride,
  const uint npt,
  const void *const in, struct findpts_data_3 *const fd,
  const void *const findptsData);

#endif
