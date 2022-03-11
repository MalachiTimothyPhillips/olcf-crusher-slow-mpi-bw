#ifndef FINDPTS_TYPES_H
#define FINDPTS_TYPES_H
// Okay for both C/C++ files to look at this

#if !defined(MEM_H) || !defined(FINDPTS_H) || !defined(FINDPTS_LOCAL_H) || !defined(FINDPTS_EL_H) || !defined(OBBOX_H)
#warning "findptsTypes.h" requires "mem.h", "findpts.h", "findpts_local.h", "findpts_el.h", "obbox.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif
struct eval_src_pt_3 {
  double r[3];
  int index, proc, el;
};
struct eval_out_pt_3 {
  double out;
  int index, proc;
};

struct hash_data_3 {
  ulong hash_n;
  struct dbl_range bnd[3];
  double fac[3];
  uint *offset;
};
struct findpts_data_3 {
  struct crystal cr;
  struct findpts_local_data_3 local;
  struct hash_data_3 hash;
};
#ifdef __cplusplus
}
#endif
#endif