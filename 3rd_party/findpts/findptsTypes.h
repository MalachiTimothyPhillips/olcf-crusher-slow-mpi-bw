#ifndef FINDPTS_TYPES_H
#define FINDPTS_TYPES_H

#if !defined(FINDPTS_LOCAL_H)
#warning "findptsTypes.h" requires "findpts_local.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

//struct dbl_range{
//  double min, max;
//};

struct evalSrcPt_t {
  double r[3];
  int index, proc, el;
};
struct evalOutPt_t {
  double out;
  int index, proc;
};

struct hashData_t {
  ulong hash_n;
  struct dbl_range bnd[3];
  double fac[3];
  uint *offset;
};
struct findpts_data_3 {
  struct crystal cr;
  struct findpts_local_data_3 local;
  struct hashData_t hash;
};
#ifdef __cplusplus
}
#endif
#endif