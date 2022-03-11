
#ifndef INTERFACE_FINDPTS_H
#define INTERFACE_FINDPTS_H

#include "ogstypes.h"

extern "C" {
struct findpts_data_3;

struct findpts_data_3 *legacyFindptsSetup(MPI_Comm comm,
                                             const dfloat *const elx[3],
                                             const dlong n[3],
                                             const dlong nel,
                                             const dlong m[3],
                                             const dfloat bbox_tol,
                                             const hlong local_hash_size,
                                             const hlong global_hash_size,
                                             const dlong npt_max,
                                             const dfloat newt_tol);

void legacyFindptsFree(struct findpts_data_3 *fd);

void legacyFindptsLagData(struct findpts_data_3 *const fd, dfloat **lag_data, dlong *lag_data_size);

void legacyFindpts(dlong *const code_base,
                      const dlong code_stride,
                      dlong *const proc_base,
                      const dlong proc_stride,
                      dlong *const el_base,
                      const dlong el_stride,
                      dfloat *const r_base,
                      const dlong r_stride,
                      dfloat *const dist2_base,
                      const dlong dist2_stride,
                      const dfloat *const x_base[3],
                      const dlong x_stride[3],
                      const dfloat npt,
                      struct findpts_data_3 *const fd);

void legacyFindptsEval(dfloat *const out_base,
                          const dlong out_stride,
                          const dlong *const code_base,
                          const dlong code_stride,
                          const dlong *const proc_base,
                          const dlong proc_stride,
                          const dlong *const el_base,
                          const dlong el_stride,
                          const dfloat *const r_base,
                          const dlong r_stride,
                          const dlong npt,
                          const dfloat *const in,
                          struct findpts_data_3 *const fd);

void devFindpts(dlong *const code_base,
                     dlong *const proc_base,
                     dlong *const el_base,
                     dfloat *const r_base,
                     dfloat *const dist2_base,
                     const dfloat *const x_base[3],
                     const dlong npt,
                     struct findpts_data_3 *const fd,
                     const void *const findptsData);

void devFindptsEval(dfloat *const out_base,
                         const dlong *const code_base,
                         const dlong *const proc_base,
                         const dlong *const el_base,
                         const dfloat *const r_base,
                         const dlong npt,
                         void *const in,
                         struct findpts_data_3 *const fd,
                         const void *const findptsData);

void legacyFindptsLocalEval(dfloat *const out_base,
                               const dlong out_stride,
                               const dlong *const el_base,
                               const dlong el_stride,
                               const dfloat *const r_base,
                               const dlong r_stride,
                               const dlong npt,
                               const dfloat *const in,
                               struct findpts_data_3 *const fd);

void devFindptsLocalEval(void *const out,
                              const void *const el,
                              const void *const r,
                              const dlong npt,
                              void *const in,
                              struct findpts_data_3 *const fd,
                              const void *const findptsData);
}

#endif
