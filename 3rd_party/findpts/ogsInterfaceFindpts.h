
#ifndef OGS_INTERFACE_FINDPTS_H
#define OGS_INTERFACE_FINDPTS_H

#include "ogsInterface.h"

extern "C" {
struct findpts_data_3;

struct findpts_data_3 *ogsLegacyFindptsSetup(MPI_Comm comm,
                                             const dfloat *const elx[3],
                                             const dlong n[3],
                                             const dlong nel,
                                             const dlong m[3],
                                             const dfloat bbox_tol,
                                             const hlong local_hash_size,
                                             const hlong global_hash_size,
                                             const dlong npt_max,
                                             const dfloat newt_tol);

void ogsLegacyFindptsFree(struct findpts_data_3 *fd);

void ogsLegacyFindptsLagData(struct findpts_data_3 *const fd, dfloat **lag_data, dlong *lag_data_size);

void ogsLegacyFindpts(dlong *const code_base,
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

void ogsLegacyFindptsEval(dfloat *const out_base,
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

void ogsDevFindpts(dlong *const code_base,
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
                     const dlong npt,
                     struct findpts_data_3 *const fd,
                     const void *const ogs_fd);

void ogsDevFindptsEval(dfloat *const out_base,
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
                         void *const in,
                         struct findpts_data_3 *const fd,
                         const void *const ogs_fd);

void ogsLegacyFindptsLocalEval(dfloat *const out_base,
                               const dlong out_stride,
                               const dlong *const el_base,
                               const dlong el_stride,
                               const dfloat *const r_base,
                               const dlong r_stride,
                               const dlong npt,
                               const dfloat *const in,
                               struct findpts_data_3 *const fd);

void ogsDevFindptsLocalEval(void *const out_base,
                              const dlong out_stride,
                              const void *const el_base,
                              const dlong el_stride,
                              const void *const r_base,
                              const dlong r_stride,
                              const dlong npt,
                              void *const in,
                              struct findpts_data_3 *const fd,
                              const void *const ogs_fd);
}

#endif
