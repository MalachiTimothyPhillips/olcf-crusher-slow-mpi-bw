
#ifndef INTERFACE_FINDPTS_H
#define INTERFACE_FINDPTS_H

#include "ogstypes.h"

#ifdef __cplusplus
extern "C" {
#endif
struct gslibFindptsData_t;

struct gslibFindptsData_t *legacyFindptsSetup(MPI_Comm comm,
                                              const dfloat *const elx[3],
                                              const dlong n[3],
                                              const dlong nel,
                                              const dlong m[3],
                                              const dfloat bbox_tol,
                                              const hlong local_hash_size,
                                              const hlong global_hash_size,
                                              const dlong npt_max,
                                              const dfloat newt_tol);

#ifdef __cplusplus
}
#endif
#endif
