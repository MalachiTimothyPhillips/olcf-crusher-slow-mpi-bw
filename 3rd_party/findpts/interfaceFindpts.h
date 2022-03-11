
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
}

#endif
