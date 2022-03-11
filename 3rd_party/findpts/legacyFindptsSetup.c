/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

/* compile with C compiler (not C++) */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "gslib.h"

#include "ogstypes.h"
#include "findptsTypes.h"

struct findpts_data_3 *legacyFindptsSetup(
  MPI_Comm mpi_comm,
  const dfloat *const elx[3],
  const dlong n[3], const dlong nel,
  const dlong m[3], const dfloat bbox_tol,
  const hlong local_hash_size, const hlong global_hash_size,
  const dlong npt_max, const dfloat newt_tol) {

  if (sizeof(dfloat) != sizeof(double)) {
    fail(1,__FILE__,__LINE__,"findpts's dfloat is not compatible with gslib's double");
  }
  if (sizeof(dlong) != sizeof(uint)) {
    fail(1,__FILE__,__LINE__,"findpts's dlong is not compatible with gslib's uint");
  }

  struct comm gs_comm;
  comm_init(&gs_comm, mpi_comm);

  struct findpts_data_3* fd = findpts_setup_3(&gs_comm, elx, n, nel, m, bbox_tol,
                                              local_hash_size, global_hash_size,
                                              npt_max, newt_tol);
  comm_free(&gs_comm);
  return fd;
}

void legacyFindptsFree(struct findpts_data_3 *fd) {
  findpts_free_3(fd);
}