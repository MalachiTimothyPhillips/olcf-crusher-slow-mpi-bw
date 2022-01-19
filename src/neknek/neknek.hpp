#if !defined(neknek_hpp_)
#define neknek_hpp_

#include <mpi.h>
#include "nrssys.hpp"
#include "ogsFindpts.hpp"

struct neknek_t {
  dlong nsessions, sessionID;
  MPI_Comm globalComm;
  MPI_Comm localComm;
  bool connected;

  dlong NcorrectorSteps;

  dlong Nscalar;
  bool globalMovingMesh;
  ogs_findpts_t *ogsHandle = nullptr;
  dlong npt;

  dlong *pointMap;
  occa::memory o_pointMap;

  dfloat *valInterp = nullptr;
  occa::memory o_valInterp;

  ogs_findpts_data_t *findPtsData;
};


struct nrs_t;
void neknekSetup(nrs_t *nrs);
void neknekUpdateBoundary(nrs_t *nrs);


#endif
