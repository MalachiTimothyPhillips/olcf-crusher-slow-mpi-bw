#if !defined(neknek_hpp_)
#define neknek_hpp_

#include <mpi.h>
#include "nrssys.hpp"
#include "findpts.hpp"

struct nrs_t;

struct session_data_t {
  dlong nsessions, sessionID;
  MPI_Comm globalComm;
  MPI_Comm localComm;
  bool connected;
};

struct neknek_t {
  dlong nsessions, sessionID;
  MPI_Comm globalComm;
  MPI_Comm localComm;
  bool connected;

  dlong NcorrectorSteps;

  dlong Nscalar;
  bool globalMovingMesh;
  findpts_t *ogsHandle = nullptr;
  dlong npt;

  dlong *pointMap;
  occa::memory o_pointMap;

  dfloat *valInterp = nullptr;
  occa::memory o_valInterp;

  findpts_data_t *findPtsData;
  neknek_t(nrs_t *nrs, const session_data_t &session);
};

void neknekSetup(nrs_t *nrs);
void neknekUpdateBoundary(nrs_t *nrs);


#endif
