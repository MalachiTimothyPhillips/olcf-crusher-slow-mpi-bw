
#include <cfloat>
#include "neknek.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "ogsKernelsFindpts.hpp"

static void reserveAllocation(nrs_t *nrs, dlong npt) {
  neknek_t *neknek = nrs->neknek;
  const dlong D = nrs->dim;

  if(neknek->valInterp == nullptr || neknek->npt != npt) {

    delete[] neknek->valInterp;

    neknek->valInterp = (dfloat *)calloc((D + neknek->Nscalar) * npt, sizeof(dfloat));
    neknek->pointMap = (dlong *)calloc(nrs->fieldOffset + 1, sizeof(dlong));

    neknek->o_pointMap  = platform->device.malloc((nrs->fieldOffset+1) * sizeof(dfloat));
    if(npt > 0) {
      neknek->o_valInterp = platform->device.malloc((D+neknek->Nscalar)*npt * sizeof(dfloat));
    } else {
      neknek->o_valInterp = platform->device.malloc(1 * sizeof(dfloat));
    }

    neknek->npt = npt;
  }

}


static void findInterpPoints(nrs_t* nrs){
  neknek_t *neknek = nrs->neknek;

  const dlong nsessions = neknek->nsessions;
  const dlong sessionID = neknek->sessionID;
  MPI_Comm globalComm = neknek->globalComm;

  const dlong D = nrs->dim;
  const mesh_t *mesh = nrs->meshV;
  const dlong nmsh = mesh->N;
  const dlong nelm = mesh->Nelements;
  const dlong nfac = mesh->Nfaces;
  const dlong nfpt = mesh->Nfp;

  occa::device &device = platform_t::getInstance()->device.occaDevice();

  // Setup findpts
  dfloat tol = 5e-13;
  dlong npt_max = 128;
  dfloat bb_tol = 0.01;
  dlong n1[3] = {nmsh+1, nmsh+1, nmsh+1};
  dlong nf1[3] = {2*n1[0],2*n1[1],2*n1[2]};
  dlong npt_per_elm = n1[0]*n1[1]*(D==3?n1[2]:1);
  dlong ntot = npt_per_elm*nelm;

  dfloat *elx_null[3] = {nullptr, nullptr, nullptr};
  dfloat *elx     [3] = {mesh->x, mesh->y, mesh->z};

  if (neknek->ogsHandle != nullptr) {
    ogsFindptsFree(neknek->ogsHandle);
  }

  ogs_findpts_t **ogsHandles = new ogs_findpts_t*[nsessions];
  for(dlong i = 0; i < nsessions; ++i) {
    ogsHandles[i] = ogsFindptsSetup(D, globalComm,
                                    (i == sessionID ? elx : elx_null),
                                    n1, (i == sessionID ? nelm : 0),
                                    nf1, bb_tol, ntot, ntot, npt_max, tol,
                                    &device);
  }
  neknek->ogsHandle = ogsFindptsSetup(D, globalComm, elx, n1, nelm,
                                      nf1, bb_tol, ntot, ntot, npt_max, tol,
                                      &device);

  constexpr dlong faceMap[6] = {5, 0, 1, 2, 3, 4};

  dlong num_interp_faces = 0;
  dlong *intflag = (dlong*)nek::ptr("intflag");
  for (dlong e = 0; e < nelm; ++e) {
    for (dlong f = 0; f < nfac; ++f) {
      num_interp_faces += intflag[faceMap[f]+nfac*e]!=0;
    }
  }
  dlong npt = num_interp_faces*nfpt;
  reserveAllocation(nrs, npt);

  dfloat *interpX = new dfloat[npt*D];
  dlong ip = 0;
  std::fill(neknek->pointMap, neknek->pointMap+nrs->fieldOffset, -1);
  for (dlong e = 0; e < nelm; ++e) {
    for (dlong f = 0; f < nfac; ++f) {

      for(dlong m = 0; m < nfpt; ++m) {
        dlong id = nfac*nfpt*e + nfpt*f + m;
        dlong idM = mesh->vmapM[id];

        if (intflag[faceMap[f]+nfac*e]) {
          for(dlong d = 0; d<D; ++d) {
            interpX[ip + d*npt] = elx[d][idM];
          }
          neknek->pointMap[idM] = ip;
          ++ip;
        }
      }
    }
  }
  neknek->pointMap[nrs->fieldOffset] = neknek->npt;
  neknek->o_pointMap.copyFrom(neknek->pointMap);

  auto findPtsData = new ogs_findpts_data_t(npt);
  neknek->findPtsData = findPtsData;

  dfloat *interpX_3[3] = {interpX, interpX+npt, interpX+2*npt};
  dfloat *nullptr_3[3] = {nullptr, nullptr, nullptr};
  dlong interpXStride[3] = {1*sizeof(dfloat), 1*sizeof(dfloat), 1*sizeof(dfloat)};
  for(dlong sess = 0; sess < nsessions; ++sess) {
    ogsFindpts(findPtsData,
               (sess == sessionID) ? nullptr_3 : interpX_3,
               interpXStride,
               (sess == sessionID) ? 0 : npt,
               ogsHandles[sess]);
  }
  // TODO add warning prints


  for(dlong i = 0; i < nsessions; ++i) {
    ogsFindptsFree(ogsHandles[i]);
  }

  delete[] interpX;
}

neknek_t::neknek_t(nrs_t *nrs, const session_data_t &session)
    : nsessions(session.nsessions), sessionID(session.sessionID), globalComm(session.globalComm),
      localComm(session.localComm), connected(session.connected)
{

  nrs->neknek = this;
  if(nrs->cds){
    nrs->cds->neknek = this;
  }

  int nsessmax = 0;
  platform->options.getArgs("NEKNEK MAX NUM SESSIONS", nsessmax);
  if (nsessmax > 1) {
    this->connected = true;
  }

  platform->options.getArgs("NEKNEK CORRECTOR STEPS", this->NcorrectorSteps);

  neknekSetup(nrs);
}

void neknekSetup(nrs_t *nrs)
{
  if(platform->options.compareArgs("BUILD ONLY", "TRUE")) {
    int maxSessions;
    platform->options.getArgs("NEKNEK MAX NUM SESSIONS", maxSessions);
    if (maxSessions >= 2) {
      occa::device device = platform_t::getInstance()->device.occaDevice();
      dlong Nq = nrs->meshV->Nq;
      MPI_Comm comm = platform->comm.mpiComm;

      // precompile kernels
      // OCCA automatically garbage collections
      std::pair<occa::kernel, occa::kernel> kernels = ogs::initFindptsKernel(comm, device, 3, Nq);
    }
    return;
  }

  neknek_t *neknek = nrs->neknek;
  if(!neknek->connected) {
    reserveAllocation(nrs, 0);
    neknek->pointMap[nrs->fieldOffset] = 0;
    neknek->o_pointMap.copyFrom(neknek->pointMap);
    return;
  }

  const dlong nsessions = neknek->nsessions;
  MPI_Comm globalComm = neknek->globalComm;
  dlong globalRank;
  MPI_Comm_rank(globalComm, &globalRank);

  if(globalRank == 0) printf("configuring neknek with %d sessions\n", nsessions);

  dlong nFields[2];
  nFields[0] = nrs->Nscalar;
  nFields[1] = -nFields[0];
  MPI_Allreduce(MPI_IN_PLACE, nFields, 2, MPI_DLONG, MPI_MIN, globalComm);
  nFields[1] = -nFields[1];
  if (nFields[0] != nFields[1]) {
    if(globalRank == 0) {
      std::cout << "WARNING: varying numbers of scalars; only updating " << nFields[0] << std::endl;
    }
  }
  neknek->Nscalar = nFields[0];

  platform->options.getArgs("NEKNEK CORRECTOR STEPS", neknek->NcorrectorSteps);

  const dlong movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  dlong globalMovingMesh;
  MPI_Allreduce(&movingMesh, &globalMovingMesh, 1, MPI_DLONG, MPI_MAX, globalComm);
  neknek->globalMovingMesh = globalMovingMesh;

  findInterpPoints(nrs);
}

static void fieldEval(nrs_t *nrs, dlong field, occa::memory in) {

  neknek_t *neknek = nrs->neknek;
  const dlong D = nrs->dim;
  dfloat *out = neknek->valInterp+field*neknek->npt;

  ogsFindptsEval(out, neknek->findPtsData, neknek->npt, in, neknek->ogsHandle);
}

void neknekUpdateBoundary(nrs_t *nrs)
{
  neknek_t *neknek = nrs->neknek;
  if(!neknek->connected) return;

  if (neknek->globalMovingMesh) {
    findInterpPoints(nrs);
  }

  const dlong D = nrs->dim;

  fieldEval(nrs, 0, nrs->o_U + 0 * nrs->fieldOffset * sizeof(dfloat));
  fieldEval(nrs, 1, nrs->o_U + 1 * nrs->fieldOffset * sizeof(dfloat));
  fieldEval(nrs, 2, nrs->o_U + 2 * nrs->fieldOffset * sizeof(dfloat));

  if (neknek->Nscalar > 0) {
    for(dlong i = 0; i < neknek->Nscalar; ++i) {
      fieldEval(nrs, D+i, nrs->cds->o_S+nrs->cds->fieldOffsetScan[i]*sizeof(dfloat));
    }
  }
  // TODO Allow for higher order extrapolation
  // TODO figure out chk_outflow

  neknek->o_valInterp.copyFrom(neknek->valInterp);
}
