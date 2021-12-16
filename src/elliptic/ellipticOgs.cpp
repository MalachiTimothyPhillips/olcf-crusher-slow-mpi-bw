#include "elliptic.h"
#include "platform.hpp"
#include <bcTypes.hpp>

void ellipticOgs(mesh_t *mesh,
                 dlong mNlocal,
                 int nFields,
                 dlong offset,
                 int *BCType,
                 int BCTypeOffset,
                 bool &unaligned,
                 dlong &Nmasked,
                 occa::memory &o_maskIds,
                 dlong &NmaskedLocal,
                 occa::memory &o_maskIdsLocal,
                 dlong &NmaskedGlobal,
                 occa::memory &o_maskIdsGlobal,
                 occa::memory &o_BCType,
                 ogs_t **ogs)
{
  unaligned = false;
  const int Nlocal = (nFields == 1) ? mNlocal : nFields * offset;
  const int largeNumber = 1 << 20;

  int *mapB = (int*) calloc(Nlocal, sizeof(int));
  for(int fld = 0; fld < nFields; fld++) {
    for (dlong e = 0; e < mesh->Nelements; e++) {
      for (int n = 0; n < mesh->Np; n++)
        mapB[n + e * mesh->Np + fld * offset] = largeNumber;
      for (int f = 0; f < mesh->Nfaces; f++) {
        int bc = mesh->EToB[f + e * mesh->Nfaces];
        if (bc > 0) {
          int BCFlag = BCType[bc + BCTypeOffset * fld];
          for (int n = 0; n < mesh->Nfp; n++) {
            int fid = mesh->faceNodes[n + f * mesh->Nfp];
            mapB[fid + e * mesh->Np + fld * offset] =
              mymin(BCFlag, mapB[fid + e * mesh->Np + fld * offset]);
          }
        }
      }
    }
  }
  ogsGatherScatterMany(mapB,
                       nFields,
                       offset,
                       ogsInt,
                       ogsMin,
                       mesh->ogs);

  Nmasked = 0;
  for(int fld = 0; fld < nFields; fld++) {
    for (dlong n = 0; n < mesh->Nlocal; n++) {
      if (mapB[n + fld * offset] == largeNumber) {
        mapB[n + fld * offset] = 0;
      }
      else if (mapB[n + fld * offset] == DIRICHLET) { // Dirichlet boundary
        Nmasked++;
      }
      else if (mapB[n + fld * offset] == DIRICHLETNORMAL) {
        unaligned = true;
      }
    }
  }
  dlong *maskIds = (dlong*) calloc(Nmasked, sizeof(dlong));

  Nmasked = 0;
  for(int fld = 0; fld < nFields; fld++) {
    for (dlong n = 0; n < mesh->Nlocal; n++) {
      if (mapB[n + fld * offset] == 1) maskIds[Nmasked++] = n + fld * offset;
    }
  }
  if(Nmasked) o_maskIds = platform->device.malloc(Nmasked * sizeof(dlong), maskIds);

  NmaskedLocal = 0;
  for (int fld = 0; fld < nFields; fld++) {
    for (dlong el = 0; el < mesh->NlocalGatherElements; ++el) {
      const dlong elemOffset = mesh->localGatherElementList[el] * mesh->Np;
      for (dlong qp = 0; qp < mesh->Np; qp++) {
        const dlong n = elemOffset + qp;
        if (mapB[n + fld * offset] == 1)
          NmaskedLocal++;
      }
    }
  }
  dlong *localMaskIds = (dlong *)calloc(NmaskedLocal, sizeof(dlong));
  NmaskedLocal = 0;
  for (int fld = 0; fld < nFields; fld++) {
    for (dlong el = 0; el < mesh->NlocalGatherElements; ++el) {
      const dlong elemOffset = mesh->localGatherElementList[el] * mesh->Np;
      for (dlong qp = 0; qp < mesh->Np; qp++) {
        const dlong n = elemOffset + qp;
        if (mapB[n + fld * offset] == 1)
          localMaskIds[NmaskedLocal++] = n + fld * offset;
      }
    }
  }
  if (NmaskedLocal)
    o_maskIdsLocal = platform->device.malloc(NmaskedLocal * sizeof(dlong), localMaskIds);
  free(localMaskIds);

  NmaskedGlobal = 0;
  for (int fld = 0; fld < nFields; fld++) {
    for (dlong eg = 0; eg < mesh->NglobalGatherElements; ++eg) {
      const dlong elemOffset = mesh->globalGatherElementList[eg] * mesh->Np;
      for (dlong qp = 0; qp < mesh->Np; qp++) {
        const dlong n = elemOffset + qp;
        if (mapB[n + fld * offset] == 1)
          NmaskedGlobal++;
      }
    }
  }
  dlong *globalMaskIds = (dlong *)calloc(NmaskedGlobal, sizeof(dlong));
  NmaskedGlobal = 0;
  for (int fld = 0; fld < nFields; fld++) {
    for (dlong eg = 0; eg < mesh->NglobalGatherElements; ++eg) {
      const dlong elemOffset = mesh->globalGatherElementList[eg] * mesh->Np;
      for (dlong qp = 0; qp < mesh->Np; qp++) {
        const dlong n = elemOffset + qp;
        if (mapB[n + fld * offset] == 1)
          globalMaskIds[NmaskedGlobal++] = n + fld * offset;
      }
    }
  }
  if (NmaskedGlobal)
    o_maskIdsGlobal = platform->device.malloc(NmaskedGlobal * sizeof(dlong), globalMaskIds);
  free(globalMaskIds);

  free(mapB);

  if(! *ogs) {
    if(nFields > 1) {
      if(platform->comm.mpiRank == 0)
        printf("Creating a masked gs handle for nFields > 1 is currently not supported!\n");
      ABORT(EXIT_FAILURE);
    }

    hlong* maskedGlobalIds = (hlong*) calloc(mesh->Nlocal,sizeof(hlong));
    memcpy(maskedGlobalIds, mesh->globalIds, mesh->Nlocal * sizeof(hlong));
    for (dlong n = 0; n < Nmasked; n++) maskedGlobalIds[maskIds[n]] = 0;
    *ogs = ogsSetup(mesh->Nlocal, maskedGlobalIds, platform->comm.mpiComm, 1, platform->device.occaDevice());
    free(maskedGlobalIds);
  }
  free(maskIds);

  o_BCType = platform->device.malloc(BCTypeOffset * nFields * sizeof(int), BCType);
}
