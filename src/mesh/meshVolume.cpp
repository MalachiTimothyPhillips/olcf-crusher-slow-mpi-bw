#include "mesh3D.h"
#include "platform.hpp"
#include "linAlg.hpp"

void meshVolume(mesh_t* meshV, mesh_t* meshT){

  dfloat volume = 0.0;
  const auto Np = meshT->Np;
  const auto Nggeo = meshT->Nggeo;

  for(dlong e = 0; e < meshT->Nelements; ++e) {
    
    // only count contribution of fluid domain
    if(!meshT->elementInfo[e]){
      for(dlong n = 0; n < Np; ++n){
          volume += meshT->ggeo[Nggeo * Np * e + n + Np * GWJID];
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  meshV->volume = volume;
}