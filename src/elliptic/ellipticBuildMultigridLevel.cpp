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

#include "elliptic.h"
#include "platform.hpp"

namespace{
std::string gen_suffix(const elliptic_t * elliptic, const char * floatString)
{
  const std::string precision = std::string(floatString);
  if(precision.find(pfloatString) != std::string::npos){
    return std::string("_") + std::to_string(elliptic->mesh->N) + std::string("pfloat");
  }
  else{
    return std::string("_") + std::to_string(elliptic->mesh->N);
  }
  
}
}

// create elliptic and mesh structs for multigrid levels
elliptic_t* ellipticBuildMultigridLevel(elliptic_t* baseElliptic, int Nc, int Nf)
{
  
  elliptic_t* elliptic = new elliptic_t();

  memcpy(elliptic,baseElliptic,sizeof(elliptic_t));

  //populate the mini-mesh using the mesh struct
  mesh_t* mesh = new mesh_t();
  memcpy(mesh,baseElliptic->mesh,sizeof(mesh_t));

  elliptic->mesh = mesh;

  setupAide options = elliptic->options;

  switch(elliptic->elementType) {
  case HEXAHEDRA:
    meshLoadReferenceNodesHex3D(mesh, Nc, 1);
    meshHaloSetup(mesh);
    meshPhysicalNodesHex3D(mesh);
    meshHaloPhysicalNodes(mesh);
    meshGeometricFactorsHex3D(mesh);

    if(!options.compareArgs("BOX DOMAIN", "TRUE")) {
      meshConnectFaceNodes3D(mesh);
    }else {
      if(platform->comm.mpiRank == 0)
        printf("WARNING: connecting periodic box\n");

      dfloat XMIN = -1, XMAX = +1; // default bi-unit cube
      dfloat YMIN = -1, YMAX = +1;
      dfloat ZMIN = -1, ZMAX = +1;

      options.getArgs("BOX XMIN", XMIN);
      options.getArgs("BOX YMIN", YMIN);
      options.getArgs("BOX ZMIN", ZMIN);

      options.getArgs("BOX XMAX", XMAX);
      options.getArgs("BOX YMAX", YMAX);
      options.getArgs("BOX ZMAX", ZMAX);

      meshConnectPeriodicFaceNodes3D(mesh, XMAX - XMIN, YMAX - YMIN, ZMAX - ZMIN);
    }
    meshSurfaceGeometricFactorsHex3D(mesh);
    break;
  }

  // global nodes
  meshGlobalIds(mesh);

#if 0
  mesh->o_x = platform->device.malloc(mesh->Np * mesh->Nelements * sizeof(dfloat), mesh->x);
  mesh->o_y = platform->device.malloc(mesh->Np * mesh->Nelements * sizeof(dfloat), mesh->y);
  mesh->o_z = platform->device.malloc(mesh->Np * mesh->Nelements * sizeof(dfloat), mesh->z);
#endif

  //dont need these once vmap is made
  free(mesh->x);
  free(mesh->y);
  if (elliptic->dim == 3)
    free(mesh->z);

  dlong Ntotal = mesh->Np * mesh->Nelements;

  if (elliptic->elementType == HEXAHEDRA) {
    //lumped mass matrix
    mesh->MM = (dfloat*) calloc(mesh->Np * mesh->Np, sizeof(dfloat));
    dfloat* DT = (dfloat*) calloc(mesh->Nq * mesh->Nq, sizeof(dfloat));

    for (int j = 0; j < mesh->Nq; j++)
      for (int i = 0; i < mesh->Nq; i++)
        DT[j * mesh->Nq + i] = mesh->D[i * mesh->Nq + j];

    for (int k = 0; k < mesh->Nq; k++)
      for (int j = 0; j < mesh->Nq; j++)
        for (int i = 0; i < mesh->Nq; i++) {
          int n = i + j * mesh->Nq + k * mesh->Nq * mesh->Nq;
          mesh->MM[n + n * mesh->Np] = mesh->gllw[i] * mesh->gllw[j] * mesh->gllw[k];
        }

    mesh->o_D = platform->device.malloc(mesh->Nq * mesh->Nq * sizeof(dfloat), mesh->D);
    mesh->o_DT = platform->device.malloc(mesh->Nq * mesh->Nq * sizeof(dfloat), DT); // transpose(D)

#if 0
    mesh->o_cubD = platform->device.malloc(mesh->cubNq * mesh->cubNq * sizeof(dfloat), mesh->cubD);

    dfloat* cubInterpT = (dfloat*) calloc(mesh->cubNq * mesh->Nq, sizeof(dfloat));
    for(int n = 0; n < mesh->Nq; ++n)
      for(int m = 0; m < mesh->cubNq; ++m)
        cubInterpT[m + n * mesh->cubNq] = mesh->cubInterp[m * mesh->Nq + n];

    mesh->o_cubInterpT = platform->device.malloc(mesh->cubNq * mesh->Nq * sizeof(dfloat), cubInterpT);

    free(cubInterpT);
#endif

    mesh->o_ggeo =
      platform->device.malloc(mesh->Nelements * mesh->Np * mesh->Nggeo * sizeof(dfloat),
                          mesh->ggeo);
#if 0
    mesh->o_vgeo =
      platform->device.malloc(mesh->Nelements * mesh->Nvgeo * mesh->Np * sizeof(dfloat),
                          mesh->vgeo);
    mesh->o_sgeo =
      platform->device.malloc(mesh->Nelements * mesh->Nfaces * mesh->Nfp * mesh->Nsgeo * sizeof(dfloat),
                          mesh->sgeo);

    mesh->o_vmapM =
      platform->device.malloc(mesh->Nelements * mesh->Nfp * mesh->Nfaces * sizeof(dlong),
                          mesh->vmapM);

    mesh->o_vmapP =
      platform->device.malloc(mesh->Nelements * mesh->Nfp * mesh->Nfaces * sizeof(dlong),
                          mesh->vmapP);
#endif

  }

  //set the normalization constant for the allNeumann Poisson problem on this coarse mesh
  hlong localElements = (hlong) mesh->Nelements;
  hlong totalElements = 0;
  MPI_Allreduce(&localElements, &totalElements, 1, MPI_HLONG, MPI_SUM, platform->comm.mpiComm);
  elliptic->allNeumannScale = 1.0 / sqrt(mesh->Np * totalElements);

  elliptic->allNeumannPenalty = 0;
  elliptic->allNeumannScale = 0;

  //setup an unmasked gs handle
  int verbose = options.compareArgs("VERBOSE","TRUE") ? 1:0;
  meshParallelGatherScatterSetup(mesh, Ntotal, mesh->globalIds, platform->comm.mpiComm, verbose);

  //make a node-wise bc flag using the gsop (prioritize Dirichlet boundaries over Neumann)
  elliptic->mapB = (int*) calloc(mesh->Nelements * mesh->Np,sizeof(int));
  for (dlong e = 0; e < mesh->Nelements; e++) {
    for (int n = 0; n < mesh->Np; n++) elliptic->mapB[n + e * mesh->Np] = 1E9;
    for (int f = 0; f < mesh->Nfaces; f++) {
      int bc = mesh->EToB[f + e * mesh->Nfaces];
      if (bc > 0) {
        for (int n = 0; n < mesh->Nfp; n++) {
          int BCFlag = elliptic->BCType[bc];
          int fid = mesh->faceNodes[n + f * mesh->Nfp];
          elliptic->mapB[fid + e * mesh->Np] = mymin(BCFlag,elliptic->mapB[fid + e * mesh->Np]);
        }
      }
    }
  }
  ogsGatherScatter(elliptic->mapB, ogsInt, ogsMin, mesh->ogs);

  //use the bc flags to find masked ids
  elliptic->Nmasked = 0;
  for (dlong n = 0; n < mesh->Nelements * mesh->Np; n++) {
    if (elliptic->mapB[n] == 1E9)
      elliptic->mapB[n] = 0.;
    else if (elliptic->mapB[n] == 1)     //Dirichlet boundary
      elliptic->Nmasked++;
  }
  elliptic->o_mapB = platform->device.malloc(mesh->Nelements * mesh->Np * sizeof(int), elliptic->mapB);

  elliptic->maskIds = (dlong*) calloc(elliptic->Nmasked, sizeof(dlong));
  elliptic->Nmasked = 0; //reset
  for (dlong n = 0; n < mesh->Nelements * mesh->Np; n++)
    if (elliptic->mapB[n] == 1) elliptic->maskIds[elliptic->Nmasked++] = n;

  if (elliptic->Nmasked) 
     elliptic->o_maskIds = platform->device.malloc(elliptic->Nmasked * sizeof(dlong), elliptic->maskIds);

  //make a masked version of the global id numbering
  hlong* maskedGlobalIds = (hlong*) calloc(Ntotal,sizeof(hlong));
  memcpy(maskedGlobalIds, mesh->globalIds, Ntotal * sizeof(hlong));
  for (dlong n = 0; n < elliptic->Nmasked; n++)
    maskedGlobalIds[elliptic->maskIds[n]] = 0;

  //use the masked ids to make another gs handle
  elliptic->ogs = ogsSetup(Ntotal, maskedGlobalIds, platform->comm.mpiComm, verbose, platform->device.occaDevice());
  elliptic->o_invDegree = elliptic->ogs->o_invDegree;
  free(maskedGlobalIds);

  std::string suffix = "Hex3D";

  std::string kernelName;

  MPI_Barrier(platform->comm.mpiComm);
  double tStartLoadKernel = MPI_Wtime();

  ellipticBuildPreconditionerKernels(elliptic);

  const std::string poissonPrefix = elliptic->poisson ? "poisson-" : "";

  {
      // check for trilinear
      if(elliptic->elementType != HEXAHEDRA) {
        kernelName = "ellipticPartialAx" + suffix;
      }else {
        if(elliptic->options.compareArgs("ELEMENT MAP", "TRILINEAR"))
          kernelName = "ellipticPartialAxTrilinear" + suffix;
        else
          kernelName = "ellipticPartialAx" + suffix;
      }

      {
        const std::string kernelSuffix = gen_suffix(elliptic, dfloatString);
        elliptic->AxKernel = platform->kernels.getKernel(poissonPrefix + kernelName + kernelSuffix);
      }
      if(!strstr(pfloatString,dfloatString)) {
        const std::string kernelSuffix = gen_suffix(elliptic, pfloatString);
        elliptic->AxPfloatKernel =
          platform->kernels.getKernel(poissonPrefix + kernelName + kernelSuffix);
      }
  }

  MPI_Barrier(platform->comm.mpiComm);
  if(platform->comm.mpiRank == 0) printf("done (%gs)\n", MPI_Wtime() - tStartLoadKernel);
  fflush(stdout);

  //new precon struct
  elliptic->precon = new precon_t();

  {

    const std::string kernelSuffix = std::string("_") + std::to_string(Nf);

    kernelName = "ellipticPreconCoarsen" + suffix;
    elliptic->precon->coarsenKernel = platform->kernels.getKernel(kernelName + kernelSuffix);
    kernelName = "ellipticPreconProlongate" + suffix;
    elliptic->precon->prolongateKernel = platform->kernels.getKernel(kernelName + kernelSuffix);

  }

  if(elliptic->elementType == HEXAHEDRA) {
    // pack gllz, gllw, and elementwise EXYZ
    dfloat* gllzw = (dfloat*) calloc(2 * mesh->Nq, sizeof(dfloat));

    int sk = 0;
    for(int n = 0; n < mesh->Nq; ++n)
      gllzw[sk++] = mesh->gllz[n];
    for(int n = 0; n < mesh->Nq; ++n)
      gllzw[sk++] = mesh->gllw[n];

    elliptic->o_gllzw = platform->device.malloc(2 * mesh->Nq * sizeof(dfloat), gllzw);
    free(gllzw);
  }

  if(!strstr(pfloatString,dfloatString)) {
    elliptic->o_lambdaPfloat = platform->device.malloc(1,  sizeof(pfloat));
    const pfloat one = 1.0;
    elliptic->o_lambdaPfloat.copyFrom(&one, sizeof(pfloat));
    mesh->o_ggeoPfloat = platform->device.malloc(mesh->Nelements * mesh->Np * mesh->Nggeo ,  sizeof(pfloat));
    mesh->o_DPfloat = platform->device.malloc(mesh->Nq * mesh->Nq ,  sizeof(pfloat));
    mesh->o_DTPfloat = platform->device.malloc(mesh->Nq * mesh->Nq ,  sizeof(pfloat));
    elliptic->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np * mesh->Nggeo,
                                       mesh->o_ggeo,
                                       elliptic->mesh->o_ggeoPfloat);
#if 0    
    mesh->o_ggeo.free();
#endif    
    elliptic->copyDfloatToPfloatKernel(mesh->Nq * mesh->Nq,
                                       mesh->o_D,
                                       elliptic->mesh->o_DPfloat);
    elliptic->copyDfloatToPfloatKernel(mesh->Nq * mesh->Nq,
                                       mesh->o_DT,
                                       elliptic->mesh->o_DTPfloat);
  }


  return elliptic;
}
