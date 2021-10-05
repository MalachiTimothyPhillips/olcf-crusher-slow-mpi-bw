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

elliptic_t* ellipticBuildMultigridLevelFine(elliptic_t* baseElliptic)
{
  
  elliptic_t* elliptic = new elliptic_t();
  memcpy(elliptic, baseElliptic, sizeof(*baseElliptic));

  mesh_t* mesh = elliptic->mesh;
  ellipticBuildPreconditionerKernels(elliptic);

  const int serial = platform->device.mode() == "Serial" || platform->device.mode() == "OpenMP";

  elliptic->var_coeff = 0;
  elliptic->lambda = (dfloat*) calloc(elliptic->Nfields, sizeof(dfloat)); // enforce lambda = 0

  constexpr int ndim {3};
  if(elliptic->elementType == HEXAHEDRA) {
    // pack gllz, gllw, and elementwise EXYZ
    dfloat* gllzw = (dfloat*) calloc(2 * mesh->Nq, sizeof(dfloat));
    dfloat* EXYZ = (dfloat*) calloc(mesh->Nelements * ndim * mesh->Nverts, sizeof(dfloat));

    int sk = 0;
    for(int n = 0; n < mesh->Nq; ++n)
      gllzw[sk++] = mesh->gllz[n];
    for(int n = 0; n < mesh->Nq; ++n)
      gllzw[sk++] = mesh->gllw[n];
    
    sk = 0;
    for(hlong e=0;e<mesh->Nelements;++e){
      for(int v=0;v<mesh->Nverts;++v)
        EXYZ[sk++] = mesh->EX[e*mesh->Nverts+v];
      for(int v=0;v<mesh->Nverts;++v)
        EXYZ[sk++] = mesh->EY[e*mesh->Nverts+v];
      for(int v=0;v<mesh->Nverts;++v)
        EXYZ[sk++] = mesh->EZ[e*mesh->Nverts+v];
    }

    elliptic->o_gllzw = platform->device.malloc(2 * mesh->Nq * sizeof(dfloat), gllzw);
    elliptic->o_EXYZ = platform->device.malloc(mesh->Nelements * ndim * mesh->Nverts * sizeof(dfloat), EXYZ);
    free(gllzw);
    free(EXYZ);
  }


  if(!strstr(pfloatString,dfloatString)) {
    mesh->o_ggeoPfloat = platform->device.malloc(mesh->Nelements * mesh->Np * mesh->Nggeo ,  sizeof(pfloat));
    mesh->o_DPfloat = platform->device.malloc(mesh->Nq * mesh->Nq ,  sizeof(pfloat));
    mesh->o_DTPfloat = platform->device.malloc(mesh->Nq * mesh->Nq ,  sizeof(pfloat));
    elliptic->o_gllzwPfloat = platform->device.malloc(2 * mesh->Nq * sizeof(pfloat));
    elliptic->o_EXYZPfloat = platform->device.malloc(mesh->Nelements * ndim * mesh->Nverts * sizeof(pfloat));

    elliptic->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np * mesh->Nggeo,
                                       elliptic->mesh->o_ggeoPfloat,
                                       mesh->o_ggeo);
    elliptic->copyDfloatToPfloatKernel(mesh->Nq * mesh->Nq,
                                       elliptic->mesh->o_DPfloat,
                                       mesh->o_D);
    elliptic->copyDfloatToPfloatKernel(mesh->Nq * mesh->Nq,
                                       elliptic->mesh->o_DTPfloat,
                                       mesh->o_DT);
    elliptic->copyDfloatToPfloatKernel(2 * mesh->Nq,
                                       elliptic->o_gllzwPfloat,
                                       elliptic->o_gllzw);
    elliptic->copyDfloatToPfloatKernel(mesh->Nelements * ndim * mesh->Nverts,
                                       elliptic->o_EXYZPfloat,
                                       elliptic->o_gllzwPfloat);
  }

  std::string suffix;
  if(elliptic->elementType == HEXAHEDRA)
    suffix = "Hex3D";

  std::string kernelName;

  {
      kernelName = "ellipticAx" + suffix;
      {
        const std::string kernelSuffix = gen_suffix(elliptic, dfloatString);
        elliptic->AxKernel = platform->kernels.getKernel(kernelName + kernelSuffix);
      }

      if(!strstr(pfloatString,dfloatString)) {
        kernelName = "ellipticAx" + suffix;
        const std::string kernelSuffix = gen_suffix(elliptic, pfloatString);
        elliptic->AxPfloatKernel = platform->kernels.getKernel(kernelName + kernelSuffix);
      }

      //if(elliptic->options.compareArgs("ELEMENT MAP", "TRILINEAR"))
      if(true)
        kernelName = "ellipticPartialAxTrilinear" + suffix;
      else
        kernelName = "ellipticPartialAx" + suffix;

      if(!serial) {
        {
          const std::string kernelSuffix = gen_suffix(elliptic, dfloatString);
          elliptic->partialAxKernel = platform->kernels.getKernel(kernelName + kernelSuffix);
        }
        if(!strstr(pfloatString,dfloatString)) {
          const std::string kernelSuffix = gen_suffix(elliptic, pfloatString);
          elliptic->partialAxPfloatKernel =
            platform->kernels.getKernel( kernelName + kernelSuffix);
        }
      }
  }

  return elliptic;
}
