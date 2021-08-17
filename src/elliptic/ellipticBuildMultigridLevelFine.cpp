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
  std::string precision = std::string(floatString);
  std::string FPString = "";
  if(precision.find("float") != std::string::npos)
    FPString = "FP32";
  else
    FPString = "FP64";
  
  return std::string("_") + std::to_string(elliptic->mesh->N) + FPString;
}
}

elliptic_t* ellipticBuildMultigridLevelFine(elliptic_t* baseElliptic)
{
  
  elliptic_t* elliptic = new elliptic_t();
  memcpy(elliptic, baseElliptic, sizeof(*baseElliptic));

  mesh_t* mesh = elliptic->mesh;
  occa::properties kernelInfo = ellipticKernelInfo(mesh);
  ellipticBuildPreconditionerKernels(elliptic, kernelInfo);

  const int serial = platform->device.mode() == "Serial" || platform->device.mode() == "OpenMP";

  elliptic->var_coeff = 0;
  elliptic->lambda = (dfloat*) calloc(elliptic->Nfields, sizeof(dfloat)); // enforce lambda = 0


  if(!strstr(pfloatString,dfloatString)) {
    mesh->o_ggeoPfloat = platform->device.malloc(mesh->Nelements * mesh->Np * mesh->Nggeo ,  sizeof(pfloat));
    mesh->o_DPfloat = platform->device.malloc(mesh->Nq * mesh->Nq ,  sizeof(pfloat));
    mesh->o_DTPfloat = platform->device.malloc(mesh->Nq * mesh->Nq ,  sizeof(pfloat));

    elliptic->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np * mesh->Nggeo,
                                       elliptic->mesh->o_ggeoPfloat,
                                       mesh->o_ggeo);
    elliptic->copyDfloatToPfloatKernel(mesh->Nq * mesh->Nq,
                                       elliptic->mesh->o_DPfloat,
                                       mesh->o_D);
    elliptic->copyDfloatToPfloatKernel(mesh->Nq * mesh->Nq,
                                       elliptic->mesh->o_DTPfloat,
                                       mesh->o_DT);
  }

  string suffix;
  if(elliptic->elementType == HEXAHEDRA)
    suffix = "Hex3D";

  // add custom defines
  kernelInfo["defines/" "p_Nverts"] = mesh->Nverts;

  kernelInfo["defines/" "p_eNfields"] = elliptic->Nfields;

  string filename, kernelName;

  string install_dir;
  install_dir.assign(getenv("NEKRS_INSTALL_DIR"));
  const string oklpath = install_dir + "/okl/elliptic/";

  {
      occa::properties AxKernelInfo = kernelInfo;

      filename = oklpath + "ellipticAx" + suffix + ".okl";
      kernelName = "ellipticAx" + suffix;
      if(serial) {
        filename = oklpath + "ellipticSerialAx" + suffix + ".c";
      }
      {
        const std::string kernelSuffix = gen_suffix(elliptic, dfloatString);
        elliptic->AxKernel = platform->device.buildKernel(filename,kernelName,AxKernelInfo, kernelSuffix);
      }

      if(!strstr(pfloatString,dfloatString)) {
        AxKernelInfo["defines/" "dfloat"] = pfloatString;
        kernelName = "ellipticAx" + suffix;
        const std::string kernelSuffix = gen_suffix(elliptic, pfloatString);
        elliptic->AxPfloatKernel = platform->device.buildKernel(filename,kernelName,AxKernelInfo, kernelSuffix);
        AxKernelInfo["defines/" "dfloat"] = dfloatString;
      }

      if(elliptic->options.compareArgs("ELEMENT MAP", "TRILINEAR"))
        kernelName = "ellipticPartialAxTrilinear" + suffix;
      else
        kernelName = "ellipticPartialAx" + suffix;

      if(!serial) {
        {
          const std::string kernelSuffix = gen_suffix(elliptic, dfloatString);
          elliptic->partialAxKernel = platform->device.buildKernel(filename,kernelName,AxKernelInfo, kernelSuffix);
        }
        if(!strstr(pfloatString,dfloatString)) {
          AxKernelInfo["defines/" "dfloat"] = pfloatString;
          const std::string kernelSuffix = gen_suffix(elliptic, pfloatString);
          elliptic->partialAxPfloatKernel =
            platform->device.buildKernel(filename, kernelName, AxKernelInfo, kernelSuffix);
          AxKernelInfo["defines/" "dfloat"] = dfloatString;
        }
      }
  }

  return elliptic;
}
