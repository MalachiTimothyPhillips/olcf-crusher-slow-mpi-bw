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
//#include "ogsInterface.h"
#include <iostream>

#include "omp.h"
void ellipticAx(elliptic_t* elliptic,
                dlong NelementsList,
                occa::memory &o_elementsList,
                occa::memory &o_q,
                occa::memory &o_Aq,
                const char* precision)
{

  if(NelementsList == 0) return;

  mesh_t* mesh = elliptic->mesh;
  setupAide &options = elliptic->options;

  const bool continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  const bool serial = useSerial();
  const int mapType = (elliptic->elementType == HEXAHEDRA &&
                       options.compareArgs("ELEMENT MAP", "TRILINEAR")) ? 1:0;
  const int integrationType = (elliptic->elementType == HEXAHEDRA &&
                               options.compareArgs("ELLIPTIC INTEGRATION", "CUBATURE")) ? 1:0;
  const std::string precisionStr(precision);
  const std::string pFloatStr(pfloatString);

  bool valid = true;
  valid &= continuous;
  if(!strstr(precision, dfloatString)) {
    valid &= !elliptic->var_coeff;
    valid &= !elliptic->blockSolver;
    if(!serial) {
      valid &= mapType == 0;
      valid &= integrationType == 0;
    }
  }
  if(!valid) {
    printf("Encountered invalid configuration inside ellipticAx!\n");
    if(elliptic->var_coeff)
      printf("Precision level (%s) does not support variable coefficient\n", precision);
    if(elliptic->blockSolver)
      printf("Precision level (%s) does not support block solver\n", precision);
    if(!serial) {
      if(mapType != 0)
        printf("Precision level (%s) does not support mapType %d\n", precision, mapType);
      if(integrationType != 0)
        printf("Precision level (%s) does not support integrationType %d\n", precision, integrationType);
    }
    ABORT(EXIT_FAILURE);
  }

  occa::memory & o_geom_factors = elliptic->stressForm ? mesh->o_vgeo : mesh->o_ggeo;
  occa::memory & o_ggeo = (precisionStr == pFloatStr) ? mesh->o_ggeoPfloat : mesh->o_ggeo;
  occa::memory & o_D = (precisionStr == pFloatStr) ? mesh->o_DPfloat : mesh->o_D;
  occa::memory & o_DT = (precisionStr == pFloatStr) ? mesh->o_DTPfloat : mesh->o_DT;
  occa::kernel & AxKernel = (precisionStr == pFloatStr) ? elliptic->AxPfloatKernel : elliptic->AxKernel;
  occa::kernel &partialAxKernel =
      (precisionStr == pFloatStr) ? elliptic->partialAxPfloatKernel : elliptic->partialAxKernel;

  if(serial) {
    if(elliptic->var_coeff) {
      if(elliptic->blockSolver) {
        elliptic->AxKernel(mesh->Nelements, elliptic->Ntotal, elliptic->loffset, o_geom_factors,
                           mesh->o_D, mesh->o_DT, elliptic->o_lambda,
                           o_q, o_Aq);
      }else {
        elliptic->AxKernel(mesh->Nelements, elliptic->Ntotal, mesh->o_ggeo, mesh->o_D,
                           mesh->o_DT, elliptic->o_lambda, o_q, o_Aq);
      }
    }else{
      const dfloat lambda = elliptic->lambda[0];
      if(elliptic->blockSolver) {
        elliptic->AxKernel(mesh->Nelements, elliptic->Ntotal, elliptic->loffset, o_geom_factors,
                           mesh->o_D, mesh->o_DT, elliptic->o_lambda,
                           o_q, o_Aq);
      }else {
        AxKernel(mesh->Nelements, o_ggeo, o_D, o_DT, elliptic->lambda[0],
                 o_q, o_Aq);
      }
    }
    return;
  }

  if(elliptic->var_coeff) {
    if(elliptic->blockSolver) {
      partialAxKernel(NelementsList,
                      elliptic->Ntotal,
                      elliptic->loffset,
                      o_elementsList,
                      o_geom_factors,
                      mesh->o_D,
                      mesh->o_DT,
                      elliptic->o_lambda,
                      o_q,
                      o_Aq);
    }else {
      partialAxKernel(NelementsList,
                      elliptic->Ntotal,
                      o_elementsList,
                      mesh->o_ggeo,
                      mesh->o_D,
                      mesh->o_DT,
                      elliptic->o_lambda,
                      o_q,
                      o_Aq);
    }
  }else{
    if(elliptic->blockSolver) {
      partialAxKernel(NelementsList,
                      elliptic->Ntotal,
                      elliptic->loffset,
                      o_elementsList,
                      o_geom_factors,
                      mesh->o_D,
                      mesh->o_DT,
                      elliptic->o_lambda,
                      o_q,
                      o_Aq);
    }else {
      partialAxKernel(NelementsList,
                      o_elementsList,
                      o_ggeo,
                      o_D,
                      o_DT,
                      elliptic->lambda[0],
                      o_q,
                      o_Aq);
    }
  }
}

void ellipticOperator(elliptic_t* elliptic,
                      occa::memory &o_q,
                      occa::memory &o_Aq,
                      const char* precision)
{
  mesh_t* mesh = elliptic->mesh;
  setupAide &options = elliptic->options;
  oogs_t* oogsAx = elliptic->oogsAx;
  const char* ogsDataTypeString = (!strstr(precision, dfloatString)) ?
                                  options.compareArgs("ENABLE FLOATCOMMHALF GS SUPPORT",
                                                      "TRUE") ? ogsFloatCommHalf : ogsPfloat
    :
                                  ogsDfloat;
  const bool serial = useSerial();
  if(serial) {
    occa::memory o_dummy;
    ellipticAx(elliptic, mesh->Nelements, o_dummy, o_q, o_Aq, precision);
    oogs::startFinish(o_Aq, elliptic->Nfields, elliptic->Ntotal, ogsDataTypeString, ogsAdd, oogsAx);
  } else {
    ellipticAx(elliptic, mesh->NglobalGatherElements, mesh->o_globalGatherElementList, o_q, o_Aq, precision);
    oogs::start(o_Aq, elliptic->Nfields, elliptic->Ntotal, ogsDataTypeString, ogsAdd, oogsAx);
    ellipticAx(elliptic, mesh->NlocalGatherElements, mesh->o_localGatherElementList, o_q, o_Aq, precision);
    oogs::finish(o_Aq, elliptic->Nfields, elliptic->Ntotal, ogsDataTypeString, ogsAdd, oogsAx);
  }
  occa::kernel &maskKernel = (!strstr(precision, dfloatString)) ? mesh->maskPfloatKernel : mesh->maskKernel;
  if (elliptic->Nmasked) maskKernel(elliptic->Nmasked, elliptic->o_maskIds, o_Aq);
}
