/*

   The MIT License (MIT)

   Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

 */

#include "elliptic.h"
#include "linAlg.hpp"

void ellipticUpdateJacobi(elliptic_t *elliptic, occa::memory &o_invDiagA)
{
  mesh_t *mesh = elliptic->mesh;
  setupAide options = elliptic->options;

  const dlong Nlocal = mesh->Np * mesh->Nelements;

  elliptic->updateDiagonalKernel(mesh->Nelements,
                                 elliptic->Nfields,
                                 elliptic->Ntotal,
                                 elliptic->loffset,
                                 elliptic->o_mapB,
                                 mesh->o_ggeo,
                                 mesh->o_D,
                                 mesh->o_DT,
                                 elliptic->o_lambda,
                                 o_invDiagA);

  oogs::startFinish(o_invDiagA,
                    elliptic->Nfields,
                    elliptic->Ntotal,
                    ogsPfloat,
                    ogsAdd,
                    elliptic->oogs);

  const pfloat one = 1.0;
  elliptic->adyManyPfloatKernel(Nlocal,
                                elliptic->Nfields,
                                elliptic->Ntotal,
                                one,
                                o_invDiagA);
}
