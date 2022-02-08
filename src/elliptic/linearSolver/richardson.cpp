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
#include "timer.hpp"
#include "linAlg.hpp"
#include "observer.hpp"

int richardson(elliptic_t *elliptic,
               occa::memory &o_r,
               occa::memory &o_x,
               const dfloat tol,
               const int MAXIT,
               dfloat &rdotr)
{

  mesh_t *mesh = elliptic->mesh;
  setupAide options = elliptic->options;

  const int verbose = options.compareArgs("VERBOSE", "TRUE");

  /*aux variables */
  occa::memory &o_b = elliptic->o_p;
  occa::memory &o_z = elliptic->o_z;
  occa::memory &o_Ax = elliptic->o_Ap;
  occa::memory &o_weight = elliptic->o_invDegree;
  o_b.copyFrom(o_r, elliptic->Ntotal * elliptic->Nfields * sizeof(dfloat));

  if (platform->comm.mpiRank == 0 && verbose) {
    printf("Richardson %s: initial res norm %.15e WE NEED TO GET TO %e \n",
           elliptic->name.c_str(),
           rdotr,
           tol);
  }

  int iter = 0;
  do {
    iter++;

    // M^{-1} (b - A x)
    ellipticPreconditioner(elliptic, o_r, o_z);

    preconditionerO

        // x_{k+1} = x_k + M^{-1} (b - A x)
        platform->linAlg->axpbyMany(mesh->Nlocal, elliptic->Nfields, elliptic->Ntotal, 1.0, o_z, 1.0, o_x);

    // r_{k+1} = b - A x_{k+1}
    ellipticOperator(elliptic, o_x, o_Ax, dfloatString);
    platform->linAlg
        ->axpbyzMany(mesh->Nlocal, elliptic->Nfields, elliptic->Ntotal, -1.0, o_Ax, 1.0, o_b, o_r);

    rdotr = platform->linAlg->weightedNorm2Many(mesh->Nlocal,
                                                elliptic->Nfields,
                                                elliptic->Ntotal,
                                                o_weight,
                                                o_r,
                                                platform->comm.mpiComm);

    rdotr *= sqrt(elliptic->resNormFactor);

    if (verbose && (platform->comm.mpiRank == 0))
      printf("it %d r norm %.15e\n", iter, rdotr);
  } while (rdotr > tol && iter < MAXIT);

  return iter;
}
