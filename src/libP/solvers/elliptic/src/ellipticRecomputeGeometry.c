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

void recomputeGeometry(elliptic_t* elliptic)
{
  setupAide &options = elliptic->options;

  int continuous = options.compareArgs("DISCRETIZATION", "CONTINUOUS");
  int serial = options.compareArgs("THREAD MODEL", "SERIAL");

  mesh_t* mesh = elliptic->mesh;
  dfloat* tmp = elliptic->tmp;
  dlong Nblock = elliptic->Nblock;
  dlong Nblock2 = elliptic->Nblock2;

  const dlong Nlocal = mesh->Np * mesh->Nelements;

  occa::memory &o_tmp = elliptic->o_tmp;
  occa::memory &o_tmp2 = elliptic->o_tmp2;
  occa::memory &o_wrk = elliptic->o_wrk;

  dfloat volume;

  // recompute geometric factors
  mesh->meshGeometricFactorsKernel(mesh->Nelements,
    1, // 1 for second-order geometric factors
    mesh->o_D,
    mesh->o_gllw,
    mesh->o_x,
    mesh->o_y,
    mesh->o_z,
    mesh->o_ggeo,
    o_wrk);

  // reduction on element volume
  mesh->sumKernel(Nlocal, o_wrk, o_tmp);

  if(serial && continuous) {
    o_tmp.copyTo(&volume, sizeof(dfloat));
  }else {
    /* add a second sweep if Nblock>Ncutoff */
    dlong Ncutoff = 100;
    dlong Nfinal;
    if(Nblock > Ncutoff) {
      mesh->sumKernel(Nblock, o_tmp, o_tmp2);
      o_tmp2.copyTo(tmp);
      Nfinal = Nblock2;
    }else  {
      o_tmp.copyTo(tmp);
      Nfinal = Nblock;
    }

    volume = 0;
    for(dlong n = 0; n < Nfinal; ++n)
      volume += tmp[n];

  }

  dfloat globalVolume = 0;
  MPI_Allreduce(&volume, &globalVolume, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
  mesh->volume = globalVolume;
}
