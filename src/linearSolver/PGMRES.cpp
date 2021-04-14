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

GmresData::GmresData(elliptic_t* elliptic, occa::properties& kernelInfo)
: restart(
    [&](){
      int _restart = 15;
      elliptic->options.getArgs("PGMRES RESTART", _restart);
      return _restart;
    }()
  ),
  flexible(
    [&](){
      if(elliptic->options.compareArgs("KRYLOV SOLVER", "FLEXIBLE"))
        return 1;
      return 0;
    }()
  ),
  o_V(elliptic->Nfields * elliptic->Ntotal, restart, sizeof(dfloat),elliptic->mesh->rank, elliptic->mesh->device),
  o_Z(flexible * elliptic->Nfields * elliptic->Ntotal, flexible * restart, sizeof(dfloat),elliptic->mesh->rank, elliptic->mesh->device)
{
  y = (dfloat *) calloc(restart, sizeof(dfloat));
  o_y = elliptic->mesh->device.malloc(restart*sizeof(dfloat), y);
  H = (dfloat *) calloc((restart+1)*(restart+1), sizeof(dfloat));
  sn = (dfloat *) calloc(restart, sizeof(dfloat));
  cs = (dfloat *) calloc(restart, sizeof(dfloat));
  s = (dfloat *) calloc(restart+1, sizeof(dfloat));
  const dlong Nlocal = elliptic->mesh->Nelements * elliptic->mesh->Np;
  int Nblock = elliptic->Nblock;
  const dlong Nbytes = restart * Nblock * sizeof(dfloat);
  //pinned scratch buffer
  {
    occa::properties props = kernelInfo;
    props["mapped"] = true; // props["host"] = true;
    h_scratch = elliptic->mesh->device.malloc(Nbytes, props);
    scratch = (dfloat*) h_scratch.ptr(props); // h_scratch.ptr()
  }
  o_scratch = elliptic->mesh->device.malloc(Nbytes);
}

void initializeGmresData(elliptic_t* elliptic, occa::properties& kernelInfo)
{
  GmresData* gmresData = new GmresData(elliptic, kernelInfo);
  elliptic->gmresData = gmresData;
}

namespace{
void gmresUpdate(elliptic_t* elliptic,
                 occa::memory o_x,
                 int I){
  const int restart = elliptic->gmresData->restart;
  mesh_t* mesh = elliptic->mesh;
  dfloat* y = elliptic->gmresData->y;
  dfloat* H = elliptic->gmresData->H;
  dfloat* s = elliptic->gmresData->s;
  deviceVector_t& o_V = elliptic->gmresData->o_V;
  deviceVector_t& o_Z = elliptic->gmresData->o_Z;
  occa::memory& o_y = elliptic->gmresData->o_y;
  occa::memory& o_z = elliptic->o_z;
  occa::memory& o_tmp = elliptic->o_p;
  const dlong Nlocal = elliptic->mesh->Nelements * elliptic->mesh->Np;

  for(int k=I-1; k>=0; --k){
    y[k] = s[k];

    for(int m=k+1; m<I; ++m)
      y[k] -= H[k + m*(restart+1)]*y[m];

    y[k] /= H[k + k*(restart+1)];
  }

  o_y.copyFrom(y, I * sizeof(dfloat));

  if(elliptic->options.compareArgs("KRYLOV SOLVER", "FLEXIBLE")){
    elliptic->updatePGMRESSolutionKernel(
      Nlocal,
      elliptic->Ntotal,
      I,
      o_y,
      o_Z,
      o_x
    );
  } else {
    elliptic->fillKernel(elliptic->Nfields * elliptic->Ntotal, 0.0, o_z);
    elliptic->updatePGMRESSolutionKernel(
      Nlocal,
      elliptic->Ntotal,
      I,
      o_y,
      o_V,
      o_z
    );
  
    ellipticPreconditioner(elliptic, o_z, o_tmp);
    ellipticScaledAdd(elliptic, 1.0, o_tmp, 1.0, o_x);
  }
}
}

// Ax=r
int pgmres(elliptic_t* elliptic, occa::memory &o_r, occa::memory &o_x,
        const dfloat tol, const int MAXIT)
{

  mesh_t* mesh = elliptic->mesh;

  occa::memory& o_w = elliptic->o_p;
  occa::memory& o_b = elliptic->o_rtmp;

  occa::memory& o_z = elliptic->o_z;
  occa::memory& o_Ax = elliptic->o_Ap;
  occa::memory& o_weight = elliptic->o_invDegree;
  deviceVector_t& o_V = elliptic->gmresData->o_V;
  deviceVector_t& o_Z = elliptic->gmresData->o_Z;

  occa::memory& o_y = elliptic->gmresData->o_y;

  dfloat* y = elliptic->gmresData->y;
  dfloat* H = elliptic->gmresData->H;
  dfloat* sn = elliptic->gmresData->sn;
  dfloat* cs = elliptic->gmresData->cs;
  dfloat* s = elliptic->gmresData->s;

  const int restart = elliptic->gmresData->restart;

  const int flexible = elliptic->options.compareArgs("KRYLOV SOLVER", "FLEXIBLE");

  const bool verbose = elliptic->options.compareArgs("VERBOSE", "TRUE");
  const bool serial = mesh->device.mode() == "Serial" || mesh->device.mode() == "OpenMP";
  const dlong Nlocal = elliptic->mesh->Nelements * elliptic->mesh->Np;

  //int Nblock = (Nlocal+BLOCKSIZE-1)/BLOCKSIZE;
  //const dlong Nbytes = Nblock * sizeof(dfloat);

  dfloat rdotr = ellipticWeightedNorm2(elliptic, o_weight, o_r) * elliptic->resNormFactor;
  rdotr = sqrt(rdotr);

  dfloat nr = rdotr / sqrt(elliptic->resNormFactor);
  dfloat error = rdotr;
  const dfloat TOL = tol;

  if (verbose&&(mesh->rank==0))
    printf("PGMRES: initial res norm %12.12f \n", rdotr);

  int iter=0;

  for(iter=0;iter<MAXIT;){

    s[0] = nr;

    // V(:,0) = r/nr
    ellipticScaledAdd(elliptic, 1.0 / nr, o_r, 0.0, o_V);

    //Construct orthonormal basis via Gram-Schmidt
    for(int i=0;i<restart;++i){

      occa::memory& o_Mv = flexible ? o_Z.at(i) : o_z;
      // z := M^{-1} V(:,i)
      ellipticPreconditioner(elliptic, o_V.at(i), o_Mv);

      // w := A z
      ellipticOperator(elliptic, o_Mv, o_w, dfloatString);

      elliptic->multiWeightedInnerProduct2Kernel(
        Nlocal,
        elliptic->Ntotal, 
	elliptic->Nblock,
        (i+1),
        0, 
	elliptic->o_invDegree, 
	o_V, 
	o_w, 
	elliptic->gmresData->o_scratch);

      elliptic->gmresData->o_scratch.copyTo(
        elliptic->gmresData->scratch,
        sizeof(dfloat) * (i+1) * elliptic->Nblock);

      for(dlong k = 0; k < i+1; ++k) {
        dfloat accum = 0.0;
        for(dlong n = 0; n < elliptic->Nblock; ++n)
          accum += elliptic->gmresData->scratch[n + k * elliptic->Nblock];
        y[k] = accum;
      }

      MPI_Allreduce(MPI_IN_PLACE, y, i+1, MPI_DFLOAT, MPI_SUM, elliptic->mesh->comm);

      for(int k = 0 ; k <=i; ++k)
        H[k + i*(restart+1)] = y[k];
      o_y.copyFrom(y, (i+1)*sizeof(dfloat));
      elliptic->gramSchmidtOrthogonalizationKernel(
        elliptic->Nblock,
        Nlocal,
        elliptic->Ntotal,
        (i+1),
        o_weight,
        o_y,
        o_V,
        o_w,
        elliptic->gmresData->o_scratch);
      dfloat nw = 0.0;
      elliptic->gmresData->o_scratch.copyTo(
        elliptic->gmresData->scratch,
        sizeof(dfloat) * elliptic->Nblock);
      for(int k = 0; k < elliptic->Nblock; ++k)
        nw += elliptic->gmresData->scratch[k];
      MPI_Allreduce(MPI_IN_PLACE, &nw, 1, MPI_DFLOAT, MPI_SUM, elliptic->mesh->comm);
      nw = sqrt(nw);

      // H(i+1,i) = ||w||_2
      H[i+1 + i*(restart+1)] = nw;

      // V(:,i+1) = w/nw
      if (i<restart-1)
        ellipticScaledAdd(elliptic, 1.0 / nw, o_w, 0.0, o_V.at(i+1));

      //apply Givens rotation
      for(int k=0; k<i; ++k){
        const dfloat h1 = H[k +     i*(restart+1)];
        const dfloat h2 = H[k + 1 + i*(restart+1)];

        H[k +     i*(restart+1)] =  cs[k]*h1 + sn[k]*h2;
        H[k + 1 + i*(restart+1)] = -sn[k]*h1 + cs[k]*h2;
      }

      // form i-th rotation matrix
      const dfloat h1 = H[i+    i*(restart+1)];
      const dfloat h2 = H[i+1 + i*(restart+1)];
      const dfloat hr = sqrt(h1*h1 + h2*h2);
      cs[i] = h1/hr;
      sn[i] = h2/hr;

      H[i   + i*(restart+1)] = cs[i]*h1 + sn[i]*h2;
      H[i+1 + i*(restart+1)] = 0;

      //approximate residual norm
      s[i+1] = -sn[i]*s[i];
      s[i]   =  cs[i]*s[i];

      iter++;
      error = fabs(s[i+1]) * sqrt(elliptic->resNormFactor);
      rdotr = error;

      if (verbose&&(mesh->rank==0)) {
        printf("GMRES: it %d, approx residual norm %12.12le \n", iter, error);
      }

      if(error < TOL || iter==MAXIT) {
        //update approximation
        gmresUpdate(elliptic, o_x, i+1);
        break;
      }
    }

    //exit if tolerance is reached
    if(error < TOL || iter==MAXIT) break;

    //update approximation
    gmresUpdate(elliptic, o_x, restart);

    // restart GMRES
    // compute A*x
    ellipticOperator(elliptic, o_x, o_Ax, dfloatString);

    elliptic->fusedResidualAndNormKernel(
      elliptic->Nblock,
      Nlocal,
      elliptic->Ntotal,
      elliptic->o_invDegree,
      o_b,
      o_Ax,
      o_r,
      elliptic->gmresData->o_scratch
    );

    if(serial){
      nr = *((dfloat*) elliptic->gmresData->o_scratch.ptr());
    } else {
      nr = 0.0;
      elliptic->gmresData->o_scratch.copyTo(elliptic->gmresData->scratch, elliptic->Nblock * sizeof(dfloat));
      for(dlong n=0;n<elliptic->Nblock;++n)
        nr += elliptic->gmresData->scratch[n];
    }

    MPI_Allreduce(MPI_IN_PLACE, &nr, 1, MPI_DFLOAT, MPI_SUM, mesh->comm);
    nr = sqrt(nr);

    error = nr * sqrt(elliptic->resNormFactor);
    rdotr = nr * sqrt(elliptic->resNormFactor);
    //exit if tolerance is reached
    if(error<=TOL) return iter;
  }

  return iter;
}
