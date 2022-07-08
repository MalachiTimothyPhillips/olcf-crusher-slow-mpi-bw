/*
    
The MIT License (MIT)
      
        
Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus, Rajesh Gandham

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

#include "omp.h"
#include "limits.h"
#include "stdio.h"
#include "parAlmond.hpp"

#include "timer.hpp"

#include "hypreWrapper.hpp"
#include "amgx.h"

#include "platform.hpp"
#include "linAlg.hpp"


namespace {
  static occa::kernel convertFP64ToFP32Kernel;
  static occa::kernel convertFP32ToFP64Kernel;
  static occa::memory o_rhsBuffer;
  static occa::memory o_xBuffer;
  static pfloat *xBuffer;
  static pfloat *rhsBuffer;
}

namespace parAlmond {

coarseSolver::coarseSolver(setupAide options_, MPI_Comm comm_) {
  gatherLevel = false;
  options = options_;
  comm = comm_;
}

int coarseSolver::getTargetSize() {
  return 1000;
}

void coarseSolver::setup(
               dlong Nrows, 
      	       hlong* globalRowStarts,       //global partition
               dlong nnz,                    //--
               hlong* Ai,                    //-- Local A matrix data (globally indexed, COO storage, row sorted)
               hlong* Aj,                    //--
               dfloat* Avals,                //--
               bool nullSpace)
{
  int rank, size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  const int verbose = (options.compareArgs("VERBOSE","TRUE")) ? 1: 0;
  const bool useDevice = options.compareArgs("COARSE SOLVER LOCATION", "DEVICE");
  const int useFP32 = options.compareArgs("COARSE SOLVER PRECISION", "FP32");


  if(options.compareArgs("PARALMOND SMOOTH COARSEST", "TRUE"))
    return; // bail early as this will not get used

  {
    std::string kernelName = "convertFP64ToFP32";
    convertFP64ToFP32Kernel = platform->kernels.get(kernelName);

    kernelName = "convertFP32ToFP64";
    convertFP32ToFP64Kernel = platform->kernels.get(kernelName);

    kernelName = "vectorDotStar2";
    vectorDotStarKernel2 = platform->kernels.get(kernelName);
  }

  o_rhsBuffer = platform->device.malloc(Nrows * sizeof(pfloat));
  rhsBuffer = (pfloat*) calloc(Nrows, sizeof(pfloat));
  o_xBuffer = platform->device.malloc(Nrows * sizeof(pfloat));
  xBuffer = (pfloat*) calloc(Nrows, sizeof(pfloat));

  if (options.compareArgs("COARSE SOLVER", "BOOMERAMG")){
 
    double settings[BOOMERAMG_NPARAM+1];
    settings[0]  = 1;    /* custom settings              */
    settings[1]  = 8;    /* coarsening                   */
    settings[2]  = 6;    /* interpolation                */
    settings[3]  = 1;    /* number of cycles             */
    settings[4]  = 16;   /* smoother for crs level       */
    settings[5]  = 3;    /* number of coarse sweeps      */
    settings[6]  = 16;   /* smoother                     */
    settings[7]  = 1;    /* number of sweeps             */
    settings[8]  = 0.25; /* strong threshold             */
    settings[9]  = 0.05; /* non galerkin tol             */
    settings[10] = 0;    /* aggressive coarsening levels */

    options.getArgs("BOOMERAMG COARSEN TYPE", settings[1]);
    options.getArgs("BOOMERAMG INTERPOLATION TYPE", settings[2]);
    options.getArgs("BOOMERAMG COARSE SMOOTHER TYPE", settings[4]);
    options.getArgs("BOOMERAMG SMOOTHER TYPE", settings[6]);
    options.getArgs("BOOMERAMG SMOOTHER SWEEPS", settings[7]);
    options.getArgs("BOOMERAMG ITERATIONS", settings[3]);
    options.getArgs("BOOMERAMG STRONG THRESHOLD", settings[8]);
    options.getArgs("BOOMERAMG NONGALERKIN TOLERANCE" , settings[9]);
    options.getArgs("BOOMERAMG AGGRESSIVE COARSENING LEVELS" , settings[10]);

    if(useDevice) {
      hypreWrapperDevice::BoomerAMGSetup(
        Nrows,
        nnz,
        Ai,
        Aj,
        Avals,
        (int) nullSpace,
        comm,
        platform->device.occaDevice(),
        useFP32,
        settings,
        verbose);
    } else {
      const int Nthreads = 1;
      hypreWrapper::BoomerAMGSetup(
        Nrows,
        nnz,
        Ai,
        Aj,
        Avals,
        (int) nullSpace,
        comm,
        Nthreads,
        useFP32,
        settings,
        verbose);
    }
 
    N = (int) Nrows;
    h_xLocal   = platform->device.mallocHost(N*sizeof(dfloat));
    h_rhsLocal = platform->device.mallocHost(N*sizeof(dfloat));
    xLocal   = (dfloat*) h_xLocal.ptr();
    rhsLocal = (dfloat*) h_rhsLocal.ptr();
  }
  else if (options.compareArgs("COARSE SOLVER", "AMGX")){
    std::string configFile;
    options.getArgs("AMGX CONFIG FILE", configFile);
    char *cfg = NULL;
    if(configFile.size()) cfg = (char*) configFile.c_str();
    AMGXsetup(
      Nrows,
      nnz,
      Ai,
      Aj,
      Avals,
      (int) nullSpace,
      comm,
      platform->device.id(),
      useFP32,
      std::stoi(getenv("NEKRS_GPU_MPI")),
      cfg);
    N = (int) Nrows;
  } else {
    if(platform->comm.mpiRank == 0){
      std::string amgSolver;
      options.getArgs("COARSE SOLVER", amgSolver);
      printf("COARSE SOLVER %s is not supported!\n", amgSolver.c_str());
    }
    ABORT(EXIT_FAILURE);
  }
}

void coarseSolver::setup(parCSR *A) {

  comm = A->comm;

  int rank, size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  if (rank == 0 && options.compareArgs("VERBOSE","TRUE"))
    printf("Setting up pMG coarse solver..."); fflush(stdout);

  if (options.compareArgs("COARSE SOLVER", "BOOMERAMG") || options.compareArgs("COARSE SOLVER", "AMGX")) {
    int totalNNZ = A->diag->nnz + A->offd->nnz;
    hlong *rows;
    hlong *cols;
    dfloat *vals;
 
    if (totalNNZ) {
      rows = (hlong *) calloc(totalNNZ,sizeof(hlong));
      cols = (hlong *) calloc(totalNNZ,sizeof(hlong));
      vals = (dfloat *) calloc(totalNNZ,sizeof(dfloat));
    }
 
    // populate local COO (AIJ) matrix using global (row,col) indices
    int cnt = 0;
    for (int n = 0; n < A->Nrows; n++) {
      for (int m = A->diag->rowStarts[n]; m < A->diag->rowStarts[n+1]; m++) {
        rows[cnt] = n + A->globalRowStarts[rank];
        cols[cnt] = A->diag->cols[m] + A->globalRowStarts[rank];
        vals[cnt] = A->diag->vals[m];
        cnt++;
      }
      for (int m = A->offd->rowStarts[n]; m < A->offd->rowStarts[n+1]; m++) {
        rows[cnt] = n + A->globalRowStarts[rank];
        cols[cnt] = A->colMap[A->offd->cols[m]];
        vals[cnt] = A->offd->vals[m];
        cnt++;
      }
    }

    setup(A->Nrows, A->globalRowStarts, totalNNZ, rows, cols, vals,(int) A->nullSpace); 
 
    if (totalNNZ) {
      free(rows);
      free(cols);
      free(vals);
    }
 
    return;
  } else {
    if(rank == 0) printf("coarseSolver::setup: Cannot find valid AMG solver!\n"); fflush(stdout); 
    ABORT(1);
  }

  if(rank == 0 && options.compareArgs("VERBOSE","TRUE")) 
    printf("done.\n"); fflush(stdout);
}

void coarseSolver::syncToDevice() {}

void coarseSolver::solve(dfloat *rhs, dfloat *x) {
  if(platform->comm.mpiRank == 0) 
    printf("Trying to call invalid host coarseSolver::solve!\n"); fflush(stdout); 
  ABORT(1);
}

void coarseSolver::gather(occa::memory o_rhs, occa::memory o_x)
{
  if (gatherLevel) {
    vectorDotStar(ogs->N, 1.0, ogs->o_invDegree, o_rhs, 0.0, o_Sx);
    ogsGather(o_Gx, o_Sx, ogsDfloat, ogsAdd, ogs);
    if(N)
      o_Gx.copyTo(rhsLocal, N*sizeof(dfloat), 0);
  } else {
    if(N)
      o_rhs.copyTo(rhsLocal, N*sizeof(dfloat), 0);
  }
}
void coarseSolver::scatter(occa::memory o_rhs, occa::memory o_x)
{
  if (gatherLevel) {
    if(N)
      o_Gx.copyFrom(xLocal, N*sizeof(dfloat), 0);
    ogsScatter(o_x, o_Gx, ogsDfloat, ogsAdd, ogs);
  } else {
    if(N)
      o_x.copyFrom(xLocal, N*sizeof(dfloat), 0);
  }
}

void coarseSolver::solve(occa::memory o_rhs, occa::memory o_x) {

  platform->timer.tic("coarseSolve", 1);

  const bool useDevice = options.compareArgs("COARSE SOLVER LOCATION", "DEVICE");

  if(useSEMFEM){

    semfemSolver(o_rhs, o_x);

  } else {

    platform->linAlg->fill(N, 0.0, o_x);
    convertFP64ToFP32Kernel(N, o_x, o_xBuffer);
    if(!useDevice) o_xBuffer.copyTo(xBuffer, N*sizeof(pfloat));

    if (gatherLevel) { // E->T
      vectorDotStar(ogs->N, 1.0, ogs->o_invDegree, o_rhs, 0.0, o_Sx);
      ogsGather(o_Gx, o_Sx, ogsDfloat, ogsAdd, ogs);
    }
    occa::memory o_b = gatherLevel ? o_Gx : o_rhs;
    convertFP64ToFP32Kernel(N, o_b, o_rhsBuffer);
    if(!useDevice) o_rhsBuffer.copyTo(rhsBuffer, N*sizeof(pfloat));

    if (options.compareArgs("COARSE SOLVER", "BOOMERAMG")){
      if(useDevice)
        hypreWrapperDevice::BoomerAMGSolve(o_rhsBuffer, o_xBuffer);
      else
        hypreWrapper::BoomerAMGSolve(rhsBuffer, xBuffer); 
    } else if (options.compareArgs("COARSE SOLVER", "AMGX")){
        AMGXsolve(o_rhsBuffer.ptr(), o_xBuffer.ptr());
    }

    if(useDevice) {
      convertFP32ToFP64Kernel(N, o_xBuffer, o_x);
      if(gatherLevel)
        o_Gx.copyFrom(o_x, N*sizeof(dfloat));
    } else {
      for(int i = 0; i < N; i++) 
        xLocal[i] = (dfloat) xBuffer[i]; 
      if(gatherLevel) 
        o_Gx.copyFrom(xLocal, N*sizeof(dfloat));
    }

    // T->E
    if(gatherLevel) {
      ogsScatter(o_x, o_Gx, ogsDfloat, ogsAdd, ogs);
    } else { 
      if(!useDevice) 
        o_x.copyFrom(xLocal, N*sizeof(dfloat));
    }
  }

  platform->timer.toc("coarseSolve");
}

} //namespace parAlmond
