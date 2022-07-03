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

#include "stdio.h"
#include "parAlmond.hpp"

#include "timer.hpp"

#include "omp.h"
#include "limits.h"
#include "hypreWrapper.hpp"
#include "amgx.h"
#include "platform.hpp"

namespace {
  static occa::kernel convertFP64ToFP32Kernel;
  static occa::kernel convertFP32ToFP64Kernel;
  static occa::memory o_rhsBuffer;
  static occa::memory o_xBuffer;
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


  if (options.compareArgs("AMG SOLVER", "BOOMERAMG")){
    const bool useDevice = options.compareArgs("AMG SOLVER LOCATION", "DEVICE");
    const int useFP32 = options.compareArgs("AMG SOLVER PRECISION", "FP32");
    if(useFP32) {
      if(platform->comm.mpiRank == 0) printf("FP32 is not supported in BoomerAMG.\n");
      MPI_Barrier(platform->comm.mpiComm);
      ABORT(1);
    }
    int Nthreads = 1;
 
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
    options.getArgs("BOOMERAMG SMOOTHER TYPE", settings[6]);
    options.getArgs("BOOMERAMG SMOOTHER SWEEPS", settings[7]);
    options.getArgs("BOOMERAMG ITERATIONS", settings[3]);
    options.getArgs("BOOMERAMG STRONG THRESHOLD", settings[8]);
    options.getArgs("BOOMERAMG NONGALERKIN TOLERANCE" , settings[9]);
    options.getArgs("BOOMERAMG AGGRESSIVE COARSENING LEVELS" , settings[10]);

    if(useDevice) {
      hypreWrapperDevice::BoomerAMGSetup(Nrows,
                           nnz,
                           Ai,
                           Aj,
                           Avals,
                           (int) nullSpace,
                           comm,
                           platform->device.occaDevice(),
                           0,  /* useFP32 */
                           settings,
                           verbose);
    } else {
      hypreWrapper::BoomerAMGSetup(Nrows,
                     nnz,
                     Ai,
                     Aj,
                     Avals,
                     (int) nullSpace,
                     comm,
                     Nthreads,
                     0,  /* useFP32 */
                     settings,
                     verbose);
    }
 
    N = (int) Nrows;
    h_xLocal   = platform->device.mallocHost(N*sizeof(dfloat));
    h_rhsLocal = platform->device.mallocHost(N*sizeof(dfloat));
    xLocal   = (dfloat*) h_xLocal.ptr();
    rhsLocal = (dfloat*) h_rhsLocal.ptr();
  }
  else if (options.compareArgs("AMG SOLVER", "AMGX")){
    //TODO: these checks should go into parReader
    const int useFP32 = options.compareArgs("AMG SOLVER PRECISION", "FP32");
    if(platform->device.mode() != "CUDA") {
      if(platform->comm.mpiRank == 0) printf("AmgX only supports CUDA!\n");
      MPI_Barrier(platform->comm.mpiComm);
      ABORT(1);
    } 
    if(options.compareArgs("AMG SOLVER LOCATION", "CPU")){
      if(platform->comm.mpiRank == 0) printf("AmgX only supports DEVICE!\n");
      MPI_Barrier(platform->comm.mpiComm);
      ABORT(1);
    } 
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
    if(useFP32)
    {
      o_rhsBuffer = platform->device.malloc(N * sizeof(float));
      o_xBuffer = platform->device.malloc(N * sizeof(float));
    }
  } else {
    if(platform->comm.mpiRank == 0){
      std::string amgSolver;
      options.getArgs("AMG SOLVER", amgSolver);
      printf("AMG SOLVER %s is not supported!\n", amgSolver.c_str());
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

  if (options.compareArgs("AMG SOLVER", "BOOMERAMG") || options.compareArgs("AMG SOLVER", "AMGX")) {
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

  if(useSEMFEM){

    semfemSolver(o_rhs, o_x);

  } else {
    const bool useDevice = options.compareArgs("AMG SOLVER LOCATION", "DEVICE");
    const int useFP32 = options.compareArgs("AMG SOLVER PRECISION", "FP32");

    if (gatherLevel) {
      vectorDotStar(ogs->N, 1.0, ogs->o_invDegree, o_rhs, 0.0, o_Sx);
      ogsGather(o_Gx, o_Sx, ogsDfloat, ogsAdd, ogs);
      if(N && !useDevice) o_Gx.copyTo(rhsLocal, N*sizeof(dfloat), 0);
    } else {
      if(N && !useDevice) o_rhs.copyTo(rhsLocal, N*sizeof(dfloat), 0);
    }
    occa::memory o_b = gatherLevel ? o_Gx : o_rhs;

    if (options.compareArgs("AMG SOLVER", "BOOMERAMG")){

      if(useDevice) {
        if(useFP32){
          convertFP64ToFP32Kernel(N, o_b, o_rhsBuffer);
          hypreWrapperDevice::BoomerAMGSolve(o_xBuffer, o_rhsBuffer);
          convertFP32ToFP64Kernel(N, o_xBuffer, o_x);
        } else {
          hypreWrapperDevice::BoomerAMGSolve(o_x, o_b);
        }
      } else {
        hypreWrapper::BoomerAMGSolve(xLocal, rhsLocal); 
      }

    } else if (options.compareArgs("AMG SOLVER", "AMGX")){

      if(useFP32){
        convertFP64ToFP32Kernel(N, o_b, o_rhsBuffer);
        AMGXsolve(o_xBuffer.ptr(), o_rhsBuffer.ptr());
        convertFP32ToFP64Kernel(N, o_xBuffer, o_x);
      } else {
        AMGXsolve(o_x.ptr(), o_b.ptr());
      }

    }

    if (gatherLevel) {
      if(N && !useDevice)
        o_Gx.copyFrom(xLocal, N*sizeof(dfloat));
      if(N && useDevice)
        o_Gx.copyFrom(o_x, N*sizeof(dfloat));
        ogsScatter(o_x, o_Gx, ogsDfloat, ogsAdd, ogs);
    } else {
      if(N && !useDevice)
        o_x.copyFrom(xLocal, N*sizeof(dfloat), 0);
    }

  }

  platform->timer.toc("coarseSolve");
}

} //namespace parAlmond
