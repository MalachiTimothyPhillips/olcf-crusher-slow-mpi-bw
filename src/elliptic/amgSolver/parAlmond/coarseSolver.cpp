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
#include "elliptic.h"

static occa::kernel vectorDotStarKernel;

namespace parAlmond {

coarseSolver::coarseSolver(setupAide options_, MPI_Comm comm_) {
  options = options_;
  comm = comm_;
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

  const std::string kernelName = "vectorDotStar";
  vectorDotStarKernel = platform->kernels.get(kernelName);

  N = (int) Nrows;

  o_xBuffer = platform->device.malloc(Nrows * sizeof(pfloat));
  h_xBuffer = platform->device.mallocHost(Nrows * sizeof(pfloat));
  xBuffer = (pfloat*) h_xBuffer.ptr(); 

  if (options.compareArgs("COARSE SOLVER", "BOOMERAMG")){

    auto settings = boomerAMGSettingsFromOptions(options);
 
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
        settings.data(),
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
        settings.data(),
        verbose);
    }
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

void coarseSolver::solve(occa::memory o_rhs, occa::memory o_x) 
{
  platform->timer.tic("coarseSolve", 1);

  if(options.compareArgs("MULTIGRID SEMFEM", "TRUE")){

    semfemSolver(o_rhs, o_x);

  } else {

    const bool useDevice = options.compareArgs("COARSE SOLVER LOCATION", "DEVICE");

    const pfloat zero = 0.0;
    platform->linAlg->pfill(N, zero, o_xBuffer);
    if(!useDevice) o_xBuffer.copyTo(xBuffer, N*sizeof(pfloat));

    // T->L
    const pfloat one = 1.0;
    vectorDotStarKernel(ogs->N, one, zero, o_weight, o_rhs, o_Sx); 
    ogsGather(o_Gx, o_Sx, ogsPfloat, ogsAdd, ogs);
    if(!useDevice) o_Gx.copyTo(Gx, N*sizeof(pfloat));

    if (options.compareArgs("COARSE SOLVER", "BOOMERAMG")){
      if(useDevice)
        hypreWrapperDevice::BoomerAMGSolve(o_Gx, o_xBuffer);
      else
        hypreWrapper::BoomerAMGSolve(Gx, xBuffer); 
    } else if (options.compareArgs("COARSE SOLVER", "AMGX")){
        AMGXsolve(o_Gx.ptr(), o_xBuffer.ptr());
    }

    // T->E
    if(useDevice) {
      ogsScatter(o_x, o_xBuffer, ogsPfloat, ogsAdd, ogs);
    } else {
      o_Gx.copyFrom(xBuffer, N*sizeof(pfloat));
      ogsScatter(o_x, o_Gx, ogsPfloat, ogsAdd, ogs);
    }

  }

  platform->timer.toc("coarseSolve");
}

} //namespace parAlmond
