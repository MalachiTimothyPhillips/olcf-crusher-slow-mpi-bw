#include "nrssys.hpp"
#include "platform.hpp"
#include "gslib.h"
#include "elliptic.h"
#include "ellipticBuildSEMFEM.hpp"
#include "hypreWrapper.hpp"
#include "hypreParamIndex.hpp"
#include "amgx.h"

namespace{
occa::kernel gatherKernel;
occa::kernel scatterKernel;
occa::memory o_dofMap;
occa::memory o_SEMFEMBuffer1;
occa::memory o_SEMFEMBuffer2;
void* SEMFEMBuffer1_h_d;
void* SEMFEMBuffer2_h_d;
dlong numRowsSEMFEM;
}

void ellipticSEMFEMSetup(elliptic_t* elliptic)
{
  const int verbose = (platform->options.compareArgs("VERBOSE","TRUE")) ? 1: 0;
  const int useFP32 = elliptic->options.compareArgs("COARSE SOLVER PRECISION", "FP32");
  const bool useDevice = elliptic->options.compareArgs("COARSE SOLVER LOCATION", "DEVICE");

  gatherKernel = platform->kernels.get("gather");
  scatterKernel = platform->kernels.get("scatter");

  MPI_Barrier(platform->comm.mpiComm);
  double tStart = MPI_Wtime();
  if(platform->comm.mpiRank == 0)  printf("setup SEMFEM preconditioner ... \n"); fflush(stdout);

  mesh_t* mesh = elliptic->mesh;
  double* mask = (double*) malloc(mesh->Np*mesh->Nelements*sizeof(double));
  for(int i = 0; i < mesh->Np*mesh->Nelements; ++i) mask[i] = 1.0;
  if(elliptic->Nmasked > 0){
    dlong* maskIds = (dlong*) calloc(elliptic->Nmasked, sizeof(dlong));
    elliptic->o_maskIds.copyTo(maskIds, elliptic->Nmasked * sizeof(dlong));
    for (dlong i = 0; i < elliptic->Nmasked; i++) mask[maskIds[i]] = 0.;
    free(maskIds);
  }
 
  SEMFEMData* data = ellipticBuildSEMFEM(
    mesh->Nq,
    mesh->Nelements,
    mesh->o_x,
    mesh->o_y,
    mesh->o_z,
    mask,
    platform->comm.mpiComm,
    mesh->globalIds
  );

  const dlong numRows = data->rowEnd - data->rowStart + 1;
  numRowsSEMFEM = numRows;

  o_dofMap = platform->device.malloc(numRows * sizeof(long long), data->dofMap);

  o_SEMFEMBuffer1 = platform->device.malloc(numRows * sizeof(pfloat));
  o_SEMFEMBuffer2 = platform->device.malloc(numRows * sizeof(pfloat));
  if(!useDevice){
    SEMFEMBuffer1_h_d = (pfloat*) calloc(numRows, sizeof(pfloat));
    SEMFEMBuffer2_h_d = (pfloat*) calloc(numRows, sizeof(pfloat));
  }

  int setupRetVal = 0;
  if(elliptic->options.compareArgs("COARSE SOLVER", "BOOMERAMG")){
      auto settings = boomerAMGSettingsFromOptions(elliptic->options);

      using namespace hypreParamIndex;
      if(elliptic->options.compareArgs("MULTIGRID SEMFEM", "TRUE")) {
        settings[CRS_SMOOTHER]  = 16;
        settings[SMOOTHER]  = 16;
      }

      if(platform->device.mode() != "Serial" && useDevice) {
        setupRetVal = hypreWrapperDevice::BoomerAMGSetup(
                        numRows,
                        data->nnz,
                        data->Ai,
                        data->Aj,
                        data->Av,
                        (int) elliptic->allNeumann,
                        platform->comm.mpiComm,
                        platform->device.occaDevice(),
                        useFP32,
                        settings.data(),
                        verbose);
      } else {
        setupRetVal = hypreWrapper::BoomerAMGSetup(
          numRows,
          data->nnz,
          data->Ai,
          data->Aj,
          data->Av,
          (int) elliptic->allNeumann,
          platform->comm.mpiComm,
          1, /* Nthreads */
          useFP32,
          settings.data(),
          verbose 
        );
      }
  }
  else if(elliptic->options.compareArgs("COARSE SOLVER", "AMGX")){
    if(platform->device.mode() != "CUDA") {
      if(platform->comm.mpiRank == 0) printf("AmgX only supports CUDA!\n");
      ABORT(1);
    } 
      
    std::string configFile;
    elliptic->options.getArgs("AMGX CONFIG FILE", configFile);
    char *cfg = NULL;
    if(configFile.size()) cfg = (char*) configFile.c_str();
    setupRetVal = AMGXsetup(
      numRows,
      data->nnz,
      data->Ai,
      data->Aj,
      data->Av,
      (int) elliptic->allNeumann,
      platform->comm.mpiComm,
      platform->device.id(),
      useFP32,
      std::stoi(getenv("NEKRS_GPU_MPI")),
      cfg);
  }
  else {
    if(platform->comm.mpiRank == 0){
      std::string amgSolver;
      elliptic->options.getArgs("COARSE SOLVER", amgSolver);
      printf("COARSE SOLVER %s is not supported!\n", amgSolver.c_str());
    }
    ABORT(EXIT_FAILURE);
  }

  if(setupRetVal) {
   if(platform->comm.mpiRank == 0)
     printf("AMG solver setup failed!\n");
   ABORT(1);
  }

  free(data);
  if(platform->comm.mpiRank == 0)  printf("done (%gs)\n", MPI_Wtime() - tStart); fflush(stdout);
}

void ellipticSEMFEMSolve(elliptic_t* elliptic, occa::memory& o_r, occa::memory& o_z)
{
  mesh_t* mesh = elliptic->mesh;

  const bool useDevice = elliptic->options.compareArgs("COARSE SOLVER LOCATION", "DEVICE");

  occa::memory& o_bufr = o_SEMFEMBuffer1;
  occa::memory& o_bufz = o_SEMFEMBuffer2;

  // E->T
  gatherKernel(
    numRowsSEMFEM,
    o_dofMap,
    o_r,
    o_bufr
  );

  if(elliptic->options.compareArgs("COARSE SOLVER", "BOOMERAMG")){

    if(!useDevice)
    {
      o_bufr.copyTo(SEMFEMBuffer1_h_d, numRowsSEMFEM * sizeof(pfloat));
      hypreWrapper::BoomerAMGSolve(SEMFEMBuffer1_h_d, SEMFEMBuffer2_h_d);
      o_bufz.copyFrom(SEMFEMBuffer2_h_d, numRowsSEMFEM * sizeof(pfloat));

    } else {
      hypreWrapperDevice::BoomerAMGSolve(o_bufr, o_bufz);
    }

  } else if(elliptic->options.compareArgs("COARSE SOLVER", "AMGX") && useDevice){

    AMGXsolve(o_bufr.ptr(), o_bufz.ptr());

  } else {

    if(platform->comm.mpiRank == 0)
      printf("Trying to call an unknown SEMFEM solver!\n");
    ABORT(1);

  }

  // T->E
  scatterKernel(
    numRowsSEMFEM,
    o_dofMap,
    o_bufz,
    o_z
  );

  oogs::startFinish(o_z, 1, 0, ogsPfloat, ogsAdd, elliptic->oogs);
}
