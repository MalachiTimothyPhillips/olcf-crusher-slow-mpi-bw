#include "cvodeSolver.hpp"
#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "Urst.hpp"
#include <limits>
#include <array>
#include <numeric>
#include "udf.hpp"

#include "timeStepper.hpp"
#include "plugins/lowMach.hpp"
#include "nekrs.hpp"
#include "bdry.hpp"

#ifdef ENABLE_CVODE
// cvode includes
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sundials/sundials_types.h>
#include <sundials/sundials_math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <nvector/nvector_mpiplusx.h>
#ifdef ENABLE_CUDA
#include <nvector/nvector_cuda.h>
#endif
#ifdef ENABLE_HIP
#include <nvector/nvector_hip.h>
#endif
#endif

namespace{
void computeInvLMMLMM(mesh_t* mesh, occa::memory& o_invLMMLMM)
{
  o_invLMMLMM.copyFrom(mesh->o_LMM, mesh->Nlocal * sizeof(dfloat));
  platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_invLMMLMM);
}

#ifdef ENABLE_CVODE
sunrealtype* __N_VGetDeviceArrayPointer(N_Vector u) 
{ 
  bool useDevice = false;
  useDevice |= platform->device.mode() == "CUDA";
  useDevice |= platform->device.mode() == "HIP";
  useDevice |= platform->device.mode() == "OPENCL";

  if(useDevice){
    return N_VGetDeviceArrayPointer(u);
  }
  else{
    return N_VGetArrayPointer_Serial(u);
  }
}
#endif

int check_retval(void* returnvalue, const char* funcname, int opt)
{
  int* retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int*) returnvalue;
    if (*retval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return 0;
}

}

namespace cvode {

cvodeSolver_t::cvodeSolver_t(nrs_t* nrs)
{
#ifdef ENABLE_CVODE
  auto cds = nrs->cds;

  o_coeffExt = platform->device.malloc(maxExtrapolationOrder * sizeof(dfloat));

  o_U0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (platform->options.compareArgs("MOVING MESH", "TRUE")) {
    o_meshU0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
    o_xyz0 = platform->device.malloc((nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);
  }

  if(nrs->cht){
    o_invLMMLMMT = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat));
    computeInvLMMLMM(cds->mesh[0], o_invLMMLMMT);
  }

  o_invLMMLMMV = platform->device.malloc(nrs->fieldOffset * sizeof(dfloat));
  computeInvLMMLMM(nrs->meshV, o_invLMMLMMV);

  setupEToLMapping(nrs);

  std::vector<dlong> scalarIds;
  std::vector<dlong> cvodeScalarIds(cds->NSfields, -1);

  Nscalar = 0;
  fieldOffsetScan = {0};
  fieldOffsetSum = 0;

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is]){
      continue;
    }
    if (!cds->cvodeSolve[is]){
      continue;
    }
    
    cvodeScalarIds[is] = Nscalar;
    scalarIds.push_back(is);

    fieldOffset.push_back(cds->fieldOffset[is]);
    fieldOffsetScan.push_back(fieldOffsetScan.back() + fieldOffset.back());
    fieldOffsetSum += fieldOffset.back();

    Nscalar++;

    // TODO: batch gather scatter operations as possible

    if(is == 0 && nrs->cht) continue; // gather-scatter is handled directly

    gatherScatterOperations.push_back(std::make_tuple(is, is+1, is == 0 ? cds->gshT : cds->gsh));
  }

  o_scalarIds = platform->device.malloc(scalarIds.size() * sizeof(dlong), scalarIds.data());
  o_cvodeScalarIds = platform->device.malloc(cvodeScalarIds.size() * sizeof(dlong), cvodeScalarIds.data());

  this->extrapolateInPlaceKernel = platform->kernels.get("extrapolateInPlace");
  this->mapEToLKernel = platform->kernels.get("mapEToL");
  this->mapLToEKernel = platform->kernels.get("mapLToE");
  this->packKernel = platform->kernels.get("pack");
  this->unpackKernel = platform->kernels.get("unpack");

  nEq = Nscalar * LFieldOffset;

  int retval = 0;

  int blockSize = BLOCKSIZE;

  // wrap RHS function into type expected by CVODE
  CVRhsFn cvodeRHS = [](double time, N_Vector Y, N_Vector Ydot, void* user_data)
  {

    auto data = static_cast<userData_t*>(user_data);
    auto nrs = data->nrs;
    auto platform = data->platform;
    auto cvodeSolver = data->cvodeSolver;

    occa::memory o_y = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Y)),
      cvodeSolver->numEquations());

    occa::memory o_ydot = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Ydot)),
      cvodeSolver->numEquations());
    
    cvodeSolver->rhs(nrs, time, o_y, o_ydot);

    return 0;
  };

  CVRhsFn cvodeJtvRHS = [](double time, N_Vector Y, N_Vector Ydot, void* user_data)
  {

    auto data = static_cast<userData_t*>(user_data);
    auto nrs = data->nrs;
    auto platform = data->platform;
    auto cvodeSolver = data->cvodeSolver;

    occa::memory o_y = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Y)),
      cvodeSolver->numEquations());

    occa::memory o_ydot = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Ydot)),
      cvodeSolver->numEquations());
    
    cvodeSolver->jtvRHS(nrs, time, o_y, o_ydot);

    return 0;
  };

  // same as cvLsDQJtimes, but with scaling for sig
  CVLsJacTimesVecFn cvodeLS = [](N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector work)
  {
    auto data = static_cast<userData_t*>(user_data);
    auto nrs = data->nrs;
    auto platform = data->platform;
    auto cvodeSolver = data->cvodeSolver;
    const auto sigScale = cvodeSolver->sigmaScale();

    void * cvode_mem = cvodeSolver->getCvodeMem();

    realtype sig, siginv;
    int iter, retval;

    retval = CVodeGetErrWeights(cvode_mem, work);
    if (retval != CV_SUCCESS)
      return (retval);

    /* Initialize perturbation to 1/||v|| */
    sig = sigScale / N_VWrmsNorm(v, work);

    constexpr int maxDQIters {3};

    occa::memory o_work = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(work)),
      cvodeSolver->numEquations());

    occa::memory o_Jv = platform->device.occaDevice().wrapMemory<sunrealtype>(
      __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(Jv)),
      cvodeSolver->numEquations());
    
    for (iter = 0; iter < maxDQIters; iter++) {

      /* Set work = y + sig*v */
      N_VLinearSum(sig, v, 1.0, y, work);

      /* Set Jv = f(tn, y+sig*v) */
      //retval = cvls_mem->jt_f(t, work, Jv, user_data);
      cvodeSolver->jtvRHS(nrs, t, o_work, o_Jv);
      if (retval == 0)
        break;
      if (retval < 0)
        return (-1);

      /* If f failed recoverably, shrink sig and retry */
      sig *= 0.25;
    }

    /* If retval still isn't 0, return with a recoverable failure */
    if (retval > 0)
      return (+1);

    /* Replace Jv by (Jv - fy)/sig */
    siginv = 1.0 / sig;
    N_VLinearSum(siginv, Jv, -siginv, fy, Jv);

    return (0);
  };

  SUNContext sunctx = nullptr;
  retval = SUNContext_Create((void*) &platform->comm.mpiComm, &sunctx);
  if(check_retval(&retval, "SUNContext_Create", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  {
    this->y = nullptr;
    if(platform->device.mode() == "CUDA") {
  #ifdef ENABLE_CUDA
      SUNCudaThreadDirectExecPolicy stream_exec_policy(blockSize);
      SUNCudaBlockReduceExecPolicy reduce_exec_policy(blockSize, 0);
  
      this->y = N_VNew_Cuda(this->nEq, sunctx);
      if(check_retval((void*)this->y, "N_VNew_Cuda", 0)) MPI_Abort(platform->comm.mpiComm, 1);
  
      retval = N_VSetKernelExecPolicy_Cuda(this->y, &stream_exec_policy, &reduce_exec_policy);
      if(check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0)) MPI_Abort(platform->comm.mpiComm, 1);
  #endif
    } else if(platform->device.mode() == "HIP") {
  #ifdef ENABLE_HIP
      this->y = N_VNew_Hip(data->nEq, sunctx);
      if(check_retval((void *)this->y, "N_VNew_Hip", 0)) MPI_Abort(platform->comm.mpiComm, 1);
  #endif
    } else if(platform->device.mode() == "Serial") {
      this->y = N_VNew_Serial(this->nEq, sunctx);
      if(check_retval((void *)this->y, "N_VNew_Serial", 0)) MPI_Abort(platform->comm.mpiComm, 1);
    }
    this->cvodeY = N_VMake_MPIPlusX(platform->comm.mpiComm, this->y, sunctx);
    this->nEqTotal = N_VGetLength(this->cvodeY);
  }

  o_cvodeY = platform->device.occaDevice().wrapMemory<sunrealtype>(
    __N_VGetDeviceArrayPointer(N_VGetLocalVector_MPIPlusX(cvodeY)),
    this->numEquations());

  // set initial condition
  pack(nrs, nrs->cds->o_S, o_cvodeY);

  auto integrator = CV_BDF;
  if(platform->options.compareArgs("CVODE INTEGRATOR", "ADAMS")){
    integrator = CV_ADAMS;
  }

  this->cvodeMem = CVodeCreate(integrator, sunctx);

  const auto T0 = nekrs::startTime();

  double relTol = 1e-4;
  platform->options.getArgs("CVODE RELATIVE TOLERANCE", relTol);

  double absTol = 1e-14;
  platform->options.getArgs("CVODE ABSOLUTE TOLERANCE", absTol);

  this->sigScale = 1.0;
  platform->options.getArgs("CVODE SIGMA SCALE", this->sigScale);

  if(check_retval((void*)this->cvodeMem, "CVodeCreate", 0)) MPI_Abort(platform->comm.mpiComm, 1);
  retval = CVodeInit(this->cvodeMem, cvodeRHS, T0, this->cvodeY);
  if(check_retval(&retval, "CVodeInit", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  retval = CVodeSStolerances(this->cvodeMem, relTol, absTol);
  if (check_retval(&retval, "CVodeSStolerances", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  int nvectorsGMR = 10;
  platform->options.getArgs("CVODE GMR VECTORS", nvectorsGMR);

  SUNLinearSolver LS = SUNLinSol_SPGMR(cvodeY, PREC_NONE, nvectorsGMR, sunctx);
  if(check_retval(&retval, "SUNLinSol_SPFGMR", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  retval = CVodeSetLinearSolver(this->cvodeMem, LS, NULL);
  if(check_retval(&retval, "CVodeSetLinearSolver", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  retval = CVodeSetJacTimesRhsFn(this->cvodeMem, cvodeJtvRHS);
  if(check_retval(&retval, "CVodeSetJacTimesRhsFn", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  retval = CVodeSetJacTimes(this->cvodeMem, NULL, cvodeLS);
  if(check_retval(&retval, "CVodeSetJacTimes", 1)) MPI_Abort(platform->comm.mpiComm, 1);

  // custom settings
  int mxsteps = 10000;
  platform->options.getArgs("CVODE MAX STEPS", mxsteps);
  retval = CVodeSetMaxNumSteps(this->cvodeMem, mxsteps);

  double dt0;
  platform->options.getArgs("DT", dt0);

  double hmax = 3 * dt0;
  platform->options.getArgs("CVODE HMAX", hmax);
  retval = CVodeSetMaxStep(this->cvodeMem, hmax);

  int maxOrder = 3;
  platform->options.getArgs("CVODE MAX TIMESTEPPER ORDER", maxOrder);
  retval = CVodeSetMaxOrd(this->cvodeMem, 3);

  double epsLin = 0.1;
  platform->options.getArgs("CVODE EPS LIN", epsLin);
  retval = CVodeSetEpsLin(this->cvodeMem, epsLin);

  userdata = std::make_shared<userData_t>(platform, nrs, this);

  // set user data as
  retval = CVodeSetUserData(this->cvodeMem, userdata.get());
  if(check_retval(&retval, "CVodeSetUserData", 1)) MPI_Abort(platform->comm.mpiComm, 1);

#else
  if(platform->comm.mpiRank == 0){
    std::cout << "No cvode installation found. Bailing...\n";
  }
  ABORT(1);
  
#endif
}

void cvodeSolver_t::setupEToLMapping(nrs_t *nrs)
{
  static_assert(sizeof(dlong) == sizeof(int), "dlong and int must be the same size");
  auto *mesh = nrs->meshV;
  if (mesh->cht)
    mesh = nrs->cds->mesh[0];

  // Note: these operations seem to be relatively expensive...
  auto o_Lids = platform->device.malloc(mesh->Nlocal * sizeof(dlong));
  std::vector<dlong> Eids(mesh->Nlocal);
  std::iota(Eids.begin(), Eids.end(), 0);
  o_Lids.copyFrom(Eids.data(), mesh->Nlocal * sizeof(dlong));

  {
    const auto saveNhaloGather = mesh->ogs->NhaloGather;
    mesh->ogs->NhaloGather = 0;
    ogsGatherScatter(o_Lids, "int", "ogsMin", mesh->ogs);
    mesh->ogs->NhaloGather = saveNhaloGather;
  }

  std::vector<dlong> Lids(mesh->Nlocal);
  o_Lids.copyTo(Lids.data(), mesh->Nlocal * sizeof(dlong));

  std::set<dlong> uniqueIds;
  for (auto &&id : Lids) {
    uniqueIds.insert(id);
  }

  const auto NL = uniqueIds.size();

  this->LFieldOffset = NL;

  std::vector<dlong> EToL(mesh->Nlocal);
  std::map<dlong, std::set<dlong>> LToE;
  for (auto &&Eid : Eids) {
    const auto Lid = std::distance(uniqueIds.begin(), uniqueIds.find(Eid));
    EToL[Eid] = Lid;
    LToE[Lid].insert(Eid);
  }

  std::vector<dlong> EToLUnique(mesh->Nlocal);
  std::copy(EToL.begin(), EToL.end(), EToLUnique.begin());

  // use the first Eid value for the deciding unique Lid value
  for (int eid = 0; eid < mesh->Nlocal; ++eid) {
    const auto lid = EToL[eid];
    if (*LToE[lid].begin() != eid) {
      EToLUnique[eid] = -1;
    }
  }

  this->o_EToL = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToL.data());
  this->o_EToLUnique = platform->device.malloc(mesh->Nlocal * sizeof(dlong), EToLUnique.data());

  o_Lids.free();
}

void cvodeSolver_t::rhs(nrs_t *nrs, dfloat time, occa::memory o_y, occa::memory o_ydot)
{
  if(userRHS){
    userRHS(nrs, tstep, time, tnekRS, o_y, o_ydot);
  } else {
    defaultRHS(nrs, tstep, time, tnekRS, o_y, o_ydot);
  }
}

void cvodeSolver_t::jtvRHS(nrs_t *nrs, dfloat time, occa::memory o_y, occa::memory o_ydot)
{
  this->setJacobianEvaluation();

  if(userJacobian){
    userJacobian(nrs, tstep, time, tnekRS, o_y, o_ydot);
  } else {
    this->rhs(nrs, time, o_y, o_ydot);
  }

  this->unsetJacobianEvaluation();
}

void cvodeSolver_t::defaultRHS(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)
{
  const bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  auto * cds = nrs->cds;

  if (time != tprev) {
    tprev = time;

    const auto cvodeDt = time - t0;
    dtCvode[0] = cvodeDt;
    dtCvode[1] = nrs->dt[1];
    dtCvode[2] = nrs->dt[2];

    const int bdfOrder = std::min(tstep, nrs->nBDF);
    const int extOrder = std::min(tstep, nrs->nEXT);
    nek::extCoeff(coeffEXT.data(), dtCvode.data(), extOrder, bdfOrder);
    nek::bdfCoeff(&this->g0, coeffBDF.data(), dtCvode.data(), bdfOrder);
    for (int i = nrs->nEXT; i > extOrder; i--){
      coeffEXT[i - 1] = 0.0;
    }
    for (int i = nrs->nBDF; i > bdfOrder; i--){
      coeffBDF[i - 1] = 0.0;
    }
    
    o_coeffExt.copyFrom(coeffEXT.data(), maxExtrapolationOrder * sizeof(dfloat));

    extrapolateInPlaceKernel(nrs->meshV->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, this->o_U0, nrs->o_U);

    if (movingMesh) {

      mesh->coeffs(dtCvode.data(), tstep);

      // restore mesh coordinates prior to integration
      {
        mesh->o_x.copyFrom(this->o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (0 * sizeof(dfloat)) * nrs->fieldOffset);
        mesh->o_y.copyFrom(this->o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (1 * sizeof(dfloat)) * nrs->fieldOffset);
        mesh->o_z.copyFrom(this->o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (2 * sizeof(dfloat)) * nrs->fieldOffset);
      }

      mesh->move();

      extrapolateInPlaceKernel(mesh->Nlocal, nrs->NVfields, extOrder, nrs->fieldOffset, o_coeffExt, this->o_meshU0, mesh->o_U);
    }

    computeUrst(nrs, true);

  }

  unpack(nrs, o_y, cds->o_S);

  // apply boundary condition
  applyDirichlet(nrs, time);

  evaluateProperties(nrs, time);

  // terms to include: user source, advection, filtering, add "weak" laplacian
  platform->linAlg->fillKernel(cds->fieldOffsetSum, 0.0, cds->o_FS);
  makeq(nrs, time);

  auto applyOgsOperation = [&](auto ogsFunc)
  {
    dlong startId, endId;
    oogs_t * gsh;
    for(const auto p : gatherScatterOperations){
      std::tie(startId, endId, gsh) = p;
      const auto Nfields = endId - startId;
      auto o_fld = cds->o_FS + nrs->cds->fieldOffsetScan[startId] * sizeof(dfloat);
      ogsFunc(o_fld, Nfields, nrs->cds->fieldOffset[startId], ogsDfloat, ogsAdd, gsh);
    }
  };

  applyOgsOperation(oogs::start);

  auto o_ptSource = cds->o_FS + cds->fieldOffsetSum * sizeof(dfloat);
  platform->linAlg->fill(cds->fieldOffsetSum, 0.0, o_ptSource);
  if (userLocalPointSource) {
    userLocalPointSource(nrs, cds->o_S, o_ptSource);
  }

  applyOgsOperation(oogs::finish);

  // weight by invLM
  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;

    // already applied elsewhere
    if(is == 0 && nrs->cht) continue;

    mesh_t *mesh;
    (is) ? mesh = cds->meshV : mesh = cds->mesh[0];
    const dlong isOffset = cds->fieldOffsetScan[is];

    auto o_FS_i = cds->o_FS + isOffset * sizeof(dfloat);
    platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_FS_i);

  }

  // o_FS += o_ptSource;
  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;

    mesh_t *mesh;
    (is) ? mesh = cds->meshV : mesh = cds->mesh[0];
    const dlong isOffset = cds->fieldOffsetScan[is];

    auto o_FS_i = cds->o_FS + isOffset * sizeof(dfloat);
    auto o_ptSource_i = cds->o_FS + cds->fieldOffsetSum * sizeof(dfloat) + isOffset * sizeof(dfloat);

    platform->linAlg->axpby(mesh->Nlocal, 1.0, o_ptSource_i, 1.0, o_FS_i);
  }

  if(platform->options.compareArgs("LOWMACH", "TRUE")){
    lowMach::cvodeArguments_t args{this->coeffBDF, this->g0, this->dtCvode[0]};
    platform->linAlg->fill(mesh->Nlocal, 0.0, nrs->o_div);

    lowMach::qThermalIdealGasSingleComponent(time, nrs->o_div, &args);
    const auto gamma0 = lowMach::gamma();

    // RHS += 1/vtrans * dp0thdt * (gamma-1)/gamma
    platform->o_mempool.slice0.copyFrom(cds->o_rho, mesh->Nlocal * sizeof(dfloat));
    platform->linAlg->ady(mesh->Nlocal, nrs->dp0thdt * (gamma0-1.0)/gamma0, platform->o_mempool.slice0);
    platform->linAlg->axpby(mesh->Nlocal, 1.0, platform->o_mempool.slice0, 1.0, cds->o_FS);
  }

  // mask Dirichlet portion of RHS
  auto o_zero = platform->o_mempool.slice0;
  platform->linAlg->fill(cds->mesh[0]->Nlocal, 0.0, o_zero);

  for (int is = 0; is < cds->NSfields; is++) {
    if(!cds->compute[is]) continue;
    if(!cds->cvodeSolve[is]) continue;
    occa::memory o_FSi =
        cds->o_FS.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));
    
    if (cds->solver[is]->Nmasked)
      cds->maskCopyKernel(cds->solver[is]->Nmasked,
                          0,
                          cds->solver[is]->o_maskIds,
                          o_zero,
                          o_FSi);
  }

  pack(nrs, cds->o_FS, o_ydot);
}

void cvodeSolver_t::makeq(nrs_t* nrs, dfloat time)
{

  auto * cds = nrs->cds;
  auto o_FS = nrs->cds->o_FS;

  bool useRelativeVelocity = platform->options.compareArgs("MOVING MESH", "TRUE");
  auto & o_Urst = useRelativeVelocity ? cds->o_relUrst : cds->o_Urst;

  if (udf.sEqnSource) {
    platform->timer.tic("udfSEqnSource", 1);
    udf.sEqnSource(nrs, time, cds->o_S, o_FS);
    platform->timer.toc("udfSEqnSource");
  }

  for (int is = 0; is < cds->NSfields; is++) {
    if (!cds->compute[is])
      continue;
    if (!cds->cvodeSolve[is])
      continue;
    mesh_t *mesh;
    (is) ? mesh = cds->meshV : mesh = cds->mesh[0];
    const dlong isOffset = cds->fieldOffsetScan[is];

    if(cds->options[is].compareArgs("REGULARIZATION METHOD", "RELAXATION")){
      cds->filterRTKernel(
        cds->meshV->Nelements,
        is,
        cds->o_filterMT,
        cds->filterS[is],
        isOffset,
        cds->o_rho,
        cds->o_S,
        o_FS);

      double flops = 6 * mesh->Np * mesh->Nq + 4 * mesh->Np;
      flops *= static_cast<double>(mesh->Nelements);
      platform->flopCounter->add("scalarFilterRT", flops);
    }

    if (cds->options[is].compareArgs("ADVECTION", "TRUE")) {
      if (cds->options[is].compareArgs("ADVECTION TYPE", "CUBATURE")){
        cds->strongAdvectionCubatureVolumeKernel(cds->meshV->Nelements,
                                                 mesh->o_vgeo,
                                                 mesh->o_cubDiffInterpT,
                                                 mesh->o_cubInterpT,
                                                 mesh->o_cubProjectT,
                                                 cds->vFieldOffset,
                                                 isOffset,
                                                 cds->vCubatureOffset,
                                                 cds->o_S,
                                                 o_Urst,
                                                 cds->o_rho,
                                                 platform->o_mempool.slice0);
      }
      else{
        cds->strongAdvectionVolumeKernel(cds->meshV->Nelements,
                                         mesh->o_vgeo,
                                         mesh->o_D,
                                         cds->vFieldOffset,
                                         isOffset,
                                         cds->o_S,
                                         o_Urst,
                                         cds->o_rho,
                                         platform->o_mempool.slice0);
      }
      platform->linAlg->axpby(cds->meshV->Nelements * cds->meshV->Np,
          -1.0,
          platform->o_mempool.slice0,
          1.0,
          o_FS,
          0,
          isOffset);

      auto o_FS_i = o_FS + isOffset * sizeof(dfloat);

      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, o_FS_i);

      timeStepper::advectionFlops(cds->mesh[is], 1);
    }

    // weak laplcian + boundary terms
    occa::memory o_Si = cds->o_S.slice(cds->fieldOffsetScan[is] * sizeof(dfloat), cds->fieldOffset[is] * sizeof(dfloat));

    platform->linAlg->fill(mesh->Nlocal, 0.0, platform->o_mempool.slice1);

    cds->helmholtzRhsBCKernel(mesh->Nelements,
                              mesh->o_sgeo,
                              mesh->o_vmapM,
                              mesh->o_EToB,
                              is,
                              time,
                              cds->fieldOffset[is],
                              mesh->o_x,
                              mesh->o_y,
                              mesh->o_z,
                              o_Si,
                              cds->o_EToB[is],
                              *(cds->o_usrwrk),
                              platform->o_mempool.slice1);
    cds->o_ellipticCoeff.copyFrom(cds->o_diff, mesh->Nlocal * sizeof(dfloat),
      cds->fieldOffsetScan[is] * sizeof(dfloat), 0);
    
    // make purely Laplacian
    auto o_helmholtzPart = cds->o_ellipticCoeff + nrs->fieldOffset * sizeof(dfloat);
    platform->linAlg->fill(mesh->Nlocal, 0.0, o_helmholtzPart);

    platform->linAlg->fill(mesh->Nlocal, 0.0, platform->o_mempool.slice2);

    // no masking, no gather scatter
    const bool applyMask = false;
    const bool skipGatherScatter = true;
    ellipticOperator(cds->solver[is], o_Si, platform->o_mempool.slice2, dfloatString, applyMask, skipGatherScatter);

    platform->linAlg->axpby(mesh->Nlocal, 1.0, platform->o_mempool.slice1,
      -1.0, platform->o_mempool.slice2);

    platform->linAlg->axpby(mesh->Nlocal,
        1.0,
        platform->o_mempool.slice2,
        1.0,
        o_FS,
        0,
        isOffset);

    auto o_FS_i = o_FS + isOffset * sizeof(dfloat);
    auto o_rho_i = cds->o_rho + isOffset * sizeof(dfloat);

    if(nrs->cht && is == 0){
      auto gsh = cds->mesh[0]->oogs;

      oogs::startFinish(o_FS_i, 1, nrs->fieldOffset, ogsDfloat, ogsAdd, gsh);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, o_FS_i);

      platform->o_mempool.slice0.copyFrom(o_rho_i, cds->fieldOffset[is] * sizeof(dfloat));
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_LMM, platform->o_mempool.slice0);
      oogs::startFinish(platform->o_mempool.slice0, 1, nrs->fieldOffset, ogsDfloat, ogsAdd, gsh);
      platform->linAlg->axmy(mesh->Nlocal, 1.0, mesh->o_invLMM, platform->o_mempool.slice0);

      platform->linAlg->aydx(mesh->Nlocal, 1.0, platform->o_mempool.slice0, o_FS_i);

    } else {
      platform->linAlg->aydx(mesh->Nlocal, 1.0, o_rho_i, o_FS_i);
    }

  }
}

void cvodeSolver_t::pack(nrs_t * nrs, occa::memory o_field, occa::memory o_y)
{
  if(userPack){
    userPack(nrs, o_field, o_y);
  } else {
    packKernel(nrs->cds->mesh[0]->Nlocal,
               nrs->meshV->Nlocal,
               nrs->Nscalar,
               nrs->fieldOffset,
               this->LFieldOffset,
               this->o_cvodeScalarIds,
               this->o_EToLUnique,
               o_field,
               o_y);
  }
}

void cvodeSolver_t::unpack(nrs_t * nrs, occa::memory o_y, occa::memory o_field)
{
  if(userUnpack){
    userUnpack(nrs, o_y, o_field);
  } else {
    unpackKernel(nrs->cds->mesh[0]->Nlocal,
               nrs->meshV->Nlocal,
               this->Nscalar,
               nrs->fieldOffset,
               this->LFieldOffset,
               this->o_cvodeScalarIds,
               this->o_EToL,
               o_y,
               o_field);
  }
}

void cvodeSolver_t::solve(nrs_t *nrs, double t0, double t1, int tstep)
{
#ifdef ENABLE_CVODE
  mesh_t *mesh = nrs->meshV;
  if (nrs->cht)
    mesh = nrs->cds->mesh[0];

  bool movingMesh = platform->options.compareArgs("MOVING MESH", "TRUE");

  o_U0.copyFrom(nrs->o_U, nrs->NVfields * nrs->fieldOffset * sizeof(dfloat));

  if (movingMesh) {

    o_meshU0.copyFrom(mesh->o_U, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

    o_xyz0.copyFrom(mesh->o_x, mesh->Nlocal * sizeof(dfloat), (0 * sizeof(dfloat)) * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_y, mesh->Nlocal * sizeof(dfloat), (1 * sizeof(dfloat)) * nrs->fieldOffset, 0);
    o_xyz0.copyFrom(mesh->o_z, mesh->Nlocal * sizeof(dfloat), (2 * sizeof(dfloat)) * nrs->fieldOffset, 0);

  }

  pack(nrs, nrs->cds->o_S, o_cvodeY);

  this->tnekRS = t0;
  this->tstep = tstep;

  double t;

  // call cvode solver
  int retval = 0;
  retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
  if(retval < 0){
    if(platform->comm.mpiRank == 0){
      std::cout << "... Restarting CVODE integrator\n";
    }
    pack(nrs, nrs->cds->o_S, o_cvodeY);
    retval = CVodeReInit(cvodeMem, t0, cvodeY);
    check_retval(&retval, "CVodeReInit", 1);
    this->tprev = std::numeric_limits<dfloat>::max();

    retval = CVode(cvodeMem, t1, cvodeY, &t, CV_NORMAL);
  }

  if(retval < 0){
    if(platform->comm.mpiRank == 0){
      std::cout << "CVODE failed after restart. Ending simulation.\n";
    }
    ABORT(1);
  }

  unpack(nrs, o_cvodeY, nrs->cds->o_S);

  // restore previous state
  nrs->o_U.copyFrom(o_U0, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

  if (movingMesh) {
    mesh->o_x.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (0 * sizeof(dfloat)) * nrs->fieldOffset);
    mesh->o_y.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (1 * sizeof(dfloat)) * nrs->fieldOffset);
    mesh->o_z.copyFrom(o_xyz0, mesh->Nlocal * sizeof(dfloat), 0, (2 * sizeof(dfloat)) * nrs->fieldOffset);

    mesh->o_U.copyFrom(o_meshU0, (nrs->NVfields * sizeof(dfloat)) * nrs->fieldOffset);

    mesh->update();

  }

  computeUrst(nrs, false);

  // apply boundary condition at time t1
  applyDirichlet(nrs, t1);
#endif
}

void cvodeSolver_t::printFinalStats() const
{
#ifdef ENABLE_CVODE
  auto cvodeMem = this->cvodeMem;

  long lenrw, leniw;
  long lenrwLS, leniwLS;
  long int nst, nfe, nsetups, nni, ncfn, netf;
  long int nli, npe, nps, ncfl, nfeLS;
  long int njtv;
  int retval;

  retval = CVodeGetWorkSpace(cvodeMem, &lenrw, &leniw);
  check_retval(&retval, "CVodeGetWorkSpace", 1);
  retval = CVodeGetNumSteps(cvodeMem, &nst);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  retval = CVodeGetNumRhsEvals(cvodeMem, &nfe);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  retval = CVodeGetNumLinSolvSetups(cvodeMem, &nsetups);
  check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
  retval = CVodeGetNumErrTestFails(cvodeMem, &netf);
  check_retval(&retval, "CVodeGetNumErrTestFails", 1);
  retval = CVodeGetNumNonlinSolvIters(cvodeMem, &nni);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  retval = CVodeGetNumNonlinSolvConvFails(cvodeMem, &ncfn);
  check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

  retval = CVodeGetNumJtimesEvals(cvodeMem, &njtv);
  check_retval(&retval, "CVodeGetNumJtimesEvals", 1);

  retval = CVodeGetLinWorkSpace(cvodeMem, &lenrwLS, &leniwLS);
  check_retval(&retval, "CVodeGetLinWorkSpace", 1);
  retval = CVodeGetNumLinIters(cvodeMem, &nli);
  check_retval(&retval, "CVodeGetNumLinIters", 1);
  retval = CVodeGetNumPrecEvals(cvodeMem, &npe);
  check_retval(&retval, "CVodeGetNumPrecEvals", 1);
  retval = CVodeGetNumPrecSolves(cvodeMem, &nps);
  check_retval(&retval, "CVodeGetNumPrecSolves", 1);
  retval = CVodeGetNumLinConvFails(cvodeMem, &ncfl);
  check_retval(&retval, "CVodeGetNumLinConvFails", 1);
  retval = CVodeGetNumLinRhsEvals(cvodeMem, &nfeLS);
  check_retval(&retval, "CVodeGetNumLinRhsEvals", 1);

  if(platform->comm.mpiRank == 0){
    printf("\nCVODE Statistics:\n");
    printf("nst     = %5ld\n", nst);
    printf("nni     = %5ld     nli     = %5ld     nli/nni = %5ld\n", nni, nli, nli / nni);
    printf("nfe     = %5ld     nfeLS   = %5ld\n", nfe, nfeLS);
    printf("netf    = %5ld     ncfn    = %5ld     ncfl    = %5ld\n", netf, ncfn, ncfl);
  }

#endif
}
} // namespace cvode