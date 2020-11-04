#include "nrs.hpp"
#include "udf.hpp"

namespace tombo
{
occa::memory pressureSolve(ins_t* ins, dfloat time)
{
  mesh_t* mesh = ins->mesh;

  //enforce Dirichlet BCs
  ins->fillKernel((1+ins->NVfields)*ins->fieldOffset, std::numeric_limits<dfloat>::min(), ins->o_wrk6);
  for (int sweep = 0; sweep < 2; sweep++) {
    ins->pressureDirichletBCKernel(mesh->Nelements,
                                   time,
                                   ins->dt,
                                   ins->fieldOffset,
                                   mesh->o_sgeo,
                                   mesh->o_x,
                                   mesh->o_y,
                                   mesh->o_z,
                                   mesh->o_vmapM,
                                   mesh->o_EToB,
                                   ins->o_EToB,
                                   ins->o_usrwrk,
                                   ins->o_U,
                                   ins->o_P,
                                   ins->o_wrk6);

    ins->velocityDirichletBCKernel(mesh->Nelements,
                                   ins->fieldOffset,
                                   time,
                                   mesh->o_sgeo,
                                   mesh->o_x,
                                   mesh->o_y,
                                   mesh->o_z,
                                   mesh->o_vmapM,
                                   mesh->o_EToB,
                                   ins->o_EToB,
                                   ins->o_usrwrk,
                                   ins->o_U,
                                   ins->o_wrk7);

    //take care of Neumann-Dirichlet shared edges across elements
    if (sweep == 0) oogs::startFinish(ins->o_wrk6, 1+ins->NVfields, ins->fieldOffset, ogsDfloat, ogsMax, ins->gsh);
    if (sweep == 1) oogs::startFinish(ins->o_wrk6, 1+ins->NVfields, ins->fieldOffset, ogsDfloat, ogsMin, ins->gsh);
  }

  if (ins->pSolver->Nmasked) ins->maskCopyKernel(ins->pSolver->Nmasked, 0, ins->pSolver->o_maskIds,
                                                 ins->o_wrk6, ins->o_P); 

  if (ins->uvwSolver) {
    if (ins->uvwSolver->Nmasked) ins->maskCopyKernel(ins->uvwSolver->Nmasked, 0*ins->fieldOffset, ins->uvwSolver->o_maskIds,
                                                     ins->o_wrk7, ins->o_U);
  } else {
    if (ins->uSolver->Nmasked) ins->maskCopyKernel(ins->uSolver->Nmasked, 0*ins->fieldOffset, ins->uSolver->o_maskIds, 
                                                   ins->o_wrk7, ins->o_U);
    if (ins->vSolver->Nmasked) ins->maskCopyKernel(ins->vSolver->Nmasked, 1*ins->fieldOffset, ins->vSolver->o_maskIds, 
                                                   ins->o_wrk7, ins->o_U);
    if (ins->wSolver->Nmasked) ins->maskCopyKernel(ins->wSolver->Nmasked, 2*ins->fieldOffset, ins->wSolver->o_maskIds, 
                                                   ins->o_wrk7, ins->o_U);
  }

  ins->curlKernel(mesh->Nelements,
                  mesh->o_vgeo,
                  mesh->o_Dmatrices,
                  ins->fieldOffset,
                  ins->o_Ue,
                  ins->o_wrk0);

  oogs::startFinish(ins->o_wrk0, ins->NVfields, ins->fieldOffset,ogsDfloat, ogsAdd, ins->gsh);

  ins->invMassMatrixKernel(
    mesh->Nelements,
    ins->fieldOffset,
    ins->NVfields,
    mesh->o_vgeo,
    ins->mesh->o_invLMM,
    ins->o_wrk0);

  ins->curlKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_Dmatrices,
    ins->fieldOffset,
    ins->o_wrk0,
    ins->o_wrk3);

  ins->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_Dmatrices,
    ins->fieldOffset,
    ins->o_div,
    ins->o_wrk0);

  //if (ins->options.compareArgs("VARIABLE VISCOSITY", "TRUE"))
  if(ins->options.compareArgs("STRESSFORMULATION", "TRUE"))
    ins->pressureStressKernel(
         mesh->Nelements,
         mesh->o_vgeo,
         mesh->o_Dmatrices,
         ins->fieldOffset,
         ins->o_mue,
         ins->o_Ue,
         ins->o_div,
         ins->o_wrk3);

  occa::memory o_irho = ins->o_ellipticCoeff;
  ins->pressureRhsKernel(
    mesh->Nelements * mesh->Np,
    ins->fieldOffset,
    ins->o_mue,
    o_irho,
    ins->o_BF,
    ins->o_wrk3,
    ins->o_wrk0,
    ins->o_wrk6);

  oogs::startFinish(ins->o_wrk6, ins->NVfields, ins->fieldOffset,ogsDfloat, ogsAdd, ins->gsh);

  ins->invMassMatrixKernel(
    mesh->Nelements,
    ins->fieldOffset,
    ins->NVfields,
    mesh->o_vgeo,
    ins->mesh->o_invLMM,
    ins->o_wrk6);

  ins->divergenceVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_Dmatrices,
    ins->fieldOffset,
    ins->o_wrk6,
    ins->o_wrk3);

  ins->pressureAddQtlKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    ins->g0 * ins->idt,
    ins->o_div,
    ins->o_wrk3);

  ins->divergenceSurfaceKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_sgeo,
    mesh->o_vmapM,
    mesh->o_EToB,
    ins->o_EToB,
    time,
    ins->g0 * ins->idt,
    mesh->o_x,
    mesh->o_y,
    mesh->o_z,
    ins->fieldOffset,
    ins->o_usrwrk,
    ins->o_wrk6,
    ins->o_U,
    ins->o_wrk3);

  oogs::startFinish(ins->o_wrk3, 1, 0, ogsDfloat, ogsAdd, ins->gsh);

  ins->o_wrk1.copyFrom(ins->o_P, ins->Ntotal * sizeof(dfloat));
  ins->NiterP = ellipticSolve(ins->pSolver, ins->presTOL, ins->o_wrk3, ins->o_wrk1);

  return ins->o_wrk1;
}

occa::memory velocitySolve(ins_t* ins, dfloat time)
{
  mesh_t* mesh = ins->mesh;

  dfloat scale = -1./3;
  if(ins->options.compareArgs("STRESSFORMULATION", "TRUE")) scale = 2./3;

#if 0
  ins->PQKernel(
       mesh->Nelements*mesh->Np,
       -scale,
       ins->o_mue,
       ins->o_div,
       ins->o_P,
       ins->o_wrk3); 

  ins->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_Dmatrices,
    ins->fieldOffset,
    ins->o_wrk3,
    ins->o_wrk0);
#else
  ins->mueDivKernel(
       mesh->Nelements*mesh->Np,
       scale,
       ins->o_mue,
       ins->o_div,
       ins->o_wrk3); 

  ins->gradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_Dmatrices,
    ins->fieldOffset,
    ins->o_wrk3,
    ins->o_wrk0);

  ins->wgradientVolumeKernel(
    mesh->Nelements,
    mesh->o_vgeo,
    mesh->o_Dmatrices,
    ins->fieldOffset,
    ins->o_P,
    ins->o_wrk3); 

  ins->scaledAddKernel(
    ins->NVfields*ins->fieldOffset,
    1.0,
    0*ins->fieldOffset,
    ins->o_wrk3,
    -1.0,
    0*ins->fieldOffset,
    ins->o_wrk0);
#endif

  ins->velocityNeumannBCKernel(
       mesh->Nelements,
       ins->fieldOffset,
       mesh->o_sgeo,
       mesh->o_vmapM,
       mesh->o_EToB,
       ins->o_EToB,
       time,
       mesh->o_x,
       mesh->o_y,
       mesh->o_z,
       ins->o_usrwrk,
       ins->o_U,
       ins->o_wrk0); 

  ins->velocityRhsKernel(
    mesh->Nelements,
    ins->fieldOffset,
    ins->o_BF,
    ins->o_wrk0,
    ins->o_rho,
    ins->o_wrk3);

  oogs::startFinish(ins->o_wrk3, ins->NVfields, ins->fieldOffset,ogsDfloat, ogsAdd, ins->gsh);

  if(ins->options.compareArgs("VELOCITY INITIAL GUESS DEFAULT", "EXTRAPOLATION")) { 
    ins->o_wrk0.copyFrom(ins->o_Ue, ins->NVfields * ins->fieldOffset * sizeof(dfloat));
    if (ins->uvwSolver) {
      if (ins->uvwSolver->Nmasked) ins->maskCopyKernel(ins->uvwSolver->Nmasked, 0*ins->fieldOffset, ins->uvwSolver->o_maskIds,
                                                       ins->o_U, ins->o_wrk0);
    } else {
      if (ins->uSolver->Nmasked) ins->maskCopyKernel(ins->uSolver->Nmasked, 0*ins->fieldOffset, ins->uSolver->o_maskIds,
                                                     ins->o_U, ins->o_wrk0);
      if (ins->vSolver->Nmasked) ins->maskCopyKernel(ins->vSolver->Nmasked, 1*ins->fieldOffset, ins->vSolver->o_maskIds,
                                                     ins->o_U, ins->o_wrk0);
      if (ins->wSolver->Nmasked) ins->maskCopyKernel(ins->wSolver->Nmasked, 2*ins->fieldOffset, ins->wSolver->o_maskIds,
                                                     ins->o_U, ins->o_wrk0);
    }
  } else {
    ins->o_wrk0.copyFrom(ins->o_U, ins->NVfields * ins->fieldOffset * sizeof(dfloat));
  }

  if(ins->uvwSolver) {
    ins->NiterU = ellipticSolve(ins->uvwSolver, ins->velTOL, ins->o_wrk3, ins->o_wrk0);
  } else {
    ins->NiterU = ellipticSolve(ins->uSolver, ins->velTOL, ins->o_wrk3, ins->o_wrk0);
    ins->NiterV = ellipticSolve(ins->vSolver, ins->velTOL, ins->o_wrk4, ins->o_wrk1);
    ins->NiterW = ellipticSolve(ins->wSolver, ins->velTOL, ins->o_wrk5, ins->o_wrk2);
  }

  return ins->o_wrk0;
}

// see ptbgeom for an example of what steps are needed
occa::memory meshSolve(ins_t* ins, dfloat time)
{
  // elastic material constants
  double vnu = 0.0;
  const double eps = 1e-8;
  ins->meshOptions.getArgs("MESH VISCOSITY", vnu);
  if(std::abs(vnu) < eps)
    vnu = 0.4;
  vnu = std::abs(vnu);
  vnu = std::min(0.499,vnu);
  const double Ce = 1.0 / ( 1.0 + vnu);
  const double C2 = vnu * Ce / (1.0 - 2.0 * vnu);
  const double C3 = 0.5 * Ce;

  // call to ibdgeom here, e.g.

  // fill arrays here...

  const double eps = 1e-12;

  // TODO: fill out rest...
  dfloat diff = ellipticWeightedInnerProduct(ins->meshSolver,
    ins->meshSolver->o_invDegree, ins->meshSolver->);
  MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DFLOAT, MPI_MAX, ins->mesh->comm);

  if(diff < eps){
    return; // mesh solve not needed
  }
  

  ellipticOperator(ins->meshSolver, ins->o_wrk0, ins->o_wrk3, dfloatString);
  ins->linAlg->scale(ins->meshSolver->Ntotal, -1.0, ins->o_wrk3);

  ins->NiterMeshSolve = ellipticSolve(ins->meshSolver, ins->meshTOL, ins->o_wrk3, ins->o_wrk0);
  return ins->o_wrk0;
}

} // namespace
