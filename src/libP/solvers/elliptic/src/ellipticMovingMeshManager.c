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
#include "ellipticMovingMeshManager.h"
#include <iostream>
#include <timer.hpp>
void MovingMeshManager::meshSolve(ins_t* ins, dfloat time)
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

  updmsys(ins->meshSolver);
  move_boundary(ins->meshSolver);

  linAlg->fill(ins->meshSolver->Ntotal, C2, ins->meshSolver->o_lambda);
  occa::memory h2 = ins->meshSolver->o_lambda + ins->fieldOffset * sizeof(dfloat);
  linAlg->fill(ins->meshSolver->Ntotal, C3, h2);

  const double eps = 1e-12;

  // TODO: fill out rest...
  dfloat diff = ellipticWeightedInnerProduct(ins->meshSolver,
    ins->meshSolver->o_invDegree, ins->meshSolver->);
  MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DFLOAT, MPI_MAX, ins->mesh->comm);

  if(diff < eps){
    return; // mesh solve not needed
  }
  

  ellipticOperator(ins->meshSolver, ins->o_wrk0, ins->o_wrk3, dfloatString);
  ins->linAlg->scale(ins->meshSolver->Ntotal*ndim, -1.0, ins->o_wrk3);

  ins->NiterMeshSolve = ellipticSolve(ins->meshSolver, ins->meshTOL, ins->o_wrk3, ins->o_wrk0);
  // add in solution
  // dsavg the solution
  return ins->o_wrk0;
}
void MovingMeshManager::updmsys(elliptic_t* elliptic)
{
}
void MovingMeshManager::move_boundary(elliptic_t* elliptic)
{
  extractFaceKernel(Nlocal, fieldOffset, o_meshVelocity, o_normal);
  oogs::startFinish(o_normal, ndim, 0, ogsDfloat, ogsAdd, oogs);
  const dfloat norm2 = linAlg->norm2(elliptic, o_normal, MPI_COMM_NULL);
  linAlg->scale(Ntotal*ndim, 1.0/norm2, o_normal);
  constexpr int nsweep = 2;
  for(int sweep = 0; sweep < nsweep; sweep++)
  {
    scaleFaceKernel(Ntotal*ndim, o_normal);
    oogs::startFinish(o_normal, ndim, 0, ogsDfloat, ogsAdd, oogs);
    for(int dim = 0 ; dim < ndim; ++dim){
      occa::memory o_slice = o_normal + dim * fieldOffset * sizeof(dfloat);
      linAlg->axmy(Ntotal, 1.0, o_invDegree, o_slice);
    }
    // TODO: opcopy on istep0
    // TODO: stuff for conjugateHeatTransfer problem
    // some other logical stuff related to symmetry boundary conditions, fix b.c.?

    if(sweep == 0){
      oogs::startFinish(o_normal, ndim, 0, ogsDfloat, ogsMax, oogs);
    } else{
      oogs::startFinish(o_normal, ndim, 0, ogsDfloat, ogsMin, oogs);
    }

    // wx += wvx, wy += wvy, wz += wvz
    linAlg->axpby(Ntotal*ndim, 1.0, ..., 1.0, o_meshVelocity);
  }


}