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

#ifndef ELLIPTIC_MOVING_MESH_MANAGER_H
#define ELLIPTIC_MOVING_MESH_MANAGER_H
#include <occa.hpp>
#include <types.h>
#include <vector>
#include <sstream>
#include <elliptic.h>
#include <functional>
#include <ins.h>

class MovingMeshManager final
{
public:
  MovingMeshManager(elliptic_t* meshSolver);
  occa::memory meshSolve(ins_t* ins, dfloat time);
private:

  int ndim;
  
  /** data **/
  dlong Nlocal;
  dlong fieldOffset;
  dlong Nelements;
  dlong Nfaces;
  oogs* oogs;
  linAlg_t* linAlg;

  /** ugly bit of state used for initialization **/
  bool velocitiesInitialized;

  /** memory **/
  // TODO: optimize for memory later...

  // Persistent memory (likely will need to keep a version here)
  occa::memory o_W;
  occa::memory o_wx; // slices of o_W corresponding to the correct component
  occa::memory o_wy;
  occa::memory o_wz;

  occa::memory o_Un;
  occa::memory o_Unx;
  occa::memory o_Uny;
  occa::memory o_Unz;

  occa::memory o_Rn;
  occa::memory o_Rnx;
  occa::memory o_Rny;
  occa::memory o_Rnz;
  
  occa::memory o_WV;
  occa::memory o_wvx; // not sure if needed
  occa::memory o_wvy;
  occa::memory o_wvz;

  // slices of meshSolver->o_lambda
  occa::memory o_h1;
  occa::memory o_h2;

  // References to weights, scratch space
  occa::memory & o_invDegree;
  occa::memory & o_wrk;

  /** kernels **/
  occa::kernel updateFaceVectorKernel;
  occa::kernel cartesianVectorDotProdKernel;

  /** helpers **/
  void update_system(elliptic_t* elliptic);
  void move_boundary(elliptic_t* elliptic);
  void area3();
};

#endif