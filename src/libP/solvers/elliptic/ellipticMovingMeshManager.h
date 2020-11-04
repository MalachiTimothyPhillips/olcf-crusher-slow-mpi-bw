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
  occa::memory meshSolve(ins_t* ins, dfloat time);
private:

  static constexpr ndim = 3;
  
  /** data **/
  dlong Nlocal;
  dlong fieldOffset;
  oogs* oogs;
  linAlg_t* linAlg;

  /** memory **/
  occa::memory o_meshVelocity;
  occa::memory o_normal;
  occa::memory & o_invDegree;

  /** kernels **/
  occa::kernel extractFaceKernel;
  occa::kernel scaleFaceKernel;

  /** helper functions **/
  void move_boundary(elliptic_t*);
  void updmsys(elliptic_t*);
};

#endif