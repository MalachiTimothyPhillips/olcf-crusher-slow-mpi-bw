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

/* compile with C compiler (not C++) */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "gslib.h"

#include "findptsTypes.h"
#include "internal_findpts.h"

#include "ogstypes.h" // for dfloat

void devFindpts(      dlong  *const  code_base   ,
                           dlong  *const  proc_base   ,
                           dlong  *const    el_base   ,
                           dfloat *const     r_base   ,
                           dfloat *const dist2_base   ,
                     const dfloat *const     x_base[3],
                     const dlong npt, struct findpts_data_3 *const fd,
                     const void *const findptsData) {
  findpts_impl( code_base,
                 proc_base,
                   el_base,
                    r_base,
                dist2_base,
                    x_base,
                npt, fd, findptsData);
}

void devFindptsEval(
        dfloat *const  out_base,
  const dlong  *const code_base,
  const dlong  *const proc_base,
  const dlong  *const   el_base,
  const dfloat *const    r_base,
  const dlong npt, void *const in, struct findpts_data_3 *const fd,
  const void *const findptsData) {

  findpts_eval_impl( out_base,
                     code_base,
                     proc_base,
                       el_base,
                        r_base,
                     npt, in, fd, findptsData);
}

void devFindptsLocalEval(
        void *const  out,
  const void *const   el,
  const void *const    r,
  const dlong npt, void *const in, struct findpts_data_3 *const fd,
  const void *const findptsData) {

  findpts_local_eval(out, el, r, npt, in, &fd->local, findptsData);
}
