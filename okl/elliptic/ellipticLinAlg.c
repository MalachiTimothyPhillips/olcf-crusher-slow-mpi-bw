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

extern "C" void FUNC(scaledAdd) (const dlong & N,
  const pfloat & alpha,
  const  pfloat *  __restrict__ x,
  const pfloat & beta,
  pfloat *  __restrict__ y){
  
#ifdef __NEKRS__OMP__
  #pragma omp parallel for
#endif
  for(dlong n = 0; n < N; ++n){
      y[n] = alpha*x[n] + beta*y[n];
  }
}

extern "C" void FUNC(fusedCopyDfloatToPfloat) (const dlong & N,
  const dfloat * __restrict__ x_dfloat,
  const dfloat * __restrict__ y_dfloat,
  pfloat * __restrict__ x_pfloat,
  pfloat * __restrict__ y_pfloat){
#ifdef __NEKRS__OMP__
  #pragma omp parallel for
#endif
  for(dlong n=0;n<N;++n){
    x_pfloat[n] = x_dfloat[n];
    y_pfloat[n] = y_dfloat[n];
  }
}

extern "C" void FUNC(copyDfloatToPfloat) (const dlong & N,
  const dfloat * __restrict__ x,
  pfloat *  __restrict__ y){
  
#ifdef __NEKRS__OMP__
  #pragma omp parallel for
#endif
  for(dlong n=0;n<N;++n){
    y[n]=x[n];
  }
}

extern "C" void FUNC(copyPfloatToDfloat) (const dlong & N,
  const pfloat *  __restrict__ y,
  dfloat * __restrict__ x){
  
#ifdef __NEKRS__OMP__
  #pragma omp parallel for
#endif
  for(dlong n=0;n<N;++n){
    x[n]=y[n];
  }
}

extern "C" void FUNC(dotMultiply) (const dlong & N,
  const  pfloat *  __restrict__ w,
  const  pfloat *  __restrict__ v,
  pfloat *  __restrict__ wv){

#ifdef __NEKRS__OMP__
  #pragma omp parallel for
#endif
  for(dlong n=0;n<N;++n){
    wv [n] = w[n]*v[n];
  }

}