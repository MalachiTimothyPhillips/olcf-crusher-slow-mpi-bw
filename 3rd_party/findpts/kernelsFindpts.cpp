
#include "ogstypes.h"
#include "findpts.hpp"
#include <cfloat>
#include <tuple>

// compute nearest power of two larger than v
// from: https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
unsigned nearestPowerOfTwo(unsigned int v){
  static_assert(sizeof(unsigned int) == 4);
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

std::vector<occa::kernel> initFindptsKernels(MPI_Comm comm, occa::device device,
                                                             dlong D, dlong Nq) {

  occa::kernel findpts_local;
  occa::kernel findpts_local_eval;
  occa::kernel findpts_local_eval_many;

  occa::properties kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  kernelInfo["defines/p_D"]  = D;
  kernelInfo["defines/p_Nq"] = Nq;
  kernelInfo["defines/p_Np"] = Nq*Nq*Nq;
  kernelInfo["defines/p_nptsBlock"] = 4;

  unsigned int Nq2 = Nq * Nq;
  const auto blockSize = nearestPowerOfTwo(Nq2);

  kernelInfo["defines/p_blockSize"] = blockSize;
  kernelInfo["defines/p_Nfp"] = Nq*Nq;
  kernelInfo["defines/dlong"] = dlongString;
  kernelInfo["defines/hlong"] = hlongString;
  kernelInfo["defines/dfloat"] = dfloatString;
  kernelInfo["defines/DBL_MAX"] = DBL_MAX;

  kernelInfo["includes"] += DFINDPTS "/okl/findpts.okl.hpp";
  kernelInfo["includes"] += DFINDPTS "/okl/poly.okl.hpp";

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  for (int r=0;r<2;r++){
    if ((r==0 && rank==0) || (r==1 && rank>0)) {
      findpts_local = device.buildKernel(DFINDPTS "/okl/findpts_local.okl", "findpts_local", kernelInfo);
      findpts_local_eval =
          device.buildKernel(DFINDPTS "/okl/findpts_local_eval.okl", "findpts_local_eval", kernelInfo);
      findpts_local_eval_many =
          device.buildKernel(DFINDPTS "/okl/findpts_local_eval_many.okl", "findpts_local_eval_many", kernelInfo);
    }
    MPI_Barrier(comm);
  }

  return {findpts_local_eval, findpts_local_eval_many, findpts_local};
}
