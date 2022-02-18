
#include "ogstypes.h"
#include "findpts.hpp"
#include "findptsKernels.hpp"
#include <cfloat>
#include <tuple>

std::tuple<occa::kernel, occa::kernel, occa::kernel> initFindptsKernels(MPI_Comm comm, occa::device device,
                                                             dlong D, dlong Nq) {

  occa::kernel findpts_local;
  occa::kernel findpts_local_eval;
  occa::kernel findpts_local_eval_vector;

  occa::properties kernelInfo;
  kernelInfo["defines"].asObject();
  kernelInfo["includes"].asArray();
  kernelInfo["header"].asArray();
  kernelInfo["flags"].asObject();
  kernelInfo["include_paths"].asArray();

  kernelInfo["defines/p_D"]  = D;
  kernelInfo["defines/p_Nq"] = Nq;
  kernelInfo["defines/p_Nfp"] = Nq*Nq;
  kernelInfo["defines/p_Np"] = Nq*Nq*Nq;
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
      findpts_local_eval_vector =
          device.buildKernel(DFINDPTS "/okl/findpts_local_eval_vector.okl", "findpts_local_eval_vector", kernelInfo);
    }
    MPI_Barrier(comm);
  }

  return std::make_tuple(findpts_local_eval_vector, findpts_local_eval, findpts_local);
}
