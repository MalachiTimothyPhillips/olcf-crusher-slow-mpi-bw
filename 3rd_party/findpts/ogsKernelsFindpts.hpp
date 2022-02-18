
#ifndef OGS_KERNELS_FINDPTS_HPP
#define OGS_KERNELS_FINDPTS_HPP

#include "ogsKernels.hpp"
#include <tuple>

namespace ogs {
  std::tuple<occa::kernel, occa::kernel, occa::kernel> initFindptsKernel(
        MPI_Comm comm, occa::device device, dlong D, dlong Nq);
}

#endif
