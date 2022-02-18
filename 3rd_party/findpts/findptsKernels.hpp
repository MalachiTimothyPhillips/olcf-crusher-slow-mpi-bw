
#ifndef KERNELS_FINDPTS_HPP
#define KERNELS_FINDPTS_HPP

#include "occa.h"
#include <tuple>

std::tuple<occa::kernel, occa::kernel, occa::kernel> initFindptsKernels(
      MPI_Comm comm, occa::device device, dlong D, dlong Nq);

#endif
