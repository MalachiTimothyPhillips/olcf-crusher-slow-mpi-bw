#ifndef ellipticBuildSEMFEM_hpp_
#define ellipticBuildSEMFEM_hpp_

#include <platform.hpp>
#include <mpi.h>

struct SEMFEMData{
  hypreWrapper::BigInt* Ai;
  hypreWrapper::BigInt* Aj;
  hypreWrapper::Real  * Av;
  hypreWrapper::Int     nnz;
  hypreWrapper::BigInt  rowStart;
  hypreWrapper::BigInt  rowEnd;
  dlong* dofMap;
};

SEMFEMData* ellipticBuildSEMFEM(const int N_, const int n_elem_, 
                   occa::memory _o_x, occa::memory _o_y, occa::memory _o_z,
                   dfloat *pmask_,
                   MPI_Comm comm,
                   long long int* gatherGlobalNodes);


#endif
