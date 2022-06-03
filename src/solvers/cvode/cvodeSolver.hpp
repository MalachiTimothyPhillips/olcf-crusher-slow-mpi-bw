#ifndef CVODE_SOLVER_HPP_
#define CVODE_SOLVER_HPP_

#include <limits>
#include "nrssys.hpp"
#include "occa.hpp"

class nrs_t;
namespace cvode {

struct Parameters_t {};
struct cvodeSolver_t{

  cvodeSolver_t(nrs_t* nrs, const Parameters_t & params);

  void solve(nrs_t *nrs, dfloat t0, dfloat t1, int tstep);

private:

  // default RHS implementation
  void rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)

  mutable dfloat tprev = std::numeric_limits<dfloat>::max();
  static constexpr int maxExtrapolationOrder = 3;
  void setup(nrs_t* nrs, const Parameters_t & params);
  void reallocBuffer(dlong Nbytes);

  occa::memory o_wrk;
  occa::memory o_coeffAB;
  occa::memory o_EToLUnique;
  occa::memory o_EToL;
  dlong LFieldOffset;

  occa::kernel extrapolateInPlaceKernel;
  occa::kernel mapEToLKernel;
  occa::kernel mapLToEKernel;
};

/**
// user-facing routines
// add methods for registering pack/unpack fptrs
void setPack(...);
void setUnpack(...);
void setRHS(...);

// point-wise, local source term -- cannot assume to be called collectively
// will be overlapped with, e.g., gather/scatter
void setLocalPointSource(...);


void setup(nrs_t *, Parameters_t params);

// private
void rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot);
**/
// default pack/unpack
} // namespace cvode

#endif