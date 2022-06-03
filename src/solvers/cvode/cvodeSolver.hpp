#ifndef CVODE_SOLVER_HPP_
#define CVODE_SOLVER_HPP_

#include "nrssys.hpp"
#include "occa.hpp"

class nrs_t;
namespace cvode {

// user-facing routines
// add methods for registering pack/unpack fptrs
void setPack(...);
void setUnpack(...);
void setRHS(...);

// point-wise, local source term -- cannot assume to be called collectively
// will be overlapped with, e.g., gather/scatter
void setLocalPointSource(...);

struct Parameters_t {};

void setup(nrs_t *, Parameters_t params);
void solve(nrs_t *nrs, dfloat t0, dfloat t1, int tstep);

// private
void rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot);
// default pack/unpack
} // namespace cvode

#endif