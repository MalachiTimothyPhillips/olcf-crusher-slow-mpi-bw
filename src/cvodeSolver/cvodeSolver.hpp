#ifndef CVODE_SOLVER_HPP_
#define CVODE_SOLVER_HPP_

#include "nrssys.hpp"
#include "occa.hpp"

class nrs_t;
namespace cvode {

struct Parameters_t {};
void setup(nrs_t *, Parameters_t params);
void solve(nrs_t *nrs, dfloat t0, dfloat t1, int tstep);
void rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot);
} // namespace cvode

#endif