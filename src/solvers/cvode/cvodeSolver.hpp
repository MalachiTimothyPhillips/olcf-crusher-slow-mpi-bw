#ifndef CVODE_SOLVER_HPP_
#define CVODE_SOLVER_HPP_

#include <limits>
#include "nrssys.hpp"
#include "occa.hpp"
#include <functional>

class nrs_t;
namespace cvode {

struct Parameters_t {};
class cvodeSolver_t{
public:

  cvodeSolver_t(nrs_t* nrs, const Parameters_t & params);

  void solve(nrs_t *nrs, dfloat t0, dfloat t1, int tstep);

  // TODO: this should be fine, right?
  // (attempt to sheild the user from needing to mess with N_Vector, e.g.)
  void setRHS(std::function<void(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)> _userRHS);

  void setPack();
  void setUnpack();
  void setLocalPointSource();

private:

  void setupEToLMapping(nrs_t *nrs, cvodeSolver_t * cvodeSolver);

  // default RHS implementation
  void rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)
  std::function<void(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)> userRHS;

  void pack();
  void unpack();
  void makeqImpl();

  void setup(nrs_t* nrs, const Parameters_t & params);
  void reallocBuffer(dlong Nbytes);

  mutable dfloat tprev = std::numeric_limits<dfloat>::max();
  static constexpr int maxExtrapolationOrder = 3;

  occa::memory o_wrk;
  occa::memory o_coeffExt;
  occa::memory o_EToLUnique;
  occa::memory o_EToL;
  dlong LFieldOffset;

  occa::kernel extrapolateInPlaceKernel;
  occa::kernel mapEToLKernel;
  occa::kernel mapLToEKernel;
};
} // namespace cvode

#endif