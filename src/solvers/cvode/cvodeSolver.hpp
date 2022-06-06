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

  using userRHS_t = std::function<void(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)>;
  using userPack_t = std::function<void(occa::memory o_YE, occa::memory o_YL)>;
  using userUnpack_t = std::function<void(occa::memory o_YE, occa::memory o_YL)>;
  using userLocalPointSource_t = std::function<void(nrs_t* nrs, occa::memory o_Y, occa::memory o_Ydot)>;

  cvodeSolver_t(nrs_t* nrs, const Parameters_t & params);

  void solve(nrs_t *nrs, dfloat t0, dfloat t1, int tstep);

  void setRHS(userRHS_t _userRHS) { userRHS = _userRHS; }
  void setPack(userPack_t _userPack) { userPack = _userPack; }
  void setUnpack(userUnpack_t _userUnpack){ userUnpack = _userUnpack;}
  void setLocalPointSource(userLocalPointSource_t _userLocalPointSource){ userLocalPointSource = _userLocalPointSource;}

private:

  void setupEToLMapping(nrs_t *nrs, cvodeSolver_t * cvodeSolver);

  void rhs(nrs_t *nrs, int tstep, dfloat time, dfloat tf, occa::memory o_y, occa::memory o_ydot)
  userRHS_t userRHS;

  userLocalPointSource_t userLocalPointSource;
  userPack_t userPack;
  userUnpack_t userUnpack;

  void pack(occa::memory o_YE, occa::memory o_YL);
  void unpack(occa::memory o_YE, occa::memory o_YL);
  void makeqImpl(nrs_t* nrs);

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