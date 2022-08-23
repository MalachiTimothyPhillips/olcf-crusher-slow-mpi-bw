#ifndef CVODE_SOLVER_HPP_
#define CVODE_SOLVER_HPP_

#include <limits>
#include "nrssys.hpp"
#include "occa.hpp"
#include <functional>
#include <map>
#include <vector>
#include <tuple>
#include <memory>

#ifdef ENABLE_CVODE
#include <cvode/cvode.h>
#endif

class nrs_t;
namespace cvode {

class cvodeSolver_t{
public:

  using userRHS_t = std::function<void(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)>;
  using userJacobian_t = std::function<void(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot)>;
  using userPack_t = std::function<void(nrs_t*, occa::memory o_field, occa::memory o_y)>;
  using userUnpack_t = std::function<void(nrs_t*, occa::memory o_y, occa::memory o_field)>;
  using userLocalPointSource_t = std::function<void(nrs_t* nrs, occa::memory o_y, occa::memory o_ydot)>;

  cvodeSolver_t(nrs_t* nrs);

  void solve(nrs_t *nrs, dfloat t0, dfloat t1, int tstep);

  void setRHS(userRHS_t _userRHS) { userRHS = _userRHS; }
  void setJacobian(userJacobian_t _userJacobian) { userJacobian = _userJacobian; }
  void setPack(userPack_t _userPack) { userPack = _userPack; }
  void setUnpack(userUnpack_t _userUnpack){ userUnpack = _userUnpack;}
  void setLocalPointSource(userLocalPointSource_t _userLocalPointSource){ userLocalPointSource = _userLocalPointSource;}
  void printInfo(bool printVerboseInfo) const;

  void setJacobianEvaluation() { jacEval = true; }
  void unsetJacobianEvaluation() { jacEval = false; }
  bool jacobianEvaluation() const { return jacEval; }

  int timeStep() const { return tstep; }
  double time() const { return tnekRS; }

  void rhs(nrs_t *nrs, dfloat time, occa::memory o_y, occa::memory o_ydot);
  void jtvRHS(nrs_t *nrs, dfloat time, occa::memory o_y, occa::memory o_ydot);
  dlong numEquations() const { return nEq;}

  // getters needed for CVLsJacTimesVecFn
  void * getCvodeMem() { return cvodeMem; }
  double sigmaScale() const { return sigScale; }

private:

  // package data to pass in as user data
  struct userData_t{

    userData_t(platform_t* _platform, nrs_t* _nrs, cvodeSolver_t* _cvodeSolver)
    : platform(_platform),
      nrs(_nrs),
      cvodeSolver(_cvodeSolver)
    {}


    platform_t* platform;
    nrs_t* nrs;
    cvodeSolver_t* cvodeSolver;
  };
  std::shared_ptr<userData_t> userdata;

  // most recent time from nekRS -- used to compute dt in CVODE integration call
  mutable double tnekRS;
  mutable int tstep;
  mutable bool jacEval = false;

  int minCvodeScalarId;
  int maxCvodeScalarId;

  mutable long int prevNsteps = 0;
  mutable long int prevNrhs = 0;
  mutable long int prevNli = 0;
  mutable long int prevNni = 0;

  void defaultRHS(nrs_t *nrs, int tstep, dfloat time, dfloat t0, occa::memory o_y, occa::memory o_ydot);
  dlong LFieldOffset;
  void pack(nrs_t * nrs, occa::memory o_field, occa::memory o_y);
  void unpack(nrs_t * nrs, occa::memory o_y, occa::memory o_field);
  dfloat tprev = std::numeric_limits<dfloat>::max();
  occa::memory o_U0;
  occa::memory o_meshU0;
  occa::memory o_xyz0;


  std::vector<std::tuple<dlong,dlong, oogs_t*>> gatherScatterOperations;

  void setupEToLMapping(nrs_t *nrs);

  userRHS_t userRHS;
  userJacobian_t userJacobian;

  userLocalPointSource_t userLocalPointSource;
  userPack_t userPack;
  userUnpack_t userUnpack;

  void makeq(nrs_t* nrs, dfloat time);

  static constexpr int maxExtrapolationOrder = 3;
  std::array<dfloat, maxExtrapolationOrder> coeffBDF;
  std::array<dfloat, maxExtrapolationOrder> coeffEXT;
  std::array<dfloat, maxExtrapolationOrder> dtCvode;
  dfloat g0;

  dlong Nscalar;
  dlong fieldOffsetSum;

  std::vector<dlong> fieldOffset;
  std::vector<dlong> fieldOffsetScan;

  occa::memory o_coeffExt;
  occa::memory o_EToLUnique;
  occa::memory o_EToL;

  occa::memory o_cvodeScalarIds;
  occa::memory o_scalarIds;

  // combined invLMM * LMM
  occa::memory o_invLMMLMMT;
  occa::memory o_invLMMLMMV;


  occa::kernel extrapolateInPlaceKernel;
  occa::kernel mapEToLKernel;
  occa::kernel mapLToEKernel;
  occa::kernel packKernel;
  occa::kernel unpackKernel;

  dlong nEq;

  long long int nEqTotal;

  dfloat sigScale = 1.0;

  // cvode internals
  void * cvodeMem;
#ifdef ENABLE_CVODE
  N_Vector y;
  N_Vector cvodeY;
#endif
  occa::memory o_cvodeY;

};
} // namespace cvode

#endif