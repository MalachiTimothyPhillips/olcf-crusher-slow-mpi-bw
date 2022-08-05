#if !defined(ellipticAutomaticPreconditioner_h_)
#define ellipticAutomaticPreconditioner_h_
#include <set>
#include <map>
#include <tuple>
#include <string>
#include <iostream>
#include <array>
#include <ellipticMultiGrid.h>
#include <functional>
class elliptic_t;

class MGLevel;

struct solverDescription_t {

  std::string to_string() const
  {
    std::ostringstream ss;

    if (chebyshevSmoother == SmootherType::CHEBYSHEV) {
      ss << "1st Cheby+";
    }
    else if (chebyshevSmoother == SmootherType::FOURTH_CHEBYSHEV) {
      ss << "4th Cheby+";
    }
    else if (chebyshevSmoother == SmootherType::OPT_CHEBYSHEV) {
      ss << "Opt. 4th Cheby+";
    }

    if (smoother == ChebyshevSmootherType::ASM) {
      ss << "ASM";
    }
    else if (smoother == ChebyshevSmootherType::RAS) {
      ss << "RAS";
    }
    else if (smoother == ChebyshevSmootherType::JACOBI) {
      ss << "Jacobi";
    }

    const auto actualChebOrder = usePostSmoothing ? chebyOrder : 2 * chebyOrder + 1;
    ss << "(" << actualChebOrder << ")";
    ss << ",(";
    for (auto &&i = schedule.rbegin(); i != schedule.rend(); ++i) {
      ss << *i;
      auto nextIt = i;
      std::advance(nextIt, 1);
      if (nextIt != schedule.rend())
        ss << ",";
    }
    ss << ")";
    if(usePostSmoothing){
      ss << ",P.S.";
    } else {
      ss << ",N.P.S.";
    }
    return ss.str();
  }

  solverDescription_t(bool mUsePostSmoothing,
                      SmootherType mChebyshevSmoother,
                      ChebyshevSmootherType mSmoother,
                      unsigned mChebyOrder,
                      std::set<unsigned> mSchedule)
      : usePostSmoothing(mUsePostSmoothing), chebyshevSmoother(mChebyshevSmoother), smoother(mSmoother), chebyOrder(mChebyOrder),
        schedule(mSchedule)
  {
  }

  bool usePostSmoothing;
  SmootherType chebyshevSmoother;
  ChebyshevSmootherType smoother;
  unsigned chebyOrder;
  std::set<unsigned> schedule;

  solverDescription_t(const solverDescription_t &rhs) = default;

  solverDescription_t &operator=(const solverDescription_t &rhs) = default;

  inline bool operator==(const solverDescription_t &other) const
  {
    return std::tie(usePostSmoothing, chebyshevSmoother, smoother, chebyOrder, schedule) ==
           std::tie(other.usePostSmoothing, other.chebyshevSmoother, other.smoother, other.chebyOrder, other.schedule);
  }
  inline bool operator<(const solverDescription_t &other) const
  {
    return std::tie(usePostSmoothing, chebyshevSmoother, smoother, chebyOrder, schedule) <
           std::tie(other.usePostSmoothing, other.chebyshevSmoother, other.smoother, other.chebyOrder, other.schedule);
  }
  inline bool operator>(const solverDescription_t &other) const { return *this < other; }
  inline bool operator<=(const solverDescription_t &other) const { return !(*this > other); }
  inline bool operator>=(const solverDescription_t &other) const { return !(*this < other); }
  inline bool operator!=(const solverDescription_t &other) const { return !(*this == other); }
};

class automaticPreconditioner_t {
public:
  automaticPreconditioner_t(elliptic_t &m_elliptic);

  std::string to_string() const;

  void tune(int tstep, std::function<void(occa::memory & o_r, occa::memory & o_x)> solverFunc, occa::memory & o_r, occa::memory & o_x);

private:
  void saveState(occa::memory & o_r, occa::memory & o_x);
  void restoreState(occa::memory & o_r, occa::memory & o_x);

  bool tuneStep(int tstep) const;
  void reinitializePreconditioner(solverDescription_t solver);

  elliptic_t &elliptic;
  unsigned long trialFrequency;
  unsigned long autoStart;
  unsigned int minChebyOrder;
  unsigned int maxChebyOrder;
  int NSamples;

  std::map<int, MGLevel *> multigridLevels;

  std::set<solverDescription_t> allSolvers;
  std::map<solverDescription_t, std::vector<double>> solverToTime;
  std::map<solverDescription_t, std::vector<unsigned int>> solverToIterations;

  // for storing rhs, x information
  occa::memory o_rSave, o_xSave;
};
#endif