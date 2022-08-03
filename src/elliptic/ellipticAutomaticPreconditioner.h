#if !defined(ellipticAutomaticPreconditioner_h_)
#define ellipticAutomaticPreconditioner_h_
#include <set>
#include <map>
#include <tuple>
#include <string>
#include <iostream>
#include <array>
#include <ellipticMultiGrid.h>
class elliptic_t;

class MGLevel;

struct solverDescription_t {

  solverDescription_t() {}

  std::string to_string() const
  {
    std::ostringstream ss;

    if (chebyshevSmoother == SmootherType::CHEBYSHEV) {
      ss << "1st Kind Chebyshev     +";
    }
    else if (chebyshevSmoother == SmootherType::FOURTH_CHEBYSHEV) {
      ss << "4th Kind Chebyshev     +";
    }
    else if (chebyshevSmoother == SmootherType::OPT_CHEBYSHEV) {
      ss << "Opt. 4th Kind Chebyshev+";
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
    ss << "+Degree=" << chebyOrder;
    ss << ",(";
    for (auto &&i = schedule.rbegin(); i != schedule.rend(); ++i) {
      ss << *i;
      auto nextIt = i;
      std::advance(nextIt, 1);
      if (nextIt != schedule.rend())
        ss << ",";
    }
    ss << ")";
    return ss.str();
  }

  solverDescription_t(SmootherType mChebyshevSmoother,
                      ChebyshevSmootherType mSmoother,
                      unsigned mChebyOrder,
                      std::set<unsigned> mSchedule)
      : chebyshevSmoother(mChebyshevSmoother), smoother(mSmoother), chebyOrder(mChebyOrder),
        schedule(mSchedule)
  {
  }

  SmootherType chebyshevSmoother;
  ChebyshevSmootherType smoother;
  unsigned chebyOrder;
  std::set<unsigned> schedule;

  solverDescription_t(const solverDescription_t &rhs) = default;

  solverDescription_t &operator=(const solverDescription_t &rhs) = default;

  inline bool operator==(const solverDescription_t &other) const
  {
    return std::tie(chebyshevSmoother, smoother, chebyOrder, schedule) ==
           std::tie(other.chebyshevSmoother, other.smoother, other.chebyOrder, other.schedule);
  }
  inline bool operator<(const solverDescription_t &other) const
  {
    return std::tie(chebyshevSmoother, smoother, chebyOrder, schedule) <
           std::tie(other.chebyshevSmoother, other.smoother, other.chebyOrder, other.schedule);
  }
  inline bool operator>(const solverDescription_t &other) const { return *this < other; }
  inline bool operator<=(const solverDescription_t &other) const { return !(*this > other); }
  inline bool operator>=(const solverDescription_t &other) const { return !(*this < other); }
  inline bool operator!=(const solverDescription_t &other) const { return !(*this == other); }
};

class automaticPreconditioner_t {
  static constexpr int NSmoothers{3};

public:
  automaticPreconditioner_t(elliptic_t &m_elliptic);

  std::string to_string() const;

  bool apply(int tstep);
  void measure(bool evaluatePreconditioner);

private:
  void evaluateCurrentSolver();
  bool selectSolver();
  void reinitializePreconditioner();
  solverDescription_t determineFastestSolver();
  elliptic_t &elliptic;
  bool activeTuner;
  unsigned long trialFrequency;
  unsigned long autoStart;
  unsigned int minChebyOrder;
  unsigned int maxChebyOrder;
  unsigned int sampleCounter;
  int NSamples;
  ChebyshevSmootherType fastestSmoother;

  std::map<int, MGLevel *> multigridLevels;

  solverDescription_t defaultSolver;
  solverDescription_t currentSolver;
  std::set<solverDescription_t> allSolvers;
  std::set<solverDescription_t> visitedSolvers;
  std::vector<solverDescription_t> remainingSolvers;
  std::set<ChebyshevSmootherType> visitedSmoothers;
  std::map<solverDescription_t, std::vector<double>> solverToTime;
  std::map<solverDescription_t, std::vector<double>> solverTimePerIter;
  std::map<solverDescription_t, double> solverStartTime;
  std::map<solverDescription_t, double> solverToEval;
  std::map<solverDescription_t, std::vector<unsigned int>> solverToIterations;
};
#endif