#if !defined(ellipticAutomaticPreconditioner_h_)
#define ellipticAutomaticPreconditioner_h_
#include <array>
#include <elliptic.h>
#include <ellipticMultiGrid.h>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <tuple>
class elliptic_t;

class MGLevel;

struct solverDescription_t{

  solverDescription_t(){}

  std::string to_string() const {
    std::ostringstream ss;

    if (preconditioner == PreconditionerType::SEMFEM) {
      ss << "SEMFEM";
    } else { // pMG
      ss << "Chebyshev+";
      if (preconditioner == PreconditionerType::CHEB_ASM) {
        ss << "ASM";
      } else if (preconditioner == PreconditionerType::CHEB_RAS) {
        ss << "RAS";
      } else if (preconditioner == PreconditionerType::CHEB_JAC) {
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
    }
    return ss.str();
  }

  solverDescription_t(PreconditionerType mPreconditioner,
      unsigned mChebyOrder,
      std::set<unsigned> mSchedule)
      : preconditioner(mPreconditioner), chebyOrder(mChebyOrder),
        schedule(mSchedule) {}

  PreconditionerType preconditioner;
  unsigned chebyOrder;
  std::set<unsigned> schedule;

  solverDescription_t(const solverDescription_t& rhs) = default;

  solverDescription_t& operator=(const solverDescription_t& rhs) = default;

  inline bool operator==(const solverDescription_t& other) const
  {
    return std::tie(preconditioner, chebyOrder, schedule) ==
           std::tie(other.preconditioner, other.chebyOrder, other.schedule);
  }
  inline bool operator<(const solverDescription_t& other) const
  {
    return std::tie(preconditioner, chebyOrder, schedule) <
           std::tie(other.preconditioner, other.chebyOrder, other.schedule);
  }
  inline bool operator> (const solverDescription_t& other) const { return *this < other; }
  inline bool operator<=(const solverDescription_t& other) const { return !(*this > other); }
  inline bool operator>=(const solverDescription_t& other) const { return !(*this < other); }
  inline bool operator!=(const solverDescription_t& other) const { return !(*this == other); }
};

class automaticPreconditioner_t{
  static constexpr int NSmoothers {3};
  public:
  automaticPreconditioner_t(elliptic_t& m_elliptic);

  std::string
  to_string() const;

  bool apply(int tstep);
  void measure(bool evaluatePreconditioner);

  private:
  void evaluateCurrentSolver();
  bool selectSolver();
  void reinitializePreconditioner();
  solverDescription_t determineFastestSolver();
  elliptic_t& elliptic;
  bool activeTuner;
  unsigned long trialFrequency;
  unsigned long autoStart;
  unsigned int minChebyOrder;
  unsigned int maxChebyOrder;
  unsigned int sampleCounter;
  int NSamples;
  ChebyshevSmootherType fastestSmoother;

  std::map<int, MGLevel*> multigridLevels;

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