#if !defined(ellipticAutomaticPreconditioner_h_)
#define ellipticAutomaticPreconditioner_h_
#include <set>
#include <map>
#include <tuple>
#include <random>
#include <string>
#include <iostream>
#include <ellipticMultiGrid.h>
class elliptic_t;

struct solverDescription_t{

  solverDescription_t(){}

  std::string to_string() const {
    std::ostringstream ss;
    ss << "Chebyshev+";
    if(smoother == ChebyshevSmootherType::ASM)
      ss << "ASM";
    else if(smoother == ChebyshevSmootherType::RAS)
      ss << "RAS";
    else if(smoother == ChebyshevSmootherType::JACOBI)
      ss << "Jacobi";
    ss << "+Degree=" << chebyOrder;
    return ss.str();
  }

  solverDescription_t(ChebyshevSmootherType mSmoother, unsigned mChebyOrder)
  : smoother(mSmoother), chebyOrder(mChebyOrder) {}

  ChebyshevSmootherType smoother;
  unsigned chebyOrder;

  solverDescription_t(const solverDescription_t& rhs) = default;

  solverDescription_t& operator=(const solverDescription_t& rhs) = default;

  inline bool operator==(const solverDescription_t& other) const
  {
    return std::tie(smoother, chebyOrder) == std::tie(other.smoother, other.chebyOrder);
  }
  inline bool operator<(const solverDescription_t& other) const
  {
    return std::tie(smoother, chebyOrder) < std::tie(other.smoother, other.chebyOrder);
  }
  inline bool operator> (const solverDescription_t& other) const { return *this < other; }
  inline bool operator<=(const solverDescription_t& other) const { return !(*this > other); }
  inline bool operator>=(const solverDescription_t& other) const { return !(*this < other); }
  inline bool operator!=(const solverDescription_t& other) const { return !(*this == other); }
};

class automaticPreconditioner_t{
  enum class Strategy{
    STEPWISE,
  };
  static constexpr int NSmoothers {3};
  public:
  automaticPreconditioner_t(elliptic_t& m_elliptic);

  std::string
  to_string() const;

  void apply();

  private:
  void evaluateCurrentSolver();
  void selectSolver();
  void reinitializePreconditioner();
  solverDescription_t stepwiseSelection();
  solverDescription_t determineFastestSolver();
  elliptic_t& elliptic;
  unsigned long solveCount;
  unsigned long trialFrequency;
  unsigned long trialCount;
  unsigned long maxTrials;
  unsigned long autoStart;
  unsigned int minChebyOrder;
  unsigned int maxChebyOrder;
  ChebyshevSmootherType fastestSmoother;

  solverDescription_t currentSolver;
  std::set<solverDescription_t> allSolvers;
  std::set<solverDescription_t> visitedSolvers;
  std::set<ChebyshevSmootherType> visitedSmoothers;
  std::set<unsigned int> visitedChebyOrders;
  std::map<solverDescription_t, double> solverToTime;

  Strategy strategy;

  std::random_device rd;
  std::mt19937 gen;
};
#endif