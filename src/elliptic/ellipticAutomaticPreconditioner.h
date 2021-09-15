#if !defined(ellipticAutomaticPreconditioner_h_)
#define ellipticAutomaticPreconditioner_h_
#include <set>
#include <map>
#include <tuple>
#include <random>
#include <string>
#include <iostream>
class elliptic_t;

struct solverDescription_t{

  solverDescription_t(){}

  std::string to_string() const {
    std::ostringstream ss;
    ss << "Chebyshev+";
    if(smoother == 0)
      ss << "ASM";
    else if(smoother == 1)
      ss << "RAS";
    else if(smoother == 2)
      ss << "Jacobi";
    ss << "+Degree=" << chebyOrder;
    return ss.str();
  }

  solverDescription_t(unsigned m_smoother, unsigned m_chebyOrder)
  : smoother(m_smoother), chebyOrder(m_chebyOrder) {}

  unsigned smoother; // 0 - ASM, 1 - RAS, 2 - Jacobi
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
    RANDOM_SAMPLE,
    RANDOM_SAMPLE_NO_REPLACEMENT,
    STEPWISE,
  };
  static constexpr int NSmoothers {3};
  public:
  automaticPreconditioner_t(elliptic_t& m_elliptic);

  std::string
  to_string() const;

  void apply();

  private:
  void evaluate_current_solver();
  void select_solver();
  void reinitializePreconditioner();
  solverDescription_t fastest_solver();
  elliptic_t& elliptic;
  unsigned long solveCount;
  unsigned long trialFrequency;
  unsigned long trialCount;
  unsigned long maxTrials;
  unsigned long autoStart;
  unsigned int minChebyOrder;
  unsigned int maxChebyOrder;
  unsigned int fastestSmoother;

  solverDescription_t currentSolver;
  std::set<solverDescription_t> allSolvers;
  std::set<solverDescription_t> visitedSolvers;
  std::set<unsigned int> visitedSmoothers;
  std::set<unsigned int> visitedChebyOrders;
  std::map<solverDescription_t, double> solverToTime;

  Strategy strategy;

  std::random_device rd;
  std::mt19937 gen;
};
#endif