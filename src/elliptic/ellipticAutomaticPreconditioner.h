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
      ss << "Jacobi";
    else if(smoother == 1)
      ss << "ASM";
    else if(smoother == 2)
      ss << "RAS";
    ss << "+Degree=" << chebyOrder;
    return ss.str();
  }

  solverDescription_t(unsigned m_smoother, unsigned m_chebyOrder)
  : smoother(m_smoother), chebyOrder(m_chebyOrder) {}

  unsigned smoother; // 0 - JAC, 1 - ASM, 2 - RAS
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
  };
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
  unsigned int minChebyOrder;
  unsigned int maxChebyOrder;

  solverDescription_t currentSolver;
  std::set<solverDescription_t> allSolvers;
  std::set<solverDescription_t> visitedSolvers;
  std::map<solverDescription_t, double> solverToTime;

  Strategy strategy;

  std::random_device rd;
  std::mt19937 gen;
};
#endif