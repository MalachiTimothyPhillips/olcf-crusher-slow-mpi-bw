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
    ss << ",(";
    for (auto&& i = levels.rbegin(); 
        i != levels.rend(); ++i ) { 
      ss << *i << ",";
    } 
    ss << ")";
    return ss.str();
  }

  solverDescription_t(ChebyshevSmootherType mSmoother, unsigned mChebyOrder, std::set<unsigned> mLevels)
  : smoother(mSmoother), chebyOrder(mChebyOrder), levels(mLevels){}

  ChebyshevSmootherType smoother;
  unsigned chebyOrder;
  std::set<unsigned> levels;

  solverDescription_t(const solverDescription_t& rhs) = default;

  solverDescription_t& operator=(const solverDescription_t& rhs) = default;

  inline bool operator==(const solverDescription_t& other) const
  {
    return std::tie(smoother, chebyOrder, levels) == std::tie(other.smoother, other.chebyOrder, levels);
  }
  inline bool operator<(const solverDescription_t& other) const
  {
    return std::tie(smoother, chebyOrder, levels) < std::tie(other.smoother, other.chebyOrder, levels);
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

  solverDescription_t currentSolver;
  std::set<solverDescription_t> allSolvers;
  std::set<solverDescription_t> visitedSolvers;
  std::vector<solverDescription_t> remainingSolvers;
  std::set<ChebyshevSmootherType> visitedSmoothers;
  std::map<solverDescription_t, std::vector<double>> solverToTime;
  std::map<solverDescription_t, double> solverStartTime;
  std::map<solverDescription_t, unsigned int> solverToIterations;
};
#endif