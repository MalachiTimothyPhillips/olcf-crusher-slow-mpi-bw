#include "elliptic.h"
#include "ellipticAutomaticPreconditioner.h"
#include <random>
#include <limits>
#include <vector>
#include <algorithm>

automaticPreconditioner_t::automaticPreconditioner_t(elliptic_t& m_elliptic)
: elliptic(m_elliptic),
  solveCount(0),
  trialCount(0),
  gen(rd())
{
  converged = 0;
  elliptic.options.getArgs("AUTO PRECONDITIONER START", autoStart);
  elliptic.options.getArgs("AUTO PRECONDITIONER TRIAL FREQUENCY", trialFrequency);
  elliptic.options.getArgs("AUTO PRECONDITIONER MAX CHEBY ORDER", maxChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER MIN CHEBY ORDER", minChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER MAX TRIALS", maxTrials);
  const std::string sampling = 
    elliptic.options.getArgs("AUTO PRECONDITIONER SAMPLING");
  
  std::set<ChebyshevSmootherType> allSmoothers = {
    ChebyshevSmootherType::JACOBI,
    ChebyshevSmootherType::ASM,
    ChebyshevSmootherType::RAS,
  };

  for(auto && smoother : allSmoothers)
  {
    for(unsigned chebyOrder = minChebyOrder; chebyOrder <= maxChebyOrder; ++chebyOrder)
    {
      allSolvers.insert({smoother, chebyOrder});
    }
  }
  if(sampling == "STEPWISE")
    strategy = Strategy::STEPWISE;
  else
    strategy = Strategy::EXHAUSTIVE;
}

void
automaticPreconditioner_t::apply()
{
  if(trialCount >= maxTrials || converged > 1) return;
  if(solveCount % trialFrequency == 0 && solveCount >= autoStart){
    trialCount++;
    evaluateCurrentSolver();
    selectSolver();
  }
  solveCount++;
}

void
automaticPreconditioner_t::evaluateCurrentSolver()
{
  const dfloat currentSolverTime = 
    platform->timer.query(elliptic.name + std::string("Solve"), "DEVICE:MAX");
  const dfloat lastRecordedTime = solverToTime[currentSolver];
  const dfloat elapsed = currentSolverTime - lastRecordedTime;
  solverToTime[currentSolver] = elapsed;
}

solverDescription_t
automaticPreconditioner_t::stepwiseSelection()
{
  constexpr unsigned int evaluationChebyOrder {2};
  if(visitedSmoothers.size() < NSmoothers){
    std::set<ChebyshevSmootherType> allSmoothers = {
      ChebyshevSmootherType::JACOBI,
      ChebyshevSmootherType::ASM,
      ChebyshevSmootherType::RAS,
    };
    std::vector<ChebyshevSmootherType> unvisitedSmoothers;
    std::set_difference(allSmoothers.begin(), allSmoothers.end(),
      visitedSmoothers.begin(), visitedSmoothers.end(),
      std::inserter(unvisitedSmoothers, unvisitedSmoothers.begin()));
    auto smoother = unvisitedSmoothers.front();
    visitedSmoothers.insert(smoother);
    return {smoother, evaluationChebyOrder};
  }
  if(visitedChebyOrders.empty()){
    // choose fastest smoother for the evaluationChebyOrder
    dfloat fastestTime = std::numeric_limits<dfloat>::max();
    for(auto&& smoother : visitedSmoothers){
      const dfloat smootherTime = solverToTime[{smoother, evaluationChebyOrder}];
      if(smootherTime < fastestTime){
        fastestTime = smootherTime;
        fastestSmoother = smoother;
      }
    }

    visitedChebyOrders.insert(evaluationChebyOrder);
  }
  const unsigned int newOrder = 
    [&](){
      std::set<unsigned int> allOrders;
      for(unsigned int order = minChebyOrder; order <= maxChebyOrder; ++order){
        allOrders.insert(order);
      }
      std::vector<unsigned int> unvisitedOrders;
      std::set_difference(allOrders.begin(), allOrders.end(),
        visitedChebyOrders.begin(), visitedChebyOrders.end(),
        std::inserter(unvisitedOrders, unvisitedOrders.begin()));
      if(unvisitedOrders.empty()){
        dfloat fastestTime = std::numeric_limits<dfloat>::max();
        unsigned int fastestOrder = 0;
        for(auto && order : visitedChebyOrders){
          const dfloat smootherTime = solverToTime[{fastestSmoother, order}];
          if(smootherTime < fastestTime){
            fastestTime = smootherTime;
            fastestOrder = order;
          }
        }
        return fastestOrder;
      }
      return unvisitedOrders.front();
    }();
  visitedChebyOrders.insert(newOrder);
  return {fastestSmoother, newOrder};
}

solverDescription_t
automaticPreconditioner_t::exhaustiveSelection()
{
  std::vector<solverDescription_t> remainingSolvers;
  std::set_difference(allSolvers.begin(), allSolvers.end(),
    visitedSolvers.begin(), visitedSolvers.end(),
    std::inserter(remainingSolvers, remainingSolvers.begin()));
  
  // select fastest solver
  solverDescription_t candidateSolver;
  if(remainingSolvers.empty() || trialCount >= maxTrials)
  {
    dfloat fastestTime = std::numeric_limits<dfloat>::max();
    for(auto&& solver : visitedSolvers){
      const dfloat solverTime = solverToTime[solver];
      if(solverTime < fastestTime){
        fastestTime = solverTime;
        candidateSolver = solver;
      }
    }
    converged++;
  } else {
    candidateSolver = remainingSolvers.back();
  }
  
  return candidateSolver;
}

void
automaticPreconditioner_t::selectSolver()
{
  if(trialCount >= maxTrials || converged > 0)
  {
    currentSolver = determineFastestSolver();
    if(platform->comm.mpiRank == 0){
      std::cout << "Determined fastest solver is : " << currentSolver.to_string() << "\n";
      std::cout << "Will now continue using this solver for the remainder of the simulation!\n";

      std::cout << "Summary of times:\n";
      std::cout << this->to_string() << std::endl;
      fflush(stdout);
    }
  } else {
    if(strategy == Strategy::STEPWISE){
      currentSolver = stepwiseSelection();
    }
    else if (strategy == Strategy::EXHAUSTIVE){
      currentSolver = exhaustiveSelection();
    }

    if(platform->comm.mpiRank == 0){
      std::cout << "Evaluating : " << currentSolver.to_string() << std::endl;
    }
    fflush(stdout);

    visitedSolvers.insert(currentSolver);
    const dfloat currentSolverTime = 
      platform->timer.query(elliptic.name + std::string("Solve"), "DEVICE:MAX");
    solverToTime[currentSolver] = currentSolverTime;

    reinitializePreconditioner();
  }
}

solverDescription_t
automaticPreconditioner_t::determineFastestSolver()
{
  dfloat minSolveTime = std::numeric_limits<dfloat>::max();
  solverDescription_t minSolver;

  for(auto&& solver : visitedSolvers){
    dfloat solverTime = solverToTime[solver];
    if(solverTime < minSolveTime){
      minSolveTime = solverTime;
      minSolver = solver;
    }
  }

  converged++;

  return minSolver;
}

void
automaticPreconditioner_t::reinitializePreconditioner()
{
  dfloat minMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minMultiplier);

  dfloat maxMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxMultiplier);
  auto** levels = elliptic.precon->parAlmond->levels;
  for(int levelIndex = 0; levelIndex < elliptic.nLevels; ++levelIndex)
  {
    auto level = dynamic_cast<MGLevel*>(levels[levelIndex]);
    level->ChebyshevIterations = currentSolver.chebyOrder;
    if(currentSolver.smoother == ChebyshevSmootherType::ASM || currentSolver.smoother == ChebyshevSmootherType::RAS){
      level->chebyshevSmoother = currentSolver.smoother;
      if(currentSolver.smoother == ChebyshevSmootherType::ASM){
        elliptic.options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+ASM");
        const dfloat rho = level->lambdaMax[0];
        level->lambda1 = maxMultiplier * rho;
        level->lambda0 = minMultiplier * rho;
      }
      if(currentSolver.smoother == ChebyshevSmootherType::RAS){
        elliptic.options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+RAS");
        const dfloat rho = level->lambdaMax[1];
        level->lambda1 = maxMultiplier * rho;
        level->lambda0 = minMultiplier * rho;
      }

    } else {
      level->chebyshevSmoother = ChebyshevSmootherType::JACOBI;
      elliptic.options.setArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI,CHEBYSHEV");
      const dfloat rho = level->lambdaMax[2];
      level->lambda1 = maxMultiplier * rho;
      level->lambda0 = minMultiplier * rho;
    }
  }

}

std::string
automaticPreconditioner_t::to_string() const
{
  std::ostringstream ss;
  std::cout.setf(std::ios::scientific);
  int outPrecisionSave = std::cout.precision();
  std::cout.precision(5);
  for(auto && solver : visitedSolvers){
    ss << "Solver : " << solver.to_string() << " took " << solverToTime.at(solver) << " s\n";
  }
  std::cout.unsetf(std::ios::scientific);
  std::cout.precision(outPrecisionSave);

  return ss.str();
}