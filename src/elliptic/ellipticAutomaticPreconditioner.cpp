#include "elliptic.h"
#include "ellipticAutomaticPreconditioner.h"
#include <random>
#include <limits>
#include <vector>
#include <algorithm>

automaticPreconditioner_t::automaticPreconditioner_t(elliptic_t& m_elliptic)
: elliptic(m_elliptic),
  solveCount(0),
  gen(rd())
{
  elliptic.options.getArgs("AUTO PRECONDITIONER TRIAL FREQUENCY", trialFrequency);
  elliptic.options.getArgs("AUTO PRECONDITIONER MAX CHEBY ORDER", maxChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER MIN CHEBY ORDER", minChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER MAX TRIALS", maxTrials);
  const std::string sampling = 
    elliptic.options.getArgs("AUTO PRECONDITIONER SAMPLING");

  constexpr unsigned NSmoothers {1}; // Whatever is currently the selected smoother
  for(unsigned smoother = 0; smoother < NSmoothers; ++smoother)
  {
    for(unsigned chebyOrder = minChebyOrder; chebyOrder <= maxChebyOrder; ++chebyOrder)
    {
      allSolvers.insert({smoother, chebyOrder});
    }
  }
  if(sampling == "RANDOM+REPLACEMENT")
    strategy = Strategy::RANDOM_SAMPLE;
  if(sampling == "RANDOM")
    strategy = Strategy::RANDOM_SAMPLE_NO_REPLACEMENT;
}

void
automaticPreconditioner_t::apply()
{
  if(trialCount > maxTrials) return;
  solveCount++;
  if(solveCount % trialFrequency == 0){
    trialCount++;
    evaluate_current_solver();
    select_solver();
  }
}

void
automaticPreconditioner_t::evaluate_current_solver()
{
  const dfloat currentSolverTime = 
    platform->timer.query(elliptic.name + std::string("Solve"), "DEVICE:MAX");
  const dfloat lastRecordedTime = solverToTime[currentSolver];
  const dfloat elapsed = currentSolverTime - lastRecordedTime;
  solverToTime[currentSolver] = elapsed;
}

void
automaticPreconditioner_t::select_solver()
{
  if(trialCount > maxTrials)
  {
    currentSolver = fastest_solver();
  } else {
    if(strategy == Strategy::RANDOM_SAMPLE){
      std::uniform_int_distribution<> dist(0, allSolvers.size()-1);
      unsigned randomIndex = dist(gen);
      auto it = allSolvers.begin();
      std::advance(it, randomIndex);
      currentSolver = *it;
    }
    else if(strategy == Strategy::RANDOM_SAMPLE_NO_REPLACEMENT){
      std::vector<solverDescription_t> unvisitedSolvers;
      std::set_difference(allSolvers.begin(), allSolvers.end(), 
        visitedSolvers.begin(), visitedSolvers.end(),
        std::inserter(unvisitedSolvers, unvisitedSolvers.begin()));

      std::uniform_int_distribution<> dist(0, unvisitedSolvers.size()-1);
      unsigned randomIndex = dist(gen);

      currentSolver = unvisitedSolvers[randomIndex];
    }

    visitedSolvers.insert(currentSolver);
    const dfloat currentSolverTime = 
      platform->timer.query(elliptic.name + std::string("Solve"), "DEVICE:MAX");
    solverToTime[currentSolver] = currentSolverTime;

    reinitializePreconditioner();
  }
}

solverDescription_t
automaticPreconditioner_t::fastest_solver()
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

  return minSolver;
}

void
automaticPreconditioner_t::reinitializePreconditioner()
{
  auto** levels = elliptic.precon->parAlmond->levels;
  for(int levelIndex = 0; levelIndex < elliptic.nLevels; ++levelIndex)
  {
    auto level = dynamic_cast<MGLevel*>(levels[levelIndex]);
    level->ChebyshevIterations = currentSolver.chebyOrder;
#ifdef READY_FOR_SMOOTHERS
    if(currentSolver.smoother == 1 || currentSolver.smoother == 2){
      level->smtypeDown = SecondarySmootherType::SCHWARZ;
      level->smtypeUp = SecondarySmootherType::SCHWARZ;
      if(currentSolver.smoother == 1){
        options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+ASM");
      }
      if(currentSolver.smoother == 2){
        options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+RAS");
      }

    } else {
      level->smtypeDown = SecondarySmootherType::JACOBI;
      level->smtypeUp = SecondarySmootherType::JACOBI;
      options.setArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI,CHEBYSHEV");
    }
#endif
  }

}