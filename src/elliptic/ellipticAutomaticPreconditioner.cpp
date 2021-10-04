#include "elliptic.h"
#include "ellipticAutomaticPreconditioner.h"
#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <nrs.hpp>

automaticPreconditioner_t::automaticPreconditioner_t(elliptic_t& m_elliptic)
: elliptic(m_elliptic),
  activeTuner(true),
  sampleCounter(0)
{
  platform->timer.tic("autoPreconditioner", 1);
  elliptic.options.getArgs("AUTO PRECONDITIONER START", autoStart);
  elliptic.options.getArgs("AUTO PRECONDITIONER TRIAL FREQUENCY", trialFrequency);
  elliptic.options.getArgs("AUTO PRECONDITIONER MAX CHEBY ORDER", maxChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER MIN CHEBY ORDER", minChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER NUM SAMPLES", NSamples);
  
  std::set<ChebyshevSmootherType> allSmoothers = {
    ChebyshevSmootherType::JACOBI,
    ChebyshevSmootherType::ASM,
    ChebyshevSmootherType::RAS,
  };

  std::vector<std::set<unsigned>> schedules;
  for(int pass = 0; pass < 2; ++pass){
    std::set<unsigned> levels;
    auto vLevels = determineMGLevels("pressure", pass);
    for(auto&& level : vLevels)
      levels.insert(level);
    schedules.push_back(levels);
  }

  for(auto && schedule : schedules){
    for(auto && smoother : allSmoothers)
    {
      for(unsigned chebyOrder = minChebyOrder; chebyOrder <= maxChebyOrder; ++chebyOrder)
      {
        allSolvers.insert({smoother, chebyOrder, schedule});
        solverToTime[{smoother, chebyOrder, schedule}] = std::vector<double>(NSamples, -1.0);
      }
    }
  }
  platform->timer.toc("autoPreconditioner");
}

bool
automaticPreconditioner_t::apply(int tstep)
{
  bool evaluatePreconditioner = true;
  evaluatePreconditioner &= activeTuner;
  evaluatePreconditioner &= tstep >= autoStart;
  if(evaluatePreconditioner){
    evaluatePreconditioner &= (tstep - autoStart) % trialFrequency == 0;
  }

  if(evaluatePreconditioner){
    evaluatePreconditioner = selectSolver();
    if(evaluatePreconditioner){
      const dfloat currentSolverTime = 
        platform->timer.query("autoPreconditioner", "DEVICE:MAX");
      solverStartTime[currentSolver] = currentSolverTime;
      platform->timer.tic("autoPreconditioner", 1);
    }
  }
  return evaluatePreconditioner;
}

void
automaticPreconditioner_t::measure(bool evaluatePreconditioner)
{
  if(evaluatePreconditioner){
    platform->timer.toc("autoPreconditioner");
    const dfloat currentSolverTime = 
      platform->timer.query("autoPreconditioner", "DEVICE:MAX");
    const dfloat lastRecordedTime = solverStartTime[currentSolver];
    const dfloat elapsed = currentSolverTime - lastRecordedTime;
    solverToTime[currentSolver][sampleCounter] = elapsed;
    solverToIterations[currentSolver] = elliptic.Niter;
  }
}

bool
automaticPreconditioner_t::selectSolver()
{
  std::vector<solverDescription_t> remainingSolvers;
  std::set_difference(allSolvers.begin(), allSolvers.end(),
    visitedSolvers.begin(), visitedSolvers.end(),
    std::inserter(remainingSolvers, remainingSolvers.begin()));
  if(remainingSolvers.empty())
  {
    currentSolver = determineFastestSolver();
    if(platform->comm.mpiRank == 0 && sampleCounter == (NSamples-1)){
      std::cout << this->to_string() << std::endl;
      std::cout << "Fastest solver : " << currentSolver.to_string() << "\n";
      fflush(stdout);
    }
    visitedSolvers.clear();
    if(sampleCounter == (NSamples-1)){
      sampleCounter = 0;
      return false;
    } else {
      sampleCounter++;
      return true;
    }
  } else {

    currentSolver = remainingSolvers.back();
    visitedSolvers.insert(currentSolver);
    reinitializePreconditioner();
  }
  return true;
}

solverDescription_t
automaticPreconditioner_t::determineFastestSolver()
{
  dfloat minSolveTime = std::numeric_limits<dfloat>::max();
  solverDescription_t minSolver;

  for(auto&& solver : visitedSolvers){
    const auto& times = solverToTime[solver];
    dfloat solverTime = std::numeric_limits<dfloat>::max();
    for(int i = 0 ; i < NSamples; ++i){
      solverTime = solverTime < times.at(i) ? solverTime : times.at(i);
    }
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
  dfloat minMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minMultiplier);

  dfloat maxMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxMultiplier);
  auto** levels = elliptic.precon->parAlmond->levels;
  for(int levelIndex = 0; levelIndex < elliptic.nLevels; ++levelIndex)
  {
    auto level = dynamic_cast<MGLevel*>(levels[levelIndex]);
    level->active = true;
    auto levelOrder = elliptic.levels[levelIndex];
    if(currentSolver.schedule.count(levelOrder) == 0){
      level->active = false;
    }
    
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
  ss << "===================================================================\n";
  ss << "| " << std::internal << std::setw(36) << "Preconditioner" << " ";
  ss << "| " << std::internal << std::setw(5) << "Niter" << " ";
  ss << "| " << std::internal << std::setw(17) << "Time (min/max)" << " |\n";
  ss << "===================================================================\n";
  for(auto && solver : visitedSolvers){
    ss << "| " << std::internal << std::setw(36) << solver.to_string() << " ";
    ss << "| " << std::internal << std::setw(5) << solverToIterations.at(solver) << " ";
    dfloat minTime = std::numeric_limits<dfloat>::max();
    dfloat maxTime = -1.0 * std::numeric_limits<dfloat>::max();
    auto& times = solverToTime.at(solver);
    for(int i = 0; i < NSamples; ++i){
      minTime = minTime < times.at(i) ? minTime : times.at(i);
      maxTime = maxTime > times.at(i) ? maxTime : times.at(i);
    }
    ss << "| " << std::internal << std::setw(7) << std::setprecision(2) << std::scientific << minTime << "/";
    ss << std::internal << std::setw(7) << std::setprecision(2) << std::scientific << maxTime << " |\n";
  }
  ss << "===================================================================\n";

  return ss.str();
}