#include "elliptic.h"
#include "ellipticAutomaticPreconditioner.h"
#include <random>
#include <limits>
#include <vector>
#include <algorithm>

automaticPreconditioner_t::automaticPreconditioner_t(elliptic_t& m_elliptic)
: elliptic(m_elliptic),
  activeTuner(true)
{
  elliptic.options.getArgs("AUTO PRECONDITIONER START", autoStart);
  elliptic.options.getArgs("AUTO PRECONDITIONER TRIAL FREQUENCY", trialFrequency);
  elliptic.options.getArgs("AUTO PRECONDITIONER MAX CHEBY ORDER", maxChebyOrder);
  elliptic.options.getArgs("AUTO PRECONDITIONER MIN CHEBY ORDER", minChebyOrder);
  
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
    evaluateCurrentSolver();
    evaluatePreconditioner = selectSolver();
  }
  return evaluatePreconditioner;
}

void
automaticPreconditioner_t::evaluateCurrentSolver()
{
  const dfloat currentSolverTime = 
    platform->timer.query(elliptic.name + std::string("Solve"), "DEVICE:MAX");
  const dfloat lastRecordedTime = solverStartTime[currentSolver];
  const dfloat elapsed = currentSolverTime - lastRecordedTime;
  solverToTime[currentSolver] = elapsed;
  solverToIterations[currentSolver] = elliptic.Niter;
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
    if(platform->comm.mpiRank == 0){
      std::cout << "Determined fastest solver is : " << currentSolver.to_string() << "\n";

      std::cout << "Summary of times:\n";
      std::cout << this->to_string() << std::endl;
      fflush(stdout);
    }
    visitedSolvers.clear();
    return false;
  } else {

    currentSolver = remainingSolvers.back();

    if(platform->comm.mpiRank == 0){
      std::cout << "Evaluating : " << currentSolver.to_string() << std::endl;
    }
    fflush(stdout);

    visitedSolvers.insert(currentSolver);
    const dfloat currentSolverTime = 
      platform->timer.query(elliptic.name + std::string("Solve"), "DEVICE:MAX");
    solverStartTime[currentSolver] = currentSolverTime;

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
  std::cout << "===============================================================\n";
  std::cout << "| Preconditioner       |  Niter  |          Time              |\n";
  for(auto && solver : visitedSolvers){
    ss << "Solver : " << solver.to_string() << " took " << solverToTime.at(solver) << " s\n";
  }
  std::cout << "===============================================================\n";
  std::cout.unsetf(std::ios::scientific);
  std::cout.precision(outPrecisionSave);

  return ss.str();
}