#include "elliptic.h"
#include "ellipticMultiGrid.h"
#include "ellipticAutomaticPreconditioner.h"
#include <random>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <nrs.hpp>

automaticPreconditioner_t::automaticPreconditioner_t(elliptic_t &m_elliptic)
    : elliptic(m_elliptic), activeTuner(true), sampleCounter(0)
{

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

  std::set<SmootherType> allChebyshevSmoothers = {
      SmootherType::CHEBYSHEV,
      SmootherType::OPT_CHEBYSHEV,
      SmootherType::FOURTH_CHEBYSHEV,
  };

  std::vector<std::set<unsigned>> schedules;
  for (int pass = 0; pass < 2; ++pass) {
    std::set<unsigned> levels;
    auto vLevels = determineMGLevels("pressure", pass);
    for (auto &&level : vLevels)
      levels.insert(level);
    schedules.push_back(levels);
  }

  defaultSolver = {SmootherType::CHEBYSHEV, ChebyshevSmootherType::ASM, 2, schedules[0]};

  for (auto &&schedule : schedules) {
    for (auto &&chebyshevSmoother : allChebyshevSmoothers) {
      for (auto &&smoother : allSmoothers) {
        for (unsigned chebyOrder = minChebyOrder; chebyOrder <= maxChebyOrder; ++chebyOrder) {
          allSolvers.insert({chebyshevSmoother, smoother, chebyOrder, schedule});
          solverToTime[{chebyshevSmoother, smoother, chebyOrder, schedule}] =
              std::vector<double>(NSamples, -1.0);
          solverTimePerIter[{chebyshevSmoother, smoother, chebyOrder, schedule}] =
              std::vector<double>(NSamples, -1.0);
          solverToIterations[{chebyshevSmoother, smoother, chebyOrder, schedule}] =
              std::vector<unsigned int>(NSamples, 0);
        }
      }
    }
  }

  auto **levels = elliptic.precon->parAlmond->levels;
  for (int levelIndex = 0; levelIndex < elliptic.nLevels; ++levelIndex) {
    auto level = dynamic_cast<MGLevel *>(levels[levelIndex]);
    const auto degree = level->degree;
    multigridLevels[degree] = level;
  }
}

bool automaticPreconditioner_t::apply(int tstep)
{
  bool evaluatePreconditioner = true;
  evaluatePreconditioner &= activeTuner;
  evaluatePreconditioner &= tstep >= autoStart;
  if (evaluatePreconditioner) {
    evaluatePreconditioner &= (tstep - autoStart) % trialFrequency == 0;
  }

#if 0
  if(platform->comm.mpiRank == 0){
    std::cout << "tstep = " << tstep;
    std::cout << ", autoStart = " << autoStart;
    std::cout << ", activeTuner = " << std::boolalpha << activeTuner;
    std::cout << ", trialFrequency = " << trialFrequency;
    std::cout << ", evaluatePreconditioner = " << std::boolalpha << evaluatePreconditioner << "\n";
  }
#endif

  if (evaluatePreconditioner) {
    evaluatePreconditioner = selectSolver();
    if (evaluatePreconditioner) {
      const dfloat currentSolverTime = platform->timer.query("autoPreconditioner", "DEVICE:MAX");
      solverStartTime[currentSolver] = currentSolverTime;
      platform->timer.tic("autoPreconditioner", 1);
    }
  }
  return evaluatePreconditioner;
}

void automaticPreconditioner_t::measure(bool evaluatePreconditioner)
{
  if (evaluatePreconditioner) {
    platform->timer.toc("autoPreconditioner");
    const dfloat currentSolverTime = platform->timer.query("autoPreconditioner", "DEVICE:MAX");
    const dfloat lastRecordedTime = solverStartTime[currentSolver];
    const dfloat elapsed = currentSolverTime - lastRecordedTime;
    solverToTime[currentSolver][sampleCounter] = elapsed;
    solverToIterations[currentSolver][sampleCounter] = elliptic.Niter;
    solverTimePerIter[currentSolver][sampleCounter] = elapsed / elliptic.Niter;
  }
}

bool automaticPreconditioner_t::selectSolver()
{
  std::vector<solverDescription_t> remainingSolvers;
  std::set_difference(allSolvers.begin(),
                      allSolvers.end(),
                      visitedSolvers.begin(),
                      visitedSolvers.end(),
                      std::inserter(remainingSolvers, remainingSolvers.begin()));
  if (remainingSolvers.empty()) {
    if (sampleCounter == (NSamples - 1)) {
      currentSolver = determineFastestSolver();
      if (platform->comm.mpiRank == 0 && sampleCounter == (NSamples - 1)) {
        std::cout << this->to_string() << std::endl;
        std::cout << "Fastest solver : " << currentSolver.to_string() << "\n";
        fflush(stdout);
      }
    }
    else {
      currentSolver = defaultSolver;
    }
    reinitializePreconditioner();
    visitedSolvers.clear();

    if (sampleCounter == (NSamples - 1)) {
      sampleCounter = 0;
      return false;
    }
    else {
      sampleCounter++;
      return true; // <-
    }
  }
  else {

    currentSolver = remainingSolvers.back();
    visitedSolvers.insert(currentSolver);
    reinitializePreconditioner();
  }
  return true;
}

solverDescription_t automaticPreconditioner_t::determineFastestSolver()
{
  dfloat minSolveTime = std::numeric_limits<dfloat>::max();
  solverDescription_t minSolver;

  // minimize min solve time
  for (auto &&solver : visitedSolvers) {
    const auto &times = solverTimePerIter[solver];
    const auto &iters = solverToIterations[solver];
    dfloat solveTime = std::numeric_limits<dfloat>::max();
    for (auto &&tSolve : times) {
      solveTime = solveTime < tSolve ? solveTime : tSolve;
    }

    solverToEval[solver] = solveTime;
    if (solveTime < minSolveTime) {
      minSolveTime = solveTime;
      minSolver = solver;
    }
  }

  return minSolver;
}

void automaticPreconditioner_t::reinitializePreconditioner()
{
  dfloat minMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minMultiplier);

  dfloat maxMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxMultiplier);

  // reset eigenvalue multipliers for all levels
  for (auto &&orderAndLevelPair : multigridLevels) {
    auto level = orderAndLevelPair.second;

    level->stype = currentSolver.chebyshevSmoother;
    level->chebyshevSmoother = currentSolver.smoother;
    level->ChebyshevIterations = currentSolver.chebyOrder;

    // re-initialize betas_opt, betas_fourth due to change in Chebyshev order
    level->betas_opt = optimalCoeffs(level->ChebyshevIterations);
    level->betas_fourth = std::vector<pfloat>(level->betas_opt.size(), 1.0);

    if (currentSolver.smoother == ChebyshevSmootherType::ASM ||
        currentSolver.smoother == ChebyshevSmootherType::RAS) {
      if (currentSolver.smoother == ChebyshevSmootherType::ASM) {
        elliptic.options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+ASM");
        const dfloat rho = level->lambdaMax[0];
        level->lambda1 = maxMultiplier * rho;
        level->lambda0 = minMultiplier * rho;
      }
      if (currentSolver.smoother == ChebyshevSmootherType::RAS) {
        elliptic.options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+RAS");
        const dfloat rho = level->lambdaMax[1];
        level->lambda1 = maxMultiplier * rho;
        level->lambda0 = minMultiplier * rho;
      }
    }
    else {
      elliptic.options.setArgs("MULTIGRID SMOOTHER", "DAMPEDJACOBI,CHEBYSHEV");
      const dfloat rho = level->lambdaMax[2];
      level->lambda1 = maxMultiplier * rho;
      level->lambda0 = minMultiplier * rho;
    }
  }

  elliptic.precon->parAlmond->baseLevel = currentSolver.schedule.size() - 1;
  unsigned ctr = 0;
  std::vector<int> orders;
  for (auto &&it = currentSolver.schedule.rbegin(); it != currentSolver.schedule.rend(); ++it) {
    elliptic.precon->parAlmond->levels[ctr] = this->multigridLevels[*it];
    orders.push_back(*it);
    ctr++;
  }

  const std::string suffix = "Hex3D";
  // also need the correct kernels, too
  // reconstruct coarsening/prolongation operators
  for (int levelIndex = 1; levelIndex < orders.size(); ++levelIndex) {
    const auto Nf = orders[levelIndex - 1];
    const auto Nc = orders[levelIndex];
    const std::string kernelSuffix =
        std::string("_Nf_") + std::to_string(Nf) + std::string("_Nc_") + std::to_string(Nc);
    auto *level = this->multigridLevels[Nc];
    auto *fineLevel = this->multigridLevels[Nf];
    level->NpF = (Nf + 1) * (Nf + 1) * (Nf + 1);
    level->buildCoarsenerQuadHex(Nf, Nc);
    level->o_invDegree = fineLevel->elliptic->o_invDegree;
    std::string kernelName = "ellipticPreconCoarsen" + suffix;
    level->elliptic->precon->coarsenKernel = platform->kernels.get(kernelName + kernelSuffix);
    kernelName = "ellipticPreconProlongate" + suffix;
    level->elliptic->precon->prolongateKernel = platform->kernels.get(kernelName + kernelSuffix);
  }
}

std::string automaticPreconditioner_t::to_string() const
{

  auto findLengthCol = [&](auto& solverToColFunc, std::string colHeader){
    auto maxColLength = colHeader.length();
    for(auto&& solver : visitedSolvers){
      maxColLength = std::max(maxColLength, solverToColFunc(solver).length());
    }
    return maxColLength;
  };

  const std::string solverDescriptionHeader = "Preconditioner";
  const std::string iterationHeader = "Iterations";
  const std::string timeHeader = "Time (min/max/avg) (s)";

  auto solverDescriptionWriter = [&](auto& solver){
    return solver.to_string();
  };

  auto iterationWriter = [&](auto& solver){
    std::ostringstream iterationsText;
    iterationsText << "(";
    bool first = true;
    for(auto&& iter : solverToIterations.at(solver)){
      if(!first) iterationsText << ", ";
      iterationsText << iter;
      first = false;
    }
    iterationsText << ")";
    return iterationsText.str();
  };

  auto timeWriter = [&](auto& solver){
    dfloat avgTime = 0.0;
    dfloat minTime = std::numeric_limits<dfloat>::max();
    dfloat maxTime = -1.0 * std::numeric_limits<dfloat>::max();
    int n = 1;
    auto &times = solverToTime.at(solver);
    for (auto&& tSolve : times){
      minTime = std::min(minTime, tSolve);
      maxTime = std::max(maxTime, tSolve);
      // on-line algorithm for computing arithmetic mean
      avgTime += (tSolve - avgTime)/n;
      n++;
    }
    std::ostringstream timeText;
    timeText << std::setprecision(2) << minTime << "/" << maxTime << "/" << avgTime;
    return timeText.str();
  };

  std::vector<long unsigned int> colLengths = {
    findLengthCol(solverDescriptionWriter, solverDescriptionHeader),
    findLengthCol(iterationWriter, iterationHeader),
    findLengthCol(timeWriter, timeHeader),
  };

  auto printEntry = [&](auto& solver){
    std::ostringstream entry;
    entry << "| " << std::left << std::setw(colLengths[0]) << solverDescriptionWriter(solver) << " ";
    entry << "| " << std::left << std::setw(colLengths[1]) << iterationWriter(solver) << " ";
    entry << "| " << std::left << std::setw(colLengths[2]) << timeWriter(solver) << " |";
    return entry.str();
  };

  // find length of line
  const auto lineLength = printEntry(*visitedSolvers.begin()).length();
  const std::string hline(lineLength, '=');

  std::ostringstream tableOutput;
  tableOutput << hline << "\n";
  tableOutput << "| " << std::left << std::setw(colLengths[0]) << solverDescriptionHeader << " ";
  tableOutput << "| " << std::left << std::setw(colLengths[1]) << iterationHeader << " ";
  tableOutput << "| " << std::left << std::setw(colLengths[2]) << timeHeader << " |\n";
  tableOutput << hline << "\n";

  for(auto&& entry : visitedSolvers){
    tableOutput << printEntry(entry) << "\n";
  }

  tableOutput << hline << "\n";

  return tableOutput.str();
}