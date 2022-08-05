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
    : elliptic(m_elliptic)
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

  std::vector<bool> usePostSmooth = {true, false};

  for (auto &&schedule : schedules) {
    for(auto && postSmooth : usePostSmooth){
      for (auto &&chebyshevSmoother : allChebyshevSmoothers) {
        for (auto &&smoother : allSmoothers) {
          for (unsigned chebyOrder = minChebyOrder; chebyOrder <= maxChebyOrder; ++chebyOrder) {
            allSolvers.insert({postSmooth, chebyshevSmoother, smoother, chebyOrder, schedule});
            solverToTime[{postSmooth, chebyshevSmoother, smoother, chebyOrder, schedule}] =
                std::vector<double>(NSamples, -1.0);
            solverToIterations[{postSmooth, chebyshevSmoother, smoother, chebyOrder, schedule}] =
                std::vector<unsigned int>(NSamples, 0);
          }
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

  o_rSave = platform->device.malloc(elliptic.Nfields * elliptic.Ntotal * sizeof(dfloat));
  o_xSave = platform->device.malloc(elliptic.Nfields * elliptic.Ntotal * sizeof(dfloat));

}

void automaticPreconditioner_t::saveState(occa::memory & o_r, occa::memory & o_x)
{
  o_rSave.copyFrom(o_r, elliptic.Nfields * elliptic.Ntotal * sizeof(dfloat));
  o_xSave.copyFrom(o_x, elliptic.Nfields * elliptic.Ntotal * sizeof(dfloat));
}

void automaticPreconditioner_t::restoreState(occa::memory & o_r, occa::memory & o_x)
{
  o_rSave.copyTo(o_r, elliptic.Nfields * elliptic.Ntotal * sizeof(dfloat));
  o_xSave.copyTo(o_x, elliptic.Nfields * elliptic.Ntotal * sizeof(dfloat));
}

void automaticPreconditioner_t::tune(int tstep, std::function<void(occa::memory & o_r, occa::memory & o_x)> solverFunc, occa::memory & o_r, occa::memory & o_x)
{
  if(tuneStep(tstep)){
    platform->timer.tic("autoPreconditioner", 1);

    const auto hPreco = platform->timer.hostElapsed(elliptic.name + " preconditioner");
    const auto dPreco = platform->timer.deviceElapsed(elliptic.name + " preconditioner");
    const auto cPreco = platform->timer.count(elliptic.name + " preconditioner");

    const auto hSmoother = platform->timer.hostElapsed(elliptic.name + " preconditioner smoother");
    const auto dSmoother = platform->timer.deviceElapsed(elliptic.name + " preconditioner smoother");
    const auto cSmoother = platform->timer.count(elliptic.name + " preconditioner smoother");

    const auto hCoarseGrid = platform->timer.hostElapsed("coarseSolve");
    const auto dCoarseGrid = platform->timer.deviceElapsed("coarseSolve");
    const auto cCoarseGrid = platform->timer.count("coarseSolve");

    this->saveState(o_r, o_x);

    // loop over every solver configuration, determining the one that minimizes
    // the time to solution
    solverDescription_t fastestSolver;
    double tFastestSolver = std::numeric_limits<double>::max();
    for(auto&& solver : allSolvers){
      reinitializePreconditioner(solver);

      double tMinSolve = std::numeric_limits<double>::max();

      for(int sample = 0; sample < NSamples; ++sample){

        this->restoreState(o_r, o_x);

        platform->device.finish();
        MPI_Barrier(platform->comm.mpiComm);
        const double start = MPI_Wtime();

        solverFunc(o_r, o_x);

        platform->device.finish();
        double elapsed = MPI_Wtime() - start;
        MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, platform->comm.mpiComm);

        tMinSolve = std::min(tMinSolve, elapsed);
        solverToTime[solver][sample] = elapsed;
        solverToIterations[solver][sample] = elliptic.Niter;
      }

      if(tMinSolve < tFastestSolver){
        tFastestSolver = tMinSolve;
        fastestSolver = solver;
      }
    }

    reinitializePreconditioner(fastestSolver);
    if (platform->comm.mpiRank == 0){
      std::cout << this->to_string() << std::endl;
      std::cout << "Fastest solver : " << fastestSolver.to_string() << "\n";
      std::cout << "tFastestSolver = " << tFastestSolver << "\n";
      std::cout << "min/max/avg (s) ";
      dfloat avgTime = 0.0;
      dfloat minTime = std::numeric_limits<dfloat>::max();
      dfloat maxTime = -1.0 * std::numeric_limits<dfloat>::max();
      int n = 1;
      auto &times = solverToTime.at(fastestSolver);
      for (auto&& tSolve : times){
        minTime = std::min(minTime, tSolve);
        maxTime = std::max(maxTime, tSolve);
        // on-line algorithm for computing arithmetic mean
        avgTime += (tSolve - avgTime)/n;
        n++;
      }
      std::cout << std::setprecision(2) << minTime << "/" << maxTime << "/" << avgTime;
      fflush(stdout);
    }

    // adjust timers for preconditioner, pMG smoother, and coarse grid
    platform->timer.resetState(elliptic.name + " preconditioner", cPreco, hPreco, dPreco);
    platform->timer.resetState(elliptic.name + " preconditioner smoother", cSmoother, hSmoother, dSmoother);
    platform->timer.resetState("coarseSolve", cCoarseGrid, hCoarseGrid, dCoarseGrid);

    // do final solve with fastest preconditioner
    this->restoreState(o_r, o_x);

    platform->timer.toc("autoPreconditioner");

    solverFunc(o_r, o_x);
  } else {
    // solve the step without any tuning
    solverFunc(o_r, o_x);
  }
}

bool automaticPreconditioner_t::tuneStep(int tstep) const
{
  bool evaluatePreconditioner = true;
  evaluatePreconditioner &= tstep >= autoStart;
  if (evaluatePreconditioner) {
    evaluatePreconditioner &= (tstep - autoStart) % trialFrequency == 0;
  }
  return evaluatePreconditioner; 
}

void automaticPreconditioner_t::reinitializePreconditioner(solverDescription_t solver)
{
  dfloat minMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MIN EIGENVALUE BOUND FACTOR", minMultiplier);

  dfloat maxMultiplier;
  elliptic.options.getArgs("MULTIGRID CHEBYSHEV MAX EIGENVALUE BOUND FACTOR", maxMultiplier);

  const int nPostSmoothing = solver.usePostSmoothing ? 1 : 0;
  elliptic.precon->parAlmond->options.setArgs("MULTIGRID NUMBER POST SMOOTHINGS", std::to_string(nPostSmoothing));

  // reset eigenvalue multipliers for all levels
  for (auto &&orderAndLevelPair : multigridLevels) {
    auto level = orderAndLevelPair.second;

    level->stype = solver.chebyshevSmoother;
    level->chebyshevSmoother = solver.smoother;

    // when omitting post smoothing, can apply higher-order smoother at the same cost/iteration
    level->ChebyshevIterations = solver.usePostSmoothing ? solver.chebyOrder
                                                                : 2 * solver.chebyOrder + 1;

    // re-initialize betas_opt, betas_fourth due to change in Chebyshev order
    level->betas_opt = optimalCoeffs(level->ChebyshevIterations);
    level->betas_fourth = std::vector<pfloat>(level->betas_opt.size(), 1.0);

    if (solver.smoother == ChebyshevSmootherType::ASM ||
        solver.smoother == ChebyshevSmootherType::RAS) {
      if (solver.smoother == ChebyshevSmootherType::ASM) {
        elliptic.options.setArgs("MULTIGRID SMOOTHER", "CHEBYSHEV+ASM");
        const dfloat rho = level->lambdaMax[0];
        level->lambda1 = maxMultiplier * rho;
        level->lambda0 = minMultiplier * rho;
      }
      if (solver.smoother == ChebyshevSmootherType::RAS) {
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

  elliptic.precon->parAlmond->baseLevel = solver.schedule.size() - 1;
  unsigned ctr = 0;
  std::vector<int> orders;
  for (auto &&it = solver.schedule.rbegin(); it != solver.schedule.rend(); ++it) {
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
    for(auto&& solver : allSolvers){
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
  const auto lineLength = printEntry(*allSolvers.begin()).length();
  const std::string hline(lineLength, '=');

  std::ostringstream tableOutput;
  tableOutput << hline << "\n";
  tableOutput << "| " << std::left << std::setw(colLengths[0]) << solverDescriptionHeader << " ";
  tableOutput << "| " << std::left << std::setw(colLengths[1]) << iterationHeader << " ";
  tableOutput << "| " << std::left << std::setw(colLengths[2]) << timeHeader << " |\n";
  tableOutput << hline << "\n";

  for(auto&& entry : allSolvers){
    tableOutput << printEntry(entry) << "\n";
  }

  tableOutput << hline << "\n";

  return tableOutput.str();
}