#include <elliptic.h>
#include <iostream>

std::string ellipticSolverDescription(elliptic_t *elliptic)
{
  std::ostringstream output;
  output << "Solver : " << elliptic->options.getArgs("KRYLOV SOLVER") << "\n";

  std::string preconditioner(elliptic->options.getArgs("PRECONDITIONER"));
  output << "Preconditioner : " << preconditioner;
  if (preconditioner.find("MULTIGRID") != std::string::npos) {
    output << ", ";
    std::string smoother(elliptic->options.getArgs("MULTIGRID SMOOTHER"));
    if (smoother.find("CHEBY") != std::string::npos) {
      output << "Cheby-";
      if (smoother.find("ASM") != std::string::npos)
        output << "ASM";
      if (smoother.find("RAS") != std::string::npos)
        output << "RAS";
      if (smoother.find("JAC") != std::string::npos)
        output << "JAC";
      output << "(" << elliptic->options.getArgs("MULTIGRID CHEBYSHEV DEGREE") << ")";
      output << ",(" << elliptic->levels[0];
      for (int level = 1; level < elliptic->nLevels; level++)
        output << "," << elliptic->levels[level];
      output << ")\n";
    }
    else {
      output << smoother << "\n";
    }
  }

  return output.str();
}