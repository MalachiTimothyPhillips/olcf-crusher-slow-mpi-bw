#include "occa.hpp"
occa::kernel benchmarkAx(int Nelements, int Nq, int Ng, 
  bool constCoeff,
  bool poisson,
  bool computeGeom,
  int wordSize,
  int Ndim,
  int verbosity,
  int Ntests = 0,
  double elapsedTarget = 1.0,
  bool requiresBenchmark = false);