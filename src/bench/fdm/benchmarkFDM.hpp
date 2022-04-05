#include "occa.hpp"
occa::kernel benchmarkFDM(const occa::properties& baseProps, int Nelements, int Nq,
  bool verbose = false,
  int Ntests = 0,
  double elapsedTarget = 1.0);