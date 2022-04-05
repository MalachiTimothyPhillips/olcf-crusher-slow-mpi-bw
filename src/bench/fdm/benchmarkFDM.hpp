#include "occa.hpp"
occa::kernel benchmarkFDM(int Nelements, int Nq_e,
  int wordSize,
  bool useRAS,
  bool overlap,
  int verbosity = 0,
  int Ntests = 0,
  double elapsedTarget = 1.0);