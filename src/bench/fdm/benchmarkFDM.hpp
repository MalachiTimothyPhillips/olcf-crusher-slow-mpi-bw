#include "occa.hpp"
occa::kernel benchmarkFDM(int wordSize, int Nelements, int Nq_e,
  bool useRAS,
  bool overlap,
  int verbosity = 0,
  int Ntests = 0,
  double elapsedTarget = 1.0);