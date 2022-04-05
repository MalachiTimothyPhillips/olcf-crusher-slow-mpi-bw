#include "occa.hpp"
struct mesh_t;
occa::kernel benchmarkAx(const occa::properties& baseProps, int Nelements, int Nq,
  bool verbose = false,
  int Ntests = 0,
  double elapsedTarget = 1.0);