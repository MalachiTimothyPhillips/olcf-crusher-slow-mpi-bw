#include "occa.hpp"
struct mesh_t;
occa::kernel benchmarkAx(const occa::properties& baseProps, const mesh_t& mesh,
  bool verbose = false,
  int Ntests = 0,
  double elapsedTarget = 1.0);