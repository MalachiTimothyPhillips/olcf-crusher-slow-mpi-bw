#include "benchmarkAdvsub.hpp"
#include <vector>
#include <numeric>
#include <iostream>
#include "nrs.hpp"

#include "randomVector.hpp"
#include "kernelBenchmarker.hpp"
#include "omp.h"

template <typename T>
occa::kernel benchmarkAdvsub(int Nelements,
                             int Nq,
                             int cubNq,
                             bool overlap,
                             int verbosity,
                             T NtestsOrTargetTime,
                             bool requiresBenchmark)
{
  // TODO: implement
  return occa::kernel();
}

template occa::kernel benchmarkAdvsub<int>(int Nelements,
                                           int Nq,
                                           int cubNq,
                                           bool overlap,
                                           int verbosity,
                                           int Ntests,
                                           bool requiresBenchmark);
template occa::kernel benchmarkAdvsub<double>(int Nelements,
                                              int Nq,
                                              int cubNq,
                                              bool overlap,
                                              int verbosity,
                                              double targetTime,
                                              bool requiresBenchmark);