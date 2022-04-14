#include "occa.hpp"

template <typename T>
occa::kernel benchmarkAdvsub(int Nelements,
                             int Nq,
                             int cubNq,
                             bool overlap,
                             int verbosity,
                             T NtestsOrTargetTime,
                             bool requiresBenchmark);