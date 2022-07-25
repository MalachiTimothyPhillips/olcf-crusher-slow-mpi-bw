#include "occa.hpp"

template <typename T>
occa::kernel benchmarkFDM(int Nelements,
                          int Nq_e,
                          int wordSize,
                          bool useRAS,
                          int verbosity,
                          T NtestsOrTargetTime,
                          bool requiresBenchmark,
                          std::string suffix);
