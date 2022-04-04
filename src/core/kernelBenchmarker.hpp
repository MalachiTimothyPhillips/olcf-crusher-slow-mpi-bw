#include "occa.hpp"
#include <utility>
#include <functional>

std::pair<occa::kernel, double> benchmarkKernel(std::function<std::string(int kernelNumber)> kernelNamer,
                                                std::function<void(occa::kernel &)> kernelRunner,
                                                int NKernels);