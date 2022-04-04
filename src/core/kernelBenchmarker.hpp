#include "occa.hpp"
#include <utility>
#include <functional>

std::pair<occa::kernel, double>
benchmarkKernel(std::function<occa::kernel(int kernelVariant)> kernelBuilder,
                std::function<void(occa::kernel &)> kernelRunner,
                std::function<void(int kernelVariant, double tKernel, int Ntests)> kernelTimingCallback,
                const std::vector<int> &kernelVariants,
                int Ntests = 0,
                double targetTime = 1.0);