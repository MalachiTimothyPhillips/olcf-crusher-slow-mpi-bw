#include "occa.hpp"
#include <utility>
#include <functional>

std::pair<occa::kernel, double>
benchmarkKernel(std::function<occa::kernel(int kernelVariant)> kernelBuilder,
                std::function<void(occa::kernel &)> kernelRunner,
                std::function<void(int kernelVariant, double tKernel)> kernelTimingCallback,
                const std::vector<int> &kernelVariants);