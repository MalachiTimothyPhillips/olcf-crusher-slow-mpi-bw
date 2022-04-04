#include "kernelBenchmarker.hpp"
#include <limits>
#include "nrs.hpp"
namespace {
double run(int Nsamples, std::function<void(occa::kernel &)> kernelRunner, occa::kernel &kernel)
{
  platform->device.finish();
  MPI_Barrier(platform->comm.mpiComm);
  const double start = MPI_Wtime();

  for (int test = 0; test < Nsamples; ++test) {
    kernelRunner(kernel);
  }

  platform->device.finish();
  MPI_Barrier(platform->comm.mpiComm);
  return (MPI_Wtime() - start) / Nsamples;
}
} // namespace
std::pair<occa::kernel, double>
benchmarkKernel(std::function<occa::kernel(int kernelVariant)> kernelBuilder,
                std::function<void(occa::kernel &)> kernelRunner,
                std::function<void(int kernelVariant, double tKernel)> kernelTimingCallback,
                const std::vector<int> &kernelVariants)
{
  occa::kernel fastestKernel;
  double fastestTime = std::numeric_limits<double>::max();
  for (auto &&kernelVariant : kernelVariants) {

    auto candidateKernel = kernelBuilder(kernelVariant);

    // warmup
    double elapsed = run(10, kernelRunner, candidateKernel);

    // evaluation
    const double targetTime = 1.0;
    const int Ntests = static_cast<int>(targetTime / elapsed);
    const double candidateKernelTiming = run(Ntests, kernelRunner, candidateKernel);

    if (candidateKernelTiming < fastestTime) {
      fastestTime = candidateKernelTiming;
      fastestKernel = candidateKernel;
    }

    kernelTimingCallback(kernelVariant, candidateKernelTiming);
  }

  return std::make_pair(fastestKernel, fastestTime);
}