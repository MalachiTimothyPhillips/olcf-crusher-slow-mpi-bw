#include "kernelTuner.hpp"
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
std::pair<occa::kernel, double> tuneKernel(std::function<occa::kernel(occa::properties &props)> kernelBuilder,
                                           std::function<void(occa::kernel &)> kernelRunner,
                                           occa::properties baseProps,
                                           int NKernels)
{
  occa::kernel fastestKernel;
  double fastestTime = std::numeric_limits<double>::max();
  for (int kernelNumber = 0; kernelNumber < NKernels; ++kernelNumber) {
    auto props = baseProps;
    props["defines/p_knl"] = kernelNumber;

    auto candidateKernel = kernelBuilder(props);

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
  }

  return std::make_pair(fastestKernel, fastestTime);
}