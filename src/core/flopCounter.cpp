#include <mpi.h>
#include "flopCounter.hpp"
#include "platform.hpp"
#include <array>

void flopCounter_t::add(const std::string &entry, dfloat work)
{
  if (!flopMap.count(entry)) {
    flopMap[entry] = 0.0;
  }
  flopMap[entry] += work;
}

dfloat flopCounter_t::count(const std::string &entry) const { return flopMap.at(entry); }

dfloat flopCounter_t::count() const
{
  dfloat err = 0;
  dfloat total = 0.0;
  for (auto const &entry : flopMap) {
    if (entry.second < 0.0)
      err += 1;
    total += entry.second;
  }

  std::array<dfloat, 2> errAndTotal = {err, total};
  MPI_Allreduce(MPI_IN_PLACE, errAndTotal.data(), 2, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);

  err = errAndTotal[0];
  total = errAndTotal[1];

  if (err > 0.0) {
    if (platform->comm.mpiRank == 0) {
      std::cout << "Encountered error in flopCounter_t::count" << std::endl;
    }
    ABORT(1)
  }

  return total;
}

void flopCounter_t::clear() { flopMap.clear(); }

std::vector<std::string> flopCounter_t::entries() const
{
  std::vector<std::string> loggedCategory;
  for (auto const &entry : flopMap) {
    loggedCategory.push_back(entry.first);
  }

  // sort by flops (largest first)
  std::sort(loggedCategory.begin(), loggedCategory.end(), [&](const std::string &a, const std::string &b) {
    return count(a) > count(b);
  });
  return loggedCategory;
}