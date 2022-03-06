#include <mpi.h>
#include "flopCounter.hpp"
#include "platform.hpp"

void flopCounter_t::addWork(const std::string &entry, dfloat work)
{
  if (!flopMap.count(entry)) {
    flopMap[entry] = 0.0;
  }
  flopMap[entry] += work;
}

dfloat flopCounter_t::count(const std::string &entry) const { return flopMap.at(entry); }

dfloat flopCounter_t::count() const
{
  dfloat total = 0.0;
  for (auto const &entry : flopMap) {
    total += entry.second;
  }

  MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  return total;
}

void flopCounter_t::clear() { flopMap.clear(); }

std::vector<std::string> flopCounter_t::entries() const
{
  std::vector<std::string> loggedCategory;
  for (auto const &entry : flopMap) {
    loggedCategory.push_back(entry.first);
  }
  return loggedCategory;
}