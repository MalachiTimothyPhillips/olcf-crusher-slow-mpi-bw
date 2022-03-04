#include <mpi.h>
#include "flopCounter.hpp"

void flopCounter_t::logWork(const std::string &entry, dfloat work)
{
  if (!flopMap.count(entry)) {
    flopMap[entry] = 0.0;
  }
  flopMap[entry] += work;
}

dfloat flopCounter_t::flopCount(const std::string &entry) const { return flopMap.at(entry); }

dfloat flopCounter_t::flopCount() const
{
  dfloat total = 0.0;
  for (auto const &entry : flopMap) {
    total += entry.second;
  }

  MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm);
  return total;
}

void flopCounter_t::clear() { flopMap.clear(); }