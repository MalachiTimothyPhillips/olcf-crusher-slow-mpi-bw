#include "flopCounter.hpp"

void flopCounter_t::logWork(const std::string &entry, dfloat work) { flopMap[entry] += work; }

dfloat flopCounter_t::flopCount(const std::string &entry) const { return flopMap.at(entry); }

dfloat flopCounter_t::flopCount() const
{
  dfloat total = 0.0;
  for (auto const &entry : flopMap) {
    total += entry.second;
  }
  return total;
}

void flopCounter_t::clear() { flopMap.clear(); }