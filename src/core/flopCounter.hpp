#if !defined(nekrs_counter_hpp_)
#define nekrs_counter_hpp_
#include "nrssys.hpp"
#include <map>
#include <vector>
class flopCounter_t {
public:
  void clear();
  void add(const std::string &entry, dfloat work);

  dfloat count(const std::string &entry, MPI_Comm comm) const;

  std::vector<std::string> entries() const;

  // Note: must be called collectively
  dfloat count(MPI_Comm comm) const;

private:
  std::map<std::string, dfloat> flopMap;
};
#endif