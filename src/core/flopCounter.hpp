#if !defined(nekrs_flopCounter_hpp_)
#define nekrs_flopCounter_hpp_
#include "nrssys.hpp"
#include <map>
class flopCounter_t {
public:
  void clear();
  void logWork(const std::string &entry, dfloat work);
  dfloat flopCount(const std::string &entry) const;

  // all entries
  dfloat flopCount() const;

private:
  std::map<std::string, dfloat> flopMap;
};
#endif