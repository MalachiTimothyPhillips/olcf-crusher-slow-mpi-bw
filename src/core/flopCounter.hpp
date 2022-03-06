#if !defined(nekrs_counter_hpp_)
#define nekrs_counter_hpp_
#include "nrssys.hpp"
#include <map>
#include <vector>
class flopCounter_t {
public:
  void clear();
  void addWork(const std::string &entry, dfloat work);
  dfloat count(const std::string &entry) const;

  std::vector<std::string> entries() const;

  // all entries
  dfloat count() const;

private:
  std::map<std::string, dfloat> flopMap;
};
#endif