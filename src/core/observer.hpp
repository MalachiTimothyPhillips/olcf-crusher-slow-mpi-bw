#ifndef observer_hpp_
#define observer_hpp_

#include <map>
#include <string>

class observer_t {

  static observer_t *get();

  int count(std::string entry) const;

  void increment(std::string entry);

private:
  static observer_t *singleton;

  observer_t() = default;

  std::map<std::string, int> operatorCounts;
};
#endif