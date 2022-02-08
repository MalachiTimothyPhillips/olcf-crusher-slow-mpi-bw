#ifndef observer_hpp_
#define observer_hpp_

#include <map>
#include <string>

class observer_t {
public:
  static observer_t *get();

  int count(std::string entry) const;

  void increment(std::string entry);

  void clear();

  auto begin() const { return operatorCounts.begin(); }
  auto end() const { return operatorCounts.end(); }

private:
  static observer_t *singleton;

  observer_t() = default;

  std::map<std::string, int> operatorCounts;
};
#endif