#include <observer.hpp>

observer_t *observer_t::singleton = nullptr;

observer_t *observer_t::get()
{
  if (!singleton)
    singleton = new observer_t();
  return singleton;
}

int observer_t::count(std::string entry) const
{
  if (operatorCounts.count(entry)) {
    return operatorCounts.at(entry);
  }
  return 0;
}

void observer_t::increment(std::string entry) { operatorCounts[entry]++; }
void observer_t::clear() { operatorCounts.clear(); }