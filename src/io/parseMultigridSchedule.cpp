#include "parseMultigridSchedule.hpp"
#include "nrs.hpp"

#include <set>
#include <limits>

std::pair<std::map<std::pair<int, bool>, int>, std::string>
parseMultigridSchedule(const std::string &schedule)
{
  std::string errorString;
  std::map<std::pair<int, bool>, int> scheduleMap;
  auto entries = serializeString(schedule, ',');

  const auto INVALID = -std::numeric_limits<int>::max();
  int prevOrder = std::numeric_limits<int>::max();
  int minOrder = std::numeric_limits<int>::max();
  bool downLeg = true;

  for (auto &&entry : entries) {
    auto tokens = serializeString(entry, '+');

    int order = INVALID;
    int degree = INVALID;
    for (auto &&token : tokens) {
      if (token.find("p") != std::string::npos) {
        order = std::stoi(serializeString(entry, '=').at(1));
        minOrder = std::min(minOrder, order);

        if (order > prevOrder) {
          downLeg = false;
        }
        prevOrder = order;
      }
      else if (token.find("degree") != std::string::npos) {
        degree = std::stoi(serializeString(entry, '=').at(1));
      }
      else {
        errorString += "ERROR: Unknown token '" + token + "' in schedule '" + schedule + "'!\n";
      }
    }

    if (order == -std::numeric_limits<int>::max()) {
      errorString += "ERROR: Order not specified in " + entry + "\n";
    }

    scheduleMap[{order, downLeg}] = degree;
  }

  // up leg and down leg orders must be identical
  std::set<int> downLegOrders;
  std::set<int> upLegOrders;
  for (auto &&entry : scheduleMap) {
    if (entry.first.second) {
      downLegOrders.insert(entry.first.first);
    }
    else {
      upLegOrders.insert(entry.first.first);
    }
  }

  if (downLegOrders != upLegOrders) {
    errorString += "ERROR: Down leg and up leg orders must be identical!\n";
  }

  // all orders, except the coarse grid, must have a degree associated with them
  for (auto &&entry : scheduleMap) {
    if (entry.first.first == minOrder) {
      continue;
    }
    if (entry.second == INVALID) {
      errorString += "ERROR: Degree not specified for order " + std::to_string(entry.first.first) + "!\n";
    }
  }

  return {scheduleMap, errorString};
}