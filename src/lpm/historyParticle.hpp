#ifndef history_particle_hpp_
#define history_particle_hpp_
#include "particle.hpp"

class historyData_t {
public:
  dfloat v_hist[2][3];
  hlong  id;
  dfloat color;

  historyData_t()
  {
  }

  historyData_t(dfloat v_hist_[2][3], dfloat color_, hlong id_);
};

using historyParticle_t = particle_set<historyData_t>;

#endif