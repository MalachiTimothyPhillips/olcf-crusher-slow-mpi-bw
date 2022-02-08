#ifndef history_particle_hpp_
#define history_particle_hpp_
#include "particle.hpp"

class nrs_t;

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

using historyParticles_t = particle_set<historyData_t>;


void particleOut(const historyParticles_t& particles);
void particleUpdate(historyParticles_t& particles, nrs_t* nrs, int tstep);

#endif