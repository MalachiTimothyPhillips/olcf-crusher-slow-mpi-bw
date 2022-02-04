#include "historyParticle.hpp"
historyData_t::historyData_t(dfloat v_hist_[2][3], dfloat color_, hlong id_)

{
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      v_hist[i][j] = v_hist_[i][j];
    }
  }
  id = id_;
  color = color_;
}