#if !defined(nekrs_particle_hpp_)
#define nekrs_particle_hpp_

#include <array>

#include "nrs.hpp"
#include "gslib.h"
#include "ogsFindpts.hpp"

#include "interp.hpp"

class historyData_t {
public:
  static constexpr int integrationOrder{3};
  dfloat v_hist[integrationOrder - 1][3];
  hlong id;
  dfloat color;

  historyData_t() {}

  historyData_t(dfloat v_hist_[integrationOrder - 1][3], dfloat color_, hlong id_)
  {
    for (int i = 0; i < integrationOrder - 1; ++i) {
      for (int j = 0; j < 3; ++j) {
        v_hist[i][j] = v_hist_[i][j];
      }
    }
    id = id_;
    color = color_;
  }
};

using Extra = historyData_t;

struct particle_t {
  std::array<dfloat, 3> x;
  std::array<dfloat, 3> r;
  dlong code;
  dlong proc;
  dlong el;
  Extra extra;

  particle_t() {}

  particle_t(std::array<dfloat, 3> x_,
             dlong code_,
             dlong proc_,
             dlong el_,
             std::array<dfloat, 3> r_,
             Extra extra_)
      : code(code_), proc(proc_), el(el_), extra(extra_)
  {
    for (int i = 0; i < 3; ++i) {
      x[i] = x_[i];
      r[i] = r_[i];
    }
  }

  };

  // Contains a set of particles and the information needed to interpolate on the mesh
  class particles_t {
  public:
    // helper class to pass components of a single particle
    std::shared_ptr<pointInterpolation_t> interp_;

    std::vector<dfloat> x[3];
    std::vector<dlong> code;
    std::vector<dlong> proc;
    std::vector<dlong> el;
    std::vector<Extra> extra;
    std::vector<std::array<dfloat, 3>> r;

    particles_t(nrs_t *nrs_, double newton_tol_) : interp_(new pointInterpolation_t(nrs_, newton_tol_)) {}

    particles_t(particles_t &set)
        : interp_(set.interp_), x(set.x), code(set.code), proc(set.proc), el(set.el), extra(set.extra),
          r(set.r)
    {
    }

    ~particles_t() {}

    //// Set management ////
    void reserve(int n);

    dlong size() const { return x[0].size(); }

    dlong capacity() const { return x[0].capacity(); }

    particle_t operator[](int i)
    {
      std::array<dfloat, 3> x_;
      for (int j = 0; j < 3; ++j) {
        x_[j] = x[j][i];
      }
      return particle_t(x_, code[i], proc[i], el[i], r[i], extra[i]);
    }

    void push(particle_t particle);

    particle_t remove(int i);

    void swap(int i, int j);

    //// particle operations ////

    // Locates the element and process for each particle
    void find(bool printWarnings = true, dfloat *dist2In = nullptr, dlong dist2Stride = 1);

    // Moves each particle to the process that owns it's current element
    // this->find must have been called since the last change in position
    void migrate();

    // Interpolates the fields at each particle with the assumption that all particles belong to local
    // elements this->migrate must have been called since the last change in position
    //   fld            ... source field(s)
    //   out            ... array of pointers to the output arrays (dfloat[n][3])
    //   nfld           ... number of fields
    void interpLocal(occa::memory field, dfloat *out[], dlong nFields);
    void interpLocal(dfloat *field, dfloat *out[], dlong nFields);

    void write();

    void update(occa::memory o_fld, dfloat *dt, int tstep);
  };

#endif
