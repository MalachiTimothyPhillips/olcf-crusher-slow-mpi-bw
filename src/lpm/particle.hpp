#if !defined(nekrs_particle_hpp_)
#define nekrs_particle_hpp_

#include <array>

#include "nrs.hpp"
#include "gslib.h"
#include "ogsFindpts.hpp"

#include "pointInterpolation.hpp"

struct particle_t {
  static constexpr int integrationOrder{3};
  dfloat x, y, z;
  std::array<dfloat, 3> r;
  dlong code;
  dlong proc;
  dlong el;
  std::array<dfloat, 3 * integrationOrder> v;

  particle_t() {}

  particle_t(dfloat x_, dfloat y_, dfloat z_, 
             dlong code_,
             dlong proc_,
             dlong el_,
             std::array<dfloat, 3> r_,
             std::array<dfloat, 3 * integrationOrder> v_)
      : code(code_), proc(proc_), el(el_), v(v_), x(x_), y(y_), z(z_), r(r_)
  {
  }

  };

  // Contains a set of particles and the information needed to interpolate on the mesh
  class lpm_t {
  public:

    lpm_t(nrs_t *nrs_, double newton_tol_) : interp_(new pointInterpolation_t(nrs_, newton_tol_)) {}

    lpm_t(lpm_t &set) = delete;

    ~lpm_t() {}

    particle_t operator[](int i) const
    {
      const dfloat xp = x(i);
      const dfloat yp = y(i);
      const dfloat zp = z(i);
      return particle_t(xp, yp, zp, code[i], proc[i], el[i], r[i], v[i]);
    }

    dlong size() const { return _x.size(); }

    //// Set management ////
    void reserve(int n);

    dlong capacity() const { return _x.capacity(); }

    void push(particle_t particle);

    particle_t remove(int i);

    void swap(int i, int j);

    void write(dfloat time) const;
    void update(occa::memory o_fld, dfloat *dt, int tstep);

    dfloat& x(int i) { return _x[i]; }
    const dfloat& x(int i) const { return _x[i]; }
    dfloat& y(int i) { return _y[i]; }
    const dfloat& y(int i) const { return _y[i]; }
    dfloat& z(int i) { return _z[i]; }
    const dfloat& z(int i) const { return _z[i]; }

  private:

    static constexpr bool profile = true; // toggle for timing blocks

    // helper class to pass components of a single particle
    std::shared_ptr<pointInterpolation_t> interp_;

    std::vector<dfloat> _x;
    std::vector<dfloat> _y;
    std::vector<dfloat> _z;

    std::vector<dlong> code;
    std::vector<dlong> proc;
    std::vector<dlong> el;
    std::vector<std::array<dfloat, 3*particle_t::integrationOrder>> v;
    std::vector<std::array<dfloat, 3>> r;

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

  };

#endif
