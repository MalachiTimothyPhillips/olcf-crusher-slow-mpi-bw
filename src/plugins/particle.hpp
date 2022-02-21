#if !defined(nekrs_particle_hpp_)
#define nekrs_particle_hpp_

#include <array>

#include "nrs.hpp"
#include "gslib.h"
#include "findpts.hpp"

#include "pointInterpolation.hpp"
#include <functional>

// TODO: need to page align occa::memory objects...
// TODO: ask Stefan about adding function ptr

struct particle_t {
  static constexpr int integrationOrder{3};
  dfloat x, y, z;
  dfloat r, s, t;
  dlong code;
  dlong proc;
  dlong el;
  dlong id;
  std::array<dfloat, 3 * integrationOrder> v;

  particle_t() {}

  particle_t(dfloat x_, dfloat y_, dfloat z_, 
             dlong code_,
             dlong proc_,
             dlong el_,
             dlong id_,
             dfloat r_, dfloat s_, dfloat t_,
             std::array<dfloat, 3 * integrationOrder> v_)
      : code(code_), proc(proc_), el(el_), id(id_), v(v_), x(x_), y(y_), z(z_), r(r_), s(s_), t(t_)
  {
  }

  };

  // Contains a set of particles and the information needed to interpolate on the mesh
  class lpm_t {
  public:

    lpm_t(nrs_t *nrs_, double newton_tol_);

    lpm_t(lpm_t &set) = delete;

    ~lpm_t() {}

    particle_t operator[](int i) const
    {
      auto& data = interp_->data();
      const dfloat xp = x(i);
      const dfloat yp = y(i);
      const dfloat zp = z(i);
      return particle_t(xp, yp, zp, data.code[i], data.proc[i], data.el[i], id[i], data.r[3*i+0], data.r[3*i+1], data.r[3*i+2], v[i]);
    }

    dlong size() const { return _x.size(); }
    dlong offset() const;

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

    dlong& particleIndex(int i) { return id[i]; }
    const dlong& particleIndex(int i) const { return id[i]; }

    pointInterpolation_t& interp() { return *interp_; }

    void addPostIntegrationWork(std::function<void(lpm_t&)> work);

    occa::memory& device_x() { return o_x; }
    occa::memory& device_y() { return o_y; }
    occa::memory& device_z() { return o_z; }

  private:

    bool needsSync;
    static constexpr bool profile = true; // toggle for timing blocks

    std::shared_ptr<pointInterpolation_t> interp_;

    std::vector<dfloat> _x;
    std::vector<dfloat> _y;
    std::vector<dfloat> _z;
    std::vector<dlong> id;

    std::vector<std::function<void(lpm_t&)>> postIntegrationWork;

    // TODO: avoid vector<array<...>>
    std::vector<std::array<dfloat, 3*particle_t::integrationOrder>> v;

    dfloat* scratch_v;
    occa::memory h_scratch_v;

    occa::memory o_Uinterp; // interpolated fluid velocity, including lagged states!

    occa::memory o_x;
    occa::memory o_y;
    occa::memory o_z;

    occa::memory o_coeffAB;

    //// particle operations ////

    // Locates the element and process for each particle
    void find(bool printWarnings = true);

    // Moves each particle to the process that owns it's current element
    // this->find must have been called since the last change in position
    void migrate();

    // Interpolates the fields at each particle with the assumption that all particles belong to local
    // elements this->migrate must have been called since the last change in position
    void interpLocal(occa::memory field, occa::memory o_out, dlong nFields);

    // Perform integration
    void advance(dfloat *dt, int tstep);

    void syncToDevice();

    occa::kernel nStagesSumVectorKernel;

  };

#endif
