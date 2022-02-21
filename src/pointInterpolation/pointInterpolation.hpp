#if !defined(nekrs_interp_hpp_)
#define nekrs_interp_hpp_

#include "nrssys.hpp"

class nrs_t;
class findpts_t;

class lpm_t;

class pointInterpolation_t {
public:
  pointInterpolation_t(nrs_t *nrs_, dfloat newton_tol_ = 0, bool profile_ = false);
  ~pointInterpolation_t();

  // Finds the process, element, and reference coordinates of the given points
  void find(bool printWarnings = true);

  // Evaluates the points using the (code, proc, el, r) tuples computed by findPoints
  void eval(occa::memory fields,
            dlong nFields,
            dfloat *out);

  auto *ptr() { return findpts_; }
  auto &data() {return data_;}

  void addPoints(int n, dfloat* x, dfloat* y, dfloat *z);

private:
  nrs_t *nrs;
  double newton_tol;
  findpts_t *findpts_;
  findpts_data_t data_;
  bool profile;

  int nPoints;

  dfloat * _x;
  dfloat * _y;
  dfloat * _z;

  occa::memory o_r;
  occa::memory o_el;

private:

  // Evalutes points located on this process
  // Given a (code, proc, el, r) tuple computed by findPoints, proc must be this
  // process's rank, code must be 0 or 1, and el and r are passed to this function
  void evalLocalPoints(occa::memory fields,
                       const dlong nFields,
                       const dlong *el,
                       const dfloat *r,
                       occa::memory o_out,
                       dlong n);
  friend class lpm_t;
};

#endif
