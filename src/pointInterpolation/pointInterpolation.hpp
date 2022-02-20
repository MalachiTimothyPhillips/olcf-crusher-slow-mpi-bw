#if !defined(nekrs_interp_hpp_)
#define nekrs_interp_hpp_

#include "nrssys.hpp"

class nrs_t;
class findpts_t;

class pointInterpolation_t {
public:
  pointInterpolation_t(nrs_t *nrs_, dfloat newton_tol_ = 0, bool profile_ = false);
  ~pointInterpolation_t();

  // Finds the process, element, and reference coordinates of the given points
  void find(bool printWarnings = true);

  // Evaluates the points using the (code, proc, el, r) tuples computed by findPoints
  void eval(const dfloat *fields,
            const dlong nFields,
            findpts_data_t *findPtsData,
            dfloat **out,
            const dlong out_stride[],
            dlong n);
  // Evaluates the points using the (code, proc, el, r) tuples computed by findPoints
  void eval(occa::memory fields,
            dlong nFields,
            findpts_data_t *findPtsData,
            dfloat **out,
            const dlong out_stride[],
            dlong n);

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

  //std::vector<dfloat> _x;
  //std::vector<dfloat> _y;
  //std::vector<dfloat> _z;
  dfloat * _x;
  dfloat * _y;
  dfloat * _z;

  std::vector<dlong> code;
  std::vector<dlong> proc;
  std::vector<dlong> el;
  std::vector<dfloat> dist2;
  std::vector<dfloat> r;

public:
  // Evalutes points located on this process
  // Given a (code, proc, el, r) tuple computed by findPoints, proc must be this
  // process's rank, code must be 0 or 1, and el and r are passed to this function
  void evalLocalPoints(const dfloat *fields,
                       const dlong nFields,
                       const dlong *el,
                       const dlong el_stride,
                       const dfloat *r,
                       const dlong r_stride,
                       dfloat **out,
                       const dlong out_stride[],
                       dlong n);

  // Evalutes points located on this process
  // Given a (code, proc, el, r) tuple computed by findPoints, proc must be this
  // process's rank, code must be 0 or 1, and el and r are passed to this function
  void evalLocalPoints(occa::memory fields,
                       const dlong nFields,
                       const dlong *el,
                       const dlong el_stride,
                       const dfloat *r,
                       const dlong r_stride,
                       dfloat **out,
                       const dlong out_stride[],
                       dlong n);
};

#endif
