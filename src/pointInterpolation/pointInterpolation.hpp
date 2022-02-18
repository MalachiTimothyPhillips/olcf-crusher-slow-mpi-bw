#if !defined(nekrs_interp_hpp_)
#define nekrs_interp_hpp_

#include "nrssys.hpp"

class nrs_t;
class ogs_findpts_t;

class pointInterpolation_t {
public:
  pointInterpolation_t(nrs_t *nrs_, dfloat newton_tol_ = 0, bool profile_ = false);
  ~pointInterpolation_t();

  // Finds the process, element, and reference coordinates of the given points
  void find(const dfloat *const *x,
            const dlong xStride[],
            ogs_findpts_data_t *findPtsData,
            dlong n,
            bool printWarnings = true);

  // Evaluates the points using the (code, proc, el, r) tuples computed by findPoints
  void eval(const dfloat *fields,
            const dlong nFields,
            ogs_findpts_data_t *findPtsData,
            dfloat **out,
            const dlong out_stride[],
            dlong n);
  // Evaluates the points using the (code, proc, el, r) tuples computed by findPoints
  void eval(occa::memory fields,
            dlong nFields,
            ogs_findpts_data_t *findPtsData,
            dfloat **out,
            const dlong out_stride[],
            dlong n);

  // Evaluates a field at the given points
  void interpField(dfloat *fields,
                   dlong nFields,
                   const dfloat *x[],
                   const dlong x_stride[],
                   dfloat *out[],
                   const dlong out_stride[],
                   dlong n);

  void interpField(occa::memory fields,
                   dlong nFields,
                   const dfloat *x[],
                   const dlong x_stride[],
                   dfloat *out[],
                   const dlong out_stride[],
                   dlong n);

  auto *ptr() { return findpts; }

private:
  nrs_t *nrs;
  double newton_tol;
  ogs_findpts_t *findpts;
  bool profile;

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
