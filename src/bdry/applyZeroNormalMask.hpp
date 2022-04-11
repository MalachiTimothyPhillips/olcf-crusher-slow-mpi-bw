#if !defined(apply_zero_normal_mask_hpp_)
#define apply_zero_normal_mask_hpp_

#include "nrssys.hpp"
#include "occa.hpp"

class nrs_t;

void applyZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory &o_mask, occa::memory &o_x);

void applyZeroNormalMask(nrs_t *nrs,
                         dlong Nelements,
                         occa::memory &o_elementList,
                         occa::memory &o_EToB,
                         occa::memory &o_mask,
                         occa::memory &o_x);

#endif