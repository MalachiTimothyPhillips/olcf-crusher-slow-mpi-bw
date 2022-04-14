#if !defined(create_zero_normal_mask_hpp_)
#define create_zero_normal_mask_hpp_

#include "occa.hpp"

class nrs_t;

void createZeroNormalMask(nrs_t *nrs, occa::memory &o_EToB, occa::memory& o_EToBV, occa::memory &o_mask);

#endif