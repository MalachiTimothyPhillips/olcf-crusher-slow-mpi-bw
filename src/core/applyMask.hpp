#ifndef apply_mask_hpp
#define apply_mask_hpp
#include <elliptic.h>

class nrs_t;
void applyMask(elliptic_t *solver, occa::memory &o_x, std::string precision);
void applyMaskInterior(elliptic_t *solver, occa::memory &o_x, std::string precision);
void applyMaskExterior(elliptic_t *solver, occa::memory &o_x, std::string precision);

void applyMaskUnaligned(nrs_t *nrs, elliptic_t *solver, occa::memory &o_x, std::string precision);
void applyMaskUnalignedInterior(nrs_t *nrs, elliptic_t *solver, occa::memory &o_x, std::string precision);
void applyMaskUnalignedExterior(nrs_t *nrs, elliptic_t *solver, occa::memory &o_x, std::string precision);
#endif