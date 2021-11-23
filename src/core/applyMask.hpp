#ifndef apply_mask_hpp
#define apply_mask_hpp
#include <elliptic.h>
void applyMask(elliptic_t *solver, occa::memory &o_x, std::string precision, bool isGlobal);
#endif