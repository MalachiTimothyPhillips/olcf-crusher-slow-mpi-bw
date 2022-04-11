#ifndef ellipticApplyMask_hpp
#define ellipticApplyMask_hpp

#include <string>
#include "occa.hpp"

class elliptic_t;

void ellipticApplyMask(elliptic_t *solver,
                       dlong Nelements,
                       occa::memory &o_elementList,
                       occa::memory &o_x,
                       std::string precision)
#endif