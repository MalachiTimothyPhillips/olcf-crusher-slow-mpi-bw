#ifndef MAP_L_VECTOR_HPP_
#define MAP_L_VECTOR_HPP_

#include "occa.hpp"

class nrs_t;

namespace cvode{
class cvodeSolver_t;
void setupEToLMapping(nrs_t *nrs, cvodeSolver_t* cvodeSolver);
}

#endif