#if !defined(nekrs_avm_hpp_)
#define nekrs_avm_hpp_

#include "cds.hpp"
#include "nrs.hpp"
#include <functional>
namespace avm{
void filterSetup(nrs_t* nrs);
void filterSetup(nrs_t* nrs, std::function<dfloat(dfloat r, const dlong is)> viscosity);
void applyAVM(nrs_t* nrs, const dfloat time, const dlong scalarIndex, occa::memory o_S, occa::memory o_FS);
}

#endif
