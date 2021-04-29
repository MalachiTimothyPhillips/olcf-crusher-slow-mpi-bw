#if !defined(nekrs_avm_hpp_)
#define nekrs_avm_hpp_

#include "cds.hpp"
#include "nrs.hpp"
#include <functional>
namespace avm{
void filterSetup(nrs_t* nrs, bool userSetProps = false);
void filterSetup(nrs_t* nrs, std::function<dfloat(dfloat r, const dlong is)> viscosity, bool userSetProps = false);
void applyAVM(nrs_t* nrs, const dfloat time, const dlong scalarIndex, occa::memory o_S, occa::memory o_avm);
}

#endif
