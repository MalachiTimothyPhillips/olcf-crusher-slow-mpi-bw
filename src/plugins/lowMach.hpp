#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include <array>

namespace lowMach
{

struct cvodeArguments_t{
  std::array<dfloat, 3> coeffBDF;
  dfloat g0;
  dfloat dt;
};

dfloat gamma();
void setup(nrs_t* nrs, dfloat gamma0);
void buildKernel(occa::properties kernelInfo);
void qThermalIdealGasSingleComponent(dfloat time, occa::memory o_div, cvodeArguments_t * args = nullptr);
void dpdt(occa::memory o_FU);
}
