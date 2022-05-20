#if !defined(nekrs_udf_hpp_)
#define nekrs_udf_hpp_

#define ins_t nrs_t

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "parReader.hpp"
#include "constantFlowRate.hpp"
#include "postProcessing.hpp"

extern "C" {
void UDF_Setup0(MPI_Comm comm, setupAide &options);
void UDF_Setup(nrs_t* nrs);
void UDF_LoadKernels(occa::properties& kernelInfo);
void UDF_ExecuteStep(nrs_t* nrs, dfloat time, int tstep);
}

using udfsetup0 = void (*)(MPI_Comm, setupAide &);
using udfsetup = void (*)(nrs_t *);
using udfloadKernels = void (*)(occa::properties &);
using udfexecuteStep = void (*)(nrs_t *, double, int);

using udfuEqnSource = void (*)(nrs_t *, double, occa::memory, occa::memory);
using udfsEqnSource = void (*)(nrs_t *, double, occa::memory, occa::memory);
using udfproperties = void (*)(nrs_t *, double, occa::memory, occa::memory, occa::memory, occa::memory);
using udfdiv = void (*)(nrs_t *, double, occa::memory);
using udfconv = int (*)(nrs_t *, int);

struct UDF
{
  udfsetup0 setup0;
  udfsetup setup;
  udfloadKernels loadKernels;
  udfexecuteStep executeStep;
  udfuEqnSource uEqnSource;
  udfsEqnSource sEqnSource;
  udfproperties properties;
  udfdiv div;
  udfconv timeStepConverged;
};

extern UDF udf;

void oudfFindDirichlet(std::string &field);
void oudfFindNeumann(std::string &field);
void oudfInit(setupAide &options);
void udfBuild(const char* udfFile, setupAide& options);
void udfLoad(void);
void* udfLoadFunction(const char* fname, int errchk);
occa::kernel oudfBuildKernel(occa::properties kernelInfo, const char *function);

#endif
