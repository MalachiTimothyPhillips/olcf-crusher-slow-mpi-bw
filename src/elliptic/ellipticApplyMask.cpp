#include <elliptic.h>
#include <ellipticApplyMask.hpp>
void ellipticApplyMask(elliptic_t *solver, occa::memory &o_x, std::string precision)
{
  ellipticApplyMask(solver, mesh->Nelements, mesh->o_elementList, o_x, precision);
}
void ellipticApplyMask(elliptic_t *solver,
                       dlong Nelements,
                       occa::memory &o_elementList,
                       occa::memory &o_x,
                       std::string precision)
{
  mesh_t *mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;
  occa::kernel &enforceUnKernel =
      (precision != dfloatString) ? solver->enforceUnPfloatKernel : solver->enforceUnKernel;

  const dlong Nmasked = solver->NmaskedGlobal;
  occa::memory &o_maskIds = solver->o_maskIdsGlobal;

  if (solver->applyZeroNormalMask) {
    if (precision != dfloatString) {
      std::cout << "Precision level (" << precision << ") not supported in applyZeroNormalMask\n";
      ABORT(EXIT_FAILURE);
    }
    solver->applyZeroNormalMask(Nelements, o_elementList, o_x);
  }
  if (Nmasked) {
    maskKernel(Nmasked, o_maskIds, o_x);
  }
}