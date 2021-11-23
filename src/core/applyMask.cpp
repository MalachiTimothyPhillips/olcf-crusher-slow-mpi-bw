#include <applyMask.hpp>
void applyMask(elliptic_t *solver, occa::memory &o_x, std::string precision, bool isGlobal)
{
  mesh_t* mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;

  const dlong Nmasked = isGlobal ? solver->NmaskedGlobal : solver->NmaskedLocal;
  occa::memory &o_maskIds = isGlobal ? solver->o_maskIdsGlobal : solver->o_maskIdsLocal;
  if (Nmasked)
    maskKernel(Nmasked, o_maskIds, o_x);
}