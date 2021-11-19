#include <applyMask.hpp>
void applyMask(elliptic_t* solver, occa::memory& o_x, std::string precision)
{
  mesh_t* mesh = solver->mesh;
  occa::kernel &maskKernel = (precision != dfloatString) ? mesh->maskPfloatKernel : mesh->maskKernel;
  if (solver->Nmasked) maskKernel(solver->Nmasked, solver->o_maskIds, o_x);
}