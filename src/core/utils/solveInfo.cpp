#include "nrs.hpp"

solveInfo_t::solveInfo_t(const nrs_t* nrs)
{
  if(nrs->uvwSolver){
    NiterUVW = nrs->uvwSolver->Niter;
    res00NormUVW = nrs->uvwSolver->res00Norm;
    res0NormUVW = nrs->uvwSolver->res0Norm;
    resNormUVW = nrs->uvwSolver->resNorm;
  }
  else {
    NiterU = nrs->uSolver->Niter;
    res00NormU = nrs->uSolver->res00Norm;
    res0NormU = nrs->uSolver->res0Norm;
    resNormU = nrs->uSolver->resNorm;

    NiterV = nrs->vSolver->Niter;
    res00NormV = nrs->vSolver->res00Norm;
    res0NormV = nrs->vSolver->res0Norm;
    resNormV = nrs->vSolver->resNorm;

    NiterW = nrs->wSolver->Niter;
    res00NormW = nrs->wSolver->res00Norm;
    res0NormW = nrs->wSolver->res0Norm;
    resNormW = nrs->wSolver->resNorm;

  }

  NiterP = nrs->pSolver->Niter;
  res00NormP = nrs->pSolver->res00Norm;
  res0NormP = nrs->pSolver->res0Norm;
  resNormP = nrs->pSolver->resNorm;
}

void
solveInfo_t::addSolveInfo(nrs_t* nrs) const
{
  if(nrs->uvwSolver){
    nrs->uvwSolver->Niter += NiterUVW;
    nrs->uvwSolver->res00Norm = res00NormUVW;
    nrs->uvwSolver->res0Norm = res0NormUVW;
    nrs->uvwSolver->resNorm = resNormUVW;
  }
  else {
    nrs->uSolver->Niter += NiterU;
    nrs->uSolver->res00Norm = res00NormU;
    nrs->uSolver->res0Norm = res0NormU;
    nrs->uSolver->resNorm = resNormU;

    nrs->vSolver->Niter += NiterV;
    nrs->vSolver->res00Norm = res00NormV;
    nrs->vSolver->res0Norm = res0NormV;
    nrs->vSolver->resNorm = resNormV;

    nrs->wSolver->Niter += NiterW;
    nrs->wSolver->res00Norm = res00NormW;
    nrs->wSolver->res0Norm = res0NormW;
    nrs->wSolver->resNorm = resNormW;
  }
  nrs->pSolver->Niter += NiterP;
  nrs->pSolver->res00Norm = res00NormP;
  nrs->pSolver->res0Norm = res0NormP;
  nrs->pSolver->resNorm = resNormP;
}