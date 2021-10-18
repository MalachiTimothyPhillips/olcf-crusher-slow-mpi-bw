#include <elliptic.h>

void
ellipticUpdateLambda(elliptic_t* elliptic)
{
  mesh_t* mesh = elliptic->mesh;
  precon_t* precon = elliptic->precon;
  parAlmond::multigridLevel** levels = precon->parAlmond->levels;
  const int numMGLevels = elliptic->nLevels;
  for(int levelIndex = 0; levelIndex < numMGLevels; levelIndex++){
    auto mgLevel = dynamic_cast<MGLevel*>(levels[levelIndex]);
    
    if(levelIndex == 0){
      elliptic_t* ellipticFine = mgLevel->elliptic;
      ellipticFine->copyDfloatToPfloatKernel(mesh->Nelements * mesh->Np,
        elliptic->o_lambda,
        ellipticFine->o_lambdaPfloat);
    }
    else {
      auto prevLevel = dynamic_cast<MGLevel*>(levels[levelIndex-1]);
      elliptic_t* ellipticFine = prevLevel->elliptic;
      elliptic_t* ellipticCoarse = mgLevel->elliptic;
      const int Nfq = ellipticFine->mesh->Nq;
      const int Ncq = ellipticCoarse->mesh->Nq;
      occa::memory o_lambdaCoarse = elliptic->o_Ap;
      occa::memory o_lambdaFine = elliptic->o_p;
      platform->linAlg->fill(ellipticCoarse->mesh->Nelements * ellipticCoarse->mesh->Np, 0.0, o_lambdaCoarse);
      ellipticCoarse->copyPfloatToDPfloatKernel(ellipticFine->mesh->Nelements * ellipticFine->mesh->Np,
        ellipticFine->o_lambdaPfloat,
        o_lambdaFine);

      ellipticCoarse->precon->coarsenKernel(ellipticCoarse->mesh->Nelements, mgLevel->o_interp, o_lambdaFine, o_lambdaCoarse);

      ellipticCoarse->copyDfloatToPfloatKernel(ellipticCoarse->mesh->Nelements * ellipticCoarse->mesh->Np,
        o_lambdaCoarse,
        ellipticCoarse->o_lambdaPfloat);
    }
  }
}