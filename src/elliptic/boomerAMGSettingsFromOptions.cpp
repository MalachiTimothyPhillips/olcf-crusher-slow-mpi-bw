#include "elliptic.h"
#include "hypreParamIndex.hpp"

std::vector<double>
boomerAMGSettingsFromOptions(setupAide& options){
  using namespace hypreParamIndex;
  std::vector<double> settings(BOOMERAMG_NPARAM, 0.0);

  settings[CUSTOM                   ]  = 1;    /* custom settings              */
  settings[COARSENING               ]  = 8;    /* coarsening                   */
  settings[INTERPOLATION            ]  = 6;    /* interpolation                */
  settings[NUM_CYCLES               ]  = 1;    /* number of cycles             */
  settings[CRS_SMOOTHER             ]  = 18;   /* smoother for crs level       */
  settings[NUM_CRS_SWEEPS           ]  = 3;    /* number of coarse sweeps      */
  settings[SMOOTHER                 ]  = 18;   /* smoother                     */
  settings[NUM_SWEEPS               ]  = 1;    /* number of sweeps             */
  settings[STRONG_THRESHOLD         ]  = 0.25; /* strong threshold             */
  settings[NON_GALERKIN_TOL         ]  = 0.05; /* non galerkin tol             */
  settings[NUM_AGG_COARSENING_LEVELS]  = 0;    /* aggressive coarsening levels */
  settings[CHEBY_DEGREE             ]  = 2;    /* Chebysehv Order              */
  settings[CHEBY_VARIANT            ]  = 2;    /* Chebyshev Variant            */

  if(options.compareArgs("MULTIGRID SEMFEM", "TRUE")) {
    settings[CRS_SMOOTHER]  = 16;
    settings[SMOOTHER]  = 16;
  }  

  options.getArgs("BOOMERAMG COARSEN TYPE", settings[COARSENING]);
  options.getArgs("BOOMERAMG INTERPOLATION TYPE", settings[INTERPOLATION]);
  options.getArgs("BOOMERAMG COARSE SMOOTHER TYPE", settings[CRS_SMOOTHER]);

  options.getArgs("BOOMERAMG SMOOTHER TYPE", settings[SMOOTHER]);

  // by default, use the same smoother during the post cycle
  options.getArgs("BOOMERAMG SMOOTHER TYPE", settings[POST_SMOOTHER]);

  options.getArgs("BOOMERAMG SMOOTHER SWEEPS", settings[NUM_SWEEPS]);
  options.getArgs("BOOMERAMG ITERATIONS", settings[NUM_CYCLES]);
  options.getArgs("BOOMERAMG STRONG THRESHOLD", settings[STRONG_THRESHOLD]);
  options.getArgs("BOOMERAMG NONGALERKIN TOLERANCE" , settings[NON_GALERKIN_TOL]);
  options.getArgs("BOOMERAMG AGGRESSIVE COARSENING LEVELS" , settings[NUM_AGG_COARSENING_LEVELS]);
  options.getArgs("BOOMERAMG CHEBYSHEV DEGREE" , settings[CHEBY_DEGREE]);
  options.getArgs("BOOMERAMG CHEBYSHEV VARIANT" , settings[CHEBY_VARIANT]);

  options.getArgs("BOOMERAMG POST SMOOTHER TYPE", settings[POST_SMOOTHER]);

  return settings;
}