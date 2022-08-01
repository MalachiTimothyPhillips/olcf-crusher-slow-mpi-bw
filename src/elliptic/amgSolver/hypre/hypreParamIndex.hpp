#ifndef HYPRE_PARAM_INDEX_HPP_
#define HYPRE_PARAM_INDEX_HPP_

// Cheby variants:
// 2 : Standard, 1st Kind Chebyshev
// 3 : Fourth Kind Chebyshev
// 4 : Optimized, Fourth Kind Chebyshev
// variants 3, 4 are from https://arxiv.org/abs/2202.08830
namespace hypreParamIndex{
  inline constexpr int BOOMERAMG_NPARAM = 14;
  inline constexpr int CUSTOM                    = 0;  /* custom settings              */
  inline constexpr int COARSENING                = 1;  /* coarsening                   */
  inline constexpr int INTERPOLATION             = 2;  /* interpolation                */
  inline constexpr int NUM_CYCLES                = 3;  /* number of cycles             */
  inline constexpr int CRS_SMOOTHER              = 4;  /* smoother for crs level       */
  inline constexpr int NUM_CRS_SWEEPS            = 5;  /* number of coarse sweeps      */
  inline constexpr int SMOOTHER                  = 6;  /* smoother                     */
  inline constexpr int NUM_SWEEPS                = 7;  /* number of sweeps             */
  inline constexpr int STRONG_THRESHOLD          = 8;  /* strong threshold             */
  inline constexpr int NON_GALERKIN_TOL          = 9;  /* non galerkin tol             */
  inline constexpr int NUM_AGG_COARSENING_LEVELS = 10; /* aggressive coarsening levels */
  inline constexpr int CHEBY_DEGREE              = 11; /* Chebyshev order */
  inline constexpr int CHEBY_VARIANT             = 12;  /* Chebyshev variant*/
  inline constexpr int POST_SMOOTHER             = 13;  /* Post Smoother */
};

#endif