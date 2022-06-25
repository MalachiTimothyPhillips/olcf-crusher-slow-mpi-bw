#ifndef HYPRE_WRAPPER_H
#define HYPRE_WRAPPER_H

#ifdef NEKRS_HYPRE_DEVICE 
#define HYPRE_API_PREFIX NEKRS_DEVICE_
#else
#define HYPRE_API_PREFIX NEKRS_
#endif

#include <dlfcn.h>

#define BOOMERAMG_NPARAM 10

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NEKRS_HYPRE_DEVICE
#include "device/HYPRE.h"    
#include "device/HYPRE_parcsr_ls.h"
#include "device/_hypre_utilities.h"
#else
#include "HYPRE.h"    
#include "HYPRE_parcsr_ls.h"
#endif

#define _TOKEN_PASTE(a,b) a##b
#define _TOKEN_PASTE_(a,b) _TOKEN_PASTE(a,b) 
#define ADD_PREFIX(a) _TOKEN_PASTE_(HYPRE_API_PREFIX,a)

#define DECLARE(a,b) HYPRE_Int ADD_PREFIX(a) b; typedef HYPRE_Int (*t_##a) b; static t_##a __##a 

//
// definitions have match HYPRE_parcsr_ls.h
//
DECLARE(
HYPRE_BoomerAMGSolve,
(
  HYPRE_Solver       solver,
  HYPRE_ParCSRMatrix A,
  HYPRE_ParVector    b,
  HYPRE_ParVector    x
)
);

DECLARE(
HYPRE_BoomerAMGSetup,
(
  HYPRE_Solver       solver,
  HYPRE_ParCSRMatrix A,
  HYPRE_ParVector    b,
  HYPRE_ParVector    x
)
);

DECLARE(
HYPRE_BoomerAMGCreate,
(
  HYPRE_Solver *solver
)
);

DECLARE(
HYPRE_BoomerAMGSetCoarsenType,
(
  HYPRE_Solver solver,
  HYPRE_Int    coarsen_type
)
);

DECLARE(
HYPRE_BoomerAMGSetCycleRelaxType,
(
  HYPRE_Solver  solver,
  HYPRE_Int     relax_type,
  HYPRE_Int     k
)
);

DECLARE(
HYPRE_BoomerAMGSetCycleNumSweeps,
(
   HYPRE_Solver  solver,
   HYPRE_Int     num_sweeps,
   HYPRE_Int     k
)
);

DECLARE(
HYPRE_BoomerAMGSetMinCoarseSize,
(
  HYPRE_Solver solver,
  HYPRE_Int    min_coarse_size
)
);

DECLARE(
HYPRE_BoomerAMGSetStrongThreshold,
(
  HYPRE_Solver solver,
  HYPRE_Real   strong_threshold
)
);

DECLARE(
HYPRE_BoomerAMGSetNonGalerkinTol,
(
   HYPRE_Solver solver,
   HYPRE_Real  nongalerkin_tol
)
);

DECLARE(
HYPRE_BoomerAMGSetLevelNonGalerkinTol,
(
  HYPRE_Solver solver,
  HYPRE_Real   nongalerkin_tol,
  HYPRE_Int  level
)
);

DECLARE(
HYPRE_BoomerAMGSetAggNumLevels,
(
  HYPRE_Solver solver,
  HYPRE_Int    agg_num_levels
)
);

DECLARE(
HYPRE_BoomerAMGSetMaxIter,
(
  HYPRE_Solver solver,
  HYPRE_Int          max_iter
)
);

DECLARE(
HYPRE_BoomerAMGSetTol,
(  
  HYPRE_Solver solver,
  HYPRE_Real   tol
)
);

DECLARE(
HYPRE_BoomerAMGSetPrintLevel,
(
  HYPRE_Solver solver,
  HYPRE_Int    print_level
)
);


DECLARE(
HYPRE_IJMatrixPrint,
(
  HYPRE_IJMatrix matrix,
  const char *filename
)
);

DECLARE(
HYPRE_IJVectorPrint,
(
  HYPRE_IJVector vector,
  const char *filename
)
);


DECLARE(
HYPRE_BoomerAMGSetInterpType,
(
  HYPRE_Solver solver,
  HYPRE_Int    interp_type
)
);

DECLARE(
HYPRE_BoomerAMGDestroy,
(
  HYPRE_Solver solver
)
);

//
// definitions have match HYPRE_IJ_mv.h
// 
DECLARE(
HYPRE_IJVectorCreate,
(  
  MPI_Comm        comm,
  HYPRE_BigInt    jlower,
  HYPRE_BigInt    jupper,
  HYPRE_IJVector *vector
)
);

DECLARE(
HYPRE_IJVectorSetObjectType,
(
   HYPRE_IJVector vector,
   HYPRE_Int      type
)
);

DECLARE(
HYPRE_IJVectorInitialize,
(
  HYPRE_IJVector vector
)
);

DECLARE(
HYPRE_IJVectorAssemble,
(
  HYPRE_IJVector vector
)
);

DECLARE(
HYPRE_IJVectorSetValues,
(
  HYPRE_IJVector       vector,
  HYPRE_Int            nvalues,
  const HYPRE_BigInt  *indices,
  const HYPRE_Complex *values
)
);

DECLARE(
HYPRE_IJVectorGetValues,
(
  HYPRE_IJVector   vector,
  HYPRE_Int        nvalues,
  const HYPRE_BigInt *indices,
  HYPRE_Complex   *values
)
);

DECLARE(
HYPRE_IJVectorGetObject,
(
  HYPRE_IJVector  vector,
  void          **object
)
);

DECLARE(
HYPRE_IJVectorDestroy,
(
  HYPRE_IJVector vector
)
);

DECLARE(
HYPRE_IJMatrixGetObject,
(
  HYPRE_IJMatrix  matrix,
  void          **object
)
);

DECLARE(
HYPRE_IJMatrixDestroy,
(
  HYPRE_IJMatrix matrix
)
);

DECLARE(
HYPRE_IJMatrixSetValues,
(
  HYPRE_IJMatrix       matrix,
  HYPRE_Int            nrows,
  HYPRE_Int           *ncols,
  const HYPRE_BigInt  *rows,
  const HYPRE_BigInt  *cols,
  const HYPRE_Complex *values
)
);

DECLARE(
HYPRE_IJMatrixAssemble,
(
  HYPRE_IJMatrix matrix
)
);

DECLARE(
HYPRE_IJMatrixCreate,
(
  MPI_Comm        comm,
  HYPRE_BigInt    ilower,
  HYPRE_BigInt    iupper,
  HYPRE_BigInt    jlower,
  HYPRE_BigInt    jupper,
  HYPRE_IJMatrix *matrix
)
);

DECLARE(
HYPRE_IJMatrixSetObjectType,
(  
  HYPRE_IJMatrix matrix,
  HYPRE_Int      type
)
);

DECLARE(
HYPRE_IJMatrixInitialize,
(
  HYPRE_IJMatrix matrix
)
);

DECLARE(
HYPRE_IJMatrixGetRowCounts,
(
  HYPRE_IJMatrix  matrix,
  HYPRE_Int       nrows,
  HYPRE_BigInt   *rows,
  HYPRE_Int      *ncols
)
);

DECLARE(
HYPRE_IJMatrixGetValues,
(
  HYPRE_IJMatrix  matrix,
  HYPRE_Int       nrows,
  HYPRE_Int      *ncols,
  HYPRE_BigInt   *rows,
  HYPRE_BigInt   *cols,
  HYPRE_Complex  *values
)
);

DECLARE(
HYPRE_IJMatrixAddToValues,
(
  HYPRE_IJMatrix       matrix,
  HYPRE_Int            nrows,
  HYPRE_Int           *ncols,
  const HYPRE_BigInt  *rows,
  const HYPRE_BigInt  *cols,
  const HYPRE_Complex *values
)
);

DECLARE(
HYPRE_SetSpGemmUseVendor,
( 
  HYPRE_Int use_vendor 
)
);

DECLARE(
HYPRE_SetSpMVUseVendor,
( 
  HYPRE_Int use_vendor 
)
);

DECLARE(
HYPRE_SetUseGpuRand,
( 
  HYPRE_Int use_gpurand 
)
);

DECLARE(
HYPRE_BoomerAMGSetModuleRAP2,
(
  HYPRE_Solver solver,
  HYPRE_Int    mod_rap2
)
);

DECLARE(
HYPRE_Init,
(
)
);

DECLARE(
HYPRE_BoomerAMGSetKeepTranspose,
(
  HYPRE_Solver solver,
  HYPRE_Int    keepTranspose
)
);

DECLARE
(HYPRE_SetMemoryLocation,
(
  HYPRE_MemoryLocation memory_location 
)
);

DECLARE(
HYPRE_SetExecutionPolicy,
(
  HYPRE_ExecutionPolicy exec_policy
)
);



#undef DECLARE

static void check_error(const char* error)
{
  if(error != NULL) {
    fprintf(stderr, "Error: %s!\n", error);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
}

#define STRING_IT(a) #a
#define STRING_IT_(a) STRING_IT(a) 
#define LIB_LOAD(a) __##a = (t_##a) dlsym(lib_handle, STRING_IT_(ADD_PREFIX(a))); check_error(dlerror());;
static void __HYPRE_Load()
{
  //TODO: bcast + load from node-local storage

  const char* install_dir = getenv("NEKRS_HOME");
#define MAX_PATH 4096
  char libpath[MAX_PATH];
  char libname[MAX_PATH];
#undef MAX_PATH
  char ext[6];

  snprintf(ext, sizeof(ext), "%s", "so");
#ifdef __APPLE__
  snprintf(ext, sizeof(ext), "%s", "dylib");
#endif

#ifdef NEKRS_HYPRE_DEVICE
  snprintf(libname, sizeof(libname), "%s", "hypre/libHYPREDevice");
#else
  snprintf(libname, sizeof(libname), "%s", "hypre/libHYPRE");
#endif

  snprintf(libpath, sizeof(libpath), "%s/lib/%s.%s", install_dir, libname, ext);

  void* lib_handle = dlopen(libpath, RTLD_LAZY | RTLD_LOCAL);
  if(!lib_handle) check_error(dlerror());

  LIB_LOAD(HYPRE_BoomerAMGSolve);
  LIB_LOAD(HYPRE_BoomerAMGSetup);
  LIB_LOAD(HYPRE_BoomerAMGCreate); 
  LIB_LOAD(HYPRE_BoomerAMGSetCoarsenType);
  LIB_LOAD(HYPRE_BoomerAMGSetCycleRelaxType);
  LIB_LOAD(HYPRE_BoomerAMGSetCycleNumSweeps);
  LIB_LOAD(HYPRE_BoomerAMGSetMinCoarseSize); 
  LIB_LOAD(HYPRE_BoomerAMGSetStrongThreshold); 
  LIB_LOAD(HYPRE_BoomerAMGSetNonGalerkinTol); 
  LIB_LOAD(HYPRE_BoomerAMGSetLevelNonGalerkinTol);
  LIB_LOAD(HYPRE_BoomerAMGSetAggNumLevels); 
  LIB_LOAD(HYPRE_BoomerAMGSetMaxIter);
  LIB_LOAD(HYPRE_BoomerAMGSetTol); 
  LIB_LOAD(HYPRE_BoomerAMGSetPrintLevel);
  LIB_LOAD(HYPRE_IJMatrixPrint);
  LIB_LOAD(HYPRE_IJVectorPrint);
  LIB_LOAD(HYPRE_BoomerAMGSetInterpType);
  LIB_LOAD(HYPRE_BoomerAMGDestroy);
  LIB_LOAD(HYPRE_IJVectorCreate);
  LIB_LOAD(HYPRE_IJVectorSetObjectType);
  LIB_LOAD(HYPRE_IJVectorInitialize); 
  LIB_LOAD(HYPRE_IJVectorAssemble);
  LIB_LOAD(HYPRE_IJVectorSetValues); 
  LIB_LOAD(HYPRE_IJVectorGetValues);
  LIB_LOAD(HYPRE_IJVectorGetObject);
  LIB_LOAD(HYPRE_IJVectorDestroy);
  LIB_LOAD(HYPRE_IJMatrixGetObject);
  LIB_LOAD(HYPRE_IJMatrixDestroy);
  LIB_LOAD(HYPRE_IJMatrixSetValues);
  LIB_LOAD(HYPRE_IJMatrixAssemble);
  LIB_LOAD(HYPRE_IJMatrixCreate);
  LIB_LOAD(HYPRE_IJMatrixSetObjectType);
  LIB_LOAD(HYPRE_IJMatrixInitialize);
  LIB_LOAD(HYPRE_IJMatrixGetRowCounts);
  LIB_LOAD(HYPRE_IJMatrixGetValues);
  LIB_LOAD(HYPRE_IJMatrixAddToValues);
  LIB_LOAD(HYPRE_SetSpGemmUseVendor);
  LIB_LOAD(HYPRE_SetSpMVUseVendor);
  LIB_LOAD(HYPRE_SetUseGpuRand);
  LIB_LOAD(HYPRE_BoomerAMGSetModuleRAP2);
  LIB_LOAD(HYPRE_BoomerAMGSetKeepTranspose);
  LIB_LOAD(HYPRE_SetMemoryLocation);
  LIB_LOAD(HYPRE_SetExecutionPolicy);
  LIB_LOAD(HYPRE_Init);
}

#undef STRING_IT
#undef STRING_IT_ 
#undef LIB_LOAD

#undef _TOKEN_PASTE
#undef _TOKEN_PASTE_
#undef ADD_PREFIX
#undef DECLARE

#ifdef __cplusplus
}
#endif

#endif
