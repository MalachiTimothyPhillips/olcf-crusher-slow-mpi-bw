#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"

#define TOKEN_PASTE(a,b) a##b
#define TOKEN_PASTE_(a,b) TOKEN_PASTE(a,b) 
#define DECLARE(a) HYPRE_Int __attribute__((visibility("default"))) TOKEN_PASTE_(HYPRE_API_PREFIX,a)

//
// definitions have match HYPRE_parcsr_ls.h
//

DECLARE(HYPRE_BoomerAMGSolve)
(
  HYPRE_Solver       solver,
  HYPRE_ParCSRMatrix A,
  HYPRE_ParVector    b,
  HYPRE_ParVector    x
)
{ 
  HYPRE_BoomerAMGSolve(solver, A, b, x);
}


DECLARE(HYPRE_BoomerAMGSetup)
(
  HYPRE_Solver       solver,
  HYPRE_ParCSRMatrix A,
  HYPRE_ParVector    b,
  HYPRE_ParVector    x
)
{ 
  HYPRE_BoomerAMGSetup(solver, A, b, x);
}


DECLARE(HYPRE_BoomerAMGCreate)
(
  HYPRE_Solver *solver
)
{ 
  HYPRE_BoomerAMGCreate(solver);
}


DECLARE(HYPRE_BoomerAMGSetCoarsenType)
(
  HYPRE_Solver solver,
  HYPRE_Int    coarsen_type
)
{ 
  HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
}


DECLARE(HYPRE_BoomerAMGSetCycleRelaxType)
(
  HYPRE_Solver  solver,
  HYPRE_Int     relax_type,
  HYPRE_Int     k
)
{ 
  HYPRE_BoomerAMGSetCycleRelaxType(solver, relax_type, k);
}


DECLARE(HYPRE_BoomerAMGSetCycleNumSweeps)
(
   HYPRE_Solver  solver,
   HYPRE_Int     num_sweeps,
   HYPRE_Int     k
)
{
  HYPRE_BoomerAMGSetCycleNumSweeps(solver, num_sweeps, k);
}


DECLARE(HYPRE_BoomerAMGSetMinCoarseSize)
(
  HYPRE_Solver solver,
  HYPRE_Int    min_coarse_size
)
{
  HYPRE_BoomerAMGSetMinCoarseSize(solver, min_coarse_size);
}


DECLARE(HYPRE_BoomerAMGSetStrongThreshold)
(
  HYPRE_Solver solver,
  HYPRE_Real   strong_threshold
)
{ 
  HYPRE_BoomerAMGSetStrongThreshold(solver, strong_threshold);
}


DECLARE(HYPRE_BoomerAMGSetNonGalerkinTol)
(
   HYPRE_Solver solver,
   HYPRE_Real    nongalerkin_tol
)
{
  HYPRE_BoomerAMGSetNonGalerkinTol(solver, nongalerkin_tol);
}


DECLARE(HYPRE_BoomerAMGSetLevelNonGalerkinTol)
(
  HYPRE_Solver solver,
  HYPRE_Real   nongalerkin_tol,
  HYPRE_Int    level
)
{
  HYPRE_BoomerAMGSetLevelNonGalerkinTol(solver, nongalerkin_tol, level);
}


DECLARE(HYPRE_BoomerAMGSetAggNumLevels)
(
  HYPRE_Solver solver,
  HYPRE_Int    agg_num_levels
)
{
  HYPRE_BoomerAMGSetAggNumLevels(solver, agg_num_levels);
}


DECLARE(HYPRE_BoomerAMGSetMaxIter)
(
  HYPRE_Solver solver,
  HYPRE_Int    max_iter
)
{
  HYPRE_BoomerAMGSetMaxIter(solver, max_iter);
}


DECLARE(HYPRE_BoomerAMGSetTol)
(  
  HYPRE_Solver solver,
  HYPRE_Real   tol
)
{
  HYPRE_BoomerAMGSetTol(solver, tol);
}


DECLARE(HYPRE_BoomerAMGSetPrintLevel)
(
  HYPRE_Solver solver,
  HYPRE_Int    print_level
)
{
  HYPRE_BoomerAMGSetPrintLevel(solver, print_level);
}


DECLARE(HYPRE_BoomerAMGSetInterpType)
(
  HYPRE_Solver solver,
  HYPRE_Int    interp_type
)
{
  HYPRE_BoomerAMGSetInterpType(solver, interp_type);
} 


DECLARE(HYPRE_BoomerAMGDestroy)
(
  HYPRE_Solver solver
)
{
  HYPRE_BoomerAMGDestroy(solver);
}

//
// definitions have match HYPRE_IJ_mv.h
// 

DECLARE(HYPRE_IJVectorCreate)
(  
  MPI_Comm        comm,
  HYPRE_BigInt    jlower,
  HYPRE_BigInt    jupper,
  HYPRE_IJVector *vector
)
{ 
  HYPRE_IJVectorCreate(comm, jlower, jupper, vector);
}


DECLARE(HYPRE_IJVectorSetObjectType)
(
   HYPRE_IJVector vector,
   HYPRE_Int      type
)
{
  HYPRE_IJVectorSetObjectType(vector, type);
}


DECLARE(HYPRE_IJVectorInitialize)
(
  HYPRE_IJVector vector
)
{
  HYPRE_IJVectorInitialize(vector);
}


DECLARE(HYPRE_IJVectorAssemble)
(
  HYPRE_IJVector vector
)
{
  HYPRE_IJVectorAssemble(vector);
}


DECLARE(HYPRE_IJVectorSetValues)
(
  HYPRE_IJVector       vector,
  HYPRE_Int            nvalues,
  const HYPRE_BigInt  *indices,
  const HYPRE_Complex *values
)
{
  HYPRE_IJVectorSetValues(vector, nvalues, indices, values);
}


DECLARE(HYPRE_IJVectorGetValues)
(
  HYPRE_IJVector   vector,
  HYPRE_Int        nvalues,
  const HYPRE_BigInt *indices,
  HYPRE_Complex   *values
)
{
  HYPRE_IJVectorGetValues(vector, nvalues, indices, values);
}


DECLARE(HYPRE_IJVectorGetObject)
(
  HYPRE_IJVector  vector,
  void          **object
)
{
  HYPRE_IJVectorGetObject(vector, object);
}


DECLARE(HYPRE_IJVectorDestroy)
(
  HYPRE_IJVector vector
)
{
  HYPRE_IJVectorDestroy(vector);
}


DECLARE(HYPRE_IJMatrixGetObject)
(
  HYPRE_IJMatrix  matrix,
  void          **object
)
{
  HYPRE_IJMatrixGetObject(matrix, object);
}


DECLARE(HYPRE_IJMatrixDestroy)
(
  HYPRE_IJMatrix matrix
)
{
  HYPRE_IJMatrixDestroy(matrix);
}


DECLARE(HYPRE_IJMatrixSetValues)
(
  HYPRE_IJMatrix       matrix,
  HYPRE_Int            nrows,
  HYPRE_Int           *ncols,
  const HYPRE_BigInt  *rows,
  const HYPRE_BigInt  *cols,
  const HYPRE_Complex *values
)
{
  HYPRE_IJMatrixSetValues(matrix, nrows, ncols, rows, cols, values);
}


DECLARE(HYPRE_IJMatrixAssemble)
(
  HYPRE_IJMatrix matrix
)
{
  HYPRE_IJMatrixAssemble(matrix);
}


DECLARE(HYPRE_IJMatrixCreate)
(
  MPI_Comm        comm,
  HYPRE_BigInt    ilower,
  HYPRE_BigInt    iupper,
  HYPRE_BigInt    jlower,
  HYPRE_BigInt    jupper,
  HYPRE_IJMatrix *matrix
)
{
  HYPRE_IJMatrixCreate(comm, ilower, iupper, jlower, jupper, matrix);
}


DECLARE(HYPRE_IJMatrixSetObjectType)
(  
  HYPRE_IJMatrix matrix,
  HYPRE_Int      type
)
{
  HYPRE_IJMatrixSetObjectType(matrix, type);
}


DECLARE(HYPRE_IJMatrixInitialize)
(
  HYPRE_IJMatrix matrix
)
{
  HYPRE_IJMatrixInitialize(matrix);
}


DECLARE(HYPRE_IJMatrixGetRowCounts)
(
  HYPRE_IJMatrix  matrix,
  HYPRE_Int       nrows,
  HYPRE_BigInt   *rows,
  HYPRE_Int      *ncols
)
{
  HYPRE_IJMatrixGetRowCounts(matrix, nrows, rows, ncols);
}


DECLARE(HYPRE_IJMatrixGetValues)
(
  HYPRE_IJMatrix  matrix,
  HYPRE_Int       nrows,
  HYPRE_Int      *ncols,
  HYPRE_BigInt   *rows,
  HYPRE_BigInt   *cols,
  HYPRE_Complex  *values
)
{
  HYPRE_IJMatrixGetValues(matrix, nrows, ncols, rows, cols, values);
}


DECLARE(HYPRE_IJMatrixAddToValues)
(
  HYPRE_IJMatrix       matrix,
  HYPRE_Int            nrows,
  HYPRE_Int           *ncols,
  const HYPRE_BigInt  *rows,
  const HYPRE_BigInt  *cols,
  const HYPRE_Complex *values
)
{
  HYPRE_IJMatrixAddToValues(matrix, nrows, ncols, rows, cols, values);
}

DECLARE(HYPRE_IJMatrixPrint)
( 
  HYPRE_IJMatrix matrix, 
  const char *filename
)
{
  HYPRE_IJMatrixPrint(matrix, filename);
}

 DECLARE(HYPRE_IJVectorPrint)
( 
  HYPRE_IJVector vector, 
  const char *filename
)
{
  HYPRE_IJVectorPrint(vector, filename);
}

DECLARE(HYPRE_SetSpGemmUseVendor)
( 
  HYPRE_Int use_vendor 
)
{
  HYPRE_SetSpGemmUseVendor(use_vendor);
}

DECLARE(HYPRE_SetSpMVUseVendor)
( 
  HYPRE_Int use_vendor 
)
{
  HYPRE_SetSpMVUseVendor(use_vendor);
}

DECLARE(HYPRE_SetUseGpuRand)
( 
  HYPRE_Int use_gpurand 
)
{
  HYPRE_SetUseGpuRand(use_gpurand);
}


DECLARE(HYPRE_BoomerAMGSetModuleRAP2)
(
  HYPRE_Solver solver,
  HYPRE_Int    mod_rap2
)
{
  HYPRE_BoomerAMGSetModuleRAP2(solver, mod_rap2);
}

DECLARE(HYPRE_Init)
(
)
{
  HYPRE_Init();
}

DECLARE(HYPRE_BoomerAMGSetKeepTranspose)
(
  HYPRE_Solver solver,
  HYPRE_Int    keepTranspose
)
{
  HYPRE_BoomerAMGSetKeepTranspose(solver, keepTranspose);
}

DECLARE(HYPRE_SetMemoryLocation)
(
  HYPRE_MemoryLocation memory_location 
)
{
  HYPRE_SetMemoryLocation(memory_location);
}

DECLARE(HYPRE_SetExecutionPolicy)
(
  HYPRE_ExecutionPolicy exec_policy
)
{
  HYPRE_SetExecutionPolicy(exec_policy);
}

