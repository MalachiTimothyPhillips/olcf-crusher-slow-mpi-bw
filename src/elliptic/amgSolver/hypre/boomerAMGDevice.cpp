#include "nrssys.hpp"
#include "platform.hpp"
#include <map>
#include <vector>

#ifdef ENABLE_HYPRE_GPU_SUPPORT
#define NEKRS_HYPRE_DEVICE
#include "__HYPRE.h"
#undef NEKRS_HYPRE_DEVICE 

static double boomerAMGParam[BOOMERAMG_NPARAM];

class hypre_data_t;

struct hypre_data_t {
  MPI_Comm comm;
  HYPRE_Solver solver;
  HYPRE_IJMatrix A;
  HYPRE_IJVector b;
  HYPRE_IJVector x;
  HYPRE_BigInt iupper;
  HYPRE_BigInt ilower;
  occa::memory o_ii;
  int nRows;
};
static hypre_data_t *data;

int boomerAMGSetupDevice(int nrows, int nz,
                         const long long int * Ai, const long long int * Aj, const double * Av,
                         const int null_space, const MPI_Comm ce,
                         const int useFP32, const double *param, const int verbose)
{
  MPI_Comm comm;
  MPI_Comm_dup(ce, &comm);

  int rank;
  MPI_Comm_rank(comm,&rank);

  __HYPRE_Load();

  if(sizeof(HYPRE_Real) != ((useFP32) ? sizeof(float) : sizeof(double))) {
    if(rank == 0) printf("HYPRE has not been built to support FP32.\n");
    MPI_Abort(ce, 1);
  } 

  data = new hypre_data_t();
  data->comm = comm;
  data->nRows = nrows;
  data->ilower = data->nRows;
  MPI_Scan(MPI_IN_PLACE, &data->ilower, 1, MPI_LONG_LONG, MPI_SUM, ce);
  data->ilower -= data->nRows;
  data->iupper = (data->ilower + data->nRows) - 1; 

  __HYPRE_Init();

  __HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
  __HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);

#if 0
  hypre_uint mempool_bin_growth   = 8,
             mempool_min_bin      = 3,
             mempool_max_bin      = 9;
  size_t mempool_max_cached_bytes = 2000LL * 1024 * 1024;
  __HYPRE_SetGPUMemoryPoolSize(mempool_bin_growth, mempool_min_bin,
                               mempool_max_bin, mempool_max_cached_bytes);
#endif

  HYPRE_Int use_vendor = 0; 
#if defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)
  use_vendor = 1;

  HYPRE_Int  spgemm_alg = 1;
  HYPRE_Int  spgemm_rowest_mtd = 3;
  HYPRE_Int  spgemm_rowest_nsamples = 32;
  HYPRE_Real spgemm_rowest_mult = 1.5;
  char       spgemm_hash_type = 'L';

  __HYPRE_SetSpGemmAlgorithm(spgemm_alg);
  __HYPRE_SetSpGemmRownnzEstimateMethod(spgemm_rowest_mtd);
  __HYPRE_SetSpGemmRownnzEstimateNSamples(spgemm_rowest_nsamples);
  __HYPRE_SetSpGemmRownnzEstimateMultFactor(spgemm_rowest_mult);
  __HYPRE_SetSpGemmHashType(spgemm_hash_type);
#endif
  __HYPRE_SetSpGemmUseVendor(use_vendor);
  __HYPRE_SetSpMVUseVendor(1);

  __HYPRE_SetUseGpuRand(1);

  // Setup matrix
  {
    __HYPRE_IJMatrixCreate(comm,data->ilower,data->iupper,data->ilower,data->iupper,&data->A);
    __HYPRE_IJMatrixSetObjectType(data->A,HYPRE_PARCSR);
    __HYPRE_IJMatrixInitialize(data->A);

    std::map<HYPRE_BigInt, std::vector<std::pair<HYPRE_BigInt, HYPRE_Real>>> rowToColAndVal;
    for(int i=0; i<nz; i++) 
    {
      HYPRE_BigInt mati = (HYPRE_BigInt)(Ai[i]);
      HYPRE_BigInt matj = (HYPRE_BigInt)(Aj[i]);
      HYPRE_Real matv = (HYPRE_Real) Av[i]; 
      rowToColAndVal[mati].emplace_back(std::make_pair(matj, matv));
    }

    const HYPRE_Int rowsToSet = rowToColAndVal.size();
    std::vector<HYPRE_Int> ncols(rowsToSet);
    std::vector<HYPRE_BigInt> rows(rowsToSet);
    std::vector<HYPRE_BigInt> cols(nz);
    std::vector<HYPRE_Real> vals(nz);

    unsigned rowCtr = 0;
    unsigned colCtr = 0;
    for(auto&& rowAndColValPair : rowToColAndVal){
      const auto & row = rowAndColValPair.first;
      const auto & colAndValues = rowAndColValPair.second;

      rows[rowCtr] = row;
      ncols[rowCtr] = colAndValues.size();

      for(auto&& colAndValue : colAndValues){
        const auto & col = colAndValue.first;
        const auto & val = colAndValue.second;
        cols[colCtr] = col;
        vals[colCtr] = val;
        ++colCtr;
      }
      ++rowCtr;
    }

    auto o_ncols = platform->device.malloc(ncols.size() * sizeof(HYPRE_Int),ncols.data());
    auto o_rows = platform->device.malloc(rows.size() * sizeof(HYPRE_BigInt),rows.data());
    auto o_cols = platform->device.malloc(cols.size() * sizeof(HYPRE_BigInt),cols.data());
    auto o_vals = platform->device.malloc(vals.size() * sizeof(HYPRE_Real),vals.data());

    __HYPRE_IJMatrixSetValues(data->A, 
                              rowsToSet /* values for nrows */, 
                              (HYPRE_Int*) o_ncols.ptr() /* cols for each row */, 
                              (HYPRE_BigInt*) o_rows.ptr(),
                              (HYPRE_BigInt*) o_rows.ptr(),
                              (HYPRE_Real*) o_vals.ptr());

    auto o_ncols.free();
    auto o_rows.free();
    auto o_cols.free();
    auto o_vals.free();

    __HYPRE_IJMatrixAssemble(data->A);
#if 0
    __HYPRE_IJMatrixPrint(data->A, "matrix.dat");
#endif
  }

  if ((int) param[0]) {
    for (int i = 0; i < BOOMERAMG_NPARAM; i++)
      boomerAMGParam[i] = param[i+1]; 
  } else {
    boomerAMGParam[0]  = 8;    /* coarsening */
    boomerAMGParam[1]  = 6;    /* interpolation */
    boomerAMGParam[2]  = 1;    /* number of cycles */
    boomerAMGParam[3]  = 16;   /* smoother for crs level */
    boomerAMGParam[4]  = 3;    /* sweeps */
    boomerAMGParam[5]  = 16;   /* smoother */
    boomerAMGParam[6]  = 1;    /* sweeps   */
    boomerAMGParam[7]  = 0.25; /* threshold */
    boomerAMGParam[8]  = 0.0;  /* non galerkin tolerance */
  }

  // Setup solver
  __HYPRE_BoomerAMGCreate(&data->solver);

  __HYPRE_BoomerAMGSetCoarsenType(data->solver,boomerAMGParam[0]);
  __HYPRE_BoomerAMGSetInterpType(data->solver,boomerAMGParam[1]);

  __HYPRE_BoomerAMGSetModuleRAP2(data->solver, 1);
  __HYPRE_BoomerAMGSetKeepTranspose(data->solver, 1);

  //__HYPRE_BoomerAMGSetChebyFraction(data->solver, 0.2); 

  if (boomerAMGParam[5] > 0) {
    __HYPRE_BoomerAMGSetCycleRelaxType(data->solver, boomerAMGParam[5], 1);
    __HYPRE_BoomerAMGSetCycleRelaxType(data->solver, boomerAMGParam[5], 2);
  } 
  __HYPRE_BoomerAMGSetCycleRelaxType(data->solver, 9, 3);

  __HYPRE_BoomerAMGSetCycleNumSweeps(data->solver, boomerAMGParam[6], 1);
  __HYPRE_BoomerAMGSetCycleNumSweeps(data->solver, boomerAMGParam[6], 2);
  __HYPRE_BoomerAMGSetCycleNumSweeps(data->solver, 1, 3);

  if (null_space) {
    __HYPRE_BoomerAMGSetMinCoarseSize(data->solver, 2);
    __HYPRE_BoomerAMGSetCycleRelaxType(data->solver, boomerAMGParam[3], 3);
    __HYPRE_BoomerAMGSetCycleNumSweeps(data->solver, boomerAMGParam[4], 3);
  }

  __HYPRE_BoomerAMGSetStrongThreshold(data->solver,boomerAMGParam[7]);

// not supported yet
#if 0
  if (boomerAMGParam[8] > 1e-3) {
    __HYPRE_BoomerAMGSetNonGalerkinTol(data->solver,boomerAMGParam[8]);
    __HYPRE_BoomerAMGSetLevelNonGalerkinTol(data->solver,0.0 , 0);
    __HYPRE_BoomerAMGSetLevelNonGalerkinTol(data->solver,0.01, 1);
    __HYPRE_BoomerAMGSetLevelNonGalerkinTol(data->solver,0.05, 2);
  }
#endif

  __HYPRE_BoomerAMGSetAggNumLevels(data->solver, boomerAMGParam[9]); 

  __HYPRE_BoomerAMGSetMaxIter(data->solver, boomerAMGParam[2]); // number of V-cycles
  __HYPRE_BoomerAMGSetTol(data->solver,0);

  if(verbose)
    __HYPRE_BoomerAMGSetPrintLevel(data->solver,3);
  else
    __HYPRE_BoomerAMGSetPrintLevel(data->solver,1);

  // Create and initialize rhs and solution vectors
  __HYPRE_IJVectorCreate(comm,data->ilower,data->iupper,&data->b);
  __HYPRE_IJVectorSetObjectType(data->b,HYPRE_PARCSR);
  __HYPRE_IJVectorInitialize(data->b);
  __HYPRE_IJVectorAssemble(data->b);

  __HYPRE_IJVectorCreate(comm,data->ilower,data->iupper,&data->x);
  __HYPRE_IJVectorSetObjectType(data->x,HYPRE_PARCSR);
  __HYPRE_IJVectorInitialize(data->x);
  __HYPRE_IJVectorAssemble(data->x);

  // Perform AMG setup
  HYPRE_ParVector par_b;
  HYPRE_ParVector par_x;
  HYPRE_ParCSRMatrix par_A;
  __HYPRE_IJVectorGetObject(data->b,(void**) &par_b);
  __HYPRE_IJVectorGetObject(data->x,(void**) &par_x);
  __HYPRE_IJMatrixGetObject(data->A,(void**) &par_A);

  __HYPRE_BoomerAMGSetup(data->solver,par_A,par_b,par_x);

  HYPRE_BigInt *ii = (HYPRE_BigInt*) malloc(data->nRows*sizeof(HYPRE_BigInt));
  for(int i=0;i<data->nRows;++i) 
    ii[i] = data->ilower + i;
  data->o_ii = platform->device.malloc(data->nRows*sizeof(HYPRE_BigInt), ii);
  free(ii);

  return 0;
}

int boomerAMGSolveDevice(const occa::memory& o_x, const occa::memory& o_b)
{
  int err;

  // note x is ALWAYS zero

  __HYPRE_IJVectorSetValues(data->b,data->nRows,(HYPRE_BigInt*) data->o_ii.ptr(),(HYPRE_Real*) o_b.ptr());
  __HYPRE_IJVectorAssemble(data->b);

  HYPRE_ParVector par_x;
  HYPRE_ParVector par_b;
  HYPRE_ParCSRMatrix par_A;

  __HYPRE_IJVectorGetObject(data->x,(void **) &par_x);
  __HYPRE_IJVectorGetObject(data->b,(void **) &par_b);
  __HYPRE_IJMatrixGetObject(data->A,(void **) &par_A);

#if 0
  __HYPRE_IJVectorPrint(data->b, "b.dat");
  __HYPRE_IJVectorPrint(data->x, "x.dat");
#endif

  if(__HYPRE_BoomerAMGSolve(data->solver,par_A,par_b,par_x) > 0) { 
    int rank;
    MPI_Comm_rank(data->comm,&rank);
    if(rank == 0) printf("HYPRE_BoomerAMGSolve failed!\n");
    return 1;
  }

  __HYPRE_IJVectorGetValues(data->x,data->nRows,(HYPRE_BigInt*) data->o_ii.ptr(),(HYPRE_Real*) o_x.ptr());

  return 0; 
}

void boomerAMGFreeDevice()
{
  __HYPRE_BoomerAMGDestroy(data->solver);
  __HYPRE_IJMatrixDestroy(data->A);
  __HYPRE_IJVectorDestroy(data->x);
  __HYPRE_IJVectorDestroy(data->b);
  data->o_ii.free();
  free(data);
}

#else

int boomerAMGSetupDevice(int nrows, int nz,
                         const long long int * Ai, const long long int * Aj, const double * Av,
                         const int null_space, const MPI_Comm ce,
                         const int useFP32, const double *param, const int verbose)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);  
  if(rank == 0) printf("ERROR: Recompile with HYPRE GPU support!\n");
  return 1;
}

int boomerAMGSolveDevice(const occa::memory& o_x, const occa::memory& o_b)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);  
  if(rank == 0) printf("ERROR: Recompile with HYPRE GPU support!\n");
  return 1;
}

void boomerAMGFreeDevice()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);  
  if(rank == 0) printf("ERROR: Recompile with HYPRE GPU support!\n");
}
#endif