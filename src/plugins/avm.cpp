#include "cds.hpp"
#include "avm.hpp"
#include <string>
#include <functional>
#include "nekInterfaceAdapter.hpp"

/**
 * C_0^{\infty} artificial viscosity method (https://arxiv.org/pdf/1810.02152.pdf)
 **/

namespace avm {

static occa::kernel filterScalarNormKernel;
static occa::kernel applyAVMKernel;
static occa::kernel computeMaxViscKernel;
static occa::kernel computeLengthScaleKernel;

static occa::memory o_artVisc;
static occa::memory o_diffOld; // diffusion from initial state

static bool setProp = false;

void filterFunctionRelaxation1D(int Nmodes, int Nc, dfloat *A);

void filterVandermonde1D(int N, int Np, dfloat *r, dfloat *V);

dfloat filterJacobiP(dfloat a, dfloat alpha, dfloat beta, int N);

dfloat filterFactorial(int n);

void allocateMemory(nrs_t* nrs)
{
  cds_t* cds = nrs->cds;
  // nu may differ per field, but is a constant w.r.t. each element for a given scalar
  o_artVisc = platform->device.malloc(cds->NSfields * cds->mesh[0]->Np, sizeof(dfloat));

  if(!setProp) o_diffOld = platform->device.malloc(cds->fieldOffsetSum, sizeof(dfloat), cds->o_diff);
}

void compileKernels(nrs_t* nrs)
{
  cds_t* cds = nrs->cds;
  std::string install_dir;
  install_dir.assign(getenv("NEKRS_INSTALL_DIR"));
  const std::string oklpath = install_dir + "/okl/plugins/";
  std::string filename = oklpath + "filterScalarNorm.okl";
  occa::properties info = *nrs->kernelInfo;
  info["defines/" "p_Nq"] = cds->mesh[0]->Nq;
  info["defines/" "p_Np"] = cds->mesh[0]->Np;
  filterScalarNormKernel =
    platform->device.buildKernel(filename,
                             "filterScalarNorm",
                             info);

  filename = oklpath + "applyAVM.okl";
  applyAVMKernel =
    platform->device.buildKernel(filename,
                             "applyAVM",
                             info);

  filename = oklpath + "computeMaxVisc.okl";
  computeMaxViscKernel =
    platform->device.buildKernel(filename,
                             "computeMaxVisc",
                             info);

  filename = oklpath + "computeLengthScale.okl";
  computeLengthScaleKernel =
    platform->device.buildKernel(filename,
                             "computeLengthScale",
                             info);
}

void filterSetup(nrs_t* nrs, bool userSetProperties)
{
  // eq (55)
  auto gegenbauer = [](const dfloat x, const dfloat lambda)
  {
    const dfloat base = 1.0 - x * x;
    return std::pow(base, lambda);
  };
  const dfloat machine_eps = std::numeric_limits<dfloat>::epsilon();
  const dfloat alpha = -1.0 * std::log(machine_eps);
  // eq (56)
  auto superGaussian = [alpha] (const dfloat x, const dfloat lambda)
  {
    const dfloat exponent = -1.0 * alpha * std::pow(x, 2.0 * lambda);
    return std::exp(exponent);
  };
  // eq (57)
  auto gevrey = [](const dfloat x, const dfloat lambda)
  {
    const dfloat absX = std::fabs(x);
    if(absX < 1.0 && absX >= 0.0){
      const dfloat exponent = x * x / ( lambda * ( x * x - 1 ) );
      return std::exp(exponent);
    }
    return 0.0;
  };

  cds_t* cds = nrs->cds;
  const int rank = platform->comm.mpiRank;
  filterSetup(nrs,
    [cds,rank,gegenbauer, superGaussian,gevrey](const dfloat r, const dlong is){
      dfloat lambda = 0.0;
      cds->options[is].getArgs("AVM LAMBDA", lambda);
      if(cds->options[is].compareArgs("ARTIFICIAL VISCOSITY", "GEGENBAUER")){
        return gegenbauer(r, lambda);
      } else if (cds->options[is].compareArgs("ARTIFICIAL VISCOSITY", "GAUSSIAN")){
        return superGaussian(r, lambda);
      } else if (cds->options[is].compareArgs("ARTIFICIAL VISCOSITY", "GEVREY")){
        return gevrey(r, lambda);
      } else {
        if(rank == 0){
          std::string badAVMName = cds->options[is].getArgs("ARTIFICIAL VISCOSITY");
          printf("ERROR: AVM with name %s is not supported!\n", badAVMName.c_str());
          printf("Supported AVMs are: "
            "GEGENBAUER, "
            "GAUSSIAN, and "
            "GEVREY\n");
        }
        EXIT(1);
      }

      return 0.0;
    },
    userSetProperties
    );
}

void filterSetup(nrs_t *nrs,
  std::function<dfloat(dfloat r, const dlong is)> viscosity,
  bool userSetProperties
) {

  setProp = userSetProperties;

  cds_t* cds = nrs->cds;
  allocateMemory(nrs);
  compileKernels(nrs);

  // Construct Filter Function
  const int Nmodes = cds->mesh[0]->N + 1; // N+1, 1D GLL points
  if(!platform->options.compareArgs("FILTER STABILIZATION", "RELAXATION"))
  {
    cds->o_filterMT = platform->device.malloc(cds->NSfields * Nmodes * Nmodes, sizeof(dfloat));
  }
  for (int s = 0; s < cds->NSfields; ++s) {

    mesh_t* mesh = cds->mesh[s];

    // initialize viscosity distribution for field
    {
      dfloat* visc = (dfloat*) calloc(mesh->Np, sizeof(dfloat));
      for(int point = 0 ; point < mesh->Np; ++point){
        visc[point]  = viscosity(mesh->r[point], s);
        visc[point] *= viscosity(mesh->s[point], s);
        visc[point] *= viscosity(mesh->t[point], s);
      }
      o_artVisc.copyFrom(visc, mesh->Np * sizeof(dfloat), s * mesh->Np * sizeof(dfloat));
      free(visc);
    }

    // First construct filter function
    dfloat filterS = 10.0; // filter Weight...
    int filterNc;
    cds->options[s].getArgs("HPFRT STRENGTH", filterS);
    cds->options[s].getArgs("HPFRT MODES", filterNc);
    filterS = -1.0 * fabs(filterS);
    cds->filterS[s] = filterS;
    // Construct Filter Function

    // Vandermonde matrix
    dfloat *V = (dfloat *)calloc(Nmodes * Nmodes, sizeof(dfloat));
    // Filter matrix, diagonal
    dfloat *A = (dfloat *)calloc(Nmodes * Nmodes, sizeof(dfloat));

    // Construct Filter Function
    filterFunctionRelaxation1D(Nmodes, filterNc, A);

    // Construct Vandermonde Matrix
    filterVandermonde1D(mesh->N, Nmodes, mesh->r, V);

    // Invert the Vandermonde
    int INFO;
    int N = Nmodes;
    int LWORK = N * N;
    double *WORK = (double *)calloc(LWORK, sizeof(double));
    int *IPIV = (int *)calloc(Nmodes + 1, sizeof(int));
    double *iV = (double *)calloc(Nmodes * Nmodes, sizeof(double));

    for (int n = 0; n < (Nmodes + 1); n++)
      IPIV[n] = 1;
    for (int n = 0; n < Nmodes * Nmodes; ++n)
      iV[n] = V[n];

    dgetrf_(&N, &N, (double *)iV, &N, IPIV, &INFO);
    dgetri_(&N, (double *)iV, &N, IPIV, (double *)WORK, &LWORK, &INFO);

    if (INFO) {
      printf("DGE_TRI/TRF error: %d \n", INFO);
      ABORT(EXIT_FAILURE);
    }

    // V*A*V^-1 in row major
    char TRANSA = 'T';
    char TRANSB = 'T';
    double ALPHA = 1.0, BETA = 0.0;
    int MD = Nmodes;
    int ND = Nmodes;
    int KD = Nmodes;
    int LDA = Nmodes;
    int LDB = Nmodes;

    double *C = (double *)calloc(Nmodes * Nmodes, sizeof(double));
    int LDC = Nmodes;
    dgemm_(&TRANSA, &TRANSB, &MD, &ND, &KD, &ALPHA, A, &LDA, iV, &LDB, &BETA, C,
           &LDC);

    TRANSA = 'T';
    TRANSB = 'N';
    dgemm_(&TRANSA, &TRANSB, &MD, &ND, &KD, &ALPHA, V, &LDA, C, &LDB, &BETA, A,
           &LDC);

    const dlong Nbytes = Nmodes * Nmodes * sizeof(double);
    cds->o_filterMT.copyFrom(A, Nbytes, s * Nbytes);

    if (platform->comm.mpiRank == 0)
      printf("High pass filter relaxation: chi = %.4f using %d mode(s) on scalar %d\n",
             fabs(cds->filterS[s]), filterNc, s);

    free(A);
    free(C);
    free(V);
    free(iV);
    free(IPIV);
    free(WORK);
  }
}

// low Pass
void filterFunctionRelaxation1D(int Nmodes, int Nc, dfloat *A) {
  // Set all diagonal to 1
  for (int n = 0; n < Nmodes; n++)
    A[n * Nmodes + n] = 1.0;

  int k0 = Nmodes - Nc;
  for (int k = k0; k < Nmodes; k++) {
    dfloat amp = ((k + 1.0 - k0) * (k + 1.0 - k0)) / (Nc * Nc);
    A[k + Nmodes * k] = 1.0 - amp;
  }
}

void filterVandermonde1D(int N, int Np, dfloat *r, dfloat *V) {
  int sk = 0;
  for (int i = 0; i <= N; i++) {
    for (int n = 0; n < Np; n++)
      V[n * Np + sk] = filterJacobiP(r[n], 0, 0, i);
    sk++;
  }
}

// jacobi polynomials at [-1,1] for GLL
dfloat filterJacobiP(dfloat a, dfloat alpha, dfloat beta, int N) {
  dfloat ax = a;

  dfloat *P = (dfloat *)calloc((N + 1), sizeof(dfloat));

  // Zero order
  dfloat gamma0 = pow(2, (alpha + beta + 1)) / (alpha + beta + 1) *
                  filterFactorial(alpha) * filterFactorial(beta) /
                  filterFactorial(alpha + beta);
  dfloat p0 = 1.0 / sqrt(gamma0);

  if (N == 0) {
    free(P);
    return p0;
  }
  P[0] = p0;

  // first order
  dfloat gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0;
  dfloat p1 = ((alpha + beta + 2) * ax / 2 + (alpha - beta) / 2) / sqrt(gamma1);
  if (N == 1) {
    free(P);
    return p1;
  }

  P[1] = p1;

  /// Repeat value in recurrence.
  dfloat aold = 2 / (2 + alpha + beta) *
                sqrt((alpha + 1.) * (beta + 1.) / (alpha + beta + 3.));
  /// Forward recurrence using the symmetry of the recurrence.
  for (int i = 1; i <= N - 1; ++i) {
    dfloat h1 = 2. * i + alpha + beta;
    dfloat anew = 2. / (h1 + 2.) *
                  sqrt((i + 1.) * (i + 1. + alpha + beta) * (i + 1 + alpha) *
                       (i + 1 + beta) / (h1 + 1) / (h1 + 3));
    dfloat bnew = -(alpha * alpha - beta * beta) / h1 / (h1 + 2);
    P[i + 1] = 1. / anew * (-aold * P[i - 1] + (ax - bnew) * P[i]);
    aold = anew;
  }

  dfloat pN = P[N];
  free(P);
  return pN;
}

dfloat filterFactorial(int n) {
  if (n == 0)
    return 1;
  else
    return n * filterFactorial(n - 1);
}

occa::memory computeEps(nrs_t* nrs, const dfloat time, const dlong scalarIndex, occa::memory o_S)
{

  cds_t* cds = nrs->cds;
  dfloat sensorSensitivity = 1.0;
  cds->options[scalarIndex].getArgs("SENSOR SENSITIVITY", sensorSensitivity);
  mesh_t* mesh = cds->mesh[scalarIndex];
  int Nblock = (cds->mesh[0]->Nlocal+BLOCKSIZE-1)/BLOCKSIZE;

  occa::memory& o_logShockSensor = platform->o_mempool.slice0;
  occa::memory& o_elementLengths = platform->o_mempool.slice1;

  // artificial viscosity magnitude
  occa::memory o_epsilon = platform->o_mempool.slice2;

  const dfloat p = mesh->N;
  
  filterScalarNormKernel(
    mesh->Nelements,
    scalarIndex,
    cds->fieldOffsetScan[scalarIndex],
    cds->o_filterMT,
    sensorSensitivity,
    p,
    o_S,
    o_logShockSensor
  );

  // see footnote after eq 63
  const dfloat logReferenceSensor = -2.0;

  const dfloat rampParameter = 1.0;

  computeLengthScaleKernel(
    mesh->Nelements,
    mesh->o_x,
    mesh->o_y,
    mesh->o_z,
    o_elementLengths // <- min element lengths
  );

#define DEBUG
#ifdef DEBUG
  dfloat* elemLengths = platform->mempool.slice0;
  o_elementLengths.copyTo(elemLengths, mesh->Nelements * sizeof(dfloat));
  dfloat minLength = 1e8;
  for(dlong e = 0 ; e < mesh->Nelements; ++e){
    minLength = (minLength < elemLengths[e]) ? minLength : elemLengths[e];
  }
  MPI_Allreduce(MPI_IN_PLACE, &minLength, 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiComm);
  //printf("min GLL spacing: %f\n", minLength);

  //dfloat* elemLengths = platform->mempool.slice0;
  //o_elementLengths.copyTo(elemLengths, mesh->Nelements * sizeof(dfloat));
  //dfloat minLength = -1e8;
  //for(dlong e = 0 ; e < mesh->Nelements; ++e){
  //  minLength = (minLength > elemLengths[e]) ? minLength : elemLengths[e];
  //}
  //MPI_Allreduce(MPI_IN_PLACE, &minLength, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm);
  //printf("max GLL spacing: %f\n", minLength);
#endif

  computeMaxViscKernel(
    mesh->Nelements,
    nrs->fieldOffset,
    logReferenceSensor,
    rampParameter,
    o_elementLengths, // h_e
    cds->o_U,
    o_logShockSensor,
    o_epsilon // max(|df/du|) <- max visc
  );

  return o_epsilon;

}

void applyAVM(nrs_t* nrs, const dfloat time, const dlong scalarIndex, occa::memory o_S, occa::memory o_avm)
{
  cds_t* cds = nrs->cds;
  mesh_t* mesh = cds->mesh[scalarIndex];
  const dlong scalarOffset = cds->fieldOffsetScan[scalarIndex];
  occa::memory o_eps = computeEps(nrs, time, scalarIndex, o_S);
  platform->linAlg->fill(nrs->fieldOffset, 0.0, o_avm);
  applyAVMKernel(
    mesh->Nelements,
    scalarOffset,
    scalarIndex,
    o_eps,
    o_avm
  );

  const dfloat maxVisc = platform->linAlg->max(
    mesh->Nlocal,
    o_avm,
    platform->comm.mpiComm
  );
  //printf("vismx: %f\n", maxVisc);

  //platform->linAlg->axpby(
  //  mesh->Nlocal,
  //  1.0,
  //  o_avm,
  //  1.0,
  //  cds->o_diff,
  //  0,
  //  cds->fieldOffsetScan[scalarIndex]
  //);

  if(nrs->isOutputStep && 1)
  {
    writeFld(
      "avm", time, 1, 1,
      &nrs->o_U,
      &nrs->o_P,
      &o_avm,
      1);

    // compare to n5k results
    //nek::ocopyToNek(time, 1); // whatever
    //nek::userchk();
  }
}

} // namespace