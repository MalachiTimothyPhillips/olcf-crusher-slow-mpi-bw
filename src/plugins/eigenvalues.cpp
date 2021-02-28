#include "nrs.hpp"
#include "elliptic.h"
#include "eigenvalues.hpp"
#include "nekInterfaceAdapter.hpp"
namespace eig{
namespace{
  static int Nmax;
  static int k;
  static int N;
  static int M;
  static dfloat* H;
  static dfloat* Vx;
  static dfloat* Vhost;
  static occa::memory* o_V;
  static occa::memory o_Vx;
  static occa::memory o_AVx;
  static occa::memory o_VxPfloat;
  static occa::memory o_AVxPfloat;
  static dfloat* WORK;
  static dfloat* VL;
  static dfloat* VR;
  static dfloat* WR;
  static dfloat* WI;
  static dfloat* eigVecs;
  static occa::memory o_eigVecs;
}
static void eig(const int size, double* A, double* WR, double* WI)
{
  int NB  = 256;
  char JOBVL  = 'V';
  char JOBVR  = 'V';
  int N = size;
  int LDA = size;
  int LWORK  = (NB + 2) * N;

  int INFO = -999;

  dgeev_ (&JOBVL, &JOBVR, &N, A, &LDA, WR, WI,
          VL, &LDA, VR, &LDA, WORK, &LWORK, &INFO);

  assert(INFO == 0);
}
void setup(nrs_t* nrs, int _Nmax)
{
  Nmax = _Nmax;
  kmax = Nmax;

  mesh_t* mesh = nrs->mesh;
  MPI_Barrier(mesh->comm);
  const double tStart = MPI_Wtime();
  if(mesh->rank == 0)  printf("setting up eigenvalue plugin..."); fflush(stdout);

  const dlong Nrows = mesh->Nelements * mesh->Np;
  const dlong Ncols = mesh->Nelements * mesh->Np;
     
  N = Nrows;
  M = Ncols;

  hlong Nlocal = (hlong) Nrows;
  hlong Ntotal = 0;
  MPI_Allreduce(&Nlocal, &Ntotal, 1, MPI_HLONG, MPI_SUM, mesh->comm);

  k = std::min((hlong)kmax, Ntotal); 

  // allocate memory for Hessenberg matrix
  H = (dfloat*) calloc(k * k,sizeof(dfloat));
  Vx = (dfloat*) calloc(M, sizeof(dfloat));
  eigVecs = (dfloat*) calloc(nrs->fieldOffset*k, sizeof(dfloat));
  Vhost = (dfloat*) calloc(M*k, sizeof(dfloat));

  o_V = new occa::memory[k + 1];

  o_Vx  = mesh->device.malloc(M * sizeof(dfloat),Vx);
  o_AVx = mesh->device.malloc(M * sizeof(dfloat),Vx);
  o_AVxPfloat = mesh->device.malloc(M * sizeof(pfloat));
  o_VxPfloat = mesh->device.malloc(M * sizeof(pfloat));
  for(int i = 0; i <= k; i++)
    o_V[i] = mesh->device.malloc(M * sizeof(dfloat),Vx);
  o_eigVecs = mesh->device.malloc(nrs->fieldOffset*k*sizeof(dfloat), eigVecs);
  WR = (dfloat*) calloc(k,sizeof(dfloat));
  WI = (dfloat*) calloc(k,sizeof(dfloat));
  int NB  = 256;
  int LWORK  = (NB + 2) * N;
  WORK  = new dfloat[LWORK];
  VL  = new dfloat[k * k];
  VR  = new dfloat[k * k];

}

// mode 0: no smoother
// mode 1: smoother
static void compute_eigs_impl(nrs_t* nrs, int mode)
{
  elliptic_t* elliptic = nrs->pSolver; // only consider pressure solver
  mesh_t* mesh = elliptic->mesh;
  MGLevel* fineLevel = dynamic_cast<MGLevel*>(elliptic->precon->parAlmond->levels[0]);
  // do an arnoldi
  // generate a random vector for initial basis vector
  for (dlong i = 0; i < N; i++) Vx[i] = (dfloat) drand48();

  //gather-scatter
  ogsGatherScatter(Vx, ogsDfloat, ogsAdd, mesh->ogs);
  for (dlong i = 0; i < elliptic->Nmasked; i++) Vx[elliptic->maskIds[i]] = 0.;

  o_Vx.copyFrom(Vx); //copy to device
  dfloat norm_vo = ellipticWeightedInnerProduct(elliptic, elliptic->o_invDegree, o_Vx, o_Vx);
  norm_vo = sqrt(norm_vo);

  ellipticScaledAdd(elliptic, 1. / norm_vo, o_Vx, 0., o_V[0]);

  for(int j = 0; j < k; j++) {
    // v[j+1] = invD*(A*v[j])
    //this->Ax(o_V[j],o_AVx);
    ellipticOperator(elliptic,o_V[j],o_AVx,dfloatString);
    elliptic->copyDfloatToPfloatKernel(M, o_AVxPfloat, o_AVx);
    if(mode) fineLevel->smoother(o_AVxPfloat, o_VxPfloat, true);
    else o_VxPfloat.copyFrom(o_AVxPfloat, M * sizeof(pfloat));
    elliptic->copyPfloatToDPfloatKernel(M, o_VxPfloat, o_V[j + 1]);

    // modified Gram-Schmidth
    for(int i = 0; i <= j; i++) {
      // H(i,j) = v[i]'*A*v[j]
      dfloat hij =
        ellipticWeightedInnerProduct(elliptic, elliptic->o_invDegree, o_V[i], o_V[j + 1]);

      // v[j+1] = v[j+1] - hij*v[i]
      ellipticScaledAdd(elliptic, -hij, o_V[i], 1., o_V[j + 1]);

      H[i + j * k] = (dfloat) hij;
    }

    if(j + 1 < k) {
      // v[j+1] = v[j+1]/||v[j+1]||
      dfloat norm_vj = ellipticWeightedInnerProduct(elliptic,
                                                    elliptic->o_invDegree,
                                                    o_V[j + 1],
                                                    o_V[j + 1]);
      norm_vj = sqrt(norm_vj);
      ellipticScaledAdd(elliptic, 1 / norm_vj, o_V[j + 1], 0., o_V[j + 1]);

      H[j + 1 + j * k] = (dfloat) norm_vj;
    }
  }

  eig(k, H, WR, WI);

  double rho = 0.;

  std::cout << "Eigs are:\n";
  for(int i = 0; i < k; i++) {
    double rho_i  = sqrt(WR[i] * WR[i] + WI[i] * WI[i]);
    std::cout << "\t" << rho_i << "\n";

    if(rho < rho_i)
      rho = rho_i;
  }

  // Q v is an estimate of the eigenvector
  for(int i = 0; i < k; ++i){
    o_V[i].copyTo(Vhost + i*M, M*sizeof(dfloat));
  }

  for(int neig = 0; neig < k; ++neig){
    for(int row = 0; row < M; ++row){
      dfloat rowDotCol = 0.0;
      for(int col = 0; col < k; ++col){
        rowDotCol += Vhost[row + col * M] * VL[col + neig * k];
      }
      eigVecs[row + neig*nrs->fieldOffset] = rowDotCol;
    }
  }

  occa::memory o_null;
  o_eigVecs.copyFrom(eigVecs, k*nrs->fieldOffset*sizeof(dfloat));

  const int FP64 = 1;

  writeFld("eig", 0.0, 1, FP64,
        &nrs->o_U,
        &nrs->o_P,
        &o_eigVecs,
        k);
}
void compute_eigs(nrs_t* nrs)
{
  compute_eigs_impl(nrs, 0); // eigs A only
  compute_eigs_impl(nrs, 1); // eigs SA
}
}