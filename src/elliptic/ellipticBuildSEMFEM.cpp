/*
 * Low-Order finite element preconditioner computed with HYPRE's AMG solver
*/

#include <math.h>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

#include "nrssys.hpp"
#include "platform.hpp"
#include "hypreWrapper.hpp"
#include "gslib.h"
#include "ellipticBuildSEMFEM.hpp"

namespace{
 void quadrature_rule(hypreWrapper::Real q_r[4][3], hypreWrapper::Real q_w[4]) {
    hypreWrapper::Real a = (5.0 + 3.0 * sqrt(5.0)) / 20.0;
    hypreWrapper::Real b = (5.0 - sqrt(5.0)) / 20.0;

    q_r[0][0] = a;
    q_r[0][1] = b;
    q_r[0][2] = b;
    q_r[1][0] = b;
    q_r[1][1] = a;
    q_r[1][2] = b;
    q_r[2][0] = b;
    q_r[2][1] = b;
    q_r[2][2] = a;
    q_r[3][0] = b;
    q_r[3][1] = b;
    q_r[3][2] = b;

    q_w[0] = 1.0 / 24.0;
    q_w[1] = 1.0 / 24.0;
    q_w[2] = 1.0 / 24.0;
    q_w[3] = 1.0 / 24.0;
}
hypreWrapper::BigInt bisection_search_index(const hypreWrapper::BigInt* sortedArr, hypreWrapper::BigInt value, hypreWrapper::BigInt start, hypreWrapper::BigInt end)
{
  hypreWrapper::BigInt fail = -1;
  hypreWrapper::BigInt L = start;
  hypreWrapper::BigInt R = end-1;
  while (L <= R){
    const hypreWrapper::BigInt m = (L+R)/2;
    if(sortedArr[m] < value){
      L = m + 1;
    } else if (sortedArr[m] > value){
      R = m - 1;
    } else {
      return m;
    }
  }
  return fail;
}

hypreWrapper::BigInt linear_search_index(const hypreWrapper::BigInt* unsortedArr, hypreWrapper::BigInt value, hypreWrapper::BigInt start, hypreWrapper::BigInt end)
{
  hypreWrapper::BigInt fail = -1;
  for(hypreWrapper::BigInt idx = start; idx < end; ++idx){
    if(unsortedArr[idx] == value){
      return idx;
    }
  }
  return fail;
}

/* Basis functions and derivatives in 3D */
 hypreWrapper::Real phi_3D_1(hypreWrapper::Real q_r[4][3], int q) { return q_r[q][0]; }
 hypreWrapper::Real phi_3D_2(hypreWrapper::Real q_r[4][3], int q) { return q_r[q][1]; }
 hypreWrapper::Real phi_3D_3(hypreWrapper::Real q_r[4][3], int q) { return q_r[q][2]; }
 hypreWrapper::Real phi_3D_4(hypreWrapper::Real q_r[4][3], int q) { return 1.0 - q_r[q][0] - q_r[q][1] - q_r[q][2]; }
 void dphi(hypreWrapper::Real deriv[3], int q)
{
  if(q==0){
    deriv[0] = 1.0;
    deriv[1] = 0.0;
    deriv[2] = 0.0;
  }

  if(q==1){
    deriv[0] = 0.0;
    deriv[1] = 1.0;
    deriv[2] = 0.0;
  }

  if(q==2){
    deriv[0] = 0.0;
    deriv[1] = 0.0;
    deriv[2] = 1.0;
  }

  if(q==3){
    deriv[0] = -1.0;
    deriv[1] = -1.0;
    deriv[2] = -1.0;
  }
}

/* Math functions */
hypreWrapper::BigInt maximum(hypreWrapper::BigInt a, hypreWrapper::BigInt b) { return a > b ? a : b; }

hypreWrapper::Real determinant(hypreWrapper::Real A[3][3]) {
  /*
   * Computes the determinant of a matrix
   */

  hypreWrapper::Real d_1 = A[0][0] * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
  hypreWrapper::Real d_2 = A[0][1] * (A[1][0] * A[2][2] - A[2][0] * A[1][2]);
  hypreWrapper::Real d_3 = A[0][2] * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);

  return d_1 - d_2 + d_3;
}

void inverse(hypreWrapper::Real invA[3][3], hypreWrapper::Real A[3][3]) {
  /*
   * Computes the inverse of a matrix
   */
  hypreWrapper::Real inv_det_A = 1.0 / determinant(A);
  invA[0][0] = inv_det_A * (A[1][1] * A[2][2] - A[2][1] * A[1][2]);
  invA[0][1] = inv_det_A * (A[0][2] * A[2][1] - A[2][2] * A[0][1]);
  invA[0][2] = inv_det_A * (A[0][1] * A[1][2] - A[1][1] * A[0][2]);
  invA[1][0] = inv_det_A * (A[1][2] * A[2][0] - A[2][2] * A[1][0]);
  invA[1][1] = inv_det_A * (A[0][0] * A[2][2] - A[2][0] * A[0][2]);
  invA[1][2] = inv_det_A * (A[0][2] * A[1][0] - A[1][2] * A[0][0]);
  invA[2][0] = inv_det_A * (A[1][0] * A[2][1] - A[2][0] * A[1][1]);
  invA[2][1] = inv_det_A * (A[0][1] * A[2][0] - A[2][1] * A[0][0]);
  invA[2][2] = inv_det_A * (A[0][0] * A[1][1] - A[1][0] * A[0][1]);
}

 void x_map(hypreWrapper::Real x[3], hypreWrapper::Real q_r[4][3], hypreWrapper::Real x_t[3][4], int q) {
  const int n_dim = 3;
  int i, d;

  for (d = 0; d < n_dim; d++) {
    x[d] = x_t[d][0] * phi_3D_1(q_r, q);
    x[d] += x_t[d][1] * phi_3D_2(q_r, q);
    x[d] += x_t[d][2] * phi_3D_3(q_r, q);
    x[d] += x_t[d][3] * phi_3D_4(q_r, q);
  }
}

 void J_xr_map(hypreWrapper::Real J_xr[3][3], hypreWrapper::Real q_r[4][3], hypreWrapper::Real x_t[3][4]){
  const int n_dim = 3;
  int i, j, k;
  hypreWrapper::Real deriv[3];

  for (i = 0; i < n_dim; i++) {
    for (j = 0; j < n_dim; j++) {
      J_xr[i][j] = 0.0;

      for (k = 0; k < n_dim + 1; k++) {
        dphi(deriv, k);

        J_xr[i][j] += x_t[i][k] * deriv[j];
      }
    }
  }
}

template<typename T>
occa::memory scratchOrAllocateMemory(int nWords, T* src, size_t& bytesRemaining, size_t& byteOffset, size_t& bytesAllocated, bool& allocated);
static occa::kernel computeStiffnessMatrixKernel;
static occa::memory o_stiffness;
static occa::memory o_x;
static occa::memory o_y;
static occa::memory o_z;
static bool constructOnHost = false;

void load();

void construct_coo_graph();
void fem_assembly_device();
void fem_assembly_host();

void matrix_distribution();
void fem_assembly();
void mesh_connectivity(int[8][3], int[8][4]);
hypreWrapper::BigInt maximum(hypreWrapper::BigInt, hypreWrapper::BigInt);

static constexpr int n_dim = 3;
static int n_x, n_y, n_z, n_elem;
static int n_xyz, n_xyze;
static hypreWrapper::BigInt *glo_num;
static dfloat *pmask;
static int num_loc_dofs;
static hypreWrapper::BigInt *dof_map;
static hypreWrapper::BigInt row_start;
static hypreWrapper::BigInt row_end;
static hypreWrapper::IJMatrix *A_bc;

struct COOGraph
{
  hypreWrapper::Int nrows;
  hypreWrapper::BigInt nnz;
  hypreWrapper::BigInt * rows;
  hypreWrapper::BigInt * rowOffsets;
  hypreWrapper::Int * ncols;
  hypreWrapper::BigInt * cols;
  hypreWrapper::Real* vals;
};

static COOGraph coo_graph;

}

static struct comm comm;
struct gs_data {
  struct comm comm;
};
static struct gs_data *gsh;

SEMFEMData* ellipticBuildSEMFEM(const int N_, const int n_elem_,
                          occa::memory _o_x, occa::memory _o_y, occa::memory _o_z,
                          dfloat *pmask_, MPI_Comm mpiComm,
                          long long int *gatherGlobalNodes
                          )
{
  n_x = N_;
  n_y = N_;
  n_z = N_;
  o_x = _o_x;
  o_y = _o_y;
  o_z = _o_z;
  n_elem = n_elem_;
  pmask = pmask_;

  n_xyz = n_x * n_y * n_z;
  n_xyze = n_x * n_y * n_z * n_elem;
  int NuniqueBases = n_xyze;

  {
    comm_ext world;
    world = (comm_ext)mpiComm;
    comm_init(&comm, world);
    gsh = gs_setup(gatherGlobalNodes, NuniqueBases, &comm, 0, gs_pairwise,
                   /* mode */ 0);
  }

  constructOnHost = !platform->device.deviceAtomic;

  if(!constructOnHost) load();

  matrix_distribution();

  fem_assembly();

  SEMFEMData* data;

  {

    const int numRows = row_end - row_start + 1;

    hypreWrapper::BigInt *ownedRows = (hypreWrapper::BigInt*) calloc(numRows, sizeof(hypreWrapper::BigInt));
    int ctr = 0;
    for(hypreWrapper::BigInt row = row_start; row <= row_end; ++row)
      ownedRows[ctr++] = row;
  
    hypreWrapper::Int *ncols = (hypreWrapper::Int*) calloc(numRows, sizeof(hypreWrapper::Int));
    hypreWrapper::IJMatrixGetRowCounts(&A_bc,
      numRows,
      ownedRows,
      ncols);

    int nnz = 0;
    for(int i = 0; i < numRows; ++i)
      nnz += ncols[i];
  
    // construct COO matrix from Hypre matrix
    hypreWrapper::BigInt *hAj = (hypreWrapper::BigInt*) calloc(nnz, sizeof(hypreWrapper::BigInt));
    hypreWrapper::Real   *hAv = (hypreWrapper::Real*) calloc(nnz, sizeof(hypreWrapper::Real));
    hypreWrapper::IJMatrixGetValues(&A_bc,
      -numRows,
      ncols,
      ownedRows,
      &hAj[0],
      &hAv[0]);

    hypreWrapper::BigInt *Ai = (hypreWrapper::BigInt*) calloc(nnz, sizeof(hypreWrapper::BigInt));
    hypreWrapper::BigInt *Aj = (hypreWrapper::BigInt*) calloc(nnz, sizeof(hypreWrapper::BigInt));
    hypreWrapper::Real    *Av = (hypreWrapper::Real*) calloc(nnz, sizeof(hypreWrapper::Real));
    for(int n = 0; n < nnz; ++n) {
       Aj[n] = hAj[n];
       Av[n] = hAv[n];
    } 
    ctr = 0;
    for(int i = 0; i < numRows; ++i){
      hypreWrapper::BigInt row = ownedRows[i];
      for(int col = 0; col < ncols[i]; ++col)
        Ai[ctr++] = row;
    }

    free(hAj);
    free(hAv);
    free(ownedRows);
    free(ncols);
    hypreWrapper::IJMatrixDestroy(&A_bc);

    data = (SEMFEMData*) malloc(sizeof(SEMFEMData));
    data->Ai = Ai;
    data->Aj = Aj;
    data->Av = Av;
    data->nnz = nnz;
    data->rowStart = row_start;
    data->rowEnd = row_end;
    data->dofMap = dof_map;

  }

  return data;
}

namespace{

/* FEM Assembly definition */
void matrix_distribution() {
  /*
   * Ranks the global numbering array after removing the Dirichlet nodes
   * which is then used in the assembly of the matrices to map degrees of
   * freedom to rows of the matrix
   */

  int idx;
  buffer my_buffer;
  hypreWrapper::BigInt idx_start = n_xyze;
  hypreWrapper::BigInt scan_out[2], scan_buf[2];
  comm_scan(scan_out, &comm, gs_long_long, gs_add, &idx_start, 1, scan_buf);
  idx_start = scan_out[0];

  glo_num = (hypreWrapper::BigInt*) malloc(n_xyze * sizeof(hypreWrapper::BigInt));

  for (idx = 0; idx < n_xyze; idx++) {
    if (pmask[idx] > 0.0)
      glo_num[idx] = idx_start + (hypreWrapper::BigInt)idx;
    else
      glo_num[idx] = -1;
  }

  gs(glo_num, gs_long_long, gs_min, 0, gsh, 0);

  /* Rank ids */
  hypreWrapper::BigInt maximum_value_local = 0;
  hypreWrapper::BigInt maximum_value = 0;

  for (idx = 0; idx < n_xyze; idx++) {
    maximum_value_local = (glo_num[idx] > maximum_value_local)
                              ? glo_num[idx]
                              : maximum_value_local;
  }

  comm_allreduce(&comm, gs_long_long, gs_max, &maximum_value_local, 1,
                 &maximum_value);
  const hypreWrapper::BigInt nstar = maximum_value / comm.np + 1;

  struct ranking_tuple {
    hypreWrapper::BigInt rank;
    unsigned int proc;
    unsigned int idx;
  };

  struct array ranking_transfer;
  array_init(ranking_tuple, &ranking_transfer, n_xyze);
  ranking_transfer.n = n_xyze;
  struct ranking_tuple *ranking_tuple_array =
      (struct ranking_tuple *)ranking_transfer.ptr;

  for (idx = 0; idx < ranking_transfer.n; idx++) {
    ranking_tuple_array[idx].rank = glo_num[idx];
    ranking_tuple_array[idx].proc = glo_num[idx] / nstar;
    ranking_tuple_array[idx].idx = idx;
  }

  struct crystal crystal_router_handle;
  crystal_init(&crystal_router_handle, &comm);
  sarray_transfer(ranking_tuple, &ranking_transfer, proc, 1,
                  &crystal_router_handle);
  ranking_tuple_array = (struct ranking_tuple *)ranking_transfer.ptr;

  buffer_init(&my_buffer, 1);
  sarray_sort(ranking_tuple, ranking_transfer.ptr, ranking_transfer.n, rank, 1,
              &my_buffer);

  hypreWrapper::BigInt current_rank = ranking_tuple_array[0].rank;
  hypreWrapper::BigInt current_count = 0;
  ranking_tuple_array[0].rank = current_count;

  for (idx = 1; idx < ranking_transfer.n; idx++) {

    if (ranking_tuple_array[idx].rank > current_rank) {
      current_count++;
      current_rank = ranking_tuple_array[idx].rank;
      ranking_tuple_array[idx].rank = current_count;
    } else if (ranking_tuple_array[idx].rank == current_rank) {
      ranking_tuple_array[idx].rank = current_count;
    } else {
      break;
    }
  }

  current_count += 1;

  hypreWrapper::BigInt rank_start;
  comm_scan(scan_out, &comm, gs_long_long, gs_add, &current_count, 1, scan_buf);
  rank_start = scan_out[0];

  for (idx = 0; idx < ranking_transfer.n; idx++) {
    ranking_tuple_array[idx].rank += rank_start;
  }

  sarray_transfer(ranking_tuple, &ranking_transfer, proc, 1,
                  &crystal_router_handle);
  ranking_tuple_array = (struct ranking_tuple *)ranking_transfer.ptr;

  buffer_init(&my_buffer, 1);
  sarray_sort(ranking_tuple, ranking_transfer.ptr, ranking_transfer.n, idx, 0,
              &my_buffer);

  for (idx = 0; idx < n_xyze; idx++) {
    glo_num[idx] = ranking_tuple_array[idx].rank;
  }

  array_free(&ranking_transfer);
  crystal_free(&crystal_router_handle);
}
void construct_coo_graph() {

  /* Mesh connectivity (Can be changed to fill-out or one-per-vertex) */
  constexpr int num_fem = 8;
  int v_coord[8][3];
  int t_map[8][4];

  mesh_connectivity(v_coord, t_map);

  int E_x = n_x - 1;
  int E_y = n_y - 1;
  int E_z = n_z - 1;

  std::unordered_map<hypreWrapper::BigInt, std::unordered_set<hypreWrapper::BigInt>> graph;
  std::unordered_map<hypreWrapper::Int, std::unordered_set<hypreWrapper::Int>> rowIdxToColIdxMap;
  const int nvert = 8;
  for (int s_z = 0; s_z < E_z; s_z++) {
    for (int s_y = 0; s_y < E_y; s_y++) {
      for (int s_x = 0; s_x < E_x; s_x++) {
        /* Get indices */
        int s[n_dim];

        s[0] = s_x;
        s[1] = s_y;
        s[2] = s_z;

        int idx[nvert];

        for (int i = 0; i < nvert; i++) {
          idx[i] = 0;

          idx[i] += (s[0] + v_coord[i][0]) * 1;
          idx[i] += (s[1] + v_coord[i][1]) * n_x;
          idx[i] += (s[2] + v_coord[i][2]) * n_x * n_x;
        }
        for (int t = 0; t < num_fem; t++) {
          for (int i = 0; i < n_dim + 1; i++) {
            for (int j = 0; j < n_dim + 1; j++) {
              rowIdxToColIdxMap[idx[t_map[t][i]]].insert(idx[t_map[t][j]]);
            }
          }
        }
      }
    }
  }
  for(int e = 0; e < n_elem; ++e){
    for(auto&& idx_row_lookup_and_col_lookups : rowIdxToColIdxMap){
      auto row_lookup = idx_row_lookup_and_col_lookups.first;
      for(auto && col_lookup : idx_row_lookup_and_col_lookups.second)
      {
        if(pmask[e * n_xyz + row_lookup] > 0.0 && pmask[e * n_xyz + col_lookup] > 0.0)
        {
          hypreWrapper::BigInt row = glo_num[e*n_xyz + row_lookup];
          hypreWrapper::BigInt col = glo_num[e*n_xyz + col_lookup];
          graph[row].emplace(col);
        }
      }
    }
  }
  const int nrows = graph.size();
  hypreWrapper::BigInt * rows = (hypreWrapper::BigInt*) malloc(nrows * sizeof(hypreWrapper::BigInt));
  hypreWrapper::BigInt * rowOffsets = (hypreWrapper::BigInt*) malloc((nrows+1) * sizeof(hypreWrapper::BigInt));
  hypreWrapper::Int * ncols = (hypreWrapper::Int*) malloc(nrows * sizeof(hypreWrapper::Int));
  hypreWrapper::BigInt nnz = 0;
  hypreWrapper::BigInt ctr = 0;

  for(auto&& row_and_colset : graph){
    rows[ctr++] = row_and_colset.first;
    nnz += row_and_colset.second.size();
  }
  hypreWrapper::BigInt * cols = (hypreWrapper::BigInt*) malloc(nnz * sizeof(hypreWrapper::BigInt));
  hypreWrapper::Real* vals = (hypreWrapper::Real*) calloc(nnz,sizeof(hypreWrapper::Real));
  std::sort(rows, rows + nrows);
  hypreWrapper::BigInt entryCtr = 0;
  rowOffsets[0] = 0;
  for(auto localrow = 0; localrow < nrows; ++localrow){
    const hypreWrapper::BigInt row = rows[localrow];
    const auto& colset = graph[row];
    const auto size = colset.size();
    ncols[localrow] = size;
    rowOffsets[localrow+1] = rowOffsets[localrow] + size;
    for(auto&& col : colset){
      cols[entryCtr++] = col;
    }
  }

  coo_graph.nrows = nrows;
  coo_graph.nnz = nnz;
  coo_graph.rows = rows;
  coo_graph.rowOffsets = rowOffsets;
  coo_graph.ncols = ncols;
  coo_graph.vals = vals;
  coo_graph.cols = cols;
}

void fem_assembly_host() {

  const hypreWrapper::Int nrows = coo_graph.nrows;
  hypreWrapper::BigInt * rows = coo_graph.rows;
  hypreWrapper::BigInt * rowOffsets = coo_graph.rowOffsets;
  hypreWrapper::Int * ncols = coo_graph.ncols;
  hypreWrapper::BigInt nnz = coo_graph.nnz;
  hypreWrapper::BigInt * cols = coo_graph.cols;
  hypreWrapper::Real* vals = coo_graph.vals;

  hypreWrapper::Real q_r[4][3];
  hypreWrapper::Real q_w[4];
  int v_coord[8][3];
  int t_map[8][4];
  quadrature_rule(q_r, q_w);
  mesh_connectivity(v_coord, t_map);

  dfloat* x = (dfloat*) malloc(n_xyze * sizeof(dfloat));
  dfloat* y = (dfloat*) malloc(n_xyze * sizeof(dfloat));
  dfloat* z = (dfloat*) malloc(n_xyze * sizeof(dfloat));
  o_x.copyTo(x, n_xyze * sizeof(dfloat));
  o_y.copyTo(y, n_xyze * sizeof(dfloat));
  o_z.copyTo(z, n_xyze * sizeof(dfloat));

  const int n_quad = 4;
  const int num_fem = 8;

  for (int e = 0; e < n_elem; e++) {

    /* Cycle through collocated quads/hexes */
    for (int s_z = 0; s_z < n_x-1; s_z++) {
      for (int s_y = 0; s_y < n_x-1; s_y++) {
        for (int s_x = 0; s_x < n_x-1; s_x++) {
          hypreWrapper::Real A_loc[4][4];
          hypreWrapper::Real J_xr[3][3];
          hypreWrapper::Real J_rx[3][3];
          hypreWrapper::Real x_t[3][4];
          hypreWrapper::Real q_x[3];
          /* Get indices */
          int s[n_dim];

          s[0] = s_x;
          s[1] = s_y;
          s[2] = s_z;

          int idx[8];

          for (int i = 0; i < 8; i++) {
            idx[i] = 0;
            idx[i] += (s[0] + v_coord[i][0]) * 1;
            idx[i] += (s[1] + v_coord[i][1]) * n_x;
            idx[i] += (s[2] + v_coord[i][2]) * n_x * n_x;
          }

          /* Cycle through collocated triangles/tets */
          for (int t = 0; t < num_fem; t++) {
            /* Get vertices */
            for (int i = 0; i < n_dim + 1; i++) {
                x_t[0][i] = x[idx[t_map[t][i]] + e * n_x * n_x * n_x];
                x_t[1][i] = y[idx[t_map[t][i]] + e * n_x * n_x * n_x];
                x_t[2][i] = z[idx[t_map[t][i]] + e * n_x * n_x * n_x];
            }

            /* Local FEM matrices */
            /* Reset local stiffness and mass matrices */
            for (int i = 0; i < n_dim + 1; i++) {
              for (int j = 0; j < n_dim + 1; j++) {
                A_loc[i][j] = 0.0;
              }
            }

            /* Build local stiffness matrices by applying quadrature rules */
            J_xr_map(J_xr, q_r, x_t);
            inverse(J_rx, J_xr);
            const hypreWrapper::Real det_J_xr = determinant(J_xr);
            for (int q = 0; q < n_quad; q++) {
              /* From r to x */
              x_map(q_x, q_r, x_t, q);

              /* Integrand */
              for (int i = 0; i < n_dim + 1; i++) {
                hypreWrapper::Real deriv_i[3];
                dphi(deriv_i, i);
                for (int j = 0; j < n_dim + 1; j++) {
                  hypreWrapper::Real deriv_j[3];
                  dphi(deriv_j, j);
                  int alpha, beta;
                  hypreWrapper::Real func = 0.0;

                  for (alpha = 0; alpha < n_dim; alpha++) {
                    hypreWrapper::Real a = 0.0, b = 0.0;

                    for (beta = 0; beta < n_dim; beta++) {
                      a += deriv_i[beta] * J_rx[beta][alpha];

                      b += deriv_j[beta] * J_rx[beta][alpha];
                    }

                    func += a * b;
                  }

                  A_loc[i][j] += func * det_J_xr * q_w[q];
                }
              }
            }
            for (int i = 0; i < n_dim + 1; i++) {
              for (int j = 0; j < n_dim + 1; j++) {
                if ((pmask[idx[t_map[t][i]] + e * n_x * n_x * n_x] > 0.0) &&
                    (pmask[idx[t_map[t][j]] + e * n_x * n_x * n_x] > 0.0)) {
                  hypreWrapper::BigInt row = glo_num[idx[t_map[t][i]] + e * n_x * n_x * n_x];
                  hypreWrapper::BigInt col = glo_num[idx[t_map[t][j]] + e * n_x * n_x * n_x];
                  hypreWrapper::BigInt local_row_id = bisection_search_index(rows, row, 0, nrows);
                  hypreWrapper::BigInt start = rowOffsets[local_row_id];
                  hypreWrapper::BigInt end = rowOffsets[local_row_id+1];

                  hypreWrapper::BigInt id = linear_search_index(cols, col, start, end);
                  vals[id] += A_loc[i][j];
                }
              }
            }
          }
        }
      }
    }
  }


  int err = hypreWrapper::IJMatrixAddToValues(&A_bc, nrows, ncols, rows, cols, vals);
  if (err != 0) {
    if (comm.id == 0)
      printf("hypreWrapper::IJMatrixAddToValues failed!\n");
    ABORT(EXIT_FAILURE);
  }

  free(rows);
  free(rowOffsets);
  free(ncols);
  free(cols);
  free(vals);
  free(x);
  free(y);
  free(z);
}
void fem_assembly_device() {

  const hypreWrapper::Int nrows = coo_graph.nrows;
  hypreWrapper::BigInt * rows = coo_graph.rows;
  hypreWrapper::BigInt * rowOffsets = coo_graph.rowOffsets;
  hypreWrapper::Int * ncols = coo_graph.ncols;
  hypreWrapper::BigInt nnz = coo_graph.nnz;
  hypreWrapper::BigInt * cols = coo_graph.cols;
  hypreWrapper::Real* vals = coo_graph.vals;

  struct AllocationTracker{
    bool o_maskAlloc;
    bool o_glo_numAlloc;
    bool o_rowOffsetsAlloc;
    bool o_rowsAlloc;
    bool o_colsAlloc;
    bool o_valsAlloc;
  };
  AllocationTracker allocations;
  size_t bytesRemaining = platform->o_mempool.bytesAllocated;
  size_t byteOffset = 0;
  size_t bytesAllocated = 0;
  occa::memory o_mask = scratchOrAllocateMemory(
    n_xyze,
    pmask,
    bytesRemaining,
    byteOffset,
    bytesAllocated,
    allocations.o_maskAlloc
  );
  occa::memory o_glo_num = scratchOrAllocateMemory(
    n_xyze,
    glo_num,
    bytesRemaining,
    byteOffset,
    bytesAllocated,
    allocations.o_glo_numAlloc
  );
  occa::memory o_rows = scratchOrAllocateMemory(
    nrows,
    rows,
    bytesRemaining,
    byteOffset,
    bytesAllocated,
    allocations.o_rowsAlloc
  );
  occa::memory o_rowOffsets = scratchOrAllocateMemory(
    nrows+1,
    rowOffsets,
    bytesRemaining,
    byteOffset,
    bytesAllocated,
    allocations.o_rowOffsetsAlloc
  );
  occa::memory o_cols = scratchOrAllocateMemory(
    nnz,
    cols,
    bytesRemaining,
    byteOffset,
    bytesAllocated,
    allocations.o_colsAlloc
  );
  occa::memory o_vals = scratchOrAllocateMemory(
    nnz,
    vals,
    bytesRemaining,
    byteOffset,
    bytesAllocated,
    allocations.o_valsAlloc
  );

  computeStiffnessMatrixKernel(
    n_elem,
    (int)nrows,
    o_x,
    o_y,
    o_z,
    o_mask,
    o_glo_num,
    o_rows,
    o_rowOffsets,
    o_cols,
    o_vals
  );
  o_vals.copyTo(vals, nnz * sizeof(hypreWrapper::Real));

  if(allocations.o_maskAlloc) o_mask.free();
  if(allocations.o_glo_numAlloc) o_glo_num.free();
  if(allocations.o_rowOffsetsAlloc) o_rowOffsets.free();
  if(allocations.o_rowsAlloc) o_rows.free();
  if(allocations.o_colsAlloc) o_cols.free();
  if(allocations.o_valsAlloc) o_vals.free();

  int err = hypreWrapper::IJMatrixAddToValues(&A_bc, nrows, ncols, rows, cols, vals);
  if (err != 0) {
    if (comm.id == 0)
      printf("hypreWrapper::IJMatrixAddToValues failed!\n");
    ABORT(EXIT_FAILURE);
  }

  free(rows);
  free(rowOffsets);
  free(ncols);
  free(cols);
  free(vals);
}

void fem_assembly() {

  hypreWrapper::Real tStart = MPI_Wtime();
  if(comm.id == 0) printf("building matrix ... ");
  int i, j, k, e, d, t, q;
  int idx;
  hypreWrapper::BigInt row;

  row_start = 0;
  row_end = 0;

  for (idx = 0; idx < n_xyze; idx++)
    if (glo_num[idx] >= 0)
      row_end = maximum(row_end, glo_num[idx]);

  hypreWrapper::BigInt scan_out[2], scan_buf[2];
  comm_scan(scan_out, &comm, gs_long_long, gs_max, &row_end, 1, scan_buf);
  if (comm.id > 0)
    row_start = scan_out[0] + 1;

  num_loc_dofs = row_end - row_start + 1;

  dof_map = (dlong *) malloc(num_loc_dofs * sizeof(dlong));

  for (idx = 0; idx < n_xyze; idx++) {
    if ((row_start <= glo_num[idx]) && (glo_num[idx] <= row_end)) {
      dof_map[glo_num[idx] - row_start] = idx;
    }
  }

  /* Assemble FE matrices with boundary conditions applied */
  hypreWrapper::IJMatrixCreate(comm.c, row_start, row_end, row_start, row_end, &A_bc);
  hypreWrapper::IJMatrixSetObjectType(&A_bc);
  hypreWrapper::IJMatrixInitialize(&A_bc);

  construct_coo_graph();

  if(constructOnHost)
  {
    fem_assembly_host();
  }
  else
  {
    fem_assembly_device();
  }

  {
    hypreWrapper::IJMatrixAssemble(&A_bc);
  }

  free(glo_num);

  MPI_Barrier(comm.c);
  if(comm.id == 0) printf("done (%gs)\n", MPI_Wtime() - tStart);
}

void load(){
  computeStiffnessMatrixKernel = platform->kernels.get(
    "computeStiffnessMatrix"
  );
}
void mesh_connectivity(int v_coord[8][3], int t_map[8][4]) {

  (v_coord)[0][0] = 0;
  (v_coord)[0][1] = 0;
  (v_coord)[0][2] = 0;
  (v_coord)[1][0] = 1;
  (v_coord)[1][1] = 0;
  (v_coord)[1][2] = 0;
  (v_coord)[2][0] = 0;
  (v_coord)[2][1] = 1;
  (v_coord)[2][2] = 0;
  (v_coord)[3][0] = 1;
  (v_coord)[3][1] = 1;
  (v_coord)[3][2] = 0;
  (v_coord)[4][0] = 0;
  (v_coord)[4][1] = 0;
  (v_coord)[4][2] = 1;
  (v_coord)[5][0] = 1;
  (v_coord)[5][1] = 0;
  (v_coord)[5][2] = 1;
  (v_coord)[6][0] = 0;
  (v_coord)[6][1] = 1;
  (v_coord)[6][2] = 1;
  (v_coord)[7][0] = 1;
  (v_coord)[7][1] = 1;
  (v_coord)[7][2] = 1;

  (t_map)[0][0] = 0;
  (t_map)[0][1] = 2;
  (t_map)[0][2] = 1;
  (t_map)[0][3] = 4;
  (t_map)[1][0] = 1;
  (t_map)[1][1] = 0;
  (t_map)[1][2] = 3;
  (t_map)[1][3] = 5;
  (t_map)[2][0] = 2;
  (t_map)[2][1] = 6;
  (t_map)[2][2] = 3;
  (t_map)[2][3] = 0;
  (t_map)[3][0] = 3;
  (t_map)[3][1] = 2;
  (t_map)[3][2] = 7;
  (t_map)[3][3] = 1;
  (t_map)[4][0] = 4;
  (t_map)[4][1] = 5;
  (t_map)[4][2] = 6;
  (t_map)[4][3] = 0;
  (t_map)[5][0] = 5;
  (t_map)[5][1] = 7;
  (t_map)[5][2] = 4;
  (t_map)[5][3] = 1;
  (t_map)[6][0] = 6;
  (t_map)[6][1] = 7;
  (t_map)[6][2] = 2;
  (t_map)[6][3] = 4;
  (t_map)[7][0] = 7;
  (t_map)[7][1] = 3;
  (t_map)[7][2] = 6;
  (t_map)[7][3] = 5;
}

template<typename T>
occa::memory scratchOrAllocateMemory(int nWords, T* src, size_t& bytesRemaining, size_t& byteOffset, size_t& bytesAllocated, bool& allocated)
{
  occa::memory o_mem;
  if(nWords * sizeof(T) < bytesRemaining){
    o_mem = platform->o_mempool.o_ptr.slice(byteOffset);
    o_mem.copyFrom(src, nWords * sizeof(T));
    bytesRemaining -= nWords * sizeof(T);
    byteOffset += nWords * sizeof(T);
    allocated = false;
  } else {
    o_mem = platform->device.malloc(nWords * sizeof(T), src);
    allocated = true;
    bytesAllocated += nWords * sizeof(T);
  }
  return o_mem;
}

}
