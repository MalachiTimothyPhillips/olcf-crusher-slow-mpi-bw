#ifndef FINDPTS_HPP
#define FINDPTS_HPP

#include "occa.hpp"
#include <mpi.h>
#include "ogstypes.h" // for dfloat, dlong
#include <limits>
#include <tuple>
#include <vector>

class gslibFindptsData_t;

struct findpts_t {
  int D;
  gslibFindptsData_t *findpts_data;
  occa::device device;
  occa::kernel local_eval_kernel;
  occa::kernel local_eval_many_kernel;
  occa::kernel local_kernel;
  occa::memory o_fd_local;

  occa::memory o_x;
  occa::memory o_y;
  occa::memory o_z;
};

struct findpts_data_t {
  std::vector<dlong> code;
  std::vector<dlong> proc;
  std::vector<dlong> el;
  std::vector<dfloat> r;
  std::vector<dfloat> dist2;

  dlong *code_base;
  dlong *proc_base;
  dlong *el_base;
  dfloat *r_base;
  dfloat *dist2_base;

  findpts_data_t(){}

  findpts_data_t(int npt)
  {
    code = std::vector<dlong>(npt, 0);
    proc = std::vector<dlong>(npt, 0);
    el = std::vector<dlong>(npt, 0);
    r = std::vector<dfloat>(3*npt, 0);
    dist2 = std::vector<dfloat>(npt, 0);

    code_base = code.data();
    proc_base = proc.data();
    el_base = el.data();
    r_base = r.data();
    dist2_base = dist2.data();

    for (dlong i = 0; i < npt; ++i) {
      dist2_base[i] = std::numeric_limits<dfloat>::max();
      code_base[i] = 2;
    }
  }

};

findpts_t *findptsSetup(
  MPI_Comm comm,
  const dfloat* const x,
  const dfloat* const y,
  const dfloat* const z,
  const dlong Nq,
  const dlong Nelements,
  const dlong m,
  const dfloat bbox_tol,
  const hlong local_hash_size, const hlong global_hash_size,
  const dlong npt_max, const dfloat newt_tol,
  occa::device &device);
void findptsFree(findpts_t *fd);
void findpts(findpts_data_t *findPtsData,
                const dfloat *const x_base[],
                const dlong npt,
                findpts_t *const fd);

void findptsEval(const dlong npt,
  occa::memory o_in,
  findpts_t* fd,
  findpts_data_t* findPtsData,
 
  dfloat * out_base);
void findptsEval(const dlong npt,
  const dlong nFields,
  const dlong inputOffset,
  const dlong outputOffset,
  occa::memory o_in,
  findpts_t* fd,
  findpts_data_t* findPtsData,
  dfloat * out_base);

void findptsLocalEval(
  const dlong npt,
  occa::memory o_in,
  occa::memory o_el,
  occa::memory o_r,
  findpts_t * fd,
  occa::memory o_out);

void findptsLocalEval(
  const dlong npt,
  const dlong nFields,
  const dlong inputOffset,
  const dlong outputOffset,
  occa::memory o_in,
  occa::memory o_el,
  occa::memory o_r,
  findpts_t * fd,
  occa::memory o_out);

struct crystal;
crystal* crystalRouter(findpts_t *const fd);

std::vector<occa::kernel> initFindptsKernels(
      MPI_Comm comm, occa::device device, dlong D, dlong Nq);

#endif
