#ifndef FINDPTS_HPP
#define FINDPTS_HPP

#include "occa.hpp"
#include <mpi.h>
#include "ogstypes.h" // for dfloat, dlong
#include <limits>
#include <tuple>
#include <vector>

struct findpts_t {
  int D;
  void *findpts_data;
  occa::device *device;
  occa::kernel local_eval_kernel;
  occa::kernel local_eval_vector_kernel;
  occa::kernel local_kernel;
  occa::memory o_fd_local;
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
  const dlong D, MPI_Comm comm,
  const dfloat *const elx[],
  const dlong n[], const dlong nel,
  const dlong m[], const dfloat bbox_tol,
  const hlong local_hash_size, const hlong global_hash_size,
  const dlong npt_max, const dfloat newt_tol,
  occa::device *device = nullptr);
void findptsFree(findpts_t *fd);
void findpts(findpts_data_t *findPtsData,
                const dfloat *const x_base[],
                const dlong x_stride[],
                const dlong npt,
                findpts_t *const fd);
void findptsEval(dfloat *const out_base,
                    findpts_data_t *findPtsData,
                    const dlong npt,
                    const dfloat *const in,
                    findpts_t *const fd);

void findptsEval(dfloat *const out_base,
                    findpts_data_t *findPtsData,
                    const dlong npt,
                    occa::memory o_in,
                    findpts_t *const fd);

void findptsLocalEval(
        dfloat *const  out_base, const dlong  out_stride,
  const dlong  *const   el_base, const dlong   el_stride,
  const dfloat *const    r_base, const dlong    r_stride,
  const dlong npt, const dfloat *const in, findpts_t *const fd);

void findptsLocalEval(
  occa::memory o_out, const dlong  out_stride,
  occa::memory  o_el, const dlong   el_stride,
  occa::memory   o_r, const dlong    r_stride,
  const dlong npt, occa::memory o_in, findpts_t *const fd);

struct crystal;
crystal* crystalRouter(findpts_t *const fd);

std::tuple<occa::kernel, occa::kernel, occa::kernel> initFindptsKernels(
      MPI_Comm comm, occa::device device, dlong D, dlong Nq);

#endif
