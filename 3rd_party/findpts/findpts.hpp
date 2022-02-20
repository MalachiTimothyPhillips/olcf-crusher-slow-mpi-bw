#ifndef FINDPTS_HPP
#define FINDPTS_HPP

#include "occa.hpp"
#include <mpi.h>
#include "ogstypes.h" // for dfloat, dlong
#include <limits>
#include <tuple>

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
  bool owned;
  dlong *code_base;
  dlong *proc_base;
  dlong *el_base;
  dfloat *r_base;
  dfloat *dist2_base;

  findpts_data_t() { owned = false; }

  findpts_data_t(dlong *_code_base,
                     dlong *_proc_base,
                     dlong *_el_base,
                     dfloat *_r_base,
                     dfloat *_dist2_base)
      : code_base(_code_base), proc_base(_proc_base), el_base(_el_base), r_base(_r_base),
        dist2_base(_dist2_base)
  {
    owned = false;
  }

  findpts_data_t(int npt)
  {
    code_base = (dlong *)calloc(npt, sizeof(dlong));
    proc_base = (dlong *)calloc(npt, sizeof(dlong));
    el_base = (dlong *)calloc(npt, sizeof(dlong));
    r_base = (dfloat *)calloc(3 * npt, sizeof(dfloat));
    dist2_base = (dfloat *)calloc(npt, sizeof(dfloat));

    for (dlong i = 0; i < npt; ++i) {
      dist2_base[i] = std::numeric_limits<dfloat>::max();
      code_base[i] = 2;
    }
    owned = true;
  }

  ~findpts_data_t()
  {
    if (owned) {
      free(code_base);
      free(proc_base);
      free(el_base);
      free(r_base);
      free(dist2_base);
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
                    occa::memory d_in,
                    findpts_t *const fd);

void findptsLocalEval(
        dfloat *const  out_base, const dlong  out_stride,
  const dlong  *const   el_base, const dlong   el_stride,
  const dfloat *const    r_base, const dlong    r_stride,
  const dlong npt, const dfloat *const in, findpts_t *const fd);

void findptsLocalEval(
  occa::memory out_base, const dlong  out_stride,
  occa::memory  el_base, const dlong   el_stride,
  occa::memory   r_base, const dlong    r_stride,
  const dlong npt, occa::memory d_in, findpts_t *const fd);

struct crystal;
crystal* crystalRouter(findpts_t *const fd);

std::tuple<occa::kernel, occa::kernel, occa::kernel> initFindptsKernels(
      MPI_Comm comm, occa::device device, dlong D, dlong Nq);

#endif
