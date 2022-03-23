#ifndef FINDPTS_HPP
#define FINDPTS_HPP

#include "occa.hpp"
#include <mpi.h>
#include "ogstypes.h" // for dfloat, dlong
#include <limits>
#include <tuple>
#include <vector>

struct crystal;
struct hashData_t;

struct findpts_t {
  MPI_Comm comm;
  int rank;
  dfloat tol;
  crystal *cr;
  hashData_t *hash;
  occa::device device;
  occa::kernel local_eval_kernel;
  occa::kernel local_eval_many_kernel;
  occa::kernel local_kernel;

  // data for elx
  occa::memory o_x;
  occa::memory o_y;
  occa::memory o_z;

  // data for wtend
  occa::memory o_wtend_x;
  occa::memory o_wtend_y;
  occa::memory o_wtend_z;

  // SoA variant of obbox
  occa::memory o_c;
  occa::memory o_A;
  occa::memory o_min;
  occa::memory o_max;

  occa::memory o_offset;

  dfloat hashMin[3];
  dfloat hashFac[3];
  dlong hash_n;
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

  findpts_data_t() {}

  findpts_data_t(int npt)
  {
    code = std::vector<dlong>(npt, 0);
    proc = std::vector<dlong>(npt, 0);
    el = std::vector<dlong>(npt, 0);
    r = std::vector<dfloat>(3 * npt, 0);
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

findpts_t *findptsSetup(MPI_Comm comm,
                        const dfloat *const x,
                        const dfloat *const y,
                        const dfloat *const z,
                        const dlong Nq,
                        const dlong Nelements,
                        const dlong m,
                        const dfloat bbox_tol,
                        const hlong local_hash_size,
                        const hlong global_hash_size,
                        const dlong npt_max,
                        const dfloat newt_tol,
                        occa::device &device);
void findptsFree(findpts_t *fd);
void findpts(findpts_data_t *findPtsData, const dfloat *const x_base[], const dlong npt, findpts_t *const fd);

void findptsEval(const dlong npt,
                 occa::memory o_in,
                 findpts_t *fd,
                 findpts_data_t *findPtsData,

                 dfloat *out_base);
void findptsEval(const dlong npt,
                 const dlong nFields,
                 const dlong inputOffset,
                 const dlong outputOffset,
                 occa::memory o_in,
                 findpts_t *fd,
                 findpts_data_t *findPtsData,
                 dfloat *out_base);

void findptsLocalEval(const dlong npt,
                      occa::memory o_in,
                      occa::memory o_el,
                      occa::memory o_r,
                      findpts_t *fd,
                      occa::memory o_out);

void findptsLocalEval(const dlong npt,
                      const dlong nFields,
                      const dlong inputOffset,
                      const dlong outputOffset,
                      occa::memory o_in,
                      occa::memory o_el,
                      occa::memory o_r,
                      findpts_t *fd,
                      occa::memory o_out);

struct crystal;
crystal *crystalRouter(findpts_t *const fd);

std::vector<occa::kernel> initFindptsKernels(MPI_Comm comm, occa::device device, dlong D, dlong Nq);

#endif
