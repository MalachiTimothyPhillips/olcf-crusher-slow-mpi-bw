#include "particle.hpp"
#include "nekInterfaceAdapter.hpp" // for nek::coeffAB

namespace {
std::array<dfloat, historyData_t::integrationOrder> particleTimestepperCoeffs(dfloat *dt, int tstep)
{
  constexpr int integrationOrder = historyData_t::integrationOrder;
  std::array<dfloat, integrationOrder> coeffs;
  const int particleOrder = mymin(tstep, integrationOrder);
  nek::coeffAB(coeffs.data(), dt, particleOrder);
  for (int i = 0; i < particleOrder; ++i)
    coeffs[i] *= dt[0];
  for (int i = integrationOrder; i > particleOrder; i--)
    coeffs[i - 1] = 0.0;

  return coeffs;
}
} // namespace

void particles_t::reserve(int n)
{
  for (int i = 0; i < 3; ++i) {
    x[i].reserve(n);
  }
  code.reserve(n);
  proc.reserve(n);
  el.reserve(n);
  r.reserve(n);
  extra.reserve(n);
}
void particles_t::push(particle_t particle)
{
  for (int j = 0; j < 3; ++j) {
    x[j].push_back(particle.x[j]);
  }
  code.push_back(particle.code);
  proc.push_back(particle.proc);
  el.push_back(particle.el);
  r.push_back(particle.r);
  extra.push_back(particle.extra);
}

particle_t particles_t::remove(int i)
{
  particle_t part;
  if (i == size() - 1) {
    // just pop the last element
    for (int j = 0; j < 3; ++j) {
      part.x[j] = x[j].back();
      x[j].pop_back();
      part.r[j] = r.back()[j];
    }
    r.pop_back();
    part.code = code.back();
    code.pop_back();
    part.proc = proc.back();
    proc.pop_back();
    part.el = el.back();
    el.pop_back();
    part.extra = extra.back();
    extra.pop_back();
  }
  else {
    // swap last element to i'th position
    for (int j = 0; j < 3; ++j) {
      part.x[j] = x[j][i];
      x[j][i] = x[j].back();
      x[j].pop_back();
      part.r[j] = r[i][j];
      r[i][j] = r.back()[j];
    }
    r.pop_back();
    part.code = code[i];
    code[i] = code.back();
    code.pop_back();
    part.proc = proc[i];
    proc[i] = proc.back();
    proc.pop_back();
    part.el = el[i];
    el[i] = el.back();
    el.pop_back();
    part.extra = extra[i];
    extra[i] = extra.back();
    extra.pop_back();
  }
  return part;
}

void particles_t::swap(int i, int j)
{
  if (i == j)
    return;

  for (int d = 0; d < 3; ++d) {
    std::swap(x[d][i], x[d][j]);
    std::swap(r[d][i], r[d][j]);
  }
  std::swap(code[i], code[j]);
  std::swap(proc[i], proc[j]);
  std::swap(el[i], el[j]);
  std::swap(extra[i], extra[j]);
}

void particles_t::find(bool printWarnings, dfloat *dist2In, dlong dist2Stride)
{
  dlong n = size();
  dfloat *dist2;
  if (dist2In != nullptr) {
    dist2 = dist2In;
  }
  else {
    dist2 = new dfloat[n];
    dist2Stride = 1;
  }
  dfloat *xBase[3];
  dlong xStride[3];
  for (int i = 0; i < 3; ++i) {
    xBase[i] = x[i].data();
    xStride[i] = 1;
  }

  ogs_findpts_data_t data(code.data(), proc.data(), el.data(), &(r.data()[0][0]), dist2);

  interp_->find(xBase, xStride, &data, size(), printWarnings);
  if (dist2In == nullptr) {
    delete[] dist2;
  }
}
void particles_t::migrate()
{
  int mpi_rank = platform_t::getInstance()->comm.mpiRank;

  struct array transfer;
  array_init(particle_t, &transfer, 128);

  int index = 0;
  int unfound_count = 0;
  while (index < size()) {
    if (code[index] == 2) {
      swap(index, unfound_count);
      ++unfound_count;
      ++index;
    }
    else if (proc[index] != mpi_rank) {
      // remove index'th element and move the last point to index'th storage
      array_reserve(particle_t, &transfer, transfer.n + 1);
      ((particle_t *)transfer.ptr)[transfer.n] = remove(index);
      ++transfer.n;
    }
    else {
      // keep point on this process
      ++index;
    }
  }

  sarray_transfer(particle_t, &transfer, proc, true, ogsCrystalRouter(interp_->ptr()));

  reserve(size() + transfer.n);
  particle_t *transfer_ptr = (particle_t *)transfer.ptr;
  for (int i = 0; i < transfer.n; ++i) {
    transfer_ptr[i].proc = mpi_rank; // sarray_transfer sets proc to be the sender
    push(transfer_ptr[i]);
  }

  array_free(&transfer);
}

void particles_t::interpLocal(occa::memory field, dfloat *out[], dlong nFields)
{
  dlong pn = size();
  dlong offset = 0;
  while (offset < pn && code[offset] == 2)
    ++offset;
  pn -= offset;

  dfloat **outOffset = new dfloat *[nFields];
  dlong *outStride = new dlong[nFields];
  for (dlong i = 0; i < nFields; ++i) {
    outOffset[i] = out[i] + offset;
    outStride[i] = 1;
  }

  interp_->evalLocalPoints(field,
                           nFields,
                           el.data() + offset,
                           1,
                           &(r.data()[offset][0]),
                           3,
                           outOffset,
                           outStride,
                           pn);
  delete[] outOffset;
}
void particles_t::interpLocal(dfloat *field, dfloat *out[], dlong nFields)
{
  dlong pn = size();
  dlong offset = 0;
  while (offset < pn && code[offset] == 2)
    ++offset;
  pn -= offset;

  dfloat **outOffset = new dfloat *[nFields];
  dlong *outStride = new dlong[nFields];
  for (dlong i = 0; i < nFields; ++i) {
    outOffset[i] = out[i] + offset;
    outStride[i] = 1;
  }

  interp_->evalLocalPoints(field,
                           nFields,
                           el.data() + offset,
                           1,
                           &(r.data()[offset][0]),
                           3,
                           outOffset,
                           outStride,
                           pn);
  delete[] outOffset;
}

void particles_t::write()
{

  static int out_step = 0;
  ++out_step;

  MPI_Comm mpi_comm = platform->comm.mpiComm;
  int mpi_rank = platform->comm.mpiRank;
  int mpi_size = platform->comm.mpiCommSize;
  dlong npart = this->size();

  char fname[128];
  sprintf(fname, "part%05d.3d", out_step);

  constexpr hlong p_size = 18 * 4;

  hlong p_offset = npart;
  MPI_Exscan(MPI_IN_PLACE, &p_offset, 1, MPI_HLONG, MPI_SUM, mpi_comm);
  hlong file_offset = p_offset * p_size + 12;

  MPI_File file_out;
  MPI_File_open(mpi_comm, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_out);
  if (mpi_rank == 0) {
    MPI_File_write(file_out, "X Y Z Color\n", 12, MPI_CHAR, MPI_STATUS_IGNORE);
  }
  else {
    MPI_File_seek(file_out, file_offset, MPI_SEEK_SET);
  }

  char *char_buffer = new char[npart * p_size + 1];
  for (dlong ii = 0; ii < npart; ++ii) {
    sprintf(char_buffer + ii * p_size,
            "%17.9e %17.9e %17.9e %17.9e\n",
            this->x[0][ii],
            this->x[1][ii],
            this->x[2][ii],
            this->extra[ii].color);
  }
  MPI_File_write_all(file_out, char_buffer, p_size * npart, MPI_CHAR, MPI_STATUS_IGNORE);
  delete[] char_buffer;

  MPI_File_close(&file_out);
}
void particles_t::update(occa::memory o_fld, dfloat *dt, int tstep)
{

  this->find();
  this->migrate();

  dfloat *u1[3];
  u1[0] = new dfloat[3 * this->size()];
  u1[1] = u1[0] + this->size();
  u1[2] = u1[1] + this->size();
  this->interpLocal(o_fld, u1, 3);

  auto coeffs = particleTimestepperCoeffs(dt, tstep);

  for (int i = 0; i < this->size(); ++i) {
    // Update particle position and velocity history
    for (int k = 0; k < 3; ++k) {
      this->x[k][i] += coeffs[0] * u1[k][i];
      for (int j = 1; j < historyData_t::integrationOrder - 1; ++j) {
        this->x[k][i] += coeffs[j] * this->extra[i].v_hist[j - 1][k];
      }
      this->extra[i].v_hist[1][k] = this->extra[i].v_hist[0][k];
      this->extra[i].v_hist[0][k] = u1[k][i];
    }
  }

  delete[] u1[0];
}