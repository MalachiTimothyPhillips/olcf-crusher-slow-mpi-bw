#include "particle.hpp"
#include "nekInterfaceAdapter.hpp" // for nek::coeffAB
#include <iomanip>
#include <iostream>
#include <algorithm>

namespace {

std::array<dfloat, particle_t::integrationOrder> particleTimestepperCoeffs(dfloat *dt, int tstep)
{
  constexpr int integrationOrder = particle_t::integrationOrder;
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

lpm_t::lpm_t(nrs_t *nrs_, double newton_tol_) {

  interp_ = std::make_shared<pointInterpolation_t>(nrs_, newton_tol_);
  needsSync = false;

  std::string path;
  int rank = platform->comm.mpiRank;
  path.assign(getenv("NEKRS_INSTALL_DIR"));
  path += "/okl/core/";
  std::string kernelName, fileName;
  const std::string extension = ".okl";

  kernelName = "nStagesSumVector";
  fileName = path + kernelName + extension;
  nStagesSumVectorKernel = platform->device.buildKernel(fileName, platform->kernelInfo, true);

  o_coeffAB = platform->device.malloc(particle_t::integrationOrder * sizeof(dfloat));
}

void lpm_t::reserve(int n)
{
  _x.reserve(n);
  _y.reserve(n);
  _z.reserve(n);
  v.reserve(n);
  id.reserve(n);

  auto& data = interp_->data();

  data.code.reserve(n);
  data.proc.reserve(n);
  data.el.reserve(n);
  data.r.reserve(3*n);
}
void lpm_t::push(particle_t particle)
{
  needsSync = true;
  _x.push_back(particle.x);
  _y.push_back(particle.y);
  _z.push_back(particle.z);
  v.push_back(particle.v);
  id.push_back(particle.id);

  auto& data = interp_->data();

  data.code.push_back(particle.code);
  data.proc.push_back(particle.proc);
  data.el.push_back(particle.el);

  data.r.push_back(particle.r);
  data.r.push_back(particle.s);
  data.r.push_back(particle.t);
}

particle_t lpm_t::remove(int i)
{
  needsSync = true;
  auto& data = interp_->data();
  particle_t part;
  if (i == size() - 1) {
    // just pop the last element
    part.x = _x.back();
    _x.pop_back();
    part.y = _y.back();
    _y.pop_back();
    part.z = _z.back();
    _z.pop_back();

    part.id = id.back();
    id.pop_back();

    part.v = v.back();
    v.pop_back();

    part.r = data.r[3 * i + 0];
    part.s = data.r[3 * i + 1];
    part.t = data.r[3 * i + 2];

    data.r.pop_back(); // each coordinate
    data.r.pop_back();
    data.r.pop_back();

    part.code = data.code.back();
    data.code.pop_back();
    part.proc = data.proc.back();
    data.proc.pop_back();
    part.el = data.el.back();
    data.el.pop_back();
  }
  else {

    // swap last element to i'th position
    swap(i, size() - 1);

    // remove last element
    part = remove(size() - 1);
  }
  return part;
}

void lpm_t::swap(int i, int j)
{
  if (i == j)
    return;

  needsSync = true;

  auto& data = interp_->data();

  std::swap(_x[i], _x[j]);
  std::swap(_y[i], _y[j]);
  std::swap(_z[i], _z[j]);
  std::swap(id[i], id[j]);
  std::swap(v[i], v[j]);

  std::swap(data.r[3*i + 0], data.r[3*j + 0]);
  std::swap(data.r[3*i + 1], data.r[3*j + 1]);
  std::swap(data.r[3*i + 2], data.r[3*j + 2]);
  std::swap(data.code[i], data.code[j]);
  std::swap(data.proc[i], data.proc[j]);
  std::swap(data.el[i]  , data.el[j]);
}

void lpm_t::find(bool printWarnings)
{
  if(profile){
    platform->timer.tic("lpm_t::find", 1);
  }
  dlong n = size();

  interp_->addPoints(n, _x.data(), _y.data(), _z.data());

  interp_->find(printWarnings);

  if(profile){
    platform->timer.toc("lpm_t::find");
  }
}
void lpm_t::migrate()
{
  if(profile){
    platform->timer.tic("lpm_t::migrate", 1);
  }
  auto & data = interp_->data();
  int mpi_rank = platform_t::getInstance()->comm.mpiRank;

  struct array transfer;
  array_init(particle_t, &transfer, 128);

  int index = 0;
  int unfound_count = 0;
  while (index < size()) {
    if (data.code[index] == 2) {
      swap(index, unfound_count);
      ++unfound_count;
      ++index;
    }
    else if (data.proc[index] != mpi_rank) {
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

  sarray_transfer(particle_t, &transfer, proc, true, crystalRouter(interp_->ptr()));

  reserve(size() + transfer.n);
  particle_t *transfer_ptr = (particle_t *)transfer.ptr;
  for (int i = 0; i < transfer.n; ++i) {
    transfer_ptr[i].proc = mpi_rank; // sarray_transfer sets proc to be the sender
    push(transfer_ptr[i]);
  }

  array_free(&transfer);

  if(profile){
    platform->timer.toc("lpm_t::migrate");
  }
}

void lpm_t::interpLocal(occa::memory field, occa::memory o_out, dlong nFields)
{
  if(profile){
    platform->timer.tic("lpm_t::interpLocal", 1);
  }

  auto & data = interp_->data();

  dlong pn = size();
  dlong offset = 0;
  while (offset < pn && data.code[offset] == 2)
    ++offset;

  // TODO: is the offset ever non-zero??? If so, this would be much simpler.
  pn -= offset;

  interp_->evalLocalPoints(field,
                           nFields,
                           data.el.data() + offset,
                           data.r.data() + 3 * offset,
                           o_out,
                           pn);
  if(profile){
    platform->timer.toc("lpm_t::interpLocal");
  }
}

namespace{
std::string lpm_vtu_data(std::string fieldName, int nComponent, int distance)
{
  return "<DataArray type=\"Float32\" Name=\"" + fieldName + "\" NumberOfComponents=\"" 
    + std::to_string(nComponent) + "\" format=\"append\" offset=\""
    + std::to_string(distance) + "\"/>\n";
}
}

void lpm_t::write(dfloat time) const
{
  if(profile){
    platform->timer.tic("lpm_t::write", 1);
  }
  static_assert(sizeof(float) == 4, "Requires float be 32-bit");
  static_assert(sizeof(int) == 4, "Requires int be 32-bit");

  static int out_step = 0;
  ++out_step;

  MPI_Comm mpi_comm = platform->comm.mpiComm;
  int mpi_rank = platform->comm.mpiRank;
  int mpi_size = platform->comm.mpiCommSize;
  dlong npart = this->size();
  
  dlong global_npart = npart;

  MPI_Allreduce(MPI_IN_PLACE, &global_npart, 1, MPI_DLONG, MPI_SUM, platform->comm.mpiComm);

  std::ostringstream output;
  output << "par" << std::setw(5) << std::setfill('0') << out_step << ".vtu";
  std::string fname = output.str();

  hlong p_offset = 0;
  MPI_Exscan(&p_offset, &npart, 1, MPI_HLONG, MPI_SUM, mpi_comm);

  if(mpi_rank == 0){
    std::ofstream file(fname, std::ios::trunc);
    file.close();
  }

  MPI_File file_out;
  MPI_File_open(mpi_comm, fname.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file_out);

  std::string message = "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
  message += "\t<UnstructuredGrid>\n";
  message += "\t\t<FieldData>\n";
  message += "\t\t\t<DataArray type=\"Float32\" Name=\"TIME\" NumberOfTuples=\"1\" format=\"ascii\"> " + std::to_string(time) + " </DataArray>\n";
  message += "\t\t\t<DataArray type=\"Int32\" Name=\"CYCLE\" NumberOfTuples=\"1\" format=\"ascii\"> " + std::to_string(out_step) + " </DataArray>\n";
  message += "\t\t</FieldData>\n";
  message += "\t\t<Piece NumberOfPoints=\"" + std::to_string(global_npart) + "\" NumberOfCells=\"0\">\n";
  message += "\t\t\t<Points>\n";
  message += "\t\t\t\t" + lpm_vtu_data("Position", 3, 0);
  message += "\t\t\t</Points>\n";

  // TODO: add once the ability to add arbitrary particle equations is finished
  message += "\t\t\t<PointData>\n";
  message += "\t\t\t</PointData>\n";

  message += "\t\t\t<Cells>\n";
  message += "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\"/>\n";
  message += "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\"/>\n";
  message += "\t\t\t\t<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\"/>\n";
  message += "\t\t\t</Cells>\n";
  message += "\t\t</Piece>\n";
  message += "\t</UnstructuredGrid>\n";
  message += "\t<AppendedData encoding=\"raw\">\n";
  message += "_";
  message += std::to_string(3 * sizeof(float) * global_npart);

  if (mpi_rank == 0) {
    MPI_File_write(file_out, message.c_str(), message.length(), MPI_CHAR, MPI_STATUS_IGNORE);
  }

  // byte displacements
  MPI_Offset position = message.length() + sizeof(float) * (3 * p_offset + 1);
  int count_pos = 3 * npart;

  MPI_File_set_view(file_out, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

  // buffer
  std::vector<float> positions(3 * npart, 0.0);
  std::ostringstream coords;
  for(int particle = 0; particle < npart; ++particle){
    positions[3 * particle + 0] = static_cast<float>(this->_x[particle]);
    positions[3 * particle + 1] = static_cast<float>(this->_y[particle]);
    positions[3 * particle + 2] = static_cast<float>(this->_z[particle]);
  }

  // TODO:
  // will need to output other fields than just position
  MPI_File_write_all(file_out, positions.data(), positions.size(), MPI_FLOAT, MPI_STATUS_IGNORE);

  MPI_File_get_size(file_out, &position);

  MPI_File_set_view(file_out, position, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

  if(mpi_rank == 0){
    message = "";
    message += "</AppendedData>\n";
    message += "</VTKFile>";
    MPI_File_write(file_out, message.c_str(), message.length(), MPI_CHAR, MPI_STATUS_IGNORE);
  }

  MPI_File_close(&file_out);
  if(profile){
    platform->timer.toc("lpm_t::write");
  }
}

void lpm_t::syncToDevice()
{
  if(!needsSync) return;

  needsSync = false;

  const auto n = this->size();

  if(profile){
    platform->timer.tic("lpm_t::syncToDevice", 1);
  }

  if(o_Uinterp.size() < 3 * particle_t::integrationOrder * this->size() * sizeof(dfloat)){

    o_Uinterp.free();

    o_Uinterp = platform->device.malloc(3 * particle_t::integrationOrder * this->size() * sizeof(dfloat));

  }

  if(o_x.size() < 3 * this->size() * sizeof(dfloat)){

    o_x.free();
    o_y.free();
    o_z.free();

    o_x = platform->device.malloc(this->size() * sizeof(dfloat));
    o_y = platform->device.malloc(this->size() * sizeof(dfloat));
    o_z = platform->device.malloc(this->size() * sizeof(dfloat));

  }

  o_x.copyFrom(_x.data(), this->size() * sizeof(dfloat));
  o_y.copyFrom(_y.data(), this->size() * sizeof(dfloat));
  o_z.copyFrom(_z.data(), this->size() * sizeof(dfloat));

  // copy lagged states
  const dlong Nbyte = 3 * n * sizeof(dfloat);
  std::vector<dfloat> velocity(3*n, 0.0);
  for(int state = 1; state < particle_t::integrationOrder; ++state)
  {
    for(int i = 0; i < this->size(); ++i){
      velocity[i + this->size() * 0] = v[i][3*state + 0];
      velocity[i + this->size() * 1] = v[i][3*state + 1];
      velocity[i + this->size() * 2] = v[i][3*state + 2];
    }

    o_Uinterp.copyFrom(velocity.data(),
      Nbyte,
      state * Nbyte);
  }



  if(profile){
    platform->timer.toc("lpm_t::syncToDevice");
  }
}

void lpm_t::addPostIntegrationWork(std::function<void(lpm_t&)> work)
{
  postIntegrationWork.push_back(work);
}

void lpm_t::advance(dfloat * dt, int tstep)
{
  if(profile){
    platform->timer.tic("lpm_t::advance", 1);
  }

  auto coeffs = particleTimestepperCoeffs(dt, tstep);
  o_coeffAB.copyFrom(coeffs.data(), particle_t::integrationOrder * sizeof(dfloat));

  const auto n = this->size();

#if 1
  nStagesSumVectorKernel(n, n, particle_t::integrationOrder, o_coeffAB, o_Uinterp, o_x, o_y, o_z);

  for(auto&& work : postIntegrationWork)
  {
    work(*this);
  }

  // lag velocity states
  const dlong Nbyte = 3 * n * sizeof(dfloat);
  for (int s = particle_t::integrationOrder; s > 1; s--) {
    o_Uinterp.copyFrom(
        o_Uinterp, Nbyte, (s - 1) * Nbyte, (s - 2) * Nbyte);
  }

  // copy position results back
  o_x.copyTo(_x.data(), n * sizeof(dfloat));
  o_y.copyTo(_y.data(), n * sizeof(dfloat));
  o_z.copyTo(_z.data(), n * sizeof(dfloat));

  // copy lagged velocity state back
  std::vector<dfloat> velocity(3*n, 0.0);
  for(int state = 0; state < particle_t::integrationOrder; ++state)
  {
    o_Uinterp.copyTo(velocity.data(),
      Nbyte,
      state * Nbyte);

    for(int i = 0; i < this->size(); ++i){
      v[i][3*state + 0] = velocity[i + this->size() * 0];
      v[i][3*state + 1] = velocity[i + this->size() * 1];
      v[i][3*state + 2] = velocity[i + this->size() * 2];
    }

  }

#else
  for (int i = 0; i < this->size(); ++i) {
    // Update particle position and velocity history

    this->v[i][0] = u1[0 * this->size() + i];
    this->v[i][1] = u1[1 * this->size() + i];
    this->v[i][2] = u1[2 * this->size() + i];

    int k = 0;
    this->_x[i] += coeffs[0] * this->v[i][k];
    for (int j = 1; j < particle_t::integrationOrder; ++j) {
      this->_x[i] += coeffs[j] * this->v[i][3*j + k];
    }
    this->v[i][2*3 + k] = this->v[i][1*3 + k];
    this->v[i][1*3 + k] = this->v[i][0*3 + k];

    k++;
    this->_y[i] += coeffs[0] * this->v[i][k];
    for (int j = 1; j < particle_t::integrationOrder; ++j) {
      this->_y[i] += coeffs[j] * this->v[i][3*j + k];
    }
    this->v[i][2*3 + k] = this->v[i][1*3 + k];
    this->v[i][1*3 + k] = this->v[i][0*3 + k];

    k++;
    this->_z[i] += coeffs[0] * this->v[i][k];
    for (int j = 1; j < particle_t::integrationOrder; ++j) {
      this->_z[i] += coeffs[j] * this->v[i][3*j + k];
    }
    this->v[i][2*3 + k] = this->v[i][1*3 + k];
    this->v[i][1*3 + k] = this->v[i][0*3 + k];
  }
#endif

  if(profile){
    platform->timer.toc("lpm_t::advance");
  }

}

void lpm_t::update(occa::memory o_fld, dfloat *dt, int tstep)
{
  if(profile){
    platform->timer.tic("lpm_t::update", 1);
  }

  this->find();
  this->migrate();

  this->syncToDevice();

  this->interpLocal(o_fld, o_Uinterp, 3);

  this->advance(dt, tstep);

  if(profile){
    platform->timer.toc("lpm_t::update");
  }
}