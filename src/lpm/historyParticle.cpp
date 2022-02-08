#include "historyParticle.hpp"

#define OUT_CHUNK_SIZE 100000 /* chunk size for outputing particles */
#define USE_MPIIO      true

historyData_t::historyData_t(dfloat v_hist_[2][3], dfloat color_, hlong id_)

{
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      v_hist[i][j] = v_hist_[i][j];
    }
  }
  id = id_;
  color = color_;
}

void particleOut(const historyParticles_t& particles){
  static dfloat x_temp[USE_MPIIO?0:OUT_CHUNK_SIZE][4];
  static dfloat x_root[USE_MPIIO?0:OUT_CHUNK_SIZE][4];

  static int out_step = 0;
  ++out_step;

  MPI_Comm mpi_comm = platform_t::getInstance()->comm.mpiComm;
  int mpi_rank = platform_t::getInstance()->comm.mpiRank;
  int mpi_size = platform_t::getInstance()->comm.mpiCommSize;
  dlong npart = particles.size();

  char fname[128];
  sprintf(fname, "part%05d.3d", out_step);

  if (USE_MPIIO) {
    constexpr hlong p_size = 18*4;

    hlong p_offset = npart;
    MPI_Exscan(MPI_IN_PLACE, &p_offset, 1, MPI_HLONG, MPI_SUM, mpi_comm);
    hlong file_offset = p_offset*p_size+12;

    MPI_File file_out;
    MPI_File_open(mpi_comm, fname, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file_out);
    if(mpi_rank == 0) {
      MPI_File_write(file_out, "X Y Z Color\n", 12, MPI_CHAR, MPI_STATUS_IGNORE);
    } else {
      MPI_File_seek(file_out, file_offset, MPI_SEEK_SET);
    }

    char *char_buffer = new char[npart*p_size + 1];
    for (dlong ii = 0; ii < npart; ++ii) {
      sprintf(char_buffer+ii*p_size, "%17.9e %17.9e %17.9e %17.9e\n",
              particles.x[0][ii], particles.x[1][ii], particles.x[2][ii], particles.extra[ii].color);
    }
    MPI_File_write_all(file_out, char_buffer, p_size*npart, MPI_CHAR, MPI_STATUS_IGNORE);
    delete [] char_buffer;

    MPI_File_close(&file_out);
  } else {
    std::FILE *file_out;
    if (mpi_rank == 0) {
      file_out = fopen(fname, "w+");
      fprintf(file_out, "X Y Z Color\n");
    }
    hlong l_min = INT_MAX, l_max = INT_MIN, min_points, max_points;
    for (int i = 0; i < npart; ++i) {
      hlong id = particles.extra[i].id;
      if (l_min > id) {
        l_min = id;
      }
      if (l_max < id) {
        l_max = id;
      }
    }
    MPI_Allreduce(&l_min, &min_points, 1, MPI_HLONG, MPI_MIN, mpi_comm);
    MPI_Allreduce(&l_max, &max_points, 1, MPI_HLONG, MPI_MAX, mpi_comm);

    hlong n_active = max_points - min_points + 1;
    dlong npass = n_active / OUT_CHUNK_SIZE;
    if (n_active > npass*OUT_CHUNK_SIZE) ++npass;
    hlong ilast=min_points;

    for (int ipass = 0; ipass < npass; ++ipass) {
      dlong mpart = (dlong)std::min((hlong)OUT_CHUNK_SIZE, max_points - ilast+1);
      hlong i0 = ilast;
      hlong i1 = i0 + mpart;
      ilast = i1;

      memset(&x_temp[0][0], 0, mpart*4*sizeof(dfloat));
      for (int ii=0; ii < npart; ++ii) {
        hlong id = particles.extra[ii].id;
        if (i0 <= id && id < i1) {
          int i = id-i0;
          for (int j = 0; j < 3; ++j) {
            x_temp[i][j] = particles.x[j][ii];       // coordinates
          }
          x_temp[i][3] = particles.extra[ii].color;  // color
        }
      }

      MPI_Reduce(&x_temp[0][0], &x_root[0][0], mpart*4, MPI_DFLOAT, MPI_SUM, 0, mpi_comm);
      if (mpi_rank == 0) {
        for (int i = 0; i < mpart; ++i) {
          fprintf(file_out, " %17.9e %17.9e %17.9e %17.9e\n",
                  x_root[i][0], x_root[i][2], x_root[i][3], x_root[i][4]);
        }
      }
    }
    if (mpi_rank == 0) fclose(file_out);
  }
}

void particleUpdate(historyParticles_t& particles, nrs_t* nrs, int tstep){

  particles.find();
  particles.migrate();

  dfloat *u1[3];
  u1[0] = new dfloat[3*particles.size()];
  u1[1] = u1[0] + particles.size();
  u1[2] = u1[1] + particles.size();
  occa::memory o_U = nrs->o_U.cast(occa::dtype::get<dfloat>());
  particles.interpLocal(o_U, u1, 3);

  double c1, c2, c3, dt = nrs->dt[0];
  if (tstep == 0) { // AB1
    c1 = 1.0;
    c2 = 0.0;
    c3 = 0.0;
  } else if (tstep == 1) { // AB2
    c1 =  3.0 / 2.0;
    c2 = -1.0 / 2.0;
    c3 =  0.0 / 2.0;
  } else { // AB3
    c1 =  23.0 / 12.0;
    c2 = -16.0 / 12.0;
    c3 =   5.0 / 12.0;
  }

  for (int i = 0; i < particles.size(); ++i) {
    // Update particle position and velocity history
    for (int k=0; k < 3; ++k) {
       particles.x[k][i] += dt*(c1*u1[k][i] + c2*particles.extra[i].v_hist[0][k] + c3*particles.extra[i].v_hist[1][k]);
       particles.extra[i].v_hist[1][k] = particles.extra[i].v_hist[0][k];
       particles.extra[i].v_hist[0][k] = u1[k][i];
    }
  }

  delete[] u1[0];
}