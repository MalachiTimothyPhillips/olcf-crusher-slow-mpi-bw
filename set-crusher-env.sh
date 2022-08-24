module load PrgEnv-gnu
module load craype-accel-amd-gfx90a
module load cray-mpich
module load rocm
module unload cray-libsci
module load cmake
module list

export MPICH_GPU_SUPPORT_ENABLED=1

## These must be set before compiling so the executable picks up GTL
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

# NEKRS specific stuff

#export NEKRS_INSTALL_DIR=${HOME}/.local/nekrs-crusher-next-nbc-device-experiment
export NEKRS_INSTALL_DIR=${HOME}/.local/nekrs-crusher-olcf-mpi-slowdown
export NEKRS_HOME=$NEKRS_INSTALL_DIR

export NEKRS_CC="cc"
export NEKRS_CXX="CC"
export NEKRS_FC="ftn"

# acct stuff
export PROJ_ID=CSC262
export QUEUE=batch
