module load cray-mpich
module load rocm
module load PrgEnv-gnu
module load craype-accel-amd-gfx90a

module load openblas # for hipBone

module unload cray-libsci

export MPICH_GPU_SUPPORT_ENABLED=1

## These must be set before compiling so the executable picks up GTL
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"
