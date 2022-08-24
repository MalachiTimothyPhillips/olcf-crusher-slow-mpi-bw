# Instructions to replicate perceived "slow" MPI communication

## Cloning Directory
```
git clone --depth=1 git@github.com:MalachiTimothyPhillips/olcf-crusher-slow-mpi-bw.git
cd olcf-crusher-slow-mpi-bw
```

## Setting modules

Modules/env variables have already been set in `set-crusher-env.sh`, which needs to be sourced:
```
source set-crusher-env.sh
```

Please alter things like the account/installation directory as needed.

## Configuration
Run configuration script. Necessary environment variables have already been altered above.
```
./nrsconfig
```

## Compilation

This doesn't take too long, maybe 5-10 minutes?
```
cd build && make install -j 12
```

## Setting up a case

Navigate to GPFS to set up a case, e.g.:
```
cd $MEMBERWORK/csc262
```

Copy `kershaw` example from repo:
```
cp -r path/to/olcf-crusher-slow-mpi-bw/examples/kershaw .
```

*CAUTIONARY NOTE*: The full path to the case, e.g.:
`/gpfs/alpine/scratch/malachi/csc262/kershaw/kershaw.re2`
must be less than or equal to 132 characters!
This is due to a nek5000 limitation that's currently used to read the mesh files.
I know it's a pain...

Navigate to dir:
```
cd kershaw
```

Copy job configuration script
```
cp path/to/olcf-crusher-slow-mpi-bw/scripts/nrsqsub_crusher
```

Configure job to run on a single Crusher node for 00:30:00 minutes:
```
./nrsqsub_crusher kershaw 1 00:30:00
```

As output, you should see an s.bin file, as below:
```
#!/bin/bash
#SBATCH -A CSC262
#SBATCH -J nekRS_kershaw
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=8
module load PrgEnv-gnu
module load craype-accel-amd-gfx90a
module load cray-mpich
module load rocm
module unload cray-libsci
module list
rocm-smi
rocm-smi --showpids
# which nodes am I running on?
squeue -u $USER
export MPICH_GPU_SUPPORT_ENABLED=1
export PE_MPICH_GTL_DIR_amd_gfx90a="-L/opt/cray/pe/mpich/8.1.16/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"
ulimit -s unlimited
export NEKRS_HOME=/ccs/home/<your-account-name>/.local/nekrs-crusher-slow-mpi
export NEKRS_GPU_MPI=1
export NVME_HOME=/mnt/bb/<your-account-name>/
export ROMIO_HINTS=<path-to-the-current-dir>/.romio_hint
# actual run
date
srun -n 8 /ccs/home/<your-account-name>/.local/nekrs-crusher-slow-mpi/bin/nekrs --backend HIP --device-id 0 --setup kershaw
```

You may also want to manually add in, e.g., prior to the `srun` command:
```
export MPICH_OFI_NIC_POLICY=NUMA
```

To launch the job, simply `sbatch s.bin`.

After the job finishes, you should see output like the following at the bottom:

```
occa memory usage: 6.40133 GB
initialization took 41.1226 s
Benchmarking d->d gather-scatter handle.

used config: pw+device (MPI min/max/avg: 4.15e-05s 4.68e-05s 4.39e-05s / avg bi-bw: 10.9GB/s/rank)
Benchmarking h->h gather-scatter handle.

used config: pw+hybrid (MPI min/max/avg: 2.33e-05s 2.59e-05s 2.47e-05s / avg bi-bw: 19.5GB/s/rank)
Benchmarking d->d gather-scatter handle, using direct all-to-all exchange.

running benchmarks

BPS5
..................................................
repetitions: 50
solve time: min: 0.552199s  avg: 0.553751s  max: 0.592319s
iterations: 38
throughput: 1.88831e+08 (DOF x iter)/s/rank
throughput: 4.96922e+06 DOF/s/rank
flops/rank: 4.01044e+11

BP5
done (5.5e-06s)
iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.iteration limit of pressure reached!
.
repetitions: 25
solve time: min: 0.485137s  avg: 0.485999s  max: 0.487046s
iterations: 500
throughput: 2.82807e+09 (DOF x iter)/s/rank
flops/rank: 5.37142e+11
End
```

The bijection bandwidths reported are for doing the point-to-point MPI communication with GPU buffers or host buffers.
To query the message sizes, scroll up in the output to:

```
 setup mesh topology
   Right-handed check complete for       64000 elements. OK.
gs_setup: 236041 unique labels shared
   pairwise times (avg, min, max): 8.91632e-05 8.67524e-05 9.06878e-05
   crystal router                : 0.000370038 0.000369652 0.000370557
   used all_to_all method: pairwise
   handle bytes (avg, min, max): 2.75687e+07 27568660 27568660
   buffer bytes (avg, min, max): 961072 961072 961072
   setupds time 5.4103E-01 seconds   0  8     8364041       64000

 nElements   max/min/bal: 8000 8000 1.00
 nMessages   max/min/avg: 7 7 7.00
 msgSize     max/min/avg: 19881 1 8581.00
 msgSizeSum  max/min/avg: 60067 60067 60067.00
```

This provides information about the number of messages required, and the message size required. On the OLCF office hour, I was mistaken about how many words needed to be exchanged.
The message size will be as large as ~20,000 words, in either double or single precision. The timings reported in the d->d gather-scatter and h->h gather scatter handle are for double precision words.

Example output from a real job is found in `nekRS_kershaw-166189.out`.
