This benchmark applies the subcycling operator.

# Usage

```
Usage: ./nekrs-subcycling --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>
                    [--ext-order <n>] [--c-order<n>] [--iterations <n>]
```

# Examples

### Nvidia V100 
```
>mpirun -np 1 nekrs-subcycling --p-order 7 --elements 4096 --backend CUDA

MPItasks=1 OMPthreads=12 NRepetitions=8199 N=7 cubN=10 Nelements=4096 elapsed time=0.00116622 wordSize=64 GDOF/s=1.20468 GB/s=54.6604 GFLOPS/s=2038.45
```

```
>mpirun -np 1 nekrs-subcycling --p-order 7 --c-order 7 --elements 4096 --backend CUDA

MPItasks=1 OMPthreads=2 NRepetitions=21207 N=7 cubN=7 Nelements=4096 elapsed time=0.000460586 wordSize=64 GDOF/s=3.0503 GB/s=72.8517 GFLOPS/s=846.899
```

### AMD EPYC 7402 24-Core Processor
```
>OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 24 --bind-to core --map-by ppr:24:socket nekrs-subcycling --p-order 7 --elements 4096 --backend CPU

MPItasks=24 OMPthreads=1 NRepetitions=646 N=7 cubN=10 Nelements=4080 elapsed time=0.0156808 wordSize=64 GDOF/s=0.0892453 GB/s=4.04966 GFLOPS/s=151.012
```

```
>OCCA_CXXFLAGS='-O3 -march=native -mtune=native' mpirun -np 24 --bind-to core --map-by ppr:24:socket nekrs-subcycling --p-order 7 --c-order 7 --elements 4096 --backend CPU

MPItasks=24 OMPthreads=1 NRepetitions=1823 N=7 cubN=7 Nelements=4080 elapsed time=0.00546482 wordSize=64 GDOF/s=0.256081 GB/s=6.11637 GFLOPS/s=71.0995
```
