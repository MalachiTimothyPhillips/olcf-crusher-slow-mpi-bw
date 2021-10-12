This benchmark applies the fast diagonalization method (FDM)
```
Su = (S_x \cross S_y \cross S_z) \Lambda^{-1} (S_x^T \cross S_y^T \cross S_z^T)u
```

# Usage

```
Usage: ./nekrs-fdm --p-order <n> --elements <n> --backend <CPU|CUDA|HIP|OPENCL>
                    [--fp32] [--iterations <n>]
```

Note here that n refers to the polynomial order of the unextended domain.
Hene, the true FDM problem dimensions are (n+3)^3.

# Examples

### Nvidia V100
```
>nekrs-fdm --p-order 7 --elements 4000 --backend CUDA --fp32 --iterations 100

MPItasks=1 OMPthreads=12 NRepetitions=100 N=7 Nelements=4000 elapsed time=0.000159026 wordSize=32 GDOF/s=18.3366 GB/s=83.0051 GFLOPS/s=3043.52

>nekrs-fdm --p-order 7 --elements 4000 --backend CUDA --iterations 100

MPItasks=1 OMPthreads=12 NRepetitions=100 N=7 Nelements=4000 elapsed time=0.000321472 wordSize=64 GDOF/s=9.07078 GB/s=41.0612 GFLOPS/s=1505.58
```

### Intel i7-7800X CPU @ 3.50GHz
```
>nekrs-fdm --p-order 7 --elements 4000 --backend CPU --fp32 --iterations 100

MPItasks=1 OMPthreads=12 NRepetitions=100 N=7 Nelements=4000 elapsed time=0.0264158 wordSize=32 GDOF/s=0.110389 GB/s=0.499701 GFLOPS/s=18.3224

>nekrs-fdm --p-order 7 --elements 4000 --backend CPU --iterations 100

MPItasks=1 OMPthreads=12 NRepetitions=100 N=7 Nelements=4000 elapsed time=0.0265916 wordSize=64 GDOF/s=0.109659 GB/s=0.496398 GFLOPS/s=18.2013
```