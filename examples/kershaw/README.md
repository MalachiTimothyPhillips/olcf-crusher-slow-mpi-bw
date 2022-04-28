# Kershaw BP5 and BPS5

## Performance Results

### NVIDIA V100
```
BPS5
solve time: 0.521428s
  preconditioner 0.402965s
    smoother 0.251669s
    coarse grid 0.103442s
iterations: 31
throughput: 2.17135e+08 (DOF x iter)/s
throughput: 7.00436e+06 DOF/s
FLOPS/s: 4.90859e+11

BP5
solve time: 1.47599s
throughput: 2.47446e+09 (DOF x iter)/s
FLOPS/s: 4.69537e+11
```

### NVIDIA A100
```
BPS5
solve time: 0.296066s
  preconditioner 0.224607s
    smoother 0.154094s
    coarse grid 0.0426403s
iterations: 31
throughput: 3.82415e+08 (DOF x iter)/s
throughput: 1.2336e+07 DOF/s
FLOPS/s: 8.64493e+11

BP5
solve time: 0.895191s
throughput: 4.07987e+09 (DOF x iter)/s
FLOPS/s: 7.7417e+11
```

### NVIDIA A100, 2 GPU
```
BPS5
..................................................
repetitions: 50
solve time: min: 0.199604s  avg: 0.201367s  max: 0.261248s
  preconditioner 0.161264s
    smoother 0.112551s
    coarse grid 0.0285382s
iterations: 31
throughput: 5.67224e+08 (DOF x iter)/s
throughput: 1.82976e+07 DOF/s
FLOPS/s: 1.28228e+12

BP5
done (1.538e-05s)
.........................
repetitions: 25
solve time: min: 0.513648s  avg: 0.514179s  max: 0.516034s
iterations: 1000
throughput: 7.11045e+09 (DOF x iter)/s
FLOPS/s: 1.34923e+12
```

### AMD MI100
```
BPS5
solve time: 0.508355s
  preconditioner 0.389843s
    smoother 0.28957s
    coarse grid 0.051806s
iterations: 31
throughput: 2.22719e+08 (DOF x iter)/s
throughput: 7.18447e+06 DOF/s
FLOPS/s: 5.03481e+11

BP5
solve time: 1.41609s
throughput: 2.57912e+09 (DOF x iter)/s
FLOPS/s: 4.89396e+11
```

### AMD MI250X/1
```
BPS5
solve time: 0.440587s
  preconditioner 0.335607s
    smoother 0.242305s
    coarse grid 0.0527129s
iterations: 31
throughput: 2.56976e+08 (DOF x iter)/s
throughput: 8.28954e+06 DOF/s
FLOPS/s: 5.80923e+11

BP5
solve time: 1.17002s
throughput: 3.12153e+09 (DOF x iter)/s
FLOPS/s: 5.92321e+11
```

### AMD MI250X
```
BPS5
..................................................
repetitions: 50
solve time: min: 0.550918s  avg: 0.553299s  max: 0.576202s
  preconditioner 0.44601s
    smoother 0.333078s
    coarse grid 0.0597616s
iterations: 31
throughput: 2.05512e+08 (DOF x iter)/s
throughput: 6.62941e+06 DOF/s
FLOPS/s: 4.64583e+11

BP5
done (4.158e-06s)
.........................
repetitions: 25
solve time: min: 1.35397s  avg: 1.35479s  max: 1.35617s
iterations: 1000
throughput: 2.69744e+09 (DOF x iter)/s
FLOPS/s: 5.11849e+11
```