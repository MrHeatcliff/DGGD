# PACS concurrency benchmark

One optimizer update consumes a minibatch from each of three source domains.
Each process warms up for 25 updates, synchronizes at a barrier, then measures 250 updates plus one full source validation and checkpoint write.
Native FP32 algorithms, 1 Torch thread and 1 DataLoader worker per source; H200, 132 SMs, approximately 140 GiB VRAM.
Warmup/penalty counters start past annealing thresholds for representative steady-state compute.
132 Hopper SMs × 128 FP32 lanes/SM correspond to 16,896 FP32 CUDA cores; this arithmetic is not a job-count limit.

| Algorithm | MPS | Jobs/GPU | Aggregate updates/s | Peak device MiB | Success |
| --- | --- | ---: | ---: | ---: | --- |
| ERMPlusPlus | no_mps | 1 | 9.54 | 11305.0 | True |
| ERMPlusPlus | no_mps | 2 | 12.25 | 22607.0 | True |
| ERMPlusPlus | no_mps | 4 | 12.79 | 45201.0 | True |
| ERMPlusPlus | no_mps | 8 | 12.92 | 90399.0 | True |
| IGA | no_mps | 1 | 1.90 | 20977.0 | True |
| IGA | no_mps | 2 | 1.80 | 41951.0 | True |
| IGA | no_mps | 4 | 0.00 | 77319.0 | SKIPPED |
| RDM | no_mps | 1 | 3.59 | 27587.0 | True |
| RDM | no_mps | 2 | 6.39 | 55171.0 | True |
| RDM | no_mps | 4 | 7.19 | 110339.0 | True |
| ERMPlusPlus | mps | 1 | 9.26 | 11365.0 | True |
| ERMPlusPlus | mps | 2 | 15.53 | 22662.0 | True |
| ERMPlusPlus | mps | 4 | 17.36 | 45255.0 | True |
| ERMPlusPlus | mps | 8 | 17.98 | 90427.0 | True |
| IGA | mps | 1 | 1.86 | 21038.0 | True |
| IGA | mps | 2 | 2.61 | 42007.0 | True |
| IGA | mps | 4 | 3.05 | 83944.0 | True |
| RDM | mps | 1 | 3.53 | 27648.0 | True |
| RDM | mps | 2 | 6.94 | 55227.0 | True |
| RDM | mps | 4 | 9.15 | 110386.0 | True |

Choose the smallest MPS concurrency within 5% of measured peak throughput.
These are short representative measurements, not a guarantee for every algorithm or full training run.

References: [NVIDIA MPS](https://docs.nvidia.com/deploy/mps/latest/index.html),
[NVIDIA Hopper SM architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/),
[NVIDIA H200 architecture overview](https://developer-blogs.nvidia.com/wp-content/uploads/2024/08/CUDA-Programming-and-Optimization.pdf).
