# Public hipBLASLt gfx1250 A8W8 reproduction with FP32 output

This benchmark builds and launches the public hipBLASLt A8W8 assembly with an
FP32 output epilogue. It preserves the input packing, compute loop, register
allocation, and input pipeline of the published kernel. Both inputs use FP8
E4M3 with E8M0 scales; accumulation and output are FP32, with alpha=1 and beta=0.

The source is pinned to ROCm/rocm-libraries commit
[`74186f70c3861b4b17cbad886def36fdaf106374`](https://github.com/ROCm/rocm-libraries/tree/74186f70c3861b4b17cbad886def36fdaf106374/projects/hipblaslt/tensilelite)
on `hipblaslt-gfx1250-custom-kernels`. See the upstream
[custom-kernel notes](https://github.com/ROCm/rocm-libraries/blob/74186f70c3861b4b17cbad886def36fdaf106374/projects/hipblaslt/tensilelite/README_custom_mxfp8_compute_memory_gemm.md).
The `async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep` assembly
for 8192x8192x8192 and 8192x8192x4096 is byte-identical. This reproducer covers
that A8W8 source. The separate A8W4 and `Permute_interleave_HiLnMX_halfBufStore`
code-object bundles are outside its scope.

## Run on gfx1250

Use a ROCm PyTorch/Triton environment, a HIP runtime with cluster launch support,
and LLVM with gfx1250 support. The default assembler/linker are under
`LLVM_SYSPATH`, falling back to `~/llvm/llvm-build`. Override with `--llvm PATH`.

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/hipblaslt_repro/bench.py \
  -M 8192 -N 8192 -K 8192

gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/hipblaslt_repro/bench.py \
  -M 8192 -N 8192 -K 4096
```

The first run downloads the pinned assembly and verifies its SHA256. Generated
assembly, object, HSACO, and build metadata are cached under
`~/.cache/tlx-hipblaslt-mxfp-f32`; `--cache-dir PATH` selects another directory.
The builder can also run separately without a GPU:

```bash
python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/hipblaslt_repro/build.py
```

The benchmark validates against a CPU FP32 reference before timing. The default
eager timing budget is 256 ms (`--benchmark-ms 256`), using Triton's `do_bench`
on the same stream as the HIP launches. Packing, transfers, and reference
computation are outside the timed region. `--output-dir PATH` saves the exact
source and HSACO, launch/input provenance, and correctness/timing JSON.

`--output staged` is the default FP32 epilogue. Use `--output direct` to compare
direct `global_store_b128` from the native accumulator layout. Both store FP32.

## Kernel and output layout

| Property | Value |
| --- | --- |
| Tile | 256x256x256 |
| Workgroup | Four wave32 waves, 128 threads |
| Cluster | 4x4 workgroups |
| Physical launch grid | `(M/512, N/512, 1)` |
| Output tiles per workgroup | Four, across the M/2 and N/2 quadrants |
| VGPRs / SGPRs | 1024 / 106 |
| LDS / private scratch | 287744 bytes / 0 bytes |
| Output | FP32, column-major `(M,N)`, strides `(1,M)` |

A and B are stored as contiguous `(M,K)` and `(N,K)` arrays. Each has one
E8M0 scale per 32 K elements. The public kernel's scale packing is
`[K/128][M or N][4]`, which differs from the TLX tutorial's packing. The harness
packs the scales and uses the kernel's universal v2 argument ABI directly.

The FP32 staged epilogue writes four 256x64 panels. It stores the native
accumulators with `ds_store_b128` and copies contiguous output with
`global_store_async_from_lds_b128`. Two 69632-byte LDS slots, with a 1088-byte
column pitch, fit in the original allocation. A partial async wait and a
workgroup barrier protect slot reuse. The existing next-tile and final full
async waits are retained.

For an odd number of BK256 steps, the input ring changes phase between output
tiles. The builder selects an epilogue that follows the free LDS half, preserving
the next tile's prefetched input. The requested K=4096 and K=8192 shapes use a
fixed output half and the same binary.

The final input-loop cluster wait is consumed before the output workgroup
barriers. This ordering completes validation in both FFM and AM; leaving that
cluster wait pending across the added output barriers stalled in AM. The
adaptation preserves the rendezvous count and local memory lifetime waits.

## Model checks and measurement scope

`--benchmark-mode none` performs one dispatch and correctness check. In an
environment configured for FFM or AM, for example:

```bash
python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/hipblaslt_repro/bench.py \
  -M 2048 -N 2048 -K 2048 --benchmark-mode none --output-dir /tmp/hipblaslt-f32
```

M and N must be positive multiples of 2048; K must be a multiple of 256 and at
least 768. `--grid-x` and `--grid-y` bound the physical workgroup grid for model
runs; both dimensions must remain multiples of four. For example, this checks
64 output tiles across all four quadrants using full 8192-sized strides:

```bash
python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/hipblaslt_repro/bench.py \
  -M 8192 -N 8192 -K 8192 --grid-x 4 --grid-y 4 \
  --benchmark-mode none --output-dir /tmp/hipblaslt-f32-bounded
```

The harness checks every visited output and verifies that the unvisited region
remains NaN. Bounded launches are not complete 8192x8192 output validation.
Default hardware launches cover the complete output.

Validation with r8.05 FFM/AM and `~/llvm/llvm-build`:

| Check | Output coverage | Maximum absolute error |
| --- | --- | ---: |
| FFM, 2048x2048x2048, signed inputs | Complete | 0.00006104 |
| FFM, 8192x8192x4096, signed inputs | 4194304 of 67108864 elements | 0.00012207 |
| FFM, 8192x8192x8192, signed inputs | 4194304 of 67108864 elements | 0.00024414 |
| FFM, 2048x4096x768, signed inputs | Complete | 0.00005341 |
| FFM, 2048x2048x1280, signed inputs | Complete | 0.00004578 |
| AM, 2048x2048x2048, tutorial input distribution | Complete | 0.00003052 |

All checks used nonuniform scales and the fixed tolerance `atol=2e-3,
rtol=1e-4`. The default input distribution is signed normal values scaled by
0.5 and quantized to E4M3. `--input-mode tutorial` uses positive E4M3 byte codes
20 through 39 for the matched AM comparison. The same staged HSACO and input
bytes also passed FFM; bitwise equality between FFM and AM is not required by
the numerical check.

On the complete 2048-cubed problem with 16 workgroups and four tiles per
workgroup, the following AM results were observed:

| FP32 kernel | Dispatch cycles | Official steady XDL, active CUs | Sampled final compute tile XDL |
| --- | ---: | ---: | ---: |
| Public compute + direct output | 191061 | 21.28% | 95.96% |
| Public compute + staged output | 121734 | 31.94% | 91.99% |
| Current TLX persistent A8W8 | 108596 | 36.16% | 60.83% |

The TLX control at revision `52d3e7c4bd541faeddd5fa6462d46723c5ee5130` uses
256x256x128 tiles, three buffers, group-M 8, partial fusion, four-workgroup
multicast with barrier interval 4, output staging, and cross-tile prefetch.
It uses 16 persistent programs to cover the same complete 2048-cubed problem.

The sampled compute percentages count actual XDL service clocks between the
first and last WMMA service in the final tile on one SIMD. Each has 8192 matrix
service clocks; the intervals are 8537, 8905, and 13466 clocks respectively.
These percentages are distinct from AM's official steady XDL statistic.
Only 16 of the model's 32 CUs are active. Startup, output, and tile handoff
account for the different dispatch ranking; these results do not establish
hardware throughput at the full 8192 shapes.
