# TLX MXFP GEMM Benchmark

Wrapper:

```bash
third_party/tlx/tutorials/amd-mxfp-gemm-tdm-pipelined_bench.py
```

Run from the repo root. Prefer the built package first in `PYTHONPATH` so the
benchmark uses the freshly built `libtriton.so`:

```bash
PYTHONPATH=build/lib.linux-x86_64-cpython-311:python \
python third_party/tlx/tutorials/amd-mxfp-gemm-tdm-pipelined_bench.py \
  -M 2048 -N 1024 -K 8192 \
  -BM 256 -BN 256 -BK 256 \
  --num-warps 4 --num-buffers 2 \
  --scale-preshuffled --with-a-scale \
  --schedule sliceMNK --tdm-fusion 2way \
  --l2-prefetch-distance 2 \
  --dtype-a float8_e4m3 --dtype-b float8_e4m3 \
  --benchmark-mode eager --benchmark-num-iters 32
```

Recommended starting config for gfx1250 TLX MXFP GEMM:

- Problem: `M=2048`, `N=1024`, `K=8192`
- Tile: `BM=256`, `BN=256`, `BK=256`
- Pipeline: `num_warps=4`, `num_buffers=2`
- Data: `float8_e4m3` A/B, preshuffled scales, A scale enabled
- Schedule: `sliceMNK`
- L2 prefetch: `2`
- TDM fusion: `2way` as the current baseline; also compare `4way` and `partial`

Comparison configs:

- This command uses the current TLX MXFP variant comparison shape and compares
  `4way`, `2way`, and `partial`. Use `2way` as the default starting point, then
  rerun with the other fusion modes.
- To reproduce comparisons that pin LLVM scheduling, pass
  `TRITON_AMD_LLVM_FLAGS=amdgpu-loop-carried-load-percent=0` in the environment;
  the wrapper does not set LLVM flags itself.
- For the smaller `K=4096` Gluon MXFP comparison:

  ```bash
  TRITON_AMD_LLVM_FLAGS="amdgpu-loop-carried-load-percent=0" \
  python3 third_party/amd/python/examples/gluon/mxfp_gemm_gfx1250.py \
    -M 2048 -N 1024 -K 4096 \
    -BM 256 -BN 256 -BK 256 \
    --num_warps 4 --num_buffers 2 \
    --dtype_a float8_e4m3 --dtype_b float8_e4m3 \
    --schedule sliceNK --group_size_m 8 \
    --scale_preshuffled --with_a_scale
  ```

  TLX wrapper equivalent:

  ```bash
  TRITON_AMD_LLVM_FLAGS="amdgpu-loop-carried-load-percent=0" \
  PYTHONPATH=build/lib.linux-x86_64-cpython-311:python \
  python third_party/tlx/tutorials/amd-mxfp-gemm-tdm-pipelined_bench.py \
    -M 2048 -N 1024 -K 4096 \
    -BM 256 -BN 256 -BK 256 \
    --num-warps 4 --num-buffers 2 \
    --group-size-m 8 \
    --dtype-a float8_e4m3 --dtype-b float8_e4m3 \
    --scale-preshuffled --with-a-scale \
    --schedule sliceNK --tdm-fusion none \
    --benchmark-mode eager --benchmark-num-iters 32
  ```

  A Gluon MXFP command with the same shape/tiles and no explicit `--schedule`
  uses the Gluon CLI default `sliceNK`, so it is equivalent to the explicit
  `--schedule sliceNK` command above. If a baseline schedule is intended, pass
  `--schedule baseline` explicitly.
- Gluon benchmark CLIs may have different defaults (`M=N=8192`, `K=1024`,
  `BK=128`, `schedule=sliceNK`, `l2_prefetch_distance=-1`). Pass the shape and
  knobs above explicitly when comparing TLX to Gluon.
- The current Gluon file in this checkout has the same older CLI default shape
  and does not expose the donor branch's benchmark/resolve flags unless the
  stashed Gluon port is applied.

For compile/static-profile only, use:

```bash
--benchmark-mode none
```

For a quick correctness smoke, use a smaller shape and add `--check`:

```bash
PYTHONPATH=build/lib.linux-x86_64-cpython-311:python \
python third_party/tlx/tutorials/amd-mxfp-gemm-tdm-pipelined_bench.py \
  -M 256 -N 256 -K 512 \
  -BM 256 -BN 256 -BK 256 \
  --scale-preshuffled --with-a-scale \
  --schedule sliceMNK --tdm-fusion none \
  --l2-prefetch-distance 2 \
  --dtype-a float8_e4m3 --dtype-b float8_e4m3 \
  --benchmark-mode none --check
```

To compare variants, rerun the recommended command with:

```bash
--tdm-fusion 2way
--tdm-fusion 4way
--tdm-fusion partial
```

# TLX F16 GEMM Benchmark

Wrapper for the warp-pipelined f16 benchmark:

```bash
third_party/tlx/tutorials/amd-f16-gemm-warp-pipeline_bench.py
```

Recommended starting config for the TLX AMD f16 warp-pipelined GEMM:

- Problem: `M=N=K=8192`
- Tile: `BM=256`, `BN=256`, `BK=32`
- Pipeline: `num_warps=8`, `num_buffers=2`
- Grouping: `GROUP_M=16`
- AMD codegen knobs: `matrix_instr_nonkdim=16`, `waves_per_eu=0`
- XCD remap: `num_xcds=8`, `xcd_chunk=4`

Cross-reference:

- Matches the first/default config in
  `third_party/tlx/tutorials/amd-gemm-warp-pipeline_test.py`: `256x256x32`,
  `num_buffers=2`, `num_warps=8`, `GROUP_M=16`,
  `matrix_instr_nonkdim=16`, `waves_per_eu=0`, `xcd_chunk=4`, run over
  `M=N=K=4096` and `8192`.
- The single-warp f16 comparison is a different config:
  `M=N=K=1024`, `BM=BN=256`, `BK=128`, `num_warps=4`, `num_buffers=2`,
  `transpose_b=True`, and `l2_prefetch_distance=0`. Use `--variant single` in
  this wrapper for that comparison.
- There is no directly comparable Gluon MXFP file for f16; Gluon f16 references
  use `third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py`.

Gluon f16 single-warp command:

```bash
TRITON_AMD_LLVM_FLAGS="amdgpu-loop-carried-load-percent=0" \
python3 third_party/amd/python/examples/gluon/f16_gemm_gfx1250.py \
  -M 1024 -N 1024 -K 1024 \
  --block_m 256 --block_n 256 --block_k 128 \
  --num-warps 4 --prefetch-lds --single-warp-schedule
```

TLX wrapper equivalent for the single-warp f16 config:

```bash
TRITON_AMD_LLVM_FLAGS="amdgpu-loop-carried-load-percent=0" \
PYTHONPATH=build/lib.linux-x86_64-cpython-311:python \
python third_party/tlx/tutorials/amd-f16-gemm-warp-pipeline_bench.py \
  --variant single \
  -M 1024 -N 1024 -K 1024 \
  -BM 256 -BN 256 -BK 128 \
  --num-warps 4 --num-buffers 2 \
  --l2-prefetch-distance 0 --transpose-b \
  --benchmark-mode eager --benchmark-num-iters 200
```

Copy-paste benchmark command:

```bash
PYTHONPATH=build/lib.linux-x86_64-cpython-311:python \
python third_party/tlx/tutorials/amd-f16-gemm-warp-pipeline_bench.py \
  -M 8192 -N 8192 -K 8192 \
  -BM 256 -BN 256 -BK 32 \
  --num-warps 8 --num-buffers 2 \
  --group-m 16 \
  --matrix-instr-nonkdim 16 --waves-per-eu 0 \
  --num-xcds 8 --xcd-chunk 4 \
  --benchmark-mode eager --benchmark-num-iters 200
```

For a quick correctness smoke, use:

```bash
PYTHONPATH=build/lib.linux-x86_64-cpython-311:python \
python third_party/tlx/tutorials/amd-f16-gemm-warp-pipeline_bench.py \
  -M 256 -N 256 -K 256 \
  -BM 128 -BN 128 -BK 32 \
  --num-warps 4 --num-buffers 2 \
  --group-m 4 \
  --matrix-instr-nonkdim 16 \
  --benchmark-mode none --check
```

To compare against `torch.matmul`, add:

```bash
--bench-ref
```
