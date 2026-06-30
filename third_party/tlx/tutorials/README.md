# TLX MXFP GEMM Benchmark

Local wrapper:

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

Local wrapper:

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
