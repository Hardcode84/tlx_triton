# gfx1250 MXFP GEMM

This directory contains the FP8/FP4 TDM-pipelined GEMM tutorial and a persistent
A8W8 benchmark sweep.

- `amd_mxfp_gemm_tdm_pipelined.py`: kernels, config-based `matmul` API,
  `mxgemm_tdm_pipelined` API, and single-shape benchmark CLI.
- `bench.py`: multi-shape benchmark runner with separate processes and CSV output.

The benchmark defaults to FP8 E4M3 activations and weights, FP32 output,
256x256x256 tiles, two buffers, partial TDM fusion, persistent scheduling, and
cross-tile prefetch. It runs M=N=8192 with K=8192 and K=4096.

Run from the repository root in an environment configured for gfx1250:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv a8w8.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -M 8192 -N 8192 -K 4096
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

For one simulator dispatch per shape, use `--benchmark-mode none --output-dir
<directory>` to keep each case's artifacts in a fresh subdirectory. This mode
leaves timing fields blank. Both scripts provide `--help` for configuration
options.

Compile checks live in `python/test/unit/language/test_tlx_codegen.py`; gfx1250
runtime checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`.
