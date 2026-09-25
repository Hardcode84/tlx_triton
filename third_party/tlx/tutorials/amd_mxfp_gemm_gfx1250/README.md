# gfx1250 MXFP GEMM

This directory contains the FP8/FP4 TDM-pipelined GEMM tutorial and a persistent
MXFP8 x MXFP8 / MXFP8 x MXFP4 benchmark sweep.

- `amd_mxfp_gemm_tdm_pipelined.py`: kernels, config-based `matmul` API,
  `mxgemm_tdm_pipelined` API, and single-shape benchmark CLI.
- `bench.py`: multi-shape benchmark runner with separate processes and CSV output.

The benchmark runs both variants at M=N=8192 with K=8192 and K=4096, for four
runs by default:

| Variant | Activations | Weights |
| --- | --- | --- |
| `mx8xmx8` | FP8 E4M3 | FP8 E4M3 |
| `mx8xmx4` | FP8 E4M3 | FP4 E2M1 |

Both use E8M0 scales, FP32 output, 256x256x256 tiles, two buffers, partial TDM
fusion, persistent scheduling, and cross-tile prefetch. The summary and CSV
identify each variant and its operand dtypes.

Run from the repository root in an environment configured for gfx1250:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -M 8192 -N 8192 -K 4096
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

Use repeatable `--variant` options to select variants. `--dtype-a float8_e5m2`
changes activations for the selected variants. Alternatively, `--dtype-b`
selects a single weight dtype (`float8_e4m3`, `float8_e5m2`, or `float4`) and
cannot be combined with `--variant`. Three buffers with 256x256 tiles are
supported for `mx8xmx4`; `mx8xmx8` exceeds LDS capacity with that configuration.

For one simulator dispatch per shape/variant, use `--benchmark-mode none
--output-dir <directory>` to keep each run's artifacts in a fresh subdirectory
whose name includes the variant. This mode leaves timing fields blank. Both
scripts provide `--help` for configuration options.

Compile checks live in `python/test/unit/language/test_tlx_codegen.py`; gfx1250
runtime checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`.
