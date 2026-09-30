# XDL efficiency preparation record

This file retains the comparison data and collection details while the XDL
efficiency slide is deferred. Any future presentation of these results should
use only XDL efficiency percentages and percentage-point changes. AM/package
details, raw counters, and validation artifacts stay in preparation records.

## Controlled comparison

Collection completed on 30 September 2026 from branch `gfx1250-preso`, revision
`9f48680b75`, with documentation changes in progress. Kernel/compiler sources
are unchanged from the slide code reference `b266fe4c1d`.

- Workload: `G=2, M=4096, N=1024, K=2048`, with the same M in both groups.
- Tile: `256×256×128`, depth 2, `GROUP_M=4`, no L2 prefetch.
- Launch: 32 physical workgroups, four Wave32 waves per workgroup.
- Each group has 64 output tiles: two tiles per program per group. This
  exercises within-group prefetch and a group transition in every variant.
- FP16 inputs/output, FP32 accumulation; CPU input generation uses seed 123.
- All variants use the same tensor allocation order, addresses, inputs,
  compiler flags, and counter settings.

| Variant | Cross-tile prefetch | Program remapping | Cluster size | Multicast |
| :--- | :--- | :--- | :---: | :---: |
| Alias-C | Off | None | 1 | Off |
| Hybrid | On | None | 1 | Off |
| Hybrid + remap | On | Chunked, 8 logical XCDs, chunk 2 | 1 | Off |
| + cluster, multicast off | On | Chunked, 8 logical XCDs, chunk 2 | 4 | Off |
| + multicast | On | Chunked, 8 logical XCDs, chunk 2 | 4 | On |

The two cluster rows retain identical launch and synchronization settings;
only multicast changes. The first transition changes both next-tile prefetch
and C staging, as implemented by the hybrid option.

This diagnostic workload replaces the deferred slide's original, unfilled
`G=16, M=N=K=4096` proposal. The two shapes must not be presented as the same
measurement. The assembly excerpts retain their separately recorded build
configuration in [assembly.md](assembly.md).

## Results

| Variant | Steady XDL efficiency | Delta vs. alias-C | Samples in steady window |
| :--- | ---: | ---: | ---: |
| Alias-C | 74.661414% | — | 351 |
| Hybrid | 68.436292% | −6.225121 pp | 383 |
| Hybrid + remap | 68.781250% | −5.880164 pp | 381 |
| + cluster, multicast off | 66.689007% | −7.972407 pp | 393 |
| + multicast | 68.804720% | −5.856693 pp | 380 |

All five runs passed correctness for all 8,388,608 output elements and produced
bit-identical outputs. The multicast binary was also byte-identical between
the AM run and a separate successful FFM correctness run. The official active-unit
and all-unit steady metrics are identical in each case: all 128 SIMD streams
participated. An independent reconstruction from the raw counter window matched
the official statistic within 1e-12. Each generated HTML report's steady-state
field was also checked against the extracted statistic within 1e-12.

Alias-C has the highest steady-state efficiency on this diagnostic shape.
Within the clustered configuration, multicast improves efficiency by
2.115714 percentage points. This comparison does not establish an efficiency
gain for the hybrid over alias-C, or the ordering on larger workloads.

Resource metadata for these exact binaries:

| Variant | LDS bytes | VGPRs | SGPRs | Scratch bytes |
| :--- | ---: | ---: | ---: | ---: |
| Alias-C | 286672 | 815 | 104 | 0 |
| Hybrid | 320448 | 782 | 100 | 0 |
| Hybrid + remap | 320448 | 782 | 100 | 0 |
| + cluster, multicast off | 320448 | 886 | 102 | 0 |
| + multicast | 320448 | 886 | 104 | 0 |

## Metric and normalization

Use the official r8.05 perftools report field **XDL Steady efficiency (Include
all CU)**, computed as `100 × xdl_steady_state_all_cu` by
`perftools_core.util.stat_util.get_stats_miperf`.

The implementation selects all per-SIMD `wmma_ops` counter columns, locates the
first and last rows with matrix activity, excludes those two edge rows, and
averages the remaining rows and all columns divided by the dump period.
The period is 500 clocks. This is the official steady window, not a selected
hot interval or the maximum CSV row. Recurring stores and tile/group
transitions remain inside it. It is also different from an isolated K-loop
measurement and from whole-kernel efficiency.

The model exposes 32 workgroup processors, represented by 128 SIMD counter
streams. The package labels these fields using “CU”; the slide uses the
ISA terminology WGP for the four-SIMD workgroup placement boundary. The
active-unit and all-unit metrics are collected separately so coverage can
be checked. Presentation deltas are computed from unrounded percentages
before rounding for display.

The package configuration models one XCD. The source's chunked remapping
still changes the assignment of logical tiles to physical programs, but this
collection does not evaluate placement across eight physical XCDs. Interpret
the remapping row within this workload and modeled capacity.

## Environment and artifacts

- Active package: `rocdtif-7.15-am+ffmlite-mi400-r8.05`.
- `AM_CLOCK_MT=0`.
- `TRITON_AMD_LLVM_FLAGS=amdgpu-loop-carried-load-percent=0`.
- `TRITON_DISABLE_LINE_INFO=1`; separate JIT cache per run.
- Triton: `build/lib.linux-x86_64-cpython-311`, with
  `LLVM_SYSPATH=/home/ibutygin/llvm/llvm-build`.
- `libtriton.so` SHA-256:
  `f4ff2d1c6dfedee6f674135314b5067cb7e9e44f1c42193a9debc76e3c31ba8f`.
- Perftools wheel: `perftools_core-1.2.5+608e506.112-py3-none-any.whl`;
  SHA-256 `f9d41c039d4d6898bf361bc5f69c842ad9e5bcf5ed89ed3399eed86d796597c6`.
- Raw run root:
  `/home/ibutygin/tlx/am-runs/gfx1250_slides_steady_20260930/`.

The existing analysis dependency environment is reused with modules extracted
from the r8.05 wheel. `stat_util.py` was verified byte-for-byte against that
wheel before collection; no packages were installed globally.

Each run records its source/library hashes, compiler settings, input/output
hashes, addresses, generated assembly and code object, compiler resource
metadata, complete counters, and correctness result. Every output is checked
against CPU FP32 `torch.matmul`, converted to FP16, using `atol=rtol=1e-2`.
There is exactly one target kernel launch per run. Input generation and the
reference multiplication execute on the CPU.

## Reproduction

[collect-efficiency.py](scripts/collect-efficiency.py) runs one variant after
the package's AM or FFM environment has been sourced in a fresh shell. Run
it in a dedicated scratch directory. It applies the parent guide's ROCm
library shim before importing PyTorch and imports the kernel from this
checkout explicitly.

The local `run.sh` under the raw run root supplies the recorded environment:

```bash
bash /home/ibutygin/tlx/am-runs/gfx1250_slides_steady_20260930/run.sh am alias
```

Use `hybrid`, `remap`, `cluster`, and `multicast` for the other rows. The launcher
refuses to reuse an existing run directory. For a new collection, use a fresh
run root with the same settings.

The raw root's `summarize.py` extracts official report fields, independently
checks the steady-window arithmetic, and validates matching workload, input,
address, compiler, and normalization settings before writing `summary.json`.
That summary retains full-precision values, sample ranges, source/binary/counter
hashes, and validation records for every row. The independent counter check
uses the same first/last activity boundaries and all 128 columns.

`report.sh` generates the active package's plots and HTML report for each run.
The reports are stored at `runs/<variant>-am/report/report.html`, alongside
the corresponding `official_stats.json`, `result.json`, and raw counter files.
The reports include the matrix-activity plot. The package's auxiliary draw-log
parser reports an unsupported format; the XDL summary comes from the completed
counter files and was independently verified. One target dispatch per run was
checked directly from `msg.log`.
