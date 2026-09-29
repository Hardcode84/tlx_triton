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

Both use E8M0 scales, FP32 output, 256x256 M/N tiles, three buffers, partial TDM
fusion, persistent scheduling, and output staging. MX8xMX8 uses BK128 with
cross-tile prefetch; MX8xMX4 uses BK256 with cross-tile prefetch disabled so
output can reuse the A ring. The summary and CSV identify each variant and
its configuration.

Both variants default to 256 persistent programs (capped by tile count),
group-M 8, no XCD remapping, and four-workgroup multicast with a cluster
barrier every four input K blocks. This captures the selected hardware
configuration across both default shapes. Interval 8 was slightly faster
at K8192, while interval 4 was faster at K4096, so the benchmark uses 4
for both. The eager timing budget remains 256 ms.

Run from the repository root in an environment configured for gfx1250:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -M 8192 -N 8192 -K 4096
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

Use `--sched-mode-2` to enable the persistent kernel's hardware WMMA queuing
setting (`SCHED_MODE[2]`). It is disabled by default; `--no-sched-mode-2`
selects the default behavior. The summary and CSV record `sched_mode_2`.
For example, compare the same sweep with the setting off and on:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --no-sched-mode-2 --csv sched-off.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --sched-mode-2 --csv sched-on.csv
```

The standalone tutorial accepts `--persistent --sched_mode_2`; the Python
API and `matmul` configuration use `SCHED_MODE_2=True`.

Persistent program ordering is selectable with `--xcd-remap none|balanced|chunked`,
using `--num-xcds` (default 8) and `--xcd-chunk` (default 2). Remapping defaults
to `none`. Balanced mode assigns each logical XCD a contiguous range; chunked
mode groups small runs of tiles and leaves an incomplete final chunk unchanged.

Use `--cluster-size 2|4` with `--xcd-remap none|chunked` to multicast inputs
across independent workgroups. The benchmark defaults to cluster size 4;
use `--cluster-size 1` to disable clustering. Data and their
E8M0 scales use the same recipient masks. Without remapping, a cluster shares
B and its scales across consecutive M tiles, for either `--group-m 4` or
`--group-m 8`. With chunked remapping, a two-workgroup cluster shares B and
its scales; a four-workgroup cluster with `--group-m 4` shares both A and B
across a two-by-two output region, while `--group-m 8` shares B and its scales
across four M tiles. `--no-cluster-multicast` keeps cluster synchronization
with independent loads, for comparison.

Cluster barriers are optional for AMD multicast. Only workgroups that have
issued matching requests receive a combined load; late requests receive a
separate load after timeout. `--cluster-barrier-interval N` aligns requests
every N input K blocks, restarting at K=0 of each output tile. The benchmark
default is 4; the standalone tutorial and Python API default to 1.
Set it to 0 to remove **all** cluster barriers, including the exit
barrier. Local workgroup synchronization and TDM completion waits still
protect each workgroup's LDS. Positive intervals retain an exit barrier.

```bash
# Multicast with no cluster synchronization.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --xcd-remap none --cluster-size 4 --group-m 8 --num-programs 256 \
  --cluster-barrier-interval 0

# Selected benchmark defaults: align requests every four input K blocks.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --xcd-remap none --cluster-size 4 --group-m 8 --num-programs 256 \
  --cluster-barrier-interval 4
```

Use interval 1 for the original per-load synchronization. Add
`--no-cluster-multicast` at any interval to compare independent loads with
the same synchronization cadence.

```bash
# Remapping alone, with the current per-variant tile defaults.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --cluster-size 1 --xcd-remap balanced

# Multicast with the original tile ordering.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --xcd-remap none --cluster-size 4 --group-m 8 --num-programs 256

# Same ordering and cluster synchronization, without shared loads.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --xcd-remap none --cluster-size 4 --group-m 8 --num-programs 256 --no-cluster-multicast

# Four-workgroup sharing of A/B and scales.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --xcd-remap chunked --cluster-size 4 --group-m 4 --num-programs 256

# Same workgroup mapping and barriers, with multicast disabled.
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --xcd-remap chunked --cluster-size 4 --group-m 4 --num-programs 256 --no-cluster-multicast
```

Clustering supports no remapping, or chunked remapping with eight XCDs and
chunk size two. Both require group-M of four or eight, full M groups,
an even number of N tiles, both
scales, and partial/two-way/four-way TDM fusion. The actual program count
must be divisible by 16 and divide the output tile count, keeping all cluster
members on the same loop boundaries. Override `--num-programs` to select
a different count satisfying these conditions. The count includes all
workgroups, not clusters. Both BK128's separate output staging and BK256's
reuse of A for output are supported. Cluster barriers align requests to
improve sharing; they are not required for multicast correctness.

The summary and CSV record the remapping and cluster configuration. Standalone
flags use underscores (`--xcd_remap`, `--num_xcds`, `--xcd_chunk`, `--cluster_size`,
`--cluster_multicast`); Python API/config keys are `XCD_REMAP`, `NUM_XCDS`,
`XCD_CHUNK`, `CLUSTER_SIZE`, and `CLUSTER_MULTICAST`.
The barrier interval uses standalone flag `--cluster_barrier_interval` and
Python API/config key `CLUSTER_BARRIER_INTERVAL`; the summary and CSV record it.

Use repeatable `--variant` options to select variants. `--dtype-a float8_e5m2`
changes activations for the selected variants. Alternatively, `--dtype-b`
selects a single weight dtype (`float8_e4m3`, `float8_e5m2`, or `float4`) and
cannot be combined with `--variant`. Three buffers with 256x256x256 tiles are
supported for `mx8xmx4`; `mx8xmx8` exceeds LDS capacity with that configuration.

For one simulator dispatch per shape/variant, use `--benchmark-mode none
--output-dir <directory>` to keep each run's artifacts in a fresh subdirectory
whose name includes the variant. This mode leaves timing fields blank. Both
scripts provide `--help` for configuration options.

Compile checks live in `python/test/unit/language/test_tlx_codegen.py`; gfx1250
runtime checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`.
