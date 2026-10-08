# gfx1250 MXFP GEMM

This directory contains the FP8/FP4 TDM-pipelined GEMM tutorial and a persistent
MXFP8 x MXFP8 / MXFP8 x MXFP4 benchmark sweep.

- `amd_mxfp_gemm_tdm_pipelined.py`: kernels, config-based `matmul` API,
  `mxgemm_tdm_pipelined` API, and single-shape benchmark CLI.
- `bench.py`: multi-shape benchmark runner with separate processes and CSV output.
- `collect_traces.py`: real-device instruction traces, eager timings, and
  timestamped power, clocks, temperatures, and power limits.
- `amd_mxfp_gemm_operand_pipeline.py`: experimental four-wave A8W8 kernel
  with operand prefetch across K steps and FP32 output.
- `amd_mxfp_gemm_operand_pipeline_paired.py`: two K256 input stages feeding
  native K128 computation, with partitioned input storage and cache prefetch.
- [`../../tools/perf_model/`](../../tools/perf_model/README.md): generic resource,
  dependency, and buffer-overlap model with an MXFP work-graph generator.
- [`hipblaslt_repro/`](hipblaslt_repro/README.md): public hipBLASLt gfx1250 A8W8
  assembly reproduction with FP32 output and a standalone HIP benchmark.

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

The default MX8xMX8 kernel handles its final three K stages in one loop.
It carries the physical ring slot across the tail and selects the required
immediate wait counts as pending transfers drain. This limits the tail's
code footprint while retaining cross-tile prefetch and native K128 WMMAs.

As the tail retires input slots, it fills them with the next tile's A/B and
scales. The two dedicated C slots can start their first output chunks while
those inputs remain in flight. Output waits begin with the third chunk,
when a C slot is reused, and leave the other chunk in flight. The current
tile's input waits retire any previous tile's C stores before this output
phase, so there is no need to drain incoming A/B before the first C chunk.

Run from the repository root in an environment configured for gfx1250:

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -M 8192 -N 8192 -K 4096
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

To collect hardware evidence for the persistent and streamed A8W8 schedules,
run the collector on an idle gfx1250 device in the same Python environment:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/collect_traces.py \
  --output /tmp/mxfp-hardware-01 --package
```

This compares both schedules at M=N=8192 and K=8192/4096, placing each
baseline next to its candidate. Each case repeats the ordinary eager
benchmark for 10 seconds of warmup and 5 seconds of measurement. Individual
timing batches retain `bench.py`'s 256 ms budget and cache-clearing behavior.
The power sampler polls the selected device's hwmon sensors every 100 ms.
Sampling records watts, available clocks and temperatures, the power cap,
utilization, and performance level, with wall-clock, monotonic, and raw
monotonic timestamps. The manifest maps every sensor to its source and unit.

ATT collection runs separately, with 1024 matching warmup launches followed
by one traced launch. Its selected dispatch is **1025, one-based**; the
collector checks the adjacent kernel trace against the decoded dispatch.
Cache clearing precedes that launch, matching the eager timing path. The
default trace selects CU 0, shader-engine mask `0x1`, and SIMD mask `0xF`;
`--target-cu`, `--shader-engine-mask`, `--simd-select`, and `--activity`
override these. Activity defaults to the profiler's architecture defaults.

The selected ROCm installation must provide an ATT-capable `rocprofv3` and
its matching trace decoder. Use `--profiler /path/to/rocprofv3` and
`--decoder-dir /path/to/decoder/lib` to select them explicitly. Collection
succeeds only after validating the raw trace, code objects, decoded UI files,
and results database. The compiled assembly must also match between the
timing and ATT processes.

The output contains `summary.csv` with timing and measured device power;
`manifest.json` with device identity, configuration, versions, and commands;
and a source snapshot. Each case has `power/` and `att/` directories with
`telemetry.csv`, `telemetry.jsonl`, `telemetry_summary.json`, `workload.json`,
logs, generated assembly, and the compiled code object. The viewer bundle is
under `att/trace/`. Available AMD SMI snapshots retain per-domain clocks and
throttle counters before and after each process. `--package` archives the
whole collection and prints its SHA-256 digest. Existing non-empty output
directories and archives are rejected.

Power describes the repeated benchmark workload, including its cache clears
and host launch gaps. The summary averages the valid samples wholly inside
the measurement phase. Sensor averaging and update periods are controlled by
the driver; polling at 100 ms does not give per-dispatch power resolution.
The ATT run has its own power series and timing windows. GPU limits and clocks
are read without changing device settings. Existing GPU visibility is
preserved; `--gpu INDEX_OR_UUID` explicitly selects a physical ROCr device.
The sampler uses the PCI address reported by HIP to select its sensors.

Power sampling prefers `gpu_metrics` socket power in watts, supporting table
versions 1.4 through 1.9. If that reading is unavailable, each sample tries hwmon
`power1_input`, then `power1_average`. Unavailable readings remain missing.
Every sample records `power_source` and `gpu_metrics_version`; phase summaries
and `summary.csv` include source counts so a fallback to averaged power is
visible. Raw sensor errors are retained even when another source succeeds.

Preview the cases without accessing a GPU, select one shape, or collect
timing and power when a decoder is unavailable:

```bash
python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/collect_traces.py --dry-run

gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/collect_traces.py \
  --output /tmp/mxfp-k4096 -- --case 8192,8192,4096

gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/collect_traces.py \
  --no-att --output /tmp/mxfp-power --duration-seconds 10
```

Options after `--` use the benchmark's argument validation. For example,
`--kernels persistent -- --variant mx8xmx4` profiles the A8W4 baseline.
The streamed schedule supports A8W8. Use `--warmup-seconds`,
`--duration-seconds`, `--warmup-dispatches`, and `--sample-ms` to adjust
collection lengths and sampling. These diagnostic runs do not perform
numerical correctness checks.

If power validation fails, inspect the saved capture without rerunning the
workload or accessing the device:

```bash
python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/collect_traces.py \
  --inspect /tmp/mxfp-hardware-01
```

The report includes total and valid readings, power sources, sensor errors,
read-batch durations, and counts both inside and overlapping each workload phase.
Collection also saves `telemetry_diagnostics.json` and
`telemetry_summary.json` before reporting insufficient power samples.

Use `--first-use-prefetch` to test the model-derived operand schedule:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --first-use-prefetch --csv mxfp-first-use.csv
```

This selects persistent E4M3 x E4M3 with three BK128 buffers, four waves,
and dedicated FP32 output staging. The C00/C10/C01/C11 compute order lets the
final two quadrants cover the next stage's operand loads after the C10 refill.
B1 stays in the current iteration. C01 places four LDS reads between pairs
of WMMAs, spreading next A0/B0 reads across its matrix windows. C11 covers
next A1 reads, and C00 covers current B1 reads.

Two K iterations rotate the operand register sets, eliminating the payload
copies needed by a single-step loop backedge. Packed 32-bit loop carries
preserve the native FP8 layout. The last K step uses the same loop, keeping
the even benchmark shapes at two static compute bodies. Its lookahead fills
the next tile's LDS stages; the final prefetched register operands are unused.
Global packing and scale packing are the same as the default kernel. Both
default K shapes run with the usual 256 ms timing budget.
The standalone flag is `--first_use_prefetch`; the API and `matmul`
configuration use `FIRST_USE_PREFETCH=True`. Compare it with
`--variant mx8xmx8` using interleaved runs on the target hardware before
selecting a default. This is an opt-in scheduling experiment; fewer loop
instructions or shorter matrix phases alone do not establish a complete
dispatch speedup.

Use `--output-tail-reuse` to test output staging in the retired third A/B
input stage:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --output-tail-reuse --csv mxfp-output-reuse.csv
```

This selects persistent E4M3 x E4M3 with three BK128 buffers and four waves.
Two next-tile input stages remain prefetched while the third A/B stage holds
two 64x128 FP32 output panels. After the output stores complete, that stage
is refilled before its next consumer. The output views retain the input
ring's physical slot stride across tile boundaries. This removes the separate
output allocation while retaining b128 stores and eight output transfers per
tile; the tradeoff is later prefetch of the third next-tile stage.
The standalone flag is `--output_tail_reuse`; the API and `matmul`
configuration use `OUTPUT_TAIL_REUSE=True`. The saved benchmark defaults
remain available without this flag.

Use `--operand-pipeline` to test the four-wave operand pipeline:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --operand-pipeline --csv mxfp-operands.csv
```

This selects nonpersistent E4M3 x E4M3, 256x256x128 tiles, four payload/scale
buffers, and four waves. Both default K shapes run with a 256 ms timing budget.
The kernel retains one rolling A register set, one extra 64-row A fragment,
and two B register sets. Next-stage operands load while current-stage matrix
instructions execute. A two-step K loop rotates the register banks without
bulk operand copies. A and B scales share the payload ring's four-slot lifetime,
avoiding a separate scale-reuse synchronization point. The 64x64 subtiles keep
scales in the matrix layout; payload reads and FP32 output stores use b128.

The path requires full M/N tiles and K >= 512 divisible by 256. It is available
as an explicit experiment for hardware comparison; benchmark defaults remain
the selected persistent configurations. To compare with the existing four-wave
kernel at the same tile and input-ring depth:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --variant mx8xmx8 --no-persistent --cluster-size 1 --num-buffers 4 \
  --no-output-staging --csv mxfp-baseline.csv
```

The standalone flag is `--operand_pipeline`; the Python API and `matmul`
configuration use `OPERAND_PIPELINE=True`, with four waves, partial fusion,
and staged output. The summary and CSV identify the kernel as `operand_pipeline`.

Add `-BK 256` to transfer two native K128 steps in each batch:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --operand-pipeline -BK 256 --csv mxfp-paired.csv
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --operand-pipeline -BK 256 --l2-prefetch-distance -1 --csv mxfp-paired-no-prefetch.csv
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --operand-pipeline -BK 256 --l2-prefetch-distance 2 --csv mxfp-paired-prefetch-2.csv
```

This selects two payload/scale buffers. Partitioned input storage separates
simultaneous read streams. The kernel reads the high K128 half early, refills
the retired input stage, and computes two high-half rows before waiting for
the next batch. It then reads A0 and all B fragments first, since the next
step uses them in its first row. Scheduling groups place address preparation
and LDS reads between independent matrix instructions. FP32 output uses
padded storage, b128 stores, and one TDM transfer.

The default cache-prefetch distance for this experiment is one input batch;
two gives more lookahead with the same LDS capacity, and -1 disables it.
Prefetch uses fixed descriptors and clamps the K index to the last valid
batch, including short K and the tail. The standalone/API settings are
`BLOCK_K=256`, `NUM_BUFFERS=2`, and `L2_PREFETCH_DISTANCE=1` (or 2/-1).
The original operand path uses `BLOCK_K=128`, `NUM_BUFFERS=4`, and
`L2_PREFETCH_DISTANCE=-1`. Persistent benchmark defaults remain unchanged.

Use `--variant mx8xmx8 --register-pipeline` to test the register pipeline:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py \
  --variant mx8xmx8 --register-pipeline --sched-mode-2 --csv mxfp-register.csv
```

This selects 256x256x256 tiles, two input buffers, and four waves. Each K256
input stage supplies two K128 register stages, with upcoming operand reads
interleaved with independent matrix work. After the refill, three quadrants
provide 48 WMMAs per wave to cover the next operand reads. The register
pipeline uses cluster rendezvous points to align multicast requests; local
TDM barriers still protect the input slots. Four 256x64 FP32 output panels
reuse the free A/B input stage while one stage of the next output tile stays
prefetched. It requires FP8 A and B, both scales, partial TDM fusion, and
output staging. The existing benchmark defaults remain available for paired
hardware comparisons. `--no-cross-tile-prefetch` disables the tile prefetch
while retaining the register pipeline and output panels.

The standalone flag is `--register_pipeline`; the Python API and `matmul`
configuration use `REGISTER_PIPELINE=True`. Specify the required BK256 and
two buffers when using those interfaces. The summary and CSV record
`register_pipeline`.

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

For one dispatch per shape/variant, use `--benchmark-mode none
--output-dir <directory>` to keep each run's artifacts in a fresh subdirectory
whose name includes the variant. This mode leaves timing fields blank. Both
scripts provide `--help` for configuration options.

Compile checks live in `python/test/unit/language/test_tlx_codegen.py`; gfx1250
runtime checks live in `python/test/unit/language/test_tlx_amd_gfx1250.py`.
