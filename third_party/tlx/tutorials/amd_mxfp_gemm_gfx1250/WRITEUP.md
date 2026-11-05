# MXFP GEMM on gfx1250: design, configuration, and performance

This document describes the persistent MXFP8 x MXFP8 (A8W8) and MXFP8 x
MXFP4 (A8W4) GEMM in this directory: how the kernel is built, how the
benchmark configures it, how fast it runs, and what limits it. It is a
summary. The measurements, failed experiments, and archives behind each
statement are in [HANDOFF.md](HANDOFF.md); usage and options are in
[README.md](README.md).

## Summary

- A8W8 reaches about 7,350 TFLOPS at 8192³ and about 7,460 TFLOPS at
  16384³ and 32768³ (FP32 output). A8W4 reaches 8,600-9,300 TFLOPS. FP16
  output is 3-9% faster than FP32.
- The device runs at its 2,500 W power cap in every sustained measurement.
  The shader clock settles at 1.5-1.9 GHz depending on shape, so cycle
  savings only show in wall time if they do not add energy.
- At 8192³, A8W8 issues WMMAs about half of the time (52% of wave cycles in
  ATT). The rest goes to the steady K loop's step synchronization and
  exposed LDS latency (about 290 cycles per 849-cycle step), and to tile
  transitions bound by C-store latency (about 15% of the wave).
- Many plausible fixes (removing waits, wave specialization, L2 prefetch,
  more lookahead, LDS-conflict-free layouts, direct stores) measured slower.
  The kernel behaves like a contended shared resource: stalls removed in one
  place come back as waits elsewhere or as a lower clock.

## Problem

`C[M, N] = (A * sA) @ (B * sB)^T` with block scaling along K:

| | A8W8 (`mx8xmx8`) | A8W4 (`mx8xmx4`) |
| --- | --- | --- |
| A | FP8 E4M3, `[M, K]` | FP8 E4M3, `[M, K]` |
| B | FP8 E4M3, `[N, K]` (B transposed) | FP4 E2M1, packed two per byte |
| Scales | E8M0, one per 32 K elements, preshuffled | same |
| C | FP32 (default) or FP16 | FP32 (default) or FP16 |

Accumulation is FP32 in both cases. FP16 output rounds the accumulators once
(round to nearest even) before staging.

## Hardware facts the design depends on

- 256 CUs in 8 XCDs; four SIMDs per CU.
- `v_wmma_scale_f32_16x16x128_f8f6f4` issues every 8 cycles per SIMD. At
  one wave per SIMD that is 32,768 FLOP per cycle per CU, about 15 PFLOPS
  at 1.8 GHz and 20 PFLOPS at 2.4 GHz for the whole device.
- 320 KiB of LDS per CU, in five 64 KiB partitions. Measured
  `ds_load_b128` latency is about 75 cycles; one wave issues one per cycle;
  a CU sustains 512 B/cycle across at least two partitions and 256 B/cycle
  within one.
- TDM (tensor data mover) copies 2D/3D tiles between global memory and
  LDS asynchronously. Transfers complete in order per wave
  (`s_wait_tensorcnt N` waits until at most N are outstanding). Loads can
  be multicast to the workgroups of a cluster (up to 16).
- The pinned AMDGPU backend inserts `s_wait_tensorcnt 10` before a TDM
  instruction when more than ten may be outstanding (hardware limit).
- 1,024 VGPRs per wave at one wave per SIMD; about 106 SGPRs.
- Sustained runs hit the 2,500 W socket power cap.

## Kernel design

The benchmark kernel is `mxgemm_tdm_persistent_kernel` in
[amd_mxfp_gemm_tdm_pipelined.py](amd_mxfp_gemm_tdm_pipelined.py).

**Work split.** Each program owns a sequence of 256x256 output tiles
(persistent scheduling, 256 programs, group-M 8 ordering). Four waves per
workgroup, one per SIMD, split the tile 2x2, so each wave holds a 128x128
FP32 accumulator (512 VGPRs). One A8W8 K step (BK128) is 64 WMMAs per wave;
A8W4 uses BK256.

**Input ring.** A, B, and both scale tensors stream into LDS through a
three-stage TDM ring. Loads use partial fusion: A data on waves 0/2, B data
on waves 1/3, scales likewise, two TDM instructions per wave per stage.
Each wave reads its 128-row A and B subtiles from LDS with `ds_load_b128`
(32 KB per wave per A8W8 step) and its scales into the WMMA scale operands.
A step does:

1. release barrier: all waves finished reading the slot to be refilled;
2. TDM refill of that slot, two stages ahead;
3. `s_wait_tensorcnt` for the next stage;
4. acquire barrier: the next stage is visible to all waves;
5. every eighth step, the cluster barrier.

About five WMMAs of the current step straddle that rendezvous, which is
what hides it. Several failed experiments lost exactly this property.

**Clusters and multicast.** 16 workgroups form a 4x4 cluster of output
tiles. Each A tile is multicast to the four workgroups of its cluster row,
each B tile to the four of its column, halving the unique input each CU
fetches compared with 2x2 clusters. A cluster barrier every eight K blocks
keeps the 16 workgroups close enough together for their multicast requests
to merge; without it, late peers issue separate loads that hit in L2.
Whole clusters are remapped across XCDs (`chunked`, eight XCDs, chunk two),
so clusters on one XCD share B columns and keep A rows across persistent
waves.

**Tile transition.** With cross-tile prefetch, the last three K steps of a
tile (the K tail) refill each freed slot with the next tile's data. The C
epilogue then writes the accumulators to LDS in 64x128 chunks and stores
each with a TDM store. A8W8 uses three dedicated C slots; a slot is reused
once its previous TDM store has completed. A8W4 instead places C in A
stages it no longer needs, and starts C stores early.

**A8W8 options enabled by the benchmark:**

- `STATIC_SLOT_UNROLL` (K ≥ 8192, three stages): unrolls the steady loop by
  the ring depth so each step reads a loop-invariant slot. LDS addresses and
  operand copies leave the loop, and each tile's first step is peeled so its
  WMMAs take zero accumulators as an inline constant.
- `LDS_PARTITIONED_AB` (with static unroll): A and B rows alternate in
  64-row pieces between two LDS partitions, so the waves reading one subtile
  contend less for a partition.
- `OVERLAP_PEEL` (three stages): the next tile's first K step runs inside
  the C epilogue. After a C quadrant is staged, its registers take that
  quadrant's next-tile WMMAs, which run during later slot waits. The next
  tile's first refill is issued before the last C chunks, because TDM
  completes in order.
- Scales first (dynamic loop, K < 8192): each quadrant waits only for its
  own LDS reads.
- For K ≥ 16384, a four-stage ring with two C slots replaces the three-stage
  ring; those shapes are power bound and gain from the extra lookahead.

**Resources (A8W8 8192³ default).** 859 VGPRs, no spills; 104 of about 106
SGPRs; 316,584 of 327,680 bytes of LDS with three FP32 C slots. VGPRs allow
one tile's accumulators, which bounds how much next-tile work can overlap
the epilogue. SGPRs are the tight file: any new live descriptor group in the
K loop triggers rematerialization and rescheduling that breaks the WMMA
overlap around the step rendezvous.

**Other kernels** (opt-in, not used by the benchmark defaults): eight-wave
persistent, register pipeline, warp pipeline, operand pipelines
([amd_mxfp_gemm_operand_pipeline.py](amd_mxfp_gemm_operand_pipeline.py),
[amd_mxfp_gemm_operand_pipeline_paired.py](amd_mxfp_gemm_operand_pipeline_paired.py)),
streamed operands with optional async output
([amd_mxfp_gemm_streamed.py](amd_mxfp_gemm_streamed.py)), and
wave-specialized tile handoff. None beat the default on hardware.

## Benchmark configuration

[bench.py](bench.py) is the source of truth. It runs each shape in a fresh
process: 8192²×4096, 8192³, 16384³, 32768³, both variants, FP32 then FP16
output (16 runs).

| Setting | A8W8 | A8W4 |
| --- | --- | --- |
| Tile, waves | 256x256, BK128, 4 waves | 256x256, BK256, 4 waves |
| Input ring | 3 stages; 4 stages with 2 C slots for K ≥ 16384 | 3 stages |
| C staging | 3 dedicated 64x128 slots (6 for FP16) | C in retired A stages |
| Cross-tile prefetch | on | off |
| Static slot unroll, 64-row LDS partition pieces | K = 8192 (three stages) | off |
| Overlap peel | three-stage ring | off |
| Split cluster barrier, early C stores | off | on |
| Clusters | 4x4, multicast, barrier every 8 K blocks | same |
| Program order | group-M 8, chunked XCD remap (8 XCDs x 2) | same |
| Programs | 256 (capped by tile count) | same |

Timing: eager 256 ms batches, 1 s warmup, median batch over 2 s. A fresh
process's first batch is inflated by slow first launches, so single-batch
numbers are not comparable.

```bash
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx8 -M 8192 -N 8192 -K 8192
python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

Build with the pinned AMDGPU code generator (`cmake/amd-llvm-info.json`,
`d26ff26e`); the build now always installs it. The Meta-built backend was
2.4% slower, and the newer upstream backend `05e432a8` is 13% slower at
8192³ because its scheduler stops straddling the step rendezvous with WMMAs.

## Performance

Sustained timing at the power cap, measured before the remote host was
reprovisioned on 2026-10-25 (A8W8 with `hw_power2.py` on 2026-10-24, A8W4
and A8W8 32768³ with `bench.py` on 2026-10-23):

| Shape | A8W8 FP32 | A8W8 FP16 | A8W4 FP32 |
| --- | ---: | ---: | ---: |
| 8192²×4096 | 0.0833 ms, 6,600 TFLOPS (1.89 GHz) | 0.0780 ms, 7,050 | 0.0733 ms, 7,504 |
| 8192³ | 0.1497 ms, 7,346 TFLOPS (1.80 GHz) | 0.1437 ms, 7,654 | 0.1276 ms, 8,619 |
| 16384³ | 1.1788 ms, 7,462 TFLOPS (1.63 GHz) | 1.1354 ms, 7,747 | 0.9566 ms, 9,195 |
| 32768³ | 9.4373 ms, 7,456 TFLOPS | — | 7.5451 ms, 9,326 |

A8W4 FP16 output is 4.7% (8192³), 8.7% (8192²×4096) and 2.6% (16384³)
faster than its FP32 output.

**Machine change.** After the 2026-10-25 reprovisioning (new build of the
GPU driver and firmware packages, different board firmware), the same
binaries run 6-7% slower with unchanged cycle counts: the sustained clock
at the cap dropped by the same fraction (8192³: 0.15989 ms at 1,685 MHz).
Compare only measurements from the same machine state.

## Where the time goes (A8W8, 8192³)

ATT traces of the default kernel (one wave per shader engine, 48 waves).
ATT runs below the power cap at about 2.4 GHz, so it shows the split of
cycles, not the power-capped wall time; it also inflates issue-level
differences, so do not use it for effects under about 2%.

| Share of wave cycles | 8192³ (static loop) | 8192²×4096 (dynamic loop) |
| --- | ---: | ---: |
| WMMA floor (8 cycles per WMMA) | 52.5% | 47.3% |
| Steady K loop | 83.0% | 71.6% |
| … WMMA issue | 41.2% | 34.1% |
| … `ds_load` issue + LDS waits | 14.7% + 8.1% | 14.8% + 9.7% |
| … workgroup + cluster barriers | 6.9% + 4.1% | 0.9% + 2.5% |
| … SALU / VALU | 3.4% / 0.8% | 4.6% / 3.4% |
| … TDM waits | 3.4% | 1.0% |
| Tile transitions (3 per wave) | 11.3% | 19.3% |
| Final epilogue | 3.9% | 6.3% |
| Prologue | 1.7% | 2.8% |

### Steady K step

A step takes about 849 cycles against a 512-cycle WMMA floor:

| Cycles | Part |
| ---: | --- |
| 512 | WMMA floor (64 WMMAs x 8) |
| ~140 | step synchronization: stage acquire `s_wait_tensorcnt` ~33, workgroup barrier 41-75, cluster barrier ~40 amortized |
| ~79 | LDS wait: latency of the new stage's first operand loads, issued after the acquire barrier while ~5 deferred WMMAs cover it |
| ~69 | extra `ds_load` issue cost beyond what overlaps the matrix pipe, incl. ~22 cycles of two full WMMA drains (`s_wait_alu depctr_va_vdst(0)`) from register reuse |
| ~49 | SALU/VALU (descriptor groups, addresses, cluster-barrier check) |

- **LDS bandwidth is not the limit.** The kernel reads about 154 B/cycle
  and TDM writes about 78 B/cycle per CU, far below the measured 512
  B/cycle. The cost is latency exposed at the step boundary.
- **No wave imbalance.** The four waves arrive at workgroup barriers within
  a few cycles of each other. The barrier tail comes from late TDM data:
  in about 17% of steps (outside the two steps after a cluster barrier) one
  wave's data arrives after the ring's two-step lookahead, and the others
  wait. Across the 16 CTAs the same tails accumulate into cluster-barrier
  waits of 240-310 cycles; no CTA is consistently last.
- **LDS partition conflicts are real but secondary.** AM and ATT confirm
  them; 64-row partition pieces gain 1.6%, but layouts that remove more
  conflicts unbalance the waves across partitions and lose.

### Tile transition and epilogue

Outside the K loop each wave spends about 41k of 247k cycles (17%), about
27k of them above the steady-loop rate. A transition costs about 9k
cycles: a K tail of ~3.8k, then a C epilogue of ~5.3k overlapped with the
next tile's first step.

- **The C path is the bottleneck.** 256 KB of FP32 C per CU per tile is
  written to LDS at about 130 B/cycle and drained by TDM at about
  64 B/cycle. A C store takes 600+ cycles to complete even with no input
  loads in the wave's queue, so slot reuse waits dominate.
- **Order matters.** The first slot-reuse wait (~650 cycles) also waits for
  the next tile's stage-2 load issued just before it, because TDM completes
  in order.
- **Device-wide store contention is bounded:** about 400 cycles per
  transition and 1.4k in the final epilogue, the same with 128 or 64
  programs.
- **Scalar work is cheap.** About 342 SALU per tile transition and 56 per
  loop iteration; adding 60 independent SALU per iteration costs 0.3%, so
  removing all redundant SALU is worth about 0.1%.

### A8W4

A8W4 runs a 687-718 cycle step per 64 WMMAs (WMMA utilization 0.61-0.65)
with almost no barrier waits; its LDS traffic per FLOP is 25% lower. At
8192³ its tile transitions are a larger share (about 25% of the wave,
mostly waits on C stores and next-tile loads), at 16384³ about 12%.

## What did not help

All measured on hardware against the default at the time, with bitwise
identical outputs; details in [HANDOFF.md](HANDOFF.md).

| Change | Result (8192³ unless noted) | Why |
| --- | ---: | --- |
| Cluster barrier every 16 blocks / split barrier | +3.6% / +3.2% | more drift, fewer merged multicasts |
| Four-stage ring at 8192³ | +1.6% | loses static unroll and the third C slot |
| L2 prefetch, next to refill / mid-step | +20% / +8% | doubles L2 requests, no fewer misses |
| E8M0 scales through VGPRs | +9.6% (+72% at 16384³) | exposed global latency |
| Static K tail; batched first C fill | +0.2%; +0.4% | waits move to cluster barrier / later stores |
| Removing in-loop LDS address adds | +0.45% | loads bunch up, barrier waits double |
| Avoiding WMMA register-reuse drains (codegen pass) | +0.15% | live ranges cost as much as drains |
| `sched_group_barrier` LDS interleaving | +0.6% to +11% | full WMMA drains before pinned loads |
| Contiguous wave rows / upstream partition-aware layouts | +5% / +1.7% | partitions unevenly loaded across waves |
| Direct C stores for one quadrant | +9.5% | partial-line stores, ~8x the requests |
| Wave-specialized epilogue; C stores on scale waves | +11.9%; +4.7% | SGPR pressure reschedules the K loop; fewer store waves drain slower |
| `SCHED_MODE[2]` WMMA queuing | +0.4% | step bound by sync and LDS latency |
| Eight waves | +9.8% | two waves share each matrix unit |
| Automatic wave specialization | +0.3% (−0.5% at K=4096) | doubled code size |
| Newer AMDGPU backend `05e432a8` | +13% | scheduler drops the rendezvous overlap |
| Cache hints on C stores | 0% | L2 misses come from the input working set |

## Remaining opportunities

Each is small or needs compiler work:

- **Issue the next tile's stage-2 load after the first C chunks** so the
  first slot-reuse wait does not wait for it (up to ~0.5%).
- **Hoist A/B descriptor selects out of the tile transition** with a
  transform before fused-TDM lowering (select tile-invariant bases once,
  apply per-tile offsets after); about 0.2-0.5% at K=4096.
- **Cover post-barrier LDS latency with more matrix work**; a deeper
  register prefetch needs ~256 more VGPRs per wave than the budget allows.
- **Fewer LDS bytes per FLOP** (one operand fed without an LDS round trip)
  would raise the ceiling, but no design found so far fits the register
  budget.
- **Energy.** Under the cap, a change must save energy, not only cycles.
  FP16 output is the one change that clearly does both.

## Measurement practice

- Run all GPU work under `gpu-lock`; record power and clock with every
  timing (`hw_power2.py` samples `amd-smi metric --clock --power` in a
  background thread and reports medians over the timed window, plus cycles
  = time x clock and energy per launch).
- Compare variants interleaved in the same session; differences under
  about 0.3% are noise.
- Verify outputs bitwise against the default for every variant, and check
  the emitted assembly before interpreting a timing change.
- Use ATT for the split of cycles, not for small comparisons, and trace one
  wave per shader engine: single-CU traces can pick an unrepresentative
  wave.
- AM rewards LDS-overlap changes that hardware does not; do not tune on AM
  alone.
- FFM tests run in parallel (`pytest -n 48` or more).
