# gfx1250 MXFP performance handoff

Snapshot: 2026-10-21. This is an investigation checkpoint, not a claim that
the latest compiler build reproduces the recorded hardware performance.

## Current conclusion

Keep the ordinary persistent kernel as the benchmark default. The latest
wave-specialized tile handoff is correct in the tested cases and removes
unnecessary control work, but has **no established whole-device speedup**.
It remains opt-in. FP32 output is required throughout this investigation.

The current A8W8 8K bottleneck is the steady K-loop schedule. WMMAs keep
the matrix unit busy only about 44% of the time because LDS operand
loads, barriers, and register copies are serialized against them by
compiler-emitted waits. The device also runs at its 2500 W power cap with
the shader clock near 1.85 GHz.

The worst of these waits (`va_vdst(0)` before LDS loads) came from the
wrong AMDGPU LLVM. Because `cmake/meta-llvm-info.json` exists,
`build_helpers.download_and_copy_amd_codegen` used to build the AMD codegen
against Meta's core LLVM (`a6533ab0`) instead of the `cmake/amd-llvm-info.json`
pin (`d26ff26e`). Both the local and remote installs, and therefore the
post-rebase hardware profile, used `a6533ab0`. The pinned `d26ff26e` codegen
emits partial `va_vdst` counts. On hardware at 8192 cubed, it is 2.4% faster
than the Meta-built codegen (2.2% at K4096). The build now always installs
the pinned codegen. The remote `tlx-env` install still carries the Meta build
until it is reinstalled; until then select the pin with
`TRITON_AMD_CODEGEN_PATH`. That reverts the Meta fork's gfx950
`V_PK_MUL_F32` encoding fix from `e565f60368`, which motivated the
override. The opt-in
`STATIC_SLOT_UNROLL` removes most remaining loop waits. With the pinned
codegen it gains another 1.2% at 8192 cubed (3.6% in total, 6455 TFLOPS) but
loses 1.8% at K4096. `bench.py` therefore enables it for MX8xMX8 cases with
K >= 8192; the kernel and `matmul` default remains off. The 16384- and
32768-cubed cases have not been measured with it. See
[Wait reduction experiments](#wait-reduction-experiments) and
[Hardware A/B](#hardware-ab).

The branch has been rebased and tested locally. After the rebase, only one
diagnostic hardware profile exists (A8W8 8192 cubed, see
[Post-rebase A8W8 profile](#post-rebase-a8w8-profile)). It locates the
current bottleneck but is not a controlled baseline. Collect interleaved
timings before making further performance comparisons or changing defaults.

## Source and build checkpoint

- Branch: `gfx1250-kernels-mxfp`.
- Kernel checkpoint: `6c9a6d9108`, `[TLX][AMD] Add wave-specialized MXFP tile handoff`.
- Rebased 47 commits onto `origin/main` at `b6ed2d1e6d`.
- Previous tip: `b49ab88e08`; preserved as
  `backup/gfx1250-kernels-mxfp-pre-rebase-20261021-b49ab88e08`.
- No push was performed. Tracked files were clean before this document.
- Preserve the untracked `third_party/amd/backend/lib/libtriton_amd_codegen.so`.
  Its snapshot SHA256 is
  `7eb3c96689737e0d8d9813a2f29dfd5ea2487771af6ba8e547d9726e87242401`.

Upstream now loads the AMDGPU backend separately from core MLIR/libtriton.
Record both identities: the core LLVM selection alone does not establish
which AMDGPU backend produced a binary. At this checkpoint the local core
LLVM cache points to `~/.triton/llvm/llvm-a6533ab0-ubuntu-x64-1/lib/cmake/llvm`;
`third_party/amd/backend/compiler.py` loads the separate codegen library.
Previous hardware measurements used `~/llvm/llvm-build`; do not silently
treat the rebased setup as the same toolchain.

Rebase validation: `make` and the Python package refresh succeeded;
13 lit regression files and 101 Python checks passed. Logs are local at
`/tmp/mxfp-rebase-20261021-Wgw30m/`. These checks establish build/regression
coverage, not performance equivalence or post-rebase GPU correctness.
The saved logs record first-run failures that were later resolved:
`amdgpu_membar.mlir` failed once and passed on rerun, and 10
`test_mxfp_trace_collection_packages_timing_power_and_decoded_bundle`
cases expected four default cases instead of eight. All 89
`test_tlx_benchmark_gen.py` tests and the membar lit test pass at
`6c9a6d9108`. Locally, nine native sources are newer than the installed
`python/triton/_C/libtriton.so`; run `make` before native testing.

The remote `tlx-env` has a non-editable install of
`triton-3.9.0+git6c9a6d91`, built 2026-10-21 from the remote checkout at
the same commit. Its AMD codegen library reports LLVM
`a6533ab0f5a1df16ff3dbf7358fb79097c60855e-1`, the Meta core LLVM, rather than
the AMDGPU pin `d26ff26e` (see Current conclusion). The pinned prebuilt
library is cached locally at
`~/.triton/amd/amd-codegen-d26ff26e-ubuntu-x64-1/lib/libtriton_amd_codegen.so`
and reports `d26ff26ef4e94a4de9ce5228ef5b32b5290cca8f-1`. The remote `~/llvm/llvm-build` is still `d4ad49fb3e5e`, so the
installed environment no longer uses the toolchain behind the October 15
measurements.

## Captured benchmark defaults

[bench.py](bench.py) is the source of truth for resolved defaults. Both
variants use E4M3 activations, E8M0 scales with preshuffled scale storage,
FP32 output, persistent 256x256 M/N tiles, three payload/scale buffers,
four waves, partial unsplit TDM, sliceMNK scheduling, and C output staging.

| Setting | A8W8 (`mx8xmx8`) | A8W4 (`mx8xmx4`) |
|---|---|---|
| B format | E4M3 | Packed FP4 E2M1 |
| BK | 128 | 256 |
| Cross-tile prefetch | On | Off |
| Split cluster barrier | Off | On |
| Early C stores | Off | On |

Common launch settings: group-M 8, 256 programs capped by tile count,
four-workgroup multicast with a 2x2 cluster layout, and a cluster barrier
every four input K blocks. XCD remapping and SCHED_MODE[2] are disabled;
L2 prefetch distance is -1. Optional cluster rendezvous promotes multicast
matching; it must not be confused with required LDS reuse synchronization.

Experimental pipelines, output-tail reuse, first-use prefetch, compact K
tail, streamed operands, async output, and wave-specialized handoff default
off. Handoff scheduling remains `current`, with partial handoff TDM fusion.

The default sweep is both variants at 8192x8192x8192, 8192x8192x4096,
16384x16384x16384, and 32768x32768x32768: eight runs. Eager timing uses a
**256 ms budget**, not 256 launches. Keep the timing mode and cache-clearing
behavior identical across comparisons.

## Latest controlled hardware comparison

Collected 2026-10-15, before the rebase. A8W8, M=N=8192, FP32 output.
Each value is the median of six fresh-process timings. The experiment
interleaved all six order permutations of ordinary, previous-specialized,
and fixed-specialized variants, alternating shape order. All 36 timings
ran under `gpu-lock`, with `do_bench(warmup=30, rep=256)` in milliseconds.

| K | Ordinary ms | Previous specialized ms | Fixed specialized ms | Fixed TFLOPS |
|---|---:|---:|---:|---:|
| 4096 | 0.099762 | 0.099253 | 0.099154 | 5544.47 |
| 8192 | 0.185241 | 0.185245 | 0.185230 | 5935.93 |

Fixed versus ordinary median time reductions were 0.609% and 0.006%.
Fixed versus previous specialization was faster in three pairs and slower
in three pairs at each K. The fixed ranges were 0.096253–0.101400 ms and
0.184754–0.187487 ms. This does not establish a throughput improvement.

Validation accompanying these measurements:

- 34 focused checks passed: 20 signed ring-phase cases, six codegen cases,
  and eight incompatible-configuration rejection cases.
- Specialized outputs matched ordinary outputs bitwise for all 67,108,864
  FP32 elements at each shape. Ordinary outputs were independently checked
  on 524,288 elements per shape using the existing CPU reference tolerance.
- Every timed `.text` matched the audited code, and each variant's HSACO
  was identical across repetitions.
- All variants reported 282,760 LDS bytes, 851 registers, and zero spills.
  The specialization's padded `.text` shrank from 13,056 to 12,928 bytes.

The hardware run used the existing `tlx-env`, Python 3.12.13, and
`~/llvm/llvm-build` at `d4ad49fb3e5e295fd37dea4817f02fe39c8091ca`.
Native libtriton SHA256:
`82bc688feb259cee6871cb64c1a9532710fb42ed41f05c2d586dec28eff6c53f`.
Environment configuration and device controls were unchanged.

Private local evidence directory:
`/home/ibutygin/tlx/perf-data/mxfp-wave-handoff-wait-hardware-20261015-2ww6t007/`.
Read `REPORT.md`, `summary.json`, `timings/summary.csv`, and
`hardware-validation.json`; `binaries/` contains the actual compiler output.
An archive is alongside the directory with the same name plus `.tar.gz`.
These are machine-local artifacts, not repository dependencies.

## Post-rebase A8W8 profile

Collected 2026-10-21 at `6c9a6d9108` with the installed remote build
(codegen LLVM `a6533ab0`). Ordinary persistent A8W8, M=N=K=8192, captured
benchmark defaults, FP32 output. The run was one `collect_traces.py`
invocation under `gpu-lock`:

```bash
gpu-lock python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/collect_traces.py \
  --kernels persistent --output /tmp/mxfp-a8w8-8k-profile-20261021 \
  -- --variant mx8xmx8 --case 8192,8192,8192
```

Median 0.177709 ms (6187 TFLOPS) over 17 eager 256 ms batches after
10 s of warmup. This uses a different procedure from the October 15
fresh-process timings; do not treat the 4% difference as a measured
rebase speedup. Correctness was not checked in this run.

Archive: `/home/ibutygin/tlx/perf-data/mxfp-a8w8-8k-profile-20261021.tar.gz`,
SHA256 `146f1ea9617c8a8c4d7d1f71ee66842addbbad3f2062cadaea581e17027a2355`,
extracted alongside; the remote copy is in `/tmp`. The ATT bundle passed
validation: dispatch 1025 (one-based), CU 0, shader-engine mask `0x1`,
one fully decoded wave (`se0_sm3_sl0_wv0`) plus LDS issue times for one
other SIMD. Timed and traced HSACOs are identical.

Binary versus October 15 ordinary K8192: `.text` is 11,392 bytes instead
of 11,520, with the same 851 VGPRs, no spills, and the same WMMA and LDS
instruction counts. It has 22 tensor waits instead of 12, 31 barriers
instead of 33, and one new `global_prefetch_b8`. The codegen change
explains these; they are not source changes.

**The device is power-limited.** During measurement, socket power averaged
2480 W (max 2498 W) against a 2500 W cap. The shader clock averaged
1846 MHz (1829–1869 MHz), compared with 2383 MHz at idle. The traced dispatch
lasted 165.6 µs over 298,174 cycles, also about 1.8 GHz. Cycle savings
will therefore not translate one for one into time: higher matrix
utilization will draw more power.

**Matrix utilization is about 44%.** Each wave issues 16,384
`v_wmma_scale_f32_16x16x128_f8f6f4`: four tiles × 64 K steps × 64 WMMAs.
Back-to-back WMMAs issue every 8 cycles (13,361 of 16,383 intervals). The
matrix floor is therefore 131,072 cycles against 298,122 measured. With
282,760 LDS bytes there is one wave per SIMD, so no other wave fills its
gaps. The other traced SIMD issues its LDS bursts within about 0–40 cycles
of the decoded wave, so the waves run in near lockstep.

Wave time by phase: the K loop takes 86% (61 iterations per tile, 244
total), the three-stage tail plus epilogue and tile handoff take 12%
(about 9k cycles per tile), and the prologue takes 2% (5.6k cycles).
Exposed TDM latency is small: steady-state `s_wait_tensorcnt` costs about
10 cycles per iteration.

**The steady K loop is the bottleneck.** The median iteration takes about
975 cycles against a 512-cycle WMMA floor. Over 214 iterations under
1100 cycles, the mean is 986. Attributing WMMA idle time inside those
iterations:

| Cycles/iteration | Cause |
|---:|---|
| 512 | WMMA floor (64 × 8 cycles) |
| ~263 | LDS operand loads issued while the matrix unit is idle |
| ~109 | Two workgroup barriers plus TDM issue, with only 2 WMMAs inside |
| ~97 | Loop-carried operand copies gated by an LDS completion countdown |
| ~30 | SALU/VALU address and slot bookkeeping |

Slow iterations (up to 3,684 cycles, likely cluster-barrier and multicast
iterations) raise the all-iteration mean to roughly 1045 cycles.

The idle time comes from specific waits in the emitted code. The trace
does not record `s_wait_alu`, so these waits appear as gaps after the
preceding instruction.

- **`va_vdst(0)` after WMMA.** The kernel enables expert scheduling mode
  (`s_setreg … HW_REG_WAVE_MODE, 25` at entry), so `SIInsertWaitcnts`
  emits explicit VALU dependency waits. Each `ds_load` burst uses an
  address VGPR produced by a `v_add` issued just before a WMMA. VALU and
  XDL writes are both pending, which the pass treats as out-of-order
  completion, so it emits `s_wait_alu depctr_va_vdst(0)`. That waits for
  the WMMA as well. WMMA→`ds_load` is 31 cycles in all 744 occurrences
  (p90 37), compared with 8 for WMMA→WMMA and 4 for `ds_load`→WMMA. With
  three bursts per iteration, this costs about 70 cycles. The LLVM source
  notes this limitation: "Per-pipe tracking would allow computing tighter
  va_vdst values".
- **`vm_vsrc(0)` before scale loads.** In the 34-load burst,
  `ds_load_2addr_b32 v[0:1], v1` and `v[2:3], v3` overwrite `v0` and `v2`,
  the address registers of the 16 queued `ds_load_b128` each. The
  resulting `s_wait_alu depctr_vm_vsrc(0)` waits until the LDS queue has
  read every queued address: 14–16 cycles, then 65–75 cycles while all
  waves load concurrently. The b128 loads themselves issue at about one
  per cycle.
- **Operand rotation at the loop end.** The final 16 loads land in
  temporary registers. Then 32 `v_dual_mov_b32` copy 64 VGPRs into the
  registers the next iteration reads, alternating with `s_wait_dscnt`
  0x10 down to 0x0 at about 5 cycles per step, with no WMMA overlap.
- **Barrier section.** Each iteration performs `s_wait_dscnt 0`, slot
  release, a barrier wait of about 42 cycles, two `tensor_load_to_lds`
  with counted tensor waits, and a second barrier wait of about 13
  cycles. Only two WMMAs execute in this section.

These items are emitted waits and register choices, not hardware limits.
Candidate changes:

- Make LDS addresses loop-invariant so no per-iteration VALU result feeds
  a `ds_load`. For example, unroll by the three-slot ring depth so slot
  offsets become immediates. Unrolling by two or more also allows operand
  register rotation without the end-of-iteration copies.
- Keep scale-load destinations away from in-flight address registers, or
  issue scale loads first.
- Place WMMAs between barrier signal/wait and around TDM issue.
- In LLVM, add per-pipe `va_vdst` tracking in `SIInsertWaitcnts`.

Unresolved ceiling: each CU reads about 135 KB of LDS per iteration
(4 waves × 64 b128 loads × 512 B, plus scales), because each operand
is read by two waves in the 2x2 wave layout. That is 137 B/clk at the
measured mean, and at least 145 B/clk in the fastest iterations. A
512-cycle iteration would need about 264 B/clk. Establish gfx1250's peak
LDS read bandwidth before estimating the remaining gain from schedule
fixes.

## Wait reduction experiments

Done 2026-10-21 on the AM model (package r8.05, `AM_CLOCK_MT=0`, expert
scheduling on, default LLVM flags). [Hardware A/B](#hardware-ab) below
follows up on hardware. Scratch harnesses and every run are in
`/home/ibutygin/tlx/am-runs/mxfp-waits-20261021/`, which is machine-local.
`compile_variant.py` compiles the persistent kernel offline. Its default
output is instruction-for-instruction identical to the hardware-profiled
binary above, so offline assembly is a faithful proxy for that binary.
`run_am_case.py` launches one bench-configured dispatch with CPU-generated
inputs and checks the result on the CPU. `run_model.sh` runs it under AM or
FFM, and `collect.py` summarizes the runs.

Configuration: bench A8W8 defaults (2x2 four-CTA clusters, barrier interval
four) at M=2048, N=1024, 16 programs, so each program handles two tiles. This
is the smallest shape the clustering checks accept. Every run passed the CPU
reference check.

**Use plateau utilization, not total cycles.** AM's per-500-cycle
`wmma_ops` counters show a steady plateau and two dips: start-up and the
tile handoff. The dips are dominated by cold instruction-cache misses on
code executing for the first time. Even baseline waves spend about 16% of
the dispatch waiting on instruction fetch (`wave_icache_clocks`), with
mean fetch-miss latency in the thousands of cycles while TDM traffic
saturates memory. Total cycles therefore penalize any code-size increase in
these short runs. With the same Meta-built binary as the hardware profile,
AM's baseline plateau was 0.564, against 0.525 in the hardware ATT steady
loop, so AM tracks the hardware loop reasonably.

The reference is the pinned AMDGPU codegen (`d26ff26e`). Percentages are
relative to that pinned baseline at the same K.

| Codegen | Variant | Steady waits per K step (`va_vdst(0)` / `vm_vsrc(0)` / copies) | K4096 plateau | K4096 cycles | K8192 plateau | K8192 cycles |
|---|---|---|---:|---:|---:|---:|
| Pinned `d26ff26e` | Baseline | 0 / 5 / 25 | 0.586 | 88,835 | 0.583 | 145,727 |
| Pinned `d26ff26e` | `STATIC_SLOT_UNROLL` | 1.33 / 0 / 0 | +4.4% | +16.6% | +3.2% | +6.8% |
| Meta `a6533ab0` (installed) | Baseline | 4 / 4 / 33 | −3.8% | +0.8% | −3.4% | +1.7% |
| Meta `a6533ab0` | Expert scheduling off | 0 / 0 / 33 | −10.7% | +6.0% | | |
| Meta `a6533ab0` | LLVM unroll ×2 | 2.5 / 1 / 0 | −4.0% | +8.6% | | |
| Meta `a6533ab0` | `STATIC_SLOT_UNROLL` | 2 / 0.33 / 0 | −2.5% | +15.7% | −2.1% | +7.8% |
| Internal `d4ad49fb` | Baseline | 0 / 5 / 25 | +0.4% | −1.3% | −0.1% | −0.9% |
| Internal `d4ad49fb` | LLVM unroll ×2 | 0 / 1.5 / 0 | +1.7% | +8.5% | | |
| Internal `d4ad49fb` | `STATIC_SLOT_UNROLL` | 0.33 / 0 / 0 | +4.7% | +14.6% | +3.5% | +5.5% |

Findings:

- **`va_vdst(0)` comes from building the codegen against the Meta LLVM.**
  With identical source, the pinned AMDGPU LLVM (`d26ff26e`) and the October
  15 internal LLVM (`d4ad49fb`) emit partial counts such as
  `va_vdst(14/6/7/14)`. The Meta-built codegen (`a6533ab0`) emits
  `va_vdst(0)` instead. The AMDGPU LLVMs model 32-bit-accumulator WMMA writes
  as in-order; internally that change is `605ea4f2`, "Model VaVdst-ordered
  XDL WMMA/SWMMAC writes in SIInsertWaitcnts". The pinned and internal
  codegens perform within 1% of each other on AM. Select the pinned library
  with `TRITON_AMD_CODEGEN_PATH`; the default build replaces it with the
  Meta build whenever `cmake/meta-llvm-info.json` is present.
- **Keep expert scheduling.** Hardware interlocks are 7% slower in the
  steady loop than explicit waits.
- **Unrolling alone does not help.** LLVM unroll by two removes the copies
  and most `s_wait_dscnt`, but the plateau does not move with the Meta
  codegen. The knob was removed.
- **Scales first does not help.** Issuing scale loads before payload loads
  changes only one `vm_vsrc(0)` and was not kept.
- **`STATIC_SLOT_UNROLL` (opt-in, default off).** This unrolls the A8W8
  steady loop by the three-slot ring. Each step's slot is loop-invariant, and
  static steps handle the `(K/128 − 3) mod 3` remainder. LLVM IR hoists all
  LDS address arithmetic, operands land directly in the registers the next
  step reads, and almost all `vm_vsrc(0)` and copies disappear. Registers
  stay at 851–873 with no spills, but static code grows from 1,508 to 2,073
  instructions. The backend rematerializes several address adds back into
  the loop, so some `va_vdst(0)` remain. With the pinned codegen, the
  steady loop gains 3–4%. In short AM runs the extra cold instruction-cache
  misses outweigh that: about 48% more unique code lines and +13k cycles of
  fetch wait per wave at K8192. Hardware amortizes this over four tiles per
  program and warmer caches, so only a hardware A/B can decide it.

Validation (with both the Meta and the pinned codegen): 12 new FFM
ring-phase cases cover empty, partial, and multiple
unrolled steady loops with and without clustering. A codegen test checks
that the unrolled loop carries no operand copies. Existing persistent and
wave-specialized ring tests still pass on FFM, as do all 89
benchmark-tooling tests.

### Hardware A/B

Collected 2026-10-21 on the remote gfx1250 host with the installed `tlx-env`
(core `6c9a6d9108`). The kernel source was the working-tree tutorial copied
to the remote scratch directory `/tmp/mxfp-hw-ab-20261021/`; the remote
checkout was not modified. The pinned codegen was selected per process
through `TRITON_AMD_CODEGEN_PATH`. The pinned library already present in the
remote cache is byte-identical to the local one (SHA256 `d3c0b56f…`). All
GPU work ran under `gpu-lock`. The local archive is
`/home/ibutygin/tlx/perf-data/mxfp-hw-ab-20261021.tar.gz` (SHA256
`7fd87e5f57a024ebbf9e382c1a3bab0068a3889845287d0925b01467e4526a7a`). It
contains the scripts, correctness results, assembly, timing CSVs,
telemetry, and decoded ATT bundles.

Correctness: every variant matched the CPU reference on 2048x1024x4096
(clustered, 16 programs) and on eight sampled 256x256 tiles at 8192 cubed.
All four 8192-cubed outputs were bitwise identical
(`6a41bbf3…`). Every hardware assembly was instruction-identical to the
local offline compile used for the AM runs.

Timing: warmed-up `collect_traces.py` runs (10 s warmup, 5 s of 256 ms eager
batches, power sampled), in three interleaved rounds plus the trace run at
8192 cubed:

| Variant | 8192³ median ms (range) | vs installed | TFLOPS | K4096 median ms | vs installed |
|---|---:|---:|---:|---:|---:|
| Meta `a6533ab0` baseline (installed) | 0.17665 (0.17633–0.17733) | — | 6224 | 0.10591 | — |
| Pinned `d26ff26e` baseline | 0.17240 (0.17169–0.17299) | −2.4% | 6378 | 0.10359 | −2.2% |
| Pinned + `STATIC_SLOT_UNROLL` | 0.17034 (0.16995–0.17238) | −3.6% | 6455 | 0.10549 | −0.4% |
| Meta + `STATIC_SLOT_UNROLL` | 0.17422 (0.17402–0.17483) | −1.4% | 6311 | 0.11084 | +4.7% |

All runs sat at the 2500 W power cap (about 2470 W mean) with mean shader
clock 1814–1844 MHz at 8192 cubed and 1893–1927 MHz at K4096. Six
fresh-process `bench.py` rounds, with only 30 ms of warmup, gave the same
ordering at 8192 cubed but were slower (about 0.19 ms) and had 0.27 ms
outliers. Use warmed-up timings for comparisons.

ATT (one decoded wave, dispatch 1025, 8192 cubed):

| Variant | Median K step (cycles) | Mean K step | Wave span | WMMA→`ds_load` avg |
|---|---:|---:|---:|---:|
| Meta baseline | 981 | 1201 | 318,438 | 33.0 |
| Pinned baseline | 900 | 1148 | 304,508 | 5.8 |
| Meta + static slot | 839 | 1145 | 304,076 | 23.1 |
| Pinned + static slot | 835 | 1069 | 285,536 | 4.9 |

The pinned codegen removes the 31-cycle WMMA→LDS stall, and the median K
step falls from 981 to 835 cycles with both changes. Whole-kernel gains are
smaller because the remaining time is elsewhere:

- **Cluster rendezvous.** Every fourth step (the cluster barrier interval)
  takes 1,130–1,270 cycles at the median, 300–400 cycles more than other
  steps in every variant.
- **K tail and tile handoff.** The last four steps of each tile take about
  41k of the wave's roughly 280k cycles and did not improve.
- **Power cap.** The higher utilization runs at the same 2470 W, so clocks
  drop slightly.

At K4096 there are half as many steady steps per tile. There, the larger
`STATIC_SLOT_UNROLL` code costs more than it saves, as AM predicted.

Pre-existing post-rebase breakage found during validation (not fixed):

- `SCHED_MODE_2=True` no longer compiles. The kernel calls
  `tlx.amd_set_wave_sched_mode`, which the pre-rebase branch defined in
  `third_party/tlx/language/tlx/utility.py`. Upstream `33856162ca` replaced it
  with the `enable_wmma_queuing` kernel option, so 44 codegen and 12 FFM
  tests fail. The default is off.
- `test_gfx1250_mxgemm_wave_specialized_handoff_separates_queues` fails in
  all six cases on its tensor-wait-count assertions, with the unmodified
  tutorial source and with both the Meta and the pinned codegen.

## Earlier shape-scaling evidence

These are user-reported measurements of the captured default configuration
from a different batch. Do not combine them with the October 15 comparison
as if they were matched runs.

| Shape M x N x K | A8W8 TFLOPS | A8W4 TFLOPS |
|---|---:|---:|
| 8192 x 8192 x 8192 | 5651.44 | 7277.62 |
| 8192 x 8192 x 4096 | 5109.20 | 6091.88 |
| 16384 x 16384 x 16384 | 6462.55 | 8576.72 |
| 32768 x 32768 x 32768 | 6497.95 | 8464.43 |

Scaling supports investigating startup, K-tail, and output costs, especially
for A8W4. It does not isolate those costs: increasing M/N/K together also
changes input reuse and the number of tiles per program. A8W8 still has an
unresolved sustained K-loop throughput/overlap limitation. Neither 90%
compute utilization nor parity with the reported approximately 9 PFLOPS
Gluon result has been demonstrated.

## Latest implementation and lessons

The opt-in wave-specialized epilogue assigns next-tile A/B and scale TDM
issues to waves 0/1 and C TDM store issues to waves 2/3. **All four waves
still fill C in LDS.** One role branch contains the specialized sequence;
there are no individual runtime role branches per C panel.

Separate wave-local tensor counts keep output reuse waits from draining
input transfers. The first two C slots impose no input completion wait;
the final two C stores can remain pending into the next tile. The latest
cleanup removed dummy scalar moves in waits, redundant entry control,
a duplicate LDS-release/cluster fence, and redundant tail-drain decisions.
The emitted code changed as intended, but whole-device timings were neutral.

Retained baseline work includes persistent scheduling, preshuffled scales,
b128 accesses, layout/address/scheduling fixes, periodic multicast, and
A8W4 early C stores. Earlier deeper rings, operand/streamed/eight-wave
pipelines, async output, hybrid output, and handoff/fusion variants have not
established a replacement for that baseline. The hybrid output experiment
was saved separately; it is not the default path.

Lessons to retain when designing the next change:

- Separate wait counters do not remove physical C-slot reuse dependencies
  or contention on shared memory/transfer resources.
- More input buffers can consume C staging capacity and increase output
  transfer count. Evaluate the whole tile's LDS lifetime, not just K depth.
- A removed wait or shorter instruction sequence is only useful if it
  shortens the exposed critical path. Confirm with emitted assembly and
  matching hardware traces before interpreting a timing change.
- Model startup, steady K service, peeled tail, tile handoff, and final
  drain separately. Account for resource backpressure and finite buffers;
  do not assume nominal transfer latency is always hideable.
- Keep unsuccessful candidates as evidence. Diagnose whether the expected
  overlap was emitted, whether it occurred, and whether it mattered.

## Resume plan

1. Toolchain identities and the A8W8 K8192 binary comparison are recorded
   above. Repeat the binary comparison for A8W8 K4096 and both A8W4 shapes.
2. Revalidate correctness and collect interleaved A8W8/A8W4 baseline timings
   on the rebased build. Keep output type, input packing, device settings,
   and timing method fixed. Preserve commands, configurations, and binaries.
   Record power and clocks with every timing, since results are power-capped.
3. Reinstall the remote `tlx-env` from a checkout with the pinned-codegen
   build, after getting permission for that environment change. Measure the
   16384- and 32768-cubed cases with `bench.py`'s automatic static slot
   unroll. Next targets are the cluster rendezvous step and the K tail and
   tile handoff, both measured above. Fix the `SCHED_MODE_2` API break and the
   wave-specialized codegen tests separately. Measure peak LDS read
   bandwidth to bound the remaining steady-loop gain.
4. Vary M/N at fixed K and K at fixed M/N to separate one-time startup from
   per-output-tile costs. At 256 programs with 256x256 tiles, square M/N
   sizes 4096, 8192, and 16384 give one, four, and sixteen tiles per program.
5. Update the [resource/dependency model](../../tools/perf_model/README.md)
   with the measured service times and exposed dependencies. Choose the next
   schedule change from a quantified gap; retain ordinary defaults until
   repeated hardware results justify replacing them.

Local command preparation (no GPU workload):

```bash
python3 third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run
```

After remote access is authorized, example baseline/candidate commands in
the existing environment are:

```bash
mxfp_bench=third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py
gpu-lock python3 "$mxfp_bench" --csv /tmp/mxfp-rebased-baseline.csv
gpu-lock python3 "$mxfp_bench" --variant mx8xmx8 \
  --case 8192,8192,8192 --case 8192,8192,4096 \
  --wave-specialized-handoff --csv /tmp/mxfp-rebased-specialized.csv
```

Use fresh result paths and repeated interleaved runs for conclusions; the
two example invocations alone are not a controlled comparison. For actual
instruction traces use [collect_traces.py](collect_traces.py) and the
collection/preflight instructions in [README.md](README.md). Successful
collection packages a single archive with timings, telemetry, source,
code objects, and decoded traces. Verify timing and trace code identity.

Remote access requires user permission for the task. Read the personal
remote-access skill at `~/.codex/skills/gfx1250-remote-access/SKILL.md` before
connecting. Use remote `~/tlx/triton`, accompanying `~/llvm/llvm-build`,
and the existing `tlx-env`. Do not change environment configuration or
device controls; run all GPU work, including correctness and warmups,
under `gpu-lock`. A handoff document does not authorize remote access.

Relevant regression locations are
`python/test/unit/language/test_tlx_amd_gfx1250.py` (ring phases) and
`python/test/unit/language/test_tlx_codegen.py` (queue separation and
configuration rejection). Follow root `AGENTS.md`: rebuild with `make`
before native/compiler tests, and run pytest with `-s --tb=short`.
