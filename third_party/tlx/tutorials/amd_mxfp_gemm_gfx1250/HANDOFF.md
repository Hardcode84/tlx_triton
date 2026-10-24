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
K >= 8192; the kernel and `matmul` default remains off. That schedule now
peels each tile's first K step. `bench.py` also stages A8W8 output through
three C slots instead of two, cutting tile transitions by about 35%. The
current bench defaults measure 0.16825 ms at 8192 cubed (6535 TFLOPS),
0.10176 ms at K4096, and 1.38280 ms at 16384 cubed. 32768 cubed has not been
measured. See [Wait reduction experiments](#wait-reduction-experiments),
[Hardware A/B](#hardware-ab), and [Tile transition](#tile-transition).

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
| Static slot unroll | K = 8192 (three-stage ring) | Off |
| Overlap peel (`OVERLAP_PEEL`) | Three-stage ring (K < 16384) | Off |
| A/B LDS partition pieces (`LDS_PARTITIONED_AB`) | 64 rows with static slot unroll | Off |
| Input ring | 3 stages; 4 stages, 2 C slots for K >= 16384 | 3 stages |
| Dedicated C slots | 3 (K < 16384) | — (C reuses A stages) |
| Scales-first K loop (without static unroll) | On | — (BK256 path) |

Common launch settings: group-M 8, 256 programs capped by tile count,
16-workgroup multicast with a 4x4 cluster layout, a cluster barrier every
eight input K blocks, and chunked XCD remapping of whole clusters (eight
XCDs, chunk two). SCHED_MODE[2] is disabled;
L2 prefetch distance is -1. Optional cluster rendezvous promotes multicast
matching; it must not be confused with required LDS reuse synchronization.

Experimental pipelines, output-tail reuse, first-use prefetch, compact K
tail, streamed operands, async output, and wave-specialized handoff default
off. Handoff scheduling remains `current`, with partial handoff TDM fusion.

The default sweep is both variants at 8192x8192x8192, 8192x8192x4096,
16384x16384x16384, and 32768x32768x32768: eight runs. Eager timing repeats
**256 ms batches** (a duration, not 256 launches): 1 s of warmup, then the
median batch over 2 s. Keep the timing mode and cache-clearing behavior
identical across comparisons.

A fresh process's first batch is inflated by a few slow first launches. On
hardware at 8192 cubed, its mean was 0.1875 ms against a median of
0.1666 ms in the same batch. Later batches settle at 0.167–0.169 ms, then
drift up about 1% over 12 s under the power cap. Older single-batch `bench.py`
numbers (about 0.19–0.20 ms at 8192 cubed) are therefore not comparable with
warmed-up numbers. With the pinned codegen, the current defaults (4x4
clusters with chunked XCD remapping) measured on 2026-10-23, next to the
4x4 defaults without remapping (same session) and the 2x2 configuration
(2026-10-22 session); ms, TFLOPS:

| Shape | MX8xMX8 | no remap | 2x2 | MX8xMX4 | no remap | 2x2 |
|---|---:|---:|---:|---:|---:|---:|
| 8192³ | 0.1530 (7185) | 0.1563 | 0.1707 | 0.1276 (8619) | 0.1286 | 0.1330 |
| 8192²×4096 | 0.0848 (6483) | 0.0849 | 0.0884 | 0.0733 (7504) | 0.0738 | 0.0758 |
| 16384³ | 1.1789 (7461) | 1.2105 | 1.3805 | 0.9566 (9195) | 0.9749 | 1.0230 |
| 32768³ | 9.4373 (7456) | 9.7822 | 11.2131 | 7.5451 (9326) | 7.7185 | 8.1754 |

Since `43869ddbf0` the 8192³ A8W8 default also spreads A/B over two LDS
partitions (64-row pieces): about −1.6% in `hw_ship.py` timing (0.1505–0.1509
ms against 0.1528–0.1532 ms in the same sessions; see [CDNA5 LDS partition
conflicts](#bottlenecks-after-4x4-clusters-dc853dbc9d)). It needs the
`memdesc_reshape` rule from `639c951130`, so `bench.py` on the remote
(older Triton) cannot run it until that environment is rebuilt; the table
above has not been re-measured with `bench.py`.

`OVERLAP_PEEL` (default for three-stage A8W8 in `bench.py`, see "Overlap
peel option") measured −0.21% at 8192³ and −1.20% at 8192²×4096 in
`hw_power.py` timing; it was not measured with `bench.py` either.

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

### Tile transition

Hardware ATT, pinned codegen with static slot unroll, 8192 cubed: between the
last WMMA of one tile and the first of the next, the wave idles for 7.1–9.2k
cycles (16 ideal K steps) in each of the three transitions. That is about 9%
of the wave. The A8W8 epilogue writes the 256x256 FP32 tile as eight
64x128 chunks through two dedicated 32 KB LDS slots. Each chunk takes about
210 cycles to fill (16 `ds_store_b128` per wave, LDS wait, barrier, TDM store),
but reusing a slot waits for that slot's previous store:

- The first reuse wait (`s_wait_tensorcnt 1`, 1.9–2.9k cycles) also waits for
  the next tile's prefetched input stages, because the counter retires in
  order.
- Later reuse waits take 1.1–1.5k cycles each. With two slots, at most two
  32 KB stores are in flight, so the epilogue is bound by store latency
  rather than fill work.
- After the last store, about 800 cycles go to zeroing 512 accumulator VGPRs
  and next-tile setup. The K-tail step that issues the next-tile prefetch
  takes 1.3–2.1k cycles instead of about 850.

`OUTPUT_C_SLOTS` and `OUTPUT_C_COLS` (kernel constexprs, default 2 and
automatic) make the dedicated staging configurable. Chunks rotate through the
slots across quadrants, and each reuse waits for `slots − 1` pending stores.
Three 64x128 slots use 316,552 bytes of LDS; six 64x64 slots use 319,624.
The default assembly is unchanged.

AM, pinned codegen and static slot unroll, M=N=K=2048, 16 programs, four
tiles per program. All outputs passed the CPU check and were bitwise
identical. "Lost" counts the cycles below plateau WMMA utilization in each of
the last two transitions, where code is warm:

| Output staging | Dispatch cycles | Warm transitions lost (cycles) |
|---|---:|---|
| 2 × 64x128 (current) | 111,542 | 5,637 / 5,473 |
| 3 × 64x128 | 107,692 (−3.5%) | 4,243 / 4,335 |
| 4 × 64x64 | 111,894 (+0.3%) | 5,477 / 5,430 |
| 6 × 64x64 | 108,936 (−2.3%) | 4,554 / 4,428 |
| `WAVE_SPECIALIZED_HANDOFF` | 114,159 (+2.3%) | 5,382 / 5,114 |

Three 32 KB slots cut warm transitions by about 23% on AM. AM runs only 16
workgroups, so it does not reproduce all 256 CUs storing at once. Hardware
timing at 8192 cubed must decide whether store bandwidth, rather than
latency, limits the gain there.

**Zero-accumulator peel.** `STATIC_SLOT_UNROLL` now peels the first steady
K step of each tile. That step's dots see the tile's zero accumulators as
constants, so all 64 of its WMMAs encode C as an inline `0` and the per-tile
zeroing disappears (`v_dual_mov_b32` 303 → 65, `v_mov_b32` 38 → 2). The
remaining steps keep `slots[k % 3]`, and the peel replaces one remainder step,
so static code shrinks from 2,229 to 1,937 instructions with 857 VGPRs.
Outputs are bitwise identical to the unpeeled schedule; the 12 FFM ring-phase
cases pass. Same AM setup as above:

| Variant | Dispatch cycles | Warm transitions lost (cycles) |
|---|---:|---|
| Static slot unroll (before peel) | 111,542 | 5,637 / 5,473 |
| + peel | 105,319 (−5.6%) | 4,981 / 4,986 |
| + peel, 3 × 64x128 C slots | 101,818 (−8.7%) | 4,203 / 4,126 |

Some of the peel's gain is AM's cheaper cold instruction fetch from the
smaller code. Overlapping accumulator zeroing with C-slot waits would still
help the non-unrolled path (K < 8192 in `bench.py`). It cannot be combined
with the peel, because zeros carried across tiles are no longer constants.

With a remainder after the unrolled loop (K = 4096 or 16384), the peeled and
remainder dots initially selected a different WMMA warp distribution from the
loop. That inserted 23 `convert_layout` operations and 16 KB of LDS scratch,
which made three C slots exceed 320 KB. Static-slot steps, and every other
dot under `STATIC_SLOT_UNROLL`, now use `_mxgemm_native_dot`, the store's
native layout. No conversions remain at K = 4096, 8192, or 16384; LDS is
282,760 bytes with two slots and 316,552 with three. The K8192 assembly is
unchanged from the AM runs, and the default (non-static) assembly is
unchanged.

**Hardware A/B (2026-10-21).** Remote gfx1250, pinned codegen,
`collect_traces.py` warmed-up timing: four interleaved rounds, plus the trace
run at 8192 cubed. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-hw-tt-20261021.tar.gz` (SHA256
`1fc8281526497d3539908a05b4620a6b0360a15233153827b2a3515db0fd94ad`). All
variants passed the CPU checks; 8192-cubed outputs were bitwise identical.

| Variant | 8192³ ms (TFLOPS) | 8192²×4096 ms | 16384³ ms |
|---|---:|---:|---:|
| Committed static slot unroll (`f8aca875c4`) | 0.17075 (6439) | 0.10365 | 1.38529 |
| + zero-accumulator peel | 0.17052 (−0.14%) | 0.10365 (±0%) | 1.38772 (+0.18%) |
| + peel, 3 C slots | **0.16825 (−1.47%, 6535)** | **0.10176 (−1.82%)** | 1.38280 (−0.18%) |

The 8192²×4096 case does not use static slot unroll, so only the C slots
apply there. ATT at 8192 cubed (one wave, three transitions):

| Variant | Transition gap (cycles) | After last C store | Slot-reuse waits |
|---|---|---|---|
| Committed | 6,611 / 7,290 / 7,009 | 791–892 | 1.2–1.7k each |
| + peel | 6,325 / 6,283 / 6,400 | 411–548 | 1.0–1.5k |
| + peel, 3 C slots | 4,419 / 4,050 / 4,502 | 452–459 | 0.1–0.8k |

The peel removes about 400 cycles of zeroing per transition, which is below
whole-kernel noise. The third C slot cuts transitions by about 35%. At 16384
cubed, with 128 K steps per tile, transitions matter less, and clocks drop to
about 1.72 GHz under the power cap.

### Next A8W8 bottleneck (after `156c70871f`)

Source: the hardware ATT of the committed configuration at 8192 cubed
(`mxfp-hw-tt-20261021`, variant `peel-s3`; one wave, 278,722 cycles, WMMA
utilization 0.47).

| Phase | Cycles | Share |
|---|---:|---:|
| Steady K steps (60 per tile) | 232,970 | 83.6% |
| … above the 512-cycle WMMA floor | 110,090 | 39.5% |
| K tail (3 steps per tile, ~1.2k each) | 15,073 | 5.4% |
| Tile transitions | 12,971 | 4.7% |
| Prologue | 10,009 | 3.6% |
| Final epilogue | 4,730 | 1.7% |

Steady steps: median 836 cycles, mean 971. They split into two kinds:

- **Late input (about 226 cycles per step on average, 23% of steady time).**
  The slowest 10% of steps average 1,794 cycles. In them, the traced wave
  waits 1–2k cycles at the workgroup rendezvous after the TDM refill
  (`s_barrier_wait` at 0x2e34/0x35f8/0x3da4), while its own tensor wait is
  short: a sibling wave is waiting for late TDM input. Slow steps cluster at
  k mod 4 = 3 (12 of the slowest 24) and at tile starts. Cluster rendezvous
  steps (k mod 4 = 1) add another ~50 cycles per step on average. The likely
  cause is multicast coupling and memory latency spikes not covered by the
  three-stage ring. AM cannot reproduce 256 CUs, so this needs a hardware
  sweep of cluster barrier interval, split cluster barrier, multicast, and
  cluster size.
- **Built-in overhead (fast half: 755 cycles, 243 above the floor).** Mostly
  `s_wait_dscnt 0x0` drains (~84 cycles at each of three sites). The
  next-stage burst of 19 `ds_load_b128` is followed by only 16 WMMAs before
  its consumer, plus two rendezvous per step.

AM experiment, M=N=K=2048, 16 programs, deliberately incorrect variants that
reuse operands from registers instead of reading LDS (temporary code, not
committed). Steady step from plateau WMMA utilization:

| Variant | Steady step (cycles) |
|---|---:|
| Committed configuration | 826 |
| B reused (half the LDS operand reads) | 762 (−8%) |
| A reused | 764 (−7.5%) |
| No operand LDS reads | 666 (−19%) |

LDS operand traffic (~132 KB per CU per step with the 2x2 wave layout) costs
about half of the excess, but halving it saves only 8%. The step is not
purely LDS-bandwidth bound. Without any LDS reads, about 150 cycles remain
for the two workgroup rendezvous, TDM issue and waits, and the periodic
cluster rendezvous. Candidates, in order of measured size: make input
delivery robust to cross-CTA skew (hardware sweep above), give the
next-stage operand burst more matrix work to hide behind, and merge the
release and acquire rendezvous of each step.

**Cluster and multicast sweep (hardware, 2026-10-21).** Committed
`37b13785a6` defaults (static slot unroll with peel, three C slots), pinned
codegen, warmed-up `bench.py` timing. Medians of three interleaved rounds.
All variants passed the CPU checks, with bitwise-identical 8192-cubed
outputs. Archive: `/home/ibutygin/tlx/perf-data/mxfp-hw-sweep-20261021.tar.gz`
(SHA256 `5df288d27a8b4dc2a2161dec82fe41922c39dd2cc74658b2d93a72315bab3fdd`).

| Variant | 8192³ | 8192²×4096 | 16384³ |
|---|---:|---:|---:|
| Default: 4-CTA 2x2, multicast, interval 4 | 0.16709 ms | 0.10142 ms | 1.37282 ms |
| Interval 1 | +0.6% | +3.9% | −0.8% |
| Interval 2 | +0.4% | +0.4% | −0.7% |
| Interval 8 | +4.6% | +2.5% | +2.5% |
| Interval 16 | +13.3% | +11.9% | +13.7% |
| No cluster rendezvous (interval 0) | +70.7% | +43.0% | +82.4% |
| Split cluster barrier | +8.6% | +4.8% | +4.8% |
| Multicast off | +21.9% | +12.4% | +22.5% |
| 2-CTA linear cluster | +8.7% | +3.9% | +10.1% |
| No cluster | +22.8% | +10.9% | +23.7% |

The default stays best except for a possible 0.7% gain at 16384 cubed with
interval 2, within that case's spread. Every move toward less sharing or
less alignment costs 4–80%. So the late-input stalls are not caused by too
much synchronization. Input delivery is limited by the memory system: with
2x2 multicast each CU still needs 32 KB of unique input per K step, about
18 TB/s across 256 CUs at the measured step rate (36 TB/s without
multicast). The next lever is fewer bytes per FLOP from L2: larger multicast
groups (eight or sixteen CTAs; the tutorial supports at most four) or larger
per-CTA tiles. Within one CTA, the remaining costs are LDS operand traffic
and per-step rendezvous, as measured on AM above.

**L2 prefetch (2026-10-22): no gain, reverted.** The experiment issued
`tlx.amd_descriptor_prefetch_tensor` for A, B and both scales of stage
`k + d` in every persistent K step, skipped past the tile's last stage.
`dedup` let only the cluster ranks that own a multicast A (or B) tile
prefetch it, and `speculative` set the speculative hint. With `d = 0`, the
instructions match the committed kernel. Every variant was slower on
hardware, so the source was restored to `HEAD`. The patch, harnesses, logs
and AM results are in
`/home/ibutygin/tlx/perf-data/mxfp-hw-l2pf-20261022.tar.gz` (SHA256
`eb8dfbe05e2729bb3d9a98e73fcb45a1fc2dccce0cc7cadf7e6788c24e801b7d`).

Lowering: each tile is spread over lanes in 256-byte chunks, one
`global_prefetch_b8 ... scope:SCOPE_SE` per lane. That is about 128 L2
requests per 32 KB tile, one instruction per wave per tile, plus about 20
VALU per step for the addresses. Non-speculative prefetches carry
`th:TH_LOAD_NT`.

Correctness pitfall: do not pass `pred=` to prefetches through these tile
descriptors. A masked prefetch is still issued, redirected to the last
element of the descriptor view. The kernel's tile descriptors come from
`update_tensor_descriptor(..., add_offsets=[off_m, 0], clamp_bounds=False)`,
which moves the base but keeps the full `[M, K]` bounds. So that element lies
up to `off_m` rows past the allocation. Non-speculative prefetches faulted
there intermittently (`hipErrorIllegalAddress` for d=2, 8 and 4n, depending
on what memory happened to be mapped). Speculative prefetches never faulted.
Branching on the block-uniform predicate (`if pa:`) fixed it. The lowering is
consistent with the descriptor semantics; the descriptor view was too large.

Hardware, committed defaults plus prefetch, median of three interleaved
`bench.py`-style rounds. Every 8192-sized output was bitwise identical to
d=0:

| Variant | 8192³ | 8192²×4096 | 16384³ |
|---|---:|---:|---:|
| No prefetch | 0.16453 ms | 0.08992 ms | 1.38391 ms |
| d=1, dedup | +23.5% | +20.4% | +24.8% |
| d=2, dedup | +23.2% | +19.6% | +23.6% |
| d=4, dedup | +22.4% | +18.3% | +22.4% |
| d=8, dedup | +20.7% | +15.5% | +21.4% |
| d=2, no dedup | +24.2% | +21.3% | +22.0% |
| d=4, no dedup | +22.3% | +19.7% | +20.1% |
| d=2, dedup, speculative | +23.2% | +19.6% | +23.6% |
| d=4, dedup, speculative | +22.4% | +18.4% | +22.5% |

L2 counters (rocprofv3, one dispatch):

| Shape, variant | GL2C_REQ | GL2C_MISS |
|---|---:|---:|
| 8192³, none | 17.5M | 1.95M (11%) |
| 8192³, d=2 dedup | 35.4M | 1.94M |
| 8192³, d=4 no dedup | 47.9M | 1.93M |
| 16384³, none | 136.5M | 24.7M (18%) |
| 16384³, d=2 dedup | 293.4M | 25.6M |
| 16384³, d=4 no dedup | 394.8M | 24.8M |

Prefetch doubles or triples L2 requests but does not reduce L2 misses. The
misses are the compulsory per-XCD fills, and a prefetch can only move them
earlier. The three-stage ring already overlaps their latency. The extra
requests then compete with TDM on the input path that limits the kernel, at
about 20% for every distance and hint. Dedup barely helps because rank 0
still prefetches both A and B, and the multicast rendezvous waits for the
slowest rank. A balanced split (one tile per rank) might halve the loss but
cannot turn it into a gain while misses stay flat. AM (2048 cubed, where
everything hits in L2) showed the same direction: 26–30% more dispatch
cycles. The AM dedup runs used `pred=`, so the skipped ranks still issued
redirected prefetches.

**Next-tile loads after the C store (2026-10-22): slower, reverted.** With
cross-tile prefetch, the K tail issues the next tile's three input stages
before the epilogue. The tensor counter retires in order, so the fourth C
chunk's slot wait also waits for those nine input loads. The alternative was
`CROSS_TILE_PREFETCH=False`, which issues the next-tile TDM loads after the
C stores. It was tested alone and with an L2 prefetch of the next tile's
three stages at the start of the K tail (`NEXT_TILE_L2_PREFETCH`). That
prefetch used the full-tensor descriptors and uniform branches, one tile per
cluster rank. All variants passed FFM at 2048 cubed, and hardware outputs at
8192 were bitwise identical. Patch and logs:
`/home/ibutygin/tlx/perf-data/mxfp-hw-ntl2-20261022.tar.gz` (SHA256
`c0421b893f235ba682fbc0aafe0da6a6329757a441de77d37f7e82fea53dc202`).

| Variant | AM 2048³ cycles | HW 8192³ | HW 8192²×4096 | HW 16384³ |
|---|---:|---:|---:|---:|
| Cross-tile prefetch (default) | 101,818 | 0.16404 ms | 0.09129 ms | 1.38177 ms |
| Loads after C stores | +6.2% | +1.57% | +2.91% | +0.95% |
| + next-tile L2 prefetch | +8.2% | +1.29% | +2.79% | +1.19% |

On AM, warm transitions lose 6.1–6.3k cycles instead of 4.1–4.2k: the next
tile's first step waits for the full TDM latency, about 2k cycles. With three
C slots, the ATT slot-reuse waits that include input loads are only
0.1–0.8k cycles, so decoupling can save less than the exposed latency costs.
The L2 prefetch recovers at most 0.3%, because most input already hits in
L2. A hybrid (cross-tile TDM for stage 0 only) would be bounded by the same
≤0.8k-cycle wait, under 1% at 8192 cubed.

**K-step schedule and barrier placement (2026-10-22).** Each steady A8W8
step per wave runs four 16-WMMA quadrants and issues 64 `ds_load_b128`:

1. c00 reads `b01` (this slot).
2. c01 reads `a10` (this slot).
3. Release barrier, TDM refill of this slot, `s_wait_tensorcnt` for the
   next stage, acquire barrier.
4. c10 and c11 read `a00`/`b00` of the next slot, in a 34-load burst.

Both barriers are split (signal, about five WMMAs, wait), so on fast steps
they are cheap. Each operand group ends with its scale load, and the next
quadrant's first WMMA needs that scale, so every operand wait was
`s_wait_dscnt 0x0`. Both barriers come from Membar. Because the three ring
slots are selected at runtime, Membar cannot tell them apart, so it also
places a barrier after every TDM refill and before the next LDS read.

Variants, all bitwise identical to the baseline on hardware and passing
FFM where barriers changed. Hardware is the median of three interleaved
`bench.py`-style rounds. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-hw-ksched-20261022.tar.gz` (SHA256
`beac0986add988b8cf41b12301d3d3cce0272552f07a78293fcc47f9009204fa`; it
includes the experiment patch and the asm timeline scripts).

| Variant | AM plateau | 8192³ (static) | 8192²×4096 (dynamic) | 16384³ (static) |
|---|---:|---:|---:|---:|
| Baseline | 0.628 | — | — | — |
| Scales before payload (partial `dscnt` waits) | 0.646 | +4.6% | −1.6 to −1.9% | +4.2% |
| `GROUPED_PREFETCH` (1 WMMA : 2 loads) | 0.603 | +0.1% | −0.5% | +0.8% |
| Scales first + grouped | 0.638 | +3.4% | −2.1% | +3.6% |
| One merged barrier, refill after c10 | 0.584 | +8.9% | — | +7.9% |
| Merged + scales first | 0.627 | +10.0% | −1.9% | +10.1% |
| Early refill (read `b01` and `a10` in c00, refill before c01) | 0.644 | −0.2% | — | +0.1% |
| Static slot unroll at K=4096 | — | — | −0.5% | — |

Conclusions:

- The merged barrier needs relaxed `local_load`s and a refill placed before
  any next-slot read, or Membar adds the refill barrier back. That forced the
  refill one quadrant later, and losing that quadrant of TDM lead time costs
  about 8%. Gaining a quadrant (early refill) changes nothing. The current
  lead is just sufficient, and the full-device static loop is limited by
  input delivery, not by in-CTA scheduling.
- AM (16 workgroups, no memory contention) rewards LDS-overlap changes that
  hardware does not. Scales-first even slows the static loop on hardware,
  likely by keeping more LDS reads in flight, which delays the release drain
  and competes with incoming TDM writes. Do not tune the static loop on AM
  plateau alone.
- Adopted: `DYNAMIC_SCALES_FIRST` (default on) applies scales-first only to
  the three-buffer dynamic-slot deferred-bottom loop (A8W8 K < 8192 in
  `bench.py`). The
  static-loop asm is unchanged. Hardware A/B of the final code: 8192²×4096
  −0.5%, 4096³ −0.4%, 8192²×2048 −1.0%, 16384²×4096 −0.9%, against −1.6 to
  −1.9% at 8192²×4096 in the three earlier rounds. AM: −1.9% dispatch
  cycles. FFM passes. The mxgemm codegen tests fail in the same 125 cases
  as on `HEAD`.
- Static unroll at K=4096 is now −0.5% (it was +1.8% before the peel and
  third C slot), small enough that `bench.py` keeps the K ≥ 8192 threshold.

**Larger multicast clusters (2026-10-22): adopted.** Grid layouts `RxC`
(`CLUSTER_SIZE=R*C` up to 16) generalize the 2x2 cluster. Rank r covers M
tile r % R and N tile r // R. A is multicast to the C ranks with equal
r % R, B to the R consecutive ranks with equal r // R, and clusters tile
each M group as R x C blocks. The default 2x2 assembly is unchanged. FFM
passes 27 new ring-phase cases (4x2, 2x4, 4x4; A8W8 three- and four-stage,
A8W4), and all hardware outputs at 8192 were bitwise identical. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-hw-cluster-20261022.tar.gz` (SHA256
`9075c860aa0d0bd98bfea84d116aa45981915e99f1fb92e4c6ac02488ae50632`).

Unique A/B input per CU per K step: 2x2 32 KB, 4x2/2x4 24 KB, 4x4 16 KB,
8x2/2x8 20 KB. A8W8 hardware A/B, medians of three interleaved rounds, change
against the 2x2 default in the same run:

| Variant | 8192³ | 8192²×4096 | 16384³ | 32768³ |
|---|---:|---:|---:|---:|
| 4x2 | −2.4% | −3.4% | −5.0% | — |
| 2x4 | −2.9% | −1.7% | −1.6% | — |
| 4x4, interval 4 | −4.8 to −7.5% | −4.1 to −4.7% | −7.3 to −7.6% | −4.5% |
| 4x4, interval 8 | −5.8 to −8.5% | −4.4 to −4.9% | −7.3 to −7.6% | −0.8% |
| 4x4, interval 2 | −4.0% | −3.1% | −7.9% | — |
| 4x4, `GROUP_SIZE_M=4` | −4.9% | −4.3% | −6.8% | — |
| 4x4, split cluster barrier | −0.8% | −4.5% | −2.5% | — |
| 8x2 | +12.7% | +6.6% | +28.4% | — |
| 2x8 | +18.5% | +9.0% | +35.0% | — |
| 2x2, four-stage ring | +2.6% | +5.8% | −1.0% | — |
| 4x4, four-stage ring | −5.0 to −7.6% | −2.1 to −3.1% | −11.5% | −11.8% |
| 4x4, four-stage ring, interval 8 | −8.1% | −3.6% | −12.4% | **−12.4%** |
| 4x4, four-stage ring, interval 16 | −8.3% | −3.4% | **−13.0%** | −11.6% |

A8W4 via `bench.py` (two rounds each), 2x2 → 4x4 interval 8: 8192³
0.1329 → 0.1286 ms, 8192²×4096 0.0759 → 0.0735 ms, 16384³ 1.022 → 0.973 ms,
32768³ 8.163 → 7.692 ms (−3.2, −3.2, −4.8, −5.8%; 9150 TFLOPS at 32768³).

Halving the unique bytes per CU is the largest win since the static unroll,
which confirms that input delivery limits the full device. Square sharing
matters: 8x2 and 2x8 move nearly as few bytes as 4x4 but are much slower.
A deeper ring pays off only once multicast is wide (2x2 + 4 stages loses at
K ≤ 8192), and from K=16384 the 4x4 four-stage ring adds 4–5%. `bench.py`
now defaults to 4x4 clusters with interval 8 for both variants, and to the
four-stage A8W8 ring (two C slots, dynamic slots) for K ≥ 16384.

### Bottlenecks after 4x4 clusters (`dc853dbc9d`)

Hardware ATT of the committed `bench.py` defaults (one traced wave, CU 0,
dispatch 1025 after warmup), analyzed with `att_phases.py` (in the archive
below). Every executed instruction is assigned to its innermost loop
(backward branches in `code.json`); the cycles until the next issue are
charged to that instruction's class, so waits and barriers carry their
stall time. "WMMA issue" is time charged to WMMAs and their dependency
gaps; the WMMA floor is 512 cycles per K step. The earlier 2x2 trace is
included for comparison.

| | A8W8 8192³ 2x2 (old) | A8W8 8192³ | A8W8 16384³ (4-stage) | A8W4 8192³ | A8W4 16384³ |
|---|---:|---:|---:|---:|---:|
| Wave WMMA utilization | 0.47 | 0.50 | 0.52 | 0.61 | 0.65 |
| Steady loop share of wave | 82% | 81% | 83% | 74% | 87% |
| Steady step, mean (median) | 954 (934) | 884 (827) | 842 (807) | 687 (683) | 718 (685) |
| … LDS reads: `ds_load` issue + `s_wait_dscnt` | 260 | 265 | 261 | 193 | 197 |
| … workgroup barrier waits | 146 | 61 | 44 | 7 | 20 |
| … cluster barrier waits | 32 | 58 | 56 (separate block) | 0 | 7 (separate block) |
| … tensor (TDM) waits | 49 | 34 | 19 | 2 | 17 |
| … scalar + vector ALU | 29 | 29 | 82 | 45 | 45 |
| Tile loop outside the steady K loop | 14% | 14% | 11% | 25% | 12% |

Per-step values are cycles per 64-WMMA step. 4x4 cut the late-input
barrier waits from 146 to 61 cycles per step at 8192 cubed; the remaining
input waits (workgroup + cluster barrier + TDM, ~150 cycles) now cost about
half as much as LDS operand reads.

1. **LDS operand reads (largest in-loop cost, both variants).** About 260
   cycles per A8W8 step and 195 per A8W4 step go to issuing `ds_load_b128`
   and waiting on them: 28-31% of A8W8 steady time and 27% of A8W4's. The
   A8W8 count did not change with 4x4. AM had already shown −19% for a step
   without LDS reads. Each wave reads 32 KB of operands per BK128 step.
2. **Input arrival and cluster synchronization (A8W8).** ~150 cycles per
   step at 8192 cubed (barrier waits behind late TDM data, plus the 16-CTA
   cluster rendezvous every eight steps, which now costs more than with four
   CTAs). In the four-stage dynamic loop at 16384 cubed the cluster block is
   5.5% of the wave. A8W4 barely waits.
3. **Tile transitions and K tail.** A8W8: 14% of the wave at 8192 cubed,
   11% at 16384 cubed; A8W4: 25% at 8192 cubed (about 13k cycles per tile,
   mostly `s_wait_tensorcnt` on C stores and next-tile loads, plus
   `ds_store` of C), 12% at 16384 cubed.
4. **Scalar overhead in the four-stage A8W8 loop.** 68 SALU cycles per step
   (27 instructions) against 27 in the unrolled three-stage loop; the
   dynamic slot loop carries slot and address arithmetic.

L2 counters (rocprofv3, one warm dispatch, A8W8, bench defaults otherwise):

| | 8192³ 2x2 | 8192³ 4x4 | 16384³ 2x2 | 16384³ 4x4 (4-stage) |
|---|---:|---:|---:|---:|
| GL2C_REQ | 17.5M | 9.4M | 136.6M | 70.4M |
| GL2C_READ_SECTORS | 67.1M | 33.9M | 537M | 269M |
| Hit rate | 89% | 84% | 82% | 71% |
| GL2C_MISS | 1.92M | 1.54M | 24.7M | 20.5M |
| GL2C_EA_RDREQ_DRAM | 7.4M | 4.3M | 160M | 131M |
| GL2C_BUSY / GL2C_CYCLE | 68% | 64% | 94% | 93% |

4x4 halves L2 requests and read sectors as predicted. At 8192 cubed the L2
is busy 64% of the time; at 16384 cubed it stays busy 93% even with 4x4,
and 29% of requests miss (L2 capacity, since each XCD's L2 holds a larger
working set). `GL2C_BUSY` is not a throughput limit, though: the
scales-in-VGPR experiment below ran 2.2x longer at 16384 cubed with only 7%
more requests and still read 93% busy, so the counter tracks outstanding
work. The miss rate is the more useful signal; L2 locality (XCD-aware
ordering of 4x4 clusters, which currently requires `XCD_REMAP=none`) is the
input-side lever left to test. SQ counters (`SQ_INSTS_LDS`, `SQ_WAIT_ANY`) read zero on this
device and were not used.

Archive: `/home/ibutygin/tlx/perf-data/mxfp-hw-prof-20261022.tar.gz` (SHA256 `50ce93e3e0adc630b71b09038598e852d41995a761ba95b423737e488c93c7c9`).

**LDS read scheduling (2026-10-22): no gain, reverted. The loop is LDS
bandwidth bound.** Experiments on the 4x4 defaults. All 8192-sized hardware
outputs were bitwise identical; FFM passed for the riskiest variants. Patch
and logs: `/home/ibutygin/tlx/perf-data/mxfp-hw-lds-20261022.tar.gz` (SHA256
`5dac54f1af9ae0acb80fb29f7c023cc7015ed9bbec104bfaa0f492834db053f3`).

- `STEP_LDS_INTERLEAVE=r`: `sched_group_barrier` groups of r LDS reads per
  WMMA in the quadrants that read operands (c00, c01, and c11 when B moves
  there), leaving the c10 barrier region free.
- `STEP_B_AFTER_C10`: read the next stage's B during c11 instead of c10
  (17 + 17 loads instead of a 34-load burst).
- `FOUR_BUFFER_GROUPED=False`: drop the 1-WMMA:2-read grouping that the
  four-stage loop uses by default.
- Static slot unroll for the four-stage ring (no loop-carried operand
  copies, loop-invariant slots).

| Static A8W8 loop, 4x4 | 8192³ | 16384³ |
|---|---:|---:|
| LLVM schedule (base) | 0.15458 ms | 1.27905 ms |
| r = 1 / 2 / 3 | +0.6 / +0.6 / +11.3% | −0.6 / +7.9 / +11.9% |
| B after C10 | +0.5% | −1.9% |
| r = 2 / 3 + B after C10 | +3.0 / +8.3% | +2.0 / +9.3% |
| r = 2 in all four quadrants | +2.0% | +6.6% |

| Four-stage A8W8 loop (16384³ / 32768³ default) | 16384³ | 32768³ |
|---|---:|---:|
| Grouped, dynamic slots (default) | 1.2140 ms | 9.879 ms |
| Ungrouped | +1.1% | +0.2% |
| Ungrouped + B after C10 | +0.4% | +0.7% |
| Static unroll, grouped | +1.2% (8192³ −0.8%) | +0.7% |
| Static unroll, ungrouped | +1.9% (8192³ +1.0%) | +0.5% |

B after C10 in the three-stage dynamic loop (8192²×4096) was +0.7%. Pinned
interleaves move loads behind WMMAs that still read the destination
registers, so LLVM inserts `s_wait_alu depctr_va_vdst(0)` (a full WMMA drain)
before them; LLVM's own front-loaded order avoids most of those.

Why scheduling cannot help: per 64-WMMA step a CU's LDS takes the TDM writes
plus the 2x2 wave reads. A8W8: 64 KB written + 128 KB read = 192 KB in
884 (8192³) / 842 (16384³) cycles, 217-228 B/clk. A8W4 per 128 K: 48 + 96 =
144 KB in 687 / 718 cycles, 200-210 B/clk. The two variants' step ratio
(0.78) tracks their LDS byte ratio (0.75), and both sit at about the same
LDS throughput, so the steady step is set by LDS bandwidth, not by latency
or ordering. The 2x2 wave layout already minimizes reads for four waves
(4x1 or 1x4 would read 160 KB). Further steady-loop gains need fewer LDS
bytes per FLOP: for example one operand fed to VGPRs without an LDS round
trip, or LDS-write/read reuse that four waves cannot provide today.

**E8M0 scales through VGPRs (2026-10-23): much slower, reverted.** The
static A8W8 step loaded its four 128-row scale subtiles per K step straight
from the preshuffled global scale tensor into the WMMA scale registers
(two `buffer_load_b32` per subtile per lane, D steps ahead in a register
FIFO), keeping TDM scale loads to LDS for the K tail. The payload LDS reads
then end with payload, so quadrant waits became partial without reordering.
This needed a compiler fix: coalescing replaced the load's layout (whose
warps replicate along one axis) with a blocked layout and converted both
ways through LDS. `Coalesce.cpp` now keeps a user-pinned pointer layout
(`tlx.require_layout` on the pointer tensor) for global loads and stores,
like the existing pinned-store case (lit test in
`test/TLX/user-register-layout.mlir`; uncommitted). Kernels were compiled
locally with that fix and launched on the remote host from their code
objects (`hw_ship.py`), since the remote Triton build lacks it.

| | 8192³ | 16384³ | VGPRs / spills |
|---|---:|---:|---:|
| Scales via TDM + LDS (base) | 0.15432 ms | 1.28042 ms | 857 / 0 |
| VGPR scales, 1 step ahead | +9.6% | +72.4% | 861 / 0 |
| VGPR scales, 2 steps ahead | +227% | +226% | 878 / 31 |

Outputs were bitwise identical (FFM and hardware). L2 requests rose only 7%
(9.3M → 10.0M, 73.4M → 78.6M), so the loss is exposed global latency: one
step (~800 cycles) of lead does not cover a load, especially at 16384 cubed
where 28% of L2 requests miss, while TDM loads scales three stages ahead
and multicasts them across the 4x4 cluster. Two steps ahead spills
(register-bank pressure at 878 of 1024 VGPRs). The upside was small in any
case: scales are about 6 KB (2 KB written, 4 KB read) of the 192 KB of LDS
traffic per step. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-hw-vgpr-20261023.tar.gz` (SHA256
`44cf735f99f9152b82b4c9e72b305c66a04c6d0dc1be8561615ba59e8ec2ed3e`).

**Related work in other branches and `../TDM.md` (surveyed 2026-10-23).**
Remote refs as last fetched: `upstream` and `int` (AMD internal
triton-mi450) on 2026-10-23, `my` up to 2026-10-22. SSH fetches from this
host fail (publickey), so newer work may exist.

`/home/ibutygin/tlx/TDM.md` (single-CTA microbenchmarks of the same
BM=BN=256, BK=128 slot, 66 KiB with scales):

- Warm repeated fill 370 ns (183 GB/s per CTA); warm single fill 440 ns;
  cold 840 ns. An L2 prefetch issued 320 ns ahead restores warm fill time
  on an idle device.
- Full-grid TDM streaming 17.5-18.4 TB/s, about 95% of the BabelStream HBM
  read reference (19.3 TB/s).
- Our steady step takes in 66 KiB of LDS per CU in about 420-460 ns
  (145-157 GB/s), close to the warm single-CTA fill rate. This agrees with
  the LDS-bound analysis above. Its GEMM-with-prefetch attempt was also
  about 10% slower, as were our L2 prefetch experiments.

Kernel-relevant work not in this branch (it shares only a 2025-08 merge
base with `upstream/main`):

- **Clustered warp-pipeline GEMM in Gluon**
  (`third_party/amd/python/examples/gluon/gemm_warp_pipeline_cdna5.py`,
  upstream #12073, also on `int/shared/gfx1250`). The closest design to
  ours: 4x4 (or 2x2) CTA clusters with TDM multicast, but eight waves per
  CTA in a warp pipeline. MXFP8 uses BK256 with 256x256 CTA tiles (two
  data, three scale slots) or BK512 with 64/128 tiles, a five-stage
  pipeline and six-tile unroll; optional two-buffer output epilogue and a
  fully unrolled `--flat-pipeline`. No performance numbers are recorded.
  With eight waves on a 256x256 tile each operand is read from LDS more
  often than with our 2x2 four-wave layout, so it must win on latency
  hiding. Benchmark it against our 7050 TFLOPS (8192 cubed); this needs a
  Triton build from `int/shared/gfx1250` or upstream.
- **Gluon MXFP GEMM** (`mxfp_gemm_cdna5.py`; multi-CTA #12119, CTA-local
  slicing #12143). Clusters come from a CGA layout on the cluster tile
  rather than per-CTA masks. It has a sliceMNK variant that splits A and B
  into two TDM loads each, and an option to load scales with async copies
  instead of TDM.
- **GL2 residency hints** (`int/danherna/gfx1250-cache-residency`):
  `eviction_policy="evict_last"` on TDM load, store, fused-load and
  prefetch, buffer ops and async copies. It only helps with persisting L2
  ways reserved via
  `hipDeviceSetLimit(hipLimitPersistingL2CacheSize, bytes)`. Relevant to the
  28% L2 miss rate at 16384 cubed: FP32 C stores (1 GiB at 16384 cubed) may
  evict reused A/B, so try streaming C stores or residency hints on A/B.
- **Newer AMD codegen**: upstream #12471 builds it at LLVM `05e432a8`
  (llvm/llvm-project); we pin `d26ff26e`. No `05e432a8` build exists
  locally or on the remote host.
- **Barrier placement**: upstream #12342 always inserts a workgroup barrier
  before a cluster barrier; #12423 schedules async-wait barriers before
  potential global-memory waits. Check against our cluster rendezvous
  sequence after a rebase.
- Smaller items: `int/saananth/fused-tdm-offsets-api` exposes counted
  instruction-group scheduling hints and an LDS-DMA group mask in Gluon;
  `int/qian/lds-prefetch-dot-scaled` adds LDS prefetch for scaled dot to
  the compiler pipeliner (not Gluon); `int/qian/mxfp_gemm_slice` documents
  that slicing only helps by avoiding register spills and is neutral at
  matched pipeline depth (fp16 x fp8); upstream #12315 adds native fp4/e8m0
  types. `my/*` branches carry TLX work (cluster launches, WMMA queuing
  option, automatic wave specialization, grouped-GEMM API), no Gluon.

**Cache hints on C stores (2026-10-23): no effect.** `tlx.async_amd_descriptor_store`
gained `cache_modifier=` (passed to `amdgpu.async_tdm_copy_local_to_global`,
whose lowering already mapped it to gfx1250 cache-policy bits; TLX builder
binding updated, uncommitted). A temporary `C_STORE_CACHE` kernel constexpr
applied it to every staged C store. `.cs` emits `tensor_store_from_lds ...
th:TH_STORE_NT`; `.wt` emits `th:TH_STORE_BYPASS scope:SCOPE_SYS`. Kernels
were compiled locally and launched remotely (`hw_ship.py`). Bench defaults;
outputs bitwise identical at 8192 cubed:

| A8W8 | 8192³ | 16384³ | 32768³ | GL2C_MISS 8192³ / 16384³ |
|---|---:|---:|---:|---:|
| No hint | 0.15649 ms | 1.21488 ms | 9.85818 ms | 1.54M / 20.49M |
| `.cs` (non-temporal) | +0.09% | +0.01% | +0.04% | 1.54M / 20.48M |
| `.wt` (bypass, system scope) | +0.03% | +0.03% | +0.11% | 1.55M / 20.48M |

Even bypassing the cache for the 1 GiB FP32 output at 16384 cubed leaves
L2 misses and DRAM read requests unchanged, so the 28% miss rate is not C
stores evicting A/B; it comes from the input working set itself (L2
capacity and ordering). Residency hints on A/B, XCD-aware cluster ordering,
or larger reuse windows are the remaining levers there. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-hw-cstore-20261023.tar.gz` (SHA256
`b6637349816479000e2a5715bfd1fecce541c2a73b23aedefcdcbf059a32361e`).

**XCD remapping of 4x4 clusters (2026-10-23): adopted.** Grid layouts now
accept the persistent XCD remaps; `_mxgemm_cluster_program_id` applies them
to whole clusters (any bijection on cluster ids keeps multicast partners
together). With 256 programs there are 16 clusters per wave of tiles, and
`chunked` (eight XCDs, chunk two) and `balanced` give the same permutation.
FFM tile-coverage tests pass for every grid layout with 16 clusters and the
chunked remap; all 8192-sized hardware outputs were bitwise identical.

A diagnostic probe could not read the XCD: on gfx1250 the assembler rejects
`HW_REG_XCC_ID`, `HW_REG_HW_ID1` only carries WGP/SA bits (17 distinct
values, SE bits zero), and `HW_REG_HW_ID2` and `ttmp8` are constant across
programs. The results fit clusters dispatched round-robin (XCD = cluster %
8): without remapping each XCD's two clusters share A rows and need two B
column blocks; chunked remapping gives them the same B column block, and
successive persistent waves on an XCD keep the same A row pair.

`hw_ab.py`, A8W8 bench defaults (four-stage ring from K=16384):

| Variant | 8192³ | 16384³ | 32768³ |
|---|---:|---:|---:|
| No remap | 0.15636 ms | 1.21584 ms | 9.87164 ms |
| Chunked, 8 XCDs x 2 | −1.75% | −2.72% | −3.44% |
| Chunked, 4 XCDs x 4 | +3.50% | +3.42% | +1.57% |
| `GROUP_SIZE_M=4` | −1.78% | +0.61% | +0.10% |
| `GROUP_SIZE_M=4` + chunked 8x2 | +3.40% | +0.77% | +0.28% |

`GROUP_SIZE_M=4` matches the chunked remap at 8192 cubed (where it also
puts B-sharing clusters on one XCD) but not at larger N. `bench.py`, two
alternating rounds per configuration, chunked vs none: MX8xMX8 −2.0%
(8192³), −0.4% (8192²×4096), −2.6% (16384³), −3.6% (32768³); MX8xMX4
−0.8%, −0.6%, −1.8%, −2.1%. `bench.py` now defaults to `--xcd-remap
chunked` for 4x4 clusters. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-hw-xcd-20261023.tar.gz` (SHA256
`3583b8e0113f0329ac799b15975516664e73ab9665f6c5d02e431ccaaf42e4c4`).

**CDNA5 LDS partition conflicts (2026-10-23): real; 64-row partition
pieces adopted for the static-slot loop.** gfx1250 LDS is five 64 KiB
partitions (`getSharedMemoryPartitionSize`; partition = byte offset / 64 KiB).
AM counts concurrent accesses to one partition in
`tcp0.cu_cache_lds_lds_partition_{read,write}_conflicts`
(`am-runs/mxfp-waits-20261021/lds_conflicts.py` sums them; DS bank conflicts
are zero). Each A/B ring slot is one contiguous 34,816-byte block, so a
quadrant's 128-row operand subtile sits in one partition, and all four waves
read it together: with the native WMMA layout, `wn = 0` waves read rows
0–31/64–95 and `wn = 1` rows 32–63/96–127 of each B subtile (A by `wm`).
Hardware exposes no LDS conflict counter (`SQ_*LDS*` read 0 on this driver), so
hardware evidence comes from ATT.

`LDS_PARTITIONED_AB = r` alternates r-row pieces of A and B between two
partitions. Fused TDM members are issued by two waves, so an ordinary 2D
partitioned destination with 32-row pieces needs four TDM instructions per
member, five per stage instead of two. The pinned backend inserts
`s_wait_tensorcnt 0xa` before any TDM issue with more than ten outstanding,
which collapses a three- or four-stage ring (hardware: +9.6% at 8192³, +8.5%
at 8192²×4096, +23% at 16384³; still +1–3% with two buffers). The kept
version instead views A/B as group-major `[rows / 2r, 2r, K]` descriptors and
`[G, 2r, BK]` buffers partitioned along dim 1 with one group, so each issuing
wave fills its partition with one 3D TDM instruction (same instruction and
wait counts as the baseline). Reads reshape the slot to `[256, BK]`; this
needed a `memdesc_reshape` rule for partitioned layouts (`Ops.cpp`, lit tests in
`test/TritonGPU/{ops,invalid}.mlir`). Previously the reshape fell back to a
shared-linear encoding that silently dropped the partition bases.

| A8W8 | Base | 32-row pieces | 64-row | 128-row |
|---|---:|---:|---:|---:|
| AM 2x2 2048³: read conflicts / LDS issue | 0.881 | 0.300 | 0.740 | — |
| AM: avg LDS latency (cycles) | 123 | 111 | 119 | — |
| AM: cycles (plateau) | 101,818 (0.628) | 99,011 (0.656) | 101,875 (0.635) | — |
| HW 8192³ (static loop) | 0.1528 ms | +3.6% | **−1.6%** | 0.0% |
| HW 8192²×4096 (dynamic, 3 stages) | 0.0847 ms | −1.1% | +0.8% | — |
| HW 16384³ (dynamic, 4 stages) | 1.1776 ms | +1.4% | +0.3% | +0.4% |
| HW 32768³ (dynamic, 4 stages) | 9.444 ms | +1.1% | −0.1% | — |

Hardware timings are `hw_ship.py` medians (bench-style warm timing), 64-row
over three sessions (−1.69/−1.55/−1.75% at 8192³). All 8192-sized outputs
were bitwise identical; FFM passes the 4x4 three-stage static and four-stage
dynamic cases and the 55 ring-phase/grid-cluster pytest cases; AM 2x2 passes.

Hardware ATT at 8192³ (one wave, CU 0), cycles per steady step:

| | Base | 32-row | 64-row |
|---|---:|---:|---:|
| LDS reads (`ds_load` + `s_wait_dscnt`) | 264 | 181 | 231 |
| Workgroup barrier | 82 | 48 | 83 |
| Cluster barrier | 27 | 107 | 80 |
| Tensor waits | 30 | 44 | 23 |
| Step mean (median) | 868 (848) | 843 (795) | 879 (825) |

So the conflicts are real on hardware: removing most of them cuts this wave's
LDS read cost by 31%, yet the kernel is slower. 128-row pieces cost nothing.
64-row pieces remove fewer conflicts but help the static loop; the dynamic
loops do not gain. `LDS_PARTITIONED_AB=None` therefore selects 64 rows with
`STATIC_SLOT_UNROLL` (the 8192³ bench default) and 0 otherwise. The first
explanation, that 32-row pieces lose because each TDM instruction fetches
four 4 KiB chunks, does not hold; see the next paragraph.

**Contiguous wave rows (2026-10-23): fewer conflicts, slower; not kept.**
A temporary `MXFP_WAVE_CONTIG` switch pinned WMMA warp bases `(0,4),(4,0)`
and register bases `(0,1),(1,0),(0,2),(2,0)`, so each wave pair reads one
contiguous 64-row half of every 128-row subtile. The E8M0 scale pins changed
to match (wave stride 256, second value stride 128), and C staging split on
row bit 32 into `[2, 32, 128]` slots stored with a 3D TDM descriptor (row
bit 64 now selects the wave pair). The steady loop keeps the same LDS, WMMA,
and barrier counts with no layout conversions. FFM 4x4 and AM 2x2 pass; all
8192³ hardware outputs were bitwise identical.

| A8W8 8192³ static loop | Interleaved, no pieces | Interleaved, 64-row (default) | Contiguous, no pieces | Contiguous, 64-row |
|---|---:|---:|---:|---:|
| AM read conflicts / LDS issue | 0.881 | 0.740 | 0.856 | 0.302 |
| AM cycles (plateau) | 101,818 (0.628) | 101,875 (0.635) | 101,180 (0.612) | 99,698 (0.656) |
| HW time, two passes | 0.1530 ms | −1.48 / −1.62% | −0.94 / −1.04% | +3.37 / +3.23% |
| HW shader clock during timing | 1,813 MHz | 1,795 MHz | 1,804 MHz | 1,840 MHz |
| ATT, 16 SEs: wave cycles median (max) | 258.8k (264.3k) | 249.2k (255.6k) | — | 260.8k (284.8k) |
| … steady step: LDS reads | 279 | 233 | — | 184 |
| … workgroup barrier | 63 | 73 | — | 134 |
| … cluster barrier / tensor waits | 34 / 18 | 28 / 31 | — | 41 / 35 |

Contiguous rows with 64-row pieces reach the 32-row conflict level in AM and
fetch exactly like the default (two 8 KiB chunks), yet lose as much as
32-row pieces on hardware. Every variant runs at the ~2,480 W power cap; the
slow one runs at a higher clock, so the loss is not throttling but more
cycles. Single-wave ATT on CU 0 was misleading (its wave was the fastest of
all); one wave per shader engine shows the cause: the LDS read savings come
with doubled workgroup-barrier waits and a slow tail (max 284.8k cycles).
The two partition halves are not equally loaded. Per slot, A's rows 0–63 sit
in P1, P1/P2, P2 and rows 64–127 in P3; B's rows 0–63 in P2, P2, P2/P3 and
rows 64–127 in P4; C slots and scales occupy P0–P1. When each wave pair
reads only one half, wave (0,0) reads A and B from the crowded P1/P2 region
(also hit by C traffic) while wave (1,1) reads from P3/P4 alone, so the waves
finish their reads at different times and wait at every barrier. The
interleaved 32-row-piece layout also gives each pair one half, which explains
its loss the same way; 64-row pieces with interleaved rows give every wave
both halves and stay balanced. A balanced split would need each wave pair to
see equally loaded partitions, for example A halves and B halves in four
distinct partitions with C and scales in the fifth. The allocator does not
let the kernel choose partition offsets; the one partition buffer per
alloc fits (52,224 B per 64 KiB partition), but C's 96 KiB would have to
share the leftover space. Archive (patch, AM summaries, clock samples,
16-SE ATT bundles): `/home/ibutygin/tlx/perf-data/mxfp-wave-contig-20261023.tar.gz`
(SHA256 `c9482bcd54be8a3855c47c05ea73189387a5f6ed68a67d2fe58c91299ab98a39`).

**Upstream partition-aware wave layouts (2026-10-23): no gain; not
kept.** Upstream Gluon solves the same problem with
`gl.amd.cdna5.make_partitioned_dot_layouts` (#10145, #10809, #10334, #10911;
moved to `cdna5` in #11183; the `mxfp_gemm_cdna5.py` example enables it with
`RESOLVE_PARTITION_CONFLICTS`). For our 128x128 dot slice of a 256-row
block it returns exactly our 64-row partitioned A/B layout plus a swizzled
WMMA layout: warp bases `(4,4),(2,0)`, register bases
`(0,1),(0,2),(1,0),(4,0)` (16x16 tiles, (M, N)). Warp bit 0 is XORed into
the row group, so the two wave pairs read opposite A partition halves at
every register step and each wave still reads both halves; each pair reads
one B half. Gluon pins the layouts explicitly. TLX's `dot_scaled` still runs
`TritonAMDGPUAccelerateMatmul`, which crashed on a swizzled pinned WMMA: it
built the scale encoding with `LinearEncodingAttr::get` and checked
`tiles_per_warp` via `getWarpsPerCTA`; both reject non-permutation
layouts. Two small fixes (use `inferEncodingFromLinearLayout`; skip the
check for a swizzled pinned WMMA) are in the archive patch, not in the tree.
TLX `shape/stride` scale pins express the XOR by giving a warp bit and a
register bit the same stride. C staging split on row bit 16 into
`[4, 16, 128]` slots with a 3D TDM store.

Variants on the 8192³ static loop (all with 64-row pieces): `swz1` = upstream
layout; `swz2` = A and B both swizzled (warps `(4,2),(2,4)`, registers
`(0,1),(1,0),(4,0),(0,4)`), so every wave reads both halves of both operands.
FFM 4x4 and AM 2x2 pass; hardware outputs bitwise identical.

| | Plain warps (default) | `swz1` (upstream) | `swz2` |
|---|---:|---:|---:|
| AM read conflicts / LDS issue | 0.740 | 0.359 | 0.433 |
| AM cycles (plateau) | 101,875 (0.635) | 100,905 (0.651) | 102,612 (0.625) |
| Steady-loop `ds_load` per 3 steps | 204 | 210 | 216 |
| HW vs no pieces, two passes | −1.66 / −1.69% | +0.01 / −0.05% | +1.21 / +1.18% |
| HW shader clock (sustained) | 1,795 MHz | 1,806 MHz | 1,814 MHz |
| 16-SE ATT wave cycles median (max) | 249.2k (255.6k) | 267.0k (287.8k) | 245.1k (260.2k) |
| … LDS reads / wg barrier per step | 233 / 73 | 180 / 154 | 208 / 80 |

`swz1` removes conflicts but keeps the B-side imbalance (each pair owns one
B half), so workgroup-barrier waits double as with contiguous rows. `swz2`
is balanced and has the best traced step, yet loses in sustained timing.
Its swizzled scale layouts split some scale reads (+6% LDS instructions).
ATT runs are not a reliable ranking here: their eight-launch bursts run
below the power cap (kernel 133–145 µs) while sustained timing sits at
~2,480 W (~150 µs, 1.8 GHz), and relative stall costs shift with the clock.
Use sustained `hw_ship.py`/`bench.py` timing as ground truth. Archive
(patches, AM summaries, clock samples, ATT bundles):
`/home/ibutygin/tlx/perf-data/mxfp-wave-swizzle-20261023.tar.gz` (SHA256
`574c63d0ec5e19b9d3f5dd3f3590a92c5e34036ea059731a2d46a8881b559c8a`).

**Scale reads under the swizzled layouts (2026-10-23): inconclusive.** The
swizzled variants' extra LDS loads are all E8M0 scale reads (operand loads
stay 192 `ds_load_b128` per three steps). Each lane needs rows r and r+64;
the plain layout reads both with one `ds_load_2addr_b32`, but with the row
group XORed by a warp bit the register that holds row r depends on the wave,
the two offsets are no longer constant, and each read splits: `swz1`
12 `ds_load_b32` + 6 `ds_load_b64`, `swz2` 24 `ds_load_b32` (17 instead of
11 `s_wait_dscnt`). An upper-bound build replaced the per-step scale reads
with constant E8M0 scales (TDM scale loads kept; outputs equal across
variants):

| 8192³ cycles (time × sustained clock) | No pieces | 64-row (default) | `swz1` | `swz2` |
|---|---:|---:|---:|---:|
| Real scales | 277k | 271k | 277k | 281k |
| Constant scales | 343k | 325k | 284k | 344k |

Removing scale reads made every kernel 20–27% *slower* in cycles: LLVM
rescheduled the loop (29 instead of 11 `s_wait_dscnt` per three steps,
exposing LDS waits) and the lighter power draw raised the clock ~9%. So this
is not an upper bound. In that exposed-wait regime `swz1` needs 13% fewer
cycles than the default with identical LDS instructions, which shows its
operand pattern does relieve partition stalls, but not what it would gain in
the real schedule. Recovering paired scale reads would need the scale layout
to carry the swizzled row bit in a lane bit instead of a register bit (per-lane
addresses are free; register assignment is static), i.e. a different
`chooseScaledWmmaScaleLayout` choice and matching opsel in the scaled-WMMA
lowering. Archive: `/home/ibutygin/tlx/perf-data/mxfp-scale-reads-20261023.tar.gz`
(SHA256 `782bbbc8eb272449e38a0fd0195a77bed939d2685ee96ccc4ab1f6e271ba35f3`).

**LDS partition conflicts: what we learned (2026-10-23).**

| Experiment | Conflicts (AM, per LDS read) | 8192³ hardware | Status |
|---|---:|---:|---|
| Baseline (contiguous A/B slots) | 0.88 | 0.1530 ms | — |
| 32-row pieces, 2D, 4 TDM instr/member | 0.32 | +9.6% | rejected (TDM cap) |
| 32-row pieces, group-major 3D TDM | 0.30 | +3.6% | rejected (imbalance) |
| **64-row pieces, group-major 3D TDM** | 0.74 | **−1.6%** | **default (static loop)** |
| 128-row pieces | — | 0.0% | — |
| Contiguous wave rows + 64-row pieces | 0.30 | +3.3% | rejected (imbalance) |
| Contiguous wave rows, no pieces | 0.86 | −1.0% | not kept (default is better) |
| Upstream swizzled waves (`swz1`) + 64-row | 0.36 | +0.0% | rejected (B imbalance, split scale reads) |
| A+B swizzled waves (`swz2`) + 64-row | 0.43 | +1.2% | rejected (split scale reads) |

1. Conflicts are real on hardware, not just in AM: removing most of them
   cuts traced LDS read cost per step by about a third (279 → ~180 cycles).
2. Reducing conflicts is not enough. What matters is the per-wave balance
   of partition load. When each wave pair reads only one partition half, the
   halves are unequally shared with other buffers (A, B, C slots and scales
   in five 64 KiB partitions that the allocator places), the waves finish
   their reads at different times, and workgroup-barrier waits double.
   Interleaved 64-row pieces give every wave both halves and win.
3. The pinned backend waits before any TDM issue with more than ten
   outstanding (`s_wait_tensorcnt 0xa`), which caps instructions per stage
   times ring depth. Splitting a fused member into several instructions
   collapses the ring; group-major 3D descriptors keep one instruction per
   wave.
4. Swizzled (XOR) wave layouts, as in upstream
   `make_partitioned_dot_layouts`, balance A (and B with `swz2`), but break
   the paired E8M0 scale reads: the register holding row r depends on the
   wave, so `ds_load_2addr_b32` splits. TLX also needed two
   `AccelerateAMDMatmul` fixes to compile them (archived patch).
5. Measurement: single-CU ATT can be unrepresentative (the CU-0 wave of a
   losing variant was the fastest of all); trace one wave per shader engine.
   Short ATT bursts run below the 2,480 W power cap (~135 µs vs ~150 µs per
   kernel) and can rank variants differently; sustained timing is ground
   truth. Record clocks: under the cap, data and instruction mix change the
   clock (constant scales: +9%), so compare cycles = time × clock too.
   Experiments that remove work (constant scales) can change the compiler's
   schedule and are not valid upper bounds.
6. AM fails correctness for 4x4 multicast clusters even for the baseline;
   use 2x2 clusters for AM conflict counts.

Open directions, in rough order of expected value:
- Carry the swizzled row bit in a lane bit of the scale layout
  (`chooseScaledWmmaScaleLayout` plus scaled-WMMA opsel) so swizzled waves
  keep paired scale reads; then retest `swz1`/`swz2`. In a wait-exposed build
  `swz1` needed 13% fewer cycles than the default with identical LDS loads.
- Control partition placement (explicit LDS offsets or allocator support)
  so A halves, B halves, and C/scales sit in separate partitions; this would
  also allow per-slot rotation of heavy/light partitions in the static
  unroll.
- Re-evaluate 64-row pieces in the dynamic loops (8192²×4096: +0.8%,
  16384³: +0.3%) after either of the above.

Note also that AM fails correctness for every 4x4 (16-CTA multicast)
configuration, including the baseline, while FFM and hardware pass; use 2x2
clusters for AM. Archive of the partition experiments (AM summaries,
hardware logs and ATT bundles, patches of the rejected fused
multi-instruction lowering):
`/home/ibutygin/tlx/perf-data/mxfp-lds-partition-20261023.tar.gz` (SHA256
`dee1dd2846e4a4114575b69c7bcf7a91fb62ae1de104580475c9711cd0170bdd`).

Pre-existing post-rebase breakage found during validation (not fixed):

- `SCHED_MODE_2=True` no longer compiles. The kernel calls
  `tlx.amd_set_wave_sched_mode`, which the pre-rebase branch defined in
  `third_party/tlx/language/tlx/utility.py`. Upstream `33856162ca` replaced it
  with the `enable_wmma_queuing` kernel option, so 44 codegen and 12 FFM
  tests fail. The default is off.
- `test_gfx1250_mxgemm_wave_specialized_handoff_separates_queues` fails in
  all six cases on its tensor-wait-count assertions, with the unmodified
  tutorial source and with both the Meta and the pinned codegen.

### A8W8 8192³ profile and tile transition (2026-10-24)

Committed default after `c1086cc8e2` (static slot unroll, 64-row LDS
partition pieces, three C slots, 4x4 clusters, chunked XCD remap), compiled
locally and launched with `hw_ship.py`; the binary is identical to the
`p64` kernel of the partition experiments. Archive (kernel, scripts, clock
samples, ATT bundles, per-wave phase reports):
`/home/ibutygin/tlx/perf-data/mxfp-a8w8-8k-profile-20261024.tar.gz` (SHA256
`00cf6e3a22f220e09ad06c93d31ef8548d3480186ea6b04f83838c14971034f2`).

**Sustained: 0.1502 ms (7,320 TFLOPS) at 1,800 MHz and 2,477 W**, i.e. at
the ~2,480 W power cap (unprofiled bursts and profiled runs reach 2,419 MHz).
That is about 270k cycles per kernel, 49% of the per-SIMD WMMA cycle floor.
The power cap is the top-level limiter: removing stalls raises power and
lowers the clock, which ate the cycle savings of the recent layout
experiments. A change has to save energy, or at least not add work, to show
in wall time. Under the cap the clock also depends on the data (constant
E8M0 scales: ~1.96 GHz), so `hw_ship.py` and `bench.py` inputs may land at
different clocks.

ATT, one fully decoded wave per shader engine (16 waves), dispatch 64 and
dispatch 16,000 of back-to-back launches. Under the profiler the device
stays at ~995 W and 2.4 GHz (dispatches are serialized), so these show where
cycles go, not how the power cap reshapes them. Medians:

| | Dispatch 64 | Dispatch 16,000 |
|---|---:|---:|
| Wave cycles (max) | 249,130 (256,695) | 247,489 (254,492) |
| Steady K loop share | 79% | 81% |
| Steady step (WMMA floor 512) | 823 | 830 |
| … LDS reads (`ds_load` + `s_wait_dscnt`) | 228 | 226 |
| … workgroup barrier (range over waves) | 56 (33–105) | 72 |
| … cluster barrier | 36 | 33 |
| … TDM waits | 38 | 34 |
| … scalar ALU | 31 | 31 |
| Tile transition and K tail | ~17% | ~15% |

**Tile transition.** Counting from the last steady-loop instruction of one
tile to the first of the next (K tail, epilogue, peeled first step; 96
transitions over both traces), a transition costs a median 9.7k cycles,
about 12% of the wave. By instruction class: `s_wait_tensorcnt` 3.3k,
`s_wait_dscnt` 2.1k, WMMA issue 1.7k, workgroup barriers 1.0k, scalar ALU
0.7k, `ds_load` 0.7k, `ds_store` 0.4k, cluster barrier 0.4k. One typical
transition (8.8k cycles):

| Phase | Cycles | Content |
|---|---:|---|
| K tail (3 steps) | ~3,200 | ~1,060 per step vs 823 steady. Each step issues next-tile loads and drains LDS fully (`s_wait_dscnt 0x0` ~245), copies operands (`v_dual_mov_b32` ×32, ~60); the first step also waits ~280 at the cluster barrier |
| C epilogue (8 × 64x128 FP32 chunks) | ~4,250 | per chunk: `ds_store` → drain ~120 → barrier ~20 → TDM store; slot reuse waits 608 / 531 / 194 cycles (chunks 4, 7, 8), one 446-cycle barrier where waves wait for the slowest wave's slot wait |
| Next-tile start (peeled step) | ~1,100 | includes a 346-cycle LDS wait for the first operands |

The epilogue is bound by C-store latency, not by next-tile loads: a slot
frees 1.2–1.5k cycles after its TDM store regardless of position, and the
next-tile loads issued during the tail finished 1–3k cycles earlier. With
three 32 KB slots, eight stores need about 8/3 × 1.3k ≈ 3.5k cycles
(observed ~3.7k), during which the matrix unit idles.

**Next-tile address work** is already mostly off the critical path: the
next tile's A/B TDM loads and their descriptors are issued in the K tail
(cross-tile prefetch), the eight C-store descriptors take ~35 cycles before
the first store, and each later store needs one scalar add. What remains
after the last C store and before the next tile's first LDS read is 39
scalar instructions (~120 cycles): the ring-phase update `phase = (phase +
k_iters) % 3` (emitted as a division, although `k_iters` is a compile-time
constant), the loop test and `tile = next_tile`, the static-unroll
`slot1`/`slot2` selects, the peeled step's LDS operand addresses, and the
descriptor position of its refill load. None depends on the C stores; the
`amd_sched_barrier` fences around the epilogue keep LLVM from moving them.
Hoisting them in source to just before the first C-slot reuse wait would
hide them in that stall, but the gain (~0.1–0.2k cycles per transition,
about 0.1% of the kernel) is below timing noise; fold it into a larger
transition change rather than testing it alone.

Transition ideas, roughly by expected value per effort (1–3 measured below:
no net gain; 3 saves ~1.5k cycles per transition but slows the steady loop):

1. **Static K tail.** Give the three tail steps the static-slot treatment of
   the steady loop: no full `s_wait_dscnt 0x0` drains, no 32-register copy
   blocks, less control flow. Bounded by ~1.1k cycles per transition
   (about 1% of the wave).
2. **Batch the first three chunks.** Chunks 1–3 have free slots; write all
   three with one LDS drain and one fill barrier instead of three
   (~250–300 cycles per transition).
3. **Overlap the peeled step with the remaining C stores.** After a
   quadrant's chunks are in LDS, its accumulators are free; the peeled step
   sees zero accumulators, so its WMMAs for that quadrant can run while the
   later chunks wait for slots. A deeper version computes the first
   prefetched stages quadrant by quadrant (re-reading A/B subtiles from LDS
   instead of reusing them in registers). Largest potential (up to ~3k
   cycles per transition, the idle matrix-unit time of the epilogue), and
   the largest rework; VGPRs (859 used) allow only one tile's accumulators.
4. **Even out the slot waits across waves.** Each wave's tensor counter
   includes its own next-tile loads (A on waves 0/2, B on 1/3), so slot
   waits differ by wave and the others wait at the following barrier
   (446 cycles in the traced transition).
5. Hoist the next-tile address work above (small; combine with 1–3).

**Static K tail and batched first C fill (2026-10-24): no gain; not
kept.** Temporary kernel constexprs: `STATIC_TAIL` replaced the compact
`tl.range` tail with three unrolled steps that use the steady loop's static
`slots` (refilling each freed slot with the next tile's stage j);
`BATCH_C_FILL` wrote chunks 0–2 (both halves of C00 and the top of C01)
before issuing their three TDM stores, so the compiler inserts one drain and
fill barrier instead of three; the C01 store then starts at its second half.
Static tail: no `v_dual_mov_b32` copies left, 20 instead of 16
`tensor_load_to_lds`, 881 instead of 859 VGPRs, 2,272 instead of 2,046
instructions. Batched fill: two fewer barriers and drains. FFM passed 4x4 at
2048×1024×2048 and unclustered uneven tile counts (1280×512, 3 programs,
K = 1152 and 1536); AM 2x2 passed; hardware outputs bitwise identical.

| 8192³ hardware (three passes) | vs default |
|---|---:|
| Static tail | +0.21 / +0.21 / +0.23% |
| Batched first fill | +0.43 / +0.44 / +0.39% |
| Both | +0.14 / +0.11 / +0.15% |
| AM 2x2 2048³ cycles: default / tail / batch / both | 101,875 / 105,778 / 103,519 / 106,835 |

ATT (16 SEs, dispatch 64), per transition (median gap; class totals):

| Cycles per transition | Default | Static tail | Batched fill |
|---|---:|---:|---:|
| Transition gap (median) | 10,137 | 9,926 | 10,155 |
| `s_wait_dscnt` | 2,104 | 1,635 | 1,769 |
| Workgroup barrier | 904 | 716 | 495 |
| Scalar + vector ALU | 946 | 492 | 947 |
| Cluster barrier | 320 | 648 | 368 |
| `s_wait_tensorcnt` | 3,059 | 3,114 | 3,642 |

Removing local work turns into waiting elsewhere. The static tail removes
~900 cycles of drains, copies, and ALU, but ~330 return as cluster-barrier
waits (the tail reaches the 16-CTA rendezvous earlier and waits for slower
peers); the net ~200 cycles do not pay for the larger code (an extra long
cold-start dip in AM, instruction-cache misses at tile starts) and 22 more
VGPRs. The batched fill saves the intended ~750 cycles of drains and
barriers, but the first C store now issues only after three chunks are
written, so the store-latency chain starts later and slot waits grow by
~580. The transition is paced by C-store latency and by cluster peers, not
by per-chunk overhead; useful changes must add overlap (idea 3) or issue
the first stores earlier, not trim local work. Archive (patch, scripts, AM
summaries, ATT bundles):
`/home/ibutygin/tlx/perf-data/mxfp-tile-transition-20261024.tar.gz` (SHA256
`12a4db112c83da643e1474550ed166df1c4dd0ca9d849cec015249b70da0a86d`).

**Next-tile peeled step overlapped with C stores (2026-10-24): transitions
−14%, steady loop +4%; not kept.** Temporary `OVERLAP_PEEL` (static-slot
path): the first tile's zero-accumulator step runs before the tile loop;
every later tile's runs inside the previous epilogue. Stage-0 operands of
the next tile are read once its prefetch lands, and after each quadrant's two
C chunks are written its registers take that quadrant's next-tile dot
(store C00 → dot C00′ → store C01 → dot C01′ → …), so the 16 WMMAs per
quadrant run during the following slot-reuse waits. After the last store the
stage-3 refill is issued and stage 1 acquired with `wait(12)` (stage 2, eight
C stores, and the refill are younger). The tile loop then starts at step 1
and carries the accumulators. Variants: `=1` also carries the step-1
operands; `=2` carries only the accumulators and reloads step-1 operands at
the loop top; `=3` additionally loads each next-tile dot's operands right
before it. The asm interleaves as intended (C00′/C01′/C10′ WMMAs precede the
chunk 3/6/8 slot waits). FFM passed 4x4 at 2048×1024×2048 and unclustered
1280×512 with K = 512, 1152, 1536; AM 2x2 passed; hardware outputs bitwise
identical.

| 8192³ | Default | `=1` | `=2` |
|---|---:|---:|---:|
| HW vs default (three passes) | — | +1.09 / +0.98 / +1.06% | +0.91 / +0.93 / +0.88% |
| VGPRs; steady-loop instructions | 859; 702 | 875; 714 | 859; 706 |
| ATT transition gap (median) | 10,587 | 10,218 | **9,092** |
| ATT steady step (LDS, wg barrier) | 824 (231, 64) | 874 (249, 72) | 858 (234, 66) |
| ATT wave cycles (median) | 247,217 | 260,618 | 252,717 |

The overlap itself works: `=2` shortens each transition by ~1.5k cycles
(about 4.5k per wave). But carrying the accumulators across the tile loop
changes LLVM's register assignment for the steady loop, which now
recomputes six LDS operand addresses in one of its three unrolled steps
(`v_add`/`v_dual_add` from slot SGPRs plus per-lane VGPRs, three `v_nop`, an
extra `depctr_vm_vsrc` wait): the "per-step VALU address add makes the
following `ds_load` wait for in-flight WMMAs" pattern the static unroll was
built to avoid. That costs ~34 cycles per step (~6k per wave) and the net is
+0.9%. (Later shown not to be the cause; see "Why `OVERLAP_PEEL` was
slower".) `=3` (shorter operand lifetimes in the epilogue) produces the same
steady loop, so epilogue pressure is not the cause; `=1` also moves operands
to other VGPR banks (875 VGPRs). If the steady loop could keep the
default's code (e.g. by preventing that rematerialization, or keeping the
addresses in non-rematerializable form), the overlap would be worth about
1.8% of the wave at 8192³. Archive (patch with all three variants, ATT
bundles, loop listings):
`/home/ibutygin/tlx/perf-data/mxfp-overlap-peel-20261024.tar.gz` (SHA256
`a29d7d7a4c37d18081b804cd87031c71e3f7b4b57a66732e8b75c350f8e6fe80`).

**Where the in-loop LDS address adds come from (2026-10-24).** Traced
through the stages for the default and `OVERLAP_PEEL=2`: the Triton/TLX LLVM
IR steady loops are identical (4,296 instructions) and every one of the 216
LDS loads uses a loop-invariant `getelementptr` defined before the loop.
CodeGenPrepare then sinks all 216 addresses into the loop (`sunkaddr` = slot
base + lane offset + constant) so constants fold into `ds_load` offsets;
instruction selection emits one `V_ADD_U32` per distinct (slot base SGPR,
lane-offset VGPR) pair, 24 in the loop; Machine CSE removes only those with
an identical copy in a dominating block (the tile-start operand loads and
the peeled step provide 18 in the default, 12 with the peel moved to the
epilogue), and Machine LICM does not hoist the rest although their operands
are loop-invariant (`-hoist-cheap-insts` does not help). So the default also
carries 6 such adds. `-disable-cgp` removes the sinking but spills (1,024
VGPRs). Sched barriers cannot help: they only order instructions within a
scheduling region and cannot move them out of the loop's blocks. A fix
belongs in the TLX LDS-load lowering (an opaque per-slot base + lane value,
e.g. `freeze`, before the loop) or in LLVM. Also, `slot = wmma_idx %
NUM_BUFFERS` in `_mxgemm_load_a_operand` on an already reduced slot becomes
a `V_CNDMASK_B32`, putting the slot index in a VGPR.

**Newer AMD codegen (2026-10-24): `05e432a8` is 13% slower; keep the pin.**
Packages from the upstream blob store: `amd-codegen-6bc4aaf6-ubuntu-x64-6`
(upstream `main`'s pin, checksum matches) and `amd-codegen-05e432a8-ubuntu-
x64-1` (#12471; no checksum published yet, SHA256 `1d262f1b…`). Kernels
compiled locally with each and launched with `hw_ship.py`; FFM passed with
`05e432a8` (static 4x4, overlap `=2`, dynamic K=1024); outputs bitwise
identical.

| vs pinned `d26ff26e` (two passes) | 8192³ | 8192²×4096 | 16384³ | 32768³ |
|---|---:|---:|---:|---:|
| `6bc4aaf6` | +0.38 / +0.38% | | | |
| `05e432a8` | +13.1 / +13.5% | +2.6 / +2.4% | +0.26 / +0.19% | +0.29 / +0.31% |
| `05e432a8`, `OVERLAP_PEEL=2` | +12.0 / +12.2% | | | |

`05e432a8` shrinks the 8192³ steady loop from 702 to 611 instructions
(`s_set_vgpr_msb` 177 → 101, ALU waits 19 → 9, address adds 6 → 3, partial
`s_wait_dscnt` counts) at 933 instead of 859 VGPRs, but ATT (16 SEs) shows
the step at 1,054 instead of 855 cycles: LDS reads 153 vs 227, TDM waits 147
vs 36, cluster barrier 101 vs 46, workgroup barrier 108 vs 84, WMMA issue
494 vs 423. The old scheduler keeps ~5 WMMAs in flight across each step's
rendezvous (barrier → TDM refill → `s_wait_tensorcnt 0x4` → barrier); the
new one emits that sequence back to back with at most one WMMA in between,
exposing the input waits. Adopting it needs those WMMAs placed explicitly
(e.g. `sched_group_barrier` around the refill) or a backend fix. Archive
(ATT bundles, asm/LLIR per codegen, scripts):
`/home/ibutygin/tlx/perf-data/mxfp-codegen-05e432a8-20261024.tar.gz` (SHA256
`97db038cbb465f3b2d9fdeb58598f4f0bba4db2944340bf529c97bdf9213c575`).

**Forcing address CSE / keeping addresses alive (2026-10-24): the in-loop
adds are a VGPR-pressure decision; pinning them spills.** Three probes, all
temporary:

1. Kernel-level keep-alive (`KEEP_ADDR_PROBE`): dummy `local_load`s before
   the steady loop, consumed by side-effect inline asm, to create dominating
   address copies for Machine CSE. LLVM merged the probe loads before ISel,
   so the loop was unchanged (the six adds only appeared as three
   `v_dual_add`s in one build, which first looked like a reduction).
2. Redundant modulo (`REDUCED_SLOT`): skipping `slot = wmma_idx %
   NUM_BUFFERS` on an already reduced slot removes the `V_CNDMASK` and a
   VGPR multiply. Static loop 702 → 683 instructions, ALU waits 19 → 15,
   the `v_nop`s of `OVERLAP_PEEL=2` vanish, but six adds remain. Hardware
   (three passes, 8192³): +0.32 to +0.48% slower; with `OVERLAP_PEEL=2`
   +1.0 to +1.2%. Controls with identical code stayed within ±0.2%.
3. LLIR rewrite as a proxy for a TLX "pin address" op: each (slot base,
   lane offset) sum goes through an `asm "=v,0"` identity before the loop
   so it cannot be rematerialized. This removes the in-loop adds but
   spills: pinning all 18 pairs gives 34–36 spilled VGPRs; pinning only the
   group that is still recomputed gives zero adds and 12 spills (default
   config). **The spills are an artifact (see the next paragraph):** on
   gfx1250 the inline-asm `v` constraint maps to `VGPR_32_Lo256`
   (`SIISelLowering.cpp`, `has1024AddressableVGPRs`), so the pinned values
   had to stay in v0–v255 across the loop.

Archive (patch, `pin_lds_addr.py`, `llir_to_asm.py`, asm per variant,
hardware log):
`/home/ibutygin/tlx/perf-data/mxfp-addr-pin-20261024.tar.gz` (SHA256
`858d88556c545de81458a009e0670ef68f7bf57096d9d2cd1b162364266fa4ba`).

**Live values across the steady K loop; class-neutral address pin
(2026-10-24): VGPR pressure is not binding; removing the in-loop adds is
+0.45% slower.** `mir_live.py` computes liveness on the pre-RA MIR (after
`amdgpu-pre-ra-long-branch-reg`) of the 8192³ default and classifies what
is live at the K-loop header:

| Live at loop header (8192³ default) | VGPRs | SGPRs |
|---|---:|---:|
| Accumulators c00..c11 (carried) | 512 | |
| Next step's first A/B operands, loaded before the back edge (carried) | 128 | |
| Scale operands (carried, `vreg_64_lo256`) | 4 | |
| Slot index, cluster state, TDM descriptor parts (carried) | | 13 |
| LDS addresses: 6 lane terms, 8 A/B slot-base + lane sums, 6 scale address terms (invariant, used) | 20 | |
| TDM descriptors and bases used in the loop (invariant) | | 25 |
| Used only after the loop: 5 lane terms; C/epilogue TDM descriptors, kernel args, tile-offset terms | 5 | 52 |
| Total live-in / peak inside the loop | 669 / ~807 | 90 |

Allocation is 859 VGPRs of 1,024, so there is ~165 VGPRs of slack and
almost nothing VGPR-side to recompute outside the loop (5 registers).
SGPRs are the tight file (96 used, 104 with `OVERLAP_PEEL=2`, of ~106):
52 live-through SGPRs are only needed after the loop, while the loop
rebuilds TDM descriptor groups every step (56 SALU per three steps, mostly
`s_mov_b32 sN, s29`). That is the only real "recompute outside" candidate,
and it is cheap SALU that overlaps WMMA.

Re-running the pin with an identity that does not constrain the register
class (`llvm.amdgcn.mov.dpp8` with selector `0xFAC688`, emitted as
`v_mov_b32_dpp ... dpp8:[0,1,2,3,4,5,6,7]` before the loop) and compiling
through `TRITON_KERNEL_OVERRIDE`: zero in-loop address adds, no spills,
861 VGPRs (pins land in v816+), steady loop 702 → 693 instructions for the
default and 706 → 688 for `OVERLAP_PEEL=2`. The rest of the loop is the
same code (same `ds_load`/WMMA order, same sync skeleton). Outputs are
bitwise identical. Hardware, 8192³, three passes:

| vs default | pass 1 | pass 2 | pass 3 |
|---|---:|---:|---:|
| pinned | +0.45% | +0.44% | +0.49% |
| `OVERLAP_PEEL=2` | +0.83% | +0.94% | +1.07% |
| `OVERLAP_PEEL=2` + pinned | +1.55% | +1.70% | +1.78% |

ATT (16 SEs, one dispatch each) puts the difference in the steady loop:
median loop cycles per wave 193.8k → 205.6k, mostly workgroup-barrier wait
(13.7k → 24.5k), i.e. the step barrier after the TDM refill
(`s_wait_tensorcnt 0x4` → signal → 5 WMMAs → wait) waits about twice as
long. The six adds sat between the `ds_load` groups after that barrier;
without them the loads issue back to back and peers arrive later
(hypothesis: LDS request timing across waves). Conclusions: the in-loop
address adds are not a cost, they did not cause the `OVERLAP_PEEL`
steady-loop regression, and a TLX "keep address alive" op is not worth
adding for this kernel. The `OVERLAP_PEEL` loss must be explained by
something else in its steady loop (compare its ATT barrier and LDS waits
against the default). Archive (`mir_live.py`, `dump_prera.py`, pinned
LLIR, asm, ATT bundles and per-instruction stall diffs, hardware log):
`/home/ibutygin/tlx/perf-data/mxfp-live-values-20261024.tar.gz` (SHA256
`c0af5bc3bccd803bbe1f41adf29d2eee0e4e872a56d3e50bc7e7ee5fcfa737a1`).

**Why `OVERLAP_PEEL` was slower (2026-10-24): the next tile's stage-3
refill sat behind the C stores in the TDM queue. With that fixed the
overlap breaks even.** Variants (temporary patch, archived): `ovl2` =
`OVERLAP_PEEL=2`; `g` = `OVERLAP_GUARD` (the next-tile dots run only when
`has_next`, and the last tile does plain C stores); `e` =
`OVERLAP_EARLY_REFILL` (the stage-3 refill is issued right after quadrant
2's C stores instead of after all eight, and quadrant 3's slot waits allow
`LOADS` more pending). Sustained `hw_power.py` timing at the power cap
(about 2,480 W), 3–4 interleaved rounds each. Cycles = time × sampled gfx
clock. All outputs are bitwise identical.

| 8192³ vs default | Time | Cycles | Notes |
|---|---:|---:|---|
| `ovl2` | +0.79 / +0.93% | +1.25 / +1.35% | `GRBM_GUI_ACTIVE` +1.22% |
| `ovl2g` | +0.67 to +0.75% | +1.24 to +1.33% | guard saves about 0.18% |
| `ovl2ge` | **−0.10 / −0.15%** | −0.05 / −0.14% | two sessions |
| `ovl2ge`, refill after quadrant 1 | +0.44% | +0.55% | chunk 7's slot wait then waits for the refill |

- **Wasted step:** without the guard, the last tile's epilogue still runs
  the first step of a next tile that does not exist: 64 WMMAs and 68 LDS
  loads per wave (ATT: 16,448 vs 16,384 WMMAs per wave).
- **Late stage-3 data:** in the overlap epilogue, the stage-3 refill is
  issued after the eight C stores. TDM completes in order, so the refill
  lands late. The first loop iteration's step-2 `s_wait_tensorcnt 0x4`
  rises from 169 to 651 cycles (ATT mean, 144 waves), about 480 cycles per
  tile. Issuing the refill before the last two C chunks removes the
  remaining loss. Moving it before the last four is worse, because chunk 7
  reuses chunk 4's slot and must then wait for the refill too.
- **Hardware shape factorial (`base` vs `ovl2g`, identical loop code in
  every shape):** 4096²×8192 (one tile per program) −0.37%; 4096²×32768
  −2.3%; 8192²×32768 (four tiles) −4.4% (−7% cycles; `GL2C_REQ` −5% with
  the same misses). The overlap's steady-loop code is not slower. The 8192³
  loss came only from the transitions. Long K strongly favors the overlap;
  the L2-request difference there is not explained yet.

Leads ruled out on the way:

- **The in-loop differences ATT shows are not the cause.** ATT shows `ovl2`
  iterations about 50–70 cycles longer (WMMA +18, `ds_load` +26, VALU +17)
  and two extra full drains (`s_wait_alu depctr_va_vdst(0)`) per iteration.
  The drains are write-after-read hazards: the next step's `ds_load`
  overwrites `v[64:79]` four instructions after a WMMA reads it. There is
  also a header wait merged from the epilogue path that comes out as
  `va_vdst(0)` instead of `base`'s `va_vdst(12)`. Hand-editing the asm
  through `TRITON_KERNEL_OVERRIDE` to remove the first (move the loads
  after the next drain) was neutral (+0.78% vs +0.72%). Relaxing the
  second to `va_vdst(10)`, with a full wait on the entry edge, was worse
  (+1.15%). In the one-tile shape, ATT shows the same per-iteration excess
  (+1.8% wave cycles) while hardware shows `ovl2g` 0.37% faster.
  Single-CU ATT apparently inflates issue-level differences, perhaps
  through trace overhead per instruction, so don't use ATT for effects
  under about 2%.
- **The address pin is not the fix.** Pinned builds are slower (`basepin`
  +0.36%, `ovl2gpin` +2.06%). As with the drain edits, removing stalls
  in the loop tends to make it slower: the steady loop behaves like a
  contended shared resource, where per-wave stalls act as throttles.
- **Power is not the cause.** Every variant runs at the same cap. The
  overlap variants run at the same or higher clocks, and the cycle counter
  shows the same ordering when not throttled.

Archive (probe patch with `OVERLAP_GUARD`/`OVERLAP_EARLY_REFILL`,
`hw_power.py`, ATT bundles with tile/iteration/drain scripts, PMC, asm and
override files, logs):
`/home/ibutygin/tlx/perf-data/mxfp-overlap-reason-20261024.tar.gz` (SHA256
`f44a38692b928b3e3a2927dac00f79e66c1489624665ebdc8107a61f62e6c5d8`).

**Overlap across shapes, dynamic-loop port, and multicast alignment
(2026-10-24).** `ovl` = `OVERLAP_PEEL=2` + `OVERLAP_GUARD` +
`OVERLAP_EARLY_REFILL`. The probe patch now also runs the overlap in the
three-stage dynamic `DEFER_BOTTOM` loop: the loop starts at step 1 from the
slot after `phase`, and both loop kinds use the pinned native dot so the
carried accumulators keep one layout. Outputs match the default at K=1408
(8 dynamic steps), 4096 and 8192. Sustained timing vs the `bench.py`
default for each shape:

| Shape (bench default) | 3-stage static | static + `ovl` | dynamic + `ovl` |
|---|---:|---:|---:|
| 8192³ (static) | — | **−0.10 to −0.27%** (4 runs) | +0.28% (dynamic default +0.75%) |
| 8192²×4096 (3-stage dynamic) | +6.9% | +2.6% | **−0.93%** |
| 8192²×32768 (4-stage dynamic) | +15.8% | +12.8% | n/a |
| 16384³ (4-stage dynamic) | +9.1% | +4.8% | n/a |
| 32768³ (4-stage dynamic) | +8.1% | +2.4% | n/a |

So the overlap is a small win on both three-stage default paths. Long K
would need it ported to the four-stage ring (two C slots, different
slot-wait accounting).

The large long-K overlap gains were measured against the weaker three-stage
build, and most of them come from cluster multicast alignment. A multicast
load merges only when the cluster peers request it close together;
otherwise late peers issue separate loads, which hit in L2. PMC for the
three-stage static build (40 dispatches each):

| vs `s3` (barrier interval 8) | K=8192 L2 req | K=8192 cycles | K=32768 L2 req | K=32768 cycles |
|---|---:|---:|---:|---:|
| `ovl` | −0.3% | −0.2% | −3.2% | −3.6% |
| interval 4 | −1.2% | +1.4% | −14.5% | −6.0% |
| `ovl` + interval 4 | −1.1% | +1.2% | −15.0% | −6.5% |
| interval 16 | +4.5% | +7.1% | +31.6% | +18.8% |

L2 misses and DRAM reads are unchanged; only L2 hits move, i.e. duplicate
requests from unmerged multicasts. Timing agrees: at 8192²×32768, interval
4 is −4.9% and `ovl` + interval 4 −5.8% vs the 3-stage build, and at
8192³ interval 4 is +1.05% and 16 is +7.3%. The four-stage default does not
gain from a tighter interval: 4 and 6 are +0.5 to +1.1% at 8192²×32768,
16384³ and 32768³, so 8 stays. Archive (probe patch with the dynamic port,
logs, PMC, asm):
`/home/ibutygin/tlx/perf-data/mxfp-overlap-shapes-20261024.tar.gz` (SHA256
`2ad08ff07f94fe727649ec3c704b9cec809a102d7897ff7a8a1d8d3c18c2e5e6`).

**Overlap peel option (2026-10-24).** The probe is now one option,
`OVERLAP_PEEL` (`--overlap_peel`, `bench.py --overlap-peel`), which always
includes the last-tile guard and the early refill. It runs in the static and
dynamic loops with three or four buffers. `_mxgemm_overlap_store` stores a C
quadrant and issues the next tile's stage-`NUM_BUFFERS` refill before chunk
`max(chunks − C_SLOTS, 2 × parts per quadrant)`: chunk 5 of 8 with three
64x128 slots (the probe used 6), and chunk 14 of 16 with the four-stage
ring's two 64x64 slots, splitting quadrant 3's stores. The chunks after the
refill pass `EXTRA_PENDING=LOADS` to their slot waits. Default builds
compile to identical code. Hardware (`hw_power.py`, three rounds, vs the
`bench.py` default; outputs bitwise identical at K = 1408, 2176, 4096,
8192, 9344, and 32768):

| Shape | Ring | `OVERLAP_PEEL` |
|---|---|---:|
| 8192³ | 3 stages, static | −0.21% |
| 8192²×4096 | 3 stages, dynamic | **−1.20%** |
| 8192²×32768 | 4 stages, dynamic | −0.03% |
| 16384³ | 4 stages, dynamic | +0.12% |
| 32768³ | 4 stages, dynamic | −0.03% |

The four-stage ring runs at the power cap (2,500 W, about 1.64 GHz), and
removing transition stalls does not reduce its time. `bench.py` therefore
enables the overlap only for three-buffer A8W8. The four-stage builds spill
10–12 SGPRs with the overlap, against 6–8 without. FFM:
`test_mxgemm_overlap_peel_ring_phase` (24 cases: static, dynamic, and
four-stage rings; 4–10 K blocks; with and without 2x2 clusters) passes.
Archive (patch, hardware log, asm):
`/home/ibutygin/tlx/perf-data/mxfp-overlap-option-20261024.tar.gz` (SHA256
`f603424020146799824451609d0f3ec312dc68ca4357ded216da9d4a25cb8d47`).

Not worth repeating: issuing next-tile loads after the C stores (+1.6%),
next-tile L2 prefetch, wave-specialized handoff, and more but smaller C
slots (6 × 64x64), all measured earlier in this document. A fourth 32 KB C
slot does not fit (316,584 of 327,680 bytes of LDS used).

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
5. Tile transition: see "A8W8 8192³ profile and tile transition
   (2026-10-24)" for the measured breakdown and ideas. Static K tail and
   batched first C fill gave no gain. Overlapping the next tile's peeled
   step with the C stores saves ~1.5k cycles per transition but makes LLVM
   rematerialize LDS addresses in the steady loop. The real cause was the
   late stage-3 refill plus a wasted last-tile step (see "Why
   `OVERLAP_PEEL` was slower"). With guard and early refill it breaks even
   at 8192³ (−0.1 to −0.3%) and, ported to the dynamic loop, −0.9% at
   8192²×4096 (see "Overlap across shapes"). It is now the `OVERLAP_PEEL`
   option and a three-stage `bench.py` default (see "Overlap peel
   option"); it is neutral on the power-bound four-stage ring. Re-measure
   the defaults with `bench.py` once the remote Triton is rebuilt.
6. LDS partition conflicts: see "LDS partition conflicts: what we learned"
   above for the open directions (lane-bit scale layout for swizzled waves,
   explicit partition placement). Rebuild the remote Triton with
   `639c951130` before running `bench.py` with the new 8192³ default.
7. Update the [resource/dependency model](../../tools/perf_model/README.md)
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
