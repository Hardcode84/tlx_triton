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

Pre-existing post-rebase breakage found during validation (both fixed on
2026-10-25; see "TLX AMD test repair"):

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

**Redundant scalar work (2026-10-24): present, but worth at most about
0.1%.** In the 8192³ bench default (`OVERLAP_PEEL`, static loop), the steady
loop issues 56 SALU per iteration (three K steps). Each step rebuilds its TDM
descriptor groups: the fused A+B load's group 2 (`<64,0,0,0>`) and group 3
(zeros) are constants shared by every load in LLVM IR, but the allocator
splits them and copies them back from a zero SGPR every step (about 14 movs
per iteration). Two `s_mov_b64` per iteration are fully dead and four are
half dead (overwritten by `s_mov_b32` before use). The rest is needed: global
address updates, LDS slot address writes, and the `k % 8` cluster-barrier
check.

The cause is SGPR pressure (104 of about 106). After allocation, 83 SGPRs are
live at the K-loop header and 39 of them are never referenced in the loop.
They are tile-invariant C-store state hoisted to the kernel preheader: the
C descriptor's group 1, three per-C-slot group-0 templates (flags plus each
wave's LDS slot address), and `c_ptr` and C-address terms. LLVM options do
not help: `disable-machine-licm` frees 10 SGPRs but adds SALU (62);
`sink-insts-to-avoid-spills` and `greedy-regclass-priority-trumps-globalness`
change at most one instruction.

Sensitivity on hardware (asm through `TRITON_KERNEL_OVERRIDE`, three rounds,
outputs identical): removing the dead copies and narrowing the half-dead ones
(56 → 52) is −0.04% (noise). Adding 60 independent SALU per iteration (116)
costs +0.31% (+0.52% cycles), about 0.005% per SALU per iteration. Even
removing all ~20 redundant loop SALU is worth about 0.1%, so it is not worth
TLX-level contortions to keep the C-store state out of the loop.

The tile transition (K tail, overlapped epilogue, tile entry) issues 342
SALU per tile. The avoidable parts are tile coordinates computed twice per
tile (current tile at entry, next tile in the tail, about 32 SALU each;
carrying `next_m`/`next_n` would remove one) and per-chunk C descriptor and
address work (about 10 SALU per 64-row chunk; one tile base would halve it).
That is about 70 SALU per tile in a wait-dominated transition, likely under
0.1%. Archive (asm variants, scripts, MIR liveness, hardware log):
`/home/ibutygin/tlx/perf-data/mxfp-salu-20261024.tar.gz` (SHA256
`31e4e57eb8f39c59d81b35caea29d40152facebdd022f89c687af988ca5d6b67`).

**Automatic wave specialization (2026-10-24): mixed, not adopted.** The
local `wave-specialization` branch (6 commits, MLIR uniformity analysis plus
`HIPOptions.enable_wave_specialization`) cherry-picks cleanly onto
`46500c4c5b` (local branch `ws-test`). It first failed to compile this
kernel: after cloning, the canonicalizer merged blocks that differ only in a
TDM wait count into one block argument, and LLVM rejected the non-immediate
`llvm.amdgcn.s.wait.tensorcnt` operand ("immarg operand has non-immediate
parameter"). Lowering `AsyncTDMIntrinsicWait` to `ROCDL::WaitTensorcntOp`
(count as an attribute) fixes it; `test/Conversion/amd` lit passes with the
CHECKs updated. With the option off, the 8192³ default compiles to
identical code.

The pass splits on the fused-TDM A/B wave selector (a wave-ID bit) and
clones the kernel body into two versions (code 2,500 → 4,743 instructions).
It does remove the selector work: the base LLVM IR has 29 `select`s on that
one predicate (`icmp eq (and %wave, 5), 0`: 15 LDS pointers, 6 global
bases, 6 descriptor words, 2 whole 256-bit descriptor groups) and the
specialized IR has none; `s_cselect` goes from 69 to 48 across both
versions. But 12 of the selects are in the kernel prologue and 17 in the
tile transition, none in the K loop, and after each select only the chosen
descriptor is live, so the loop's SGPR use is unchanged: each version still issues 56 SALU per iteration and the function
still uses 104 SGPRs. The loop shrinks by 16 instructions (fewer
`s_set_vgpr_msb`/`v_nop`), and per-tile transition SALU drops from 342 to
about 270. Hardware (`hw_power.py`, three rounds, vs the `OVERLAP_PEEL`
bench default; outputs identical): 8192³ +0.34% (cycles +0.36%),
8192²×4096 −0.48% (cycles −1.1%). The transition-heavy K=4096 case gains
and 8192³ loses, possibly from the doubled instruction footprint.

Why the 17 transition selects stay per tile (base LLVM IR): every one
depends on a tile-loop phi. The 8 descriptor-word and global-base selects
depend on the tile index: `select(isA, addrA(off_m), addrB(off_n))`, where
both arms differ in the base descriptor and in the per-tile offset. The 9 LDS
slot-pointer selects depend on the ring phase: both arms are
`gep(base_piece, slot * 17408)` with the same offset, but the A/B choice is
wrapped around a second wave-dependent choice (an `extractelement` from a
`<2 x ptr>` vector of LDS partition pieces, by wave bit 1). LLVM can fold
`select(c, gep(a, i), gep(b, i))` to `gep(select(c, a, b), i)`, but not
through that vector, and not for two different sums. So each wave computes
both the A and the B candidate from per-tile data and then selects. The 12
prologue selects run once per kernel.

To hoist them, the select must take tile-invariant inputs, with the
per-tile part applied once afterwards: select the base descriptor and LDS
partition base (hoisted by LICM), then per tile select only the scalar
offset (`off_m`/`off_n`), apply one descriptor update, and add one slot
offset. The fused-TDM lowering cannot do this: it sees member descriptors
that already include the per-tile offset (from `update_tensor_descriptor`),
and lowerings must not inspect producer ops. It needs a transform before
lowering: recognize fused members built as `update_tensor_descriptor(base,
offsets)` plus `local_view(buf, slot)`, and rewrite the fused load into a
form taking per-member base, offsets, and slot. A source-level rewrite that
picks `off_m`/`off_n` from the wave ID would put a hidden warp-varying scalar
in the kernel, which `AGENTS.md` forbids. Expected value: up to what wave
specialization removed from the transition (about 72 of 342 SALU per tile;
−0.48% at K=4096) without its code-size cost at 8192³, so roughly 0.2–0.5%
at K=4096 and a small gain at 8192³. Not implemented. Archive
(lowering patch, compile wrapper, binaries, log):
`/home/ibutygin/tlx/perf-data/mxfp-wave-specialization-20261024.tar.gz`
(SHA256 `920dc59809194171dc03eb1fda344d2ef9905c270239a10cb39d803fd8f0153f`).

**Profile of the current defaults with `OVERLAP_PEEL` (2026-10-24).**
Sustained: 8192³ 0.1497 ms (7,346 TFLOPS) at 1,797 MHz and 2,472 W (at the
power cap); 8192²×4096 0.0833 ms (6,602 TFLOPS) at 1,886 MHz and 2,431 W.
ATT, 48 waves per shape (3 dispatches × 16 SEs), stall cycles charged to
the instruction before the next issue (`att_breakdown.py`); percentages are
of mean wave cycles (8192³: 249.6k; K=4096: 138.6k). ATT runs below the
power cap and inflates small issue-level differences, so use it for the
split, not for sub-2% comparisons.

| Share of wave | 8192³ (static loop) | 8192²×4096 (dynamic loop) |
|---|---:|---:|
| WMMA floor (8 cycles/WMMA) | 52.5% | 47.3% |
| Steady K loop | 83.0% | 71.6% |
| … WMMA issue | 41.2% | 34.1% |
| … `ds_load` issue + LDS waits | 14.7% + 8.1% | 14.8% + 9.7% |
| … workgroup + cluster barriers | 6.9% + 4.1% | 0.9% + 2.5% |
| … SALU / VALU | 3.4% / 0.8% | 4.6% / 3.4% |
| … TDM waits | 3.4% | 1.0% |
| Tile transitions (3 per wave) | 11.3% | 19.3% |
| Final epilogue (last tile's C) | 3.9% | 6.3% |
| Prologue | 1.7% | 2.8% |

8192³ steady step (iterations 1–18, per K step): 849 cycles: WMMA issue
428, `ds_load` issue 152, LDS wait 79, workgroup barrier 67, cluster
barrier 40, SALU 41, TDM wait 33, VALU 8. A step reads about 139 KB of LDS
per CU (68 `ds_load_b128` per wave), about 164 B/cycle. The first iteration of
each tile costs 3,136 cycles for three steps (2,547 steady), mostly barrier
waits. Transitions are about 9.4k cycles each (8.9k at K=4096); the largest
stalls are the C-slot reuse wait (`s_wait_tensorcnt 0x2`, about 2.8k per
wave), the stage acquire (`s_wait_tensorcnt 0x4`, about 2.5k), and LDS
waits for C staging and the overlapped next-tile reads. The final epilogue
mostly waits for the last C stores (TDM 4.3k).

Bottlenecks in order: (1) LDS reads in the steady loop, about 23–25% of
the wave on both paths (but see "LDS reads, deeper": much of this is
matrix-pipe time charged to loads, and the rest is latency exposed at the
step boundary, not LDS bandwidth); (2) at 8192³, per-step
workgroup and cluster barriers (11%); the dynamic K=4096 loop shows only
3.4%, so the static loop's per-step rendezvous or its wave balance is worth a
targeted look; (3) tile transitions plus final epilogue (15% at 8192³, 26%
at K=4096), now bound by C-store drain and slot reuse; (4) SALU/VALU address
work (4–8%), mostly hidden (see "Redundant scalar work"). L2 hit rate is
85.9% (8192³) and 80.3% (K=4096); the EA/DRAM counters read implausibly low
and are not used. Archive (ATT bundles, PMC, binaries, scripts):
`/home/ibutygin/tlx/perf-data/mxfp-profile-overlap-defaults-20261024.tar.gz`
(SHA256 `6bac4522b2dbee4c320675de35bd8bea612dc14d7d64c3d266f061da4ecf8689`).

**LDS reads, deeper (2026-10-24): LDS bandwidth is not the limit;
exposed latency after each step's synchronization is.**

Hardware microbenchmarks (`ldsbw/` in the archive; device-only HIP code
objects launched through `libamdhip64` with ctypes, because the remote HIP
headers do not match the gfx1250 compiler; one workgroup per CU with almost
all LDS allocated; timing with `s_get_shader_cycles_u64`):

- Dependent `ds_load_b128` latency: about 75 cycles with 1, 4, or 8 waves
  per CU (load to use; 56–59 cycles for the first load of a burst).
- One wave issues `ds_load_b128` back to back at one per cycle; a burst of
  N completes after about 56 + 2·N cycles (inline asm, so the compiler
  cannot insert waits). That is about 256 B/cycle per wave.
- CU throughput with 16 waves: 512 B/cycle when loads spread over at least
  two partitions, 256 B/cycle when all hit one partition, 367 B/cycle when
  every wave alternates between the same two. Four waves move 48 KB in
  105 cycles across two partitions (about 470 B/cycle including latency) and
  in 167 cycles from one partition (about 290 B/cycle).
- Caution: a C++ version of the same test compiled with `s_wait_dscnt`
  between its loads and reported a false per-wave limit of one load per
  13–15 cycles at any width. Check the ISA of LDS microbenchmarks.

The GEMM averages about 154 B/cycle of reads plus about 78 B/cycle of TDM
writes per CU at 8192³, far below these rates. The 849-cycle steady step
(ATT, per K step, `pcm-8192.txt` timeline) decomposes as: WMMA floor 512
(64 × 8); step synchronization about 140 (stage acquire
`s_wait_tensorcnt 0x4` about 33, the workgroup barrier after it 41–75,
cluster barrier 40 amortized); LDS waits 79, which is the latency of the new
stage's first operand loads, issued only after that barrier while about five
deferred WMMAs of the old stage cover it; extra load-issue cost about 69
(68 loads at about 1.9 cycles each, beyond what overlaps the matrix pipe,
including about 22 cycles of the two full WMMA drains,
`s_wait_alu depctr_va_vdst(0)`, that ATT charges to the preceding load); and
SALU/VALU about 49. So LDS-related time is about 150 cycles per step and
synchronization about 140, both around the step boundary.

Ways to cut it: cover the post-barrier load latency with more matrix work
(a deeper register prefetch of the next stage needs about 256 more VGPRs per
wave, over the 1,024 budget at 859 now), shorten the barrier (wave skew),
and avoid the two drains (register assignment, see "Why `OVERLAP_PEEL` was
slower"). More waves do not help: `num_warps=8` (2 waves per SIMD, 512
VGPRs, 2 spills, dynamic loop) has fewer loads per wave (52 vs 68) but two
waves share each matrix unit (10.8 cycles per WMMA) and the step grows to
957 cycles; hardware +9.8% at 8192³ and +7.2% at 8192²×4096 (outputs
identical). Archive (microbenchmark sources and runners, 8-wave ATT, logs,
step timeline): `/home/ibutygin/tlx/perf-data/mxfp-lds-deep-20261024.tar.gz`
(SHA256 `78ce79d5472558fab5d9a8a465f6c4c1238bc11ba2d650c88fd3327875c5d947`).

**WMMA queuing, SCHED_MODE[2] (2026-10-24): no gain.** The kernel's
`SCHED_MODE_2` parameter no longer compiles (see the post-rebase breakage
note); the equivalent is the backend option `enable_wmma_queuing=True`,
which only adds `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1`
at kernel entry (the rest of the code is identical). Hardware against the
current defaults (`hw_power.py`, four rounds, outputs identical): 8192³
+0.43% (cycles +0.67%), 8192²×4096 +0.05% (cycles −0.21%). With one wave per
SIMD and the step bound by synchronization and exposed LDS latency, queuing
WMMAs behind scalar and memory work does not shorten it. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-wmma-queuing-20261024.tar.gz` (SHA256
`4be760dc5ca4cd5ccda67865919477b8681b13dd314b7c121e14652de066fa07`).

**Avoiding the WMMA drains from register reuse (2026-10-24): possible, but
every attempt so far costs more than it saves.** The 8192³ static loop has
five full `va_vdst(0)` drains per three K steps
(`drain_summary.py`). One is costly (about 27 cycles in ATT): the next
step's `a00` loads (`ds_load_b128 v[64:67]`) land in the registers of the
current `a00`, four instructions after `c01`'s last WMMA reads them. Each
16-VGPR operand is one virtual register that dies at its last WMMA, and the
next loads start right after, so the allocator is free to reuse it. The
dynamic K=4096 loop has no full drains.

- Allocator options (`greedy-reverse-local-assignment`,
  `greedy-regclass-priority-trumps-globalness`, `sink-insts-to-avoid-spills`,
  `amdgpu-use-amdgpu-trackers`) leave the reuse unchanged.
- Loading the next `a00` after `c10` (temporary `MXFP_A_AFTER_C10`): the
  drain moves to another operand at WAR distance 1 (875 VGPRs); hardware
  +0.86% (cycles +1.40%).
- `llvm.fake.use` on each in-loop WMMA operand, placed before the next WMMA
  after its last use (LLVM IR rewrite `fake_use.py`, 46 fake uses): removes
  all short-distance WAR drains (one left at distance 29) and shortens the
  loop (703 → 667 instructions), 910 VGPRs, no spills. But the fake uses act
  as scheduling barriers: the WMMAs that used to straddle each step's
  barrier → TDM → `s_wait_tensorcnt` → barrier rendezvous are gone
  (`B Q T Q T B` with no WMMA, then a 34-load burst), the same failure as
  codegen `05e432a8`. Hardware +13.6% (cycles +22.9%). Outputs identical in
  all variants.

A working fix has to extend the operand live ranges after scheduling,
without constraining it: for example an AMDGPU pass between the machine
scheduler and register allocation that adds each WMMA's A/B vregs as
implicit uses of a later WMMA, or allocation hints against registers read
by in-flight WMMAs. That needs a codegen change (the pinned AMDGPU codegen
is a prebuilt library). Expected value is about 20–25 cycles per K step
(about 2.5%) at best. Archive (rewrite script, LLVM IR, binaries, patch,
hardware log): `/home/ibutygin/tlx/perf-data/mxfp-war-drains-20261024.tar.gz`
(SHA256 `9418f098e2f53c8a0506a0da75047e6d7d503d522ba63ae8a0ee02d06f9fedb7`).

Not worth repeating: issuing next-tile loads after the C stores (+1.6%),
next-tile L2 prefetch, wave-specialized handoff, and more but smaller C
slots (6 × 64x64), all measured earlier in this document. A fourth 32 KB C
slot does not fit (316,584 of 327,680 bytes of LDS used).

#### Patching the AMDGPU codegen: cost and a working pass (2026-10-24)

Rebuilding the pinned codegen locally is cheap. The pinned library
(`libtriton_amd_codegen.so`, revision `d26ff26e`) is the 349-line wrapper
`third_party/amd/backend/codegen/amdgpu_codegen.cc` linked statically
against upstream llvm-project `d26ff26e` (recipe:
`.github/workflows/amd-codegen-llvm-build.yml`; the prebuilt LLVM bootstrap
package is a 404 on the blob store, so build LLVM yourself).

- LLVM `d26ff26e`, AMDGPU only, Release + assertions, clang/lld, static:
  worktree `~/llvm/llvm-d26`, build `~/llvm/llvm-d26-build`, install
  `~/llvm/llvm-d26-install`. Full build 72 s on this host (384 cores).
- Wrapper: copy `third_party/amd/backend/codegen`, point CMake at the LLVM
  install, about 4 s to build. Use it with
  `TRITON_AMD_CODEGEN_PATH=<install>/lib/libtriton_amd_codegen.so`.
- The local build produces the same machine code as the pinned library on
  the default 8192³ kernel; only the `.ident` lines differ.
- The wrapper sets every codegen flag to `true`; the local copy accepts
  `name=value` (5-line change), which integer options need.

A custom MachineFunctionPass does not need an LLVM patch and does not need
MIR round trips. LLVM calls the callbacks registered with
`RegisterTargetPassConfigCallback` before it builds the codegen pipeline,
and `TargetPassConfig::insertPass` is public, so the wrapper can insert its
own pass instances after any pass ID, including AMDGPU-internal IDs such as
`SIFixVGPRCopiesID`, which the static library exports. One trap: register the
callback from the wrapper's `initializeTarget()`, not from a static
initializer, because LLVM's callback list is constructed after the
wrapper's static initializers and wipes earlier registrations (static
initialization order).

Why `llvm.fake.use` broke scheduling: `MachineScheduler`'s `isSchedBoundary`
treats every `FAKE_USE` as a region boundary, both before and after register
allocation. The pass below therefore inserts `FAKE_USE`s only after the
pre-RA scheduler and erases them after allocation (before the post-RA
scheduler and `SIInsertWaitcnts`).

Prototype `wmma_src_live.cc` (about 200 lines):

1. After `MachineScheduler`, for each WMMA it adds a `FAKE_USE` for each
   512-bit or larger A/B source vreg after the WMMA that is `d` WMMAs later
   in the same block, then recomputes those live intervals.
2. After `SIFixVGPRCopies`, it erases every `FAKE_USE`.

Options:

- `triton-wmma-src-live-distance=d`;
- `triton-wmma-src-live-cross-block` for dominated successor blocks;
- `triton-wmma-src-live-block-end`;
- `triton-wmma-src-live-verbose`.

The machine verifier passes. Steady-loop results on the 8192³ default:

| variant | `va_vdst` waits | `va_vdst(0)` | loop instrs | VGPRs |
| --- | --- | --- | --- | --- |
| default | 14 | 5 (one at WAR distance 4) | 703 | 859 |
| `d=2`, `d=4` | 16-17 | 3 | 699-708 | 877-878 |
| `d=6` | 9 | 3 (2 WAR, block boundary) | 696 | 939 |
| `d=8` | 9 | 3 (2 WAR, block boundary) | 680 | 947 |
| `d=6/8` cross-block | 10-13 | 4 | 692-726 | 1024, 16 spilled |
| `amdgpu-va-vdst-war-hazard=false` | 3 | 1 | 692 | 859 |

The steady loop spans several basic blocks (header, conditional TDM-issue
blocks, latch). The pass removes every short-distance drain inside a block,
including the costly distance-4 `a00` one. The two remaining WAR drains sit
at block boundaries. Following WMMAs across dominated blocks pushes VGPRs
to the limit and moves the drains to the scale loads.

Hardware 8192³, compiled with the same local codegen:

- Upper bound with WAR waits switched off (`amdgpu-va-vdst-war-hazard=false`):
  bitwise-identical outputs over 10 screens, sustained 0.14969 vs 0.14998 ms
  (−0.20%, cycles −0.41%). This is not shippable: it drops waits that
  protect WMMA source reads.
- `FAKE_USE` pass, `d=6` and `d=8`: identical outputs over 10 screens;
  sustained 0.15033 and 0.15036 vs 0.15010 ms (+0.15%, +0.17%; cycles
  +0.14%, +0.18%). Fewer drains, but the extra live ranges cost as much as
  they save.

Conclusion: WMMA WAR drains are worth at most about 0.4% of cycles at
8192³, far below the 2.5% estimated from the ATT stall attribution, and
the safe fix gains nothing. Drop this direction. The
`FAKE_USE` pass is a cheap, safe way to steer allocation, and the same
callback hook works for other post-scheduler experiments (for example,
forcing operand placement or setting allocation hints).
Archive (pass source, wrapper diff, flag compile script, assembly, ship
tarballs, hardware scripts and logs):
`/home/ibutygin/tlx/perf-data/mxfp-codegen-plugin-20261024.tar.gz` (SHA256
`07258becd96989869e13e96d2a59b72db4c8af6f2d4026c036d5eba79d2e0635`).

#### Wave imbalance in the 8192³ steady loop (2026-10-24)

There is no work imbalance between the four waves of a workgroup. The
per-step synchronization cost (about 140 cycles: workgroup barrier 67,
cluster barrier 40, TDM wait 33) is mostly late TDM data, seen either
directly or through other waves and other CTAs.

Method: on gfx1250, `--att-simd-select` takes a SIMD index (the old `0xF`
traced SIMD 3 only). The other SIMDs appear only as category-5 timestamps
without a SIMD ID, so they cannot be separated. We collected four ATT runs
of the ship29 default (dispatch 40, target CU 0, all 16 SEs), one per SIMD,
giving 16 fully decoded waves per SIMD role. Steady-loop structure per K
step:

1. release barrier (`s_wait_dscnt 0` → signal, wait five WMMAs later);
2. TDM refill;
3. `s_wait_tensorcnt 0x4`;
4. acquire barrier;
5. every eighth step, a workgroup barrier and then the cluster barrier
   (wave 0 signals `-3`, all waves wait `0xfffd`).

- **Workgroup barriers:** for every SIMD role, the time from signal to
  release equals the floor (38–47 cycles, the barrier round trip) in 92–98%
  of release barriers and about 80–90% of acquire barriers. The four waves
  therefore arrive within a few cycles of each other. The release-barrier
  stall is 2 cycles median, 4–7 mean, because the round trip hides behind
  the five WMMAs between signal and wait.
- **Acquire barrier tail:** the stall is 2 cycles median but 20–90 cycles
  mean, with lead p90 100–280. It comes from `s_wait_tensorcnt 0x4`
  (median 1–2 cycles, mean 25–100, p90 up to 260): in about 10% of steps
  one wave's TDM data is late, and the other waves then wait at the
  barrier. The slow wave changes between runs, so no SIMD role is
  systematically slow. TDM tail plus acquire stall is about 55–100 cycles
  per step per wave.
- **Cluster barrier:** 240–310 cycles median per occurrence (means
  270–590, rare 2–6k outliers), 27–40 cycles per step for every wave. No
  CTA is consistently last: each traced CTA is last in 1–6 of its 20
  cluster barriers, with median lead 210–420 cycles. This is random drift
  across the 16 CTAs from the same TDM tails accumulated over eight steps.
- **The tails depend on step position:**
  - in the two steps after a cluster barrier, only 1–2% of
    `s_wait_tensorcnt 0x4` waits exceed 100 cycles;
  - from the third step on, 16–19% do;
  - the third step of each tile has the highest rate, 28% (mean 142).

  The first two steps consume loads issued before the barrier, and the
  roughly 300-cycle barrier wait gave them time to land. So the barrier
  and the TDM tails are one latency budget: TDM delivery takes longer than
  the ring's two-step lookahead (about 1.7k cycles at the 2.4 GHz trace
  clock) in roughly 17% of steps.

Caveat: ATT runs at about 2.4 GHz and 995 W with serialized dispatches.
Under the sustained power cap (about 1.8 GHz) the same memory latency is
about 25% fewer core cycles, so production sees fewer tails than these
traces. Levers, in order:

1. more lookahead without more LDS (issue the refill earlier in the step,
   or retest the four-stage ring with two C slots under current defaults);
2. retest cluster interval 16 and the split cluster barrier under current
   defaults (both predate static unroll and overlap peel);
3. shorten the transition, which causes the step-2 tail.

Archive (decoded ATT bundles for SIMD 0–3, analysis scripts, report):
`/home/ibutygin/tlx/perf-data/mxfp-wave-imbalance-20261024.tar.gz` (SHA256
`d2ef5937547abfab1ff9c25e911f0e4ab11a37d918b4ddd816f8dc2b105022d4`).

#### Sync levers and spaced L2 prefetch (2026-10-24): all slower

Hardware 8192³ against the current default (ship29 configuration), four
sustained rounds. Every output was bitwise identical to the default over 10
screens.

| Variant | Sustained | Cycles | Energy/launch |
| --- | ---: | ---: | ---: |
| Cluster barrier interval 16 | +3.6% | +5.2% | +3.3% |
| Split cluster barrier | +3.2% | +4.7% | +2.9% |
| Four-stage ring, two C slots, dynamic slots (321,736 B LDS) | +1.6% | +1.6% | +1.3% |
| Mid-step L2 prefetch, d=1 | +9.6% | +14.8% | +9.4% |
| Mid-step L2 prefetch, d=2 | +8.3% to +8.5% | +12.7% | +8.0% |
| Mid-step L2 prefetch, d=4 | +8.0% | +12.3% | +8.0% |
| d=2 with `SCOPE_DEV` (LLVM IR override, hint 16) | +8.3% | +12.7% | +8.3% |
| d=2 control: prefetches replaced by an empty asm using the address | +1.5% | +2.3% | +1.5% |

- **Cluster rendezvous:** a less frequent or split rendezvous costs more
  than it saves. The barrier keeps the 16 CTAs aligned so their multicast
  requests coalesce; more drift means more separate loads. Interval 8,
  unsplit, stays.
- **Four-stage ring:** adds one stage of lookahead but loses the static
  unroll and the third C slot. The net is still a loss at 8192³, so the
  late TDM completions are not fixed by more lookahead alone.
- **Earlier refill (not built):** with three stages the refill overwrites
  the slot read in the previous step. It must follow the release barrier,
  where it already issues.
- **L2 prefetch, mid-step:**
  - Placement: stage `i + 3 + d`, issued after the `c10` dot (about
    WMMA 48 of 64) via a temporary `MID_L2_PREFETCH` knob.
  - Dedup: ownership rotates per step among the four CTAs sharing each A
    (or B) tile, so each CTA runs a prefetch block on about half the steps.
  - Guards: branches on block-uniform predicates; speculative, so no NT
    hint.
  - Result: about 2.5x better than the October attempt next to the refill
    (+20%), but still +8%. The control isolates the cost: the per-lane
    address VALU (64-bit `v_add` after WMMAs, `va_vdst(0)` drains, loop
    703 → 837 instructions) costs 1.5%; the prefetch memory traffic itself
    costs about 7%. Scope does not matter. The earlier PMC data agrees:
    prefetch doubles L2 requests and does not reduce L2 misses (85.9% hit
    rate).
- **Removing the address VALU:** this needs the scalar-base form of
  `global_prefetch_b8` (SGPR base plus 32-bit VGPR offset). The address is
  invariant per-lane offset plus a uniform step offset, but codegen LICM
  hoists the `zext` of the lane offset out of the loop, and the scalar-base
  matcher needs the `zero_extend` in the same selection DAG. LLVM IR
  rewrites (`saddr_pf.py`: `readfirstlane` on the scalar part, an asm
  identity on the lane offset) did not get there and caused spills. A
  lowering change in `emitTDMPrefetch` would be needed, but it can recover
  at most the 1.5% overhead, not the 7% traffic cost.
- **Source:** reverted to HEAD.

Archive (knob patch, ship tarballs, LLVM IR overrides, scripts, hardware
logs): `/home/ibutygin/tlx/perf-data/mxfp-sync-l2pf-20261024.tar.gz` (SHA256
`b63a254f89c4f3ab492d6db68c133e06af751bc45447204ff89423274bcc5b3b`).

#### Outside the K loop: current breakdown (2026-10-24)

Source: the four SIMD-select ATT runs of the current default (8192³, 64
waves, 192 transitions). Each wave runs four tiles: prologue, three
transitions, final epilogue. Each transition is split at the first C
`ds_store` into the K tail (last three K steps plus the next tile's stage
loads) and the C epilogue (eight 64x128 FP32 chunks through three LDS slots,
overlapped with the next tile's first step). Mean cycles per occurrence:

| Phase | Cycles | WMMAs | Largest classes |
| --- | ---: | ---: | --- |
| K tail (×3) | 3,782 | 192 | WMMA 1,327, `s_wait_dscnt` 756, SALU 580, `ds_load` 426, cluster barrier 304, VALU 210 |
| C epilogue (×3) | 5,321 | 64 | `s_wait_tensorcnt` 1,851, `s_wait_dscnt` 1,221, `ds_store` 763, workgroup barrier 582, WMMA 489 |
| Prologue | 4,215 | 64 | `s_wait_tensorcnt` 1,649, SALU 1,029, workgroup barrier 570, WMMA 443 |
| Final epilogue | 9,829 | 192 | `s_wait_tensorcnt` 4,307, `s_wait_dscnt` 1,695, WMMA 1,327, SALU 579, cluster barrier 533 |

Totals per wave:

- **Outside the K loop:** about 41k of 247k cycles (17%), containing 1,024
  WMMAs. At the steady-loop rate (849 cycles per 64-WMMA step) that work
  takes about 13.6k cycles, so the excess is about 27k cycles (about 11% of
  the wave). Median transition: 9.0k cycles (p10 8.2k, p90 10.1k).
- **Excess by cause:**
  - TDM waits, about 11.6k: C-slot reuse and store drain in the epilogue,
    the final store drain, first-tile load latency in the prologue;
  - LDS waits, about 7.9k: the `s_wait_dscnt 0` that closes each C chunk
    fill, and full drains in the compact tail;
  - barriers, about 4.5k;
  - SALU, about 4.1k;
  - `ds_store` issue, about 2.7k.

Median transition timeline: the K tail ends at about 4.0k cycles. The last
next-tile stage loads issue just before the first chunk store, and since TDM
completes in order, the first slot-reuse wait (about 650 cycles) also waits
for them. After that, chunks retire every 450–600 cycles (`0x4` waits of
380–460), about 64 B/cycle per CU. Filling a 32 KB chunk takes about 250
cycles (16 `ds_store_b128` per wave plus the drain), about 130 B/cycle of LDS
writes.

The C path is the bottleneck: 256 KB of FP32 per CU per tile, written to LDS
at about 130 B/cycle and drained by TDM at about 64 B/cycle, with only one
next-tile step (64 WMMAs) to overlap, because the VGPR budget allows one
tile's accumulators. At about 64 B/cycle on all 256 CUs, the drain rate
(about 39 TB/s at the 2.4 GHz trace clock) is far above HBM write
bandwidth, so stores land in L2/MALL. Whether the per-CU drain rate is a
TDM-engine limit or an L2 write limit is not measured.

Already measured and not to repeat: static K tail, batched first fill,
next-tile loads after the C stores, wave-specialized handoff, more and
smaller C slots, overlap peel (now default), L2 prefetch.

Open questions, in order:

1. **Store drain limit:** run the same ATT with fewer programs (fewer CUs
   storing at once). If chunk retire time stays at 450–600 cycles, the
   limit is per CU (TDM engine or LDS read); if it drops, it is the memory
   side.
2. **LDS write conflicts:** check the C staging layout's `ds_store`
   partition and bank conflicts (AM reports partition conflicts). About
   130 B/cycle is well below the LDS read rate measured earlier.
3. **Narrower output:** BF16/FP16 C halves LDS fill, TDM drain and memory
   writes. This changes the kernel's output contract (FP32 today).
4. **Direct stores:** write part of C (one quadrant) with
   `global_store_b128` from registers alongside the TDM path, if the drain
   limit is per CU.

Archive (scripts, report; trace bundles are in
`mxfp-wave-imbalance-20261024.tar.gz`):
`/home/ibutygin/tlx/perf-data/mxfp-transition-breakdown-20261024.tar.gz`
(SHA256 `ce1d36b5efaa44235a90bcd53c462071181467a1de20ed6214ff7946bda1902d`).

#### C epilogue experiments (2026-10-24): no gain; the output type stays FP32

**1. Store-drain limit (ATT with fewer programs).** The same default kernel,
compiled for 128 and 64 programs (each CTA then runs 8 or 16 tiles), was
traced on SIMD 0 against the 256-program trace. Cycles per occurrence:

| Programs | Epilogue TDM waits | LDS fill (`ds_store` + `s_wait_dscnt`) | Final-epilogue TDM waits |
| ---: | ---: | ---: | ---: |
| 256 | 1,678 | 725 + 1,207 | 4,605 |
| 128 | 1,285 | 699 + 1,262 | 3,191 |
| 64 | 1,282 | 721 + 1,265 | 3,308 |

Per epilogue `s_wait_tensorcnt`, in order (256 / 128 / 64 programs):

- first slot-reuse wait: 659 / 646 / 619;
- next wait: 146 / 71 / 63;
- the three later `0x4` waits: 231 / 65 / 76, 364 / 208 / 239, and
  255 / 274 / 263.

Device-wide store contention is real but bounded: about 400 cycles per
transition and about 1.4k in the final epilogue, the same at 128 and 64
programs. The remainder is per CU:

- the first slot-reuse wait (about 650 cycles) waits behind the next-tile
  stage-2 loads, issued just before chunk 0 (TDM completes in order);
- the later drain waits are 210–270 cycles each;
- the LDS fill takes about 1.95k cycles per transition at any program count.

**2. LDS partition conflicts in the C fill.** AM (`runs/am-ovl`, 2048³,
16 programs) counts partition write conflicts almost only in the transition
windows: about 0 per sample in the steady loop, and 1,500–2,000 per sample
in the epilogues, about 2 per C `ds_store_b128`. DS bank conflicts are
zero. The C slots (64x128 FP32, rows padded by 16 bytes, 33,792 bytes) sit
at LDS offsets 0, 33,792 and 67,584; waves 0–1 fill rows 0–31 of a chunk
and waves 2–3 rows 32–63.

Temporary `C_PARTITIONED` gave the C slots a `partitioned_shared` layout:
32-row pieces in two partitions, the same `ds_store`/TDM instruction
counts, and 852 VGPRs. The allocator then moved A/B to 0–208,895 and C to
208,896 and up.

- Hardware: bitwise identical; sustained +0.14%, burst −0.17%.
- ATT: the fill is unchanged (`ds_store` 756 + `s_wait_dscnt` 1,157), and
  so is the epilogue (5,126 vs 4,960 cycles).

The AM partition-conflict counts do not show up as fill time on hardware.

**3. Direct stores for one quadrant.** Temporary `DIRECT_C_QUADRANTS=1`
stored quadrant 3 straight from registers (32 `buffer_store_b128` per
wave), staging six chunks through LDS/TDM instead of eight. Chunk count,
refill position and the stage-1 acquire wait were adjusted accordingly.
891 VGPRs, no spills; bitwise identical. Hardware: **+9.5%** sustained
(cycles +12%).

**Conclusion.** The fill is bound neither by partitions nor by device-wide
drain. The remaining per-CU levers:

- **Stage-2 load after chunk 2.** Issue the next tile's stage-2 load after
  the first three C chunks instead of at the end of the tail, so the first
  slot-reuse wait no longer waits on it (up to about 0.4–0.6k cycles per
  transition, about 0.5%). Wait counts that must change:
  - the stage-0 acquire before the overlapped step becomes `LOADS`;
  - reuse waits for chunks 3–5 need `+LOADS` (chunk 5 also the refill's);
  - the stage-1 acquire count stays 12.
- **The final drain** (all CTAs at once).

Patch, ship tarballs, ATT bundles, scripts, logs:
`/home/ibutygin/tlx/perf-data/mxfp-c-epilogue-exp-20261024.tar.gz` (SHA256
`e324b6b00be0e564a5190327c6afc3742411d2a10457108350e5250e81a54415`).

**3b. Direct quadrant issued first (fourth output channel).** Same knob
plus `DIRECT_C_FIRST`: quadrant 0 goes straight from registers before any
LDS chunk, so the three slots, the first slot-reuse wait and the stage-0
refill all shift later. The TDM chunks are numbered over quadrants 1–3 (six
chunks); the refill stays before the last two chunks; the stage-1 acquire
wait is 10.

- Hardware: bitwise identical; **+10.9%** sustained (cycles +14.0%),
  versus +9.5% with the direct quadrant last.
- The asm issues the 32 `buffer_store_b128`, then `s_wait_xcnt 0x0`, then
  the TDM chunk sequence.

The direct store itself is the cost. In the WMMA register layout, each
`buffer_store_b128` writes 16 bytes per lane to 16 different rows (two
16-byte pieces per row): about 8x the memory requests of full-line TDM
writes, about 4k partial-line requests per CU per quadrant. Issuing it
first only moves the transfer wait ahead of the TDM work.

A useful direct path would first need a cross-lane transpose so each
instruction writes whole 128-byte row segments (for example
`ds_bpermute`/permlane, about 128 permutes per wave per quadrant). Not
tried. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-direct-first-20261024.tar.gz` (SHA256
`b9d0e3a084a86d5ec31580a850d2080991ec1a3c1e0208474224cebd43389a30`).

**4. Wave-specialized overlap epilogue (`OVERLAP_WS`, temporary): +11.9%;
the gain is small even before the loop regression.** Waves 0/1 issue
every next-tile input load, both in the tail (stages 0–2) and in the
epilogue (the stage-3 refill, moved earlier to quadrant 2); waves 2/3 issue
every C chunk store. Each role waits only on its own TDM counter:

- C slot reuse: `wait(2)` on the output waves;
- input acquires: exact per-wave counts in the tail and the epilogue.

All barriers stay in common code, with only the TDM issue and wave waits
in uniform role branches; the asm matches. The partitioned A/B layout
cannot be used: a fused TDM load needs two to four members with disjoint
wave masks, and a partitioned member needs an even number of issuing
waves, so waves 0/1 alone cannot fill A and B pieces with multicast. The
variant therefore uses `LDS_PARTITIONED_AB=0`, compared with `np` (the
default with that same setting).

| 8192³ hardware | Sustained | Cycles |
| --- | ---: | ---: |
| `np` (unpartitioned A/B) | +1.34% | +2.31% |
| `OVERLAP_WS` | **+11.92%** | +20.46% |

Outputs are bitwise identical.

- **Why the K loop slowed:** ATT puts most of the loss inside the K loop,
  about 1,020–1,060 vs 847 cycles per step; cluster-barrier waits roughly
  triple. The LLVM IR of the loop is identical, but the steady-loop machine
  regions now exceed the 106-SGPR target (108–115). The pre-RA
  rematerialization stage then runs ("reduce spilling", `-debug-only=
  machine-scheduler` with the local d26 build) and reschedules the
  function. The barrier → TDM → `s_wait_tensorcnt` → barrier rendezvous
  loses its straddling WMMAs, the same failure as codegen `05e432a8` and
  `llvm.fake.use`. What does not fix it:
  - `amdgpu-disable-unclustered-high-rp-reschedule`;
  - recomputing the wave role at each use through an opaque asm.

  The extra SGPRs come from the role-split handoff code. Pinning them, as
  `_mxgemm_handoff_load_stage` does, would be needed.
- **The epilogue itself barely improves:** on output waves, the first
  slot-reuse wait drops from about 920 to about 610 cycles; another reuse
  wait reaches about 1,140. So a C store takes about 600 cycles or more to
  complete even with no input loads in that wave's queue: the latency is
  physical, not counter ordering. Input waves wait about 2.7k cycles per
  transition at barriers for the output waves. Even with the loop fixed,
  the gain is bounded by a few hundred cycles per transition (≤0.5%).

Archive (patch, ship, ATT bundles, scheduler logs, hardware log):
`/home/ibutygin/tlx/perf-data/mxfp-overlap-ws-20261024.tar.gz` (SHA256
`11ba61886673de8d0b86ffe02cc6932318ace9d32d514d59029b79304a616ba4`).

**5. C stores on the scale waves (`OVERLAP_SW`, temporary): +4.7%.**
Today all waves issue partial-fusion loads: A data on waves 0/2, B data on
1/3, scales likewise, two TDM instructions per wave per stage. Plain 4-way
fusion (A→w0, B→w1, As→w2, Bs→w3) does not fit partitioned A/B, since a
partitioned member needs two issuing waves.

**Version 1 (handoff loads only).** Next-tile loads use
`[A pieces (w0,w1), As (w2)]` and `[B pieces (w0,w1), Bs (w3)]`; waves 2/3
issue the C stores. Problems:

- fused members must have the same descriptor rank, so the scales need a
  3D view: descriptor `[M/256, 2, K/32·128]`, LDS slot reinterpreted as
  `[1, 2, 512]` with identical padding;
- per-role wait counts are needed in the tail and epilogue;
- the new descriptor groups stay live through the K loop: live-through-
  unused SGPR units 53 → 106–123. Building them after the loop from opaque
  pointers and shapes did not help; 70–96 SGPR spills.

**Version 2 (measured).** Every load in the kernel, K loop included, uses
`[A pieces (w0,w1), As (w2,w3)]` and `[B pieces (w0,w1), Bs (w2,w3)]`, one
scale row per wave. Every wave still issues two instructions per stage, so
all existing wait counts stay valid. Only the C slot-reuse waits (waves
2/3) and the epilogue's final stage-1 acquire (4 on waves 0/1, 12 on waves
2/3) are per role.

- Codegen: 107 SGPRs, 11 spilled; two of the three loop steps lose their
  WMMA overlap.
- Hardware: bitwise identical; sustained **+4.69%** (cycles +7.5%).

ATT (SIMD 0 = data wave, SIMD 2 = scale/C wave):

- **First slot-reuse wait:** 943 → 810 cycles. With no data loads in the
  store waves' queue it still takes about 800 cycles: C-store latency
  dominates.
- **Later reuse waits:** 394 / 320 → 412 / 645. With C issued by two
  waves instead of four, each wave stores twice the bytes per chunk and
  the chunks drain more slowly.
- **Data waves:** about 3.1k cycles per transition at barriers waiting for
  the store waves, so the epilogue grows from about 5.0–5.6k to 6.1k.
- **Steady step:** about 880–895 vs 843–866 cycles (3D scale descriptors,
  SGPR spills).

Conclusion: in the current kernel, separating the TDM counters gains at
most about 100–300 cycles per transition. Spreading C stores over fewer
waves slows the drain, and the default sits at 104 of 106 SGPRs, so any
new descriptor group breaks the static K-loop schedule. Archive (both
patches, ship, ATT, scheduler and liveness reports, hardware log):
`/home/ibutygin/tlx/perf-data/mxfp-scale-wave-split-20261024.tar.gz`
(SHA256 `0f61cf23146f430bb28bf8c4fa3fce88566583ea6e16dbfdf28980fa3b607efa`).

#### FP16 output (`OUT_DTYPE="fp16"`, 2026-10-24): kept as an option

**Kernel.** C's element type follows `c_ptr`:

- `_mxgemm_persistent_store` converts the pinned FP32 accumulators in
  registers (`v_cvt_pk_f16_f32`, round to nearest even) before staging or
  direct stores; non-persistent direct stores convert too;
- dedicated staging slots use C's type, so a 64x128 FP16 chunk is 16 KiB;
- the four-buffer default chunk is 16 KiB of columns (64 FP32, 128 FP16);
- a static assert rejects non-FP32 C for output schedules that reinterpret
  A/B input stages as FP32 C buffers (BK256 output reuse, i.e. the MX8xMX4
  default; register pipeline; output-tail reuse; `ab-reuse`).

**Launcher, CLI and bench.** `mxgemm_tdm_pipelined(OUT_DTYPE=...)`,
`matmul(config={"OUT_DTYPE": ...})`, the CLI's `--out_dtype`, and
`bench.py --out-dtype` (which rejects the MX8xMX4 BK256 default). FP32 code
is unchanged: the 8192³ and 16384³ FP32 assembly is identical.

FP16 codegen at 8192³:

- the steady loop keeps its WMMA overlap around the rendezvous and is 24
  instructions shorter;
- 104 SGPRs; 913 VGPRs (conversion temporaries), no spills;
- 128 instead of 256 `ds_store_b128`, 256 `v_cvt_pk_f16_f32`;
- LDS 267,432 instead of 316,584 bytes with three slots.

**Correctness.**

- New `test_mxgemm_fp16_output_ring_phase`: static (6 and 3 slots),
  dynamic (6 slots), four buffers, and direct stores, clusters 1 and 4,
  checked against the FP32 reference with rtol 1e-3. Passes under FFM,
  together with sampled FP32 ring-phase tests (29 passed).
- `test_mxgemm_fp16_output_rejects_input_stage_reuse` passes (CPU-only).
- Hardware: every FP16 output equals the FP32 kernel's output rounded with
  `.half()`, bit for bit (8192³ and 8192²×4096).

**Hardware sustained timing** (`hw_power2.py`, four rounds, change against
FP32 output in the same run):

| Shape | FP32 | FP16, 3 slots | FP16, 4 slots | FP16, 6 slots |
| --- | ---: | ---: | ---: | ---: |
| 8192³ | 0.15033 ms | −3.17% | −4.12% | **−4.44%** (0.14366 ms, 7,654 TFLOPS) |
| 8192²×4096 | 0.08360 ms | −5.60% | −6.40% | **−6.73%** (0.07798 ms) |

| 16384³, four buffers | FP32 | FP16, 2 × 64x64 | FP16, 2 × 64x128 | FP16, 3 × 64x64 |
| --- | ---: | ---: | ---: | ---: |
| Sustained | 1.17928 ms | +1.35% | **−3.72%** (1.13544 ms) | −0.77% |

Energy per launch drops by the same fractions (all runs at the power
cap). `bench.py` therefore uses six FP16 C slots for the three-buffer ring,
and the kernel's four-buffer default is two 64x128 FP16 slots. MX8xMX4 FP16
output would need the A-reuse C layout rebuilt for 2-byte elements (A's
69,632-byte slot stride); not done. Archive (ships, harness, logs):
`/home/ibutygin/tlx/perf-data/mxfp-fp16-output-20261024.tar.gz` (SHA256
`d0f296f9d42f78a4c5b2833ceab8ec480113ea298c751f54f554aa6d8bb0262b`).

**FP16 output for every variant (2026-10-25).**

*Aliased C buffers.* C buffers that alias input ring slots keep the input
slot pitch: FP16 allocates `C_ROW_SCALE` = 2 times the rows per slot with the
same element padding (exactly the FP32 bytes), and `_mxgemm_c_slot` slices
the leading rows. This covers:

- MX8xMX4 BK256 output reuse, with and without cross-tile prefetch;
- output-tail reuse (`ca`/`cb`), the `ab-reuse` B halves, and the streamed
  kernel's A alias.

Single C buffers placed inside one input stage (deferred C refill,
`ab-reuse` A quadrants) just change type.

*Register pipeline.* It reinterprets a free BK256 stage. TDM store padding
must equal the panel width, and the reinterpret must keep the stage's byte
padding (256 B + 16 B), so FP16 stores two 256x128 panels per tile with
`[128:+8]` padding instead of four 256x64 FP32 panels.

*Other kernels and fixes.*

- Warp and operand pipelines (dedicated output tiles) convert before
  `local_store`.
- Only the streamed kernel's `ASYNC_OUTPUT` (FP32 lane addressing) rejects
  FP16.
- Fixed in passing: the warp-pipeline launcher's absolute import broke
  package use.
- `SCHED_MODE_2` now launches with `enable_wmma_queuing` (see the test
  repair notes below).

*Bench.* `bench.py --out-dtype fp16` now works for both variants (only
`--async-output` is rejected), and the final table has an `out` column.

*Code and tests.*

- FP32 code is unchanged: MX8xMX8 and MX8xMX4 bench defaults assemble
  identically to HEAD.
- MX8xMX4 FP16: 922 VGPRs, no spills; 64 instead of 128 `ds_store`; same LDS
  (C lives in A).
- FFM: `test_mxgemm_fp16_output_schedules_ring_phase` (14 schedules) and
  `test_mxgemm_fp16_output_nonpersistent` (direct, warp pipeline, two
  operand pipelines) pass, plus an FP32 regression sample of the touched
  paths: 225 of 225 tests pass (register, output tail reuse, A8W4 C
  staging, streamed, compact K tail, early C stores with AB reuse, wave
  specialized, overlap peel, operand pipeline ring; `pytest -n 48`, 3m23s).

Hardware, MX8xMX4 bench default (BK256, early C stores, split cluster
barrier); FP16 outputs equal the FP32 kernel's output rounded to FP16, bit
for bit:

| Shape | FP32 | FP16 | Change |
| --- | ---: | ---: | ---: |
| 8192³ | 0.13486 ms | 0.12855 ms | −4.68% |
| 8192²×4096 | 0.07758 ms | 0.07083 ms | −8.70% |
| 16384³ | 1.02583 ms | 0.99883 ms | −2.63% |

The remote host rebooted before this run (its `/tmp` and the remote ship
directory were recreated), so compare these MX8xMX4 numbers only with
each other. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-fp16-all-variants-20261025.tar.gz`
(SHA256 `a2f5f11410db6dc579c8b60cb42bf08dafef5e05c476743858ab7134d19a6955`).

#### Post-reboot slowdown is the machine, not the kernel (2026-10-25)

The pre-reboot `ship49` A8W8 binaries (2026-10-24), rerun unchanged with the
same harness (`hw_power2.py`, four rounds), are 6-7% slower. Cycles per
launch are unchanged; the sustained clock at the 2500 W cap dropped by the
same fraction, and energy per launch rose:

| Shape (FP32) | Before | After | Time | Clock | Cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192³ | 0.15003 ms, 1793 MHz | 0.15989 ms, 1685 MHz | +6.6% | −6.0% | +0.2% |
| 8192²×4096 | 0.08356 ms, 1878 MHz | 0.08849 ms, 1786 MHz | +5.9% | −4.9% | +0.7% |
| 16384³ | 1.17881 ms, 1632 MHz | 1.26639 ms, 1512 MHz | +7.4% | −7.4% | −0.5% |

Short bursts at both 8192 shapes are 11% slower too, and with light load
the shader clock now sits at ~2200 MHz instead of ~2420 MHz. The
current HEAD build of the bench default is instruction-identical to `ship49`
at all three shapes (only DWARF line numbers differ), and the outputs match
bit for bit.

Machine changes during the downtime: the host was reprovisioned at
01:07 UTC on 2026-10-25. `amdgpu-dkms` and `amdgpu-dkms-firmware` were
reinstalled as a new build of the same version (7.1.0.31300009, package
build 2410483 -> 2415424), and the board now reports IFWI 00207988 (built
2026/10/17, part 113-M4500001-001). The 2026-10-21 snapshot showed IFWI
00205629 (2026/10/01, part -900B), SCLK levels 500/2400 MHz (now
250/2234/2400), DF 1250/1900 MHz (now 500/1950), and PPT1 3750 W (now
3700 W). No snapshot from the 2026-10-24 boot exists, so the IFWI may have
changed at an earlier reboot, but the driver/firmware package swap is
certain. Compare only post-reboot numbers with each other. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-reboot-check-20261025.tar.gz` (SHA256
`ffb74e6a17f05bb87a572ccc1d9f94606e718afd8213e5a8d78bf9b1a1f0a5d4`).

#### TLX AMD test repair (2026-10-25)

All TLX AMD pytest files pass under FFM (1161 passed, 502 skipped across
`test_tlx_amd_{gfx1250,gfx942,gfx950,hip,fa_bwd_gfx950}.py`), and
`test_tlx_codegen.py` passes (746 passed, 8 xfailed); the mxgemm codegen
tests previously failed in 125 cases. Fixes:

- `SCHED_MODE_2`: the persistent and streamed kernels called the removed
  `tlx.amd_set_wave_sched_mode`. The host wrapper now passes
  `enable_wmma_queuing=SCHED_MODE_2` at launch (upstream `33856162ca`), and
  the kernels no longer take a `SCHED_MODE_2` constexpr.
- `tlx.barrier_arrive` read `options.cluster_dims`, which `HIPOptions`
  lacks, so every AMD mbarrier arrive failed to compile. It now reads the
  cluster dimensions only for `cta_mask` (NVIDIA-only).
- Codegen helper: it passed handoff constants to the streamed kernel, which
  does not declare them.
- TDM hazard waits: the pinned backend's `GCNHazardRecognizer::fixTDM`
  inserts `s_wait_tensorcnt 10` directly before a TDM instruction unless an
  earlier wait already bounds the count. It caps the TDM queue at ten and
  never drains the transfers the schedule checks reason about, so the
  mxgemm codegen checks strip exactly those waits
  (`_strip_tdm_hazard_waits`). This fixed the output staging, A8W4 C
  staging, first-use prefetch, wave-specialized handoff, and operand
  pipeline checks.
- Known drift (xfail): clustered four-stage streamed operands now get
  identity `v_perm_b32 x, y, y, 0x3020104` copies in the steady loop (56 per
  loop) and, with async output, scratch spills. The kernel source from
  `4a940284a0`, where the test was last updated, does the same, so this
  comes from the code generator. Streamed operands are not a bench default.

#### Address and scalar-work scan of the default kernel (2026-10-25)

**Method.** `scan.py` walks the 8192³ default asm (unchanged by the FP16
commit) and charges each instruction its issue gap from the four
SIMD-select ATT runs. ATT does not record `s_set_vgpr_msb`; its issue time
lands on the preceding instruction. Totals per wave, about 247k cycles per
wave:

| Region | Class | Static | Executions | Gap cycles |
| --- | --- | ---: | ---: | ---: |
| Steady loop | SALU | 59 | 4,084 | 7,643 |
| Steady loop | VALU | 10 | 800 | 1,521 |
| Steady loop | `s_nop`/`v_nop` | 3 | 240 | 400 |
| Steady loop | `s_set_vgpr_msb` | 172 | — | not traced |
| Transitions (×3) | SALU / VALU | 339 / 89 | 1,439 / 684 | 2,481 / 900 |
| Prologue | SALU / VALU | 212 / 30 | 242 | 315 |

**Steady loop, per iteration (three K steps):**

- **TDM descriptor regroup copies** (`s_mov sN, s29`, about 15 per step):
  SGPR-pressure allocation copies, already measured at most ~0.1%
  ("Redundant scalar work").
- **`k % 8` cluster-barrier test:** `s_add` + `s_and` + `s_cbranch` per
  step. The branch waits about 10 cycles on the `s_and` (about 30 cycles
  per iteration). It sits right before the step's `s_wait_dscnt 0`, which
  waits for LDS loads anyway, so most of it is hidden.
- **LDS address adds** (about 6–8 VALU per step) and three `v_nop`:
  pinning them out of the loop was measured at +0.45% ("Live values").
- **Loop control:** about 11 cycles per iteration.

**Transitions, about 1.1k SALU/VALU cycles each:**

- about 400: the compact K-tail loop's 32-VGPR `v_dual_mov` copy blocks
  (a static tail removes them, +0.2% net, measured);
- tile coordinates computed twice per tile;
- slot `% 3` selects;
- per-chunk C descriptor math.

All but the copy blocks were measured earlier at under 0.1%.

**New candidate tested: the double workgroup barrier on cluster steps.**
Every eighth step runs `cluster_barrier()` (`s_wait_dscnt 0` → workgroup
barrier → wave 0 `s_barrier_signal -3` → wait), and then the refill's
release barrier that Membar inserts at the join. Temporary
`MERGED_CLUSTER_RELEASE`:

| Variant | What changes | 8192³ | 8192²×4096 |
| --- | --- | ---: | ---: |
| `mcr` | One explicit, unconditional release barrier before the `k % 8` branch, then arrive/wait. Signal and wait become adjacent (no WMMAs between), exposing the barrier round trip every step | +2.50% | +1.84% |
| `mcr2` | Keep Membar's release barrier; on cluster steps only arrive/wait (drop `cluster_barrier`'s leading workgroup barrier). Loop schedule otherwise unchanged | +0.48% | −0.13% |

Outputs are bitwise identical. The leading workgroup barrier is not needed
for LDS safety, but aligning each CTA's waves before wave 0 signals the
cluster helps multicast matching more than it costs.

**Conclusion:** after this scan, no address or scalar work in the default
kernel is removable for a measurable gain. The remaining candidates are
each at most ~0.1% or were already measured as losses. Archive (`scan.py`,
per-instruction gaps, patch, ship, hardware log):
`/home/ibutygin/tlx/perf-data/mxfp-addr-scan-20261025.tar.gz` (SHA256
`54bc799e1a807072497c860eb97552362181e398a6e38cb97b6fea49d56f7cf9`).

### Merged handoff barrier and four-stage static ring (2026-11-05)

Branch rebased on `main` `fa6bf57d99` with the `ba034c2f` AMD codegen
(facebookexperimental PR #4734). Kernels were compiled locally through the
JIT path (same specialization as `bench.py`) and run on the remote host from
their code objects; all timings are interleaved rounds of `bench.py`-style
sustained timing at the 2,500 W cap, after FP32-reference checks on the
device. Archive of the starting profile:
`/home/ibutygin/tlx/perf-data/mxfp-a8w8-f16-8k-profile-20261105.tar.gz`
(SHA256 `846574128f5d8f46998282e9a17d7f79d7e968de93968b7f6f125e57083a2ab7`).

Starting point, A8W8 FP16 8192³ bench default: 0.1490 ms (7,379 TFLOPS) at
~1,747 MHz. ATT (16 SEs, one wave each): 249k cycles per wave, steady loop
86%, 887 cycles per K128 step (median 822; floor 512): LDS reads 222
(`ds_load` 147, `s_wait_dscnt` 86), workgroup barrier 104, TDM wait 40,
cluster barrier 35, SALU 24. Non-steady time ~35k cycles per wave (four
tiles). The gluon A8W8 tutorial (non-persistent, 4x4 clusters, four-stage
ring, one split barrier per step with relaxed loads) runs its steady loop at
680-700 cycles per K128.

`MERGED_HANDOFF_BARRIER` waits for the next slot before the refill and keeps
one workgroup barrier per static step (relaxed `local_load`s, explicit
`tl.debug_barrier()` next to the refill so register-only WMMAs can fill its
window; five do). Results:

| Variant (FP16 out) | Shape | vs current default |
| --- | --- | ---: |
| Merged, three stages | 8192³ | +23.3% (+36% cycles) |
| Four stages, dynamic slots, 2 C slots | 8192³ | +0.4% |
| Four stages, static slots | 8192³ | -1.0% |
| Four stages, static, merged | 8192³ | **-1.6%** (repeat -1.6%) |
| … with grouped prefetch off (5-WMMA window) | 8192³ | +0.5% vs merged |
| … with cluster barrier interval 16 | 8192³ | +3.6% vs merged |
| Four stages, static, no merge, grouped prefetch off | 8192³ | +0.2% vs merged |
| Four stages, static, merged | 8192²×4096 | **-2.2%** |
| Four stages, static, merged | 16384³ | +0.8% |
| Four stages, static | 16384³ | +0.2% |
| Four stages, static, merged | 32768³ | +7.6% |
| Four stages, static | 32768³ | +1.5% |

ATT of the three-stage merged kernel: 1,076 cycles per step (TDM wait 40 ->
128, cluster barrier 35 -> 138, workgroup barrier 104 -> 138; LDS 222 ->
184). Waiting for the next slot before the refill makes the refill depend on
input arrival and the slowest wave, which shortens the ring's lead; the
fourth stage restores it at K <= 8192. At K >= 16384 (memory-latency bound)
even four stages do not cover it. With four stages, grouped prefetch uses
`sched_group_barrier`s, which disable LLVM's barrier-latency mutation, so the
merged barrier has no WMMAs in its window; turning grouped prefetch off
fills the window but is slower. Four stages and two FP16 C slots use
321,768 bytes of LDS; FP32 C slots do not fit.

`bench.py` now uses four stages, static slots, two C slots, the overlap peel,
and the merged barrier for FP16 MX8xMX8 below K=16384. FP32 output and
K >= 16384 keep their previous defaults.


### Interleaved C fill in the overlapped peel (2026-11-05): neutral

`INTERLEAVE_C_FILL` (opt-in, `--interleave_c_fill {1,2}`) changes only
the `OVERLAP_PEEL` epilogue: each quadrant is converted to the C type
before its region, so its next-tile K0 dot can reuse the accumulator
registers while the converted chunks are staged. The C fill, its TDM store,
and that dot then share one scheduling region.

- Mode 1 alternates each chunk's eight `ds_store_b128` with eight WMMAs
  before the fill barrier. q0 issues all remaining K0 operand reads (`nb01`,
  `na10`), so none drain at a slot-reuse barrier.
- Mode 2 issues each chunk as a write burst, the fill barrier, the TDM store,
  and then eight WMMAs to cover the store before the next slot-reuse wait.
  q0 reads the remaining operands between its two chunks, because the fill
  barrier fences LDS reads. The next-tile refill gets its own VMEM group.
  The VMEM-write mask (`0x40`) does not match `tensor_store_from_lds`; the
  generic VMEM mask (`0x10`) does.

The asm for both modes has the intended shape in all eight chunks: 929
VGPRs, 321,768 B LDS, no scratch, and an unchanged steady loop (mode 2 has
four extra `s_set_vgpr_msb`). The FFM ring-phase test passes (3/4 buffers,
FP16/FP32, clusters 1/4).

Hardware, FP16 bench default (four stages, static, merged), interleaved
rounds, after FP32-reference checks:

| Variant | 8192³ | 8192²×4096 |
| --- | ---: | ---: |
| Mode 1, first run | -0.08% | +2.0% |
| Mode 1, repeat | +0.04% | +0.09% |
| Mode 2 | +0.11% | +0.11% |

The first 4096 run's baseline was 0.0778 ms; in the repeat all three
kernels ran at 0.0802 ms. Run-to-run drift is larger than any variant
difference, so both modes are neutral. ATT was not collected: the shared
GPU lock stayed busy for more than 30 minutes.

Rearranging work inside the C-store window does not change the transition
time. This matches the 2026-10-24 epilogue study: the window is bound by the
two-slot TDM store chain (slot reuse needs the store issued one chunk
earlier to complete) and device-wide drain, not by the LDS fill or by WMMA
placement. Further gains need more output bytes in flight, so more
or larger C slots (no LDS left with four stages), or C stores spread
across the next tile's K loop.


### L2-only lookahead stage with three stages and six C slots (2026-11-05): slower

`L2_LOOKAHEAD=d` (opt-in, `--l2_lookahead d`, static-slot BK128 only) adds a
"fake" ring stage with no LDS slot. Each static step, after the `c10` dot,
it L2-prefetches the stage `d` past the TDM refill. Even stages only (one
256-byte line holds two BK128 steps of a row). Within a cluster, ownership
of each shared A or B tile rotates per pair, so each CTA issues an A and a
B prefetch about every eight steps. Speculative prefetches, block-uniform
branches. With three stages the freed LDS holds six FP16 C slots (319,656
B). Cost in the three-step steady loop: 667 -> 800 instructions (+85 SALU,
+21 VALU, 10 `global_prefetch_b8`, +7 branches), 901 -> 907 VGPRs, no
scratch.

Hardware, FP16 8192³, four interleaved rounds after FP32-reference
checks, against the current default (four stages, merged barrier, two C
slots: 0.14635 ms):

| Variant | Time | Cycles | Clock |
| --- | ---: | ---: | ---: |
| Three stages, six C slots (old default) | +1.81% | +3.16% | 1,743 MHz |
| … + lookahead 2 | +11.07% | +16.74% | 1,807 MHz |
| Three stages, merged, six C slots | +25.62% | +40.52% | 1,922 MHz |
| … + lookahead 1 | +26.38% | +40.72% | 1,914 MHz |
| … + lookahead 2 | +21.90% | +33.77% | 1,886 MHz |
| … + lookahead 4 | +22.15% | +34.23% | 1,889 MHz |

The prefetch does shorten the merged three-stage stalls (cycles +40.5% ->
+33.8% at d=2), but recovers less than a fifth of the loss. On the
unmerged three-stage ring it costs about 9% (energy +11.1% vs +1.8%): even
deduplicated and paired, the prefetch requests plus their address and
guard code cost more than the stalls they hide, at the power cap. An L2
stage does not replace an LDS stage: the merged barrier needs the refill's
data in LDS one step earlier, and an L2 hit still leaves the TDM transfer
on the critical path. Four real stages with two C slots stay the FP16
default.

ATT (16 SEs, one wave each, 3,840 steady steps; profiled runs are not
power-capped) of the three-stage ring without and with lookahead 2: the
median steady step grows 834 -> 924 cycles (mean 858 -> 972, +114). Per
step:

- Guard code, +66: `s_and_b32` (+26) and `s_cbranch_vccnz` (+22) dominate.
  Each step evaluates the pair and owner predicates for A and B and turns
  them into `s_and_b32 vcc_lo, exec_lo, sN` + `s_cbranch_vccnz`. The block
  sits between the `c10` and `c11` WMMA groups, and the matrix pipe idles
  while it runs: in-order issue does not hide SALU between WMMA groups.
- LDS reads, +36: `ds_load_2addr_b32` (scales) +17, `s_wait_dscnt` +15.
- TDM waits, +14.
- Address VALU and the prefetches themselves, about +8.
- Barrier waits unchanged in total (cluster +62, workgroup -66): owners
  lag on their prefetch steps and the skew moves to the cluster rendezvous.

So most of the loss is guard code on the WMMA critical path, not prefetch
traffic. Even without guards, the LDS and TDM waits alone (+50 cycles per
step, about +6%) exceed anything the lookahead could recover on this ring.
**Branch-free split lookahead (`L2_LOOKAHEAD_SPLIT`).** Every CTA
prefetches its own slice every step: the C CTAs sharing an A tile split its
rows, each takes half of its slice per step by step parity, so each 256-byte
line is requested once per pair (B likewise). Waves 0-1 cover A and 2-3 B;
lanes l and l+16 of a wave share a line. The per-lane base is hoisted out of
the K loop; each step adds one scalar offset: 6 SALU, one
`v_add_nc_u64` and one `global_prefetch_b8` per wave, no branches (loop
511 -> 538 instructions per three steps). The descriptor prefetch op always
covers its whole block, so this uses elementwise inline asm on a pointer
tensor.

Inline-asm pitfall: `global_prefetch_b8` reads its address VGPRs after
issue. LLVM inserts `s_wait_alu depctr_vm_vsrc(0)` before overwriting a VMEM
source only for instructions it knows, so a later write to the address
registers corrupted the address. Symptom: intermittent
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`, every launch for the
four-stage build, rarely for three stages; FFM does not catch it. A
`depctr_vm_vsrc(0)` wait inside the asm fixes it (2,000-launch stress clean).

Hardware, FP16 8192³ (0.14665 ms default):

| Variant | Time | Cycles |
| --- | ---: | ---: |
| Three stages, six C slots | +1.67% | +2.99% |
| … + split lookahead 2 | +6.23% | +9.05% |
| Four stages (default) + split lookahead 2 | +0.70% | +0.34% |

The merged three-stage split builds (lookahead 2 and 4) passed one check and
then hung in a 2,000-launch repeat (not investigated; they were +22% with
the branched prefetch).

ATT of three stages with and without the split lookahead (profiled, so not
power-capped, ~2.4 GHz): the prefetch is faster there. Mean steady step 858
-> 834 cycles, wave 241.5k -> 236.9k (-1.9%). Barrier waits fall 141 -> 73
per step (workgroup 108 -> 38, cluster unchanged), so the guard skew is
gone and the earlier input arrival shortens the slot waits. Costs: SALU +11,
the prefetch with its `vm_vsrc` wait +9, WMMA issue +18, TDM waits +3; LDS
unchanged. At the 2,500 W cap the same kernel takes 9% more cycles.

Telemetry under the cap (`run_ab2.py`: amd-smi clocks, usage and throttle
counters per timing window, four rounds, all at 2,500 W with the PPT limit
active 99.5-100% of the time):

| Kernel | Time | Cycles | GFX MHz | FCLK MHz | UMC activity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four stages (default) | 0 | 0 | 1,717 | 1,761 | 12% |
| … + split lookahead 2 | +0.57% | +0.19% | 1,710 | 1,757 | 12% |
| Three stages | +1.77% | +3.14% | 1,741 | 1,782 | 12% |
| … + split lookahead 2 | +6.46% | +9.42% | 1,765 | 1,797 | 11% |
| … + branched lookahead 2 | +10.79% | +16.45% | 1,804 | 1,811 | 11% |

The amd-smi "cycles" above are time x sampled clock, not counts.
`CYCLE_PROBE` (opt-in, `--cycle_probe`) records each wave's
`s_get_shader_cycles_u64` and 100 MHz realtime at kernel start and end
(stored at once, so nothing stays live), timed interleaved with unprobed
builds under the cap (`run_probe.py`). Medians over 1,024 waves:

| Kernel | Probed vs unprobed time | Wave cycles | Wave µs | Effective MHz | amd-smi MHz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Four stages (default) | +0.09% | 225.5k | 138.9 | 1,623 | 1,715 |
| … + split lookahead 2 | +1.58% | 237.8k | 144.1 | 1,647 | 1,723 |
| Three stages | +0.21% | 228.8k | 138.9 | 1,645 | 1,736 |
| … + split lookahead 2 | -1.43% | 240.4k | 144.3 | 1,671 | 1,750 |

- The effective clock (cycles / realtime) is about 5% below the amd-smi
  sample median for every kernel; relative comparisons are unaffected.
- Under the cap the prefetch does add cycles: +5.4% (four stages) and +5.1%
  (three stages) per wave, at a 1.5% higher effective clock. The probe
  shifts the prefetch builds by up to 1.6% (their code changed slightly),
  so these deltas are approximate; the default and three-stage builds are
  unperturbed.
- In the profiled traces (~2.4 GHz) the same kernels take more cycles per
  wave (default 235.1k, three stages 241.5k) and the prefetch reduces them
  (229.9k, 236.9k). Memory latency is roughly fixed in time, so it costs
  more cycles at the higher clock, where hiding it pays. At ~1.65 GHz the
  kernel is less latency-bound and the prefetch's extra requests cost more
  than they hide. Judge such changes at the cap, by time.

ATT of the four-stage default under the same profiling conditions: steady
loop 4 steps per iteration, median 767-782 and mean 754-811 cycles per
step (lds 243, workgroup barrier 68, TDM 46, SALU 38, VALU 28, WMMA issue
322; the cluster rendezvous is outside the steady block); wave 235.1k
cycles, against 241.5k (three stages) and 236.9k (three stages + split
lookahead). A schedule change that wins in an uncapped trace can still
lose at the cap; judge prefetch-like changes only by sustained timing.

Archive (knob patch, ships, hardware log, ATT bundles, phase reports):
`/home/ibutygin/tlx/perf-data/mxfp-l2-lookahead-20261105.tar.gz` (SHA256
`ac634a2b80fd1afcf86fc2ae1385469ee0be89efc5e3a0184a55fde3ea7152db`).


### Scalar and vector work in the four-stage K loop (2026-11-06)

Steady loop of the FP16 8192³ default (four static steps per iteration,
1,088 instructions): per K step 64 WMMAs, 68 LDS ops, 25 waits/barriers/TDM,
20 SALU, 4.2 VALU, and 90.5 `s_set_vgpr_msb`. Per iteration:

| Work | Count | Needed? |
| --- | ---: | --- |
| Fused TDM group-0 quad assembly (`s_mov_b64` x14, `s_mov_b32` x22) | 36 | ~10-12: each load copies the whole base quad with two `s_mov_b64`, then overwrites three dwords |
| Zero-register restores (`s_mov_b32 sN, s29`) | 14 | No: SGPR pressure (105/106) reuses the zero descriptor dwords as temporaries |
| `k % 8` cluster test (`s_add_co`, `s_and 7`, branch per step) | 12 | Mostly no: with four steps per iteration only one unrolled step can hit k % 8 == 0 |
| Global address advances (`s_add_nc_u64`, `s_lshl`) | 10 | Yes |
| Loop control | 3 | Yes |
| LDS operand addresses (`v_add3_u32`, `v_add_nc_u32`) | 17 VALU | Loop-invariant (static slots); recomputed each iteration, plus most of 26 `s_wait_alu` |

**Static cluster steps (`STATIC_CLUSTER_STEPS`, `--static_cluster_steps`).**
The unrolled loop starts at PEEL and steps by NUM_BUFFERS, so step j can hit
a rendezvous only if gcd(NUM_BUFFERS, interval) divides PEEL + j +
NUM_BUFFERS; other steps drop the run-time test. Loop SALU 80 -> 61 per
iteration (tests 12 -> 2, zero restores 14 -> 8), 1,088 -> 1,055
instructions, VGPRs 929 -> 940. FFM ring-phase test passes (intervals 2/4/8,
3/4 buffers, 6-21 K steps). Hardware (four rounds, FP32-reference checks):
8192³ -0.23% (pairs -0.12 to -0.31), 8192²x4096 +0.28% (pairs +0.13 to
+0.35). Neutral overall; kept opt-in.

**Descriptor quad copies (LLVM-IR overrides).** Rewriting each group-0
build as a shuffle with the 64-bit address, or as a fresh build from
scalars (`poison` base, dword 0 extracted), produced identical machine code:
the DAG combiner restores the insert-into-base-vector form and the
coalescer emits the same whole-quad copies. Removing them needs an AMDGPU
backend change (REG_SEQUENCE / copy coalescing) in the pinned codegen; at
most ~25 SALU per iteration, likely under 0.1%.

Not pursued: hoisting the LDS address VALU (measured +0.45% on the
three-stage kernel in 2026-10: it shifts VGPR pressure and the schedule)
and `s_set_vgpr_msb` (bank-aware allocation is an LLVM change; cost
untraced).


### K-loop profile, refill overlap, per-wave skew and LDS peak (2026-11-06)

**Profile** (ATT of the FP16 8192³ default, uncapped, 3,584 steady steps):
754 cycles per K step against the 512-cycle WMMA floor. WMMA plus
`ds_load` issue is 507; WMMAs average 5 cycles to the next issue, so pipe
time is charged to the interleaved `ds_load`s and adds. WMMA-pipe idle per
step (235), by the instruction it waited behind: workgroup barrier 65, LDS
68 (`ds_load` backpressure 30, `s_wait_dscnt` 39), TDM 51 (next stage not
landed 26, second refill load behind the first 26), SALU ~30 (mostly the
descriptor-quad copies), VALU 19 (rematerialized LDS addresses; the IR has
them hoisted out of the tile loop, the backend sinks them back). The
`s_wait_tensorcnt 0xa` before each second refill load is inserted by LLVM's
wait pass, not the kernel.

**Refill overlap (`REFILL_OVERLAP`, `--refill_overlap`).** Mode 1
interleaves the refill's descriptor SALU and TDM issue with the first C10
WMMAs (sched groups). Mode 2 (removed; superseded by modes 3 and 4) split
the step barrier with inline asm
(`s_wait_tensorcnt 4; s_wait_dscnt 0; s_barrier_signal -1`, C10 WMMAs,
`s_barrier_wait -1`) and overlaps the refill and both next-slot reads with
C11. Membar does not recognize the asm as a barrier: an explicit
async-TDM wait before it makes Membar add a full barrier (so the TDM wait
lives in the asm), and the refill still gets a Membar barrier because
dynamically indexed slots look like a TDM write-after-write. FFM ring-phase
test passes (20 cases). Hardware, four rounds:

| Variant | 8192³ | 8192²x4096 |
| --- | ---: | ---: |
| Mode 1 | +0.01% | -0.07% |
| Mode 2 (split + Membar barrier) | +2.96% | +1.39% |

**Per-wave skew.** Single-SIMD ATT runs of the same binary (one wave per
workgroup each; `--simd-select` acts as a SIMD index mod 4), per K step:

| Wave | Step | Barrier | TDM wait | LDS wait | Wave cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIMD 0 (two runs) | 722 / 785 | 20 / 30 | 63 / 113 | 59 / 62 | 226k / 244k |
| SIMD 1 | 724 | 49 | 40 | 55 | 230k |
| SIMD 2 | 717 | 26 | 53 | 57 | 228k |
| SIMD 3 | 754 | 68 | 46 | 58 | 235k |

Barrier plus TDM wait is 80-140 cycles per step on every wave: each step
waits for the next stage. Wave 0 waits longest for its own TDM loads and
arrives last; the others absorb that at the barrier. LDS waits and issue
are the same on all waves.

**LDS read peak** (`lds_bw.py`: 256 CTAs, one per CU, contiguous
`ds_load_b128`, XOR-reduced, per-wave shader-cycle counters): 217 B/clk
per CU with 4 waves, 260 with 16 waves (8 waves: 99.6, an unexplained
outlier). The default reads 139 KB per CU per K step (68 `ds_load_b128` per
wave, 4 waves), so at 217 B/clk LDS alone needs ~640 cycles per step,
above the 512-cycle WMMA floor. The measured 717-785 is within ~12-20% of
that LDS bound: the K loop is mainly LDS-read bound at four waves, and the
remaining slack is next-stage arrival. Larger gains need fewer LDS bytes per
WMMA (more register reuse per wave) or more LDS throughput (more waves), not
schedule changes.
Archive (traces, logs, microbenchmark, scripts, patch):
`/home/ibutygin/tlx/perf-data/mxfp-kloop-profile-20261106.tar.gz` (SHA256
`0837f399db8d24e283461add88fcf8a7d2b75f6d3d59987877d6d24a14ff681d`).


### FlyDSL comparison and split-barrier refill (2026-11-06)

Target: 620-650 cycles per K128 step. Sources: FlyDSL
(`github.com/ROCm/FlyDSL`, `kernels/gemm/gemm_a8w8_256x256_gfx1250.py`) and
the gluon tutorial (`~/gluon-tutorials/gfx12-gluon-tutorials`, A8W8 v14:
681.9 cycles/K128 with LDS wait 36, TDM wait 5.5, barrier 7; FlyDSL tiles
without long waits 617). FlyDSL's 256x256x128 MX kernel (four waves, four
buffers): preshuffled B (`[N/16, K*16]`, so B fragments are contiguous
512-byte LDS runs; only A is padded); one TDM descriptor per wave (A, B,
A scales, B scales) patched only in the LDS and global address words;
quadrants Q0 (reads B1), Q1 (reads A1), then `s_wait_dscnt 0` + signal, 3
register-only Q2 WMMAs, wait + refill, and the next stage's reads over the
rest of Q2 and Q3; odd waves mirror the quadrant order; cluster barrier
every 32 K steps. The v14 tutorial gets its split from two ordinary
barriers that an LLVM-IR stage hook rewrites into `s_barrier_signal` /
`s_barrier_wait`, so Membar and alias analysis still see full barriers.

`REFILL_OVERLAP` modes 3 and 4 first used that IR rewrite. They now use
explicit split-barrier ops (2026-11-06): on AMD,
`tlx.named_barrier_arrive(0, num_warps * 32)` /
`tlx.named_barrier_wait(0, ...)` build `amdg.workgroup_barrier_signal` /
`amdg.workgroup_barrier_wait`, which lower to a workgroup release fence plus
`s_barrier_signal -1` and `s_barrier_wait -1` plus an acquire fence (the
same LDS-only MMRA tag as `ttg.barrier local`). The verifier requires the
wait in the same block with no shared-memory access, region op, or other
barrier in between. AMD Membar (`AMDMembarAnalysis`) treats both halves as
workgroup barriers, and the generic "later sync point" scan after an async
wait now uses the analysis's virtual `getBarrierStages`, so the TDM wait
right before the signal defers to it instead of getting its own barrier
(the mode 2 failure). Membar aborts if it would need a barrier inside the
window. Mode 2 and the IR hook are removed. Mode 4 asm against the hook
build (8k and 4k): same instruction inventory and steady-loop order, minus
the hook's marker `s_mov` per step; one step issues the wait three SALU
address ops earlier.

- Mode 3: TDM wait, barrier (signal), the 16 C10 WMMAs, barrier (wait),
  then the refill and both next-slot reads across C11 (16 WMMAs).
- Mode 4: both second halves of this slot read during C00 (front-loaded:
  four reads per WMMA over the first nine), TDM wait, signal, the 16 C01
  WMMAs, wait, then the refill and the next slot read across C10 and C11
  (32 WMMAs, also front-loaded).

FFM ring-phase test passes for all modes. Hardware (four rounds,
FP32-reference checks; repeated runs agree within ~0.1%):

| Variant | 8192³ | 8192²x4096 |
| --- | ---: | ---: |
| Mode 3 | -1.51% to -1.58% | -0.10% to -0.13% |
| Mode 4, reads spread evenly | -1.24% | -0.80% / -1.39% |
| **Mode 4, front-loaded reads** | **-2.21%** (0.14345 ms, 7,665 TFLOPS) | **-1.25%** |

ATT (uncapped) per K step: default 754, mode 3 707 (barrier 65 -> 23, TDM
45 -> 14, but LDS waits 39 -> 69: the next slot is read in only 16 WMMAs),
mode 4 front-loaded 745 (release drain `s_wait_dscnt 0` before the signal
83, `ds_load` backpressure 41). The 16-WMMA window carries no LDS reads, so
a quarter of each step leaves the LDS idle while the rest must move the
step's 139 KB. Closing the gap to ~620-650 needs FlyDSL's short window
(3-8 WMMAs), which needs quadrant dots split into 8-WMMA halves (N-split of
the accumulator and B operand; the 64-column bit is a register bit in both
layouts), so reads continue around the window.
Archive (traces, logs, ships, patch):
`/home/ibutygin/tlx/perf-data/mxfp-split-refill-20261106.tar.gz` (SHA256
`9e1dc69faea57237e5e534c59f7f99e1e91b9debac8411e820e6f33ab5934d49`).

### Large-shape A8W8 realignment (2026-11-06)

After the FP16 split refill, a fresh `bench.py` run had 16384³ and 32768³
FP16 A8W8 slower than 8192³ (7,499 and 7,285 vs 7,722 TFLOPS). Device sweeps
(each config twice in alternating order; TFLOPS):

| Case | Dynamic 4-stage (old default) | Mode 4 | Mode 4 + split L2 lookahead 4 |
| --- | ---: | ---: | ---: |
| FP16 16384³ | 7,518-7,547 | 7,602-7,617 | **7,704-7,727** |
| FP16 32768³ | **7,052-7,324** | 6,534-6,763 | 6,556-6,828 (distance 8: 6,748-7,033) |
| FP16 8192³ | - | **7,672-7,734** | 7,546-7,591 |

- Mode 4 skipped the L2 lookahead (its step returns before the lookahead
  code). It now takes the branch-free split lookahead
  (`_mxgemm_split_l2_prefetch`), placed before the C11 WMMAs; the mode 4
  four-stage ring with a 4x4 cluster only. One prefetch per step, about 2.5
  SALU and 1 VALU per step, and +2 VGPRs (910). FFM ring-phase test covers
  it (cluster 4, K5-K13). The branchy single-CTA lookahead hurts every case.
- Without refill overlap, the merged static loop gains 1-2.6% from the split
  lookahead (16384³ 7,575 vs 7,485; 32768³ 6,978 vs 6,793).
- Three-stage static loops lose 6-15% at both sizes, FP32 and FP16.
- Tile order around the best configs (one change each): group-M 4 -2 to -6%;
  2x2 clusters -8 to -12%; no XCD remap -1 to -4%; cluster barrier every 4
  -1 to -2%; every 16 +0.7% at 16384³ FP32, neutral at 16384³ FP16, -7 to
  -30% at 32768³; XCD chunk 1 or 4 within noise. The persistent kernel
  rejects `l2_prefetch_distance`. No change.
- FP32 is already aligned (16384³ 7,262, 32768³ 7,043-7,200 vs 8192³ 7,100);
  the four-stage dynamic ring stays best (overlap peel neutral).

`bench.py` now uses mode 4 for FP16 up to K=16384 and adds `--l2-lookahead 4`
from K=16384; FP16 32768³ keeps the dynamic ring. Full A8W8 bench on device:
8192²x4096 7,004 / 6,488 (FP16 / FP32), 8192³ 7,699 / 7,099, 16384³
**7,710** / 7,266, 32768³ 7,298 / 7,043. FP16 32768³ remains about 5% below
8192³ in both loop forms (the dynamic ring also drops 3-4% from 16384³):
the operands no longer fit the caches, and the merged barrier exposes each
step to the slowest wave's TDM latency. 32768³ timings are noisy (the same
config spans 7,052-7,324 across rounds).
Archive (sweep scripts, logs, ships):
`/home/ibutygin/tlx/perf-data/mxfp-realign-20261106.tar.gz` (SHA256
`5f194b78f8d5dce26773bb4df53fef0761488810079e4354c547627944636a1d`).

### Cache hints on C stores and A/B loads, current defaults (2026-11-06): no gain

Repeat of the 2026-10-23 C-store test on the current `bench.py` defaults
(FP16 mode 4 / L2 lookahead, FP32 three- and four-stage), plus A/B load
hints. Temporary environment knobs (reverted) set `cache_modifier` on every
C TDM store and on the partial-fusion loads: one fused A+B payload op and
one fused scale op per stage, so A and B share a hint. Per-operand load
hints cannot be expressed in the fused op (the cache bits are an immediate
of each wave's instruction, and the rank-3 A descriptor cannot fuse with its
rank-2 scales); they need a cache argument on the unfused TDM load binding.
asm: `.cs` gives `th:TH_STORE_NT` / `th:TH_LOAD_NT`. Full A8W8 bench, two
alternating rounds, change in mean TFLOPS against no hint:

| Case | C `.cs` | C `.wt` | A+B `.cs` | A+B+scales `.cs` | all `.cs` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192²x4096 FP32 / FP16 | +0.44 / +0.09% | +0.04 / +0.30% | -6.2 / -5.3% | -7.6 / -7.2% | -10.2 / -9.7% |
| 8192³ FP32 / FP16 | +0.02 / -0.03% | -0.02 / -0.10% | -9.0 / -6.7% | -9.0 / -7.3% | -9.1 / -6.9% |
| 16384³ FP32 / FP16 | +0.13 / -0.02% | +0.08 / -0.02% | -1.8 / -0.3% | -2.2 / -0.4% | -2.2 / -0.6% |
| 32768³ FP32 / FP16 | -0.9 / -1.3% | +0.3 / -4.1% | -8.7 / -9.1% | -9.4 / -11.8% | -8.8 / -10.2% |

C-store hints stay within noise (32768³ rounds differ by up to 6%). Non-temporal
A/B loads cost 2-12%: the A/B tiles are re-read by other workgroups and later
tiles through L2, so inserting them at low priority loses that reuse. The
opposite hint (high-temporal, `evict_last`) is the remaining load-side
experiment; the TDM ops carry only a cache modifier, so it needs an eviction
policy on the fused/unfused TDM load ops. Archive:
`/home/ibutygin/tlx/perf-data/mxfp-cache-hints-20261106.tar.gz` (SHA256
`b2f25825ca74a94a5940aad3f80c074b6193a31a8b9046b5da1728fd325df04e`).

### Static cluster steps on the mode 4 loop (2026-11-06): adopted for FP16

`STATIC_CLUSTER_STEPS` was neutral on the merged four-stage loop before
refill overlap mode 4 (see "Scalar and vector work"). On the current FP16
default it removes 19 SALU and 13 sync/branch instructions per four-step
iteration (967 -> 949 instructions, +13 `s_set_vgpr_msb`, VGPRs 908 -> 929,
SGPR spills unchanged). Device, `bench.py` default commands, four
alternating rounds (TFLOPS change per round):

| Case | Paired change | Mean |
| --- | --- | ---: |
| 8192³ FP16 | +0.79, +0.62, +0.76, +0.76% | +0.73% |
| 16384³ FP16 | +0.55, +0.25, +0.41, +0.39% | +0.40% |
| 8192²x4096 FP16 | +0.26, -1.47, +1.71, +2.36% | +0.71% (noisy) |
| 8192³ FP32 (three stages, identical code) | +0.03, -1.82, +0.13, +0.31% | -0.34% (noise) |

With three stages every unrolled step can hit the rendezvous (gcd(3, 8) = 1),
so the option changes nothing there. `bench.py` now enables it
(`--static-cluster-steps`) wherever it picks the four-stage FP16 static
loop. The FFM refill-overlap test covers mode 4 with it, with and without
the L2 lookahead.

### Late L2 prefetch before the step barriers (2026-11-06): slower

Idea: prefetch the lines of the stage the step's refill loads (distance 0)
just before the barriers that precede it, so L2 starts fetching while the
waves wait (mode 4 sequence: TDM wait, signal, 16 C01 WMMAs, wait, every
eighth step `cluster_barrier()`, refill). Temporary `LATE_L2_PREFETCH`,
branch-free split prefetch (each cluster CTA covers its share of the rows):
1 = every step before the TDM wait and signal; 2 = rendezvous steps only,
before the signal, both stages of the 256-byte line pair; 3 = rendezvous
steps only, after the split wait, right before the cluster barrier. With
static cluster steps only one unrolled step of four carries 2 or 3 (loop
+29, +31, +3 instructions per iteration). Device, `bench.py` FP16
defaults, four alternating rounds, mean change against no late prefetch:

| Case | 1 every step | 2 rendezvous, before signal | 3 rendezvous, before cluster barrier |
| --- | ---: | ---: | ---: |
| 8192²x4096 | -3.9% | -1.8% | -0.7% |
| 8192³ | -3.4% | -1.3% | -1.1% |
| 16384³ (with lookahead 4) | -15.1% | -1.8% | -8.4% |
| 16384³ without lookahead | | -0.8% vs no lookahead (-2.9% vs default) | |

Every placement loses, even issued once per eight steps. The refill already
has about three steps of slack in the four-stage ring, so an earlier start
cannot shorten an exposed wait; the extra requests (and the helper's
`depctr_vm_vsrc` wait) only add L2/TA traffic next to the TDM refill and
delay the rendezvous. Stacked on the lookahead stream at 16384³ they cost
up to 15%. Archive (patch, logs, asm):
`/home/ibutygin/tlx/perf-data/mxfp-late-l2-20261106.tar.gz` (SHA256
`f52bcddb5f3b7979d4559ad11d0789260af294307e08ef713e19e44c3f21a304`).

### LDS width and partition conflicts on the FP16 mode 4 loop (2026-11-06)

LDS reads per K step: 64 `ds_load_b128` A/B operand reads (the widest LDS
read on gfx1250; each wave reads its 128x128 FP8 A and B slices) and 4
`ds_load_2addr_b32` scale reads (two dwords 8 bytes apart). Only the scales
could widen: a scale layout with adjacent dwords would merge two of them
into one `ds_load_b128`, at most 2 of 68 LDS instructions per step.

AM (r8.05, 2048³, 16 programs, 2x2 clusters with barrier interval 8; new
`run_am_v2.py` drives the current kernel through `matmul`; all passed the
CPU check; bank conflicts zero):

| Config | Read conflicts / LDS read | Write conflicts | LDS latency |
| --- | ---: | ---: | ---: |
| FP16 default (four stages, mode 4, 64-row pieces) | 0.57 | 37.9k | 109 |
| FP16, no pieces | 0.73 | 37.2k | 114 |
| FP16, 32-row pieces | 0.12 | 41.7k | 102 |
| FP32 default (three stages, 64-row pieces) | 0.64 | 59.6k | 118 |

Device, `bench.py` FP16 defaults, four alternating rounds, mean change
against 64-row pieces: 32-row pieces -3.5% (8192²x4096), -2.3% (8192³),
-1.2% (16384³); no pieces -0.05%, -1.4%, -1.8%. The 32-row layout still
loses despite removing most conflicts (as on 2026-10-23: per-wave partition
balance matters more than the conflict count), so 64-row pieces stay.
Archive (AM harness and counters, device logs):
`/home/ibutygin/tlx/perf-data/mxfp-lds-partition-20261106.tar.gz` (SHA256
`7a791d3562e5604d062127f6b82e6f6459ed3f7b6aac9b601fcd6db7dc87c7b1`).

### FlyDSL / Gluon tutorial partition handling and wave parity (2026-11-06): no gain

Neither keeps a partition-aware layout in its best kernel. FlyDSL
(`gemm_a8w8_256x256_gfx1250.py`) places all B slots at a 64 KiB-aligned
offset 0, then A (16-byte row padding), then scales, with no pieces or
swizzle; its conflict measure is a wave-parity traversal (`wave % 2` picks
one of two steady loops; odd waves read A1 before B1, so half the waves read
A while half read B). The Gluon tutorials (v16: plain 256:+16 padding) tried
partition-separated ring slots (+0.38 / +0.15%), a partitioned layout
(segment stalls -51.6%, +0.32%) and FlyDSL's parity by SIMD ID (stalls -57 /
-68%, time -0.30% on v11, no gain on v12/v13). They read hardware stalls
with `TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES`
(needs a clock-gating override); we have not used that counter.

Our port (temporary `WAVE_PARITY`): odd waves run a mirrored mode 4 step
(C00 reading A10 then B01; C10 in the split window; refill; C01 reading the
next B00; C11 reading the next A00) in a second copy of the static steady
loop. The two loops run in sequence with wave-uniform trip counts from
`tlx.assume_uniform(tlx.thread_id(0) // 32) % 2`; both issue the same
barrier, cluster, TDM and wait sequence. 929 VGPRs (unchanged), SGPR spills
28 -> 36, +850 instructions. FFM: 12 ring-phase cases pass (K 9/13/14,
clusters 1/4, static cluster steps on/off). Device, FP16 defaults, four
alternating rounds, mean change against the default:

| Case | Parity | Parity, no pieces | Parity, 32-row pieces |
| --- | ---: | ---: | ---: |
| 8192²x4096 | +0.10% (noise) | -0.64% | +0.61% (two rounds ~0) |
| 8192³ | -1.95% | -1.05% | -0.57% |
| 16384³ | -0.78% | +0.28% | +0.04% |

Parity alone loses; it makes 32-row pieces almost neutral (they were -1.2 to
-3.5% without it), but no combination beats the default. Not kept.
Archive (patch, FFM check, device log, asm):
`/home/ibutygin/tlx/perf-data/mxfp-wave-parity-20261106.tar.gz` (SHA256
`dc74d99e8bebcb9628762301704a125eab7e72b920520f4235ef089be8d98818`).

### Automatic wave specialization on the current kernel (2026-11-06): works, mostly slower

The six `wave-specialization` commits (uniformity analysis, whole-body
specialization, fused-TDM selector specialization, thread-ID states and
integer ranges) cherry-picked onto `776e29fcaf` in worktree
`/home/ibutygin/tlx/triton-ws` (branch `ws-current`; conflicts in
`CanonicalizeLLVMIR.cpp`, `compiler.py` and a lit RUN block), plus the
uncommitted fix lowering `AsyncTDMIntrinsicWait` to `ROCDL::WaitTensorcntOp`
(otherwise block merging turns the wait count into a non-immediate operand).
Built separately (`build-ws`; needs `TRITON_EXT_ENABLED=0` for the export
version script, without which `libtriton.so` crashes next to the model's ROCm
libraries, and `TRITON_BACKENDS_IN_TREE=1`). With the option off the kernel
is identical to the main build. Two lit files fail from the merge
(`test-alignment.mlir` uses unregistered `vector`/`memref` ops;
`warp_id_to_llvm.mlir` RUN lines).

It works: `enable_wave_specialization=True` clones the whole body into two
wave versions (FP16 8192³: 4,397 -> 8,308 instructions), and the wave-selector
`s_cselect`s drop 44 -> 5 (FP16), 52 -> 21 (FP32 three-stage), 41 -> 7 (FP32
dynamic). In the FP16 mode 4 loop the specialized versions recompute each
step's global addresses from `k` (SALU 58 -> 70-72 per iteration); VGPRs
929 -> 938 (956 at 16384³), SGPR spills 28 -> 30 (30 -> 48 at 16384³). FFM:
10 ring-phase cases pass with the option (FP16 mode 4, FP32 three-stage
static, dynamic four-stage; clusters 1 and 4). Device (`run_ab2.py`, local
compiles of the `bench.py` defaults, FP32-reference checks pass, four
rotated rounds at the 2.5 kW cap):

| Case | Time | Cycles | Pairs |
| --- | ---: | ---: | --- |
| FP16 8192²x4096 | -0.13% | -0.05% | +0.14, -0.30, -0.21, -0.14 |
| FP16 8192³ | +1.78% | +3.09% | +1.74 to +1.86 |
| FP16 16384³ | +2.61% | +4.89% | +2.48 to +2.66 |
| FP32 8192³ (three stages) | +1.30% | +1.93% | +1.26 to +1.44 |
| FP32 16384³ (dynamic four stages) | **-0.47%** | -1.03% | -0.33 to -0.51 |

Only the dynamic FP32 loop gains (its selectors sit in the tile loop); the
static loops lose to the larger code and extra loop SALU. Not adopted.

Why the specialized loop recomputes `base + (k << 7)` every step (reproduced
with Triton's `llc`): the per-step addresses are identical in LLVM IR, and
in the base build the backend's straight-line strength reduction (SLSR)
rewrites step j+1 as step j's address + 128 (one shift per iteration; LSR
drops its own solution in both builds). SLSR only registers an address as a
reusable basis if `ScalarEvolution::canReuseInstruction` proves reuse adds
no poison. In the base build the TDM base is `select(wave_is_A, A_off,
B_off) + ptr`; SCEV treats the select as an opaque leaf and the walk stops
there. Specialization folds that select, exposing the whole tile-offset
chain (`shl nsw`, `sext`, `ashr exact`, `sdiv`/`trunc` from the tile id, and
`wave.id` with `range(0, 4)`), the walk gives up, and no step becomes a basis.
Confirmations: `-enable-poison-reuse-guard=false` restores the folding (10 ->
4 per-step shifts); a `freeze` on the tile-offset term of the base
(`%.reass*`) gives 2 shifts and 56-57 loop SALU in the first clone (base 58).
**Freeze fix, measured (2026-11-06).** The pass now folds selects on the
version predicate itself, replacing each with `freeze(chosen operand)`
(worktree diff, archived). Loop SALU per iteration drops below the base
build (FP16 8192³ 58 -> 55-57, 16384³ 68 -> 64-65, FP32 8192³ 32 -> 27-28);
VGPRs 930 (8k), SGPR spills 32 (8k) / 24 (16k, base 30). FFM: the same 10
ring-phase cases pass.

Timing method: `run_ab2/3.py` (several kernels in one process, one C buffer
each) gave the same binary up to +/-10% depending on its position (FP16
16384³ frozen build +9.4% in one order, +0.5% with a shared C buffer, ~+10%
first in reversed order). Final numbers use `proc_ab.py`: one fresh process
per kernel and round, six rotated rounds, median. Base binaries are stable
(16384³ 1.141-1.147 ms); both specialized binaries are bimodal at FP16
16384³ (2 of 6 processes at 1.27-1.29 ms with the clock up at ~1655 MHz,
i.e. stalled), consistent with placement-dependent instruction fetch for
the doubled code that both versions run concurrently on a CU (not traced).

| Case | WS | WS + freeze |
| --- | ---: | ---: |
| FP16 8192²x4096 | +0.49% | +1.20% |
| FP16 8192³ | +1.80% | +1.55% |
| FP16 16384³ (median; bimodal) | +0.36% | +1.08% |
| FP32 8192³ | +1.29% | **+0.06%** |
| FP32 16384³ (`run_ab3`, separate buffers) | -0.47% | -0.49% |

The freeze fully removes the FP32 three-stage regression (SALU was the
cost there) but not the FP16 mode 4 one: there the remaining loss is not
loop SALU (now below base) and is likely the doubled code footprint. Not
adopted. Archive (pass diff, harnesses, logs, binaries):
`/home/ibutygin/tlx/perf-data/mxfp-auto-ws-freeze-20261106.tar.gz` (SHA256
`f0fb2d503f231a4d4896143fcae3e4695522fa7c8c491853d11340af77e3efad`).
Archive (patches, check scripts, binaries, device logs):
`/home/ibutygin/tlx/perf-data/mxfp-auto-ws-20261106.tar.gz` (SHA256
`422ee5c3ef443f2522049089b8cfb46649c8218b5991a9d254de3e2f23f8bbbb`).

### SGPR spills and live values (2026-11-07): spills sit outside the K loop; reductions do not pay

Current defaults: FP16 four-stage builds spill 28 (4k, 8k) / 30 (16k) SGPRs
at 107 allocated; FP32 builds do not spill (101-102). All spill code
(88-110 `v_readlane`) is outside the steady K loop, in tile setup and the
transition. Pre-RA MIR liveness (`mir_live2.py`, per-instruction peaks in
`mir_peak.py`) for FP16 8192³:

- K-loop header: 113 SGPR units live: 9 carried, 49 invariant and used
  (fused TDM descriptor groups, global bases), 55 live through unused. The
  last group is: 7 `base | k*BK` constants for the K tail's next-tile
  prefetch (k = 0..3, line ~3401) and the overlap-store refill (k = 4),
  hoisted by LLVM IR LICM into the tile preheader (16 units); the C-store
  TDM group 1, a constant 8-dword tuple LLVM shares with the load
  descriptor's group 1 and keeps from kernel entry (8); C group-0 templates
  and address terms (10); the kernel-argument block (10); tile terms.
- Peaks (~150 units, tuple-granular) are in the K tail, where each of the
  four steps builds next-tile A and B addresses and selects per wave, then
  in the steady step refill (~140) and the next-tile offset math (~144).

Experiments (temporary knobs, compile then device with `proc_ab2.py`, six
per-process rounds, median vs default):

| Variant | Spills (4k/8k/16k) | Loop instructions | Device 8²x4096 / 8³ / 16³ / FP32 8³ |
| --- | --- | --- | --- |
| Sched barriers around C descriptor creation (grouped GEMM) | unchanged | unchanged | - |
| C descriptor shape + per-tile opaque zero (`off_m >> 30`) | unchanged (header +2) | unchanged | - |
| Tail/refill prefetch `k` + opaque zero | 26 / 26 / 22 (header 113 -> 97) | 949 -> 931 (16k 963 -> 979) | +0.24 / +0.57 / +0.53 / +0.54% |
| Steady loop as `tl.range(..., loop_unroll_factor=4)` (runtime slots) | 68 | 948, SALU 58 -> 52 (FP32: SALU 32 -> 108) | +0.45 / +0.59 / 0.00 / +1.22% |
| Both opaque knobs | 36 | 951 | - |

The C-descriptor levers that helped grouped GEMM do nothing here: the
constant group-1 tuple is shared with the loads, and the allocator does not
rematerialize multi-def tuples (a lowering-level fix would rebuild group 1
at the store). Cutting the K-loop header to 97 units removes only 2-8
spills (the peaks are elsewhere) and is slower everywhere, including FP32,
which has no spills. The late unroll loses the static slots and per-step
cluster tests. No change kept; spills cost only transition time.
Archive (knob patch, MIR, analysis scripts, binaries, device log):
`/home/ibutygin/tlx/perf-data/mxfp-sgpr-pressure-20261107.tar.gz` (SHA256
`2051330e815455101bf11dd6483a0fc1460f5a0d3ae2433415e8c5120be320e2`).

### Scale LDS reads: format comparison and placement in mode 4 (2026-11-07): current placement kept

Scale reads per K128 step: ours 4 `ds_load_2addr_b32` (`pack_scale` 128-row
blocks, `[lane][4 row groups][4 K]`; each wave needs row groups wm and wm+2,
8 bytes apart; the two quadrants sit in different 128-row blocks, +528).
FlyDSL `gemm_a8w8_256x256_gfx1250.py` (MX32): `[M/32][K/128][32][4]`, 8
`ds_load_b32` per step (one word per 32-row group, native scale opsel).
Gluon tutorials (v10+, FlyDSL format): 4 `ds_load_2addr_stride64_b32`. All
read 8 scale bytes per lane per step; none uses b128. A b128 read would
need a 256-row wave-ordered preshuffle `[wm][lane][(blk0, wm), (blk0, wm+2),
(blk1, wm), (blk1, wm+2)]` that also reads both quadrants' scales at once
(4 -> 2 reads of 68 per step); not implemented.

Placement in the mode 4 step (loop rotated at the barrier wait; positions
in WMMAs of 64): next-slot A scale at pos 3, B scale at 17 (each before its
payload), second-half B01/A10 scales at 33/37 in the C00 window, release
drain `s_wait_dscnt 0` at 48. Variants (temporary `MXSP` knob and a
`LOAD` payload/scale mask in the operand loaders): v1 both next-slot scales
at the refill (pos 3, 8; loop 949 -> 915 instructions, VGPRs 929 -> 891);
v2 scales after payloads (adds a full `s_wait_dscnt 0` at pos 32); v3 both
C00-window scales first (pos 33); v13 = v1 + v3. Device, FP16 defaults,
`proc_ab2.py` six per-process rounds, median change:

| Variant | 8192²x4096 | 8192³ | 16384³ |
| --- | ---: | ---: | ---: |
| v1 | +0.39% | +0.09% | +2.08% |
| v2 | +1.34% | +1.25% | +2.37% |
| v3 | -0.03% | +0.19% | +0.95% |
| v13 | +0.21% | +0.06% | +2.05% |

None is faster; the current scales-first order is kept. v1's smaller loop
and lower VGPRs do not help (+2% at 16384³). Archive (patch, placement
script, binaries, device log):
`/home/ibutygin/tlx/perf-data/mxfp-scale-placement-20261107.tar.gz` (SHA256
`503a241bc6ba71a15bad5f57c5824baf0a64c585efbf6bfe2ca63243e0fde301`).

### 256-row wave-ordered scale preshuffle (W256), b128 scale reads (2026-11-07): faster at 8k/16k

Host format: per 256 rows and K128 step, 1024 bytes ordered `[row bit 5][row %
32][row bit 7][row bit 6][4 K]` (`reshape(n/256, 2, 2, 2, 32, k/4, 4)
.permute(0, 5, 3, 4, 1, 2, 6)`), so each lane's 16 bytes are the four scale
dwords its wave needs for both 128-row quadrants. Kernel (now
`SCALE_LAYOUT="wave256"` / `--scale_layout wave256`, bench
`--scale-layout wave256`; helpers detect the format from the scale buffer's
row width, `BK/32*256` bytes): scale descriptors `[M/256, K/32*256]`, block `[1, 1024]`,
slots padded `[[1024, 16]]`; a logical `[256, 4]` view over the slot; in the
mode 4 step `sa00`/`sb00` carry both quadrants (one `ds_load_b128` each with
the next slot's A00/B00, split in registers), so A10/B01 scales are no
longer read in the C00 window. Tile entry and the first-tile prologue load
the combined form; it converts back to quadrants after the steady loop.
Other readers use the new view per quadrant. The step's fallthrough code had
to sit under a static `if` (Triton type-checks returns after an early return).

Steady loop: 272 -> 264 LDS reads per 4 steps (all `ds_load_b128`; scale
reads 4 -> 2 per step), 949 -> 910 instructions (16k: 963 -> 923), VGPRs
929 -> 938, LDS 321,768 -> 321,616 B, spills unchanged. FFM: 45 refill-overlap
ring-phase cases pass with the format. Device (`proc_ab2.py`, 8 per-process
rounds, median; `run_ab5.py` packs W256 when the binary expects it):

| Case | Base | W256 | Change |
| --- | ---: | ---: | ---: |
| FP16 8192³ | 0.14233 ms (0.14221-0.14246) | 0.14094 ms (0.14081-0.14108) | **-0.97%** |
| FP16 16384³ | 1.14270 ms | 1.13801 ms (all rounds below base) | **-0.41%** |
| FP16 8192²x4096 | 0.07722 ms (0.0762-0.0788) | 0.07811 ms (0.0767-0.0787) | +1.14% (ranges overlap; noisy) |

Device A/B of the option (2026-11-07, `ship45`: block128 vs wave256 built
from the bench default CLIs; `proc_ab2.py`, 6 per-process rotated rounds,
median, all 96 runs pass the correctness check):

| Case (bench default schedule) | block128 | wave256 | Change |
| --- | ---: | ---: | ---: |
| FP16 8192³ (4-stage mode 4) | 0.14244 ms | 0.14120 ms | **-0.87%** (ranges disjoint) |
| FP32 8192³ (3-stage static) | 0.15557 ms | 0.15288 ms | **-1.73%** (ranges disjoint) |
| FP16 8192²x4096 (4-stage mode 4) | 0.07708 ms | 0.07812 ms | **+1.34%** (slower; again, as +1.14% before) |
| FP32 8192²x4096 (3-stage dynamic) | 0.08549 ms | 0.08506 ms | -0.50% (ranges overlap) |
| FP16 16384³ (mode 4 + L2 lookahead) | 1.14472 ms | 1.13848 ms | **-0.55%** |
| FP32 16384³ (4-stage dynamic) | 1.21177 ms | 1.21067 ms | -0.09% (noise) |
| FP16 32768³ (4-stage dynamic) | 9.68464 ms | 9.68086 ms | -0.04% (noise) |
| FP32 32768³ (4-stage dynamic) | 9.84004 ms | 9.81991 ms | -0.20% |

Outside mode 4 each quadrant's scale read becomes one `ds_load_b64`
instead of a `ds_load_2addr_b32`, with unchanged VGPRs; that is the FP32 8k
gain. Mode 4 adds 9-12 VGPRs (929 -> 938, 16k 935 -> 947). The FP16 K=4096
loss is reproducible and unexplained; with 32 K steps per tile the tile
transition weighs twice as much as at 8k, so the combined-scale entry loads
and the quadrant split after the steady loop are the first suspects.
Log: `/tmp/ibutygin-live/w256opt/ab45.log`. With `--scale-layout wave256` the bench's 8k FP16
build is instruction-identical to the measured experiment binary, and the
default build is unchanged. `test_mxgemm_wave256_scales_ring_phase` covers
mode 4 (with and without the split L2 lookahead), the three-stage static and
dynamic loops, and the dynamic four-stage ring on FFM (49 cases). FP32
(three-stage, quadrant reads only) not measured on hardware. The archived
experiment patch also carries the scale-placement `LOAD`-mask variants.
Archive: `/home/ibutygin/tlx/perf-data/mxfp-w256-scales-20261107.tar.gz`
(SHA256 `12d5d8470826b29b28eea07b1b9c48c1b510aa893b9ff982ae7fed97a97dabfd`).

### Tile transition profile and late next-tile refill (2026-11-09): adopted for four buffers

ATT of the FP16 8192³ default (four-stage mode 4, 2 C slots, 16 SEs,
~2.27 GHz under the profiler; the machine has been ~2.5-4% slow since the
Nov 7 reboot, so only relative values are used). A wave takes ~245k cycles:
steady loop 79%, tile transitions 13%, final epilogue 6%, prologue 2%; a
steady step is 864 cycles. The median transition (10.3k cycles) is 7 K
steps (3 leftover steady + 4 prefetch tail, ~5.5k, real work), the C
epilogue (4.2-4.7k: TDM slot waits 2.6k, LDS drains 0.74k, workgroup
barriers 0.59k, WMMA 0.36k) and the next-tile start (0.4k). A C store takes
~1.0k cycles (p10 908, median 1,010, p90 1,240); 8 chunks through 2 slots
serialize into ~4 rounds. The epilogue's last wait acquires only stage 1, so
the final C stores stay in flight into the next tile, whose first K steps
show no tensor-wait stall (9 cycles).

The cost is the first cluster rendezvous after a transition (tile step 4):
2,652 cycles with a median cluster-barrier wait of 1,632 (p90 3,186), against
886 / 158 for a steady rendezvous step. It comes from epilogue imbalance
across the 16 CTAs: occasional C stores take 1.1-3.5k cycles, and they are
always the store issued right before the next tile's multicast A/B refill
(chunk 6; chunk 1 less often). They coincide in time across SEs during
transition bursts, and the spike moves with the refill (refill after store
5: chunk 5 spikes).

AM (`~/rocdtif-7.15-am+ffmlite-mi400-r8.05`) offers no explanation. The model
has a per-SIMD TDM counter (`tdm_per_simd=6`, stalls at issue), a TA TDM
entry buffer with a 128-byte request generator, `rr` TDM arbitration
(`priority_rate_tdm=3`) and a TCP `utc_tdm_xack_fifo`. Hardware ATT shows no
TDM issue stall (`tensor_store`/`tensor_load` issue-to-next <= 5 cycles). AM
debug emitters are compiled out (`gclog_init=<cfg>` in the `DtifGeneralArgs`
`pm4p2_args` loads a config but prints nothing). `perf_counters.csv`:
`max_tdm_desc_stall_simd` and `max_store_desc_stall_simd` are always 0, TA
adapters stay at <= 0.6 TDM requests/cycle, and a tile's C stores need ~0.5k
TA cycles, so per-CU TDM is not saturated. AM 2048³ (16 programs, 2x2
cluster) gives 135,249 cycles by default and 135,333 with the late refill: it
does not reproduce the effect.

Change: with four buffers and without `INTERLEAVE_C_FILL`, `REFILL_CHUNK`
is now `C_CHUNKS`, so the refill follows the last C store. The refilled stage
is first needed at the next tile's step 4. Device A/B (`proc_ab2.py`, 6
per-process rounds, median, all runs pass the correctness check; the new
bench builds are instruction-identical to the measured binaries):

| Case | Early refill | After last C store | Change |
| --- | ---: | ---: | ---: |
| FP16 8192²x4096 | 0.08022 ms | 0.07874 ms | **-1.84%** |
| FP16 8192³ | 0.14617 ms | 0.14420 ms | **-1.35%** (earlier run -1.14%) |
| FP16 16384³ | 1.18792 ms | 1.18748 ms | -0.04% |
| FP32 8192²x4096 (3-stage) | 0.08816 ms | 0.09012 ms | +2.22% (kept early) |
| FP32 8192³ (3-stage) | 0.15951 ms | 0.16183 ms | +1.45% (kept early) |

Three buffers need the refilled stage a step sooner, so they keep the early
refill. In the FP16 8k traces the epilogue p90 drops 6,273 -> 5,037 cycles,
the first-rendezvous wait median 1,234 -> 518, and chunks 2-6 have no slow
stores. Pitfall: placing the refill before more than `C_SLOTS` chunks makes
the later slot-reuse waits (`EXTRA_PENDING=LOADS`) too permissive; such a
variant failed on hardware while passing FFM. The default and three-buffer
formulas stay within that limit.

Archives in `/home/ibutygin/tlx/perf-data/`:
`mxfp-f16-8k-transition-20261109.tar.gz` (SHA256
`26506721c988890498bbe6652a1a74d1788542016b059124a0a5e3f25123e660`),
`mxfp-f16-8k-refill-20261109.tar.gz` (SHA256
`bc363bb7afc111851a2f66aeacc3f6224c7f8bd3c11b9319db088f7bd1af7c67`),
`mxfp-refill-shapes-20261109.tar.gz` (SHA256
`d0f24c4c27d15fafb98efd97fd26abd2bff9609ae5d036768dc705695be0c55f`).

### Epilogue TDMs on separate waves via wave masks (2026-11-09): not viable, reverted

Tried without wave specialization: every wave runs the same epilogue, fills
C in LDS and executes every barrier and wait. Only the TDM wave masks change:
C stores use `warp_used_hint=0xC` (waves 2/3), and the next-tile refill
issues A and B on waves 0/1. Inactive waves issue null descriptors, so every
wave's TDM count and all uniform waits stay valid. The refill's final epilogue
wait counts three refill operations per wave. The partitioned A/B slots need
two issuing waves per member and fused members need disjoint masks, so
A-on-0/1 and B-on-0/1 had to be one-member fused loads (the verifier and
lowering were relaxed to 1-4 members; a one-member load keeps its explicit
multicast mask). FFM: 32 ring-phase cases pass (mode 4 FP16 at late refill
and chunks 6 and 4; three-stage static FP32).

Codegen breaks the K loop in every variant, as in October:

- FP16 8192³ SGPR spills go from 28 to 74-112 at 107 SGPRs.
- The steady loop loses the WMMAs around the TDM refill and clusters 17-34
  LDS reads; FP32 three-stage goes from 101 SGPRs and no spills to 107 SGPRs
  and 22 spills, and its barriers lose their straddling WMMAs.
- Masks on the C stores alone already reschedule one of the four loop
  steps; the refill alone is worse. The hinted per-wave descriptor terms are
  invariant in the tile loop and get hoisted into the K loop's live set.
- Keeping the refill's scale load on the loop's masks (5/10) and rebuilding
  the refill A/B descriptors in the epilogue from `tlx.assume_uniform`
  bases did not help (78-94 spills).

It would not help even with clean codegen. Re-analysis of the October
`OVERLAP_WS` ATT (`att3/att-ows-s2`), where waves 2/3 issue only C stores and
waves 0/1 issue the refill before chunk 4: on the store-only waves the wait
retiring chunk 5 stalls a median 1,064 cycles (p90 1.4-1.7k), against 410-655
for chunks 1, 3 and 4. The store that overlaps the multicast refill is slow
even when it is not in the refill's queue, so the effect comes from a path
shared across waves, not per-wave TDM ordering. Not measured on hardware.
Archive (patch, FFM test, builds and loop sequences, ATT analysis scripts
and output): `/home/ibutygin/tlx/perf-data/mxfp-epilogue-tdm-waves-20261109.tar.gz`
(SHA256 `e67df2a59964dc2104ebc67b2350e4436d0bf6d5400a85ab2b99335229ca137f`).

### Row sharding across the two DPX partitions (2026-11-11)

`heliosr-1b114-d01-4` (ssh port 2223) runs its gfx1250 in DPX/NPS2: two
devices of 128 CUs and 8 shader engines each, 216 GiB local HBM each, peer
access over XGMI. `mxgemm_tdm_sharded` / `--devices 0,1` splits M across
them: each device gets its rows of A and A's scales and full copies of B and
B's scales, so the halves never communicate. Binaries were compiled locally
with the bench FP16 defaults (four stages, mode 4, 4x4 clusters, 2 C slots),
`--num_programs 128`, and run with a remote launcher that loads one kernel per
device and times do_bench-style (L2 cleared per repetition, all devices
launched together, slowest device's median). Every case passes the FP32
reference on both devices.

| Shape | 1 partition, whole problem | Half alone (dev 0 / 1) | 2 partitions, M split | Speedup |
|---|---:|---:|---:|---:|
| 8192²x4096 | 0.14417 ms | 0.07972 ms | 0.08332 ms (6,598 TFLOPS) | 1.73x |
| 8192³ | 0.26908 ms (4,086 TFLOPS) | 0.14121 / 0.14367 ms | 0.14898 ms (7,380 TFLOPS) | 1.81x |
| 16384³ | 2.00661 ms (4,384 TFLOPS) | - | 1.01217 ms (8,690 TFLOPS) | 1.98x |

`--num_xcds 4` and `8` are within 0.2% (8k: 0.2689 / 0.2693 ms alone, 0.1483
/ 0.1488 ms split). Partition 0's graphics clock is about 1.80 GHz with one
partition busy and 1.62-1.69 GHz with both, at 1.8-2.0 kW and 2.0-3.1 kW
socket power (amd-smi reports partition 1's clocks as N/A); that throttling is
the 3-5% a half loses next to the other. The socket draws about 0.8 kW idle.
These are a different machine than d04-1, so compare them with its SPX
numbers only loosely. Not tried: inputs on one device read by the other over
XGMI, or C written to the peer.
Archive (ships, launcher, logs):
`/home/ibutygin/tlx/perf-data/mxfp-dpx-sharded-20261111.tar.gz` (SHA256
`365b4a4cb0785e6d590bea0fa931bd8a82c3b3dc1a5c6d241c20d50160048273`).

### Why the DPX split scales worse at smaller shapes (2026-11-11)

Two costs, both measured on the 2223 machine with the bench FP16 schedule
(128 programs, 4x4 clusters):

1. **A fixed cost per launch, about 13.5 us**, which the split doubles in
   relative terms because each partition gets half the tiles per program.
   One-partition M sweep (N=8192, flushed do_bench timing; tiles per
   program 2/4/8/16): K=8192 0.07567 / 0.14137 / 0.26921 / 0.52307 ms,
   K=4096 0.04517 / 0.07974 / 0.14612 / 0.28338 ms. Time is linear in tiles
   per program (32 us per tile at K=8192, 17 us at K=4096) plus
   2 * half - full = 13.5 us at both K. Clocks stay at 1.80-1.83 GHz, so it
   is not power. Cycle-probe builds (per-wave 100 MHz realtime at kernel
   start and end) split it:
   - outside the waves (dispatch and completion, start event to first wave
     plus last wave to end event): 3.8 us with queued launches, 8-9 us after
     an idle queue;
   - inside: program time is linear in tiles with a 6-8 us intercept, the
     prologue's fill from cold L2 and the last tile's unoverlapped epilogue.
     Without L2 flushes the per-launch cost falls to about 7 us.
   - Start skew across programs is 0.4-0.7 us and end skew 0.7-2 us, so
     neither dispatch spread nor tile imbalance matters.
   The fixed cost is 5% of an 8192^3 run on one partition and 9.6% of each
   8192^3 half; 9.2% vs 16.8% for 8192^2x4096; 1.6% of a 16384^3 half.
2. **Shared socket power under sustained load.** With both partitions busy,
   partition 0's clock drops from about 1.80 GHz to 1.62-1.69 GHz with
   flushed timing and to 1.54 GHz back to back. 8192^3 and 8192^2x4096 lose
   4.5-9% to it; 16384^3 loses under 1% at the same clocks (1.0114 vs
   1.0154 ms synced), so it is not limited by the shader clock. The same
   flush-and-sync timing shows the loss, so it is not the do_bench L2 flush
   overlapping the other device's kernel, and short probe bursts (10 queued
   launches) show none (153.07 vs 153.27 us), so it is not interference
   inside the kernels.

Together these account for the 1.81x (8192^3: 4.6% + 5.5%) and 1.73x
(8192^2x4096: about 8% + 4.5%) speedups. The probe builds allocate registers
differently (23 instead of 28 SGPR spills) and run about 8% slower, so they
were used only for the timeline. The remote launcher now passes runtime
arguments by kernel parameter position, so kernels with probe_ptr launch.
Archive: `/home/ibutygin/tlx/perf-data/mxfp-dpx-profile-20261111.tar.gz`
(SHA256 `88354c973d690156598405913d0fab86f012342bf10bf05f349768ace2f62ed2`).

### Exit TDM drain and cluster barrier (2026-11-11): no gain; stress finds a rare C corruption

The kernel ends with `async_amd_descriptor_wait(0)` (plus a workgroup
barrier), and with clusters a second wait and an exit cluster barrier. A
temporary switch removed all three (the asm loses only `s_wait_tensorcnt 0`,
one workgroup barrier and the `-3` cluster barrier).

- **Perf:** no measurable change on the 2223 machine (flushed timing, two
  interleaved runs each): 8192^2x4096 halves together 0.0836/0.0834 vs
  0.0835/0.0834 ms; 8192^3 halves together 0.1491/0.1487 vs 0.1493/0.1491;
  8192^3 half alone 0.1415 vs 0.1413; 8192^2x4096 full 0.1461 vs 0.1460;
  8192^3 full 0.2690/0.2694 vs 0.2690/0.2691; 16384^3 halves 1.0110 vs
  1.0110 ms. The last C stores are nearly done when the waves reach the
  drain. The switch was reverted.
- **Correctness stress:** each trial poisons C with NaN, launches the GEMM,
  optionally launches an LDS scribbler right behind it (every program fills
  323,584 B of LDS with a per-program pattern; FP16 0x7c00 / 0x7c00 + pid), and
  compares C bit for bit with a clean launch. Without the drain: 0 bad trials
  in 1000 for 8192^2x4096 full on each partition (1 in 300 earlier), 0 in 300
  for the 8192^3 cases and the 8192^2x4096 halves.
- **The current kernel, with the drain, also corrupts C rarely**, so this is
  not the exit drain:
  - 8192^2x4096 full, with the scribbler: 1 in 1000 per partition. One case
    held the scribbler's pattern (32 elements of row 1296, program 988's
    0x7fdc / 0x7c00), i.e. a TDM store read LDS that a later kernel had
    written.
  - Without any scribbler: 1 in 3000 on each partition.
  - Every failure is a short aligned run of 8 or 16 consecutive FP16 elements
    (16-32 bytes) at 16-aligned rows and 8/16-aligned columns, sometimes over
    a few rows (rows 3600-3605, last 16 columns of a 128-column chunk). The
    values are plausible C values from elsewhere.
  - The single-launch FP32 reference checks and the earlier ring-phase tests
    cannot see a rate this low.
  - Not yet isolated by schedule, commit, or machine (d04-1 SPX not tested).
Archive (launcher with stress mode, scribbler, ships, logs):
`/home/ibutygin/tlx/perf-data/mxfp-dpx-exit-drain-20261111.tar.gz` (SHA256
`2f86ea47722f2ba6f8f27f93316ba85debafa8e48ae4452d2ecc92b99f9e1b8e`).

### Chunk-0 C store race in the FP16 four-stage epilogue (2026-11-11)

Follow-up to the corruption above. The stress launcher now runs both
partitions together and searches the golden output for every corrupted run.

- **Where:** every corruption (about 40 events) is in chunk 0 of a tile, the
  first 64x128 C chunk, which uses C slot 0 with no reuse wait. It is at
  in-chunk row 16 (once 17-21, once 32-33), columns 64-127. Those are the
  slot rows each lane writes with its last `ds_store_b128` instructions
  (offsets 4352 + 128/160).
- **What:** the wrong values are slot 0's previous contents. Without the
  scribbler they match, at the same in-chunk offset, chunk 6 of an earlier tile
  (the last chunk to use slot 0; in-tile row +128, column +128). With the
  scribbler they are its pattern, i.e. the first tile of a launch read LDS left
  by the previous kernel. So chunk 0's TDM store sometimes reads slot 0 before
  that chunk's own LDS writes are visible.
- **The asm is ordered:** the 8 fill stores, `s_wait_dscnt 0`,
  `s_barrier_signal/wait -1`, then `tensor_store_from_lds`, in both epilogue
  copies. Chunk 0's fill directly follows the next tile's operand reads (16
  `ds_load_b128` plus scales) while the next-tile TDM loads are in flight, so
  LDS is at its busiest there. This looks like a hardware ordering hazard
  between DS writes and TDM reads of LDS, not a missing wait; not confirmed
  with AMD.
- **Scope:** scribbler stress, bad trials over 10,000 (5,000 per
  partition):

  | Case | Bad trials |
  |---|---:|
  | FP16 four-stage, 8192^2x4096 full | 4 (earlier 11 / 10,000 and 3 / 10,000 without scribbler) |
  | FP16 four-stage, 8192^3 full | 5 |
  | FP16 four-stage, 8192^3 halves | 0 |
  | FP32 three-stage, 8192^2x4096 and 8192^3 | 0 each |
- **Fixes, applied to chunk 0 only** (temporary `MX_C0` switch, reverted):
  - **One extra workgroup barrier between chunk 0's fill and its TDM store:**
    0 / 10,000 at 8192^2x4096 and 0 / 10,000 at 8192^3. A single explicit
    barrier only replaces Membar's, so the code needs two
    `tlx.workgroup_barrier()` calls. Cost: +0.55% (0.1459 -> 0.1467 ms) and
    +0.26% (0.2674 -> 0.2681 ms) on one partition, none with both halves
    concurrent (8192^3 0.1481/0.1485 vs 0.1490/0.1487 ms; 8192^2x4096
    0.0831/0.0826 vs 0.0835/0.0833 ms).
  - **`async_amd_descriptor_wait(0)` before chunk 0's fill:** 0 / 10,000 at
    8192^2x4096, but +1.0% and +1.4% on one partition and about +0.6-1% with
    both halves.

**Adopted:** `_mxgemm_persistent_store` now issues the extra barrier for the
first chunk of every dedicated 64-row-slot epilogue: the overlapped and the
final-tile copy, FP16 four-stage and FP32 three-stage. The opt-in
wave-specialized handoff (role-split C stores) is excluded: its two roles pair
one fill barrier each, and an extra pair on both sides desynchronized them
(its ring-phase tests failed). The FP16 build is
instruction-identical to the stress-tested variant at 8192^2x4096 and 8192^3.
FP32 three-stage cost on one partition: 8192^2x4096 0.1624 -> 0.1625 ms
(+0.1%), 8192^3 0.2944 -> 0.2956 ms (+0.4%).
`test_gfx1250_mxgemm_first_c_chunk_extra_barrier` checks that each
epilogue's first C TDM store follows two barrier waits after its fill and
every other C store follows one. The compile helper in `test_tlx_codegen.py`
now fills in kernel constexpr defaults, which ASTSource leaves as None; that
had broken the merged-barrier and static-slot compile tests.
Archive (launcher, scribbler, ships, logs):
`/home/ibutygin/tlx/perf-data/mxfp-dpx-c0-race-20261111.tar.gz` (SHA256
`34e52208d023065737e5fbe19107ad0c5f149e27c04707fd8e644a1f1bc78cc8`).

### C store race: failing cases and workarounds (DEGFXARCD-12240, 2026-11-12)

The chunk-0 race above is a known hardware bug, tracked as DEGFXARCD-12240
(clones DEGFX13-37685, DEGFXMI400-13350, PLAT-216123).

**Failing cases:**

- A DS store or atomic to LDS, then `s_wait_dscnt 0`, then an async or TDM copy
  that reads the same LDS bytes (`tensor_store_from_lds`,
  `global_store_async_from_lds`). The copy can read the bytes' previous
  contents. Barriers between the wait and the copy do not make it safe.
- The reverse: an async or TDM load into LDS can be overwritten by an older DS
  store to the same bytes, even after `s_wait_dscnt 0`.
- The window grows with TDM or async loads writing the same CU's LDS. In the
  GEMM it shows up right after the next tile's operand reads.
- AM/GUMMI and FFM do not reproduce it; only hardware does.

**Workarounds from the ticket** (pass on emulation, not yet proven in general,
cost unknown):

- WAR1: `TCP:TXA_CNTL.DISABLE_ASYNC_STORE_DIRECT_COPY=1` (a driver or firmware
  setting, no kernel change).
- WAR2: after the fill, exactly 6 lanes `ds_or_b32 0` to one private scratch
  dword, then `s_wait_dscnt 0`, then the copy.
- WAR3: 5 lanes `ds_store_b32` to scratch rows in one bank, a `ds_load_b32`,
  `s_wait_dscnt 0`, then the copy.

WAR2 and WAR3 need exact lane masks, so they need inline asm or a compiler
lowering; neither fits the TLX kernels as written.

**Measured mitigations** on `heliosr-1b114-d04-1` (MI450X SPX), inserted between
`s_wait_dscnt 0` and the TDM store of the standalone reproducer
(`/home/ibutygin/tlx/perf-data/gfx1250-tdm-stale-lds-repro-20261112`,
variants in `/home/ibutygin/tlx/perf-data/gfx1250-tdm-stale-lds-mitigations-20261112`):

| Between `s_wait_dscnt 0` and the store | Failing launches |
|---|---:|
| nothing | 1470 / 5000 (29%) |
| 1 barrier | 12 / 5000 |
| 2 barriers (the adopted chunk-0 fix) | 4 / 5000 |
| 3 barriers | 1 / 5000 |
| `ds_load_b32` of one written dword + wait | 15-19 / 5000 |
| `s_sleep 2` (about 128 clocks) | 3 / 5000 |
| `s_sleep 4` (about 256 clocks) | 0 / 25,000 |
| `ds_load_b128` of every written address + wait | 0 / 27,000 |
| drain all TDM loads + barrier | 0 / 1000 |

Barriers only delay the store, so the adopted two-barrier chunk-0 fix lowers the
odds without closing the window; it passed 20,000 GEMM trials because the GEMM's
base rate is far lower than the reproducer's. The clean sleep and full-readback
results may also be only delays.

The batched kernel (`amd_mxfp_batched_gemm.py`) with TDM C stores did not fail in
about 2.6 million stressed launches (1-3 barriers, either issuing wave, C shared
with or isolated from TDM-load LDS segments, K=1024, cache-resident B; a broken
control failed every launch). Its TDM store is no faster than restaging C through
LDS and using `tl.store` (0.504 ms both), so it keeps `tl.store`.

**Options for the MXFP GEMM epilogue, safest first:** no TDM C stores (global
stores from registers or LDS); draining every wave's TDM loads before the fill
(+1% in the chunk-0 test; the reproducer's `quiet` variant); WAR1 if the setting
is available; WAR2 or WAR3 through a compiler lowering; the current two
barriers, which only lower the probability.

### A8W4 split refill barrier after the amd-llvm repin (2026-11-14): adopted

Rebasing on main moved the AMD codegen LLVM from `ba034c2f` to `59389489`.
Triton IR for the bench kernels is unchanged; only the LLVM pin matters. A8W4
(MX8xMX4, four waves, BK256 output reuse) lost about 7% at 16384³ and 32768³;
A8W8 stayed within 1-2%. The new scheduler put 2 instead of 7 and 5 WMMAs
into the two refill barrier windows, and loaded the C01 B scale last.

The A8W4 K loop now waits for the next stage before the refill, so a single
split workgroup barrier (`named_barrier_arrive/wait(0, 128)`) releases the
consumed slot and publishes the next one. C11 runs before the arrive and
covers the LDS drain; C00 and C01 (32 register-only WMMAs) fill the barrier
window; C10 and C11 overlap the next-slot operand reads. Membar cannot tell
the refill slot from the next slot, so those reads are `relaxed`. Variants
at FP32 16384³: C11 in the window -3.4%; C11 before and C00 in the window
-6.2%, and loading the C01 B scale first on top of that, no change; C00 and
C01 in the window -7.4% (adopted).

Hardware (heliosr-1b114-d04-1, three interleaved rounds, medians):

| A8W4 case | Before | Split barrier | Old LLVM |
|---|---:|---:|---:|
| 8192²×4096 FP32 / FP16 | 0.0807 / 0.0741 ms | 0.0766 / 0.0702 ms | - |
| 8192³ FP32 / FP16 | 0.1396 / 0.1337 ms | 0.1296 / 0.1238 ms | - |
| 16384³ FP32 / FP16 | 1.060 / 1.033 ms | 0.981 / 0.956 ms | 0.987 / 0.962 ms |
| 32768³ FP32 / FP16 | 8.38 / 8.26 ms | 7.84 / 7.72 ms | 7.80 / 7.69 ms |

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
