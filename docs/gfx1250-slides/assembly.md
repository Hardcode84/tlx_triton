# Assembly preparation record

The deck's ten assembly blocks were checked with `llvm-mc` for their stated
targets on 29 September 2026. Slides 3, 4, 6, 10, and 12 use freshly generated grouped
GEMM assembly. Backups B/D/I use handwritten ISA examples.
Generated excerpts retain instruction order and operands; debug directives and
compiler comments showing physical registers are removed. Omissions are marked.
These windows assume the surrounding kernel's register state, memory allocations,
and outstanding operations. Assembler acceptance verifies syntax and encoding;
it does not establish kernel correctness or performance.

## Reference compilation

- Repository: `gfx1250-preso`, HEAD `7c446203a6`; kernel and compiler sources
  unchanged from code reference `b266fe4c1d`.
- Target: `GPUTarget("hip", "gfx1250", 32)`.
- Specialization: 16 groups, K=4096, P=32; 256×256×128 tiles, depth 2,
  `GROUP_M=4`, hybrid output staging, cross-tile prefetch, no L2 prefetch.
- Remapping: chunked, eight logical XCDs, chunk size two. Four-workgroup clusters
  with multicast; four waves/workgroup, `waves_per_eu=1`.
- M and N remain runtime arguments; the intended reference workload has M=N=4096.
  This compilation does not populate the XDL efficiency chart or fix its P.
- Compiler: local `build/lib.linux-x86_64-cpython-311/triton`; the AMD backend
  Python code and TLX memory, matrix, and barrier modules match the working tree.
- LLVM tool build: 23.0.0git, revision
  `d4ad49fb3e5e295fd37dea4817f02fe39c8091ca`, from its `VCSRevision.h`.
- Compiler metadata: 320,448 shared bytes, 886 VGPRs, 104 SGPRs,
  zero private-segment bytes; eight static TDM store instructions.
  These are allocation results for this one configuration, not measured performance.

Capture hashes (SHA-256):

```text
reference.amdgcn
bafdf5397cdfe6787a6fa3aa3114eeb2a34485aca03a107cc42af5a0ce94d8aa
libtriton.so
f4ff2d1c6dfedee6f674135314b5067cb7e9e44f1c42193a9debc76e3c31ba8f
AMD libtriton_amd_codegen.so
58ee818c2dbd474f10bcf1b9d5d96fffe8ac41c9f917b97d8f7d40e2c69e2156
```

From the deck directory, with the existing matching local compiler build:

```bash
PYTHONPATH=../../build/lib.linux-x86_64-cpython-311 \
TRITON_CACHE_DIR=.cache/triton \
python3 scripts/compile-assembly.py
```

The helper compiles without launching the kernel and writes assembly, TTGIR,
LLVM IR, configuration, and hashes to ignored `.cache/assembly/`. Adjust
`PYTHONPATH` to a matching build on other hosts. The installed Python package on
the preparation host lacks some current TLX APIs; the source-tree native library
also differs from this local build. Neither was used for the final excerpts.

## Excerpt map

Line numbers refer to the captured `reference.amdgcn` above, including debug
directives. Regeneration with different debug settings can change line numbers.

| Slide | Assembly source | Editing and context |
| :--- | :--- | :--- |
| 3: registers | Lines 1841–1845, inner K-loop entry | Three consecutive instructions; `0x5a` selects D/C in v256–511 and A/B in v512–767. Partial DS wait assumes the existing load queue. |
| 4: TDM | Lines 1682 and 1023 | Individual input-load and C2-store instructions from separate regions. Descriptor setup and synchronization omitted. First operand: group 0, four SGPRs with global/LDS addresses; second: group 1, eight SGPRs with shape/strides and controls. |
| 6: scheduling | Lines 16/21, 200–202, 474–478 | Three separate regions, explicitly marked. Setup writes separate scheduling fields. Metadata uses zero MSBs; WMMA uses `0x5a`. Scalar descriptor updates are independent of WMMA VGPR operands. |
| 10: K loop | Lines 1872–1886 | Nine consecutive instructions; two LDS loads interleave with WMMAs. `0x18` depends on earlier loads, not just the two shown. Only the low byte of MSB immediates controls selection. |
| 12: C staging | Lines 1025–1046 | C3 reuses slot 1 after C1 completes; C2 may remain outstanding. Seven DS stores and one ALU dependency wait are omitted at the comment. Both workgroup handoffs are retained. |
| Backup B: hazard | Handwritten gfx1250 | Dense FP16 WMMA → dependent VALU, co-execution enabled. Four V_NOPs cover the ISA's four slots; independent VALU instructions can replace them. Operands are initialized and MSBs zero. |
| Backup D: clause | Handwritten gfx1250 | Four independent global loads, one non-flat memory clause. Dependency wait precedes the clause; load completion follows it. MSBs zero, valid global addresses, free destinations. |
| Backup I: direct LDS | Handwritten gfx950 / gfx1250 | Four-byte per-lane copies; ready addresses, valid LDS, zero immediate offsets and CDNA5 MSBs. gfx950 includes the M0 hazard delay. Completion waits are per wave. |

The K-loop operands loaded into physical v650–657 are independent of both
adjacent WMMAs. Source-level `amd_sched_barrier()` bounds whole dot regions; it
does not turn into a workgroup synchronization instruction. The output excerpt's
`-1` barriers synchronize local waves after the partial TDM wait and after LDS
writes. Its scalar add prepares the other slot's descriptor state.

## ISA and compiler references

- [CDNA5 ISA Reference Guide, 27 July 2026](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf):
  §§3.3.2/15.5 for VGPR addressing; §§5.6–5.7 for barriers and waits;
  §7.12.1 for WMMA co-execution hazards; §§10.8/10.11 for direct LDS/TDM;
  §5.3 for clauses.
- LLVM `llvm/test/CodeGen/AMDGPU/llvm.amdgcn.global.load.lds.gfx950.ll`:
  CDNA4 syntax and M0 hazard delay.
- LLVM `llvm/test/MC/AMDGPU/gfx1250_asm_vflat.s`: CDNA5 global-memory syntax.
- LLVM `llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp` and
  `llvm/test/CodeGen/AMDGPU/expert_scheduling_gfx1250.mir`: expert-mode waits.
- [Current grouped kernel](../../third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py):
  `_tdm_dot_k_block`, `_tdm_wait_and_finish_k_block`, and hybrid C staging.

Each fenced `asm` block in `slides.md` was assembled independently with:

```bash
llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 --show-encoding < snippet.s
```

Use `-mcpu=gfx950` for the CDNA4 block. No kernel launches or performance
measurements are part of this check. The separately collected [XDL results](efficiency.md)
use a diagnostic workload and the compiler settings recorded there. The
two-configuration resource comparison remains a separate preparation item.
