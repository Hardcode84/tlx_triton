# TLX Wave Converter Design

This document defines the replacement TLX Wave converter architecture. The
current bridge stack is not the base for this work. The old bridge modules have
been removed from the production backend; every unsupported case must return a
diagnostic and must not fall through to any old bridge or text-emitter path. The
new converter must not inherit old bridge control flow, mutable planning model,
text emission, or terminal-op graph reconstruction.

## Goals

- Lower supported TTGIR/ROCDL/AMDG ops to Wave/WaveAMD through structural Python
  bindings only.
- Make conversion decisions in stateless rewrite/type-conversion stages, not in
  the final emitter.
- Represent layout, pointer, mask, fragment, memory-token, and arithmetic facts
  as explicit converted values.
- Make unsupported semantics fail early with precise diagnostics.
- Keep each lowering unit testable without compiling a full GEMM.
- Produce Wave IR close enough to the existing AMD backend that assembly
  differences are attributable to Wave lowering, not bridge bloat.

## Non-Goals

- No compatibility with `_BridgePlan` or `wave_bridge_plan.py`.
- No compatibility fallback through the old bridge/text-emitter stack.
- No handwritten Wave IR text.
- No best-effort fallback from the new converter to a text emitter.
- No emitter-side producer/consumer graph analysis.
- No kernel-specific pattern matching for v9 or a particular tutorial shape.
- No adapters that wrap, deserialize, subclass, reproduce, or translate through
  `_BridgePlan`-shaped mutable state. Compatibility with the old bridge is
  limited to external golden comparison.

## Hard Rules

1. Rewriters are stateless.

   A rewriter receives the source op, converted operands, source/result type
   facts, token facts, and layout facts. It must not walk MLIR def-use chains,
   inspect users, or rediscover producers.

2. Type conversion owns representation choice.

   Tensor values become explicit representations such as scalar, simd,
   simd tuple, mask tuple, fragment tuple, pointer tuple, or token. Terminal
   lowerers must not infer representation by recursively inspecting operand
   producers.

3. Layout conversion is an operation, not a forward.

   `ttg.convert_layout` must either produce an explicit same-representation
   alias, an explicit component/lane permutation, a legal fragment conversion,
   or a rejection. It must not be forwarded and reinterpreted later by stores,
   DMA, masks, or dot lowering.

   Layout conversion also remaps attached facts. Component, lane,
   mask-uniform, packet-contiguity, and pointer-range facts are invalidated
   unless the conversion explicitly remaps or re-proves them for the result
   layout. A same-representation alias is legal only when logical coordinate,
   component, lane/register, shape, and element type mappings are identical.
   Cross-lane conversions require an explicit target op with defined semantics.

4. Emission is mechanical.

   The emitter maps already-lowered Wave operations to Python bindings and
   maintains only the MLIR insertion state, SSA value table, and region nesting.
   It does not decide whether DMA is legal, whether a mask is uniform, whether a
   pointer is in range, or whether an expression tree should be deferred.

5. Proofs are facts with provenance.

   Range, divisibility, power-of-two, no-overflow, uniformity, and contiguity
   facts must identify where they came from. Facts can come from source attrs,
   `tl.assume`/`llvm.assume`, op flags such as `nuw`/`nsw`, layout invariants,
   or constants. Facts must not be invented from unsafe arithmetic identities.

6. Memory ops are semantic conversions.

   Generic load/store, buffer load/store, async copy, DMA, and local memory ops
   each get an explicit converted op form. Buffer promotion is not an emitter
   heuristic.

7. Failure is explicit.

   Unsupported cache modifiers, cross-lane remaps, unsupported layouts,
   missing no-overflow facts, or mask-dependent proofs must reject with a
   diagnostic. Silent fallback is a bug.

8. Target IR is schema-closed.

   Target op operands are target value IDs only. Target ops must not contain
   source ops, source values, callbacks, lazy materializers, resolver functions,
   owner-op handles, or layout-analysis objects except as non-semantic debug
   provenance.

9. Shared helpers are domain-owned.

   Shared helpers must be pure and belong to one domain such as layouts, facts,
   tokens, target IR, diagnostics, or emission. No helper may access source
   program state, converted values, token graph, target program, and emitter
   state together. No helper may choose representation or memory-op family
   outside the owning conversion stage.

## Pipeline

The converter is a sequence of independent stages:

1. Import

   Build a source program from the incoming MLIR module:

   - source ops in region order;
   - source values with stable IDs, types, attrs, and owning op index;
   - explicit region tree;
   - explicit function/kernel metadata;
   - source operand lists only.

   Import may walk the module structure. It must not perform lowering.

2. Type and layout analysis

   Convert source types to target value representations:

   - scalar integer/fp/index;
   - uniform pointer;
   - per-lane pointer;
   - simd value;
   - simd tuple;
   - mask or mask tuple;
   - memdesc;
   - async/memory token;
   - MFMA fragment or fragment tuple.

   This stage also constructs layout maps. A layout map describes how logical
   tensor coordinates correspond to components, lanes, registers, warps, and CTA
   tiles. It must be structural, based on TTGIR attr APIs, and not based on
   string parsing.

3. Fact analysis

   Build fact records from source-local information:

   - integer width facts;
   - lower/upper bounds;
   - nonnegative/positive facts;
   - divisibility and power-of-two facts;
   - no-overflow facts from `nuw`/`nsw` or equivalent source guarantees;
   - pointer base uniformity;
   - pointer range upper bounds;
   - mask all-active facts;
   - mask uniform-over-group facts;
   - packet contiguity facts;
   - local-memory physical packet facts.

   Fact propagation must be conservative. In particular, multiplication of two
   nonnegative values is not nonnegative unless overflow is impossible in the
   relevant fixed-width semantics.

4. Token graph

   Build async and memory-token ordering before op conversion:

   - async copy tokens;
   - commit groups;
   - wait windows;
   - local-load/local-store token dependencies;
   - global/buffer memory effects.
   - read/write/RMW kind, volatile flag, atomic ordering, sync scope, address
     space, and conservative alias class for every memory effect.

   Token facts may include token users because this graph is a dedicated token
   dependency model. General SSA value users are not allowed.

   Unknown-alias or may-alias memory effects must be ordered in source order,
   including across region boundaries through yielded tokens. Edges may be
   omitted only with a cited no-alias or disjoint-byte-interval proof. Atomic,
   volatile, barrier, and fence-like source ops require exact target semantics
   for ordering, visibility, return value, and scope; otherwise they reject.

5. Op conversion

   Run per-op conversion patterns in source order and region order. Each pattern
   consumes converted operands and produces converted values/effects.

   The result of this phase is a target program, not Wave MLIR. Target ops are
   explicit operations such as `WaveBinary`, `WaveCmp`, `WaveLoad`,
   `WaveDmaLoadLds`, `WaveMma`, `WaveFragmentStore`, `WaveAssume`,
   `WaveIf`, and `WaveFor`.

   Rewriter entrypoints receive only the current `SourceOpView`, converted
   operands/results, fact/layout/token handles, and a target builder. They must
   not receive a `SourceProgram`, source value table, owner-op lookup API, user
   map, or producer lookup API.

6. Verification

   Verify the target program before emission:

   - every converted source result has exactly one converted value unless it is
     explicitly erased;
   - erased values have an erasure reason;
   - no target op references an unconverted source value;
   - no terminal op requests recursive materialization;
   - all proof-dependent ops cite sufficient facts;
   - proof facts have compatible region scope, mask scope, and fixed-width
     semantics for the consuming op;
   - no fact is used after layout conversion, cast, extension/truncation, or
     control-flow join unless explicitly remapped, re-proven, or joined;
   - buffer and DMA ops cite byte-range, alignment, address-space, mask,
     cache/other-value, and layout-contiguity facts as applicable;
   - all layout conversions are explicit;
   - all memory effects are tokenized.

7. Structural emission

   Emit the target program through Wave Python bindings. This stage may:

   - create MLIR types;
   - create constants;
   - create Wave/WaveAMD ops;
   - manage insertion into regions;
   - bind target SSA values.

   This stage must not:

   - walk source operand trees;
   - inspect source users;
   - choose DMA vs load/store;
   - choose buffer vs global addressing;
   - prove mask uniformity;
   - prove pointer bounds;
   - synthesize missing arithmetic facts.

## Stage Acceptance Gates

Each stage must be usable and testable before the next stage is implemented.

1. Import gate

   The stage accepts MLIR module fixtures and produces a source-program
   snapshot with source ops, values, attrs, regions, and kernel metadata. It
   must run without Wave Python bindings, target IR, or GEMM compilation.
   Negative tests cover missing public kernel, unsupported regions, malformed
   operand segments, and unsupported source types.

2. Type/layout gate

   The stage accepts source-program snapshots and produces converted type
   records plus layout maps. It must not emit target ops. Negative tests cover
   unsupported encodings, cross-lane conversions without an explicit target op,
   non-structural layout attrs, and layout conversions that cannot remap facts.

3. Fact gate

   The stage accepts source-program and layout snapshots and produces scoped
   fact records. It must not choose memory op families or emit target ops.
   Negative tests cover widened arithmetic misuse, missing no-overflow,
   invalid shift amounts, branch-local fact escape, loop-carried fact escape,
   inactive masked pointer lanes, and unproven packet uniformity.

4. Token gate

   The stage accepts source-program snapshots and produces an async/memory token
   graph. It must not inspect general SSA value users. Negative tests cover
   malformed commit/wait structure, missing token dependencies, and untokenized
   memory effects.

5. Op-conversion gate

   The stage accepts converted operands, facts, layouts, and token facts and
   produces target IR. It must run without Wave Python bindings. Negative tests
   cover missing facts for proof-dependent target ops, unresolved layout
   conversions, unsupported cache/other semantics, and attempts to attach source
   values or lazy resolvers to target ops.

6. Verification gate

   The stage accepts target IR fixtures and produces success or diagnostics. It
   must run without Wave Python bindings. Negative tests cover recursive
   materialization requests, fact scope mismatches, mask-scope mismatches,
   fixed-width mismatches, unjoined facts, and untokenized effects.

7. Emission gate

   The stage accepts verified target IR and produces Wave MLIR through Python
   bindings. It must not import source analysis modules or old bridge modules.
   Negative tests reject text snippets, lazy materializers, missing target SSA
   values, and any attempt to choose a lowering family in the emitter.

## Data Model

### Source Program

The imported source program contains:

- `SourceOp`: op index, name, operands, results, attrs, parent region.
- `SourceValue`: stable value ID, owner op index, type record, argument metadata.
- `SourceRegion`: block arguments and ordered op indices.
- `KernelInfo`: target, arch, waves, CTAs, argument attrs, noinline.

The owner op index is allowed because it is definition metadata, not a def-use
walk. Consumers are not stored except in the token graph.

### Converted Value

Each converted value has:

- source value ID;
- converted type;
- representation kind;
- component count;
- optional layout map;
- optional fact set;
- optional target value ID if it is already a target op result.

Converted values must be immutable records.

### Target Program

The target program is a closed schema over target values:

- `TargetValue`: target value ID, target type, representation kind, source
  provenance, and optional debug name.
- `TargetOp`: op kind, target operand IDs, target result IDs, attrs, fact IDs,
  and region IDs.
- `TargetEffect`: memory/token effect with explicit target operands and token
  results.
- `TargetRegion`: ordered target op IDs, block argument target IDs, and yielded
  target IDs.

Target ops cannot store source values, source ops, converted-value objects,
Python callables, lazy resolvers, or emitter objects. Source references are
debug provenance only and cannot affect emission.

`TargetOp` is a closed union of typed op records. Attr fields are recursively
limited to primitives, enums, target IDs, fact IDs, layout-map IDs, region IDs,
and immutable shape/type records. Unknown op kinds, unknown attrs, object
references, callables, and non-serializable values are verifier errors.
Emission must succeed after stripping all source provenance and debug names from
a verified target program, and the emitted structural IR must be equivalent. Any
emission dependency on source provenance is a verifier failure.

### Layout Map

A layout map answers these questions without inspecting producers:

- How many components does this value have?
- For each component, which logical coordinate group does it represent?
- Which lane expression, register expression, and warp expression are used?
- Does a conversion require cross-lane data movement?
- Does a packet stay physically contiguous?
- Does a mask stay uniform for a packet or store vector group?

Supported maps:

- blocked tensor layout;
- exact linear layout;
- generic linear layout with explicit basis maps;
- dot operand layout;
- AMD MFMA accumulator layout;
- padded shared layout;
- swizzled shared layout.

Unsupported maps must fail in layout analysis or layout conversion, not inside a
store or DMA emitter.

### Fact Record

Fact records include:

- subject value ID or layout component group;
- predicate;
- fixed-width semantics, if relevant;
- provenance;
- region scope;
- mask scope, if the fact is conditional.

Examples:

- `x >= 0`, source `tl.assume`;
- `x <= INT32_MAX`, source i32 argument type;
- `x is pow2 and x > 0`, source `tl.assume`;
- `offset in [0, upper] under mask m`, source pointer range plus mask select;
- `a * b no signed overflow`, source `nsw`;
- `packet lanes write contiguous bytes`, source layout map proof.

### Diagnostics

Every rejection is a structured diagnostic. Diagnostics include:

- stable diagnostic code;
- stage name;
- source op index or target op ID, when applicable;
- source value ID, target value ID, fact ID, or layout-map ID, when applicable;
- precise reason;
- no-fallback marker when the new converter rejects an unsupported case.

Negative tests assert diagnostic codes and key fields, not just that an
exception was raised.

## Operation Families

### Arithmetic

Integer lowering must preserve fixed-width semantics.

- Every integer op records source result width and the signedness
  interpretation used by that op. Widening an operand does not widen the source
  operation.
- For source i32/i64 `add`, `sub`, `mul`, `shl`, and offset arithmetic, lower
  as the source-width bit-vector operation plus explicit extension/truncation,
  or require `nuw`/`nsw`/range facts that prove the widened operation is
  equivalent.
- Facts derived through `extsi`, `extui`, `trunci`, shifts, adds, subs, or muls
  must cite the operation width. A fact proven in widened arithmetic is not
  usable for a source-width operation unless equivalence is proven.
- Index values widened from i32 must carry i32 bounds if the source was i32.
- Signed division may use unsigned lowering only when dividend is proven
  nonnegative, divisor is proven positive, and fixed-width semantics match.
- Signed and unsigned div/rem lowering must prove a nonzero divisor. Signed
  div/rem must also prove the `INT_MIN / -1` overflow case impossible when
  relevant.
- Multiplication range facts require no-overflow proof.
- Power-of-two assumptions must preserve source width semantics. A bare
  fixed-width bit test is not the same as an unbounded symbolic predicate.
- Shift-derived facts require proof that the shift amount is in `[0, width)`.

Floating-point and comparison lowering must preserve source semantics.

- FP ops record element type, source fast-math flags, rounding/contraction
  permissions, signed-zero requirements, NaN/inf behavior, and denorm/flush
  mode. Reassociation, contraction to FMA/MFMA, approximate reciprocal/div/sqrt,
  or any target op with different rounding or denorm behavior is legal only when
  source flags and target-arch facts prove equivalence; otherwise reject.
- `arith.cmpi`/integer comparisons record operand width and signed/unsigned
  predicate. `arith.cmpf` records ordered/unordered predicate and NaN behavior.
- Casts record exact source and destination types plus sign/zero/bitcast/fp
  conversion kind. Facts do not cross a cast unless remapped or re-proven under
  that cast's exact semantics.

### Masks

Masks are first-class converted values.

- Multi-component masks must be represented explicitly.
- `arith.andi` over masks produces a mask value or mask tuple.
- `ttg.convert_layout` on masks must permute, alias, or reject explicitly.
- Packet-uniform and vector-store-uniform proofs must be fact records, not
  recursive expression checks in terminal ops.

### Selects

`arith.select`, `tt.select`, and equivalent where/select ops are dataflow joins.

- Both arms must have compatible converted representation, layout map, element
  type, and component count, or conversion must insert explicit legal
  conversions before the select.
- A fact on the selected result is unconditional only if it is proven for both
  arms under the source select semantics. A fact proven for one arm becomes
  conditional on the selector mask/path condition and is usable only under that
  same guard or after the condition is discharged.
- Pointer base, byte-range, alignment, uniformity, and contiguity facts must not
  be copied from one selected arm unless the other arm is unreachable for every
  consuming active lane.

### Pointers

Pointer values are first-class converted values.

- Uniform pointer and per-lane pointer are distinct representations.
- `tt.addptr` produces an explicit pointer operation with base, offset, element
  type, address space, and range facts.
- Buffer promotion requires a uniform base pointer, bounded i32 offsets, and
  legal masked behavior. Masked inactive lanes must not receive unconditional
  offset-range assumes.
- `pointer_range` facts are byte-interval facts over accessed bytes, not only
  element-index facts. For each active lane and access width `N`, prove
  `0 <= byte_offset` and `byte_offset + N - 1 <= upper` using the same
  fixed-width arithmetic that computes the address.
- Promotion facts have mask scope. Inactive lanes may have arbitrary offsets and
  must not receive unconditional `wave.assume`, `tl.assume`, or verifier facts.
- If a target buffer op requires all lane addresses to be representable
  independent of the lane mask/EXEC state, require an all-active fact or reject.
- Uniform base means the same proven allocation/base for every active lane, not
  merely the same-looking SSA expression.

### Memory

Load/store conversion produces explicit target memory ops.

- Generic global load/store remains generic unless buffer promotion is proven.
- Buffer ops are explicit target ops.
- Cache modifiers are either carried or rejected.
- `other` values are converted using copied element type, not pointer type.
- Local memory address mapping comes from memdesc layout maps.
- Every load/store target op records the source mask, address space, element
  type, memory width, cache/eviction semantics, and `other` value if present.
  For masked loads, inactive lanes must perform no source memory read and must
  produce the converted `other` value, or the source-defined inactive-lane value
  when no `other` is present. For masked stores, inactive lanes must perform no
  write. If the chosen target op evaluates or requires representable addresses
  for inactive lanes independent of EXEC/mask state, conversion requires an
  all-active fact or valid-inactive-address fact; otherwise it must select an
  explicit masked scalar/vector target op or reject.

### Async Copy and DMA

Async copy conversion chooses one of:

- DMA load LDS;
- buffer DMA load LDS;
- explicit scalar/vector load-store target op;
- rejection.

The choice is made before emission and records:

- packet byte width;
- packet element count;
- source packet coordinates;
- destination physical packet coordinates;
- mask uniformity fact;
- non-DMA reason, if a scalar/vector load-store target op is selected.

For padded shared layouts, the proof must show a wave-contiguous packet does not
cross padding. For swizzled shared layouts, the conversion must use the same
logical-to-physical remap as Triton/TLX layout lowering.

A scalar/vector load-store alternative is legal only as an explicit target IR op
selected by the async-copy rewriter, with a surfaced diagnostic or
test-assertable reason. It must not call old bridge/text paths, re-enter generic
emission, or be selected by `emit.py`.

DMA legality is byte-for-byte equivalence. For every packet, prove the source
byte sequence, destination byte sequence, packet alignment, and write coverage
match the original async copy semantics. DMA may replace a masked copy only when
the mask is all-active for the packet, or uniformly false and the destination
semantics allow no write. Partial packet masks, per-lane masks, or `other` fill
values must use an explicit scalar/vector target op or reject unless an explicit
target op writes identical fill bytes.

Packet proofs must include source and destination alignment, no crossing of
padding/swizzle/vector-group boundaries, and no alias/overlap condition that
would change copy order. The chosen packet width is recorded in bytes and
elements and verified against source element type, destination memdesc element
type, LDS physical layout, and address-space legality.

### Dot and MFMA

MFMA lowering uses explicit fragment values.

- Dot operand local loads produce fragment or fragment tuple values.
- Accumulators are fragment or fragment tuple values.
- Fragment stores consume explicit fragment layout maps.
- Store mapping must be derived from the fragment layout, not register count
  guesses.
- gfx942/gfx950 differences live in MFMA layout metadata.
- Fragment layout metadata defines a bijection from logical A/B/C tile
  coordinates to `(fragment component, lane, register, element-within-register)`.
- Dot lowering verifies instruction kind, arch, wave size, element types,
  signedness/fp format, K width, opIdx, transposition, parent encodings, and
  accumulator type/shape before producing `WaveMma`.
- Fragment conversions, packs, unpacks, and stores preserve this mapping. Equal
  register count, tuple length, or element count is not sufficient proof.
- gfx942/gfx950 differences include MFMA version/instruction shape, wave size,
  operand lane ownership, accumulator lane/register layout, and legal element
  types. Unsupported mismatches reject.

### Control Flow

Control flow conversion must be region-structured.

- `scf.if` and `scf.for` produce target region ops.
- Values yielded from regions are explicit target values.
- Facts have region scope.
- Memory effects inside regions are tokenized and yielded when necessary.
- Fact use is dominance-checked. A fact proven inside an `scf.if` branch is not
  available after the `scf.if` unless equivalent facts are yielded from all
  branches for the same yielded value and joined explicitly.
- Facts inside an `scf.for` body are scoped to that iteration. Facts for
  loop-carried values after the loop require an induction/invariant proof or a
  conservative join over the initial value and every yield.
- Conditional facts carry a path condition. Using them outside that path
  requires discharging the condition or keeping the consuming op under the same
  guard.
- Token and memory facts yielded from regions must dominate their consumers.

## Module Boundaries

Target module split:

- `source_import.py`: MLIR import to source program.
- `types.py`: source/converted type records and type conversion.
- `layouts.py`: structural TTGIR layout maps and layout conversion.
- `facts.py`: range/divisibility/pow2/no-overflow/uniformity facts.
- `tokens.py`: async and memory-token graph.
- `rewriters/`: one file per op family.
- `target_ir.py`: Wave target-program records.
- `verify.py`: target-program verifier.
- `emit.py`: structural Wave Python binding emission only.
- `diagnostics.py`: consistent failure formatting.

The old bridge modules must not be imported by the new converter.

Allowed dependencies are one-way:

- Schema/data-record modules are separate from stage implementations. Tests for
  type, layout, fact, token, verifier, and emitter stages must be able to import
  their schema records without importing MLIR import code, Wave bindings, later
  stages, or old bridge modules.
- `source_import.py` may import diagnostics and source data records only.
- `types.py` may import source data records and diagnostics.
- `layouts.py` may import source/type records and diagnostics.
- `facts.py` may import source/type/layout records and diagnostics.
- `tokens.py` may import source records and diagnostics.
- `rewriters/` may import source/type/layout/fact/token/target IR records and
  diagnostics, but not `emit.py`.
- `verify.py` may import target IR, type/layout/fact records, and diagnostics.
- `emit.py` may import target IR, diagnostics, and Wave binding helpers only.

Wave binding helpers are pure wrappers around structural Wave Python binding
calls. They may not import source/type/layout/fact/token modules, target
conversion modules, old bridge modules, parser APIs, or make lowering-family
decisions.

No new converter module may import any
`third_party.tlx_wave.backend.wave_bridge*` module or any helper owned by that
stack. Golden comparison code is the only exception and must cross a serialized
artifact or external process boundary. Static tests must enforce this policy by
checking direct imports, dynamic imports, and transitive imports.

## Testing Strategy

Tests should be layered.

1. Import tests

   Verify source ops, values, attrs, regions, and argument metadata.

2. Type/layout tests

   Verify component counts, lane/register maps, cross-lane detection, shared
   layout packet maps, and MFMA layout maps.

3. Fact tests

   Verify range propagation, fixed-width pow2 semantics, no-overflow gates,
   mask-uniform facts, pointer-range facts, and rejection of unsafe proofs.

   Negative arithmetic tests cover i32 wrap after widening, signed div/rem edge
   cases, pow2 after truncation, and invalid shift counts. Negative pointer
   tests cover inactive masked lanes with out-of-range offsets, masked buffer
   promotion, vector access byte intervals, and unconditional assumptions that
   should reject. Negative DMA tests cover partial packet masks, `other` fill
   values, padding crossing, swizzle mismatch, and alignment mismatch.
   Control-flow tests cover branch-local facts escaping, loop-carried facts
   being retained without an invariant proof, and joins that require both
   branches. MFMA/layout tests cover equal-register-count wrong mappings,
   gfx942/gfx950 metadata mismatches, and `ttg.convert_layout` invalidating
   contiguity facts.

4. Rewriter tests

   Verify each op family produces target IR records and diagnostics.

5. Verifier tests

   Verify the target program rejects recursive materialization requests,
   missing facts, unhandled layout conversions, and untokenized memory effects.

6. Emitter tests

   Verify target IR emits structural Wave ops through Python bindings.
   Emitter tests must fail if emission constructs Wave IR by parsing or
   formatting textual Wave snippets. Textual IR may only be produced by MLIR
   printing after structural emission.
   Static tests must reject `ir.Module.parse`, `Operation.parse`, `parse_asm`,
   textual op assembly construction, and string-fed MLIR/Wave parsing from new
   converter and binding-helper modules.

7. End-to-end tests

   Verify tutorial GEMM and v9 lower to Wave, pass Wave verification, and have
   assembly shape comparable to the existing AMD backend.

Every end-to-end test for the new converter must assert the structural emitter
path. Text fallback is not allowed.

Additional acceptance tests:

- Static import-policy tests reject forbidden imports, analysis-to-emitter
  cycles, and text-emitter imports from the new converter.
- Static import-policy tests import every pre-emission stage with Wave bindings
  and old bridge modules unavailable.
- Stage fixture tests cover import-only snapshots, type/layout-only snapshots,
  fact-only snapshots, token-graph-only snapshots, verifier-only target IR
  snapshots, and emitter-from-handwritten-target-program snapshots.
- Stage fixtures include positive minimum-support cases, so a stage cannot pass
  by rejecting everything. Positive cases cover valid no-overflow facts, legal
  all-active buffer promotion, explicit scalar/vector async copy target op with
  reason, valid control-flow fact join, and supported f16/bf16 MFMA layouts.
- Target IR schema tests round-trip target programs through a primitive
  JSON-like form, reject callables/source objects in attrs/provenance, strip
  debug provenance before emission, and compare emitted structural IR.
- Negative verifier tests cover recursive materialization requests, missing
  no-overflow facts, forwarded `ttg.convert_layout`, untokenized memory effects,
  unproven mask uniformity, and unconditional assumptions on masked lanes.
- Token/control-flow tests cover multiple async groups, `wait 0`/`wait 1`,
  consuming LDS before wait, tokens yielded through `scf.for`, and untokenized
  effects inside branches/loops.
- Bridge-removal tests run the required TLX Wave end-to-end cases with old
  text fallback disabled or absent and assert the structural path.
- Anti-overfit tests cover at least two GEMM shapes, mask all-active and
  nonuniform cases, padded and swizzled shared layouts, gfx942/gfx950 metadata
  cases, and f16/bf16 MFMA paths where applicable.
- Assembly checks assert stable invariants such as instruction families, DMA
  use, buffer use, MFMA count, and absence of known bloat patterns rather than
  exact tutorial output.

Required end-to-end matrix:

- the TLX Wave tutorial GEMM;
- the gfx9 v9 GEMM port;
- at least two GEMM shapes;
- gfx942 and gfx950 where the lowering feature is expected to support both;
- f16 and bf16 MFMA paths where applicable;
- all-active and nonuniform mask cases;
- padded and swizzled shared-memory layouts;
- all of the above in a mode where old bridge/text fallback is unavailable.

## Migration Plan

1. Freeze the existing bridge.

   The old bridge modules are removed. Do not resurrect `wave_bridge_emit.py`,
   `wave_bridge_plan.py`, `wave_bridge_text_emit.py`, or the interim structural
   rewrite. Shared tool discovery/verification helpers may remain outside the
   converter, but must not contain lowering logic.

2. Implement source import and type/layout conversion.

   Build tests around source values and layouts before adding emission.

3. Implement fact and token analysis.

   Reproduce the known div/pow2/pointer-range/mask-uniform cases as fact tests.

4. Implement target IR and verifier.

   Make the verifier reject every class of hidden late decision seen in the old
   bridge.

5. Implement structural emission from target IR.

   Start with constants, arithmetic, masks, pointers, loads/stores, local
   memory, async copy, then MFMA.

6. Port GEMM support.

   Bring up v9 and tutorial kernels only after unit coverage proves the
   underlying representations.

7. Remove old fallback.

   The text-emitter and old bridge fallback are gone. The removal gate is that
   the staged converter still lowers the required unit and tutorial kernels
   through the structural target-IR path with no legacy escape hatch.

## Review Checklist

Reject a patch if it:

- adds a source-user map outside the token graph;
- adds a target op operand that is not a target value ID;
- adds an open-ended target op kind or attr object that is not part of a closed,
  serializable target IR schema;
- lets emission depend on source provenance or debug names;
- gives a rewriter a source-program, owner-op, user-map, or producer-lookup API;
- recursively materializes operand trees in the emitter;
- forwards `ttg.convert_layout` without an explicit conversion result;
- formats Wave IR text;
- imports old bridge internals into new converter modules;
- adds a silent fallback;
- adds a semantic fallback that is not an explicit target IR op with a
  test-assertable reason;
- proves multiplication range without no-overflow;
- adds unmasked assumptions for masked lanes;
- uses a pointer range as an element-index bound instead of a byte-interval
  fact over active lanes;
- uses a fact after layout conversion, cast, extension/truncation, or
  control-flow join without remapping, re-proving, or joining it;
- emits DMA without byte-for-byte source/destination/alignment/mask proof;
- maps MFMA fragments by register count rather than a layout bijection;
- parses layout strings when structural attr APIs are available;
- adds kernel-shape-specific lowering logic without a general layout reason;
- adds a new lowering without at least two shape/layout variants where the
  feature naturally has shape or layout degrees of freedom;
- centralizes multiple domains in a shared helper or service object.
