# Periodic kernel performance model

Build a compute, transport, and storage budget before choosing an assembly or
TLX schedule. The generic engine uses the Python standard library and does not
import Triton or require a GPU. Matplotlib is optional for plotting.

The report distinguishes three results:

- An optimistic lower bound on the period, combining resource demand,
  loop-carried dependencies, and finite credit lifetimes.
- A finite schedule that satisfies the supplied constraints, including startup
  and output drain. Its measured interior is not automatically periodic.
- A certified periodic template, when found. Dependencies, credit reuse,
  resources, and issue windows are checked across the infinite wrap. The
  template may be an observed repeat or an explicitly retimed block from the
  finite schedule; the report identifies which.

A utilization ceiling is a necessary condition, not proof that a schedule can
reach it. A certified template is achievable in the abstract model, not proof
that a compiler or device implements the assumed timings. Neither list
scheduling nor the bounded template search is an optimal-schedule solver.

## A small example with a known answer

```bash
python3 third_party/tlx/tools/perf_model/model.py \
  third_party/tlx/tools/perf_model/streaming.json \
  --schedule --sweep slots=1,2,3 --target-period 8 --json /tmp/streaming.json
```

The synthetic example has a 12-tick load latency, four ticks of memory service,
and eight ticks of computation. The input remains live through computation.

| Input slots | Period lower bound | Verified period | Compute utilization |
| ---: | ---: | ---: | ---: |
| 1 | 20 | 20 | 40% |
| 2 | 10 | 10 | 80% |
| 3 | 8 | 8 | 100% |

With two buffers, the finite cadence alternates eight and twelve ticks. An
incomplete sample can exceed 80%; checking the two-iteration pattern prevents
that sample from becoming a false sustainable-utilization claim.

## Bounds and overlap

For resource `r`, work `W_r` per iteration, and capacity `B_r`:

```text
resource bound:  T >= max_r(W_r / B_r)
```

Operations on the same resource compete for capacity. Independent resources
permit overlap. A shared array adds TDM writes and DS reads; an LDS return link
counts only traffic going to registers. Do not charge TDM traffic to that
return link simply because both operations touch LDS.

An edge from `u` to `v`, with delay `L` and iteration distance `d`, means:

```text
start(v, i) >= start(u, i - d) + L
recurrence bound:  T >= max_cycles(sum(L) / sum(d))
```

The engine finds the exact maximum cycle ratio using positive-cycle relaxation
and rational arithmetic. Fractional average periods are retained. The separate
`integer_period_lower_bound` applies to an integer period of one iteration.

Buffers add consumer-to-next-producer reuse edges. Queues allocate a circular
pool of credits in an explicit admission order. An entry can require several
credits, held from acquisition through a specified release. This models
outstanding sectors, descriptors, or register slots without confusing them
with software tile-buffer depth.

For minimum lifetimes `L_j`, credit demands `q_j`, and queue capacity `Q`:

```text
credit-area bound:       T >= sum(q_j * L_j) / Q
necessary queue depth:   Q >= ceil(sum(q_j * L_j) / target_T)
```

These lifetimes follow the dependency graph. Resource conflicts can extend
them; a necessary depth is not a sufficient depth. Reports include minimum
buffer leads, queue lifetime bounds, and independently checked finite peak
credit occupancy.

Compute utilization is total declared compute service divided by available
compute-resource time. It is a CU-level fraction when the graph describes one
CU. The tool does not infer whole-device PFLOPS from it. Issue-window packing
is checked in schedules; the analytical bound stays optimistic about packing.

Finite bounds combine the longest dependency chain in the unfolded graph with
the resource bound for each serial phase. Startup and output drain therefore
remain visible. A steady-state bound multiplied by a short trip count is not
used as a finite-dispatch bound.

## Calibration and clocks

Workload structure and machine values are separate. Parameters can be numbers,
arithmetic expressions, or `null`. An unknown latency never becomes zero:
known work is reported, bounds are marked partial, and scheduling requires all
used values. Unresolved storage is not reported as feasible.

A flat bindings file remains supported. Structured profiles additionally keep
units, source, classification, and arbitrary provenance for each value:

```json
{
  "metadata": {"name": "machine-profile", "conditions": "isolated requests"},
  "parameters": {
    "ds_scheduler_native": {
      "value": 4,
      "unit": "gfx cycles",
      "kind": "configured",
      "source": "component configuration and revision"
    },
    "return_link_width": {
      "value": 128,
      "unit": "bytes / gfx cycle / DS group",
      "kind": "assumed mapping",
      "source": "interface interpretation to validate with a throughput probe"
    }
  }
}
```

Configured stage delays, measured isolated latency, loaded latency, and exposed
wait time are different quantities. Do not add a loaded mean to reservations
that already model its contention. Keep uncertain endpoint units, sharing,
queue scope, and additive route composition identifiable as assumptions.

The MXFP graph retains individual stage values and converts their clock domains:

```text
stage ticks = ceil(native cycles * native period / shader period)
bandwidth   = native bytes/cycle * shader period / native period
```

Expressions support `+`, `-`, `*`, `/`, `//`, unary signs, `ceil`, `min`, and
`max`, including references to derived parameters. Cyclic definitions are
rejected. Expressions cannot execute Python. Units are recorded, not
automatically type-checked.

CLI `--set` values override profile values and are recorded separately. Reports
include resolved values, provenance, assumptions, and hashes of the input
graph, profile, and engine.

## MXFP graph

The generator describes a new design space, independently of existing kernels:

- A8W8 or A8W4; four or eight waves with rectangular output ownership.
- Physical BK128 or BK256 transfers, with native K128 matrix substeps.
- Packed block-32 scales and b128 payload instructions.
- Independent payload and scale LDS rings, A and B register depths, and a small
  packed-scale register ring. Register slots are measured in K128 steps.
- Packetized TDM traffic with descriptor and sector credits.
- Two shared array ports, two LDS return links, separate store ingress, and
  CU/cache links. Array service can use a pooled two-port capacity, with each
  request using at most one port, or a fixed striped mapping.
- Separate WMMA issue and execution, pending-instruction credits, source-read
  lifetimes, and accumulator dependencies.
- Explicit scalar, vector-address, control, visibility, and setup work.
- Native FP32 output stores followed by packetized TDM output. Drained input
  LDS is reused for output storage.

```bash
python3 third_party/tlx/tools/perf_model/mxfp.py \
  --waves-m 2 --waves-n 2 --block-k 256 --data-slots 2 --scale-slots 3 \
  --a-register-slots 1 --b-register-slots 2 -o /tmp/a8w8-model.json

# Inspect known structure before supplying a machine profile.
python3 third_party/tlx/tools/perf_model/model.py /tmp/a8w8-model.json
```

For 256x256 output, work per native K128 step is:

| Work per CU | A8W8, 4 waves | A8W8, 8 waves | A8W4, 4 waves | A8W4, 8 waves |
| --- | ---: | ---: | ---: | ---: |
| Matrix instructions, all waves | 256 | 256 | 256 | 256 |
| Matrix service per SIMD at eight cycles/instruction | 512 | 512 | 512 | 512 |
| Unique payload and scales | 66 KiB | 66 KiB | 50 KiB | 50 KiB |
| LDS reads returning to registers | 132 KiB | 198 KiB | 100 KiB | 134 KiB |
| Array reads plus TDM input writes | 198 KiB | 264 KiB | 150 KiB | 184 KiB |

BK256 doubles each row per model period. FP32 output adds 256 KiB of global
writes and 512 KiB of array traffic per output tile. Its TDM read does not use
the LDS-to-register return link.

A8W8 and A8W4 have equal matrix-service demand here. Their operand traffic and
register footprints differ. Eight-wave ownership increases operand replication
even though total useful math stays fixed.

By default, LDS storage is released after its final array read, before return
transport finishes. This is an optimistic source-lifetime bound. Use
`--lds-release completion` when reuse must wait for each operand's register
results, or `--lds-release workgroup` when a full LDS wait joins the payload
and scale results before releasing the stage. The latter matches a common
payload/scale ring with a workgroup reuse barrier. If a barrier also drains
younger reads from another stage, add those dependencies to the graph; the
generator cannot infer them from the instruction stream.

Register storage is released after its last matrix source read.
Those are different lifetimes. A row-major matrix traversal keeps B live late
in each step; a second B register set can ease the refill burst without doubling
A. `--mma-order nm` tests the transposed traversal. Storage checks round each
wave's VGPR allocation before adding resident waves on each SIMD.

Address sets also have a bounded lifetime, through DS source decode. This
prevents the scheduler from precomputing arbitrarily many iterations of
addresses into unaccounted registers. Loop control follows the last matrix
issue in each native K step and precedes the next step's compute issue.

The initial scaled-WMMA issue pattern follows the selected LLVM backend's
`AMDGPUCoExecInfo.h` and `SISchedule.td`: `0EEIEEISVV`. Matrix execution takes
eight cycles. A distinct issue-to-execution scale stage permits the next
scaled issue at execution offset seven; it does not reduce matrix work to
seven cycles. Source hold, result readiness, and the 24-cycle WMMA-to-DS RAW
hazard are separate constraints. A profile can replace the window table.

The default cache path assumes input hits in the modeled shared cache. The
`--memory-path memory` scenario adds memory-channel traffic and miss stages.
Its bandwidth is a configurable share over `active_cus`; channel and CU counts
must describe the same scope. Inter-CTA cache reuse, multicast, bank conflicts,
fetch misses, and dispatch overhead require additional work or resources.

## Run schedules and sensitivity analysis

`hypothetical.json` supplies invented values to exercise the tool. It is not a
calibration, tuned configuration, or device performance prediction.

```bash
python3 third_party/tlx/tools/perf_model/model.py /tmp/a8w8-model.json \
  --bindings third_party/tlx/tools/perf_model/hypothetical.json \
  --schedule --policy deadline --iterations 8 --warmup 2 \
  --json /tmp/a8w8-schedule.json --plot /tmp/a8w8-schedule.svg
```

`critical-path` prioritizes the longest remaining dependency chain. `deadline`
prioritizes requests by propagating ideal compute dates backward. These dates
are priorities, not hard timing constraints. `--target-period` controls the
deadline target and reports resource load and required buffer/credit leads at
that period. Both policies validate the resulting schedule independently.

```bash
python3 third_party/tlx/tools/perf_model/model.py /tmp/a8w8-model.json \
  --bindings /tmp/machine-profile.json --target-period 1024 \
  --sweep return_width_multiplier=1,2 --sweep transfer_credits=128,256,512 \
  --json /tmp/sensitivity.json
```

Add `--schedule` for finite and periodic witnesses. Sweep uncertain widths and
queue interpretations before selecting a design. Regenerate the graph to
change wave ownership, matrix order, packet size, memory path, or fixed versus
pooled port mapping; use bindings for numeric machine parameters and ring
depths. Packet-size sensitivity checks the effect of transfer granularity.

The full JSON contains operations, resource reservations, queue occupancy,
serial-phase windows, and the certified periodic template when available.
Retimed templates carry their own period and start offsets; they are distinct
from the plotted finite schedule. Failure to find a template is not proof that
no periodic schedule exists.

## Generic graph schema

| Field | Meaning |
| --- | --- |
| `parameters` | Named `{value, unit, source, kind}` entries; extra provenance is preserved. |
| `resources` | Named `{capacity, unit}` entries; capacity is work per tick. |
| `operations` | `id`, `kind`, `latency`, `uses`, optional `domain`, `window_domain`, `phase`. |
| `dependencies` | `{source, target, distance, delay}`; distance defaults to zero, delay to source completion. |
| `buffers` | `{id, producer, consumers, slots, size, space}`, optional `release_delay`. |
| `queues` | `{id, capacity, entries, role}`; entries have `acquire`, `release`, `units`, optional `release_delay`. |
| `storage` | Per-space `capacity`, `unit`, `fixed`, and `phase_fixed` allocations. |
| `compute_resources` | Resources whose normalized service is reported as compute utilization. |
| `coexecution` | A producer class maps to allowed issue classes at each relative tick. |

Every resource use has `resource`, `work`, optional `rate`, and optional
`offset`. Its reservation lasts `ceil(work / rate)` ticks at the given offset.
Rounding reduces the reservation rate to preserve work. Multiple uses can
describe successive stages of one route. Completion is the maximum of the
explicit latency and all reservation ends. Internal resource overbooking is
rejected.

`domain` identifies an issue resource reserved by the operation. `window_domain`
identifies the issue domain constrained by its execution window; it can differ
from `domain`, and a window-producing execution node need not issue an
instruction. Allowed classes are lists, one per tick; `"*"` permits all classes.
Overlapping windows must all permit an issue. A producer with an issue domain
must reserve exactly one issue tick at offset zero.

`prologue` and `epilogue` nodes run once and surround the complete loop. To
model persistent output overlap, place output nodes in the periodic graph and
add their actual storage lifetimes. `phase_fixed` storage can reuse a drained
loop allocation; it is not added as simultaneously live storage. The tool
does not emit assembly or allocate physical registers.
