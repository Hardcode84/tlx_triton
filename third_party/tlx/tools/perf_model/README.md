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

Byte bandwidth and port occupancy are separate constraints. A short request
still occupies a port for its rounded service duration. For example, two
256-byte ports need two clocks to accept three independent 128-byte requests,
even though their combined byte bandwidth could carry all three in one clock.
The MXFP graph reserves both bytes and whole ports for every array access.

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

Admission order alone does not imply retirement order. Set
`ordered_retirement: true` when a queue must hold younger entries until older
entries have retired. The engine inserts zero-latency retirement events after
each entry's release time, including its `release_delay`, and orders those
events across iteration boundaries. An entry can name an existing zero-latency
event with `retire`; consumers and waits can depend on that event. Internal
array or return stages can still finish out of order. The MXFP generator uses
this contract for same-wave LDS instruction completion, matching the ordered
counter prefixes used by the compiler. It does not impose it on unrelated
memory types or transport queues.

Keep the queues' units and scopes explicit. An LDS instruction can leave a
short scheduler stage while still occupying its wave's outstanding-instruction
credit until the result returns. Transfer sectors, descriptors, scheduler
entries, instruction credits, and software buffer slots are separate limits.
The width of an instruction's wait-count field does not establish any of
those capacities. An unfilled queue capacity leaves bounds partial and
prevents scheduling; it never becomes an unlimited pool.

Queue release must reflect downstream blocking. For a scheduler entry held
until an array accepts the request, use the array operation's **start**, rather
than the end of a fixed scheduler delay or the final register return:

```json
{
  "id": "scheduler_entries",
  "capacity": "scheduler_depth",
  "unit": "requests",
  "entries": [{"acquire": "scheduler", "release": "array", "release_delay": 0}]
}
```

The acquisition-to-release dependency still includes the scheduler's own
latency. Array contention extends the entry's residence and delays later
admissions. A separate instruction-completion queue can retain its credits
until the result returns. These two lifetimes must not be collapsed.

For minimum lifetimes `L_j`, credit demands `q_j`, and queue capacity `Q`:

```text
credit-area bound:       T >= sum(q_j * L_j) / Q
necessary queue depth:   Q >= ceil(sum(q_j * L_j) / target_T)
```

These lifetimes follow the dependency graph. Resource conflicts can extend
them; a necessary depth is not a sufficient depth. Reports include minimum
buffer leads, queue lifetime bounds, and independently checked finite peak
credit occupancy. Finite reports also include credit-ticks, time at capacity,
and the minimum slack between buffer retirement and its next overwrite.

The `admissions` report separates the issue frontier, credit readiness, and
actual admission. A full queue does not necessarily block any admission:
independent computation may run while that queue is occupied. Conversely,
ordered circular credit allocation can wait for a particular credit before
the entire pool is full. `credit_wait_ticks` measures the union of those local
readiness delays; `credit_wait_event_ticks` retains their summed multiplicity.
Neither is an automatic attribution of compute idle time.

A resource can specify explicit arbitration for selected operation classes:

```json
"arbitration": {
  "policy": "round-robin",
  "classes": [{"kind": "lds_array", "weight": 2},
              {"kind": "tdm_array", "weight": 1}]
}
```

Weights count admitted operations, independent of byte size or service time.
Each class preserves FIFO order by dependency-ready time, iteration, and graph
order. The arbiter skips unavailable class heads and gives up their remaining
quantum, so an empty class does not idle an otherwise available resource.
`priority` instead uses the supplied class order. Arbitration applies only to
the named classes on that resource, at offset zero; other work continues to
follow the selected list-scheduling policy. The graph must choose whether an
operation represents an instruction, clause, or packet; the tool cannot infer
the hardware's arbitration granularity from a byte count.

Arbitrated schedules are finite witnesses. The periodic checker does not yet
verify arbitration state and work-conserving choices across the wrap, so it
does not certify them as sustainable repeating schedules. Resource and
recurrence lower bounds remain optimistic. Graphs without explicit arbitration
retain their existing periodic certification.

Increasing prefetch distance has two deadlines. The final required read must
complete before consumption, and all reads of a recycled LDS slot must finish
before refill. Moving a read later can avoid exhausting instruction credits
while exposing that second wait. The minimum lifetime/area bound cannot
predict this burst behavior by itself.

Use the allocated instruction stream to identify consumers. A loop-carried
register copy can need a load result before its first matrix instruction does.
Represent load-to-copy readiness, copy-to-matrix visibility, and the previous
iteration's last read before overwriting the destination registers. An
aggregate copy count alone does not capture these deadlines. Count encoded
paired instructions once, and reserve the next matrix issue in its overlap
window before assigning the remaining slots to vector work. Register-class
restrictions can also cause spills even when the total register count fits.
Include those transfers in each request's `consumers` list as well as its
dependency edges, so readiness reports use the first register copy. Packed
scale transfers need the same treatment as payload transfers.

Include the ordering imposed by source scheduling groups. Eight copies placed
before four matrix instructions cannot use the same slots as two copies
between each pair of matrix instructions. Keep pre-pairing group sizes
separate from the final encoded instruction count. Independent scheduling
groups can insert memory operations between otherwise pairable vector
operations, so verify both the grouping and pairing in the generated assembly.
Apply these constraints to memory groups too: a read burst placed between two
matrix issues can require a gap even when average memory bandwidth suffices.

Compute utilization is total declared compute service divided by available
compute-resource time. It is a CU-level fraction when the graph describes one
CU. The tool does not infer whole-device PFLOPS from it. Issue-window packing
is checked in schedules; the analytical bound stays optimistic about packing.

Finite bounds combine the longest dependency chain in the unfolded graph with
the resource bound for each serial phase. Startup and output drain therefore
remain visible. A steady-state bound multiplied by a short trip count is not
used as a finite-dispatch bound.

Check phase improvements against the complete dispatch. Shortening one matrix
phase can expose a later input wait or move delay to a workgroup rendezvous.
For multiple SIMD streams, sum service over a common clock interval rather
than averaging percentages from different windows. Record both the local
instruction-cadence change and the resulting compute, handoff, and drain costs.
Cluster rendezvous couple participating workgroups: a single-workgroup graph
omits time spent waiting for the others. Check simultaneous wait regions across
the local waves and their arrival spread before attributing that delay to local
instruction scheduling. A high multicast matching rate does not establish
that cluster rendezvous are inexpensive.

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
A mean prefetch lead does not establish readiness of the final required
request. Cache-only and memory-path schedules are separate traffic scenarios;
an aggregate cache hit rate does not identify the misses on a critical path.

A structured profile can retain `observations` separately from its parameter
bindings. Record scope, clock, window, sample count, source revision, binary,
and input/launch conditions there. Observations are copied into reports and
do not change operation latencies. A loaded latency distribution is useful
for checking a scheduled route, but inserting its mean into a component
delay would count the modeled contention again.

Schedule reports include common clock windows for the dispatch, each serial
phase, and the sampled interior. `--window NAME=START:END` adds a window to a
scheduled report. Each window reports matrix service, queue occupancy, credit
readiness delays, and latency by request kind. Occupancy includes entries
acquired before the window; its lifetime statistics retain the full lifetimes
of entries overlapping the window. Latency samples complete in `(start,end]`
and retain their full latency even when issued earlier. An empty sample has
unknown mean latency, not zero.

Declared waits can name `compute_resources`. Requests already identify their
compute resource. The report intersects their completion waits, credit delays,
resource conflicts, and arbitration delays with compute idle intervals. It
reports both per-reason overlap and their union, leaving other idle time
unclassified. These are overlaps within the chosen schedule, not independent
causal stall counts. Summing them, or replacing them with full-dispatch queue
totals, can give the wrong diagnosis.

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
- A separate pool of outstanding LDS instruction credits per wave, held
  through register-result completion. Packed scales consume instruction
  credits independently of their byte count.
- Two shared array ports, two LDS return links, separate store ingress, and
  CU/cache links. Array service can use a pooled two-port capacity, with each
  request using at most one port, or a fixed striped mapping.
- Separate WMMA issue and execution, pending-instruction credits, source-read
  lifetimes, and accumulator dependencies.
- Each WMMA waits for the packed-scale chunk covering its own M/N fragment.
  Scale reads for later quadrants can overlap the current matrix work.
- Explicit scalar, vector-address, control, visibility, and setup work.
- Native FP32 output stores followed by packetized TDM output. Drained input
  LDS is reused for output storage.

The default input transport is `--transport staged`: issue/decode, scheduler,
array access, return transport, and completion are separate nodes. A request
can issue and then queue at a shared stage while continuing to hold its
completion credit. TDM writes and LDS reads compete at the array, but only
LDS reads use the register-return link. This allows refill traffic to extend
loaded LDS latency without changing the isolated stage delays.

`--transport fixed` retains the earlier reservation model for comparison.
It reserves every stage at a fixed offset from issue; future contention can
therefore delay the issue itself. It cannot reproduce internal queueing in
the same way. Both modes are abstract arbitration scenarios. Output transport
still uses fixed reservations and runs after the loop in this generator.

`--stage-queues` additionally holds each DS scheduler entry through array
admission and adds a finite TDM return FIFO before array service. Supply
`tdm_copy_fifo_depth` in packets at the selected packet size. Its capacity,
scope, and release endpoint are explicit scenario inputs; a FIFO entry count
does not establish its byte or descriptor capacity. Also supply
`tdm_copy_fifo_cycles`: the portion of the intrinsic `tcp_output_cycles` delay
spent holding an entry. The remainder precedes FIFO admission; the entry then
stays occupied until array service starts, including any downstream queueing.
Use whole shader ticks between zero and `tcp_output_cycles` to move the
admission point without changing total intrinsic route latency. Pipe latency
alone does not establish FIFO residence. Without this option,
internal waiting rooms remain unbounded while the existing transaction and
instruction-completion credit limits still apply.

`--array-arbitration round-robin` applies weighted arbitration to LDS and TDM
input array stages, with `array_lds_weight` and `array_tdm_weight` controlling
grants per quantum. `tdm-priority` and `lds-priority` select strict priority
scenarios. The default `scheduler` leaves the choice to the list scheduler.
These options describe arbitration assumptions; they do not identify a
device's exact policy. Logical TDM command latency and constituent packet
latencies appear as separate request kinds and must not be averaged together.

```bash
python3 third_party/tlx/tools/perf_model/mxfp.py \
  --block-k 128 --data-slots 3 --scale-slots 3 \
  --stage-queues --array-arbitration round-robin \
  -o /tmp/queued-mxfp.json

python3 third_party/tlx/tools/perf_model/model.py /tmp/queued-mxfp.json \
  --bindings third_party/tlx/tools/perf_model/hypothetical.json \
  --schedule --policy deadline --iterations 8 --warmup 2 \
  --json /tmp/queued-mxfp-report.json
```

```bash
python3 third_party/tlx/tools/perf_model/mxfp.py \
  --waves-m 2 --waves-n 2 --block-k 256 --data-slots 2 --scale-slots 3 \
  --a-register-slots 1 --b-register-slots 2 -o /tmp/a8w8-model.json

# Inspect known structure before supplying a machine profile.
python3 third_party/tlx/tools/perf_model/model.py /tmp/a8w8-model.json
```

For 256x256 output, work per native K128 step is shown below. Four-wave
ownership is 2x2; eight-wave ownership is 4x2. Other eight-wave aspect ratios
can change A8W4 operand replication.

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

By default, `--lds-release workgroup` joins the payload and scale results
before releasing a stage, matching a common ring with a workgroup reuse
barrier. Its declared arrival is the earliest point after all that stage's
read instructions have issued. A particular ISA schedule can put independent
matrix work before that arrival. Use `--lds-release completion` for separate
operand completion, or `--lds-release source` for the optimistic bound that
releases storage after its final array read, before return transport finishes.
If a counter wait also drains younger reads from another stage, include those
completions in the graph's `waits`; the generator cannot infer emitted waits.

`--scale-read-bytes 256` models four scale instructions per wave for the
four-wave 256x256x128 A8W8 tile: 64 payload instructions plus four scale
instructions, or 68 outstanding-credit acquisitions per K128 step. A
load returning 512 bytes per wave reduces instruction count without changing scale
traffic. Select the width from the compiled layout. The LDS admission order
is explicit: `--lds-order operand` groups A then B, while `interleaved`
alternates matching fragments. This is a chosen realizable order, not a
claim about an unconstrained optimal schedule.

For a placement sensitivity study, use `--array-mapping striped` and
`--wave-partitions 0,1,0,1`. This changes the input-read array port assigned
to each wave while retaining its return-link grouping. It does not infer
register layout or bank addresses. A source-level partition change that
introduces layout exchanges needs extra stores, reads and visibility edges;
it is not equivalent to changing only this mapping.

Register storage is released after its last matrix source read.
Those are different lifetimes. A row-major matrix traversal keeps B live late
in each step; a second B register set can ease the refill burst without doubling
A. `--mma-order nm` tests the transposed traversal. Storage checks round each
wave's VGPR allocation before adding resident waves on each SIMD. The report
separates the modeled working set from allocated storage. Calibrate
`vgpr_next_free` from the compiled descriptor, including any occupancy
reservation, then apply allocation granularity. Used VGPR counts alone can
give a different residency limit.

`lds_limit` is a per-workgroup legality limit. `lds_residency_capacity` is a
separate aggregate capacity requiring admission evidence. The `residency`
report gives only necessary upper bounds from known constraints; it does not
multiply throughput by that number or assume a second CTA is admitted.

Address sets also have a bounded lifetime, through DS source decode. This
prevents the scheduler from precomputing arbitrarily many iterations of
addresses into unaccounted registers. Loop control follows the last matrix
issue in each native K step and precedes the next step's compute issue.
For instructions that retain scalar descriptors or vector offsets through
address translation, extend the corresponding address-credit release to
that endpoint. Result completion, payload source read, and address-source
release are distinct lifetimes. A late scale result that overwrites a payload
address also needs that allocated-register dependency, even if source code
places the scale load first.

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

The structured profile's optional `code` section records static instruction
footprint and its provenance. It does not turn bytes into a fetch latency.
Represent a demonstrated cold fetch cost as an explicit operation at the
affected boundary. A compact tail can reduce finite-dispatch cost with the
same dynamic matrix work and a slower selected interior. Compare prologue,
first compute, later compute, tile handoff, and final drain separately.
One `prologue` and `epilogue` surround the entire modeled loop; they are not
automatically repeated at persistent tile boundaries.

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

Sweep `lds_outstanding` independently of `ds_scheduler_depth`,
`transfer_credits`, and register/LDS slot counts. The synthetic profile fills
the instruction-credit and residency capacities with invented values; a
machine profile must supply its own evidence. Compare `requests.by_kind`,
`waits`, `queue_occupancy`, and `buffer_lifetimes` as well as the final period.
A lower counter or shorter wait at one instruction can move the delay to
another wait or to the following workgroup rendezvous.

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
| `queues` | `{id, capacity, entries, role, scope, unit}`; entries have `acquire`, `release`, `units`, optional `release_delay`. |
| `waits` | `{id, arrival, target, completions, role}`; adds arrival and completion dependencies to the target. |
| `requests` | `{id, issue, complete, stages, consumers, kind, scope}`, optional `source_release` and `compute_resource`. |
| `storage` | Per-space `capacity`, `unit`, `fixed`, `phase_fixed`, optional `allocated` and `allocation_basis`. |
| `residency` | Separate aggregate capacity constraints `{resource, capacity, per_cta, scope, source}`. |
| `code` | Static footprint and provenance; metadata only, with no implicit fetch-cost formula. |
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

Storage accounting is not alias analysis. Two differently shaped views can
overlap the same physical bytes. Supply their cross-view read/write ordering
edges using the real padded stage stride; separate buffer names do not prove
disjoint storage. A missing correctness dependency is not an overlap gain.

`requests` annotate existing connected routes; they do not create timing
edges. Their stage list must follow issue to completion. Consumers name
operation starts, optionally with an iteration `distance` and a `delay`.
`source_release` names an endpoint on the route, with delay defaulting to
that operation's latency. Request reports include issue, source release,
completion, consumer, first covering wait, and queueing between stages.
`route_queue_ticks` measures gaps beyond the declared direct dependency delays,
which default to source completion. It is not an independently measured
hardware counter. It is unknown for omitted intermediate stages and for
`aggregate: true` requests, such as a command joining several packet routes:
their end-to-end latency remains available, but missing transport service
must not be labeled queueing. The `route_complete` field distinguishes these
cases; unknown queueing samples are excluded from the corresponding means.

A wait's `arrival` and `completions` can be operation names or
`{op, distance, delay}` endpoints. Distance looks backward from the wait
iteration; delay defaults to endpoint completion. For example:

```json
{
  "id": "refill_wait",
  "arrival": {"op": "independent_math_done"},
  "target": "slot_reusable",
  "completions": ["old_payload_done", "old_scales_done", "younger_read_done"],
  "role": "emitted LDS counter wait before refill"
}
```

Include `younger_read_done` only if the actual wait covers it. A loop-header
merge can force a more conservative count than the steady incoming path
alone suggests. The useful latency-hiding lead then ends at this first wait,
even if the younger operand's arithmetic consumer is much later. The wait
report separates completion delay from later scheduling delay and names the
last completions. These intervals can overlap across waits and waves.

The `issue_windows` report counts permitted empty slots by issue domain and
instruction class, intersecting overlapping window rules. It does not assume
an instruction was ready to use an empty slot. Register dependencies,
address-source holds, request credits, or another resource can prevent issue.
Scheduling markers constrain ordering; they do not replace data waits or
workgroup visibility dependencies. The selected compiler's coexecution table
and wave arbitration settings remain separate inputs to an ISA comparison.

## Compare schedules with observations

`window_metrics(start, end, service, stalls)` accepts half-open intervals
keyed by explicitly named compute resources. Combine both partner waves on
the same SIMD before computing utilization. The result gives its numerator
and denominator, clips every event to the common window, intersects stalls
with matrix idle time, and reports both per-category counts and their union.
Categories are not additive: a completion delay can coincide with a peer's
barrier wait. A long gap after a scalar instruction needs direct event
evidence before it is assigned to instruction fetch or scalar execution.

`latency_statistics(bins)` combines count/sum/min/max bins using sample
weights. Zero samples produce unknown latency, not zero. Missing min/max
coverage also remains unknown. The caller must select complete bins in the
same clock domain and scope; interval endpoints cannot be reconstructed from
an aggregate counter. Ratios of partition-conflict and active-port counts
are not fractions of conflicting instructions unless those counters have
matching event definitions.

Use the following mapping when refining a graph from generated code:

| Finding | Model change or evidence needed |
| --- | --- |
| Layout conversion adds LDS traffic | Add the actual exchange reads/stores and their visibility edges; preserve useful matrix work. |
| A view changes physical stage stride | Use padded allocation sizes and actual alias lifetimes, including nested slices and output aliases. |
| A scale load overwrites an address register | Retain the address credit until source decode/translation, then permit the overwrite. |
| Fewer partition conflicts, longer reads | Check both array placement and loaded completion/credit occupancy; do not subtract a counter ratio from latency. |
| Delayed refill improves LDS latency | Include the shorter TDM-to-visibility lead and total dispatch cost. |
| Full-drain tail is fast | Treat it as evidence about concurrent transfer interference, not sustainable loop throughput. |
| Larger unroll improves an interior | Retain cold fetch, tail, output, and register-allocation costs in the finite comparison. |
| Fewer output transfers | Model output-slot reuse and final acknowledgment; one large reused slot can serialize stores. |
| More waves | Recount replicated operands and issue work, then normalize combined matrix service per SIMD. |
| Smaller used-register count | Inspect the descriptor's allocation reservation and actual admission; do not infer another resident CTA. |
| Spills despite spare total VGPRs | Represent constrained register classes as separate storage pools; a low-register operand constraint can exhaust one class. |
| Multicast or periodic cluster rendezvous | Model matching-request overlap and rendezvous cost separately from local correctness waits and remapping. |

For independent CTA-local copies, multicast requests that do not overlap can
fall back to independent transfers. Removing a performance rendezvous does
not remove each CTA's storage lifetime protection. The one-CTA generator
does not predict cross-CTA matching, cache reuse, cluster admission, or XCD
placement; represent those mechanisms explicitly before assigning a benefit.

Compare identical useful work, K ranges, allocation sizes, binary revisions,
and launch coverage. Preserve first dispatch, repeated dispatch, and later
tiles as distinct observations: a second dispatch is not guaranteed faster.
Check at least two K lengths when a change affects ring phase or tail code.
A certified abstract schedule and a sampled service fraction are different
metrics from a device profiler's utilization statistic or full-device timing.

Run the CPU regressions without a compiler rebuild:

```bash
python3 -m pytest -s --tb=short third_party/tlx/tools/perf_model/test_model.py
```
