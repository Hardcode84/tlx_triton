# Programming CDNA5 / gfx1250 — presenter transcript

Companion to [slides.md](slides.md).
The time windows follow the 30-minute main talk and include pauses to inspect
the diagrams and assembly. Slide 14 reserves most of its time for discussion.
The title and grouped GEMM divider are included in that schedule. The backup divider and
backups A–J are optional, outside it. Section numbers match all 25 deck pages.

The schedule reserves 25 minutes for the material and five minutes for discussion.
Treat the time windows as rehearsal targets, including pauses to inspect the examples.

Text beneath each timing line is written to be spoken. Italic text in square
brackets gives delivery cues. The XDL efficiency comparison is deferred; its data
remain in [efficiency.md](efficiency.md). Backup H still describes an unfilled
resource-comparison panel.

## Slide 01 — Programming CDNA5 / gfx1250

*0:00–0:10 · 10-second opening*

Hi, I’m Ivan Butygin. Today I’ll talk about programming CDNA5, the gfx1250 target.

*[Advance to the hardware overview.]*

## Slide 02 — Hardware overview

*0:10–3:00 · 2 minutes 50 seconds*

Today I’m going to look at the CDNA5 changes that affect how we write and schedule
GPU kernels: wave size and workgroup placement, registers, data movement, and
instruction scheduling.

First, I want to separate wave size from hardware organization. CDNA3 and CDNA4
use Wave64; CDNA5 uses Wave32. On CDNA5, one workgroup processor, or WGP,
contains two compute units, or CUs. Each CU contains two SIMD32 units, so there
are four SIMD32s in a WGP. CU is still a hardware term on CDNA5.

*[Point to the two CUs inside the WGP diagram.]*

All waves of a workgroup are placed within one WGP. They can execute on any of
its four SIMD32s and share its LDS. That makes the WGP the relevant boundary
for our workgroup’s resource budget. Multiple workgroups can reside in a WGP
when resources allow.

That change affects how values are distributed across lanes and how the workgroup
uses the machine. A matrix layout built around 64 cooperating lanes needs to be
reconsidered when the hardware matrix instruction operates across 32 lanes.

The LDS capacity also changes. The table shows 64 kibibytes per CU on CDNA3,
160 per CU on CDNA4, and 320 per WGP on CDNA5. That last number belongs to
the whole WGP, containing both CUs. We’re comparing the LDS budget available
to a workgroup across generations; we cannot call this doubled LDS per CU.

That larger LDS budget makes room for substantial tiles and multiple buffer
slots. Input buffering, output staging, and layout padding all have to fit in
the same budget. Larger allocations can also reduce the number of resident
workgroups, so capacity and occupancy need to be considered together.

Finally, the matrix instruction family changes from MFMA to WMMA, and the matrix
accumulators live in the VGPR namespace. That puts accumulator storage and
ordinary vector state into one register-allocation problem.

Those capabilities give us more room to organize the computation. Whether that
organization is useful depends on how well the kernel delivers operands and
overlaps independent work. I’ll follow that path from global memory, through
LDS and registers, into the matrix instructions, and then back out to the output.

## Slide 03 — More addressable VGPRs for one wave

*3:00–6:00 · 3 minutes*

Let’s start with registers, because the assembly is hard to read correctly
without understanding this part.

CDNA5 can address up to 1,024 VGPRs per thread. The instruction’s ordinary
register field supplies the low eight bits of the index. The remaining two
bits come from persistent state set by `S_SET_VGPR_MSB`.

There are separate settings for the destination and for each of the three
source operand classes. The four ranges in the diagram are therefore address
ranges within the VGPR namespace. For example, low index seven with MSB value
two resolves to register 519: two times 256, plus seven.

*[Point to the MSB instruction, then follow the four WMMA operands.]*

Here is a generated assembly excerpt. The value `0x5a` sets the
destination and source-two MSBs to one, and source-zero and source-one to two.

So the destination written as registers 202 through 209 actually resolves to
458 through 465. The A operand resolves to 554 through 561, and B to 642 through
649. The final operand is the existing accumulator, in the same register range
as the destination.

This WMMA computes a 16-by-16 result with a reduction depth of 32. A and B contain
FP16 values, while accumulation is FP32. All 32 lanes cooperate in that matrix
operation; each lane holds part of the operands and result.

The DS wait immediately above it makes the required loaded operands available.
Its particular count comes from the larger sequence of LDS loads around this
excerpt; it is not a standalone operand-loading sequence.

There are two practical consequences here. First, when inspecting assembly,
carry the MSB state forward as you read. Two identical printed register numbers
can resolve differently after a state change. Second, greater addressability
still comes with an allocation cost. A large accumulator, its operands, and
addressing state all consume registers, so their lifetimes matter.

Now that we can identify the registers correctly, let’s look at tile transfers
between global memory and LDS.

## Slide 04 — TDM moves tiles

*6:00–8:00 · 2 minutes*

The Tensor Data Mover, or TDM, transfers tiles between global memory and LDS.
It takes a description of a tile and performs the transfer while other
instructions can make progress.

On the left is a generated tile load. Its operands name two groups of scalar
registers. Group zero contains the tile’s global address and its LDS address,
along with control bits. Group one carries the dimensions, strides, element
format, padding, and multicast controls. This two-dimensional form uses four
SGPRs for the first group and eight for the second.

On the right is a tile store. Once the source data is ready in LDS, TDM copies
the tile to global memory. These instructions come from separate regions of
generated code; descriptor setup and synchronization are outside the excerpts.

*[Indicate the two instruction operands in each panel.]*

The data payload goes directly between global memory and LDS. The descriptors
still need registers and instructions to construct them, but the transfer avoids
an intermediate payload in VGPRs.

For non-gather loads and stores, one instruction transfers a tile whose size
comes from the descriptor. The byte count is not fixed by the opcode or lane
count. Tile dimensions use 16-bit fields, and the LDS footprint must fit the
allocated buffer. Gather and scatter have a separate row-index-list limit.

This lets us specialize transfers by wave and combine instruction sites. Our
grouped GEMM already does this: waves zero and one load A; waves two and three
load B. Each wave selects its descriptor at the same load instruction site.
All four waves also compute. We will return to this on slide nine.

TDM operates on whole tile requests and ignores EXEC. Choosing which waves issue
requests therefore happens separately from the per-lane execution mask. The ISA
supports one- through five-dimensional tiles, plus two-dimensional gather and
scatter. Loads can also apply LDS padding and multicast.

Completion is tracked by the issuing wave’s tensor counter. We use
`S_WAIT_TENSORCNT` where completion is required, and synchronize other waves
before they consume shared data. The multicast option extends this cooperation
across workgroups, which brings us to clusters.

## Slide 05 — Clusters share input tiles across WGPs

*8:00–10:00 · 2 minutes*

A cluster adds a level to the launch hierarchy between the grid and the
workgroup. Its members are scheduled within one shader engine, with each member
on a separate WGP. Clusters can have one-, two-, or three-dimensional shapes,
and the architecture allows up to 16 workgroups in a cluster.

Each workgroup still has its own LDS allocation. In this diagram, multicast
places a copy of the same input tile into both allocations, so two workgroups
can reuse that input in their own computations.

*[Trace the two arrows from the global tile into the LDS boxes.]*

The example selects ranks zero and one with mask `0011`. For a TDM operation,
that recipient mask is carried in the descriptor. Every selected recipient
issues a matching request. The hardware can then combine those requests and
deliver the data to the participating workgroups. Requests that arrive late
can be served separately, so the amount of sharing also depends on how well
the workgroups stay aligned.

That alignment matters for correctness as well. Before a workgroup refills an
input slot, its neighbors must have finished reading their corresponding copies.
A workgroup finishes local LDS reads, synchronizes its local waves, and then uses
cluster arrival and wait before issuing the refill.

The cluster barrier coordinates workgroups; transfer completion is still tracked
at the requesting waves. Both pieces are needed when shared input slots are
reused.

We now have a way to move and share tiles. The next piece is how a wave
schedules memory and matrix instructions together.

## Slide 06 — Two controls, different jobs

*10:00–12:00 · 2 minutes*

There are two scheduling controls here, and each addresses a different
part of execution.

The first is expert mode, selected by setting the low two scheduling-mode bits
to two. In this mode, the compiler inserts waits for selected dependencies
between vector memory and vector ALU instructions. This backend enables it by
default for gfx1250.

Look at the memory-load excerpt. The ALU wait protects the register
dependencies associated with issuing the memory operation: prior ALU writes
must be ready, and prior memory operations must have finished using registers
that would otherwise be overwritten. The later load-counter wait establishes
that the loaded data has arrived. Those waits protect different events.

*[Move from the first mode write to the second, then to the WMMA excerpt.]*

The second control is bit two, which allows a wave to queue WMMA operations and
then issue independent work while the matrix operations execute. This is useful
at low wave occupancy, when fewer other waves are available to hide latency.

At the bottom, a WMMA is followed by scalar additions that update descriptor
addresses. Those additions use scalar registers and can proceed independently
of the matrix operands in VGPRs. That is the kind of overlap the schedule is
trying to expose.

WMMA co-execution also has specific operand-hazard spacing requirements. The
compiler must satisfy those alongside the memory dependency waits; the backup
slide has a small example.

## Slide 07 — TLX grouped GEMM

*12:00–12:10 · 10-second transition*

We now have the hardware pieces. Let’s follow one persistent grouped GEMM
through its input pipeline, tile transitions, output stores, and input sharing.

*[Pause on the section title, then advance to the workload.]*

## Slide 08 — Grouped GEMM: inputs and contract

*12:10–13:10 · 1 minute*

Each group computes A times B-transpose, using FP16 inputs and output with FP32
accumulation. N and K are shared; M can vary.

A and C pack the rows from all groups. B holds each group's weight matrix.
The int32 offsets mark row boundaries; equal adjacent offsets represent an
empty group. A and B are K-contiguous; C is N-contiguous.

*[Point from the packed shapes to the contract.]*

The TDM path requires complete tiles: M divisible by BM, N by BN, and K by 128.
The depth-two hybrid also needs an even number of K128 blocks, at least two.
Multicast adds equal positive M and the launch constraints in backup G.

## Slide 09 — Keep work and buffers on the WGP

*13:10–15:00 · 1 minute 50 seconds*

The output tile is 256 by 256, with K blocks of 128. Four Wave32 waves,
or 128 threads, cooperate in each workgroup.

P counts workgroups across all groups. After remapping, logical program p takes tiles p, p plus P,
and so on, retaining buffers and preparing the next tile in the current tail.

The wrapper starts P from the runtime’s multiprocessor count,
capped by the number of tiles. For this tile, the LDS allocation limits us to
one resident workgroup per WGP, so available WGPs guide the choice. More programs
expose parallel work; fewer leave more tiles per program for within-group prefetch.
The assembly reference uses P equal to 32; the best count depends on the device
and workload.

*[Move to the pipeline column.]*

Depth two means two A slots and two B slots. Their payload already uses 256
kibibytes of LDS; depth three would need 384, exceeding the WGP’s budget before
output staging. Two slots let us overlap input transfers and compute while
leaving room for the small C buffers.

Fused TDM selects A descriptors for waves zero and one, and B for waves two and
three, at one load instruction. All four waves also compute. Multicast masks
separately choose receiving workgroups.

We also stage output in two small slots, vary M first with GROUP_M equal to four,
and optionally share inputs across four-workgroup clusters. Let’s look inside
that pipeline.

## Slide 10 — Pipeline operands, bound their lifetimes

*15:00–17:00 · 2 minutes*

There are two levels of pipelining here. At the outer level, TDM feeds two A
slots and two B slots in LDS. At the inner level, we load operand subtiles from
LDS into registers and feed the WMMAs.

Each K block is 128 elements deep. We process it as four K32 dot steps, loading
the next operand subtile ahead of the current dot. Each source-level dot expands
into multiple matrix instructions for the larger output tile.

The fused TDM loads use issuing-wave masks `0011` for A and `1100` for B. These
masks distribute issuing work among the four waves. The multicast recipient
masks we just discussed are a separate choice across workgroups.

*[Follow the first WMMA, the two LDS loads, and the second WMMA.]*

The assembly shows the inner pipeline directly. Between two WMMAs, we change
the VGPR MSB settings and load two more pieces from LDS. Those loads write
registers that neither adjacent WMMA reads, which gives the schedule independent
memory work to overlap with matrix execution.

The DS waits protect the operands that each WMMA does consume. The value
`0x18` comes from the full queue of earlier LDS operations. It depends on more
than the two loads visible in this small window.

We also bound operand lifetimes at dot boundaries with `amd_sched_barrier()`.
That lets us retain useful overlap without allowing unrelated work to extend
register lifetimes throughout the loop. It is a compiler scheduling constraint;
the TDM completion waits and workgroup handoffs are handled separately.

That organizes the steady K loop. But a persistent kernel also has to move from
one output tile to the next without repeatedly emptying and restarting its
input pipeline.

## Slide 11 — Use the tail to prime the next tile

*17:00–20:00 · 3 minutes*

At a tile boundary, we have an opportunity to prepare future work while finishing
the current tile. The final K iterations already tell us which input slots are
about to become available.

The schedule peels the last two K iterations. Once the current operands have
been read from a slot into registers, and all readers have released that slot,
we can refill it with K0 or K1 of the next tile assigned to this persistent
program. Matrix work on the current tile can continue using the operands already
in registers.

*[Trace the released slots into K0 and K1 of the next tile.]*

There are two implementations on this slide, with different behavior at group
boundaries.

The hybrid path on the left prefetches the next tile within the same group.
Its first tile in each group still needs initial priming. Once the program has
another tile in that group, the preceding tile’s tail can provide its first
two K blocks.

The separate cross-group path on the right also prepares work across group
boundaries. It carries four upcoming boundaries in scalar state, skips empty
groups, and refills that metadata in the preceding tile’s tail. Its output
uses vector stores so that the input rings remain available for prefetched data.

The amount of useful prefetch depends on the tile sequence assigned to each
program. With M equal to 2,048 and N equal to 1,024, there are 32 of these square
tiles per group. If P is also 32, each program receives one tile from that group.
There is no second tile within the group for the hybrid tail to prepare.

If we double M to 4,096 while keeping the other settings, each program gets two
tiles per group. Now within-group prefetch has a next tile to target.

The implementation also requires a depth-two ring, an even number of K128
iterations, and at least two K blocks. Those conditions make the peeled tail
match the ring rotation.

This prepares the next inputs. We still need somewhere to put the current
output without blocking those inputs, which is the LDS-budget problem on the
next slide.

## Slide 12 — Stage C in two small slots

*20:00–22:00 · 2 minutes*

The A and B input rings already occupy 256 kibibytes of logical payload. A full
256-by-256 FP16 output tile adds another 128 kibibytes. Together, that exceeds
the WGP’s 320-kibibyte LDS capacity, even before layout overhead.

The alias-C path reuses A storage for the output. That saves space, but it ties
the next A refill to completion of the C store.

The hybrid path instead divides C into eight 32-row chunks and alternates
between two small output slots. Those slots add 32 kibibytes of payload, bringing
the logical total to 288. The actual allocation also includes layout overhead.

*[Follow the wait and the two barrier pairs in the assembly.]*

This excerpt stages C3. At entry, C1 and C2 may still be in flight. The wait to
one retires C1 because TDM operations complete in issue order within the issuing
wave. C2 can remain outstanding.

The first workgroup barrier communicates that slot one is available for reuse.
The waves then write C3 into that slot. We show one LDS store and omit the
remaining stores at the marked line.

The DS wait establishes that this wave’s LDS writes have completed. The next
workgroup barrier brings all the writers together, after which the tensor-store
instruction can read the completed C3 chunk.

We repeat that process across the eight chunks. The final two stores can stay
in flight as the program enters its next tile, while the A and B rings retain
the prefetched inputs. That is how the output path fits alongside the input
pipeline without requiring a full extra C tile in LDS.

## Slide 13 — A 2 × 2 sharing pattern

*22:00–25:00 · 3 minutes*

Now let’s apply clusters to the actual tile mapping. Each workgroup still
computes one output tile. The goal is to arrange those tiles so that pairs of
workgroups can share A and B inputs.

In the diagram, ranks zero and two use A0. Ranks one and three use A2. Across
the other direction, ranks zero and one use B0, while ranks two and three use B1.

*[Trace one row for A, then one column for B.]*

That gives the A recipient masks `0101` and `1010`. Reading the bits from the
least significant bit, the first selects ranks zero and two, and the second
selects ranks one and three.

For B, the masks are `0011` and `1100`: ranks zero and one, or ranks two and
three. Every individual transfer therefore has two recipients. The four-member
cluster gives us sharing in both operand directions.

These are cluster-rank masks. The source wave masks on the pipeline slide choose
which waves issue the A and B requests inside each workgroup. We need both
choices: who issues locally, and which workgroups receive the data.

Notice also that the two M positions are zero and two. They are nonadjacent
because the program remapping and `GROUP_M` ordering determine the logical tile
coordinates. Physical IDs zero through three map to logical IDs zero, two, four,
and six, producing the four tile coordinates printed below the diagram.

The diagram represents that logical sharing relationship. The hardware places
the cluster members on separate WGPs according to the cluster launch rules.

Every selected recipient issues the matching load, and the kernel synchronizes
the cluster before refilling a released input slot. That keeps one member’s
progress from causing an overwrite while another member is still reading.

With two-workgroup clusters, this implementation shares B only. With four,
it shares both A and B. The wrapper imposes shape and program-count constraints
so the members reach compatible tile and group boundaries.

We have now changed both the pipeline and the opportunities for input reuse.
That gives us several concrete directions to explore next.

## Slide 14 — Which directions should we explore next?

*25:00–30:00 · 5 minutes, including discussion*

There are four directions I would like to explore further. Each has a starting
point in the branch history.

First, operand delivery. An experimental branch already exposed B padding of
eight or sixteen FP16 elements and interleaved A/B input slots. Those test two
different things: bank selection and physical partition placement. An earlier
experiment also tried WMMA operand-reuse hints after register allocation. I would
compare these separately, and use a native compiler path for the reuse hints;
the old code-generation hooks are no longer in this branch.

Second, how much cluster synchronization is useful? A later MXFP branch exposes
one barrier per input block, one every few blocks, or none. Matching requests can
share a transfer; late requests can receive a separate transfer after timeout.
The question is how much alignment helps sharing, and when barrier cost exceeds
that benefit. Local completion waits and safe LDS reuse must remain. We can
compare multicast on and off at each identical barrier interval.

Third, schedule selection by shape. We have alias-C, the within-group hybrid,
and a separate cross-group path. The retained diagnostic comparison does not
show the hybrid winning universally: alias-C has higher steady XDL efficiency
on that shape. With only one tile per program per group, within-group prefetch
has no next tile to target. I would map where each schedule helps as group size,
K, and tiles per program change, including uneven groups.

Fourth, low precision. The MXFP branch already pipelines data and scale rings.
It also explores BK128 versus BK256, reusing A storage for output, and retaining
operands across refills. The useful question is which combination fits each
format's register and LDS budget. Then we can test how those choices transfer
to grouped workloads, with numerical correctness checked for every format.

These are experiments to choose between, rather than a claim that each option
will improve every workload. Which direction would you prioritize?

*[Pause for discussion. Use backup C for LDS conflicts, G for the current
cluster contract, and F/H for formats and resource budgets.]*

*Evidence: `069a0121aa` on `my/gfx1250-kernels-3`; operand-reuse experiment
`647b1a4e05`, hooks removed by `f676cb6304`; barrier intervals `5920df0408` on
`my/gfx1250-kernels-mxfp`; persistent MXFP `70ea55c702`, BK256 output reuse
`d55b4d22f1`, per-format defaults `255abe4594`; [recorded comparison](efficiency.md).
Related-branch options are not part of the deck's code reference. No new
performance measurements were collected for this slide. The reverted
chained-dot changes (`02a632587a`) produced identical default binaries and need
a new production witness before further optimization work.*

## Slide 15 — Backup

*Optional transition · outside the 30-minute main talk*

That ends the main talk. These slides have the ISA details and kernel constraints
we can refer to during discussion.

*[Open the relevant backup for a question; skip the others.]*

## Slide 16 / Backup A — Partial waits need an ordering guarantee

*Optional backup · about 2 minutes*

This slide is the reference for interpreting the waits in the assembly.
CDNA5 separates vector loads, vector stores, LDS operations, scalar-memory
operations, direct asynchronous LDS transfers, and tensor transfers into
different counters. Most of these are six bits wide; KM is five bits.

A wait to N allows the issuing wave to continue when that counter is at most N.
Zero drains the counter. To conclude that a particular earlier operation has
finished, we also need to know the completion ordering for that counter.

TDM loads and stores complete in issue order within one wave. That is what
makes the C-staging example work: with C1 followed by C2, a wait to one ensures
that C1 has retired, while C2 may still be outstanding.

For direct asynchronous LDS transfers, loads are ordered with loads and stores
with stores, but the two directions can report completion out of order relative
to each other. The same partial-wait argument therefore needs more care when
those operations are mixed.

Scalar loads can also complete out of order, so the useful scalar-memory wait
is a zero KM wait. A single-DWORD scalar load increments KM by one; larger loads
increment it by two.

Combined load-plus-DS and store-plus-DS waits remain available. XCNT has another
purpose: it tracks pending address translations. It does not establish that
the payload transfer is complete.

All of these counts belong to a wave. When other waves consume the resulting
data, we still need the corresponding synchronization between them.

## Slide 17 / Backup B — Dependency waits are only part of the contract

*Optional backup · about 1½ minutes*

This example isolates a WMMA result dependency under co-execution. The WMMA
writes its FP32 result into registers zero through seven. The vector add below
it reads register zero, so it depends on that matrix result.

*[Count the four vector NOPs between producer and consumer.]*

For this dense FP16 WMMA followed by a dependent VALU instruction, the ISA
requires four intervening independent VALU instructions or vector NOPs when
co-execution is enabled. I’ve used NOPs to make the spacing visible. In a useful
schedule, independent vector work can occupy those slots.

Scalar instructions do not count as those VALU slots. The scalar descriptor
updates we saw in the main talk are useful independent work, but they do not
substitute for this particular vector-instruction spacing requirement.

The example assumes the operands are initialized and the VGPR MSB fields are
zero. It is an illustrative instruction sequence, rather than an excerpt from
the grouped kernel.

With co-execution disabled, this particular WMMA-to-VALU case needs no inserted
vector NOPs. Other operand relationships have different requirements. Feeding
the result into a following WMMA’s A or B input is one example.

So I check the relevant ISA hazard-table entry for the instruction and the
overlapping operands. Enabling WMMA queuing still leaves that responsibility
with code generation.

## Slide 18 / Backup C — Bank conflicts and partition conflicts differ

*Optional backup · about 2 minutes*

There are two levels of LDS contention to consider here: banks within a wave’s
access, and physical partitions used by different SIMD pairs.

For banks, the mapping is the byte address shifted right by two, masked to six
bits. That gives 64 banks, each four bytes wide. With a Wave32 B32 load,
address four times the lane number uses 32 distinct banks. Address 256 times
the lane number sends every lane to a different word in bank zero.

*[Move from the bank examples to the two rows in the table.]*

Now suppose two waves from opposite SIMD pairs each have a bank-conflict-free
access. One reads the byte range starting at zero, and the other reads the range
starting at 256. Their bank patterns are individually fine, but both ranges
are in the same physical 64-kibibyte LDS partition.

The WGP has five such partitions. SIMD pair zero-and-two belongs to CU zero;
pair one-and-three belongs to CU one. Accesses from these two CUs can contend
when they target the same partition together.
Moving the second range to 65,536 plus its original within-range offsets
changes the partition while preserving its bank pattern in this example.

That gives us two separate layout questions. Padding or swizzling can address
the bank pattern. Partition placement depends on the physical allocation and
which SIMD pairs access which regions. Scheduling can also separate competing
accesses.

These examples use physical LDS addresses, including the allocation base.
For a transpose load, I would analyze the source LDS addresses first, before
considering how the instruction redistributes values into registers.

## Slide 19 / Backup D — S_CLAUSE groups a supported memory class

*Optional backup · about 1½ minutes*

`S_CLAUSE` lets a wave group a sequence of compatible memory instructions.
The first instruction after it selects the memory class. During an uninterrupted
clause, other waves cannot interleave instructions from that class.

In this example, the encoded length is three, which means four following
instructions. All four are global loads, using the same per-lane base address
with successive offsets and separate destination registers.

*[Point to the wait before the clause and the wait after the loads.]*

The dependency wait before the clause makes the registers safe for the memory
operations. The load-counter wait afterward makes the returned values available
to consumers. Clause formation itself provides neither of those completion
guarantees.

Supported classes include non-flat memory, flat memory, indexed LDS operations,
and scalar memory. TDM tensor instructions are illegal inside a clause. The
tensor loads in the grouped kernel therefore cannot be wrapped in `S_CLAUSE`
as a way to group them.

This is a small ISA example to show the mechanism. Whether a clause helps a
particular kernel depends on the available independent requests and the resulting
issue schedule. A stalled clause can leave resources idle, so I would inspect
the affected memory sequence before making it part of an optimization.

## Slide 20 / Backup E — Select rows with descriptor indices

*Optional backup · about 1½ minutes*

TDM can also use a descriptor-provided list of row indices. In the load
direction, those indices select global rows to gather into LDS. In the store
direction, they select the global rows that receive data scattered from LDS.

This mode operates on two-dimensional tiles. Per instruction, the descriptor
can carry up to 16 indices when each index is 16 bits, or eight indices when
each is 32 bits. A larger row set requires more instructions.

Gather can use arbitrary row indices, including repeated rows. There is an
additional condition when relying on out-of-bounds handling: the indices need
to be nondecreasing for that handling to be correct. That condition needs to
be preserved when constructing the descriptor for an irregular access pattern.

The extra descriptor groups, two and three, carry these indices. The global
row selection is therefore expressed in descriptor state, while TDM performs
the tile movement.

Padding and multicast remain load-side capabilities. In particular, the store
operation does not remove padding previously introduced in LDS; the output
layout must be suitable for the store being issued.

The main grouped-GEMM path we discussed uses regular tiles. This backup shows
another way to express data movement when the rows we want are selected by an
index list.

## Slide 21 / Backup F — Scaling is another programming dimension

*Optional backup · about 1 minute*

The main example uses FP16 throughout, but lower-precision formats introduce
another set of layout and scheduling choices.

CDNA4 adds FP4 and FP6 support along with OCP microscaling. CDNA5 extends the
scaling options, including 16- and 32-element blocks and fractional FP4 scales.

For a kernel developer, the data path now includes both the encoded values and
the scale information needed to interpret them. The scale-block size determines
which values share metadata, and the tile layout needs to deliver that metadata
with the corresponding matrix operands.

I would treat that as part of the operand-supply problem: where the scales
reside, when they become available, and how long their registers stay live.
The appropriate tile and pipeline can change with the format and its scaling
rules.

For the kernel in this talk, I’m keeping FP16 inputs and output with FP32
accumulation fixed. That lets us evaluate the scheduling and sharing changes
under the same precision choice. A scaled-format version would need its own
layout, correctness, and efficiency comparison.

## Slide 22 / Backup G — Keep cluster members on matching boundaries

*Optional backup · about 2 minutes*

This is the contract enforced by the clustered version of this kernel. It
exists so that workgroups sharing inputs reach compatible loop and group
boundaries.

The tile is 256 by 256 by 128, the input ring has depth two, and `GROUP_M` is
four. The path uses the within-group hybrid schedule with cross-tile prefetch.
Remapping is chunked, with eight logical XCDs and a chunk size of two.

The groups have equal, positive M. M is divisible by 1,024, and N by 512. P is
divisible by 16 and divides the number of output tiles in each group. Together,
those restrictions make the work assignment regular enough for the selected
members to share the expected inputs and execute compatible synchronization.

The launch expresses the cluster shape with `ctas_per_cga`. The grid and P
count physical workgroups, and each workgroup owns an output tile. The native
fused-load masks choose recipients, and `tlx.cluster_barrier()` protects the
input refills.

The benchmark defaults to four-workgroup clusters. The general wrapper defaults
to ordinary workgroups, which also remain the path for general ragged groups.
The clustered path excludes the other options listed here, including dedicated
C staging, L2 prefetch, and automatic configuration selection.

These restrictions describe this implementation. The architectural cluster
mechanism supports a broader set of programs. To extend this kernel to less
regular shapes, I would first establish how every member follows the correct
request and barrier sequence at tile and group boundaries.

## Slide 23 / Backup H — Smaller tiles can expose more parallel work

*Optional backup · about 1½ minutes*

The large square tile gives each workgroup substantial reuse, but the workload
also needs enough output tiles to keep the available execution resources busy.
For smaller M, reducing the tile height can expose more independent workgroups.

The table compares logical payloads. The square hybrid has 256 kibibytes of
input rings and 32 kibibytes of chunked output staging. The 128-by-256 variant
uses 192 kibibytes for inputs and a dedicated 64-kibibyte output tile, totaling
256 kibibytes before layout overhead.

Register storage matters as well. A 256-by-256 FP32 accumulator distributed
across 128 threads already represents 512 accumulator values per thread.
Operands, addresses, and other state add to that allocation.

The resource panel is still a placeholder, so the table describes
the payload design rather than a completed comparison of compiler allocations.
Both configurations need to be compared with their actual layouts and register
counts.

The selection model also estimates how fully the available program slots are
used by the tile count. That is what I mean here by tile-slot utilization;
it is a scheduling estimate, not a measurement of how busy each hardware CU is.

The asymmetric prefetch path stays within a group. It therefore needs more
tiles per group than persistent programs to have another tile to prepare.
Shrinking the tile changes both the available parallel work and the work each
program performs. I would evaluate those effects together on the selected
shape, using a consistent steady-state XDL-efficiency definition across variants.

## Slide 24 / Backup I — Global → LDS: who chooses the destination?

*Optional backup · about 2 minutes*

These examples show the per-lane direct-to-LDS path. In both cases, each lane’s
global address is already in registers zero and one, and each active lane
copies four bytes.

On CDNA4, we put a shared LDS base offset into M0. The instruction supplies the
lane placement: lane i writes at M0 plus four times i. The scalar NOP after
writing M0 covers the dependency before the LDS load uses that register.
The vector-memory wait establishes completion for the issuing wave.

*[Move to the explicit LDS-address operand in the CDNA5 instruction.]*

On CDNA5, register four supplies an LDS byte offset independently for each lane.
The kernel can therefore choose each lane’s destination placement. The
asynchronous-counter wait establishes completion of these transfers for the
issuing wave.

In both examples, the payload bypasses VGPRs on its way to LDS. The address
operands still occupy registers. These are illustrative sequences with valid
allocations and ready addresses; consumers in other waves also require a
synchronization handoff.

The grouped kernel uses the TDM path from slide four. TDM receives scalar
descriptors describing a whole tile, including global and LDS placement, and
ignores EXEC. These direct copies instead operate on per-lane addresses.

So when I describe programmable LDS addresses on CDNA5, this is the per-lane
capability I mean. When I describe the grouped kernel’s tile loads, stores, and
multicast, I’m referring to TDM and its descriptor-based operations.

## Slide 25 / Backup J — Separate arrival, participants, and state

*Optional backup · split, named, and LDS barriers*

Read this comparison along three axes: when a participant arrives, which
participants must arrive, and where the barrier state lives. “Split” describes
the timing; it is also possible with named and LDS barriers.

Use the timeline to explain buffer ownership. A producer first establishes that
the data is ready, then publishes readiness. A consumer waits before reading.
A separate release handoff tells the producer when that storage can be reused.
Independent work can go between arrival and waiting if it respects those lifetimes.

Connect this to the earlier examples: the C staging excerpt uses the workgroup
signal/wait pair. A pipeline with selected producer and consumer waves can use
named barriers. An asynchronous transfer can signal an LDS barrier through its
TDM descriptor, so transfer completion participates in the handoff.

The comparison does not replace the data path's completion waits. Check the
arrival count, initialization, and reuse protocol for each buffer.

*[Return to slide 5 for cluster scope, slide 12 for C staging, or backup A for
per-wave completion counters. Reference: CDNA5 ISA §§5.6, 10.11.3, 11.2.2.]*
