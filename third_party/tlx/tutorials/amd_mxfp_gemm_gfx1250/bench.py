"""Benchmark persistent gfx1250 MXFP8 x MXFP8 and MXFP8 x MXFP4 GEMM.

Both variants run at 8192x8192x8192, 8192x8192x4096, 16384x16384x16384, and
32768x32768x32768 by default, with FP32 output, persistent 256x256 M/N tiles,
three input buffers, output staging, and partial TDM fusion. MX8xMX8 uses
BK128 with cross-tile prefetch; MX8xMX4 uses BK256 with cross-tile prefetch
disabled by default. Use --variant mx8xmx4
--cross-tile-prefetch to start refilling each A/B stage as its C slot retires.
Use --compact-k-tail to test one loop body for A8W4's final three K256 stages.
It selects MX8xMX4 and can be combined with --cross-tile-prefetch.
Both use 256 persistent programs, group-M 8, chunked XCD remapping of whole
clusters (8 XCDs, chunk 2), and 16-workgroup
multicast in a 4x4 tile layout (each A and B tile shared by four workgroups),
with a cluster barrier every eight input K blocks. MX8xMX8 uses a four-stage
input ring with two C slots for K >= 16384, and three stages otherwise.
MX8xMX4 also enables split cluster barriers and early C stores. These two
options are disabled by default for MX8xMX8 and output-tail reuse. Other
operand schedules do not support these options. SCHED_MODE[2] is off.
Use --cluster-size 4 --cluster-layout 2x2 --cluster-barrier-interval 4 for the
previous four-workgroup clusters, and --num-buffers 3 to keep three stages.
Other cluster sizes and --streamed-operands use a linear layout by default.
Use --first-use-prefetch to test the MX8xMX8 three-buffer schedule with B1 local
to each K step, operand registers rotating across two K steps, and four-read
LDS groups during C01. It retains dedicated FP32 output staging.
Use --wave-specialized-handoff to issue next-tile A/B and scales from waves
0/1, and C transfers from waves 2/3. All four waves compute and fill C; output
reuse waits leave the input queue pending. This selects MX8xMX8 and requires
the three-buffer BK128 schedule with dedicated C slots.
MX8xMX8 unrolls its steady K loop by the three-slot ring for cases with
K >= 8192 (static slot unroll), so LDS operand addresses and loop-carried
operand copies leave the loop. On hardware this is faster at K=8192 and slower
at K=4096, where the larger code costs more than it saves. Use
--static-slot-unroll or --no-static-slot-unroll to force it for every case.
--static-slot-unroll requires the three-buffer BK128 staged-output schedule
and selects MX8xMX8.
With static slot unroll, A and B alternate 64-row pieces between two physical
LDS partitions, so the waves reading one operand subtile contend less for a
partition (1.6% faster at 8192^3 on hardware; no gain in the dynamic loops).
MX8xMX8 also stages FP32 output through three dedicated 64x128 LDS slots
instead of two, keeping more C stores in flight at each tile transition.
Use --output-c-slots to choose another count.
Three-stage MX8xMX8 also overlaps each next tile's first K step with the
FP32 output epilogue (--overlap-peel): once a C quadrant is staged, its
registers take that quadrant's next-tile WMMAs, which run during later
output-slot waits, and the next tile's first ring refill precedes the last
C chunks. This is 0.2% faster at 8192^3 and 1.2% faster at K=4096 on
hardware; the power-bound four-stage ring is unchanged, so it stays off there.
Use --no-overlap-peel to disable it.
--out-dtype fp16 writes FP16 C (FP32 accumulation, one rounding before
staging) for MX8xMX8. It uses six FP16 C slots for the three-buffer ring and
two 64x128 slots for the four-buffer ring.
Without static slot unroll (K < 8192), the MX8xMX8 K loop reads each operand's
scale before its payload, so each quadrant waits only for its own LDS reads
(0.4-1.9% faster on hardware). Use --no-dynamic-scales-first for the
payload-first order. The static loop always reads payloads first.
Use --streamed-operands to test native 32x32 accumulator fragments with one
rolling A set, three spare A fragments, and two B sets. The final three rows cover
operand reads before LDS reuse; refills precede two initial rows that cover
TDM completion before the next-stage wait. The peeled tail
prefetches the next tile and keeps its first operands in registers through
FP32 output stores. It selects MX8xMX8 with three BK128 buffers by default.
Add --num-buffers 4 for another stage of TDM lookahead, using two 64x64 output
slots instead of 64x128: sixteen FP32 output transfers per tile instead of eight.
Add --output-tail-reuse to that four-buffer streamed configuration to restore
two 64x128 panels and eight transfers. One panel reuses the first next-tile A
stage after its operands reach registers; the other has dedicated storage.
Add --async-output to --streamed-operands to store FP32 C directly from LDS
with b128 async stores. Output-buffer waits then leave next-tile TDM transfers
pending. This also supports the four-buffer --output-tail-reuse configuration.
Use --output-tail-reuse to keep two next-tile input stages prefetched while
the retired third A/B stage holds FP32 output. This selects MX8xMX8 with three
BK128 buffers and removes the separate output allocation.
For MX8xMX4, add --variant mx8xmx4 --cross-tile-prefetch --output-tail-reuse.
This keeps next-tile K=0 prefetched and uses the other two A stages for 128x128
FP32 C quadrants. The first two fills leave incoming A/B in flight, and the
last two stores overlap the next tile's first 48 WMMAs. Counted waits then
retire each C slot separately before its input refill. Do not combine this
variant with --compact-k-tail.
Add --handoff-schedule b-prefetch to load B1/B2 and their scales in the old
tile's tail, or --handoff-schedule staggered to also place matrix work between
the next tile's A refills. Both require the MX8xMX4 output-tail-reuse options.
Use --handoff-schedule ab-reuse to stage three C quadrants across the later
A/B ring stages, preload K1 before the final C store, and delay the K2 refill
until 80 WMMAs of the next tile (48 with a three-stage K length).
Add --handoff-tdm-fusion 4way to fuse A, B, and their scales in each ab-reuse
handoff refill. Ordinary K refills retain partial fusion.
With four waves, MX8xMX8 also supports four input buffers: --num-buffers 4
uses smaller output staging chunks and an explicit LDS/WMMA prefetch schedule.
Use --num-warps 8 to test two waves per SIMD on persistent 256x256 tiles;
four waves remain the default.
Use --warp-pipeline to test the nonpersistent eight-wave E4M3 kernel with
two K256 payload slots, three scale slots, partitioned LDS, and FP32 output.
It selects MX8xMX8 and disables clustering by default; K must be divisible
by 512 and at least 1024.
Use --operand-pipeline to test the nonpersistent four-wave E4M3 kernel with
four K128 payload/scale slots, rolling A prefetch, and two B register sets.
Adding -BK 256 selects two partitioned slots, with each transfer feeding two
native K128 compute steps. --l2-prefetch-distance controls its cache lookahead.
It selects MX8xMX8 and disables clustering; K must be divisible by 256 and
at least 512. FP32 output is retained.
For the A8W8 register pipeline, use --variant mx8xmx8 --register-pipeline.
This selects BK256, two input buffers, and four FP32 output panels, retaining
one prefetched input stage while the other stage holds output.
Each shape/variant runs in a fresh process using the current
interpreter and environment. Tensor allocation and compilation are outside the
tutorial kernel's timed region. Timing repeats 256 ms batches (--benchmark-ms;
a duration, not an iteration count): 1 s of warmup batches, then the median
batch over 2 s (--warmup-seconds, --duration-seconds). On gfx1250 the first
batch of a fresh process is inflated by a few slow first launches; after it,
batches settle within about 0.3% of a 10 s warmup. Use --warmup-seconds 0
--duration-seconds 0 for a single cold batch. The AMD code generator revision
is printed and recorded so runs with different code generators are not mixed.

Examples::

    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx8 --num-buffers 4
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx8 --num-warps 8
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --warp-pipeline
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --operand-pipeline
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --streamed-operands
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --streamed-operands --num-buffers 4
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --streamed-operands --num-buffers 4 --output-tail-reuse
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --operand-pipeline -BK 256
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx8 --register-pipeline
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -BK 128 --num-buffers 4 --no-output-staging
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -BK 256 --num-buffers 2 --no-output-staging
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -BK 256 --num-buffers 2 --no-cross-tile-prefetch
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4 -BK 128
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -M 8192 -N 8192 -K 4096
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run

For a single dispatch per shape/variant, select
--benchmark-mode none and --output-dir to keep each run's artifacts in
its own directory.
"""

import argparse
import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_CASES = ((8192, 8192, 8192), (8192, 8192, 4096), (16384, 16384, 16384), (32768, 32768, 32768))
VARIANT_DTYPES_B = {"mx8xmx8": "float8_e4m3", "mx8xmx4": "float4"}
RESULT_RE = re.compile(r"execution time:\s*([0-9.eE+-]+)\s*ms,\s*([0-9.eE+-]+)\s*TFLOPS")


def _parse_case(value):
    try:
        dims = tuple(int(field) for field in re.split("[xX,]", value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be M,N,K or MxNxK") from exc
    if len(dims) != 3 or min(dims) <= 0:
        raise argparse.ArgumentTypeError("case must contain three positive dimensions")
    return dims


def _supports_cluster_output_options(args, dtype_b):
    return (args.persistent and args.output_staging and args.num_warps == 4 and args.num_buffers == 3
            and args.block_m == args.block_n == 256 and args.dtype_a == "float8_e4m3" and args.tdm_fusion == "partial"
            and ((args.block_k == 128 and dtype_b == "float8_e4m3") or (args.block_k == 256 and dtype_b == "float4"))
            and not (args.register_pipeline or args.first_use_prefetch or args.compact_k_tail or args.streamed_operands
                     or args.warp_pipeline or args.operand_pipeline))


# Static slot unroll pays off once the steady loop dominates the larger code's
# tile-transition cost: +1.2% at K=8192, -1.8% at K=4096 on gfx1250.
STATIC_SLOT_UNROLL_MIN_K = 8192
# Grid cluster layouts: R M tiles by C N tiles per cluster of R*C workgroups.
GRID_CLUSTER_LAYOUTS = {"2x2": (2, 2), "4x2": (4, 2), "2x4": (2, 4), "4x4": (4, 4), "8x2": (8, 2), "2x8": (2, 8)}
# With 4x4 multicast, a four-stage MX8xMX8 ring (two C slots, dynamic slots)
# beats three stages from K=16384 (-4 to -5% on gfx1250) but not at K<=8192.
FOUR_BUFFER_MIN_K = 16384


def _supports_static_slot_unroll(args, dtype_b):
    return (args.persistent and args.output_staging and (args.block_m, args.block_n, args.block_k) == (256, 256, 128)
            and args.num_buffers == 3 and args.num_warps == 4 and args.dtype_a == dtype_b == "float8_e4m3"
            and not (args.register_pipeline or args.warp_pipeline or args.operand_pipeline or args.streamed_operands
                     or args.output_tail_reuse or args.first_use_prefetch))


def _static_slot_unroll(args, case, dtype_b):
    if args.static_slot_unroll is not None:
        return args.static_slot_unroll
    return _supports_static_slot_unroll(args, dtype_b) and case[2] >= STATIC_SLOT_UNROLL_MIN_K


def _supports_overlap_peel(args):
    return (args.persistent and args.output_staging and args.cross_tile_prefetch and args.block_k == 128
            and args.num_warps == 4 and args.handoff_schedule == "current"
            and not (args.register_pipeline or args.warp_pipeline or args.operand_pipeline or args.streamed_operands
                     or args.output_tail_reuse or args.first_use_prefetch or args.wave_specialized_handoff
                     or args.split_cluster_barrier or args.early_c_stores))


# Overlapping each next tile's first K step with the C epilogue speeds up the
# three-stage MX8xMX8 schedules on gfx1250 (8192^3: -0.2%, K=4096: -1.2%); the
# power-bound four-stage ring is unchanged (-0.03 to +0.12%).
def _overlap_peel(args, case, dtype_b):
    if not (_supports_overlap_peel(args) and case[2] // args.block_k > args.num_buffers):
        return False
    if args.overlap_peel is not None:
        return args.overlap_peel
    return dtype_b == "float8_e4m3" and args.num_buffers == 3


# Three dedicated 64x128 C slots fit beside the three-buffer BK128 input ring
# and cut A8W8 tile transitions by about 35% on gfx1250. FP16 slots take half
# the LDS, so six fit (8192^3 -1.3%, 8192^2x4096 -1.2% versus three).
DEFAULT_OUTPUT_C_SLOTS = 3
DEFAULT_FP16_OUTPUT_C_SLOTS = 6


def _supports_default_output_c_slots(args, dtype_b):
    return (args.persistent and args.output_staging and (args.block_m, args.block_n, args.block_k) == (256, 256, 128)
            and args.num_buffers == 3 and args.num_warps == 4 and args.dtype_a == dtype_b == "float8_e4m3"
            and not (args.register_pipeline or args.warp_pipeline or args.operand_pipeline or args.streamed_operands
                     or args.output_tail_reuse or args.first_use_prefetch or args.wave_specialized_handoff))


def _variant_args(args, dtype_b):
    resolved = argparse.Namespace(**vars(args))
    if resolved.block_k is None:
        resolved.block_k = 256 if dtype_b == "float4" or resolved.register_pipeline or resolved.warp_pipeline else 128
    if resolved.num_buffers is None:
        if resolved.operand_pipeline:
            resolved.num_buffers = 4 if resolved.block_k == 128 else 2
        else:
            resolved.num_buffers = 2 if resolved.register_pipeline or resolved.warp_pipeline else 3
    if resolved.l2_prefetch_distance is None:
        resolved.l2_prefetch_distance = 1 if resolved.operand_pipeline and resolved.block_k == 256 else -1
    if resolved.cross_tile_prefetch is None:
        resolved.cross_tile_prefetch = not resolved.operand_pipeline and (
            resolved.register_pipeline or not (resolved.output_staging and resolved.block_k == 256))
    default_output_schedule = (dtype_b == "float4" and _supports_cluster_output_options(resolved, dtype_b)
                               and not resolved.output_tail_reuse)
    if resolved.split_cluster_barrier is None:
        resolved.split_cluster_barrier = (default_output_schedule and resolved.cluster_size > 1
                                          and resolved.cluster_barrier_interval > 0)
    if resolved.early_c_stores is None:
        resolved.early_c_stores = default_output_schedule
    # Choose the MX8xMX8 ring depth per case unless the ring, C slots, or the
    # three-stage static unroll were requested explicitly.
    resolved.auto_ring_depth = (args.num_buffers is None and args.output_c_slots is None
                                and args.static_slot_unroll is not True
                                and _supports_default_output_c_slots(resolved, dtype_b))
    if resolved.output_c_slots is None and _supports_default_output_c_slots(resolved, dtype_b):
        resolved.output_c_slots = DEFAULT_OUTPUT_C_SLOTS if args.out_dtype == "fp32" else DEFAULT_FP16_OUTPUT_C_SLOTS
    return resolved


def _case_args(args, case):
    if not (args.auto_ring_depth and case[2] >= FOUR_BUFFER_MIN_K):
        return args
    resolved = argparse.Namespace(**vars(args))
    resolved.num_buffers = 4
    resolved.output_c_slots = None
    resolved.static_slot_unroll = False
    return resolved


def _kernel_name(args):
    if args.streamed_operands:
        return "streamed_operands"
    if args.operand_pipeline:
        return "operand_pipeline"
    if args.warp_pipeline:
        return "warp_pipeline"
    return "persistent" if args.persistent else "nonpersistent"


def _command(args, case, dtype_b):
    m, n, k = case
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("amd_mxfp_gemm_tdm_pipelined.py")),
        "-M",
        str(m),
        "-N",
        str(n),
        "-K",
        str(k),
        "-BM",
        str(args.block_m),
        "-BN",
        str(args.block_n),
        "-BK",
        str(args.block_k),
        "--dtype_a",
        args.dtype_a,
        "--dtype_b",
        dtype_b,
        "--num_buffers",
        str(args.num_buffers),
        "--num_warps",
        str(args.num_warps),
        "--group_size_m",
        str(args.group_m),
        "--schedule",
        "sliceMNK",
        "--tdm_fusion",
        args.tdm_fusion,
        "--transpose_b",
        "--scale_preshuffled",
        "--with_a_scale",
        "--l2_prefetch_distance",
        str(args.l2_prefetch_distance),
        "--benchmark_mode",
        args.benchmark_mode,
        "--benchmark_num_iters",
        str(args.benchmark_num_iters),
        "--benchmark_warmup_seconds",
        str(args.warmup_seconds),
        "--benchmark_duration_seconds",
        str(args.duration_seconds),
        "--seed",
        str(args.seed),
        "--xcd_remap",
        args.xcd_remap,
        "--num_xcds",
        str(args.num_xcds),
        "--xcd_chunk",
        str(args.xcd_chunk),
        "--cluster_size",
        str(args.cluster_size),
        "--cluster_barrier_interval",
        str(args.cluster_barrier_interval),
    ]
    if args.persistent:
        command.append("--persistent")
    if args.output_staging:
        command.append("--output_staging")
    if args.output_tail_reuse:
        command.append("--output_tail_reuse")
    if args.first_use_prefetch:
        command.append("--first_use_prefetch")
    if args.compact_k_tail:
        command.append("--compact_k_tail")
    if args.wave_specialized_handoff:
        command.append("--wave_specialized_handoff")
    if _static_slot_unroll(args, case, dtype_b):
        command.append("--static_slot_unroll")
    if _overlap_peel(args, case, dtype_b):
        command.append("--overlap_peel")
    if args.out_dtype != "fp32":
        command.extend(["--out_dtype", args.out_dtype])
    if args.output_c_slots is not None:
        command.extend(["--output_c_slots", str(args.output_c_slots)])
    if not args.dynamic_scales_first:
        command.append("--no_dynamic_scales_first")
    if args.handoff_schedule != "current":
        command.extend(["--handoff_schedule", args.handoff_schedule])
    if args.handoff_tdm_fusion != "partial":
        command.extend(["--handoff_tdm_fusion", args.handoff_tdm_fusion])
    if args.cluster_layout != "linear":
        command.extend(["--cluster_layout", args.cluster_layout])
    if args.split_cluster_barrier:
        command.append("--split_cluster_barrier")
    if args.early_c_stores:
        command.append("--early_c_stores")
    if args.streamed_operands:
        command.append("--streamed_operands")
    if args.async_output:
        command.append("--async_output")
    if args.register_pipeline:
        command.append("--register_pipeline")
    if args.warp_pipeline:
        command.append("--warp_pipeline")
    if args.operand_pipeline:
        command.append("--operand_pipeline")
    if args.sched_mode_2:
        command.append("--sched_mode_2")
    if not args.cluster_multicast:
        command.append("--no-cluster_multicast")
    if args.tdm_split:
        command.append("--tdm_split")
    if args.num_programs is not None:
        command.extend(["--num_programs", str(args.num_programs)])
    if not args.cross_tile_prefetch:
        command.append("--no-cross_tile_prefetch")
    return command


def parse_benchmark_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", type=_parse_case, help="repeatable M,N,K or MxNxK override")
    parser.add_argument("-M", type=int)
    parser.add_argument("-N", type=int)
    parser.add_argument("-K", type=int)
    parser.add_argument("-BM", "--block-m", dest="block_m", type=int, choices=(128, 256), default=256)
    parser.add_argument("-BN", "--block-n", dest="block_n", type=int, choices=(128, 256), default=256)
    parser.add_argument(
        "-BK", "--block-k", dest="block_k", type=int, choices=(128, 256), default=None,
        help="default: 128 for MX8xMX8, 256 for MX8xMX4, --register-pipeline, or --warp-pipeline; "
        "an override applies to all selected variants")
    parser.add_argument(
        "--num-buffers", type=int, choices=(2, 3, 4), default=None,
        help="input buffers (default: 3, 2 with --register-pipeline/--warp-pipeline, "
        "or 4/2 with --operand-pipeline at BK128/BK256)")
    parser.add_argument("--num-warps", type=int, choices=(4, 8), default=None,
                        help="waves per workgroup (default: 4, or 8 with --warp-pipeline)")
    parser.add_argument("--group-m", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANT_DTYPES_B),
                        help="repeatable variant selection; default: sweep both variants")
    parser.add_argument("--dtype-a", choices=("float8_e4m3", "float8_e5m2"), default="float8_e4m3")
    parser.add_argument("--dtype-b", choices=("float8_e4m3", "float8_e5m2", "float4"),
                        help="select a single weight dtype instead of --variant")
    parser.add_argument("--tdm-fusion", choices=("none", "2way", "4way", "partial"), default="partial")
    parser.add_argument("--tdm-split", action="store_true", help="split descriptors in the nonpersistent kernel")
    parser.add_argument(
        "--persistent", action=argparse.BooleanOptionalAction, default=None,
        help="persistent tile scheduling (default: enabled, except with --warp-pipeline/--operand-pipeline)")
    parser.add_argument("--warp-pipeline", action="store_true",
                        help="nonpersistent E4M3 eight-wave pipeline with two payload and three scale slots")
    parser.add_argument("--register-pipeline", action="store_true",
                        help="A8W8 BK256 register pipeline with four FP32 output panels; use --variant mx8xmx8")
    parser.add_argument("--operand-pipeline", action="store_true",
                        help="nonpersistent E4M3 four-wave pipeline: four K128 or two K256 payload/scale slots")
    parser.add_argument(
        "--l2-prefetch-distance", type=int, default=None,
        help="cache lookahead in input K blocks (default: 1 for --operand-pipeline -BK 256; "
        "-1 disables it for other paths)")
    parser.add_argument("--sched-mode-2", action=argparse.BooleanOptionalAction, default=False,
                        help="set SCHED_MODE[2] to allow WMMA queuing in the persistent kernel (default: disabled)")
    parser.add_argument("--output-staging", action=argparse.BooleanOptionalAction, default=True,
                        help="stage persistent FP32 output for TDM stores; BK256 reuses input storage")
    parser.add_argument(
        "--output-tail-reuse", action="store_true",
        help="reuse input LDS for output: E4M3 BK128 with three persistent/four streamed buffers "
        "(selects MX8xMX8), or explicit --variant mx8xmx4 with BK256 and --cross-tile-prefetch")
    parser.add_argument("--first-use-prefetch", action="store_true",
                        help="test first-use prefetch with persistent E4M3 BK128 and three buffers; selects MX8xMX8")
    parser.add_argument("--compact-k-tail", action="store_true",
                        help="test one loop body for A8W4's final three K256 stages; selects MX8xMX4")
    parser.add_argument(
        "--streamed-operands", action="store_true",
        help="test native fragments with persistent E4M3 BK128 and three or four buffers; selects MX8xMX8")
    parser.add_argument("--async-output", action="store_true",
                        help="use direct LDS-to-global C stores with independent waits; requires --streamed-operands")
    parser.add_argument("--wave-specialized-handoff", action="store_true",
                        help="use separate input/output TDM waves during the three-buffer A8W8 handoff")
    parser.add_argument(
        "--output-c-slots", type=int, default=None, help=
        f"dedicated BK128 output staging slots (default for three-buffer MX8xMX8: {DEFAULT_OUTPUT_C_SLOTS} with FP32 "
        f"and {DEFAULT_FP16_OUTPUT_C_SLOTS} with FP16 output; otherwise 2)")
    parser.add_argument("--dynamic-scales-first", action=argparse.BooleanOptionalAction, default=True,
                        help="read scales before payloads in the A8W8 K loop without static slot unroll (default: on)")
    parser.add_argument(
        "--static-slot-unroll", action=argparse.BooleanOptionalAction, default=None,
        help="unroll the three-buffer A8W8 steady loop by its ring depth with loop-invariant slots "
        f"(default: MX8xMX8 cases with K >= {STATIC_SLOT_UNROLL_MIN_K})")
    parser.add_argument("--out-dtype", choices=("fp32", "fp16"), default="fp32",
                        help="C element type; FP16 needs dedicated output staging (not the MX8xMX4 BK256 output reuse)")
    parser.add_argument(
        "--overlap-peel", action=argparse.BooleanOptionalAction, default=None,
        help="run each next tile's first K step inside the BK128 C epilogue "
        "(default: three-buffer MX8xMX8; forced on only where K has more tiles than buffers)")
    parser.add_argument(
        "--cross-tile-prefetch", action=argparse.BooleanOptionalAction, default=None,
        help="default: enabled for BK128/register pipeline; with A8W4 BK256, overlap C retirement and refills")
    parser.add_argument("--num-programs", type=int, default=256,
                        help="persistent program count, capped by tile count (default: 256)")
    parser.add_argument(
        "--xcd-remap", choices=("none", "balanced", "chunked"), default=None,
        help="persistent program remapping (default: chunked for 4x4 clusters, which remaps whole "
        "clusters; none otherwise)")
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--xcd-chunk", type=int, default=2)
    parser.add_argument(
        "--cluster-size", type=int, choices=(1, 2, 4, 8, 16), default=None,
        help="workgroups per cluster (default: 16, 4 with --streamed-operands, or 1 with "
        "--warp-pipeline/--operand-pipeline)")
    parser.add_argument(
        "--cluster-layout", choices=("linear", ) + tuple(GRID_CLUSTER_LAYOUTS), default=None,
        help="tile arrangement within each cluster (default: 4x4 for 16 workgroups and 2x2 for four, without "
        "--streamed-operands; linear otherwise); RxC layouts share A across C and B across R workgroups")
    parser.add_argument(
        "--split-cluster-barrier", action=argparse.BooleanOptionalAction, default=None,
        help="overlap periodic cluster rendezvous with steady-loop matrix work "
        "(default: enabled for the MX8xMX4 output schedule when cluster barriers are active)")
    parser.add_argument(
        "--early-c-stores", action=argparse.BooleanOptionalAction, default=None,
        help="interleave final quadrant dots and staged FP32 output stores "
        "(default: enabled for the MX8xMX4 output schedule)")
    parser.add_argument("--cluster-multicast", action=argparse.BooleanOptionalAction, default=True,
                        help="share data and scales within a cluster; disable for a synchronization-only control")
    parser.add_argument(
        "--cluster-barrier-interval", type=int, default=None,
        help="align cluster requests every N input K blocks (default: 8 for 4x4, otherwise 4); 0 disables all "
        "cluster barriers")
    parser.add_argument("--benchmark-mode", choices=("eager", "graph", "none"), default="eager")
    parser.add_argument("--benchmark-ms", "--benchmark-num-iters", dest="benchmark_num_iters", type=int, default=256,
                        help="timing repetition budget in milliseconds (default: 256; not an iteration count)")
    parser.add_argument("--warmup-seconds", type=float, default=1.0,
                        help="repeat timing batches for this long before measuring (default: 1)")
    parser.add_argument("--duration-seconds", type=float, default=2.0,
                        help="report the median batch over this period; 0 runs one batch (default: 2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, help="create a fresh subdirectory for each case's artifacts")
    parser.add_argument("--dry-run", action="store_true", help="print commands without importing GPU libraries")
    parser.add_argument("--handoff-schedule", choices=("current", "b-prefetch", "staggered", "ab-reuse"),
                        default="current",
                        help="A8W4 output-tail-reuse experiment: early B loads, staggered A refills, or A/B C staging")
    parser.add_argument("--handoff-tdm-fusion", choices=("partial", "4way"), default="partial",
                        help="fusion for ab-reuse handoff refills (default: partial)")
    args = parser.parse_args(argv)
    if args.async_output and not args.streamed_operands:
        parser.error("--async-output requires --streamed-operands")
    if sum((args.warp_pipeline, args.register_pipeline, args.operand_pipeline, args.streamed_operands)) > 1:
        parser.error(
            "select only one of --warp-pipeline, --register-pipeline, --operand-pipeline, and --streamed-operands")
    if args.persistent is None:
        args.persistent = not (args.warp_pipeline or args.operand_pipeline)
    if args.num_warps is None:
        args.num_warps = 8 if args.warp_pipeline else 4
    if args.cluster_size is None:
        args.cluster_size = 1 if args.warp_pipeline or args.operand_pipeline else (4 if args.streamed_operands else 16)
    if args.cluster_layout is None:
        default_layouts = {4: "2x2", 16: "4x4"}
        args.cluster_layout = ("linear" if args.streamed_operands else default_layouts.get(args.cluster_size, "linear"))
    if args.cluster_barrier_interval is None:
        args.cluster_barrier_interval = 8 if args.cluster_layout == "4x4" else 4
    if args.xcd_remap is None:
        # Chunked remapping of whole 4x4 clusters places clusters that share a B
        # column block on the same XCD (-1.8% at 8192^3, -3.4% at 32768^3).
        args.xcd_remap = "chunked" if args.cluster_layout == "4x4" else "none"

    dims = (args.M, args.N, args.K)
    if any(dim is not None for dim in dims):
        if args.case or any(dim is None for dim in dims):
            parser.error("provide all of -M/-N/-K together, or use --case")
        cases = [dims]
    else:
        cases = args.case or DEFAULT_CASES
    if args.dtype_b is not None:
        if args.variant:
            parser.error("use --variant or --dtype-b, not both")
        variant = "mx8xmx4" if args.dtype_b == "float4" else "mx8xmx8"
        variants = [(variant, args.dtype_b)]
    else:
        default_variants = ("mx8xmx8", ) if (
            args.warp_pipeline or args.operand_pipeline or args.output_tail_reuse or args.first_use_prefetch
            or args.streamed_operands or args.wave_specialized_handoff or args.static_slot_unroll) else VARIANT_DTYPES_B
        if args.compact_k_tail:
            default_variants = ("mx8xmx4", )
        variants = [(variant, VARIANT_DTYPES_B[variant]) for variant in (args.variant or default_variants)]
    if args.num_programs is not None and args.num_programs <= 0:
        parser.error("--num-programs must be positive")
    if args.warmup_seconds < 0 or args.duration_seconds < 0:
        parser.error("--warmup-seconds and --duration-seconds must be nonnegative")
    if args.benchmark_num_iters <= 0:
        parser.error("--benchmark-num-iters must be positive")
    if args.tdm_split and args.persistent:
        parser.error("--tdm-split requires --no-persistent")
    if args.sched_mode_2 and not args.persistent:
        parser.error("--sched-mode-2 requires --persistent")
    if args.num_warps == 8 and not args.warp_pipeline:
        if not args.persistent or args.block_m != 256 or args.block_n != 256:
            parser.error("--num-warps 8 requires persistent 256x256 M/N tiles")
        if args.output_staging and args.num_buffers == 4:
            parser.error("eight-wave output staging supports at most three input buffers")
    if args.num_xcds <= 0 or args.xcd_chunk <= 0:
        parser.error("--num-xcds and --xcd-chunk must be positive")
    if args.cluster_barrier_interval < 0:
        parser.error("--cluster-barrier-interval must be nonnegative")
    if args.cluster_layout in GRID_CLUSTER_LAYOUTS:
        rows, cols = GRID_CLUSTER_LAYOUTS[args.cluster_layout]
        if args.cluster_size != rows * cols or args.streamed_operands:
            parser.error(f"--cluster-layout {args.cluster_layout} requires --cluster-size {rows * cols} "
                         "without --streamed-operands")
        if args.group_m % rows:
            parser.error(f"--cluster-layout {args.cluster_layout} requires --group-m divisible by {rows}")
    if not args.persistent and (args.xcd_remap != "none" or args.cluster_size > 1):
        parser.error("XCD remapping and clustering require --persistent")
    if args.cluster_size > 1:
        if (args.xcd_remap != "none" and args.cluster_layout in ("linear", "2x2")
                and (args.xcd_remap, args.num_xcds, args.xcd_chunk) != ("chunked", 8, 2)):
            parser.error("linear and 2x2 clusters require --xcd-remap none or chunked with --num-xcds 8 "
                         "--xcd-chunk 2")
        if args.group_m not in (4, 8) or args.tdm_fusion == "none":
            parser.error("clustering requires --group-m 4 or 8 and partial, 2way, or 4way TDM fusion")
    if args.output_staging and not (args.warp_pipeline or args.operand_pipeline):
        if not args.persistent or args.block_m != 256 or args.block_n != 256:
            parser.error("--output-staging requires persistent 256x256 M/N tiles")
    variants = [(variant, dtype_b, _variant_args(args, dtype_b)) for variant, dtype_b in variants]
    for _, dtype_b, run_args in variants:
        if run_args.wave_specialized_handoff and not (
                run_args.persistent and run_args.output_staging and run_args.cross_tile_prefetch and
            (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 128) and run_args.num_buffers == 3
                and run_args.num_warps == 4 and run_args.tdm_fusion == "partial"
                and run_args.dtype_a == dtype_b == "float8_e4m3"
                and not (run_args.register_pipeline or run_args.warp_pipeline or run_args.operand_pipeline
                         or run_args.streamed_operands or run_args.output_tail_reuse or run_args.first_use_prefetch
                         or run_args.split_cluster_barrier or run_args.early_c_stores)):
            parser.error("--wave-specialized-handoff requires persistent MX8xMX8, 256x256x128 tiles, three buffers, "
                         "four warps, partial fusion, dedicated output staging, and cross-tile prefetch "
                         "without other schedule variants")
        if run_args.output_c_slots is not None and not (
                run_args.output_c_slots >= 2 and run_args.persistent and run_args.output_staging
                and run_args.block_k == 128 and
                not (run_args.register_pipeline or run_args.warp_pipeline or run_args.operand_pipeline
                     or run_args.streamed_operands or run_args.output_tail_reuse or run_args.wave_specialized_handoff)):
            parser.error("--output-c-slots requires at least two slots and persistent BK128 dedicated output staging "
                         "without other output schedules")
        if run_args.out_dtype != "fp32" and (
                run_args.warp_pipeline or run_args.operand_pipeline or run_args.streamed_operands or
            (run_args.persistent and
             (run_args.register_pipeline or run_args.output_tail_reuse or run_args.handoff_schedule == "ab-reuse" or
              (run_args.output_staging and run_args.block_k == 256)))):
            parser.error(f"--out-dtype {run_args.out_dtype} requires dedicated BK128 output staging or direct "
                         "stores (use --variant mx8xmx8 or --block-k 128)")
        if run_args.overlap_peel and not _supports_overlap_peel(run_args):
            parser.error("--overlap-peel requires persistent BK128 tiles with four warps, dedicated output staging, "
                         "cross-tile prefetch, and the current handoff schedule without other schedule variants")
        if run_args.static_slot_unroll and not _supports_static_slot_unroll(run_args, dtype_b):
            parser.error("--static-slot-unroll requires persistent MX8xMX8, 256x256x128 tiles, three buffers, "
                         "four warps, and dedicated output staging without other operand schedules")
        if run_args.split_cluster_barrier and (run_args.cluster_size == 1 or run_args.cluster_barrier_interval == 0 or
                                               (run_args.output_tail_reuse
                                                and run_args.handoff_schedule != "ab-reuse")):
            parser.error("--split-cluster-barrier requires clustering, a positive barrier interval, "
                         "and --handoff-schedule ab-reuse when using --output-tail-reuse")
        if (run_args.split_cluster_barrier
                or run_args.early_c_stores) and not _supports_cluster_output_options(run_args, dtype_b):
            parser.error("--split-cluster-barrier and --early-c-stores require persistent E4M3 x E4M3 BK128 or "
                         "E4M3 x FP4 BK256, 256x256 tiles, three buffers, four warps, partial TDM fusion, "
                         "and output staging without an alternative operand schedule")
        if run_args.compact_k_tail and not (
                run_args.persistent and run_args.output_staging and run_args.tdm_fusion == "partial" and
            (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 256) and run_args.num_buffers == 3
                and run_args.num_warps == 4 and run_args.dtype_a == "float8_e4m3" and dtype_b == "float4"
                and not (run_args.register_pipeline or run_args.warp_pipeline or run_args.operand_pipeline
                         or run_args.streamed_operands or run_args.output_tail_reuse or run_args.first_use_prefetch)):
            parser.error("--compact-k-tail requires persistent E4M3 x FP4, 256x256x256 tiles, three buffers, "
                         "four warps, partial TDM fusion, and output staging")
        if run_args.streamed_operands and not (run_args.persistent and run_args.output_staging
                                               and run_args.cross_tile_prefetch and run_args.tdm_fusion == "partial"
                                               and run_args.num_warps == 4 and run_args.num_buffers in (3, 4) and
                                               (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 128)
                                               and run_args.dtype_a == dtype_b == "float8_e4m3" and
                                               (not run_args.output_tail_reuse or run_args.num_buffers == 4)
                                               and not run_args.first_use_prefetch and not run_args.tdm_split):
            parser.error(
                "--streamed-operands requires persistent E4M3 x E4M3, 256x256x128 tiles, three or four buffers, "
                "four warps, partial unsplit TDM, cross-tile prefetch, output staging, "
                "and four buffers for output tail reuse")
        if run_args.first_use_prefetch and not (
                run_args.persistent and run_args.output_staging and run_args.tdm_fusion == "partial"
                and run_args.num_warps == 4 and run_args.num_buffers == 3 and
            (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 128)
                and run_args.dtype_a == dtype_b == "float8_e4m3" and not run_args.output_tail_reuse
                and not run_args.register_pipeline and not run_args.warp_pipeline and not run_args.operand_pipeline):
            parser.error("--first-use-prefetch requires persistent E4M3 x E4M3, 256x256x128 tiles, three buffers, "
                         "four warps, partial TDM fusion, and dedicated output staging")
        a8w4_output_tail = ((run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 256)
                            and run_args.dtype_a == "float8_e4m3" and dtype_b == "float4" and run_args.num_buffers == 3
                            and run_args.cross_tile_prefetch and not run_args.streamed_operands)
        a8w8_output_tail = ((run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 128)
                            and run_args.dtype_a == dtype_b == "float8_e4m3"
                            and run_args.num_buffers == (4 if run_args.streamed_operands else 3))
        if run_args.handoff_schedule != "current" and not (a8w4_output_tail and run_args.output_tail_reuse
                                                           and not run_args.compact_k_tail):
            parser.error("--handoff-schedule requires --variant mx8xmx4 --cross-tile-prefetch "
                         "--output-tail-reuse without --compact-k-tail")
        if run_args.handoff_tdm_fusion != "partial" and run_args.handoff_schedule != "ab-reuse":
            parser.error("--handoff-tdm-fusion requires --handoff-schedule ab-reuse")
        if run_args.output_tail_reuse and not (run_args.persistent and run_args.output_staging
                                               and run_args.tdm_fusion == "partial" and run_args.num_warps == 4 and
                                               (a8w8_output_tail or a8w4_output_tail) and not run_args.register_pipeline
                                               and not run_args.warp_pipeline and not run_args.operand_pipeline):
            parser.error("--output-tail-reuse requires persistent E4M3 x E4M3 with 256x256x128 tiles "
                         "and three buffers (four with --streamed-operands), or E4M3 x FP4 with "
                         "256x256x256 tiles, three buffers, and --cross-tile-prefetch; "
                         "four warps, partial TDM fusion, and output staging")
        if run_args.l2_prefetch_distance < -1:
            parser.error("--l2-prefetch-distance must be at least -1")
        if (run_args.persistent or run_args.warp_pipeline) and run_args.l2_prefetch_distance != -1:
            parser.error("--l2-prefetch-distance requires a nonpersistent four-wave kernel")
        valid_operand_ring = (run_args.block_k, run_args.num_buffers) in ((128, 4), (256, 2))
        valid_operand_prefetch = run_args.l2_prefetch_distance == -1 or (run_args.block_k == 256
                                                                         and run_args.l2_prefetch_distance in (1, 2))
        if run_args.operand_pipeline and not (not run_args.persistent and run_args.output_staging
                                              and run_args.num_warps == 4 and valid_operand_ring and
                                              (run_args.block_m, run_args.block_n) == (256, 256)
                                              and run_args.dtype_a == dtype_b == "float8_e4m3"
                                              and run_args.tdm_fusion == "partial" and not run_args.tdm_split
                                              and not run_args.cross_tile_prefetch and valid_operand_prefetch):
            parser.error("--operand-pipeline requires nonpersistent E4M3 x E4M3, 256x256 tiles, four warps, "
                         "BK128/four buffers or BK256/two buffers, staged output, partial unsplit TDM, "
                         "no cross-tile prefetch, and L2 prefetch -1 (or 1/2 with BK256)")
        if run_args.warp_pipeline and not (
                not run_args.persistent and not run_args.register_pipeline and run_args.output_staging
                and run_args.num_warps == 8 and run_args.num_buffers == 2 and
            (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 256)
                and run_args.dtype_a == dtype_b == "float8_e4m3" and run_args.tdm_fusion == "partial"
                and not run_args.tdm_split and not run_args.cross_tile_prefetch):
            parser.error("--warp-pipeline requires nonpersistent E4M3 x E4M3, 256x256x256 tiles, eight warps, "
                         "two payload buffers, staged output, partial unsplit TDM, and no cross-tile prefetch")
        if run_args.register_pipeline and not (
                run_args.persistent and run_args.output_staging and run_args.tdm_fusion == "partial"
                and run_args.num_warps == 4 and run_args.num_buffers == 2 and
            (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 256) and dtype_b != "float4"):
            parser.error("--register-pipeline requires A8W8, persistent 256x256x256 tiles, two buffers, "
                         "four warps, partial TDM fusion, and output staging; select --variant mx8xmx8")
        if run_args.output_staging:
            if (run_args.block_k == 256 and run_args.cross_tile_prefetch and not run_args.register_pipeline
                    and not (dtype_b == "float4" and run_args.num_buffers == 3 and run_args.num_warps == 4)):
                parser.error("BK256 cross-tile output staging requires three buffers, FP4 B, and four warps; "
                             "use --no-cross-tile-prefetch otherwise")
            max_buffers = 4 if run_args.block_k == 128 else (3 if dtype_b == "float4" else 2)
            if run_args.num_buffers > max_buffers:
                parser.error(
                    f"BK{run_args.block_k} output staging with {dtype_b} supports at most {max_buffers} buffers")
        # Include the operand/scale padding used by the tutorial. This is an
        # upper bound for the rings; compiler scratch may need additional LDS.
        data_bytes = (run_args.block_m + run_args.block_n // (2 if dtype_b == "float4" else 1)) * run_args.block_k
        scale_bytes = (run_args.block_m + run_args.block_n) * run_args.block_k // 32
        scale_buffers = 3 if run_args.warp_pipeline else run_args.num_buffers
        ring_bytes = run_args.num_buffers * data_bytes * 272 // 256 + scale_buffers * scale_bytes * 264 // 256
        if ring_bytes > 320 * 1024:
            parser.error("input rings exceed gfx1250 LDS capacity; reduce --block-k, --num-buffers, or M/N tiles")
        for m, n, k in cases:
            if run_args.operand_pipeline and (k < 512 or k % 256):
                parser.error("--operand-pipeline requires K >= 512 divisible by 256")
            if run_args.warp_pipeline and (k < 1024 or k % 512):
                parser.error("--warp-pipeline requires K >= 1024 divisible by 512")
            if run_args.cluster_size > 1:
                cols = GRID_CLUSTER_LAYOUTS.get(run_args.cluster_layout, (1, 2))[1]
                if m % (run_args.group_m * run_args.block_m) or n % (max(2, cols) * run_args.block_n):
                    parser.error("clustering requires full M groups and N tiles divisible by the cluster columns "
                                 "(at least two)")
                tiles = (m // run_args.block_m) * (n // run_args.block_n)
                programs = min(run_args.num_programs, tiles) if run_args.num_programs is not None else None
                if programs is not None and (programs <= 0 or programs % 16 or tiles % programs):
                    parser.error("clustering requires --num-programs divisible by 16 and dividing the tile count")
            if (min(m, n, k) <= 0 or m % run_args.block_m or n % run_args.block_n or k % run_args.block_k
                    or k // run_args.block_k < run_args.num_buffers):
                parser.error("cases must contain full M/N/K tiles and at least --num-buffers K tiles")

    return args, cases, variants


def _amd_codegen_revision():
    try:
        from triton.backends.amd.compiler import get_amd_codegen_revision
        return get_amd_codegen_revision()
    except Exception as error:  # Report rather than hide an unusable code generator.
        return f"unknown ({error})"


def main():
    args, cases, variants = parse_benchmark_args()
    codegen = None if args.dry_run else _amd_codegen_revision()
    if codegen is not None:
        print(f"AMD code generator: {codegen}", flush=True)
    results = []
    runs = [(case, variant, dtype_b, run_args) for case in cases for variant, dtype_b, run_args in variants]
    for index, (case, variant, dtype_b, run_args) in enumerate(runs, 1):
        run_args = _case_args(run_args, case)
        label = "x".join(map(str, case))
        command = _command(run_args, case, dtype_b)
        print(f"\n[{index}/{len(runs)}] {variant} {label} ({args.dtype_a} x {dtype_b})", flush=True)
        print(f"$ {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        cwd = None
        if args.output_dir is not None:
            cwd = args.output_dir.resolve() / f"{index:02d}-{variant}-{label}"
            cwd.mkdir(parents=True, exist_ok=False)
        ms = tflops = None
        with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                match = RESULT_RE.search(line)
                if match:
                    ms, tflops = map(float, match.groups())
            returncode = process.wait()
        status = "ok" if returncode == 0 else f"exit {returncode}"
        if status == "ok" and args.benchmark_mode != "none" and ms is None:
            status = "missing timing"
        kernel_name = _kernel_name(run_args)
        config = dict(
            kernel=kernel_name, block_m=run_args.block_m, block_n=run_args.block_n, block_k=run_args.block_k,
            num_buffers=run_args.num_buffers, scale_buffers=3 if run_args.warp_pipeline else run_args.num_buffers,
            num_warps=run_args.num_warps, group_m=run_args.group_m, tdm_fusion=run_args.tdm_fusion,
            tdm_split=run_args.tdm_split, l2_prefetch_distance=run_args.l2_prefetch_distance,
            output_staging=run_args.output_staging, register_pipeline=run_args.register_pipeline,
            output_tail_reuse=run_args.output_tail_reuse, first_use_prefetch=run_args.first_use_prefetch,
            wave_specialized_handoff=run_args.wave_specialized_handoff, static_slot_unroll=_static_slot_unroll(
                run_args, case, dtype_b), overlap_peel=_overlap_peel(run_args, case,
                                                                     dtype_b), output_c_slots=run_args.output_c_slots
            or 2, dynamic_scales_first=run_args.dynamic_scales_first, out_dtype=run_args.out_dtype,
            handoff_schedule=run_args.handoff_schedule, compact_k_tail=run_args.compact_k_tail,
            handoff_tdm_fusion=run_args.handoff_tdm_fusion, streamed_operands=run_args.streamed_operands,
            async_output=run_args.async_output, sched_mode_2=run_args.sched_mode_2, xcd_remap=run_args.xcd_remap,
            num_xcds=run_args.num_xcds, xcd_chunk=run_args.xcd_chunk, cluster_size=run_args.cluster_size,
            cluster_multicast=run_args.cluster_multicast if run_args.cluster_size > 1 else False,
            cluster_barrier_interval=run_args.cluster_barrier_interval, cluster_layout=run_args.cluster_layout,
            split_cluster_barrier=run_args.split_cluster_barrier, early_c_stores=run_args.early_c_stores,
            cross_tile_prefetch=run_args.cross_tile_prefetch if run_args.persistent else False,
            requested_programs=run_args.num_programs if run_args.persistent else None,
            benchmark_mode=run_args.benchmark_mode, benchmark_ms=run_args.benchmark_num_iters,
            warmup_seconds=run_args.warmup_seconds, duration_seconds=run_args.duration_seconds, seed=run_args.seed,
            amd_codegen=codegen)
        results.append(
            dict(variant=variant, dtype_a=args.dtype_a, dtype_b=dtype_b, M=case[0], N=case[1], K=case[2], ms=ms,
                 tflops=tflops, status=status, **config, command=shlex.join(command)))

    if args.dry_run:
        return 0
    for variant, _, run_args in variants:
        kernel_name = _kernel_name(run_args)
        print(
            f"\nConfiguration ({variant}): {kernel_name}, "
            f"tile={run_args.block_m}x{run_args.block_n}x{run_args.block_k}, "
            f"buffers={run_args.num_buffers}"
            f"{f' (4 with 2 C slots for K>={FOUR_BUFFER_MIN_K})' if run_args.auto_ring_depth else ''}, "
            f"warps={run_args.num_warps}, group_m={run_args.group_m}, fusion={run_args.tdm_fusion}, "
            f"scale_buffers={3 if run_args.warp_pipeline else run_args.num_buffers}, "
            f"split={run_args.tdm_split}, output_staging={run_args.output_staging}, "
            f"register_pipeline={run_args.register_pipeline}, "
            f"output_tail_reuse={run_args.output_tail_reuse}, "
            f"handoff_schedule={run_args.handoff_schedule}, "
            f"handoff_tdm_fusion={run_args.handoff_tdm_fusion}, "
            f"first_use_prefetch={run_args.first_use_prefetch}, "
            f"wave_specialized_handoff={run_args.wave_specialized_handoff}, "
            f"static_slot_unroll={'auto' if run_args.static_slot_unroll is None else run_args.static_slot_unroll}, "
            f"overlap_peel={'auto' if run_args.overlap_peel is None else run_args.overlap_peel}, "
            f"output_c_slots={run_args.output_c_slots or 2}, "
            f"out_dtype={run_args.out_dtype}, "
            f"dynamic_scales_first={run_args.dynamic_scales_first}, "
            f"compact_k_tail={run_args.compact_k_tail}, "
            f"streamed_operands={run_args.streamed_operands}, "
            f"async_output={run_args.async_output}, "
            f"l2_prefetch_distance={run_args.l2_prefetch_distance}, "
            f"sched_mode_2={run_args.sched_mode_2}, "
            f"xcd_remap={run_args.xcd_remap}, num_xcds={run_args.num_xcds}, xcd_chunk={run_args.xcd_chunk}, "
            f"cluster_size={run_args.cluster_size}, "
            f"cluster_multicast={run_args.cluster_multicast if run_args.cluster_size > 1 else False}, "
            f"cluster_barrier_interval={run_args.cluster_barrier_interval}, "
            f"cluster_layout={run_args.cluster_layout}, split_cluster_barrier={run_args.split_cluster_barrier}, "
            f"early_c_stores={run_args.early_c_stores}, "
            f"cross_tile_prefetch={run_args.cross_tile_prefetch if run_args.persistent else False}, "
            f"programs={(run_args.num_programs or 'auto') if run_args.persistent else 'tile count'}, "
            f"timing={run_args.benchmark_mode}/{run_args.benchmark_num_iters} ms batches, "
            f"warmup={run_args.warmup_seconds:g} s, median over {run_args.duration_seconds:g} s, "
            f"amd_codegen={codegen}", flush=True)
    print(f"\n{'variant':>9} {'dtype A':>12} {'dtype B':>12} {'M':>7} {'N':>7} {'K':>7} "
          f"{'BK':>4} {'ms':>12} {'TFLOPS':>12}  status")
    for row in results:
        ms = "-" if row["ms"] is None else f"{row['ms']:.4f}"
        tflops = "-" if row["tflops"] is None else f"{row['tflops']:.2f}"
        print(f"{row['variant']:>9} {row['dtype_a']:>12} {row['dtype_b']:>12} "
              f"{row['M']:7d} {row['N']:7d} {row['K']:7d} {row['block_k']:4d} {ms:>12} {tflops:>12}  {row['status']}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=tuple(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
