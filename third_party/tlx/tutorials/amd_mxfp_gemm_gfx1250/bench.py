"""Benchmark persistent gfx1250 MXFP8 x MXFP8 and MXFP8 x MXFP4 GEMM.

Both variants run at 8192x8192x8192 and 8192x8192x4096 by default, with FP32
output, persistent 256x256 M/N tiles, three input buffers, output staging, and
partial TDM fusion. MX8xMX8 uses BK128 with cross-tile prefetch; MX8xMX4 uses
BK256 with cross-tile prefetch disabled so output can reuse the A ring.
Both use 256 persistent programs, group-M 8, no XCD remapping, and four-workgroup
multicast with a cluster barrier every four input K blocks. SCHED_MODE[2] is off.
Use --first-use-prefetch to test the MX8xMX8 three-buffer schedule with B1 local
to each K step, operand registers rotating across two K steps, and four-read
LDS groups during C01. It retains dedicated FP32 output staging.
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
tutorial kernel's timed region. The default timing budget is 256 ms, matching
a standalone run using --benchmark_num_iters 256 (that parameter is a duration,
not an iteration count). Override it with --benchmark-ms.

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

DEFAULT_CASES = ((8192, 8192, 8192), (8192, 8192, 4096))
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
        help="reuse input LDS for output with E4M3 BK128: three persistent buffers or "
        "four streamed buffers; selects MX8xMX8")
    parser.add_argument("--first-use-prefetch", action="store_true",
                        help="test first-use prefetch with persistent E4M3 BK128 and three buffers; selects MX8xMX8")
    parser.add_argument(
        "--streamed-operands", action="store_true",
        help="test native fragments with persistent E4M3 BK128 and three or four buffers; selects MX8xMX8")
    parser.add_argument("--async-output", action="store_true",
                        help="use direct LDS-to-global C stores with independent waits; requires --streamed-operands")
    parser.add_argument("--cross-tile-prefetch", action=argparse.BooleanOptionalAction, default=None,
                        help="default: enabled for the register pipeline and BK128; disabled for BK256 output reuse")
    parser.add_argument("--num-programs", type=int, default=256,
                        help="persistent program count, capped by tile count (default: 256)")
    parser.add_argument("--xcd-remap", choices=("none", "balanced", "chunked"), default="none",
                        help="persistent program remapping (default: none)")
    parser.add_argument("--num-xcds", type=int, default=8)
    parser.add_argument("--xcd-chunk", type=int, default=2)
    parser.add_argument("--cluster-size", type=int, choices=(1, 2, 4), default=None,
                        help="workgroups per cluster (default: 4, or 1 with --warp-pipeline/--operand-pipeline)")
    parser.add_argument("--cluster-multicast", action=argparse.BooleanOptionalAction, default=True,
                        help="share data and scales within a cluster; disable for a synchronization-only control")
    parser.add_argument(
        "--cluster-barrier-interval", type=int, default=4,
        help="align cluster requests every N input K blocks (default: 4); 0 disables all cluster barriers")
    parser.add_argument("--benchmark-mode", choices=("eager", "graph", "none"), default="eager")
    parser.add_argument("--benchmark-ms", "--benchmark-num-iters", dest="benchmark_num_iters", type=int, default=256,
                        help="timing repetition budget in milliseconds (default: 256; not an iteration count)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, help="create a fresh subdirectory for each case's artifacts")
    parser.add_argument("--dry-run", action="store_true", help="print commands without importing GPU libraries")
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
        args.cluster_size = 1 if args.warp_pipeline or args.operand_pipeline else 4

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
        default_variants = ("mx8xmx8", ) if (args.warp_pipeline or args.operand_pipeline or args.output_tail_reuse
                                             or args.first_use_prefetch or args.streamed_operands) else VARIANT_DTYPES_B
        variants = [(variant, VARIANT_DTYPES_B[variant]) for variant in (args.variant or default_variants)]
    if args.num_programs is not None and args.num_programs <= 0:
        parser.error("--num-programs must be positive")
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
    if not args.persistent and (args.xcd_remap != "none" or args.cluster_size > 1):
        parser.error("XCD remapping and clustering require --persistent")
    if args.cluster_size > 1:
        if args.xcd_remap != "none" and (args.xcd_remap, args.num_xcds, args.xcd_chunk) != ("chunked", 8, 2):
            parser.error("clustering requires --xcd-remap none or chunked with --num-xcds 8 --xcd-chunk 2")
        if args.group_m not in (4, 8) or args.tdm_fusion == "none":
            parser.error("clustering requires --group-m 4 or 8 and partial, 2way, or 4way TDM fusion")
    if args.output_staging and not (args.warp_pipeline or args.operand_pipeline):
        if not args.persistent or args.block_m != 256 or args.block_n != 256:
            parser.error("--output-staging requires persistent 256x256 M/N tiles")
    variants = [(variant, dtype_b, _variant_args(args, dtype_b)) for variant, dtype_b in variants]
    for _, dtype_b, run_args in variants:
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
        if run_args.output_tail_reuse and not (
                run_args.persistent and run_args.output_staging and run_args.tdm_fusion == "partial"
                and run_args.num_warps == 4 and run_args.num_buffers == (4 if run_args.streamed_operands else 3) and
            (run_args.block_m, run_args.block_n, run_args.block_k) == (256, 256, 128)
                and run_args.dtype_a == dtype_b == "float8_e4m3" and not run_args.register_pipeline
                and not run_args.warp_pipeline and not run_args.operand_pipeline):
            parser.error("--output-tail-reuse requires persistent E4M3 x E4M3, 256x256x128 tiles, "
                         "three buffers (four with --streamed-operands), "
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
            if run_args.block_k == 256 and run_args.cross_tile_prefetch and not run_args.register_pipeline:
                parser.error("BK256 output staging reuses the A ring and requires --no-cross-tile-prefetch")
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
                if m % (run_args.group_m * run_args.block_m) or n % (2 * run_args.block_n):
                    parser.error("clustering requires full M groups and an even number of N tiles")
                tiles = (m // run_args.block_m) * (n // run_args.block_n)
                programs = min(run_args.num_programs, tiles) if run_args.num_programs is not None else None
                if programs is not None and (programs <= 0 or programs % 16 or tiles % programs):
                    parser.error("clustering requires --num-programs divisible by 16 and dividing the tile count")
            if (min(m, n, k) <= 0 or m % run_args.block_m or n % run_args.block_n or k % run_args.block_k
                    or k // run_args.block_k < run_args.num_buffers):
                parser.error("cases must contain full M/N/K tiles and at least --num-buffers K tiles")

    return args, cases, variants


def main():
    args, cases, variants = parse_benchmark_args()
    results = []
    runs = [(case, variant, dtype_b, run_args) for case in cases for variant, dtype_b, run_args in variants]
    for index, (case, variant, dtype_b, run_args) in enumerate(runs, 1):
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
            streamed_operands=run_args.streamed_operands, async_output=run_args.async_output,
            sched_mode_2=run_args.sched_mode_2, xcd_remap=run_args.xcd_remap, num_xcds=run_args.num_xcds,
            xcd_chunk=run_args.xcd_chunk, cluster_size=run_args.cluster_size,
            cluster_multicast=run_args.cluster_multicast if run_args.cluster_size > 1 else False,
            cluster_barrier_interval=run_args.cluster_barrier_interval,
            cross_tile_prefetch=run_args.cross_tile_prefetch if run_args.persistent else False,
            requested_programs=run_args.num_programs if run_args.persistent else None,
            benchmark_mode=run_args.benchmark_mode, benchmark_ms=run_args.benchmark_num_iters, seed=run_args.seed)
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
            f"buffers={run_args.num_buffers}, warps={run_args.num_warps}, group_m={run_args.group_m}, fusion={run_args.tdm_fusion}, "
            f"scale_buffers={3 if run_args.warp_pipeline else run_args.num_buffers}, "
            f"split={run_args.tdm_split}, output_staging={run_args.output_staging}, "
            f"register_pipeline={run_args.register_pipeline}, "
            f"output_tail_reuse={run_args.output_tail_reuse}, "
            f"first_use_prefetch={run_args.first_use_prefetch}, "
            f"streamed_operands={run_args.streamed_operands}, "
            f"async_output={run_args.async_output}, "
            f"l2_prefetch_distance={run_args.l2_prefetch_distance}, "
            f"sched_mode_2={run_args.sched_mode_2}, "
            f"xcd_remap={run_args.xcd_remap}, num_xcds={run_args.num_xcds}, xcd_chunk={run_args.xcd_chunk}, "
            f"cluster_size={run_args.cluster_size}, "
            f"cluster_multicast={run_args.cluster_multicast if run_args.cluster_size > 1 else False}, "
            f"cluster_barrier_interval={run_args.cluster_barrier_interval}, "
            f"cross_tile_prefetch={run_args.cross_tile_prefetch if run_args.persistent else False}, "
            f"programs={(run_args.num_programs or 'auto') if run_args.persistent else 'tile count'}, "
            f"timing={run_args.benchmark_mode}/{run_args.benchmark_num_iters} ms", flush=True)
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
