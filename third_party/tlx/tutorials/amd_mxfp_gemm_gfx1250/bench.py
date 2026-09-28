"""Benchmark persistent gfx1250 MXFP8 x MXFP8 and MXFP8 x MXFP4 GEMM.

Both variants run at 8192x8192x8192 and 8192x8192x4096 by default, with FP32
output, persistent 256x256 M/N tiles, three input buffers, output staging, and
partial TDM fusion. MX8xMX8 uses BK128 with cross-tile prefetch; MX8xMX4 uses
BK256 with cross-tile prefetch disabled so output can reuse the A ring.
Each shape/variant runs in a fresh process using the current
interpreter and environment. Tensor allocation and compilation are outside the
tutorial kernel's timed region. The default timing budget is 256 ms, matching
a standalone run using --benchmark_num_iters 256 (that parameter is a duration,
not an iteration count). Override it with --benchmark-ms.

Examples::

    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -BK 128 --num-buffers 4 --no-output-staging
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -BK 256 --num-buffers 2 --no-output-staging
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -BK 256 --num-buffers 2 --no-cross-tile-prefetch
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4 -BK 128
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py -M 8192 -N 8192 -K 4096
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --dry-run

For a single dispatch per shape/variant under a simulator, select
--benchmark-mode none and --output-dir to keep each run's model artifacts in
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
        resolved.block_k = 256 if dtype_b == "float4" else 128
    if resolved.cross_tile_prefetch is None:
        resolved.cross_tile_prefetch = not (resolved.output_staging and resolved.block_k == 256)
    return resolved


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
        "-1",
        "--benchmark_mode",
        args.benchmark_mode,
        "--benchmark_num_iters",
        str(args.benchmark_num_iters),
        "--seed",
        str(args.seed),
    ]
    if args.persistent:
        command.append("--persistent")
    if args.output_staging:
        command.append("--output_staging")
    if args.tdm_split:
        command.append("--tdm_split")
    if args.num_programs is not None:
        command.extend(["--num_programs", str(args.num_programs)])
    if not args.cross_tile_prefetch:
        command.append("--no-cross_tile_prefetch")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", type=_parse_case, help="repeatable M,N,K or MxNxK override")
    parser.add_argument("-M", type=int)
    parser.add_argument("-N", type=int)
    parser.add_argument("-K", type=int)
    parser.add_argument("-BM", "--block-m", dest="block_m", type=int, choices=(128, 256), default=256)
    parser.add_argument("-BN", "--block-n", dest="block_n", type=int, choices=(128, 256), default=256)
    parser.add_argument("-BK", "--block-k", dest="block_k", type=int, choices=(128, 256), default=None,
                        help="default: 128 for MX8xMX8, 256 for MX8xMX4; an override applies to all selected variants")
    parser.add_argument("--num-buffers", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--group-m", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANT_DTYPES_B),
                        help="repeatable variant selection; default: sweep both variants")
    parser.add_argument("--dtype-a", choices=("float8_e4m3", "float8_e5m2"), default="float8_e4m3")
    parser.add_argument("--dtype-b", choices=("float8_e4m3", "float8_e5m2", "float4"),
                        help="select a single weight dtype instead of --variant")
    parser.add_argument("--tdm-fusion", choices=("none", "2way", "4way", "partial"), default="partial")
    parser.add_argument("--tdm-split", action="store_true", help="split descriptors in the nonpersistent kernel")
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-staging", action=argparse.BooleanOptionalAction, default=True,
                        help="stage persistent FP32 output for TDM stores; BK256 reuses the A ring")
    parser.add_argument("--cross-tile-prefetch", action=argparse.BooleanOptionalAction, default=None,
                        help="default: disabled for BK256 output staging, enabled otherwise")
    parser.add_argument("--num-programs", type=int, default=None,
                        help="default: one program per CU, capped by tile count")
    parser.add_argument("--benchmark-mode", choices=("eager", "graph", "none"), default="eager")
    parser.add_argument("--benchmark-ms", "--benchmark-num-iters", dest="benchmark_num_iters", type=int, default=256,
                        help="timing repetition budget in milliseconds (default: 256; not an iteration count)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--output-dir", type=Path, help="create a fresh subdirectory for each case's model artifacts")
    parser.add_argument("--dry-run", action="store_true", help="print commands without importing GPU libraries")
    args = parser.parse_args()

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
        variants = [(variant, VARIANT_DTYPES_B[variant]) for variant in (args.variant or VARIANT_DTYPES_B)]
    if args.num_programs is not None and args.num_programs <= 0:
        parser.error("--num-programs must be positive")
    if args.benchmark_num_iters <= 0:
        parser.error("--benchmark-num-iters must be positive")
    if args.tdm_split and args.persistent:
        parser.error("--tdm-split requires --no-persistent")
    if args.output_staging:
        if not args.persistent or args.block_m != 256 or args.block_n != 256:
            parser.error("--output-staging requires persistent 256x256 M/N tiles")
    variants = [(variant, dtype_b, _variant_args(args, dtype_b)) for variant, dtype_b in variants]
    for _, dtype_b, run_args in variants:
        if run_args.output_staging:
            if run_args.block_k == 256 and run_args.cross_tile_prefetch:
                parser.error("BK256 output staging reuses the A ring and requires --no-cross-tile-prefetch")
            max_buffers = 3 if run_args.block_k == 128 else (3 if dtype_b == "float4" else 2)
            if run_args.num_buffers > max_buffers:
                parser.error(
                    f"BK{run_args.block_k} output staging with {dtype_b} supports at most {max_buffers} buffers")
        # Include the operand/scale padding used by the tutorial. This is an
        # upper bound for the rings; compiler scratch may need additional LDS.
        data_bytes = (run_args.block_m + run_args.block_n // (2 if dtype_b == "float4" else 1)) * run_args.block_k
        scale_bytes = (run_args.block_m + run_args.block_n) * run_args.block_k // 32
        ring_bytes = run_args.num_buffers * (data_bytes * 272 // 256 + scale_bytes * 264 // 256)
        if ring_bytes > 320 * 1024:
            parser.error("input rings exceed gfx1250 LDS capacity; reduce --block-k, --num-buffers, or M/N tiles")
        for m, n, k in cases:
            if (min(m, n, k) <= 0 or m % run_args.block_m or n % run_args.block_n or k % run_args.block_k
                    or k // run_args.block_k < run_args.num_buffers):
                parser.error("cases must contain full M/N/K tiles and at least --num-buffers K tiles")

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
        config = dict(kernel="persistent" if run_args.persistent else "nonpersistent", block_m=run_args.block_m,
                      block_n=run_args.block_n, block_k=run_args.block_k, num_buffers=run_args.num_buffers,
                      group_m=run_args.group_m, tdm_fusion=run_args.tdm_fusion, tdm_split=run_args.tdm_split,
                      output_staging=run_args.output_staging,
                      cross_tile_prefetch=run_args.cross_tile_prefetch if run_args.persistent else False,
                      requested_programs=run_args.num_programs if run_args.persistent else None,
                      benchmark_mode=run_args.benchmark_mode, benchmark_ms=run_args.benchmark_num_iters,
                      seed=run_args.seed)
        results.append(
            dict(variant=variant, dtype_a=args.dtype_a, dtype_b=dtype_b, M=case[0], N=case[1], K=case[2], ms=ms,
                 tflops=tflops, status=status, **config, command=shlex.join(command)))

    if args.dry_run:
        return 0
    for variant, _, run_args in variants:
        print(
            f"\nConfiguration ({variant}): {'persistent' if run_args.persistent else 'nonpersistent'}, "
            f"tile={run_args.block_m}x{run_args.block_n}x{run_args.block_k}, "
            f"buffers={run_args.num_buffers}, group_m={run_args.group_m}, fusion={run_args.tdm_fusion}, "
            f"split={run_args.tdm_split}, output_staging={run_args.output_staging}, "
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
