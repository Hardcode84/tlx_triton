"""Benchmark persistent gfx1250 MXFP8 x MXFP8 and MXFP8 x MXFP4 GEMM.

Both variants run at 8192x8192x8192 and 8192x8192x4096 by default, with FP32
output. Each shape/variant runs in a fresh process using the current interpreter
and environment. Tensor allocation and compilation are outside the tutorial
kernel's timed region.

Examples::

    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --csv mxfp.csv
    python third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250/bench.py --variant mx8xmx4
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
        "256",
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
    parser.add_argument("--num-buffers", type=int, choices=(2, 3), default=2)
    parser.add_argument("--group-m", type=int, choices=(1, 2, 4, 8), default=8)
    parser.add_argument("--variant", action="append", choices=tuple(VARIANT_DTYPES_B),
                        help="repeatable variant selection; default: sweep both variants")
    parser.add_argument("--dtype-a", choices=("float8_e4m3", "float8_e5m2"), default="float8_e4m3")
    parser.add_argument("--dtype-b", choices=("float8_e4m3", "float8_e5m2", "float4"),
                        help="select a single weight dtype instead of --variant")
    parser.add_argument("--tdm-fusion", choices=("none", "2way", "4way", "partial"), default="partial")
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cross-tile-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-programs", type=int, default=None,
                        help="default: one program per CU, capped by tile count")
    parser.add_argument("--benchmark-mode", choices=("eager", "graph", "none"), default="eager")
    parser.add_argument("--benchmark-num-iters", type=int, default=32, help="timing repetition budget in milliseconds")
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
    if (args.block_m == args.block_n == 256 and args.num_buffers == 3
            and any(dtype_b != "float4" for _, dtype_b in variants)):
        parser.error("MXFP8 x MXFP8 256x256 tiles with three buffers exceed gfx1250 LDS capacity; "
                     "use two buffers or select --variant mx8xmx4")
    for m, n, k in cases:
        if min(m, n, k) <= 0 or m % args.block_m or n % args.block_n or k % 256 or k // 256 < args.num_buffers:
            parser.error("cases must contain full M/N/K tiles and at least --num-buffers K tiles")

    results = []
    runs = [(case, variant, dtype_b) for case in cases for variant, dtype_b in variants]
    for index, (case, variant, dtype_b) in enumerate(runs, 1):
        label = "x".join(map(str, case))
        command = _command(args, case, dtype_b)
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
        results.append(
            dict(variant=variant, dtype_a=args.dtype_a, dtype_b=dtype_b, M=case[0], N=case[1], K=case[2], ms=ms,
                 tflops=tflops, status=status))

    if args.dry_run:
        return 0
    print(f"\n{'variant':>9} {'dtype A':>12} {'dtype B':>12} {'M':>7} {'N':>7} {'K':>7} "
          f"{'ms':>12} {'TFLOPS':>12}  status")
    for row in results:
        ms = "-" if row["ms"] is None else f"{row['ms']:.4f}"
        tflops = "-" if row["tflops"] is None else f"{row['tflops']:.2f}"
        print(f"{row['variant']:>9} {row['dtype_a']:>12} {row['dtype_b']:>12} "
              f"{row['M']:7d} {row['N']:7d} {row['K']:7d} {ms:>12} {tflops:>12}  {row['status']}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=("variant", "dtype_a", "dtype_b", "M", "N", "K", "ms", "tflops", "status"))
            writer.writeheader()
            writer.writerows(results)
    return int(any(row["status"] != "ok" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
