#!/usr/bin/env python3
"""Benchmark wrapper for the TLX AMD f16 warp-pipelined GEMM tutorial."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path


def _load_f16_gemm_module():
    module_path = Path(__file__).with_name("amd-gemm-warp-pipeline_test.py")
    spec = importlib.util.spec_from_file_location("tlx_amd_f16_gemm_warp_pipeline", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _static_profile(kernel):
    amdgcn = kernel.asm["amdgcn"]
    patterns = {
        "sgpr_count": r"\.sgpr_count:\s+(\d+)",
        "sgpr_spill_count": r"\.sgpr_spill_count:\s+(\d+)",
        "vgpr_count": r"\.vgpr_count:\s+(\d+)",
        "vgpr_spill_count": r"\.vgpr_spill_count:\s+(\d+)",
        "scratch_size": r";\s+ScratchSize:\s+(\d+)",
        "code_len_in_byte": r";\s+codeLenInByte\s+=\s+(\d+)",
        "occupancy": r";\s+Occupancy:\s+(\d+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, amdgcn)
        if match is not None:
            print(f"- {name}: {match.group(1)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD f16 warp-pipelined GEMM.")
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=32, help="BLOCK_K")
    parser.add_argument("--num-warps", "--num_warps", dest="num_warps", type=int, default=8)
    parser.add_argument("--num-buffers", "--num_buffers", dest="num_buffers", type=int, default=2)
    parser.add_argument("--group-m", "--group_m", "--group-size-m", "--group_size_m", dest="group_m", type=int,
                        default=16)
    parser.add_argument("--waves-per-eu", "--waves_per_eu", dest="waves_per_eu", type=int, default=0)
    parser.add_argument("--matrix-instr-nonkdim", "--matrix_instr_nonkdim", dest="matrix_instr_nonkdim", type=int,
                        default=16)
    parser.add_argument("--num-xcds", "--num_xcds", dest="num_xcds", type=int, default=8)
    parser.add_argument("--xcd-chunk", "--xcd_chunk", dest="xcd_chunk", type=int, default=4)
    parser.add_argument("--benchmark-mode", choices=["none", "eager", "graph"], default="none",
                        help="Timing method. graph uses triton.testing.do_bench_cudagraph.")
    parser.add_argument("--benchmark-num-iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for eager benchmark mode.")
    parser.add_argument("--check", action="store_true", help="Run torch reference check after the kernel.")
    parser.add_argument("--bench-ref", action="store_true", help="Also benchmark torch.matmul.")
    parser.add_argument("--no-static-profile", action="store_true", help="Do not print static profile after compile.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _tflops(ms, m, n, k):
    return 2 * m * n * k * 1e-12 / (ms * 1e-3)


def main():
    args = parse_args()

    import torch
    import triton

    mod = _load_f16_gemm_module()
    torch.manual_seed(args.seed)

    a = torch.randn((args.M, args.K), device=mod.DEVICE, dtype=torch.float16)
    b = torch.randn((args.K, args.N), device=mod.DEVICE, dtype=torch.float16)
    c = torch.empty((args.M, args.N), device=mod.DEVICE, dtype=torch.float16)

    ref = None
    if args.check:
        ref = torch.matmul(a, b)

    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), )

    def run():
        return mod.gemm_wp[grid](
            a,
            b,
            c,
            args.M,
            args.N,
            args.K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_M=args.BM,
            BLOCK_N=args.BN,
            BLOCK_K=args.BK,
            GROUP_M=args.group_m,
            NUM_BUFFERS=args.num_buffers,
            NUM_XCDS=args.num_xcds,
            XCD_CHUNK=args.xcd_chunk,
            num_warps=args.num_warps,
            num_stages=1,
            waves_per_eu=args.waves_per_eu,
            matrix_instr_nonkdim=args.matrix_instr_nonkdim,
        )

    kernel = run()
    torch.cuda.synchronize()

    if not args.no_static_profile:
        _static_profile(kernel)

    if args.check:
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print("✅Pass")

    if args.bench_ref:
        if args.benchmark_mode == "graph":
            ref_ms = triton.testing.do_bench_cudagraph(lambda: torch.matmul(a, b), rep=args.benchmark_num_iters)
        else:
            ref_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), warmup=args.warmup,
                                             rep=args.benchmark_num_iters)
        print(f"torch.matmul: {ref_ms} ms, {_tflops(ref_ms, args.M, args.N, args.K):.2f} TFLOPS")

    if args.benchmark_mode == "graph":
        ms = triton.testing.do_bench_cudagraph(run, rep=args.benchmark_num_iters)
        print(f"execution time: {ms} ms, {_tflops(ms, args.M, args.N, args.K):.2f} TFLOPS")
    elif args.benchmark_mode == "eager":
        ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.benchmark_num_iters)
        print(f"execution time: {ms} ms, {_tflops(ms, args.M, args.N, args.K):.2f} TFLOPS")


if __name__ == "__main__":
    main()
