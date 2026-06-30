#!/usr/bin/env python3
"""Benchmark wrapper for TLX AMD f16 GEMM tutorials."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path


def _load_module(name, filename):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, module_path)
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
    parser = argparse.ArgumentParser(description="Benchmark TLX AMD f16 GEMM.")
    parser.add_argument("--variant", choices=["warp", "single"], default="warp",
                        help="warp: amd-gemm-warp-pipeline_test.py; single: AM-guide single-warp TDM schedule.")
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
    parser.add_argument("--l2-prefetch-distance", "--l2_prefetch_distance", dest="l2_prefetch_distance", type=int,
                        default=0)
    parser.add_argument("--transpose-b", dest="transpose_b", action=argparse.BooleanOptionalAction, default=True)
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

    if args.variant == "warp":
        mod = _load_module("tlx_amd_f16_gemm_warp_pipeline", "amd-gemm-warp-pipeline_test.py")
    else:
        mod = _load_module("tlx_amd_f16_gemm_tdm_pipelined", "amd-tdm-gemm-pipelined_test.py")
        assert args.BK == 128, "single-warp TDM schedule requires BK=128"
        assert args.num_warps == 4, "single-warp TDM schedule requires num_warps=4"
    torch.manual_seed(args.seed)

    a = torch.randn((args.M, args.K), device=mod.DEVICE, dtype=torch.float16)
    b_base = torch.randn((args.K, args.N), device=mod.DEVICE, dtype=torch.float16)
    if args.variant == "single" and args.transpose_b:
        b = b_base.T.contiguous()
    else:
        b = b_base
    c_dtype = torch.bfloat16 if args.variant == "single" else torch.float16
    c = torch.empty((args.M, args.N), device=mod.DEVICE, dtype=c_dtype)

    ref = None
    if args.check:
        ref_b = b.T if args.variant == "single" and args.transpose_b else b
        ref = torch.matmul(a.to(torch.float32), ref_b.to(torch.float32)).to(c_dtype)

    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), )

    def run():
        if args.variant == "single":
            stride_bk, stride_bn = (b.stride(1), b.stride(0)) if args.transpose_b else (b.stride(0), b.stride(1))
            return mod.matmul_tdm_pipelined_single_warp_per_simd_schedule_kernel[grid](
                a,
                b,
                c,
                args.M,
                args.N,
                args.K,
                a.stride(0),
                a.stride(1),
                stride_bk,
                stride_bn,
                c.stride(0),
                c.stride(1),
                BLOCK_M=args.BM,
                BLOCK_N=args.BN,
                BLOCK_K=args.BK,
                NUM_BUFFERS=args.num_buffers,
                TRANSPOSE_B=args.transpose_b,
                L2_PREFETCH_DISTANCE=args.l2_prefetch_distance,
                num_warps=args.num_warps,
                waves_per_eu=max(1, args.num_warps // 4),
            )
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
