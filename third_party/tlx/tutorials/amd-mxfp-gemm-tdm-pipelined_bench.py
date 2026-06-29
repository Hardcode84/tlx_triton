#!/usr/bin/env python3
"""Benchmark wrapper for the TLX gfx1250 MXFP TDM GEMM tutorial."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path


def _load_tlx_mxfp_module():
    module_path = Path(__file__).with_name("amd-mxfp-gemm-tdm-pipelined_test.py")
    spec = importlib.util.spec_from_file_location("tlx_mxfp_gemm_tdm_pipelined", module_path)
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


def _as_device_tensor(x, dtype, transpose=False):
    if dtype == "float4":
        data = x.data
        return data.T.contiguous().cuda() if transpose else data.contiguous().cuda()
    return x.T.contiguous().cuda() if transpose else x.contiguous().cuda()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TLX gfx1250 MXFP TDM GEMM.")
    parser.add_argument("-M", type=int, default=2048)
    parser.add_argument("-N", type=int, default=1024)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=256, help="BLOCK_K")
    parser.add_argument("--num-warps", "--num_warps", dest="num_warps", type=int, default=4, choices=[4, 8])
    parser.add_argument("--num-buffers", "--num_buffers", dest="num_buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--group-size-m", "--group_size_m", dest="group_size_m", type=int, default=8,
                        choices=[1, 2, 4, 8])
    parser.add_argument("--scale-preshuffled", "--scale_preshuffled", dest="scale_preshuffled", action="store_true")
    parser.add_argument("--with-a-scale", "--with_a_scale", dest="with_a_scale", action="store_true")
    parser.add_argument("--schedule", choices=["baseline", "sliceK", "sliceNK", "sliceMNK"], default="sliceMNK")
    parser.add_argument("--tdm-fusion", "--tdm_fusion", dest="tdm_fusion", choices=["none", "2way", "4way", "partial"],
                        default="none")
    parser.add_argument("--l2-prefetch-distance", "--l2_prefetch_distance", dest="l2_prefetch_distance", type=int,
                        default=2)
    parser.add_argument("--dtype-a", "--dtype_a", dest="dtype_a", choices=["float8_e4m3", "float8_e5m2", "float4"],
                        default="float8_e4m3")
    parser.add_argument("--dtype-b", "--dtype_b", dest="dtype_b", choices=["float8_e4m3", "float8_e5m2", "float4"],
                        default="float8_e4m3")
    parser.add_argument("--transpose-b", dest="transpose_b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-mode", choices=["none", "eager", "graph"], default="none",
                        help="Timing method. graph uses triton.testing.do_bench_cudagraph.")
    parser.add_argument("--benchmark-num-iters", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations for eager benchmark mode.")
    parser.add_argument("--check", action="store_true", help="Run torch reference check after the kernel.")
    parser.add_argument("--no-static-profile", action="store_true", help="Do not print static profile after compile.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    import torch
    import triton
    from triton.tools.mxfp import MXScaleTensor

    mod = _load_tlx_mxfp_module()
    scale_block = 32

    torch.manual_seed(args.seed)
    a = mod._init_data(args.dtype_a, args.M, args.K)
    b = mod._init_data(args.dtype_b, args.K, args.N)
    a_scale = (MXScaleTensor(size=(args.M, triton.cdiv(args.K, scale_block))).random(
        high=32.0).data if args.with_a_scale else None)
    b_scale = MXScaleTensor(size=(args.N, triton.cdiv(args.K, scale_block))).random(high=32.0).data

    ref = None
    if args.check:
        ref = mod.torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, args.M, args.N, args.K)

    a_scale_input = mod.pack_scale(a_scale) if args.scale_preshuffled else a_scale
    b_scale_input = mod.pack_scale(b_scale) if args.scale_preshuffled else b_scale

    if args.dtype_a == "float4":
        a = a.to_packed_tensor(dim=1)
    if args.dtype_b == "float4":
        b = b.to_packed_tensor(dim=0)

    a_d = _as_device_tensor(a, args.dtype_a)
    b_d = _as_device_tensor(b, args.dtype_b, transpose=args.transpose_b)
    b_scale_d = b_scale_input.cuda()
    a_scale_d = a_scale_input.cuda() if a_scale_input is not None else b_scale_d
    c_d = torch.empty((args.M, args.N), device=a_d.device, dtype=torch.float32)

    stride_bk, stride_bn = (b_d.stride(0), b_d.stride(1)) if not args.transpose_b else (b_d.stride(1), b_d.stride(0))
    grid = (triton.cdiv(args.M, args.BM) * triton.cdiv(args.N, args.BN), )

    def run():
        return mod.mxgemm_tdm_pipelined_kernel[grid](
            a_d,
            b_d,
            c_d,
            a_scale_d,
            b_scale_d,
            args.M,
            args.N,
            args.K,
            a_d.stride(0),
            a_d.stride(1),
            stride_bk,
            stride_bn,
            c_d.stride(0),
            c_d.stride(1),
            b_scale_d.stride(0),
            DTYPE_A=mod.DTYPE_TO_TRITON[args.dtype_a],
            DTYPE_B=mod.DTYPE_TO_TRITON[args.dtype_b],
            SCALE_BLOCK=scale_block,
            BLOCK_M=args.BM,
            BLOCK_N=args.BN,
            BLOCK_K=args.BK,
            GROUP_SIZE_M=args.group_size_m,
            TRANSPOSE_B=args.transpose_b,
            NUM_BUFFERS=args.num_buffers,
            SCALE_PRESHUFFLE=args.scale_preshuffled,
            WITH_A_SCALE=args.with_a_scale,
            SCHEDULE=args.schedule,
            TDM_FUSION=args.tdm_fusion,
            L2_PREFETCH_DISTANCE=args.l2_prefetch_distance,
            num_warps=args.num_warps,
            waves_per_eu=max(1, args.num_warps // 4),
        )

    kernel = run()
    torch.cuda.synchronize()

    if not args.no_static_profile:
        _static_profile(kernel)

    if args.benchmark_mode == "graph":
        ms = triton.testing.do_bench_cudagraph(run, rep=args.benchmark_num_iters)
        tflops = 2 * args.M * args.N * args.K / (ms * 1e-3) / 1e12
        print(f"execution time: {ms} ms, {tflops:.2f} TFLOPS")
    elif args.benchmark_mode == "eager":
        ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.benchmark_num_iters)
        tflops = 2 * args.M * args.N * args.K / (ms * 1e-3) / 1e12
        print(f"execution time: {ms} ms, {tflops:.2f} TFLOPS")

    if args.check:
        torch.testing.assert_close(c_d.cpu(), ref, rtol=1e-5, atol=2e-2)
        print("✅Pass")


if __name__ == "__main__":
    main()
