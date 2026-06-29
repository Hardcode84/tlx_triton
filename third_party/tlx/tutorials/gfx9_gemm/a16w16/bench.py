"""Benchmark harness for the gfx9 TLX GEMM tutorial kernels."""

import argparse
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import time

import torch
import triton
from triton import knobs

VERSION_MAP = {
    0: "v0_naive",
    1: "v1_buffer_load",
    2: "v2_async_copy",
    3: "v3_lds",
    4: "v4_global_prefetch",
    5: "v5_local_prefetch",
    6: "v6_loop_unroll",
    7: "v7_slice",
    8: "v8_warp_pipeline",
    9: "v9_beyond_hotloop",
}

PROVIDER_LABELS = {
    "rocblas": "rocBLAS",
    "tlx": "TLX",
    "wave": "Wave",
}

BENCH_DIR = Path(__file__).resolve().parent
TILE_M = 256
TILE_N = 256
TILE_K = 64
TWO_STAGE_K = 2 * TILE_K
TUTORIAL_PROVIDERS = frozenset({"tlx", "wave"})
TWO_STAGE_K_VERSIONS = frozenset(range(5, 10))
UNTILED_K_VERSIONS = frozenset({0, 1})


def get_x_vals():
    return [
        (4096, 4096, 1024),
        (4096, 4096, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
    ]


def parse_shape(text):
    parts = text.replace("x", ",").split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"shape must be MxNxK or M,N,K, got {text!r}")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape must contain integer M/N/K values, got {text!r}") from exc
    if any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape dimensions must be positive: {text!r}")
    return shape


def validate_shape_for_providers(shape, version, providers):
    if not TUTORIAL_PROVIDERS.intersection(providers):
        return
    M, N, K = shape
    if M % TILE_M:
        raise argparse.ArgumentTypeError(f"tutorial kernels require M to be a multiple of {TILE_M}, got {M}")
    if N % TILE_N:
        raise argparse.ArgumentTypeError(f"tutorial kernels require N to be a multiple of {TILE_N}, got {N}")
    if version not in UNTILED_K_VERSIONS and K % TILE_K:
        raise argparse.ArgumentTypeError(f"tutorial kernels v{version} require K to be a multiple of "
                                         f"{TILE_K}, got {K}")
    if version in TWO_STAGE_K_VERSIONS and (K < TWO_STAGE_K or K % TWO_STAGE_K):
        raise argparse.ArgumentTypeError(f"tutorial kernels v{version} prefetch two {TILE_K}-wide K tiles; "
                                         f"K must be at least {TWO_STAGE_K} and a multiple of {TWO_STAGE_K}, "
                                         f"got {K}")


def validate_shapes_for_providers(shapes, version, providers):
    for shape in shapes:
        validate_shape_for_providers(shape, version, providers)


def load_matmul_module(version_dir, suffix):
    kernel_path = BENCH_DIR / version_dir / "matmul_kernel.py"
    spec = importlib.util.spec_from_file_location(
        f"_tlx_gfx9_gemm_{version_dir}_{suffix}_{time.time_ns()}",
        kernel_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark kernel from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_driver(provider):
    if provider == "tlx":
        from triton.backends.amd import driver as amd_driver

        return amd_driver.HIPDriver()
    if provider == "wave":
        from triton.backends.tlx_wave import driver as tlx_wave_driver

        return tlx_wave_driver.TLXWaveDriver()
    return None


@contextmanager
def active_driver(driver):
    if driver is None:
        yield
        return
    previous_driver = triton.runtime.driver.active
    triton.runtime.driver.set_active(driver)
    try:
        yield
    finally:
        triton.runtime.driver.set_active(previous_driver)


def provider_defaults(version):
    if version == 9:
        return ["tlx", "wave"]
    return ["rocblas", "tlx"]


def make_inputs(M, N, K, device, b_layout):
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    if b_layout == "contiguous":
        b = torch.randn((K, N), device=device, dtype=torch.float16)
    else:
        b = torch.randn((N, K), device=device, dtype=torch.float16).T
    return a, b


def provider_matmul(provider, module, a, b):
    if provider == "rocblas":
        return torch.matmul(a, b)
    return module.matmul(a, b)


def benchmark_provider(args, provider, version_dir, a, b, ref, M, N, K):
    module = None
    if provider != "rocblas":
        module = load_matmul_module(version_dir, provider)
    driver = make_driver(provider)
    cache_dir = None
    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir) / version_dir / provider / f"M{M}_N{N}_K{K}"

    with active_driver(driver), knobs.cache.scope(), knobs.runtime.scope():
        if cache_dir is not None:
            knobs.cache.dir = str(cache_dir)
        if args.arch is not None:
            knobs.runtime.override_arch = args.arch
        c = provider_matmul(provider, module, a, b)
        torch.cuda.synchronize()
        ok = torch.allclose(c, ref, atol=args.atol, rtol=args.rtol)
        max_err = (c - ref).abs().max().item()
        if not ok:
            bad = int((~torch.isclose(c, ref, atol=args.atol, rtol=args.rtol)).sum().item())
            return {
                "ok": False,
                "max_err": max_err,
                "bad": bad,
                "ms": None,
                "tflops": None,
            }
        ms = triton.testing.do_bench(
            lambda: provider_matmul(provider, module, a, b),
            warmup=args.warmup,
            rep=args.rep,
        )
    return {
        "ok": True,
        "max_err": max_err,
        "bad": 0,
        "ms": ms,
        "tflops": tflops(ms, M, N, K),
    }


def tflops(ms, M, N, K):
    return 2 * M * N * K * 1e-12 / (ms * 1e-3)


def main():
    parser = argparse.ArgumentParser(description="TLX GEMM benchmark")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--version", type=int, default=0, choices=range(0, 10))
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=tuple(PROVIDER_LABELS),
        default=None,
        help=("providers to benchmark. Defaults to rocblas tlx, except v9 defaults "
              "to tlx wave."),
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help=("custom shape as MxNxK or M,N,K. Can be repeated. TLX/Wave "
              "providers require tutorial tile-compatible shapes."),
    )
    parser.add_argument(
        "--b-layout",
        choices=("transposed", "contiguous"),
        default="transposed",
        help="layout used for B input; transposed matches the tutorial benchmark.",
    )
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--atol", type=float, default=1e-1)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--arch", default=None, help="optional Triton runtime arch override")
    parser.add_argument("--cache-dir", default=None, help="optional Triton cache root")
    parser.add_argument("--wave-opt", default=None, help="optional path to wave-opt")
    args = parser.parse_args()

    if args.wave_opt:
        os.environ["TRITON_WAVE_OPT"] = args.wave_opt

    version_dir = VERSION_MAP[args.version]
    providers = (list(args.providers) if args.providers is not None else provider_defaults(args.version))
    sizes = list(args.shape) if args.shape is not None else get_x_vals()
    if args.K:
        sizes = [(m, n, k) for m, n, k in sizes if k == args.K]
    if not sizes:
        raise SystemExit("no shapes selected")
    try:
        validate_shapes_for_providers(sizes, args.version, providers)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    device = triton.runtime.driver.active.get_active_torch_device()

    print(f"\n{version_dir} ({args.b_layout} B):")
    header = f"{'M':>6s} {'N':>6s} {'K':>6s}"
    for provider in providers:
        label = PROVIDER_LABELS[provider]
        header += f"  {label:>17s}"
    if "tlx" in providers and "wave" in providers:
        header += f"  {'Wave/TLX':>9s}"
    print(header)

    for M, N, K in sizes:
        a, b = make_inputs(M, N, K, device, args.b_layout)
        ref = torch.matmul(a, b)
        torch.cuda.synchronize()

        row = f"{M:6d} {N:6d} {K:6d}"
        results = {}
        for provider in providers:
            try:
                result = benchmark_provider(args, provider, version_dir, a, b, ref, M, N, K)
            except Exception as exc:
                result = {
                    "ok": False,
                    "max_err": None,
                    "bad": None,
                    "ms": None,
                    "tflops": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results[provider] = result
            if result["ok"]:
                row += f"  {result['tflops']:8.1f}T/{result['ms']:6.3f}ms"
            else:
                row += f"  {'FAIL':>17s}"
                if "error" in result:
                    print(f"[{PROVIDER_LABELS[provider]}] M={M} N={N} K={K} failed: "
                          f"{result['error']}")
                else:
                    print(f"[{PROVIDER_LABELS[provider]}] M={M} N={N} K={K} failed "
                          f"correctness: max_err={result['max_err']}, bad={result['bad']}")
        if ("tlx" in results and "wave" in results and results["tlx"]["ok"] and results["wave"]["ok"]):
            ratio = results["wave"]["tflops"] / results["tlx"]["tflops"]
            row += f"  {ratio:8.3f}x"
        print(row, flush=True)


if __name__ == "__main__":
    main()
