"""Reproduce the public hipBLASLt gfx1250 A8W8 compute kernel with FP32 D.

The assembly uses a 256x256x256 tile, four wave32 waves, a 4x4 CTA cluster,
and four output tiles per CTA. Alpha is one and beta is zero.
"""
import argparse
import ctypes as ct
import ctypes.util
import json
import os
from pathlib import Path
import re
import shlex
import struct
import sys
import time

from build import DEFAULT_CACHE, DEFAULT_LLVM, build, sha256


def configure_model_runtime():
    """Keep the ROCm torch wheel from preloading a different model runtime."""
    active = bool(os.environ.get("HSA_MODEL_LIB")) or os.environ.get("HSA_ENABLE_DTIF") == "1"
    if not active:
        return None
    pack = Path(os.environ["TRITON_GFX1250_MODEL_PATH"]).expanduser()
    import rocm_sdk
    find = rocm_sdk.find_libraries
    initialize = rocm_sdk.initialize_process
    overrides = {
        name: pack / "rocm" / lib
        for name, lib in (
            ("amd_comgr", "libamd_comgr.so.3"),
            ("amdhip64", "libamdhip64.so.7"),
            ("hiprtc", "libhiprtc.so.7"),
        )
    }

    def find_libraries(*names):
        return [path for name in names for path in ([overrides[name]] if name in overrides else find(name))]

    def initialize_process(*, preload_shortnames=None, **kwargs):
        skip = {"rocprofiler-sdk", "rocprofiler-sdk-roctx", "roctracer64", "roctx64"}
        if preload_shortnames is not None:
            preload_shortnames = [name for name in preload_shortnames if name not in skip]
        return initialize(preload_shortnames=preload_shortnames, **kwargs)

    rocm_sdk.find_libraries = find_libraries
    rocm_sdk.initialize_process = initialize_process
    return pack / "rocm/libamdhip64.so.7"


class AttributeValue(ct.Union):
    _fields_ = [("pad", ct.c_char * 64), ("dims", ct.c_uint * 3), ("align", ct.c_uint64)]


class Attribute(ct.Structure):
    _fields_ = [("id", ct.c_uint), ("pad", ct.c_uint), ("val", AttributeValue)]


class LaunchConfig(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ("gx", "gy", "gz", "bx", "by", "bz", "shared")] + [
        ("stream", ct.c_void_p),
        ("attrs", ct.POINTER(Attribute)),
        ("count", ct.c_uint),
    ]


class Hip:

    def __init__(self, library):
        self.lib = ct.CDLL(str(library))
        self.lib.hipGetErrorString.argtypes = [ct.c_int]
        self.lib.hipGetErrorString.restype = ct.c_char_p
        self.allocations = []
        self.module = ct.c_void_p()
        self.malloc = self.api("hipMalloc", [ct.POINTER(ct.c_void_p), ct.c_size_t])
        self.memcpy = self.api("hipMemcpy", [ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int])
        self.sync = self.api("hipDeviceSynchronize", [])

    def api(self, name, types):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = types, ct.c_int

        def call(*args):
            error = fn(*args)
            if error:
                raise RuntimeError(f"{name}: {error}: {self.lib.hipGetErrorString(error).decode()}")

        return call

    def upload(self, tensor):
        ptr = ct.c_void_p()
        size = tensor.numel() * tensor.element_size()
        self.malloc(ct.byref(ptr), size)
        self.allocations.append(ptr)
        self.memcpy(ptr, tensor.data_ptr(), size, 1)
        return ptr.value

    def close(self):
        if self.module.value:
            self.api("hipModuleUnload", [ct.c_void_p])(self.module)
        free = self.api("hipFree", [ct.c_void_p])
        for ptr in self.allocations:
            free(ptr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-M", "--m", type=int, default=8192)
    parser.add_argument("-N", "--n", type=int, default=8192)
    parser.add_argument("-K", "--k", type=int, default=8192)
    parser.add_argument("--output", choices=("staged", "direct"), default="staged",
                        help="FP32 epilogue: LDS-staged asynchronous b128 stores or direct b128 stores")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--llvm", type=Path, default=DEFAULT_LLVM)
    parser.add_argument("--output-dir", type=Path,
                        help="Save source, HSACO, inputs' hashes, and validation/timing JSON")
    parser.add_argument("--benchmark-mode", choices=("eager", "none"), default="eager")
    parser.add_argument("--benchmark-ms", type=int, default=256, help="Eager timing duration in milliseconds")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--input-mode", choices=("signed", "tutorial"), default="signed")
    parser.add_argument("--grid-x", type=int, help="Bound the physical CTA grid for a partial check; default M/512")
    parser.add_argument("--grid-y", type=int, help="Bound the physical CTA grid for a partial check; default N/512")
    args = parser.parse_args()
    M, N, K = args.m, args.n, args.k
    gx = M // 512 if args.grid_x is None else args.grid_x
    gy = N // 512 if args.grid_y is None else args.grid_y
    if M <= 0 or N <= 0 or M % 2048 or N % 2048 or K < 768 or K % 256:
        parser.error("M and N must be positive multiples of 2048; K must be a multiple of 256 and at least 768")
    if not (0 < gx <= M // 512 and 0 < gy <= N // 512 and gx % 4 == gy % 4 == 0):
        parser.error("CTA grid dimensions must be positive multiples of 4, at most M/512 and N/512")
    if args.benchmark_ms <= 0:
        parser.error("--benchmark-ms must be positive")
    binary, build_info = build(args.cache_dir, args.llvm, output=args.output, k=K)
    model_library = configure_model_runtime()
    import torch
    if torch.version.hip is None:
        raise RuntimeError("This benchmark requires a ROCm build of PyTorch")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    arch = torch.cuda.get_device_properties(args.device).gcnArchName.split(":")[0]
    if arch != "gfx1250":
        raise RuntimeError(f"This assembly requires gfx1250; the selected device is {arch}")

    def make_input(rows):
        if args.input_mode == "tutorial":
            return torch.randint(20, 40, (rows, K), dtype=torch.uint8).view(torch.float8_e4m3fn)
        return (torch.randn(rows, K) * 0.5).to(torch.float8_e4m3fn)

    a, b = make_input(M), make_input(N)
    sa = torch.randint(125, 130, (M, K // 32), dtype=torch.uint8)
    sb = torch.randint(125, 130, (N, K // 32), dtype=torch.uint8)

    def pack_scale(scale):
        # The public kernel expects [K/128][M or N][4], unlike TLX pack_scale.
        return scale.reshape(scale.shape[0], K // 128, 4).permute(1, 0, 2).contiguous()

    output = torch.full((N, M), float("nan"), dtype=torch.float32)
    if model_library is None:
        # Torch has loaded the chosen HIP runtime; reuse that exact library.
        loaded = {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/libamdhip64.so" in line
        }
        model_library = next(iter(loaded), None) or ctypes.util.find_library("amdhip64")
        if not model_library:
            raise RuntimeError("Could not locate the HIP runtime loaded by PyTorch")
    hip = Hip(model_library)
    try:
        hip.api("hipInit", [ct.c_uint])(0)
        hip.api("hipSetDevice", [ct.c_int])(args.device)
        da, db, dsa, dsb, dd = map(hip.upload, (a, b, pack_scale(sa), pack_scale(sb), output))
        function = ct.c_void_p()
        hip.api("hipModuleLoad", [ct.POINTER(ct.c_void_p), ct.c_char_p])(ct.byref(hip.module), os.fsencode(binary))
        hip.api("hipModuleGetFunction",
                [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p])(ct.byref(function), hip.module,
                                                                     build_info["name"].encode())
        # Tensile universal ABI v2: 4 header words, M/N/batch/K, six pointers,
        # twelve strides, alpha/beta. The first 27 dwords are preloaded.
        kernargs = struct.pack("<8I6Q12I2f", 1, 1, 1, gx * gy, M, N, 1, K, dd, dd, da, dsa, db, dsb, M, M * N, M, M * N,
                               K, M * K, 4 * M, M * K // 32, K, N * K, 4 * N, N * K // 32, 1., 0.)
        assert len(kernargs) == 136
        buffer = ct.create_string_buffer(kernargs)
        length = ct.c_size_t(len(kernargs))
        extra = (ct.c_void_p * 5)(1, ct.addressof(buffer), 2, ct.addressof(length), 3)
        attrs = (Attribute * 2)()
        attrs[0].id, attrs[0].val.dims[:] = 4, (4, 4, 1)
        attrs[1].id, attrs[1].val.dims[0] = 2, 0
        # Use the same current stream as Triton's eager event-based timer.
        stream = torch.cuda.current_stream(args.device).cuda_stream
        config = LaunchConfig(gx, gy, 1, 128, 1, 1, 0, stream, attrs, 2)
        assert ct.sizeof(Attribute) == 72 and ct.sizeof(LaunchConfig) == 56
        launch_api = hip.api("hipDrvLaunchKernelEx", [ct.POINTER(LaunchConfig), ct.c_void_p, ct.c_void_p, ct.c_void_p])

        def launch():
            launch_api(ct.byref(config), function, None, extra)

        manifest = dict(
            args={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, command=shlex.join([sys.executable, *sys.argv]), build=build_info,
            grid=(gx, gy, 1), block=(128, 1, 1), cluster=(4, 4, 1), output_dtype="float32", output_strides=(1, M),
            alpha=1, beta=0, checked_elements=gx * gy * 4 * 256 * 256, allocated_output_elements=M * N, inputs_sha256={
                name: sha256(t.view(torch.uint8).numpy().tobytes())
                for name, t in (("a", a), ("b", b), ("sa", sa), ("sb", sb))
            }, addresses=dict(a=da, b=db, sa=dsa, sb=dsb, d=dd), environment={
                k: os.environ.get(k)
                for k in ("TRITON_GFX1250_MODEL_PATH", "LLVM_SYSPATH", "HSA_MODEL_LIB", "AM_CLOCK_MT",
                          "DtifGeneralArgs")
            })
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "kernel.hsaco").write_bytes(binary.read_bytes())
            (args.output_dir / "kernel.s").write_bytes(binary.with_suffix(".s").read_bytes())
            (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"hipBLASLt A8W8 -> FP32, tile=256x256x256, waves=4, cluster=4x4, "
            f"programs={gx * gy}, tiles/program=4, output={args.output}, timing={args.benchmark_mode}", flush=True)
        hip.sync()
        start = time.monotonic()
        launch()
        hip.sync()
        manifest["launch_host_seconds"] = time.monotonic() - start
        hip.memcpy(output.data_ptr(), dd, output.numel() * output.element_size(), 2)
        rows = torch.cat((torch.arange(gx * 256), torch.arange(gx * 256) + M // 2))
        cols = torch.cat((torch.arange(gy * 256), torch.arange(gy * 256) + N // 2))
        actual = output.T.index_select(0, rows).index_select(1, cols).contiguous()

        def scaled(x, scale):
            return x.float() * (scale.to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, 1)

        expected = scaled(a[rows], sa[rows]) @ scaled(b[cols], sb[cols]).T
        mismatch = ~torch.isclose(actual, expected, atol=2e-3, rtol=1e-4)
        stray = (~output.isnan()).sum().item() - (~actual.isnan()).sum().item()
        passed = not mismatch.any().item() and stray == 0
        manifest.update(correctness="passed" if passed else "failed", mismatch_count=mismatch.sum().item(),
                        nan_count=actual.isnan().sum().item(), unexpectedly_written_elements=stray,
                        max_abs_error=(actual - expected).abs().max().item(),
                        output_sha256=sha256(actual.numpy().tobytes()),
                        reference_sha256=sha256(expected.numpy().tobytes()), atol=2e-3, rtol=1e-4)
        if passed and args.benchmark_mode == "eager":
            from triton.testing import do_bench
            ms = do_bench(launch, rep=args.benchmark_ms)
            manifest.update(ms=ms, tflops=2 * manifest["checked_elements"] * K / ms / 1e9)
        log = Path("msg.log").read_text() if Path("msg.log").exists() else ""
        starts = re.findall(r"DispatchId \d+:: CP_clk\s*=\s*(\d+)", log)
        ends = re.findall(r"DumpDispatchEndTime.*?clk\s+(\d+)", log)
        manifest.update(dispatch_starts=starts, dispatch_ends=ends)
        if len(starts) == len(ends) == 1:
            manifest["dispatch_cycles"] = int(ends[0]) - int(starts[0])
        if args.output_dir:
            (args.output_dir / "result.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if not passed:
                torch.save(dict(actual=actual, expected=expected), args.output_dir / "failure.pt")
        print(f"M={M} N={N} K={K}: {manifest['correctness']}; "
              f"checked {manifest['checked_elements']:,}/{M * N:,} outputs; "
              f"max_abs_error={manifest['max_abs_error']:.6g}")
        if "ms" in manifest:
            print(f"execution time: {manifest['ms']:.6f} ms, {manifest['tflops']:.2f} TFLOPS")
        if not passed:
            raise AssertionError(f"{manifest['mismatch_count']} mismatches, {stray} unexpected writes")
    finally:
        hip.close()


if __name__ == "__main__":
    main()
