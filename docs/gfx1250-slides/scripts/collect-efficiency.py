"""Launch one controlled grouped GEMM variant in an AM/FFM scratch directory.

Source the chosen package environment first. Raw artifacts and correctness
results stay in the current directory; presentation metrics are extracted with
the package's perftools_core after the process finishes.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

VARIANTS = {
    "alias": dict(cross_tile=False, remap="none", cluster_size=1, multicast=False),
    "hybrid": dict(cross_tile=True, remap="none", cluster_size=1, multicast=False),
    "remap": dict(cross_tile=True, remap="chunked", cluster_size=1, multicast=False),
    "cluster": dict(cross_tile=True, remap="chunked", cluster_size=4, multicast=False),
    "multicast": dict(cross_tile=True, remap="chunked", cluster_size=4, multicast=True),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def patch_rocm_sdk(package):
    import rocm_sdk

    find = rocm_sdk.find_libraries
    initialize = rocm_sdk.initialize_process
    overrides = {
        "amd_comgr": package / "rocm/libamd_comgr.so.3", "amdhip64": package / "rocm/libamdhip64.so.7", "hiprtc":
        package / "rocm/libhiprtc.so.7"
    }
    skip = {"rocprofiler-sdk", "rocprofiler-sdk-roctx", "roctracer64", "roctx64"}

    def find_libraries(*names):
        paths = []
        for name in names:
            paths.extend([overrides[name]] if name in overrides and overrides[name].exists() else find(name))
        return paths

    def initialize_process(*, preload_shortnames=None, **kwargs):
        if preload_shortnames is not None:
            preload_shortnames = [name for name in preload_shortnames if name not in skip]
        return initialize(preload_shortnames=preload_shortnames, **kwargs)

    rocm_sdk.find_libraries = find_libraries
    rocm_sdk.initialize_process = initialize_process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--model", choices=("am", "ffm"), required=True)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--programs", type=int, default=32)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if Path.cwd() == root:
        parser.error("Run from a dedicated scratch directory, outside the repository root")
    package = Path(os.environ["TRITON_GFX1250_MODEL_PATH"]).resolve()
    patch_rocm_sdk(package)

    import torch
    import triton
    import triton.language as tl
    from triton._C import libtriton

    source = root / "third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py"
    spec = importlib.util.spec_from_file_location("slide_efficiency_kernel", source)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    cfg = VARIANTS[args.variant]
    g, m, n, k, programs = args.groups, args.m, args.n, args.k, args.programs
    assert min(g, m, n, k, programs) > 0
    assert m % 256 == n % 256 == k % 128 == 0
    assert k // 128 >= 2 and (k // 128) % 2 == 0
    assert programs <= g * (m // 256) * (n // 256)
    if cfg["cluster_size"] > 1:
        mod._validate_grouped_gemm_cluster_config([m] * g, n, programs, block_m=256, block_n=256, block_k=128,
                                                  group_m=4, tdm_pipeline_depth=2, l2_prefetch_distance=0,
                                                  c_staging_mode=0, cross_tile_prefetch=cfg["cross_tile"],
                                                  auto_config=False, xcd_remap_mode=cfg["remap"], num_xcds=8,
                                                  xcd_chunk=2)

    torch.set_num_threads(8)
    torch.manual_seed(123)
    inputs_a = torch.randn((g * m, k), dtype=torch.float16)
    inputs_b = torch.randn((g, n, k), dtype=torch.float16)
    offsets_cpu = torch.arange(g + 1, dtype=torch.int32) * m
    a, b, offsets = inputs_a.cuda(), inputs_b.cuda(), offsets_cpu.cuda()
    c = torch.empty((g * m, n), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    constants = dict(K=k, NUM_PROGRAMS=programs, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4, NUM_BUFFERS=2,
                     L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=0, CROSS_TILE_PREFETCH=cfg["cross_tile"],
                     XCD_REMAP_MODE=mod._XCD_REMAP_MODES[cfg["remap"]], NUM_XCDS=8, XCD_CHUNK=2,
                     CLUSTER_SIZE=cfg["cluster_size"], CLUSTER_MULTICAST=cfg["multicast"])
    options = dict(num_warps=4, waves_per_eu=1, ctas_per_cga=(cfg["cluster_size"], 1, 1))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = dict(
        args=vars(args), variant=cfg, constants=constants, compiler_options=options, revision=revision,
        package=str(package), package_env_sha256=sha256(
            (package / ("am_env.sh" if args.model == "am" else "ffmlite_env.sh")).read_bytes()), triton=triton.__file__,
        native_library=libtriton.__file__, native_library_sha256=sha256(Path(libtriton.__file__).read_bytes()),
        source=str(source), source_sha256=sha256(source.read_bytes()),
        target=str(triton.runtime.driver.active.get_current_target()),
        runtime_multiprocessor_count=torch.cuda.get_device_properties(0).multi_processor_count,
        input_sha256=sha256(inputs_a.numpy().tobytes() + inputs_b.numpy().tobytes()), addresses=dict(
            a=a.data_ptr(), b=b.data_ptr(), c=c.data_ptr(),
            offsets=offsets.data_ptr()), strides=dict(a=a.stride(), b=b.stride(), c=c.stride()), environment={
                name: os.environ.get(name)
                for name in ("AM_CLOCK_MT", "TRITON_AMD_LLVM_FLAGS", "TRITON_DISABLE_LINE_INFO", "TRITON_CACHE_DIR",
                             "LLVM_SYSPATH", "DtifGeneralArgs", "DtifExtraModelArgs")
            })
    Path("manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Launching", args.variant, vars(args), flush=True)
    start = time.monotonic()
    kernel = mod.grouped_gemm_tdm_kernel[(programs, )](a, b, c, offsets, tl.constexpr(g) if cfg["cross_tile"] else g, n,
                                                       a.stride(0), b.stride(0), b.stride(1), c.stride(0), **constants,
                                                       **options)
    torch.cuda.synchronize()
    manifest["launch_host_seconds"] = time.monotonic() - start
    for kind, value in kernel.asm.items():
        path = Path("kernel." + kind)
        path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
    manifest["metadata"] = kernel.metadata._asdict()
    manifest["assembly_sha256"] = sha256(kernel.asm["amdgcn"].encode())
    manifest["hsaco_sha256"] = sha256(kernel.asm["hsaco"])
    manifest["resources"] = {
        name: int(re.search(pattern, kernel.asm["amdgcn"])[1])
        for name, pattern in {
            "vgprs": r"\.amdhsa_next_free_vgpr (\d+)", "sgprs": r"\.amdhsa_next_free_sgpr (\d+)", "private_segment":
            r"\.amdhsa_private_segment_fixed_size (\d+)"
        }.items()
    }
    actual = c.cpu()
    expected = torch.empty_like(actual)
    for group in range(g):
        expected[group * m:(group + 1) *
                 m] = (inputs_a[group * m:(group + 1) * m].float() @ inputs_b[group].float().T).half()
    manifest.update(checked_elements=actual.numel(), output_sha256=sha256(actual.numpy().tobytes()),
                    expected_sha256=sha256(expected.numpy().tobytes()),
                    max_abs_error=(actual.float() - expected.float()).abs().max().item())
    try:
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
        manifest["correctness"] = "passed"
    except AssertionError as error:
        manifest["correctness"] = "failed"
        manifest["correctness_error"] = str(error)
        torch.save(dict(actual=actual, expected=expected), "mismatch.pt")
        raise
    finally:
        Path("result.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("Correctness passed:", actual.numel(), "elements;", manifest["resources"], flush=True)


if __name__ == "__main__":
    main()
