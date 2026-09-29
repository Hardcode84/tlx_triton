"""Compile the slide reference kernel and save assembly provenance; no launch."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/assembly"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root / "third_party/tlx/tutorials/amd_grouped_gemm_gfx1250/amd_grouped_gemm_gfx1250_test.py"
    spec = importlib.util.spec_from_file_location("slide_review_grouped", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    constants = dict(group_size=16, K=4096, NUM_PROGRAMS=32, BLOCK_M=256, BLOCK_N=256, BLOCK_K=128, GROUP_M=4,
                     NUM_BUFFERS=2, L2_PREFETCH_DISTANCE=0, C_STAGING_MODE=0, CROSS_TILE_PREFETCH=True,
                     XCD_REMAP_MODE=2, NUM_XCDS=8, XCD_CHUNK=2, CLUSTER_SIZE=4, CLUSTER_MULTICAST=True)
    options = dict(num_warps=4, waves_per_eu=1, ctas_per_cga=(4, 1, 1))
    signature = {
        key: value
        for key, value in module._grouped_gemm_tdm_compile_signature().items()
        if key not in constants
    }
    compiled = triton.compile(
        ASTSource(module.grouped_gemm_tdm_kernel, signature=signature, constexprs=constants,
                  attrs=module._grouped_gemm_tdm_compile_attrs()), target=GPUTarget("hip", "gfx1250", 32),
        options=options)
    asm = compiled.asm["amdgcn"]
    assert compiled.metadata.shared <= 320 * 1024
    assert compiled.metadata.global_scratch_size == 0
    assert compiled.metadata.ctas_per_cga == (4, 1, 1)
    assert asm.count("tensor_store_from_lds") == 8
    assert "v_wmma_f32_16x16x32_f16" in asm

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for kind in ("amdgcn", "ttgir", "llir"):
        data = compiled.asm[kind].encode()
        (args.output / f"reference.{kind}").write_bytes(data)
        hashes[kind] = hashlib.sha256(data).hexdigest()
    record = dict(
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), triton=triton.__file__, target="gfx1250",
        constants=constants, options=options, shared_bytes=compiled.metadata.shared,
        tdm_store_instructions=asm.count("tensor_store_from_lds"), resources={
            key: int(re.search(pattern, asm)[1])
            for key, pattern in {
                "vgprs": r"\.amdhsa_next_free_vgpr (\d+)", "sgprs": r"\.amdhsa_next_free_sgpr (\d+)", "private_segment":
                r"\.amdhsa_private_segment_fixed_size (\d+)"
            }.items()
        }, hashes=hashes)
    (args.output / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
