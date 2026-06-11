import json
from types import SimpleNamespace

import pytest

import triton
from triton._C.libtriton import ir
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile, make_backend

if "tlx_wave" in backends:
    from triton.backends.tlx_wave import wave_bridge
else:
    wave_bridge = None


pytestmark = pytest.mark.skipif(
    "tlx_wave" not in backends, reason="tlx_wave backend is not installed"
)

GFX950_WAVE = GPUTarget("tlx_wave", "gfx950", 64)


def _asm_text(compiled, artifact):
    text = compiled.asm[artifact]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return text


@triton.jit
def _tlx_wave_local_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    tile = tlx.local_view(buffers, 0)
    values = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tlx.local_store(tile, values)
    out = tlx.local_load(tile)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _tlx_wave_i8_async_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)

    buffers = tlx.local_alloc((BLOCK_SIZE,), tl.int8, 1)
    tile = tlx.local_view(buffers, 0)
    token = tlx.async_load(in_ptr + offs, tile)
    tlx.async_load_commit_group([token])
    tlx.async_load_wait_group(0)

    out = tlx.local_load(tile)
    tl.store(out_ptr + offs, out)


@triton.jit
def _tlx_wave_gemm_cutoff_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_ITERS: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, NUM_BUFFERS)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, NUM_BUFFERS)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile_id in tl.range(0, K_ITERS, loop_unroll_factor=K_ITERS):
        slot = tile_id % NUM_BUFFERS
        k_start = tile_id * BLOCK_K
        a_offsets = offs_m[:, None] * (BLOCK_K * K_ITERS) + k_start + offs_k[None, :]
        b_offsets = (k_start + offs_k[:, None]) * N + offs_n[None, :]

        a_view = tlx.local_view(a_buffers, slot)
        b_view = tlx.local_view(b_buffers, slot)
        tok_a = tlx.async_load(a_ptr + a_offsets, a_view, mask=offs_m[:, None] < M)
        tok_b = tlx.async_load(b_ptr + b_offsets, b_view, mask=offs_n[None, :] < N)
        tlx.async_load_commit_group([tok_a, tok_b])
        tlx.async_load_wait_group(0)

        a_tile = tlx.local_load(a_view)
        b_tile = tlx.local_load(b_view)
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    c_offsets = offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def _minimal_ttgir(public_funcs, target="hip:gfx950", threads_per_warp=64):
    return f"""
module attributes {{tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
{public_funcs}
}}
"""


def _wave_bridge_options(arch="gfx950", warp_size=64):
    return SimpleNamespace(arch=arch, warp_size=warp_size)


def _parse_ttgir(tmp_path, public_funcs, target="hip:gfx950", threads_per_warp=64):
    ctx = ir.context()
    ir.load_dialects(ctx)
    make_backend(GFX950_WAVE).load_dialects(ctx)
    path = tmp_path / "tlx_wave_test.mlir"
    path.write_text(_minimal_ttgir(public_funcs, target, threads_per_warp))
    return ir.parse_mlir_module(str(path), ctx), ctx


def test_tlx_wave_scaffold_emits_wave_skeleton():
    src = ASTSource(
        fn=_tlx_wave_local_kernel,
        signature={"in_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.target.backend == "tlx_wave"
    assert compiled.metadata.arch == "gfx950"
    assert compiled.metadata.tlx_wave_status == "emitted_wave_async_tokens"
    assert compiled.metadata.tlx_wave_num_kernel_args == 3
    assert compiled.metadata.tlx_wave_num_pointer_args == 2
    assert compiled.metadata.tlx_wave_num_scalar_args == 1
    assert compiled.metadata.tlx_wave_plan_kind == "ttgir_graph"
    assert compiled.metadata.tlx_wave_plan_num_ops > 0
    assert compiled.metadata.tlx_wave_plan_num_values > 0
    assert compiled.metadata.shared == 256
    assert compiled.metadata.tlx_wave_lds_size_bytes == 256
    assert compiled.metadata.tlx_wave_num_async_copies == 0
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 0
    assert compiled.metadata.tlx_wave_num_load_store_fallbacks == 0
    assert compiled.metadata.tlx_wave_wave_builder == "wave-dsl"
    assert compiled.metadata.tlx_wave_wave_opt.endswith("wave-opt")
    assert "ttgir" in compiled.asm
    assert "wave" in compiled.asm

    plan = json.loads(compiled.metadata.tlx_wave_plan_json)
    assert plan["kind"] == "ttgir_graph"
    assert plan["op_counts"]["ttg.local_alloc"] == 1
    assert plan["op_counts"]["ttg.local_store"] == 1
    assert plan["op_counts"]["ttg.local_load"] == 1
    assert plan["op_counts"]["tt.store"] == 1

    ttgir_artifact = _asm_text(compiled, "ttgir")
    assert "tt.func" in ttgir_artifact
    assert "ttg.local_alloc" in ttgir_artifact

    wave_artifact = _asm_text(compiled, "wave")
    assert 'waveamdmachine.target = "amdgcn-amd-amdhsa--gfx950"' in wave_artifact
    assert "func.func @_tlx_wave_local_kernel" in wave_artifact
    assert "%arg0: !wave.ptr<#wave.global, f32>" in wave_artifact
    assert "%arg1: !wave.ptr<#wave.global, f32>" in wave_artifact
    assert "%arg2: i32" in wave_artifact
    assert "wave.kernel" in wave_artifact
    assert 'tlx_wave.bridge.stage = "async-copy-tokens"' in wave_artifact
    assert 'tlx_wave.source_target = "hip:gfx950"' in wave_artifact
    assert "tlx_wave.num_warps = 4 : i32" in wave_artifact
    assert "tlx_wave.threads_per_warp = 64 : i32" in wave_artifact
    assert "tlx_wave.lds_size_bytes = 256 : i32" in wave_artifact
    assert "tlx_wave.emitted.async_copies = 0 : i32" in wave_artifact
    assert "wave.lds_size = 256 : i64" in wave_artifact
    assert 'tlx_wave.plan.kind = "ttgir_graph"' in wave_artifact
    assert "return" in wave_artifact
    assert "tt.func public" not in wave_artifact
    assert "ttg.local_alloc" not in wave_artifact


def test_tlx_wave_gemm_cutoff_preserves_async_gemm_shape():
    src = ASTSource(
        fn=_tlx_wave_gemm_cutoff_kernel,
        signature={
            "a_ptr": "*fp16",
            "b_ptr": "*fp16",
            "c_ptr": "*fp32",
            "M": "i32",
            "N": "i32",
        },
        constexprs={
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 32,
            "K_ITERS": 2,
            "NUM_BUFFERS": 2,
        },
    )

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.target == GFX950_WAVE
    assert compiled.metadata.num_ctas == 1
    assert compiled.metadata.warp_size == 64
    assert compiled.metadata.tlx_wave_status == "emitted_wave_async_tokens"
    assert compiled.metadata.tlx_wave_num_kernel_args == 5
    assert compiled.metadata.tlx_wave_num_pointer_args == 3
    assert compiled.metadata.tlx_wave_num_scalar_args == 2
    assert compiled.metadata.tlx_wave_plan_kind == "ttgir_graph"
    assert compiled.metadata.tlx_wave_plan_num_addresses >= 5
    assert compiled.metadata.tlx_wave_plan_num_memdescs >= 6
    assert compiled.metadata.tlx_wave_plan_num_tokens >= 8
    assert compiled.metadata.shared == 8192
    assert compiled.metadata.tlx_wave_lds_size_bytes == 8192
    assert compiled.metadata.tlx_wave_num_async_copies == 4
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 4
    assert compiled.metadata.tlx_wave_num_load_store_fallbacks == 0
    assert compiled.metadata.tlx_wave_num_async_commit_groups == 2
    assert compiled.metadata.tlx_wave_num_async_waits == 2
    assert compiled.metadata.tlx_wave_num_wave_barriers == 2
    assert compiled.metadata.tlx_wave_wave_builder == "wave-dsl"
    assert compiled.metadata.tlx_wave_wave_opt.endswith("wave-opt")
    assert "ttgir" in compiled.asm
    assert "wave" in compiled.asm

    ttgir_artifact = _asm_text(compiled, "ttgir")
    assert ttgir_artifact.count("ttg.local_alloc") >= 2
    assert "ttg.memdesc_index" in ttgir_artifact
    assert ttgir_artifact.count("ttg.async_copy_global_to_local") >= 4
    assert ttgir_artifact.count("ttg.async_commit_group") >= 2
    assert ttgir_artifact.count("ttg.async_wait") >= 2
    assert ttgir_artifact.count("ttg.local_load") >= 4
    assert ttgir_artifact.count("tt.dot") >= 2
    assert "tt.store" in ttgir_artifact
    assert "amdg." not in ttgir_artifact

    wave_artifact = _asm_text(compiled, "wave")
    assert 'waveamdmachine.target = "amdgcn-amd-amdhsa--gfx950"' in wave_artifact
    assert "func.func @_tlx_wave_gemm_cutoff_kernel" in wave_artifact
    assert "%arg0: !wave.ptr<#wave.global, f16>" in wave_artifact
    assert "%arg1: !wave.ptr<#wave.global, f16>" in wave_artifact
    assert "%arg2: !wave.ptr<#wave.global, f32>" in wave_artifact
    assert "%arg3: i32" in wave_artifact
    assert "%arg4: i32" in wave_artifact
    assert "wave.kernel" in wave_artifact
    assert 'tlx_wave.bridge.stage = "async-copy-tokens"' in wave_artifact
    assert "tlx_wave.num_pointer_args = 3 : i32" in wave_artifact
    assert "tlx_wave.num_scalar_args = 2 : i32" in wave_artifact
    assert "tlx_wave.wave_size = 64 : i32" in wave_artifact
    assert "tlx_wave.has_explicit_local_mem_access = true" in wave_artifact
    assert "tlx_wave.lds_size_bytes = 8192 : i32" in wave_artifact
    assert "tlx_wave.emitted.async_copies = 4 : i32" in wave_artifact
    assert "tlx_wave.emitted.dma_load_lds = 4 : i32" in wave_artifact
    assert "tlx_wave.emitted.load_store_fallbacks = 0 : i32" in wave_artifact
    assert "wave.lds_size = 8192 : i64" in wave_artifact
    assert 'tlx_wave.plan.kind = "ttgir_graph"' in wave_artifact
    assert "tlx_wave.plan.num_addresses" in wave_artifact
    assert "tlx_wave.plan.num_memdescs" in wave_artifact
    assert "tlx_wave.plan.num_tokens" in wave_artifact
    assert "return" in wave_artifact
    assert wave_artifact.count("waveamd.dma_load_lds") == 4
    assert wave_artifact.count("wave.join") == 4
    assert wave_artifact.count("wave.wait") == 2
    assert wave_artifact.count("wave.barrier") == 2
    assert "after %" in wave_artifact
    assert "tt.func public" not in wave_artifact
    assert "ttg.local_alloc" not in wave_artifact
    assert "amdg." not in wave_artifact

    plan = json.loads(compiled.metadata.tlx_wave_plan_json)
    assert plan["kind"] == "ttgir_graph"
    assert plan["op_counts"]["ttg.local_alloc"] == 2
    assert plan["op_counts"]["ttg.memdesc_index"] == 4
    assert plan["op_counts"]["ttg.async_copy_global_to_local"] == 4
    assert plan["op_counts"]["ttg.async_commit_group"] == 2
    assert plan["op_counts"]["ttg.async_wait"] == 2
    assert plan["op_counts"]["ttg.local_load"] == 4
    assert plan["op_counts"]["tt.dot"] == 2
    assert plan["op_counts"]["tt.store"] == 1

    address_ops = [address["op"] for address in plan["addresses"]]
    assert address_ops.count("ttg.async_copy_global_to_local") == 4
    assert address_ops.count("ttg.local_load") == 4
    assert address_ops.count("tt.store") == 1
    assert all("role" not in address for address in plan["addresses"])
    assert {
        address["element_type"]
        for address in plan["addresses"]
        if address["op"] == "ttg.async_copy_global_to_local"
    } == {"f16"}

    allocations = [
        memdesc for memdesc in plan["memdescs"] if memdesc["kind"] == "allocation"
    ]
    views = [memdesc for memdesc in plan["memdescs"] if memdesc["kind"] == "view"]
    assert len(allocations) == 2
    assert len(views) == 4
    assert all(memdesc["shape"] == [2, 32, 32] for memdesc in allocations)
    assert all(memdesc["alloc_shape"] == [2, 32, 32] for memdesc in allocations)
    assert all(memdesc["shape"] == [32, 32] for memdesc in views)
    assert all(memdesc["element_type"] == "f16" for memdesc in allocations + views)
    assert all(
        memdesc["memory_space"] == "#ttg.shared_memory"
        for memdesc in allocations + views
    )
    assert sorted(view["static_index"] for view in views) == [0, 0, 1, 1]

    token_ops = [token["op"] for token in plan["tokens"]]
    assert token_ops.count("ttg.async_copy_global_to_local") == 4
    assert token_ops.count("ttg.async_commit_group") == 2
    assert token_ops.count("ttg.async_wait") == 2


def test_tlx_wave_async_copy_fallback_emits_load_store_for_i8():
    src = ASTSource(
        fn=_tlx_wave_i8_async_kernel,
        signature={"in_ptr": "*i8", "out_ptr": "*i8"},
        constexprs={"BLOCK_SIZE": 64},
    )

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.tlx_wave_status == "emitted_wave_async_tokens"
    assert compiled.metadata.shared == 64
    assert compiled.metadata.tlx_wave_lds_size_bytes == 64
    assert compiled.metadata.tlx_wave_num_async_copies == 1
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 0
    assert compiled.metadata.tlx_wave_num_load_store_fallbacks == 1
    assert compiled.metadata.tlx_wave_num_async_commit_groups == 1
    assert compiled.metadata.tlx_wave_num_async_waits == 1
    assert compiled.metadata.tlx_wave_num_wave_barriers == 1

    wave_artifact = _asm_text(compiled, "wave")
    assert 'tlx_wave.bridge.stage = "async-copy-tokens"' in wave_artifact
    assert "tlx_wave.emitted.async_copies = 1 : i32" in wave_artifact
    assert "tlx_wave.emitted.dma_load_lds = 0 : i32" in wave_artifact
    assert "tlx_wave.emitted.load_store_fallbacks = 1 : i32" in wave_artifact
    assert "wave.lds_size = 64 : i64" in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.join" in wave_artifact
    assert "wave.wait" in wave_artifact
    assert "wave.barrier" in wave_artifact
    assert "after %" in wave_artifact


def test_tlx_wave_bridge_reports_unsupported_ttgir_skeleton_inputs(tmp_path):
    one_func = """
  tt.func public @one(%p: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    tt.return
  }
"""
    two_funcs = one_func + """
  tt.func public @two(%p: !tt.ptr<f32>) attributes {noinline = false} {
    tt.return
  }
"""

    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, one_func)
    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())
    assert "func.func @one" in wave
    assert "%arg0: !wave.ptr<#wave.global, f32>" in wave
    assert "%arg1: i32" in wave
    assert metadata["tlx_wave_status"] == "emitted_wave_async_tokens"
    assert metadata["tlx_wave_wave_builder"] == "wave-dsl"
    assert metadata["tlx_wave_wave_opt"].endswith("wave-opt")
    del ctx

    with pytest.raises(ValueError, match="exactly one public tt.func kernel"):
        mod, ctx = _parse_ttgir(tmp_path, two_funcs)
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx

    with pytest.raises(ValueError, match="only supports.*gfx950"):
        mod, ctx = _parse_ttgir(tmp_path, one_func, target="hip:gfx942")
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx

    with pytest.raises(ValueError, match="only supports wave64"):
        mod, ctx = _parse_ttgir(tmp_path, one_func, threads_per_warp=32)
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_bridge_records_non_f16_dot_without_gemm_rejection(tmp_path):
    dot_func = """
  tt.func public @dot_kernel(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %c: !tt.ptr<f32>) attributes {noinline = false} {
    %lhs = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)
    metadata = {}
    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())
    assert "func.func @dot_kernel" in wave
    plan = json.loads(metadata["tlx_wave_plan_json"])
    assert plan["kind"] == "ttgir_graph"
    assert plan["op_counts"]["tt.dot"] == 1
    assert any(
        value["producer"] == "tt.dot" and value["element_type"] == "f32"
        for value in plan["values"]
    )
    del ctx
