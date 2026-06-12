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
    from triton.backends.tlx_wave import wave_bridge_emit
    from triton.backends.tlx_wave import wave_bridge_plan
else:
    wave_bridge = None
    wave_bridge_emit = None
    wave_bridge_plan = None


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
def _tlx_wave_async_other_kernel(
    in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    buffers = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    tile = tlx.local_view(buffers, 0)
    token = tlx.async_load(in_ptr + offs, tile, mask=mask, other=0.0)
    tlx.async_load_commit_group([token])
    tlx.async_load_wait_group(0)

    out = tlx.local_load(tile)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _tlx_wave_unrelated_i32_math_kernel(
    i32_in, i32_out, f32_in, f32_out, n_elements, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    ints = tl.load(i32_in + offs, mask=mask, other=0)
    ints = ints + 1
    tl.store(i32_out + offs, ints, mask=mask)

    buffers = tlx.local_alloc((BLOCK_SIZE,), tl.float32, 1)
    tile = tlx.local_view(buffers, 0)
    values = tl.load(f32_in + offs, mask=mask, other=0.0)
    tlx.local_store(tile, values)
    out = tlx.local_load(tile)
    tl.store(f32_out + offs, out, mask=mask)


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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, NUM_BUFFERS)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, NUM_BUFFERS)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile_id in tl.range(0, K_ITERS, loop_unroll_factor=K_ITERS):
        slot = tile_id % NUM_BUFFERS
        k_start = tile_id * BLOCK_K
        a_offsets = (pid_m * BLOCK_M + offs_m[:, None]) * (
            BLOCK_K * K_ITERS
        ) + k_start + offs_k[None, :]
        b_offsets = (k_start + offs_k[:, None]) * N + pid_n * BLOCK_N + offs_n[None, :]

        a_view = tlx.local_view(a_buffers, slot)
        b_view = tlx.local_view(b_buffers, slot)
        tok_a = tlx.async_load(
            a_ptr + a_offsets,
            a_view,
            mask=pid_m * BLOCK_M + offs_m[:, None] < M,
        )
        tok_b = tlx.async_load(
            b_ptr + b_offsets,
            b_view,
            mask=pid_n * BLOCK_N + offs_n[None, :] < N,
        )
        tlx.async_load_commit_group([tok_a, tok_b])
        tlx.async_load_wait_group(0)

        a_tile = tlx.local_load(a_view)
        b_tile = tlx.local_load(b_view)
        acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    c_offsets = (
        (pid_m * BLOCK_M + offs_m[:, None]) * N
        + pid_n * BLOCK_N
        + offs_n[None, :]
    )
    c_mask = (
        (pid_m * BLOCK_M + offs_m[:, None] < M)
        & (pid_n * BLOCK_N + offs_n[None, :] < N)
    )
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


@triton.jit
def _tlx_wave_gemm_token_local_load_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_buffers = tlx.local_alloc((BLOCK_M, BLOCK_K), tl.float16, 1)
    b_buffers = tlx.local_alloc((BLOCK_K, BLOCK_N), tl.float16, 1)
    a_view = tlx.local_view(a_buffers, 0)
    b_view = tlx.local_view(b_buffers, 0)

    a_offsets = offs_m[:, None] * BLOCK_K + offs_k[None, :]
    b_offsets = offs_k[:, None] * N + offs_n[None, :]
    tok_a = tlx.async_load(a_ptr + a_offsets, a_view, mask=offs_m[:, None] < M)
    tok_b = tlx.async_load(b_ptr + b_offsets, b_view, mask=offs_n[None, :] < N)
    group = tlx.async_load_commit_group([tok_a, tok_b])
    wait = tlx.async_load_wait_group(0, [group])

    a_tile = tlx.local_load(a_view, token=wait)
    b_tile = tlx.local_load(b_view, token=wait)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)

    c_offsets = offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + c_offsets, acc, mask=c_mask)


def _minimal_ttgir(
    public_funcs, target="hip:gfx950", threads_per_warp=64, num_ctas=1
):
    return f"""
module attributes {{tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = {num_ctas} : i32, "ttg.num-warps" = 4 : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
{public_funcs}
}}
"""


def _wave_bridge_options(arch="gfx950", warp_size=64):
    return SimpleNamespace(arch=arch, warp_size=warp_size)


def _parse_ttgir(
    tmp_path, public_funcs, target="hip:gfx950", threads_per_warp=64, num_ctas=1
):
    ctx = ir.context()
    ir.load_dialects(ctx)
    make_backend(GFX950_WAVE).load_dialects(ctx)
    path = tmp_path / "tlx_wave_test.mlir"
    path.write_text(_minimal_ttgir(public_funcs, target, threads_per_warp, num_ctas))
    return ir.parse_mlir_module(str(path), ctx), ctx


def test_tlx_wave_lowers_non_dot_local_memory_roundtrip(tmp_path):
    local_func = """
  tt.func public @local_roundtrip(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_ptr = tt.addptr %in_base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %in_ptr : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    ttg.local_store %loaded, %alloc : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %alloc : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_ptr = tt.addptr %out_base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %out_ptr, %out : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 256
    assert metadata["tlx_wave_num_wave_local_loads"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] == 1
    assert "func.func @local_roundtrip" in wave_artifact
    assert wave_artifact.count("wave.load") == 2
    assert wave_artifact.count("wave.store") == 2
    assert "wave.lds_base" in wave_artifact
    assert "#wave.shared" in wave_artifact
    assert "#wave.global" in wave_artifact
    assert "wave.barrier" in wave_artifact
    assert "after %" in wave_artifact
    assert "waveamd.mma" not in wave_artifact
    del ctx


def test_tlx_wave_rejects_generic_tensor_layout_with_partial_coverage(tmp_path):
    partial_layout_func = """
  tt.func public @partial_layout(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %zero : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %loaded = tt.load %ptr : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, partial_layout_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    loaded = next(value for value in plan.values if value.producer == "tt.load")

    with pytest.raises(ValueError, match="full tensor extent.*dim 0.*covers only 8"):
        wave_bridge_emit._blocked_tensor_dim_bindings(
            None, loaded, None, "tt.load result"
        )
    del ctx


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
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_bridge_stage == "ttgir-op-lowering"
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
    assert compiled.metadata.tlx_wave_num_wave_local_loads == 4
    assert compiled.metadata.tlx_wave_num_fragment_packs == 4
    assert compiled.metadata.tlx_wave_num_fragment_fills == 1
    assert compiled.metadata.tlx_wave_num_mmas == 2
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
    assert ttgir_artifact.count("tt.get_program_id") >= 2
    assert "ttg.convert_layout" in ttgir_artifact
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
    assert 'tlx_wave.bridge.stage = "ttgir-op-lowering"' in wave_artifact
    assert "tlx_wave.num_pointer_args = 3 : i32" in wave_artifact
    assert "tlx_wave.num_scalar_args = 2 : i32" in wave_artifact
    assert "tlx_wave.wave_size = 64 : i32" in wave_artifact
    assert "tlx_wave.has_explicit_local_mem_access = true" in wave_artifact
    assert "tlx_wave.lds_size_bytes = 8192 : i32" in wave_artifact
    assert "tlx_wave.emitted.async_copies = 4 : i32" in wave_artifact
    assert "tlx_wave.emitted.dma_load_lds = 4 : i32" in wave_artifact
    assert "tlx_wave.emitted.load_store_fallbacks = 0 : i32" in wave_artifact
    assert "tlx_wave.emitted.local_loads = 4 : i32" in wave_artifact
    assert "tlx_wave.emitted.fragment_packs = 4 : i32" in wave_artifact
    assert "tlx_wave.emitted.fragment_fills = 1 : i32" in wave_artifact
    assert "tlx_wave.emitted.mmas = 2 : i32" in wave_artifact
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
    assert wave_artifact.count("wave.load") == 4
    assert wave_artifact.count("waveamd.fragment_pack") == 4
    assert wave_artifact.count("waveamd.fragment_fill") == 1
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 2
    assert wave_artifact.count("waveamd.fragment_unpack") == 1
    assert wave_artifact.count("wave.extract") == 4
    assert wave_artifact.count("wave.where") == 8
    assert wave_artifact.count("wave.select") >= 4
    assert wave_artifact.count("wave.store") == 4
    assert wave_artifact.count("wave.cmpi") >= 3
    assert wave_artifact.count("wave.index_expr") >= 4
    assert wave_artifact.count("wave.workgroup_id") == 2
    assert wave_artifact.count("wave.workitem_id") >= 8
    assert wave_artifact.count("wave.ptr_add") >= 8
    assert "!waveamd.fragment<0, f16, 16, 16, 64, 4>" in wave_artifact
    assert "!waveamd.fragment<1, f16, 16, 16, 64, 4>" in wave_artifact
    assert "!waveamd.fragment<2, f32, 16, 16, 64, 4>" in wave_artifact
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
    assert plan["op_counts"]["tt.get_program_id"] == 2
    assert plan["op_counts"]["tt.store"] == 1
    assert all("encoding_attr" not in value for value in plan["values"])
    assert any(
        value["producer"] == "ttg.local_load"
        and value["encoding"]
        and "#ttg.dot_op" in value["encoding"]
        for value in plan["values"]
    )

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


def test_tlx_wave_tokenized_local_load_preserves_wait_dependency():
    src = ASTSource(
        fn=_tlx_wave_gemm_token_local_load_kernel,
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
        },
    )

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies == 2
    assert compiled.metadata.tlx_wave_num_async_commit_groups == 1
    assert compiled.metadata.tlx_wave_num_async_waits == 1
    assert compiled.metadata.tlx_wave_num_wave_barriers == 1
    assert compiled.metadata.tlx_wave_num_wave_local_loads == 2
    assert compiled.metadata.tlx_wave_num_mmas == 1

    ttgir_artifact = _asm_text(compiled, "ttgir")
    assert "ttg.local_load" in ttgir_artifact
    assert " token " in ttgir_artifact

    wave_artifact = _asm_text(compiled, "wave")
    load_lines = [
        line.strip() for line in wave_artifact.splitlines() if "wave.load" in line
    ]
    assert len(load_lines) == 2
    assert all(" after %" in line for line in load_lines)
    assert wave_artifact.count("wave.barrier") == 1
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 1


def test_tlx_wave_async_copy_fallback_emits_load_store_for_i8(tmp_path):
    async_i8_func = """
  tt.func public @async_i8(%arg0: !tt.ptr<i8>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_i8_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 64
    assert metadata["tlx_wave_lds_size_bytes"] == 64
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert metadata["tlx_wave_num_load_store_fallbacks"] == 1
    assert metadata["tlx_wave_num_async_commit_groups"] == 1
    assert metadata["tlx_wave_num_async_waits"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] == 1

    assert 'tlx_wave.bridge.stage = "ttgir-op-lowering"' in wave_artifact
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
    del ctx


def test_tlx_wave_bridge_rejects_async_copy_other():
    src = ASTSource(
        fn=_tlx_wave_async_other_kernel,
        signature={"in_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )

    with pytest.raises(ValueError, match="async_copy_global_to_local.*`other`"):
        triton_compile(src, target=GFX950_WAVE)


def test_tlx_wave_bridge_lowers_async_copy_constant_i1_mask(tmp_path):
    async_mask_func = """
  tt.func public @async_mask(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %mask = arith.constant dense<true> : tensor<64xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc mask %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_mask_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_bridge_stage"] == "ttgir-op-lowering"
    assert "wave.cmpi" in wave
    del ctx


def test_tlx_wave_rejects_unlowered_non_dot_data_math_in_ordered_path():
    src = ASTSource(
        fn=_tlx_wave_unrelated_i32_math_kernel,
        signature={
            "i32_in": "*i32",
            "i32_out": "*i32",
            "f32_in": "*fp32",
            "f32_out": "*fp32",
            "n_elements": "i32",
        },
        constexprs={"BLOCK_SIZE": 64},
    )

    with pytest.raises(ValueError, match="arith\\.addi.*lowered as simd"):
        triton_compile(src, target=GFX950_WAVE)


def test_tlx_wave_bridge_uses_async_copy_operand_segments_for_other(tmp_path):
    blocked = (
        "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [1, 0]}>"
    )
    shared = "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>"
    async_other_func = f"""
  tt.func public @async_other(%arg0: !tt.ptr<i1>) attributes {{noinline = false}} {{
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<16x16xi1, {shared}, #ttg.shared_memory, mutable>
    %zero = arith.constant dense<0> : tensor<16x16xi32, {blocked}>
    %base = tt.splat %arg0 : !tt.ptr<i1> -> tensor<16x16x!tt.ptr<i1>, {blocked}>
    %ptr = tt.addptr %base, %zero : tensor<16x16x!tt.ptr<i1>, {blocked}>, tensor<16x16xi32, {blocked}>
    %other = arith.constant dense<false> : tensor<16x16xi1, {blocked}>
    %token = ttg.async_copy_global_to_local %ptr, %alloc other %other : tensor<16x16x!tt.ptr<i1>, {blocked}> -> <16x16xi1, {shared}, #ttg.shared_memory, mutable>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, async_other_func)

    kernel = wave_bridge._kernel_from_module(mod)
    ops = wave_bridge_plan._walk_ops(mod, kernel)
    owners = wave_bridge_plan._result_owner_map(ops)
    values = wave_bridge_plan._build_value_plans(mod, kernel, ops)
    addresses = wave_bridge_plan._build_address_plans(ops, values, owners)
    async_addresses = [
        address
        for address in addresses
        if address.op == "ttg.async_copy_global_to_local"
    ]

    assert len(async_addresses) == 1
    assert async_addresses[0].mask_value_id is None
    assert async_addresses[0].other_value_id is not None
    with pytest.raises(ValueError, match="async_copy_global_to_local.*`other`"):
        wave_bridge_plan._validate_address_feature_support(addresses)
    del ctx


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
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
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


def test_tlx_wave_bridge_rejects_private_helper_funcs_before_emission(tmp_path):
    helper_func = """
  tt.func private @helper() {
    tt.return
  }
  tt.func public @kernel(%p: !tt.ptr<f32>) attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, helper_func)

    with pytest.raises(ValueError, match="private/helper tt\\.func.*helper"):
        wave_bridge._kernel_from_module(mod)
    del ctx


def test_tlx_wave_bridge_rejects_nested_regions_before_emission(tmp_path):
    region_func = """
  tt.func public @region_kernel(%flag: i1) attributes {noinline = false} {
    scf.if %flag {
      %one = arith.constant 1 : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, region_func)

    with pytest.raises(ValueError, match="straight-line TTGIR.*scf\\.if"):
        wave_bridge._build_bridge_plan(
            mod, wave_bridge._kernel_from_module(mod)
        )
    del ctx


def test_tlx_wave_bridge_reaches_dot_validation_after_tt_load(tmp_path):
    dot_func = """
  tt.func public @dot_with_load(%p: !tt.ptr<f32>) attributes {noinline = false} {
    %base = tt.splat %p : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %load = tt.load %base : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %lhs = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    with pytest.raises(ValueError, match="supports only f16 x f16"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_bridge_rejects_unlowered_local_store_value_with_dot(tmp_path):
    dot_func = """
  tt.func public @dot_with_local_store() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    ttg.local_store %value, %alloc : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %lhs = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    with pytest.raises(ValueError, match="ttg\\.local_store.*arith\\.constant"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_bridge_rejects_non_f16_dot_with_encoding_diagnostic(tmp_path):
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
    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "f16 x f16" in message
    assert "#ttg.dot_op" in message
    assert "opIdx = 0" in message
    del ctx


def test_tlx_wave_bridge_rejects_unsupported_local_load_layout(tmp_path):
    dot_func = """
  tt.func public @dot_kernel(%c: !tt.ptr<f32>) attributes {noinline = false} {
    %slot = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %a_view = ttg.memdesc_index %a_alloc[%slot] : !ttg.memdesc<1x32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_view = ttg.memdesc_index %b_alloc[%slot] : !ttg.memdesc<1x32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %lhs = ttg.local_load %a_view : !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_view : !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)
    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "ttg.local_load" in message
    assert "expected f16 dot operand" in message
    assert "#ttg.dot_op" in message
    assert "opIdx = 0" in message
    del ctx


def test_tlx_wave_bridge_rejects_unsupported_shared_local_load_layout(tmp_path):
    dot_func = """
  tt.func public @dot_kernel(%c: !tt.ptr<f32>) attributes {noinline = false} {
    %slot = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %a_view = ttg.memdesc_index %a_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %b_view = ttg.memdesc_index %b_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %lhs = ttg.local_load %a_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)
    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "ttg.local_load" in message
    assert "swizzled_shared" in message
    assert "order" in message
    assert "memdesc encoding" in message
    assert "#ttg.dot_op" in message
    del ctx


def test_tlx_wave_bridge_rejects_dot_operand_layout_conversion(tmp_path):
    dot_func = """
  tt.func public @dot_kernel(%c: !tt.ptr<f32>) attributes {noinline = false} {
    %slot = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %a_view = ttg.memdesc_index %a_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_view = ttg.memdesc_index %b_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %lhs_physical = ttg.local_load %a_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs_physical = ttg.local_load %b_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %lhs = ttg.convert_layout %lhs_physical : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.convert_layout %rhs_physical : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)
    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "tt.dot operand" in message
    assert "unlowered dot operand layout conversion" in message
    assert "physical encoding" in message
    del ctx


def test_tlx_wave_bridge_rejects_unknown_accumulator_constant():
    value = SimpleNamespace(
        value_id=0,
        producer="arith.constant",
        type_kind="tensor",
        element_type="f32",
        shape=wave_bridge._GFX950_MMA_SHAPE,
        const_value=None,
        type="tensor<32x32xf32>",
        encoding="#ttg.blocked<...>",
    )
    stats = SimpleNamespace(fragment_fills=0)
    with pytest.raises(ValueError, match="zero f32 tensor constants"):
        wave_bridge._emit_accumulator_fragment(None, value, {}, None, stats)
    assert stats.fragment_fills == 0


def test_tlx_wave_bridge_rejects_multi_cta_dot_accumulation():
    plan = SimpleNamespace(op_counts={"tt.dot": 1})
    attrs = SimpleNamespace(num_ctas=2)
    with pytest.raises(ValueError, match="multi-CTA accumulation"):
        wave_bridge._validate_compute_support(attrs, plan)


def test_tlx_wave_bridge_rejects_dot_scaled():
    plan = SimpleNamespace(op_counts={"tt.dot_scaled": 1})
    attrs = SimpleNamespace(num_ctas=1)
    with pytest.raises(ValueError, match="tt\\.dot_scaled"):
        wave_bridge._validate_compute_support(attrs, plan)
