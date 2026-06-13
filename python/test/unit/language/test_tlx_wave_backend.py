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


def test_tlx_wave_lowers_i32_load_add_store(tmp_path):
    local_func = """
  tt.func public @i32_add_store(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_ptr = tt.addptr %in_base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %lhs = tt.load %in_ptr : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %rhs = tt.load %in_ptr : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %sum = arith.addi %lhs, %rhs : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_ptr = tt.addptr %out_base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %out_ptr, %sum : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.load") == 2
    assert wave_artifact.count("wave.store") == 1
    assert "wave.binary" in wave_artifact
    assert "wave.binary addi" in wave_artifact
    del ctx


def test_tlx_wave_lowers_simd_compare_masked_store(tmp_path):
    local_func = """
  tt.func public @simd_compare_store(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_ptr = tt.addptr %in_base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %lhs = tt.load %in_ptr : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %rhs = tt.load %in_ptr : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %sum = arith.addi %lhs, %rhs : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %mask = arith.cmpi ugt, %sum, %lhs : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_ptr = tt.addptr %out_base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %out_ptr, %sum, %mask : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.cmpi") >= 1
    assert "wave.where" in wave_artifact
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_rejects_deferred_formula_mixed_with_simd_data(tmp_path):
    local_func = """
  tt.func public @mixed_formula_data(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %sum = arith.addi %loaded, %range : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(ValueError, match="deferred address/index formula"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_lowers_dot_local_load_fragment_without_async_gemm(tmp_path):
    dot_func = """
  tt.func public @dot_local_load_only() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 4096
    assert metadata["tlx_wave_num_async_copies"] == 0
    assert metadata["tlx_wave_num_wave_local_loads"] == 2
    assert metadata["tlx_wave_num_fragment_packs"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 1
    assert metadata["tlx_wave_num_mmas"] == 1
    assert wave_artifact.count("waveamd.fragment_pack") == 2
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 1
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_lowers_multicomponent_blocked_local_roundtrip(tmp_path):
    local_func = """
  tt.func public @multi_component_roundtrip(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<65xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 65 : i32, start = 0 : i32} : tensor<65xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<65x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_ptr = tt.addptr %in_base, %range : tensor<65x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<65xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %in_ptr : tensor<65x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    ttg.local_store %loaded, %alloc : tensor<65xf32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<65xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %alloc : !ttg.memdesc<65xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<65xf32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<65x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_ptr = tt.addptr %out_base, %range : tensor<65x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<65xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %out_ptr, %out : tensor<65x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 260
    assert metadata["tlx_wave_num_wave_local_loads"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] == 1
    assert wave_artifact.count("wave.load") == 4
    assert wave_artifact.count("wave.store") == 4
    assert "wave.where" in wave_artifact
    del ctx


def test_tlx_wave_lowers_simd_convert_layout_component_permutation(tmp_path):
    source_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
    )
    result_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [1, 0]}>"
    )
    convert_func = f"""
  tt.func public @simd_convert_component_permutation(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {{noinline = false}} {{
    %zero = arith.constant dense<0> : tensor<8x8xi32, {source_encoding}>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, {source_encoding}>
    %in_ptr = tt.addptr %in_base, %zero : tensor<8x8x!tt.ptr<f32>, {source_encoding}>, tensor<8x8xi32, {source_encoding}>
    %loaded = tt.load %in_ptr : tensor<8x8x!tt.ptr<f32>, {source_encoding}>
    %converted = ttg.convert_layout %loaded : tensor<8x8xf32, {source_encoding}> -> tensor<8x8xf32, {result_encoding}>
    %out_zero = arith.constant dense<0> : tensor<8x8xi32, {result_encoding}>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, {result_encoding}>
    %out_ptr = tt.addptr %out_base, %out_zero : tensor<8x8x!tt.ptr<f32>, {result_encoding}>, tensor<8x8xi32, {result_encoding}>
    tt.store %out_ptr, %converted : tensor<8x8x!tt.ptr<f32>, {result_encoding}>
    tt.return
  }}
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, convert_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.load") == 4
    assert wave_artifact.count("wave.store") == 4
    del ctx


def test_tlx_wave_rejects_incompatible_simd_convert_layout(tmp_path):
    source_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
    )
    result_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 32], "
        "warpsPerCTA = [2, 1], order = [0, 1]}>"
    )
    convert_func = f"""
  tt.func public @simd_convert_incompatible(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {{noinline = false}} {{
    %zero = arith.constant dense<0> : tensor<8x8xi32, {source_encoding}>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, {source_encoding}>
    %in_ptr = tt.addptr %in_base, %zero : tensor<8x8x!tt.ptr<f32>, {source_encoding}>, tensor<8x8xi32, {source_encoding}>
    %loaded = tt.load %in_ptr : tensor<8x8x!tt.ptr<f32>, {source_encoding}>
    %converted = ttg.convert_layout %loaded : tensor<8x8xf32, {source_encoding}> -> tensor<8x8xf32, {result_encoding}>
    %out_zero = arith.constant dense<0> : tensor<8x8xi32, {result_encoding}>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>, {result_encoding}>
    %out_ptr = tt.addptr %out_base, %out_zero : tensor<8x8x!tt.ptr<f32>, {result_encoding}>, tensor<8x8xi32, {result_encoding}>
    tt.store %out_ptr, %converted : tensor<8x8x!tt.ptr<f32>, {result_encoding}>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, convert_func)

    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "ttg.convert_layout for SIMD tensor data" in message
    assert source_encoding in message
    assert result_encoding in message
    del ctx


def test_tlx_wave_lowers_static_memdesc_index_as_staged_transform(tmp_path):
    local_func = """
  tt.func public @static_view(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %slot = arith.constant 1 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %view = ttg.memdesc_index %alloc[%slot] : !ttg.memdesc<2x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %loaded, %view : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %view : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 512
    assert "wave.lds_size = 512 : i64" in wave_artifact
    assert wave_artifact.count("wave.index_expr") >= 4
    assert wave_artifact.count("wave.ptr_add") >= 4
    del ctx


def test_tlx_wave_lowers_dynamic_memdesc_index_as_ssa_transform(tmp_path):
    local_func = """
  tt.func public @dynamic_view(%arg0: !tt.ptr<f32>, %slot: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %view = ttg.memdesc_index %alloc[%slot] : !ttg.memdesc<2x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %loaded, %view : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %view : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    view = next(memdesc for memdesc in plan.memdescs if memdesc.kind == "view")

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert view.view_op == "ttg.memdesc_index"
    assert view.static_index is None
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 512
    assert "wave.index_cast" in wave_artifact
    assert wave_artifact.count("wave.index_expr") >= 4
    assert wave_artifact.count("wave.ptr_add") >= 4
    del ctx


def test_tlx_wave_records_memdesc_require_layout_constraint(tmp_path):
    local_func = """
  tt.func public @required_layout() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %req = tlx.require_layout %alloc : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %req : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    constraint = plan.layout_constraints[0]
    view = next(memdesc for memdesc in plan.memdescs if memdesc.source == "tlx.require_layout")

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert constraint.op == "tlx.require_layout"
    assert constraint.value_kind == "memdesc"
    assert view.view_op == "tlx.require_layout"
    assert view.base_value_id == constraint.source_value_id
    assert metadata["tlx_wave_plan_num_layout_constraints"] == 1
    assert "tlx_wave.plan.num_layout_constraints" in wave_artifact
    del ctx


def test_tlx_wave_rejects_conflicting_require_layout_constraints(tmp_path):
    local_func = """
  tt.func public @conflicting_layouts() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %req0 = tlx.require_layout %alloc : !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %req1 = tlx.require_layout %alloc : !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(ValueError, match="conflicting tlx\\.require_layout"):
        wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    del ctx


def test_tlx_wave_storage_alias_allocs_share_smem_arena(tmp_path):
    local_func = """
  tt.func public @alias_arena() attributes {noinline = false} {
    %spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %a = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %b = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    lds_layout = wave_bridge_plan._compute_lds_layout(plan)
    alias_allocs = [
        memdesc for memdesc in plan.memdescs if memdesc.alias_spec_value_id is not None
    ]

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert len(alias_allocs) == 2
    assert lds_layout.size_bytes == 256
    assert len({lds_layout.offsets[memdesc.value_id] for memdesc in alias_allocs}) == 1
    assert metadata["shared"] == 256
    assert metadata["tlx_wave_plan_num_storage_aliases"] == 3
    assert "tlx_wave.plan.num_storage_aliases" in wave_artifact
    del ctx


def test_tlx_wave_rejects_unsupported_storage_alias_overlap(tmp_path):
    local_func = """
  tt.func public @alias_distinct_overlap() attributes {noinline = false} {
    %spec = tlx.storage_alias_spec storage = smem : !tlx.storage_alias_spec<smem>
    %a = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %b = tlx.storage_alias_local_alloc %spec : !tlx.storage_alias_spec<smem> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = tlx.reuse_group(%a, %b) group_kind = distinct : (!ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>, !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>) -> !tlx.reuse_group<distinct>
    tlx.set_buffer_overlap(%spec, %group) : (!tlx.storage_alias_spec<smem>, !tlx.reuse_group<distinct>) -> ()
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(ValueError, match="tlx\\.set_buffer_overlap.*distinct"):
        wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    del ctx


def test_tlx_wave_rejects_unsupported_memdesc_subslice_view(tmp_path):
    local_func = """
  tt.func public @subslice_view() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %view = ttg.memdesc_subslice %alloc [0, 0] : !ttg.memdesc<8x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 8x64>
    %out = ttg.local_load %view : !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 8x64> -> tensor<8x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(ValueError, match="unsupported memdesc view ttg\\.memdesc_subslice"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_rejects_generic_2d_shared_local_addressing(tmp_path):
    local_func = """
  tt.func public @generic_shared_2d(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %zero : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %loaded = tt.load %ptr : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %loaded, %alloc : tensor<8x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(
        ValueError,
        match="unsupported shared-memory encoding for generic LDS addressing",
    ):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_rejects_required_generic_shared_encoding(tmp_path):
    local_func = """
  tt.func public @required_unsupported_shared() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %req = tlx.require_layout %alloc : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %req : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(
        ValueError,
        match="unsupported shared-memory encoding for generic LDS addressing",
    ):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
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


def test_tlx_wave_async_copy_fallback_scalarizes_multicomponent_layout(tmp_path):
    async_i8_func = """
  tt.func public @async_i8_multicomponent(%arg0: !tt.ptr<i8>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<65xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 65 : i32, start = 0 : i32} : tensor<65xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<65x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<65x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<65xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<65x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <65xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
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
    assert metadata["shared"] == 65
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert metadata["tlx_wave_num_load_store_fallbacks"] == 1
    assert wave_artifact.count("wave.load") == 2
    assert wave_artifact.count("wave.store") == 2
    assert "waveamd.dma_load_lds" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_fallback_for_unaligned_f16_view(tmp_path):
    async_f16_view_func = """
  tt.func public @async_f16_view(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %slot = arith.constant 1 : i32
    %limit = arith.constant 1 : i32
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x1xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %view = ttg.memdesc_index %alloc[%slot] : !ttg.memdesc<2x1xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<1xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %limit_splat = tt.splat %limit : i32 -> tensor<1xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %mask = arith.cmpi ult, %range, %limit_splat : tensor<1xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %view mask %mask : tensor<1x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <1xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_f16_view_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 16
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert metadata["tlx_wave_num_load_store_fallbacks"] == 1
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "waveamd.dma_load_lds" not in wave_artifact
    del ctx


def test_tlx_wave_rejects_async_copy_2d_shared_destination_without_layout_transform(
    tmp_path,
):
    async_i8_2d_func = """
  tt.func public @async_i8_2d(%arg0: !tt.ptr<i8>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<8x32x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %zero : tensor<8x32x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<8x32x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <8x32xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, async_i8_2d_func)

    with pytest.raises(
        ValueError,
        match="ttg\\.async_copy_global_to_local destination.*"
        "unsupported shared-memory encoding",
    ):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
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


def test_tlx_wave_bridge_lowers_uniform_tensor_compare_mask(tmp_path):
    uniform_mask_func = """
  tt.func public @uniform_mask(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %lhs = tt.splat %arg1 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %rhs = tt.splat %arg2 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %mask = arith.cmpi ult, %lhs, %rhs : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr, %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %loaded, %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, uniform_mask_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_bridge_stage"] == "ttgir-op-lowering"
    assert "wave.splat" in wave
    assert "wave.cmpi" in wave
    del ctx


def test_tlx_wave_bridge_discovers_normal_build_wave_runtime(tmp_path, monkeypatch):
    build_dir = tmp_path / "cmake.test"
    python_package = build_dir / "python_packages" / "wave_mlir"
    tools_dir = build_dir / "bin"
    python_package.mkdir(parents=True)
    tools_dir.mkdir()
    for tool_name in wave_bridge_emit._WAVE_TOOL_NAMES:
        tool = tools_dir / tool_name
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)

    monkeypatch.delenv("TRITON_WAVE_OPT", raising=False)
    monkeypatch.delenv("TRITON_WAVE_PYTHONPATH", raising=False)
    monkeypatch.delenv("TRITON_WAVE_TOOLS_DIR", raising=False)
    monkeypatch.setattr(
        wave_bridge_emit,
        "_cmake_build_dirs",
        lambda: iter((build_dir,)),
    )

    assert list(
        wave_bridge_emit._existing_paths(
            wave_bridge_emit._candidate_wave_python_paths()
        )
    ) == [python_package.resolve()]
    assert wave_bridge_emit._wave_opt() == str((tools_dir / "wave-opt").resolve())
    for tool_name in wave_bridge_emit._WAVE_TOOL_NAMES:
        assert wave_bridge_emit._wave_tool(tool_name) == str(
            (tools_dir / tool_name).resolve()
        )


def test_tlx_wave_bridge_async_inputs_are_ordered_shared_values(tmp_path):
    shared_values_func = """
  tt.func public @shared_async_inputs(%arg0: !tt.ptr<f32>, %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %limit = tt.splat %arg1 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %mask = arith.cmpi ult, %range, %limit : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc mask %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %loaded = tt.load %ptr, %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %loaded, %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, shared_values_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    address_by_op = {address.op: address for address in plan.addresses}

    async_address = address_by_op["ttg.async_copy_global_to_local"]
    load_address = address_by_op["tt.load"]
    store_address = address_by_op["tt.store"]
    assert (
        async_address.address_value_id
        == load_address.address_value_id
        == store_address.address_value_id
    )
    assert (
        async_address.mask_value_id
        == load_address.mask_value_id
        == store_address.mask_value_id
    )

    producer_by_result = {
        result_id: op.index
        for op in plan.ops
        for result_id in op.results
    }
    op_index = {op.name: op.index for op in plan.ops}
    for value_id in (async_address.address_value_id, async_address.mask_value_id):
        producer_index = producer_by_result[value_id]
        assert producer_index < op_index["ttg.async_copy_global_to_local"]
        assert producer_index < op_index["tt.load"]
        assert producer_index < op_index["tt.store"]

    assert not hasattr(wave_bridge_emit, "_ensure_lowered_dependency")
    assert not hasattr(wave_bridge_emit, "_ensure_async_copy_inputs_lowered")
    assert not hasattr(wave_bridge_emit, "_emit_async_tokens")
    del ctx


def test_tlx_wave_bridge_emits_staged_index_pointer_mask_values():
    class FakeExpr:
        def __init__(self, text):
            self.text = text

        def __add__(self, other):
            return FakeExpr(f"({self.text}+{other.text})")

        def __mul__(self, other):
            return FakeExpr(f"({self.text}*{other.text})")

    class FakeMaskType:
        def __init__(self, width):
            self.width = width

    class FakeValue:
        def __init__(self, name, typ=None):
            self.name = name
            self.type = typ

    class FakeW:
        @staticmethod
        def sym(name):
            return FakeExpr(name)

        class MaskType:
            def __init__(self, typ):
                self.width = typ.width

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []
            self.ptr_adds = []
            self.cmpis = []
            self.selects = []

        def index_expr(self, expr, bindings=None):
            value = FakeValue(f"idx{len(self.index_exprs)}")
            self.index_exprs.append((expr, bindings or {}, value))
            return value

        def ptr_add(self, base, offset):
            value = FakeValue(f"ptr{len(self.ptr_adds)}")
            self.ptr_adds.append((base, offset, value))
            return value

        def cmpi(self, predicate, lhs, rhs):
            value = FakeValue(f"mask{len(self.cmpis)}", FakeMaskType(64))
            self.cmpis.append((predicate, lhs, rhs, value))
            return value

        def select(self, condition, true_value, false_value):
            value = FakeValue(f"select{len(self.selects)}", true_value.type)
            self.selects.append((condition, true_value, false_value, value))
            return value

        def lane_id(self, width=64):
            return FakeValue("lane", FakeMaskType(width))

    def plan(value_id):
        return SimpleNamespace(value_id=value_id)

    builder = FakeBuilder()
    w = FakeW()
    values = {value_id: plan(value_id) for value_id in range(1, 9)}
    wave_values = {
        1: wave_bridge_emit._WaveValue("index_expr", FakeValue("a")),
        2: wave_bridge_emit._WaveValue("index_expr", FakeValue("b")),
        6: wave_bridge_emit._WaveValue("pointer_expr", FakeValue("base")),
    }

    wave_bridge_emit._emit_index_binary_op(
        builder,
        SimpleNamespace(name="arith.muli", operands=(1, 2), results=(3,)),
        values,
        wave_values,
        w,
    )
    wave_bridge_emit._emit_index_binary_op(
        builder,
        SimpleNamespace(name="arith.addi", operands=(3, 2), results=(4,)),
        values,
        wave_values,
        w,
    )
    wave_bridge_emit._emit_cmp_op(
        builder,
        SimpleNamespace(
            name="arith.cmpi", operands=(4, 2), results=(5,), attrs={"predicate": 6}
        ),
        values,
        wave_values,
        w,
    )
    wave_bridge_emit._emit_addptr_op(
        builder,
        SimpleNamespace(name="tt.addptr", operands=(6, 4), results=(7,)),
        values,
        wave_values,
        w,
    )
    wave_bridge_emit._emit_mask_and_op(
        builder,
        SimpleNamespace(name="arith.andi", operands=(5, 5), results=(8,)),
        values,
        wave_values,
        w,
    )

    assert len(builder.index_exprs) == 2
    assert builder.index_exprs[0][2] in builder.index_exprs[1][1].values()
    assert len(builder.cmpis) == 2
    assert builder.cmpis[0][0] == "ult"
    assert builder.cmpis[1][0] == "ne"
    assert len(builder.ptr_adds) == 1
    assert builder.ptr_adds[0][1] is builder.index_exprs[1][2]
    assert len(builder.selects) == 1
    for value_id in (3, 4, 5, 7, 8):
        lowered = wave_values[value_id]
        assert not isinstance(lowered.value, wave_bridge_emit._IndexExpr)
        assert not isinstance(lowered.value, wave_bridge_emit._MaskCompare)
        assert not isinstance(lowered.value, wave_bridge_emit._PointerAdd)


def test_tlx_wave_bridge_async_coords_use_blocked_layout_order():
    class FakeExpr:
        def __init__(self, text):
            self.text = str(text)

        def __str__(self):
            return self.text

        def __repr__(self):
            return self.text

        def __add__(self, other):
            return FakeExpr(f"({self}+{other})")

        def __radd__(self, other):
            return FakeExpr(f"({other}+{self})")

        def __mul__(self, other):
            return FakeExpr(f"({self}*{other})")

        def __rmul__(self, other):
            return FakeExpr(f"({other}*{self})")

        def __truediv__(self, other):
            return FakeExpr(f"({self}/{other})")

    class FakeSimdType:
        def __init__(self, width, element_type="index"):
            self.width = width
            self.element_type = element_type

    class FakeValue:
        def __init__(self, name, typ=None):
            self.name = name
            self.type = typ

    class FakeBlockedEncoding:
        def __init__(self, size_per_thread, threads_per_warp, warps_per_cta, order):
            self.size_per_thread = size_per_thread
            self.threads_per_warp = threads_per_warp
            self.warps_per_cta = warps_per_cta
            self.order = order

        def is_blocked_encoding(self):
            return True

        def get_blocked_size_per_thread(self):
            return self.size_per_thread

        def get_blocked_threads_per_warp(self):
            return self.threads_per_warp

        def get_blocked_warps_per_cta(self):
            return self.warps_per_cta

        def get_blocked_order(self):
            return self.order

    class FakeW:
        class SimdType:
            @staticmethod
            def isinstance(typ):
                return isinstance(typ, FakeSimdType)

            def __init__(self, typ):
                self.width = typ.width
                self.element_type = typ.element_type

        class sym_ctx:
            @staticmethod
            def int_(value):
                return FakeExpr(value)

        @staticmethod
        def sym(name):
            return FakeExpr(name)

        @staticmethod
        def mod(lhs, rhs):
            return FakeExpr(f"mod({lhs},{rhs})")

        @staticmethod
        def floor(value):
            return FakeExpr(f"floor({value})")

        @staticmethod
        def index_type():
            return "index"

        @staticmethod
        def i64():
            return "i64"

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []
            self.index_casts = []

        def workitem_id(self, axis=0, width=64):
            return FakeValue(f"thread{axis}", FakeSimdType(width, "i32"))

        def index_expr(self, expr, bindings=None):
            width = 1
            if bindings:
                widths = [
                    binding.type.width
                    for binding in bindings.values()
                    if isinstance(binding.type, FakeSimdType)
                ]
                width = widths[0] if widths else width
            value = FakeValue(f"idx{len(self.index_exprs)}", FakeSimdType(width))
            self.index_exprs.append((expr, bindings or {}, value))
            return value

        def constant(self, typ, value):
            return FakeValue(f"const{value}", typ)

        def splat(self, value, width):
            return FakeValue(f"splat{value.name}", FakeSimdType(width, value.type))

        def index_cast(self, value, typ):
            result = FakeValue(
                f"cast{len(self.index_casts)}", FakeSimdType(value.type.width, typ)
            )
            self.index_casts.append((value, typ, result))
            return result

        def cmpi(self, predicate, lhs, rhs):
            return FakeValue(f"{predicate}{lhs.name}", FakeSimdType(64))

        def select(self, condition, true_value, false_value):
            return FakeValue(f"select{condition.name}", true_value.type)

        def lane_id(self, width=64):
            return FakeValue("lane", FakeSimdType(width, "i32"))

    encoding = (
        "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
    )
    value_plan = SimpleNamespace(
        value_id=42,
        type_kind="tensor",
        type="tensor<8x32x!tt.ptr<f32>>",
        shape=(8, 32),
        encoding=encoding,
        encoding_attr=FakeBlockedEncoding((1, 1), (2, 32), (4, 1), (0, 1)),
    )
    builder = FakeBuilder()

    bindings, width, active = wave_bridge_emit._async_dim_bindings(
        builder, value_plan, FakeW()
    )

    assert width == 64
    assert active is not None
    dim0 = str(builder.index_exprs[0][0])
    dim1 = str(builder.index_exprs[1][0])
    assert "mod(mod(tlx_async_42_thread,64),2)" in dim0
    assert "(2*mod(floor((tlx_async_42_thread/64)),4))" in dim0
    assert "mod(floor((mod(tlx_async_42_thread,64)/2)),32)" in dim1
    assert [cast[1] for cast in builder.index_casts] == ["i64"] * 4
    assert all(cast[0].type.element_type == "index" for cast in builder.index_casts)
    assert {str(symbol) for symbol in bindings} == {"tlx_dim0", "tlx_dim1"}


def test_tlx_wave_bridge_async_coords_include_size_per_thread_component():
    class FakeExpr:
        def __init__(self, text):
            self.text = str(text)

        def __str__(self):
            return self.text

        def __repr__(self):
            return self.text

        def __add__(self, other):
            return FakeExpr(f"({self}+{other})")

        def __radd__(self, other):
            return FakeExpr(f"({other}+{self})")

        def __mul__(self, other):
            return FakeExpr(f"({self}*{other})")

        def __rmul__(self, other):
            return FakeExpr(f"({other}*{self})")

        def __truediv__(self, other):
            return FakeExpr(f"({self}/{other})")

    class FakeSimdType:
        def __init__(self, width, element_type="index"):
            self.width = width
            self.element_type = element_type

    class FakeValue:
        def __init__(self, name, typ=None):
            self.name = name
            self.type = typ

    class FakeBlockedEncoding:
        def is_blocked_encoding(self):
            return True

        def get_blocked_size_per_thread(self):
            return (2, 1)

        def get_blocked_threads_per_warp(self):
            return (2, 32)

        def get_blocked_warps_per_cta(self):
            return (4, 1)

        def get_blocked_order(self):
            return (0, 1)

    class FakeW:
        class SimdType:
            @staticmethod
            def isinstance(typ):
                return isinstance(typ, FakeSimdType)

            def __init__(self, typ):
                self.width = typ.width
                self.element_type = typ.element_type

        class sym_ctx:
            @staticmethod
            def int_(value):
                return FakeExpr(value)

        @staticmethod
        def sym(name):
            return FakeExpr(name)

        @staticmethod
        def mod(lhs, rhs):
            return FakeExpr(f"mod({lhs},{rhs})")

        @staticmethod
        def floor(value):
            return FakeExpr(f"floor({value})")

        @staticmethod
        def index_type():
            return "index"

        @staticmethod
        def i64():
            return "i64"

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []
            self.index_casts = []

        def workitem_id(self, axis=0, width=64):
            return FakeValue(f"thread{axis}", FakeSimdType(width, "i32"))

        def index_expr(self, expr, bindings=None):
            width = 1
            if bindings:
                widths = [
                    binding.type.width
                    for binding in bindings.values()
                    if isinstance(binding.type, FakeSimdType)
                ]
                width = widths[0] if widths else width
            value = FakeValue(f"idx{len(self.index_exprs)}", FakeSimdType(width))
            self.index_exprs.append((expr, bindings or {}, value))
            return value

        def constant(self, typ, value):
            return FakeValue(f"const{value}", typ)

        def splat(self, value, width):
            return FakeValue(f"splat{value.name}", FakeSimdType(width, value.type))

        def index_cast(self, value, typ):
            result = FakeValue(
                f"cast{len(self.index_casts)}", FakeSimdType(value.type.width, typ)
            )
            self.index_casts.append((value, typ, result))
            return result

        def cmpi(self, predicate, lhs, rhs):
            return FakeValue(f"{predicate}{lhs.name}", FakeSimdType(64))

        def select(self, condition, true_value, false_value):
            return FakeValue(f"select{condition.name}", true_value.type)

        def lane_id(self, width=64):
            return FakeValue("lane", FakeSimdType(width, "i32"))

    encoding = (
        "#ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [2, 32], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
    )
    value_plan = SimpleNamespace(
        value_id=43,
        type_kind="tensor",
        type="tensor<16x32x!tt.ptr<f32>>",
        shape=(16, 32),
        encoding=encoding,
        encoding_attr=FakeBlockedEncoding(),
    )
    builder = FakeBuilder()

    bindings, width, active = wave_bridge_emit._async_dim_bindings(
        builder, value_plan, FakeW(), component=1
    )

    assert width == 64
    assert active is not None
    dim0 = str(builder.index_exprs[0][0])
    dim1 = str(builder.index_exprs[1][0])
    assert "mod(1,2)" in dim0
    assert "(2*" in dim0
    assert "mod(floor((1/2)),1)" in dim1
    assert {str(symbol) for symbol in bindings} == {"tlx_dim0", "tlx_dim1"}


def test_tlx_wave_bridge_splats_uniform_tensor_compare_operands():
    class FakeSimdType:
        def __init__(self, width, element_type="index"):
            self.width = width
            self.element_type = element_type

    class FakeMaskType:
        def __init__(self, width):
            self.width = width

    class FakeValue:
        def __init__(self, name, typ="index"):
            self.name = name
            self.type = typ

    class FakeBlockedEncoding:
        def is_blocked_encoding(self):
            return True

        def get_blocked_size_per_thread(self):
            return (1,)

        def get_blocked_threads_per_warp(self):
            return (64,)

        def get_blocked_warps_per_cta(self):
            return (1,)

        def get_blocked_order(self):
            return (0,)

    class FakeW:
        class SimdType:
            @staticmethod
            def isinstance(typ):
                return isinstance(typ, FakeSimdType)

            def __init__(self, typ):
                self.width = typ.width
                self.element_type = typ.element_type

        @staticmethod
        def i64():
            return "i64"

    class FakeBuilder:
        def __init__(self):
            self.splats = []
            self.index_casts = []
            self.cmpis = []

        def splat(self, value, width):
            result = FakeValue(f"splat{len(self.splats)}", FakeSimdType(width, value.type))
            self.splats.append((value, width, result))
            return result

        def index_cast(self, value, typ):
            result = FakeValue(
                f"cast{len(self.index_casts)}", FakeSimdType(value.type.width, typ)
            )
            self.index_casts.append((value, typ, result))
            return result

        def cmpi(self, predicate, lhs, rhs):
            result = FakeValue(f"mask{len(self.cmpis)}", FakeMaskType(lhs.type.width))
            self.cmpis.append((predicate, lhs, rhs, result))
            return result

    def plan(value_id, **kwargs):
        return SimpleNamespace(value_id=value_id, **kwargs)

    builder = FakeBuilder()
    values = {
        1: plan(1),
        2: plan(2),
        3: plan(
            3,
            type_kind="tensor",
            shape=(64,),
            encoding_attr=FakeBlockedEncoding(),
            encoding="#ttg.blocked",
        ),
    }
    wave_values = {
        1: wave_bridge_emit._WaveValue("index_expr", FakeValue("lhs")),
        2: wave_bridge_emit._WaveValue("index_expr", FakeValue("rhs")),
    }

    wave_bridge_emit._emit_cmp_op(
        builder,
        SimpleNamespace(
            name="arith.cmpi", operands=(1, 2), results=(3,), attrs={"predicate": 6}
        ),
        values,
        wave_values,
        FakeW(),
    )

    assert [width for _, width, _ in builder.splats] == [64, 64]
    assert [cast[1] for cast in builder.index_casts] == ["i64", "i64"]
    assert builder.index_casts[0][0] is builder.splats[0][2]
    assert builder.index_casts[1][0] is builder.splats[1][2]
    assert builder.cmpis[0][0] == "ult"
    assert builder.cmpis[0][1] is builder.index_casts[0][2]
    assert builder.cmpis[0][2] is builder.index_casts[1][2]
    assert wave_values[3].kind == "mask_expr"
    assert wave_values[3].value is builder.cmpis[0][3]


def test_tlx_wave_lowers_unrelated_i32_data_math_in_ordered_path():
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

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_wave_local_loads == 1
    wave_artifact = _asm_text(compiled, "wave")
    assert "wave.binary" in wave_artifact
    assert "wave.store" in wave_artifact


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
