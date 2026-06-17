import subprocess
from types import SimpleNamespace

import pytest

import triton
from triton._C.libtriton import ir, passes
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

GFX942_WAVE = GPUTarget("tlx_wave", "gfx942", 64)
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
def _tlx_wave_async_other_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
        a_offsets = (
            (pid_m * BLOCK_M + offs_m[:, None]) * (BLOCK_K * K_ITERS)
            + k_start
            + offs_k[None, :]
        )
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
        (pid_m * BLOCK_M + offs_m[:, None]) * N + pid_n * BLOCK_N + offs_n[None, :]
    )
    c_mask = (pid_m * BLOCK_M + offs_m[:, None] < M) & (
        pid_n * BLOCK_N + offs_n[None, :] < N
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
    public_funcs,
    target="hip:gfx950",
    threads_per_warp=64,
    num_ctas=1,
    num_warps=4,
    preamble="",
):
    return f"""
{preamble}
module attributes {{tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = {num_ctas} : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
{public_funcs}
}}
"""


def _wave_bridge_options(arch="gfx950", warp_size=64):
    return SimpleNamespace(arch=arch, warp_size=warp_size)


def test_tlx_wave_backend_defaults_and_accepts_mfma_options():
    backend = make_backend(GFX950_WAVE)

    assert backend.parse_options({}).matrix_instr_nonkdim == 0
    assert backend.parse_options({"matrix_instr_nonkdim": 32}).matrix_instr_nonkdim == 32
    with pytest.warns(UserWarning, match="kpack is deprecated"):
        assert backend.parse_options({"kpack": 2}).kpack == 1
    assert make_backend(GFX942_WAVE).parse_options({}).matrix_instr_nonkdim == 0
    assert make_backend(GFX942_WAVE).parse_options({"kpack": 2}).kpack == 2


def _parse_ttgir(
    tmp_path,
    public_funcs,
    target="hip:gfx950",
    threads_per_warp=64,
    num_ctas=1,
    num_warps=4,
    preamble="",
):
    ctx = ir.context()
    ir.load_dialects(ctx)
    make_backend(GFX950_WAVE).load_dialects(ctx)
    path = tmp_path / "tlx_wave_test.mlir"
    path.write_text(
        _minimal_ttgir(
            public_funcs,
            target,
            threads_per_warp,
            num_ctas,
            num_warps,
            preamble,
        )
    )
    return ir.parse_mlir_module(str(path), ctx), ctx


def _run_waveamd_to_machine(wave_artifact):
    result = subprocess.run(
        [wave_bridge_emit._wave_opt(), "-", "--waveamd-to-machine"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_promote_buffer(wave_artifact):
    result = subprocess.run(
        [wave_bridge_emit._wave_opt(), "-", "--wave-promote-global-to-buffer"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_promote_buffer_to_machine(wave_artifact):
    result = subprocess.run(
        [
            wave_bridge_emit._wave_opt(),
            "-",
            "--wave-promote-global-to-buffer",
            "--waveamd-to-machine",
        ],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def test_tlx_wave_bridge_recognizes_structural_ixsimpl_bool_literals():
    import ixsimpl

    ctx = ixsimpl.Context()
    x = ctx.sym("x")
    aligned = ctx.eq(ixsimpl.mod(x, 8), ctx.int_(0))
    not_aligned = ctx.ne(ixsimpl.mod(x, 8), ctx.int_(0))
    w = SimpleNamespace(sym_ctx=ctx)

    simplified_true = [aligned]
    simplified_false = [not_aligned]
    ctx.simplify_batch(simplified_true, assumptions=[aligned])
    ctx.simplify_batch(simplified_false, assumptions=[aligned])

    assert wave_bridge_emit._ixsimpl_is_true(simplified_true[0])
    assert wave_bridge_emit._ixsimpl_is_false(simplified_false[0])
    assert wave_bridge_emit._ixsimpl_proves(aligned, [aligned], w) is True
    assert wave_bridge_emit._ixsimpl_proves(not_aligned, [aligned], w) is False


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


def test_tlx_wave_lowers_scalar_constant_splat_store(tmp_path):
    local_func = """
  tt.func public @constant_splat_store(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = arith.constant 1.250000e+00 : f32
    %tile = tt.splat %value : f32 -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %tile : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.splat" in wave_artifact
    assert wave_artifact.count("wave.store") == 1
    del ctx


def test_tlx_wave_lowers_scalar_argument_splat_store(tmp_path):
    local_func = """
  tt.func public @argument_splat_store(%arg0: !tt.ptr<f32>, %arg1: f32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %tile = tt.splat %arg1 : f32 -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %tile : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "%arg1: f32" in wave_artifact
    assert "wave.splat" in wave_artifact
    assert wave_artifact.count("wave.store") == 1
    del ctx


def test_tlx_wave_lowers_integer_argument_splat_store(tmp_path):
    local_func = """
  tt.func public @integer_argument_splat_store(%arg0: !tt.ptr<i32>, %arg1: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %tile = tt.splat %arg1 : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %tile : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.splat" in wave_artifact
    assert wave_artifact.count("wave.store") == 1
    del ctx


def test_tlx_wave_lowers_uniform_data_broadcast_to_local_store_and_store(tmp_path):
    local_func = """
  tt.func public @broadcast_data_store(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = arith.constant 2.500000e+00 : f32
    %seed = tt.splat %value : f32 -> tensor<1xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %tile = tt.broadcast %seed : tensor<1xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    ttg.local_store %tile, %alloc : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %alloc : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %out : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] == 1
    assert "wave.splat" in wave_artifact
    assert wave_artifact.count("wave.store") == 2
    assert wave_artifact.count("wave.load") == 1
    del ctx


def test_tlx_wave_rejects_nonuniform_data_broadcast_shape_change(tmp_path):
    local_func = """
  tt.func public @broadcast_nonuniform_data(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %range = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<1xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr : tensor<1x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %tile = tt.broadcast %loaded : tensor<1xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(ValueError, match="non-uniform SIMD tensor data"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
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


@pytest.mark.parametrize("k_width", [4, 8, 32])
def test_tlx_wave_lowers_amd_mfma_dot_local_load_k_width_variants(tmp_path, k_width):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @dot_local_load_mfma_kwidth() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma__K_WIDTH_ATTR__}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma__K_WIDTH_ATTR__}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma__K_WIDTH_ATTR__}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma__K_WIDTH_ATTR__}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
""".replace("__K_WIDTH_ATTR__", f", kWidth = {k_width}")
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 2
    assert metadata["tlx_wave_num_fragment_packs"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 1
    assert metadata["tlx_wave_num_mmas"] == 1
    assert wave_artifact.count("waveamd.fragment_pack") == 2
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 1
    del ctx


def test_tlx_wave_lowers_dot_accumulator_chain_and_fragment_store(tmp_path):
    dot_func = """
  tt.func public @dot_chain_store(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot0 = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %dot1 = tt.dot %lhs, %rhs, %dot0 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %zero = arith.constant dense<0> : tensor<32x32xi32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_ptr = tt.addptr %out_base, %zero : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %out_ptr, %dot1 : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 0
    assert metadata["tlx_wave_num_wave_local_loads"] == 2
    assert metadata["tlx_wave_num_fragment_packs"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 1
    assert metadata["tlx_wave_num_mmas"] == 2
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 2
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_lowers_tiled_convert_layout_dot_operand(tmp_path):
    preamble = """
#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @converted_tiled_dot_operand() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %lhs_physical = ttg.local_load %a_alloc : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #src>
    %lhs = ttg.convert_layout %lhs_physical : tensor<64x32xf16, #src> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x32xf32, #mma>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_mmas"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 2
    assert metadata["tlx_wave_num_fragment_packs"] == 3
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 2
    assert wave_artifact.count("waveamd.fragment_pack") == 3
    del ctx


def test_tlx_wave_lowers_tiled_mfma_fragment_local_store(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @tiled_mfma_fragment_local_store() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %out_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<64x32xf16, #shared, #smem, mutable> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x32xf32, #mma>
    ttg.local_store %dot, %out_alloc : tensor<64x32xf32, #mma> -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_mmas"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 2
    assert metadata["tlx_wave_num_wave_barriers"] >= 1
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_honors_converted_blocked_fragment_store_layout(tmp_path, monkeypatch):
    preamble = """
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @converted_blocked_fragment_store(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #store>
    %zero = arith.constant dense<0> : tensor<32x32xi32, #store>
    %out_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #store>
    %out_ptr = tt.addptr %out_base, %zero : tensor<32x32x!tt.ptr<f32>, #store>, tensor<32x32xi32, #store>
    tt.store %out_ptr, %out : tensor<32x32x!tt.ptr<f32>, #store>
    tt.return
  }
"""

    def fail_mfma_tile_store(*args, **kwargs):
        raise AssertionError("converted blocked store used MFMA tile-store fallback")

    monkeypatch.setattr(
        wave_bridge_emit, "_emit_mfma_fragment_tile_store", fail_mfma_tile_store
    )
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_mmas"] == 1
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_vectorizes_mfma32_fragment_store(tmp_path):
    preamble = """
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @mfma32_fragment_store(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #store>
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #store>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #store}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #store}>> -> tensor<32x1xi32, #store>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #store>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #store> -> tensor<32x32xi32, #store>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #store}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #store}>> -> tensor<1x32xi32, #store>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #store> -> tensor<32x32xi32, #store>
    %offsets = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #store>
    %out_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #store>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<32x32x!tt.ptr<f32>, #store>, tensor<32x32xi32, #store>
    tt.store %out_ptr, %out : tensor<32x32x!tt.ptr<f32>, #store>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble, num_warps=4)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_mmas"] == 2
    assert wave_artifact.count("wave.pack") == 4
    assert wave_artifact.count("wave.store") == 4
    assert "(!wave.simd<vector<4xi32>, 64>" in wave_artifact
    assert (
        "(!wave.simd<i32, 64>, !wave.simd<!wave.ptr<#wave.global, f32>"
        not in wave_artifact
    )
    del ctx


def test_tlx_wave_vectorizes_mfma32_fragment_store_with_uniform_mask(tmp_path):
    preamble = """
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @mfma32_fragment_store_mask_uniform(%arg0: !tt.ptr<f32>, %arg1: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #store>
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #store>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #store}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #store}>> -> tensor<32x1xi32, #store>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #store>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #store> -> tensor<32x32xi32, #store>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #store}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #store}>> -> tensor<1x32xi32, #store>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #store> -> tensor<32x32xi32, #store>
    %offsets = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #store>
    %out_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #store>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<32x32x!tt.ptr<f32>, #store>, tensor<32x32xi32, #store>
    %limit = tt.splat %arg1 : i32 -> tensor<1x32xi32, #store>
    %mask_cols = arith.cmpi slt, %cols_2d, %limit : tensor<1x32xi32, #store>
    %mask = tt.broadcast %mask_cols : tensor<1x32xi1, #store> -> tensor<32x32xi1, #store>
    tt.store %out_ptr, %out, %mask : tensor<32x32x!tt.ptr<f32>, #store>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble, num_warps=4)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.pack") == 4
    assert wave_artifact.count("wave.store") == 4
    assert "(!wave.simd<vector<4xi32>, 64>" in wave_artifact
    del ctx


def test_tlx_wave_keeps_mfma32_fragment_store_scalar_for_nonuniform_mask(tmp_path):
    preamble = """
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @mfma32_fragment_store_mask_nonuniform(%arg0: !tt.ptr<f32>, %arg1: i32) attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #store>
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #store>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #store}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #store}>> -> tensor<32x1xi32, #store>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #store>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #store> -> tensor<32x32xi32, #store>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #store}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #store}>> -> tensor<1x32xi32, #store>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #store> -> tensor<32x32xi32, #store>
    %offsets = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #store>
    %out_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #store>
    %out_ptr = tt.addptr %out_base, %offsets : tensor<32x32x!tt.ptr<f32>, #store>, tensor<32x32xi32, #store>
    %limit = tt.splat %arg1 : i32 -> tensor<1x32xi32, #store>
    %mask_cols = arith.cmpi slt, %cols_2d, %limit : tensor<1x32xi32, #store>
    %mask = tt.broadcast %mask_cols : tensor<1x32xi1, #store> -> tensor<32x32xi1, #store>
    tt.store %out_ptr, %out, %mask : tensor<32x32x!tt.ptr<f32>, #store>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble, num_warps=4)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.pack") == 0
    assert wave_artifact.count("wave.store") == 16
    del ctx


def test_tlx_wave_honors_converted_blocked_fragment_local_store_layout(
    tmp_path, monkeypatch
):
    preamble = """
#store = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @converted_blocked_fragment_local_store() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %out_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #store>
    ttg.local_store %out, %out_alloc : tensor<32x32xf32, #store> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
"""

    def fail_mfma_tile_local_store(*args, **kwargs):
        raise AssertionError("converted blocked local store used MFMA fallback")

    monkeypatch.setattr(
        wave_bridge_emit,
        "_emit_mfma_fragment_tile_local_store",
        fail_mfma_tile_local_store,
    )
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_mmas"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] >= 1
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_rejects_unsupported_blocked_fragment_local_store_layout(tmp_path):
    preamble = """
#bad_store = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @unsupported_blocked_fragment_local_store() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %out_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #bad_store>
    ttg.local_store %out, %out_alloc : tensor<32x32xf32, #bad_store> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "fragment store through #ttg.amd_mfma" in message
    assert "unsupported blocked store layout" in message
    assert "sizePerThread=(1, 2)" in message
    del ctx


def test_tlx_wave_rejects_unlowered_blocked_fragment_local_store_conversion(tmp_path):
    dot_func = """
  tt.func public @unlowered_blocked_fragment_local_store() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %out_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    ttg.local_store %out, %out_alloc : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "unlowered layout conversion" in message
    assert "sizePerThread = [2, 2]" in message
    assert "sizePerThread = [1, 2]" in message
    del ctx


def test_tlx_wave_rejects_unlowered_blocked_to_mfma_local_store_conversion(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    dot_func = """
  tt.func public @unlowered_blocked_to_mfma_local_store() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %out_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x32xf32, #mma>
    ttg.local_store %out, %out_alloc : tensor<32x32xf32, #mma> -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "unlowered layout conversion" in message
    assert "sizePerThread = [2, 2]" in message
    assert "#ttg.amd_mfma" in message
    del ctx


def test_tlx_wave_lowers_multicomponent_blocked_local_roundtrip(tmp_path):
    local_func = """
  tt.func public @multi_component_roundtrip(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %in_ptr = tt.addptr %in_base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %in_ptr : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    ttg.local_store %loaded, %alloc : tensor<64xf32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %alloc : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %out_ptr = tt.addptr %out_base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %out_ptr, %out : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
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
    assert wave_artifact.count("wave.load") == 4
    assert wave_artifact.count("wave.store") == 4
    assert "wave.where" in wave_artifact
    del ctx


def test_tlx_wave_lowers_f16_32x32_blocked_local_roundtrip_as_tensor_data(tmp_path):
    local_func = """
  tt.func public @f16_32x32_roundtrip(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %in_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %in_ptr = tt.addptr %in_base, %zero : tensor<32x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %loaded = tt.load %in_ptr : tensor<32x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    ttg.local_store %loaded, %alloc : tensor<32x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_ptr = tt.addptr %out_base, %zero : tensor<32x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %out_ptr, %out : tensor<32x32x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 1
    assert "waveamd.fragment_pack" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
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


def test_tlx_wave_converts_mask_tuple_layout_component_permutation(tmp_path):
    source_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
    )
    result_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [1, 0]}>"
    )
    convert_func = f"""
  tt.func public @mask_convert_component_permutation() attributes {{noinline = false}} {{
    %zero = arith.constant dense<0> : tensor<8x8xi32, {source_encoding}>
    %one = arith.constant dense<1> : tensor<8x8xi32, {source_encoding}>
    %mask = arith.cmpi slt, %zero, %one : tensor<8x8xi32, {source_encoding}>
    %converted = ttg.convert_layout %mask : tensor<8x8xi1, {source_encoding}> -> tensor<8x8xi1, {result_encoding}>
    tt.return
  }}
"""
    mod, ctx = _parse_ttgir(tmp_path, convert_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    source_plan = next(value for value in plan.values if value.producer == "arith.cmpi")
    result_plan = next(
        value for value in plan.values if value.producer == "ttg.convert_layout"
    )

    converted = wave_bridge_emit._convert_mask_layout(
        wave_bridge_emit._WaveValue("mask_tuple", tuple(range(4))),
        source_plan,
        result_plan,
    )

    assert converted.kind == "mask_tuple"
    assert converted.value == (0, 2, 1, 3)
    del ctx


def test_tlx_wave_rejects_incompatible_simd_convert_layout(tmp_path):
    source_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
    )
    result_encoding = (
        "#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 32], "
        "warpsPerCTA = [4, 1], order = [0, 1]}>"
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
    assert "arith.index_cast %arg1" not in wave_artifact
    assert "tlx_memdesc_" in wave_artifact
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
    view = next(
        memdesc for memdesc in plan.memdescs if memdesc.source == "tlx.require_layout"
    )

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


def test_tlx_wave_lowers_tensor_release_layout_as_value_forwarding(tmp_path):
    local_func = """
  tt.func public @released_tensor(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %released = tlx.release_layout %loaded : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %released : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.load") == 1
    assert wave_artifact.count("wave.store") == 1
    assert "tlx.release_layout" not in wave_artifact
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


def test_tlx_wave_lowers_memdesc_subslice_as_staged_transform(tmp_path):
    local_func = """
  tt.func public @subslice_view() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %view = ttg.memdesc_subslice %alloc [0, 32] : !ttg.memdesc<8x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 8x64>
    %out = ttg.local_load %view : !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 8x64> -> tensor<8x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
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

    assert view.view_op == "ttg.memdesc_subslice"
    assert view.view_offsets == (0, 32)
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "ttg.memdesc_subslice" not in wave_artifact
    assert wave_artifact.count("wave.index_expr") >= 4
    assert wave_artifact.count("wave.ptr_add") >= 2
    del ctx


def test_tlx_wave_async_copy_strided_subslice_falls_back_without_dma(tmp_path):
    local_func = """
  tt.func public @async_subslice(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %view = ttg.memdesc_subslice %alloc [0, 32] : !ttg.memdesc<8x64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 8x64>
    %offsets = arith.constant dense<0> : tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %offsets : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %view : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable, 8x64>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    lds_layout = wave_bridge_plan._compute_lds_layout(plan)
    memdescs = wave_bridge_plan._memdescs_by_id(plan)
    address = next(
        address
        for address in plan.addresses
        if address.op == "ttg.async_copy_global_to_local"
    )
    memdesc = memdescs[address.memdesc_value_id]

    assert memdesc.view_op == "ttg.memdesc_subslice"
    assert (
        wave_bridge_emit._dma_packet_bytes(address, memdesc, memdescs, lds_layout)
        is None
    )
    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_lowers_subslice_of_trans_view(tmp_path):
    local_func = """
  tt.func public @trans_subslice_view() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %trans = ttg.memdesc_trans %alloc {order = array<i32: 1, 0>} : !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x8xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    %view = ttg.memdesc_subslice %trans [16, 0] : !ttg.memdesc<32x8xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<16x8xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable, 32x8>
    %out = ttg.local_load %view : !ttg.memdesc<16x8xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable, 32x8> -> tensor<16x8xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "ttg.memdesc_trans" not in wave_artifact
    assert "ttg.memdesc_subslice" not in wave_artifact
    assert "tlx_memdesc_" in wave_artifact
    assert "wave.load" in wave_artifact
    del ctx


@pytest.mark.parametrize(
    ("func_name", "alloc_type", "view_op", "view_type", "tensor_type"),
    [
        (
            "trans_view",
            "!ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>",
            "ttg.memdesc_trans %alloc {order = array<i32: 1, 0>}",
            "!ttg.memdesc<32x8xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>",
            "tensor<32x8xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [1, 0]}>>",
        ),
        (
            "reshape_view",
            "!ttg.memdesc<4x16xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>",
            "ttg.memdesc_reshape %alloc",
            "!ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>",
            "tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>",
        ),
        (
            "reinterpret_view",
            "!ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>",
            "ttg.memdesc_reinterpret %alloc",
            "!ttg.memdesc<256xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>",
            "tensor<256xi8, #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>",
        ),
    ],
)
def test_tlx_wave_lowers_memdesc_transform_views(
    tmp_path, func_name, alloc_type, view_op, view_type, tensor_type
):
    local_func = f"""
  tt.func public @{func_name}() attributes {{noinline = false}} {{
    %alloc = ttg.local_alloc : () -> {alloc_type}
    %view = {view_op} : {alloc_type} -> {view_type}
    %out = ttg.local_load %view : {view_type} -> {tensor_type}
    tt.return
  }}
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    view = next(memdesc for memdesc in plan.memdescs if memdesc.kind == "view")

    view_name = view_op.split()[0]
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert view.view_op == view_name
    if view_name == "ttg.memdesc_trans":
        assert view.view_order == (1, 0)
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert view_name not in wave_artifact
    assert "wave.load" in wave_artifact
    assert wave_artifact.count("wave.index_expr") >= 2
    assert wave_artifact.count("wave.ptr_add") >= 1
    del ctx


def test_tlx_wave_lowers_2d_non_dot_local_memory_roundtrip(tmp_path):
    local_func = """
  tt.func public @local_roundtrip_2d(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %zero : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %loaded = tt.load %ptr : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_store %loaded, %alloc : tensor<8x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %out = ttg.local_load %alloc : !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<8x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_ptr = tt.addptr %out_base, %zero : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %out_ptr, %out : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["shared"] == 1024
    assert metadata["tlx_wave_num_wave_local_loads"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] == 1
    assert wave_artifact.count("wave.load") == 2
    assert wave_artifact.count("wave.store") == 2
    assert "waveamd.mma" not in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_rejects_non_contiguous_2d_shared_local_addressing(tmp_path):
    local_func = """
  tt.func public @generic_shared_2d_bad_order(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %zero : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<8x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %loaded = tt.load %ptr : tensor<8x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
    ttg.local_store %loaded, %alloc : tensor<8x32xf32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<8x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>, #ttg.shared_memory, mutable>
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


def test_tlx_wave_lowers_generic_tensor_layout_with_repeated_components(tmp_path):
    partial_layout_func = """
  tt.func public @partial_layout(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %ptr = tt.addptr %base, %zero : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %loaded = tt.load %ptr : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, partial_layout_func)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    loaded = next(value for value in plan.values if value.producer == "tt.load")

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert (
        wave_bridge_emit._blocked_layout_component_count(
            loaded, "tt.load result", "generic tensor lowering"
        )
        == 4
    )
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave_artifact.count("wave.load") == 4
    del ctx


def test_tlx_wave_gemm_cutoff_lowers_padded_async_copy_as_dma():
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
        attrs={
            (3,): [["tt.divisibility", 16]],
            (4,): [["tt.divisibility", 16]],
        },
    )

    compiled = triton_compile(src, target=GFX950_WAVE)
    wave_artifact = _asm_text(compiled, "wave")

    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies == 4
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 4
    assert wave_artifact.count("waveamd.dma_load_lds") == 4
    assert "ttg.async_copy_global_to_local" not in wave_artifact


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
        attrs={
            (3,): [["tt.divisibility", 16]],
            (4,): [["tt.divisibility", 16]],
        },
    )

    compiled = triton_compile(
        src, target=GFX950_WAVE, options={"matrix_instr_nonkdim": 16}
    )
    ttgir = _asm_text(compiled, "ttgir")
    wave_artifact = _asm_text(compiled, "wave")

    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies == 2
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 2
    assert compiled.metadata.tlx_wave_num_async_commit_groups == 1
    assert compiled.metadata.tlx_wave_num_async_waits == 1
    assert compiled.metadata.tlx_wave_num_wave_barriers == 1
    assert compiled.metadata.tlx_wave_num_wave_local_loads == 2
    assert compiled.metadata.tlx_wave_num_mmas == 1
    assert "#ttg.amd_mfma" in ttgir
    assert "version = 4" in ttgir
    assert "warpsPerCTA = [2, 2]" in ttgir
    assert "instrShape = [16, 16, 32]" in ttgir
    assert "isTransposed = true" in ttgir
    assert ttgir.count("#ttg.padded_shared") == 2
    assert "parent = #mma, kWidth = 8" in ttgir
    assert wave_artifact.count("waveamd.dma_load_lds") == 2
    assert wave_artifact.count("wave.wait") == 1
    assert wave_artifact.count("wave.barrier") == 1
    assert "ttg.local_load" not in wave_artifact


def test_tlx_wave_lowers_gemm_f16_async_tiles_as_dma(tmp_path):
    async_gemm_preamble = """
#blocked_a = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_gemm_tiles = """
  tt.func public @async_gemm_tiles(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) attributes {noinline = false} {
    %slot = arith.constant 1 : i32
    %k_limit = arith.constant 32 : i32
    %smem_a = ttg.local_alloc : () -> !ttg.memdesc<2x256x32xf16, #shared, #smem, mutable>
    %smem_b = ttg.local_alloc : () -> !ttg.memdesc<2x32x256xf16, #shared, #smem, mutable>
    %view_a = ttg.memdesc_index %smem_a[%slot] : !ttg.memdesc<2x256x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x32xf16, #shared, #smem, mutable>
    %view_b = ttg.memdesc_index %smem_b[%slot] : !ttg.memdesc<2x32x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x256xf16, #shared, #smem, mutable>

    %a_row_stride = arith.constant dense<32> : tensor<256x1xi32, #blocked_a>
    %a_rows = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
    %a_rows_2d = tt.expand_dims %a_rows {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>> -> tensor<256x1xi32, #blocked_a>
    %a_row_off = arith.muli %a_rows_2d, %a_row_stride : tensor<256x1xi32, #blocked_a>
    %a_row_offs = tt.broadcast %a_row_off : tensor<256x1xi32, #blocked_a> -> tensor<256x32xi32, #blocked_a>
    %a_cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
    %a_cols_2d = tt.expand_dims %a_cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>> -> tensor<1x32xi32, #blocked_a>
    %a_col_offs = tt.broadcast %a_cols_2d : tensor<1x32xi32, #blocked_a> -> tensor<256x32xi32, #blocked_a>
    %a_offsets = arith.addi %a_row_offs, %a_col_offs : tensor<256x32xi32, #blocked_a>
    %a_base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked_a>
    %a_ptr = tt.addptr %a_base, %a_offsets : tensor<256x32x!tt.ptr<f16>, #blocked_a>, tensor<256x32xi32, #blocked_a>
    %a_k = tt.splat %k_limit : i32 -> tensor<1x32xi32, #blocked_a>
    %a_mask_cols = arith.cmpi slt, %a_cols_2d, %a_k : tensor<1x32xi32, #blocked_a>
    %a_mask = tt.broadcast %a_mask_cols : tensor<1x32xi1, #blocked_a> -> tensor<256x32xi1, #blocked_a>

    %b_row_stride = arith.constant dense<256> : tensor<32x1xi32, #blocked_b>
    %b_rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_b}>>
    %b_rows_2d = tt.expand_dims %b_rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_b}>> -> tensor<32x1xi32, #blocked_b>
    %b_row_off = arith.muli %b_rows_2d, %b_row_stride : tensor<32x1xi32, #blocked_b>
    %b_row_offs = tt.broadcast %b_row_off : tensor<32x1xi32, #blocked_b> -> tensor<32x256xi32, #blocked_b>
    %b_cols = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>>
    %b_cols_2d = tt.expand_dims %b_cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>> -> tensor<1x256xi32, #blocked_b>
    %b_col_offs = tt.broadcast %b_cols_2d : tensor<1x256xi32, #blocked_b> -> tensor<32x256xi32, #blocked_b>
    %b_offsets = arith.addi %b_row_offs, %b_col_offs : tensor<32x256xi32, #blocked_b>
    %b_base = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x256x!tt.ptr<f16>, #blocked_b>
    %b_ptr = tt.addptr %b_base, %b_offsets : tensor<32x256x!tt.ptr<f16>, #blocked_b>, tensor<32x256xi32, #blocked_b>
    %b_k = tt.splat %k_limit : i32 -> tensor<32x1xi32, #blocked_b>
    %b_mask_rows = arith.cmpi slt, %b_rows_2d, %b_k : tensor<32x1xi32, #blocked_b>
    %b_mask = tt.broadcast %b_mask_rows : tensor<32x1xi1, #blocked_b> -> tensor<32x256xi1, #blocked_b>

    %tok_a = ttg.async_copy_global_to_local %a_ptr, %view_a mask %a_mask : tensor<256x32x!tt.ptr<f16>, #blocked_a> -> <256x32xf16, #shared, #smem, mutable>
    %tok_b = ttg.async_copy_global_to_local %b_ptr, %view_b mask %b_mask : tensor<32x256x!tt.ptr<f16>, #blocked_b> -> <32x256xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %tok_a, %tok_b
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(
        tmp_path,
        async_gemm_tiles,
        num_warps=8,
        preamble=async_gemm_preamble,
    )
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    values_by_id = {value.value_id: value for value in plan.values}
    address_plans = [
        values_by_id[address.address_value_id]
        for address in plan.addresses
        if address.op == "ttg.async_copy_global_to_local"
    ]

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert [
        wave_bridge_emit._blocked_layout_component_count(
            value, "ttg.async_copy_global_to_local source", "test"
        )
        for value in address_plans
    ] == [16, 16]
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 2
    assert metadata["tlx_wave_num_dma_load_lds"] == 4
    assert wave_artifact.count("waveamd.dma_load_lds") == 4
    assert wave_artifact.count("wave.where") == 4
    assert "ttg.memdesc_index" not in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_lowers_contiguous_f16_as_16_byte_dma(tmp_path):
    async_func = """
  tt.func public @async_f16_wide_dma(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<512x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>, tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<512x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>> -> <512xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_waveamd_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "bytes = 16" in wave_artifact
    assert "waveamdmachine.global_load_lds_b128" in machine
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_buffer_load_to_local_lowers_contiguous_f16_as_dma(tmp_path):
    buffer_func = """
  tt.func public @buffer_f16_wide_dma(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>] -> <512xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, buffer_func, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_wave_promote_buffer_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "bytes = 16" in wave_artifact
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    assert "amdg.buffer_load_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_buffer_load_to_local_mask_other_uses_fallback(tmp_path):
    buffer_func = """
  tt.func public @buffer_mask_other(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %mask = arith.constant dense<true> : tensor<64xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask other = %other into %alloc : <f32>[tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>] tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, buffer_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "amdg.buffer_load_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_buffer_load_to_local_padded_chunk_crossing_pad_falls_back(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[64:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    buffer_func = """
  tt.func public @buffer_padded_crosses_pad(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #blocked>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %offs = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = amdg.buffer_load_to_local %arg0[%offs] into %alloc : <f16>[tensor<32x32xi32, #blocked>] -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, buffer_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_buffer_load_to_local_lowers_swizzled_f16_as_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    buffer_func = """
  tt.func public @buffer_swizzled_dma(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #blocked>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %offs = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = amdg.buffer_load_to_local %arg0[%offs] into %alloc : <f16>[tensor<32x32xi32, #blocked>] -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, buffer_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "wave.load" not in wave_artifact
    del ctx


def test_tlx_wave_buffer_load_to_local_rejects_stride_operand(tmp_path):
    buffer_func = """
  tt.func public @buffer_stride(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %stride = arith.constant 64 : i32
    %token = amdg.buffer_load_to_local %arg0[%range] stride = %stride into %alloc : <f16>[tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>] -> <64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, buffer_func)

    with pytest.raises(ValueError, match="amdg\\.buffer_load_to_local with a stride"):
        wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    del ctx


def test_tlx_wave_buffer_load_to_local_rejects_cache_modifier(tmp_path):
    buffer_func = """
  tt.func public @buffer_cache_modifier(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = amdg.buffer_load_to_local %arg0[%range] cacheModifier = cv into %alloc : <f16>[tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>] -> <64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, buffer_func)

    with pytest.raises(ValueError, match="amdg\\.buffer_load_to_local.*cacheModifier"):
        wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    del ctx


def test_tlx_wave_async_copy_padded_chunk_crossing_pad_falls_back(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[64:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_padded_crosses_pad(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #blocked>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %offs = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_async_copy_lowers_swizzled_f16_as_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_swizzled_dma(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %row_stride = arith.constant dense<32> : tensor<32x1xi32, #blocked>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %row_off = arith.muli %rows_2d, %row_stride : tensor<32x1xi32, #blocked>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %offs = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    assert "wave.load" not in wave_artifact
    del ctx


def test_tlx_wave_linear_encoding_attribute_helpers(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [16, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
"""
    linear_func = """
  tt.func public @linear_helpers() attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<32x32xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, linear_func, preamble=preamble)
    plan = wave_bridge._build_bridge_plan(mod, wave_bridge._kernel_from_module(mod))
    value = next(
        value
        for value in plan.values
        if value.producer == "arith.constant" and value.shape == (32, 32)
    )
    attr = value.encoding_attr

    assert attr.is_linear_encoding()
    assert attr.get_linear_register_bases() == [[0, 1], [16, 0]]
    assert attr.get_linear_lane_bases() == [
        [0, 2],
        [0, 4],
        [0, 8],
        [0, 16],
        [1, 0],
        [2, 0],
    ]
    assert attr.get_linear_warp_bases() == [[4, 0], [8, 0]]
    assert attr.get_linear_block_bases() == []
    assert set(attr.get_linear_in_dim_names()) == {
        "register",
        "lane",
        "warp",
        "block",
    }
    assert len(attr.get_linear_out_dim_names()) == 2
    assert attr.get_linear_num_in_dims() == 4
    assert attr.get_linear_num_out_dims() == 2
    del ctx


def test_tlx_wave_async_copy_linear_source_falls_back_without_dma(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [[0, 1], [16, 0]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = []}>
#shared = #ttg.padded_shared<[64:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [2, 0], [4, 0], [8, 0], [16, 0], [1, 0]], block = []}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_linear_fallback(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %zero = arith.constant dense<0> : tensor<32x32xi32, #linear>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #linear>
    %ptr = tt.addptr %base, %zero : tensor<32x32x!tt.ptr<f16>, #linear>, tensor<32x32xi32, #linear>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<32x32x!tt.ptr<f16>, #linear> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_generic_linear_tensor_ops_and_local_roundtrip(tmp_path):
    preamble = """
#generic = #ttg.generic_linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    generic_func = """
  tt.func public @generic_linear_ops(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #generic>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #generic>
    %ptr = tt.addptr %base, %offs : tensor<128x!tt.ptr<i32>, #generic>, tensor<128xi32, #generic>
    %loaded = tt.load %ptr : tensor<128x!tt.ptr<i32>, #generic>
    %one = arith.constant dense<1> : tensor<128xi32, #generic>
    %sum = arith.addi %loaded, %one : tensor<128xi32, #generic>
    %limit = arith.constant dense<64> : tensor<128xi32, #generic>
    %mask = arith.cmpi slt, %sum, %limit : tensor<128xi32, #generic>
    %bound = arith.constant dense<96> : tensor<128xi32, #generic>
    %bounds_mask = arith.cmpi slt, %offs, %bound : tensor<128xi32, #generic>
    %combined_mask = arith.andi %mask, %bounds_mask : tensor<128xi1, #generic>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    ttg.local_store %sum, %alloc : tensor<128xi32, #generic> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %roundtrip = ttg.local_load %alloc : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #generic>
    %out_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #generic>
    %out_ptr = tt.addptr %out_base, %offs : tensor<128x!tt.ptr<i32>, #generic>, tensor<128xi32, #generic>
    tt.store %out_ptr, %roundtrip, %combined_mask : tensor<128x!tt.ptr<i32>, #generic>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, generic_func, preamble=preamble, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "ttg.generic_linear" not in wave_artifact
    assert wave_artifact.count("wave.load") >= 2
    assert wave_artifact.count("wave.store") >= 4
    del ctx


def test_tlx_wave_generic_linear_to_blocked_same_lane_convert(tmp_path):
    preamble = """
#generic = #ttg.generic_linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    convert_func = """
  tt.func public @generic_linear_same_lane_convert(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #generic>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #generic>
    %ptr = tt.addptr %base, %offs : tensor<128x!tt.ptr<i32>, #generic>, tensor<128xi32, #generic>
    %loaded = tt.load %ptr : tensor<128x!tt.ptr<i32>, #generic>
    %converted = ttg.convert_layout %loaded : tensor<128xi32, #generic> -> tensor<128xi32, #blocked>
    %limit = arith.constant dense<96> : tensor<128xi32, #generic>
    %mask = arith.cmpi slt, %loaded, %limit : tensor<128xi32, #generic>
    %out_mask = ttg.convert_layout %mask : tensor<128xi1, #generic> -> tensor<128xi1, #blocked>
    %out_offs = ttg.convert_layout %offs : tensor<128xi32, #generic> -> tensor<128xi32, #blocked>
    %out_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %out_ptr = tt.addptr %out_base, %out_offs : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %out_ptr, %converted, %out_mask : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, convert_func, preamble=preamble, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "ttg.convert_layout" not in wave_artifact
    del ctx


def test_tlx_wave_rejects_cross_lane_generic_linear_mask_convert(tmp_path):
    preamble = """
#generic = #ttg.generic_linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    convert_func = """
  tt.func public @generic_linear_cross_lane_mask_convert(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #generic>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #generic>
    %ptr = tt.addptr %base, %offs : tensor<128x!tt.ptr<i32>, #generic>, tensor<128xi32, #generic>
    %loaded = tt.load %ptr : tensor<128x!tt.ptr<i32>, #generic>
    %limit = arith.constant dense<96> : tensor<128xi32, #generic>
    %mask = arith.cmpi slt, %loaded, %limit : tensor<128xi32, #generic>
    %out_mask = ttg.convert_layout %mask : tensor<128xi1, #generic> -> tensor<128xi1, #blocked>
    %zero = arith.constant dense<0> : tensor<128xi32, #blocked>
    %value = arith.constant dense<1> : tensor<128xi32, #blocked>
    %out_base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %out_ptr = tt.addptr %out_base, %zero : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %out_ptr, %value, %out_mask : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, convert_func, preamble=preamble, num_warps=1)

    with pytest.raises(ValueError, match="cross-lane remap"):
        wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())
    del ctx


def test_tlx_wave_rejects_cross_lane_generic_linear_convert(tmp_path):
    preamble = """
#generic = #ttg.generic_linear<{register = [[1]], lane = [[2], [4], [8], [16], [32], [64]], warp = [], block = []}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    convert_func = """
  tt.func public @generic_linear_cross_lane_convert(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
    %offs = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #generic>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #generic>
    %ptr = tt.addptr %base, %offs : tensor<128x!tt.ptr<i32>, #generic>, tensor<128xi32, #generic>
    %loaded = tt.load %ptr : tensor<128x!tt.ptr<i32>, #generic>
    %converted = ttg.convert_layout %loaded : tensor<128xi32, #generic> -> tensor<128xi32, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, convert_func, preamble=preamble, num_warps=1)

    with pytest.raises(ValueError, match="cross-lane remap"):
        wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())
    del ctx


def test_tlx_wave_async_copy_pointer_range_bounds_runtime_stride_dma_source(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_pointer_range_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}, %stride: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %c0 : i32
    llvm.intr.assume %stride_nonnegative : i1
    %stride_splat = tt.splat %stride : i32 -> tensor<32x1xi32, #blocked>
    %rows = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %row_off = arith.muli %rows_2d, %stride_splat : tensor<32x1xi32, #blocked>
    %row_offs = tt.broadcast %row_off : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %col_offs = tt.broadcast %cols_2d : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %offs = arith.addi %row_offs, %col_offs : tensor<32x32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_wave_promote_buffer_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert "wave.assume" in wave_artifact
    assert "1073741815" in wave_artifact
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_store_pointer_range_promotes_runtime_stride_buffer_store(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    store_func = """
  tt.func public @store_pointer_range(%arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %stride: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %c0 : i32
    llvm.intr.assume %stride_nonnegative : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<64xi32, #blocked>
    %offs = arith.addi %stride_splat, %range : tensor<64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %value = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    tt.store %ptr, %value : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, store_func, preamble=preamble, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_wave_promote_buffer_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "536870910" in wave_artifact
    assert "waveamdmachine.buffer_store_b32" in machine
    assert "waveamdmachine.global_store" not in machine
    del ctx


def test_tlx_wave_load_pointer_range_promotes_runtime_stride_buffer_load(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    load_func = """
  tt.func public @load_pointer_range(%arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %stride: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %c0 : i32
    llvm.intr.assume %stride_nonnegative : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<64xi32, #blocked>
    %offs = arith.addi %stride_splat, %range : tensor<64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, load_func, preamble=preamble, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_wave_promote_buffer_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "536870910" in wave_artifact
    assert "waveamdmachine.buffer_load_b32" in machine
    assert "waveamdmachine.global_load_b32" not in machine
    del ctx


def test_tlx_wave_store_without_pointer_range_does_not_promote_buffer_store(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    store_func = """
  tt.func public @store_no_pointer_range(%arg0: !tt.ptr<f32>, %stride: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %c0 : i32
    llvm.intr.assume %stride_nonnegative : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<64xi32, #blocked>
    %offs = arith.addi %stride_splat, %range : tensor<64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %value = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    tt.store %ptr, %value : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, store_func, preamble=preamble, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    promoted = _run_wave_promote_buffer(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "waveamd.make_buffer" not in promoted
    assert "wave.store" in promoted
    del ctx


def test_tlx_wave_store_pointer_range_partial_block_does_not_assume_inactive_lanes(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    store_func = """
  tt.func public @store_pointer_range_partial(%arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}, %stride: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %stride_nonnegative = arith.cmpi sge, %stride, %c0 : i32
    llvm.intr.assume %stride_nonnegative : i1
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<32xi32, #blocked>
    %offs = arith.addi %stride_splat, %range : tensor<32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %offs : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %value = arith.constant dense<1.000000e+00> : tensor<32xf32, #blocked>
    tt.store %ptr, %value : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, store_func, preamble=preamble, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    promoted = _run_wave_promote_buffer(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "536870910" not in wave_artifact
    assert "waveamd.make_buffer" not in promoted
    assert "wave.store" in promoted
    del ctx


def test_tlx_wave_async_copy_pointer_range_does_not_assume_negative_offset_nonnegative(
    tmp_path,
):
    async_func = """
  tt.func public @async_pointer_range_negative_offset(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %halo = arith.constant dense<16> : tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %offsets = arith.subi %range, %halo : tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %ptr = tt.addptr %base, %offsets : tensor<512x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>, tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<512x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>> -> <512xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    promoted = _run_wave_promote_buffer(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    bounded_offset_assumes = [
        line
        for line in wave_artifact.splitlines()
        if "1073741815" in line
    ]
    assert bounded_offset_assumes
    assert all('"x >= 0"' not in line for line in bounded_offset_assumes)
    assert "waveamd.make_buffer" not in promoted
    assert "waveamd.dma_load_lds" in promoted
    del ctx


def test_tlx_wave_async_copy_lowers_contiguous_bf16_as_16_byte_dma(tmp_path):
    async_func = """
  tt.func public @async_bf16_wide_dma(%arg0: !tt.ptr<bf16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xbf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<512x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>, tensor<512xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<512x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>> -> <512xbf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_waveamd_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "bytes = 16" in wave_artifact
    assert "waveamdmachine.global_load_lds_b128" in machine
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_f16_source_falls_back_to_4_byte_dma(tmp_path):
    async_func = """
  tt.func public @async_f16_source_4_byte_dma(%arg0: !tt.ptr<f16>, %n: i32 {tt.divisibility = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %n_splat = tt.splat %n : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %offsets = arith.remsi %range, %n_splat : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %ptr = tt.addptr %base, %offsets : tensor<64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<64x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>> -> <64xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "bytes = 4" in wave_artifact
    assert "bytes = 16" not in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_partial_f16_packet_mask_falls_back(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_partial_mask(%arg0: !tt.ptr<f16>, %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64xf16, #shared, #smem, mutable>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %cols_2d : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
    %n = tt.splat %arg1 : i32 -> tensor<1x64xi32, #blocked>
    %mask = arith.cmpi slt, %cols_2d, %n : tensor<1x64xi32, #blocked>
    %token = ttg.async_copy_global_to_local %ptr, %alloc mask %mask : tensor<1x64x!tt.ptr<f16>, #blocked> -> <1x64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1, preamble=preamble)

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_proves_arg_aligned_inner_packet_mask(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_aligned_packet_mask(%arg0: !tt.ptr<f16>, %m: i32, %n: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x32xf16, #shared, #smem, mutable>
    %rows = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %row_stride = arith.constant dense<32> : tensor<8x1xi32, #blocked>
    %row_offsets = arith.muli %rows_2d, %row_stride : tensor<8x1xi32, #blocked>
    %row_offsets_b = tt.broadcast %row_offsets : tensor<8x1xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %n_splat_offsets = tt.splat %n : i32 -> tensor<1x32xi32, #blocked>
    %col_offsets_mod = arith.remsi %cols_2d, %n_splat_offsets : tensor<1x32xi32, #blocked>
    %col_offsets_b = tt.broadcast %col_offsets_mod : tensor<1x32xi32, #blocked> -> tensor<8x32xi32, #blocked>
    %offsets = arith.addi %row_offsets_b, %col_offsets_b : tensor<8x32xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<8x32x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<8x32x!tt.ptr<f16>, #blocked>, tensor<8x32xi32, #blocked>
    %m_splat = tt.splat %m : i32 -> tensor<8x1xi32, #blocked>
    %row_mask = arith.cmpi slt, %rows_2d, %m_splat : tensor<8x1xi32, #blocked>
    %row_mask_b = tt.broadcast %row_mask : tensor<8x1xi1, #blocked> -> tensor<8x32xi1, #blocked>
    %n_splat = tt.splat %n : i32 -> tensor<1x32xi32, #blocked>
    %col_mask = arith.cmpi slt, %cols_2d, %n_splat : tensor<1x32xi32, #blocked>
    %col_mask_b = tt.broadcast %col_mask : tensor<1x32xi1, #blocked> -> tensor<8x32xi1, #blocked>
    %mask = arith.andi %row_mask_b, %col_mask_b : tensor<8x32xi1, #blocked>
    %token = ttg.async_copy_global_to_local %ptr, %alloc mask %mask : tensor<8x32x!tt.ptr<f16>, #blocked> -> <8x32xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=4, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_dma_load_lds"] > 0
    assert "waveamd.dma_load_lds" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_strided_f16_packet_source_falls_back(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_strided_source(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64xf16, #shared, #smem, mutable>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %two = arith.constant dense<2> : tensor<1x64xi32, #blocked>
    %offsets = arith.muli %cols_2d, %two : tensor<1x64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<1x64x!tt.ptr<f16>, #blocked> -> <1x64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1, preamble=preamble)

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_branch_local_assume_source_uses_async_copy_fallback(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    async_func = """
  tt.func public @async_branch_assume(%arg0: !tt.ptr<f16>, %stride: i32, %flag: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cond = arith.cmpi eq, %flag, %c0 : i32
    scf.if %cond {
      %unit_stride = arith.cmpi eq, %stride, %c1 : i32
      llvm.intr.assume %unit_stride : i1
    }
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x64xf16, #shared, #smem, mutable>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %stride_splat = tt.splat %stride : i32 -> tensor<1x64xi32, #blocked>
    %offsets = arith.muli %cols_2d, %stride_splat : tensor<1x64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %offsets : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<1x64x!tt.ptr<f16>, #blocked> -> <1x64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, async_func, num_warps=1, preamble=preamble)

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_lowers_i8_as_dma(tmp_path):
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
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_strided_i8_packet_source_falls_back(tmp_path):
    async_i8_func = """
  tt.func public @async_i8_strided(%arg0: !tt.ptr<i8>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %two = arith.constant dense<2> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %offsets = arith.muli %range, %two : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %offsets : tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, async_i8_func)

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_wait_pipeline_loop_carries_mem_tokens(tmp_path):
    pipeline_func = """
  tt.func public @async_f32_pipeline(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %init_wait = ttg.async_wait {num = 0 : i32}
    %loop = scf.for %iv = %c0 to %c2 step %c1 iter_args(%prev = %init_wait) -> (!ttg.async.token) {
      %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
      %group = ttg.async_commit_group tokens %token
      %wait = ttg.async_wait %prev, %group {num = 1 : i32}
      %loaded = ttg.local_load %alloc token %wait : !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> tensor<64xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
      scf.yield %group : !ttg.async.token
    }
    %flush = ttg.async_wait %loop {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, pipeline_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert metadata["tlx_wave_num_async_commit_groups"] == 1
    assert metadata["tlx_wave_num_async_waits"] == 2
    assert metadata["tlx_wave_num_wave_barriers"] == 2
    assert "scf.for" in wave_artifact
    assert "iter_args" in wave_artifact
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert wave_artifact.count("wave.wait") == 2
    assert wave_artifact.count("wave.barrier") == 2
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_multiple_in_flight_explicit_commit_wait(tmp_path):
    async_f32_func = """
  tt.func public @async_f32_multi_in_flight(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %alloc0 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %alloc1 = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %t0 = ttg.async_copy_global_to_local %ptr, %alloc0 : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %t1 = ttg.async_copy_global_to_local %ptr, %alloc1 : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %g0 = ttg.async_commit_group tokens %t0
    %g1 = ttg.async_commit_group
    %wait0 = ttg.async_wait %g0, %g1 {num = 1 : i32}
    %wait1 = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, async_f32_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 2
    assert metadata["tlx_wave_num_dma_load_lds"] == 2
    assert metadata["tlx_wave_num_async_commit_groups"] == 2
    assert metadata["tlx_wave_num_async_waits"] == 2
    assert metadata["tlx_wave_num_wave_barriers"] == 2
    assert wave_artifact.count("waveamd.dma_load_lds") == 2
    assert wave_artifact.count("wave.wait") == 2
    assert wave_artifact.count("wave.barrier") == 2
    assert wave_artifact.count("wave.join") >= 4
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_lowers_i8_multicomponent_layout_as_dma(tmp_path):
    async_i8_func = """
  tt.func public @async_i8_multicomponent(%arg0: !tt.ptr<i8>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i8> -> tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %token = ttg.async_copy_global_to_local %ptr, %alloc : tensor<64x!tt.ptr<i8>, #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>> -> <64xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>
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
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert wave_artifact.count("waveamd.dma_load_lds") == 1
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_unaligned_f16_view_falls_back_without_dma(tmp_path):
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
    mod, ctx = _parse_ttgir(tmp_path, async_f16_view_func)

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_async_copy_noncontiguous_2d_i8_packet_source_falls_back(
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

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact
    del ctx


def test_tlx_wave_bridge_lowers_async_copy_other_with_load_store_fallback():
    src = ASTSource(
        fn=_tlx_wave_async_other_kernel,
        signature={"in_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )

    compiled = triton_compile(src, target=GFX950_WAVE)
    wave_artifact = _asm_text(compiled, "wave")

    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies == 1
    assert compiled.metadata.tlx_wave_num_dma_load_lds == 0
    assert "waveamd.dma_load_lds" not in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "ttg.async_copy_global_to_local" not in wave_artifact


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
    assert metadata["tlx_wave_num_async_copies"] == 1
    assert metadata["tlx_wave_num_dma_load_lds"] == 1
    assert "waveamd.dma_load_lds" in wave
    assert "wave.cmpi" in wave
    del ctx


def test_tlx_wave_bridge_lowers_splatted_scalar_i1_constant_mask(tmp_path):
    mask_func = """
  tt.func public @scalar_i1_mask(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<64xi1, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %loaded = tt.load %ptr, %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %loaded, %mask : tensor<64x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, mask_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.cmpi" in wave
    assert "wave.load" in wave
    assert "wave.store" in wave
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


def test_tlx_wave_bridge_lowers_llvm_assume_range_and_divisibility(tmp_path):
    assume_func = """
  tt.func public @assume_range_div(%arg0: i32) attributes {noinline = false} {
    %c16 = arith.constant 16 : i32
    %c128 = arith.constant 128 : i32
    %c0 = arith.constant 0 : i32
    %lower = arith.cmpi sgt, %arg0, %c0 : i32
    %upper = arith.cmpi sgt, %c128, %arg0 : i32
    %range = arith.andi %lower, %upper : i1
    llvm.intr.assume %range : i1
    %rem = arith.remsi %arg0, %c16 : i32
    %divisible = arith.cmpi eq, %rem, %c0 : i32
    llvm.intr.assume %divisible : i1
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave.count("wave.assume") == 3
    assert '#wave.pred<"-1 + x >= 0">' in wave
    assert '#wave.pred<"-127 + x <= 0">' in wave
    assert '#wave.pred<"Mod(x, 16) == 0">' in wave
    assert "llvm.intr.assume" not in wave
    assert "arith.remsi" not in wave
    del ctx


def test_tlx_wave_bridge_lowers_llvm_assume_power_of_two(tmp_path):
    assume_func = """
  tt.func public @assume_power_of_two(%arg0: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %minus_one = arith.subi %arg0, %c1 : i32
    %pow2_bits = arith.andi %arg0, %minus_one : i32
    %pow2 = arith.cmpi eq, %pow2_bits, %c0 : i32
    %positive = arith.cmpi sgt, %arg0, %c0 : i32
    %assumption = arith.andi %positive, %pow2 : i1
    llvm.intr.assume %assumption : i1
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert wave.count("wave.assume") == 2
    assert '#wave.pred<"-1 + x >= 0">' in wave
    assert "x &" in wave
    assert "== 0" in wave
    assert "llvm.intr.assume" not in wave
    assert "arith.andi" not in wave
    del ctx


def test_tlx_wave_bridge_drops_standalone_fixed_width_power_of_two_assume(tmp_path):
    assume_func = """
  tt.func public @assume_power_of_two(%arg0: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %minus_one = arith.subi %arg0, %c1 : i32
    %pow2_bits = arith.andi %arg0, %minus_one : i32
    %pow2 = arith.cmpi eq, %pow2_bits, %c0 : i32
    llvm.intr.assume %pow2 : i1
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.assume" not in wave
    assert "x &" not in wave
    assert "llvm.intr.assume" not in wave
    del ctx


def test_tlx_wave_bridge_keeps_pow2_assumed_product_symbolic(tmp_path):
    assume_func = """
  tt.func public @assume_power_of_two_product(%arg0: !tt.ptr<i32>, %arg1: i32) attributes {noinline = false} {
    %c16 = arith.constant 16 : i32
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %c_i32_max = arith.constant 2147483647 : i32
    %divisor = arith.muli %arg1, %c16 : i32
    %positive = arith.cmpi sgt, %divisor, %c0 : i32
    %bounded = arith.cmpi sle, %divisor, %c_i32_max : i32
    %minus_one = arith.subi %divisor, %c1 : i32
    %pow2_bits = arith.andi %divisor, %minus_one : i32
    %pow2 = arith.cmpi eq, %pow2_bits, %c0 : i32
    %bounds = arith.andi %positive, %bounded : i1
    %assumption = arith.andi %bounds, %pow2 : i1
    llvm.intr.assume %assumption : i1
    %pid = tt.get_program_id x : i32
    %group = arith.divsi %pid, %divisor : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = tt.splat %group : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %value : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.binary muli" not in wave
    assert "wave.binary divsi" in wave
    assert "x &" in wave
    assert "-2147483647 + x <= 0" in wave
    assert "llvm.intr.assume" not in wave
    del ctx


def test_tlx_wave_bridge_lowers_shared_assume_condition_helpers(tmp_path):
    assume_func = """
  tt.func public @shared_assume_condition(%arg0: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %sum = arith.addi %arg0, %c1 : i32
    %positive = arith.cmpi sgt, %sum, %c0 : i32
    llvm.intr.assume %positive : i1
    scf.if %positive {
      %one = arith.constant 1 : i32
    }
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.assume" in wave
    assert "arith.cmpi sgt" in wave
    assert "scf.if" in wave
    del ctx


def test_tlx_wave_bridge_drops_unsigned_llvm_assume_facts(tmp_path):
    assume_func = """
  tt.func public @drop_unsigned_assume(%arg0: i32) attributes {noinline = false} {
    %c3 = arith.constant 3 : i32
    %c0 = arith.constant 0 : i32
    %nonnegative = arith.cmpi uge, %arg0, %c0 : i32
    llvm.intr.assume %nonnegative : i1
    %rem = arith.remui %arg0, %c3 : i32
    %divisible = arith.cmpi eq, %rem, %c0 : i32
    llvm.intr.assume %divisible : i1
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.assume" not in wave
    assert "llvm.intr.assume" not in wave
    assert "arith.remui" not in wave
    del ctx


def test_tlx_wave_bridge_drops_unsupported_llvm_assume(tmp_path):
    assume_func = """
  tt.func public @drop_unsupported_assume(%arg0: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %c0 = arith.constant 0 : i32
    %not_zero = arith.cmpi ne, %arg0, %c0 : i32
    llvm.intr.assume %not_zero : i1
    %diff = arith.subi %arg0, %c1 : i32
    %positive_diff = arith.cmpi sgt, %diff, %c0 : i32
    llvm.intr.assume %positive_diff : i1
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, assume_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.assume" not in wave
    assert "llvm.intr.assume" not in wave
    assert "arith.subi" not in wave
    del ctx


def test_tlx_wave_bridge_lowers_scalar_index_arithmetic(tmp_path):
    arith_func = """
  tt.func public @scalar_index_arith(%arg0: !tt.ptr<i32>, %arg1: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %pid = tt.get_program_id x : i32
    %sum = arith.addi %pid, %arg1 : i32
    %diff = arith.subi %sum, %c1 : i32
    %q = arith.divsi %diff, %c3 : i32
    %r = arith.remsi %diff, %c3 : i32
    %min = arith.minsi %q, %r : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = tt.splat %min : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %value : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, arith_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.binary divsi" in wave
    assert "wave.binary remsi" in wave
    assert "wave.select" in wave
    assert "arith.divsi" not in wave
    assert "arith.remsi" not in wave
    assert "arith.minsi" not in wave
    del ctx


def test_tlx_wave_bridge_emits_nsw_for_scalar_index_product(tmp_path):
    arith_func = """
  tt.func public @scalar_index_product(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %pid0 = tt.get_program_id x : i32
    %pid1 = tt.get_program_id y : i32
    %q0 = arith.divsi %pid0, %c32 : i32
    %q1 = arith.divsi %pid1, %c32 : i32
    %product = arith.muli %q0, %q1 : i32
    %value_scalar = arith.divsi %product, %c32 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %offset_base = tt.splat %value_scalar : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %offsets = arith.addi %offset_base, %range : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %offsets : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %data_scalar = arith.addi %c1, %c1 : i32
    %data = tt.splat %data_scalar : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %data : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, arith_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.binary muli" in wave
    assert "overflow<nsw> : index, index -> index" in wave
    del ctx


def test_tlx_wave_bridge_lowers_unsigned_index_div_rem(tmp_path):
    arith_func = """
  tt.func public @unsigned_index_arith(%arg0: !tt.ptr<i32>, %arg1: i32) attributes {noinline = false} {
    %c3 = arith.constant 3 : i32
    %q = arith.divui %arg1, %c3 : i32
    %r = arith.remui %arg1, %c3 : i32
    %value_scalar = arith.addi %q, %r : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = tt.splat %value_scalar : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %value : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, arith_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.binary divui" in wave
    assert "wave.binary remui" in wave
    assert "arith.divui" not in wave
    assert "arith.remui" not in wave
    del ctx


def test_tlx_wave_bridge_lowers_lane_varying_index_min(tmp_path):
    arith_func = """
  tt.func public @lane_varying_min(%arg0: !tt.ptr<i32>, %arg1: i32) attributes {noinline = false} {
    %c1 = arith.constant 1 : i32
    %limit_scalar = arith.addi %arg1, %c1 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %limit = tt.splat %limit_scalar : i32 -> tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %offset = arith.minsi %range, %limit : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %offset : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = arith.constant dense<0> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.store %ptr, %value : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, arith_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "wave.select" in wave
    assert "wave.cmpi" in wave
    assert "arith.minsi" not in wave
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

    python_paths = list(
        wave_bridge_emit._existing_paths(
            wave_bridge_emit._candidate_wave_python_paths()
        )
    )
    assert python_paths[0] == python_package.resolve()
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
        result_id: op.index for op in plan.ops for result_id in op.results
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

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_bridge_stage"] == "ttgir-op-lowering"
    assert "wave.cmpi" in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
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

        def index_expr(self, expr, bindings=None, result_type=None):
            value = FakeValue(f"idx{len(self.index_exprs)}", result_type)
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
        return SimpleNamespace(value_id=value_id, type_kind="index")

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

        class CastKind:
            IntConvert = "IntConvert"

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

        @staticmethod
        def simd_type(element_type, width):
            return FakeSimdType(width, element_type)

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []
            self.index_casts = []
            self.casts = []

        def workitem_id(self, axis=0, width=64):
            return FakeValue(f"thread{axis}", FakeSimdType(width, "i32"))

        def index_expr(self, expr, bindings=None, result_type=None):
            width = 1
            if bindings:
                widths = [
                    binding.type.width
                    for binding in bindings.values()
                    if isinstance(binding.type, FakeSimdType)
                ]
                width = widths[0] if widths else width
            value = FakeValue(
                f"idx{len(self.index_exprs)}",
                result_type or FakeSimdType(width),
            )
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

        def cast(self, value, typ, kind):
            result = FakeValue(f"wavecast{len(self.casts)}", typ)
            self.casts.append((value, typ, kind, result))
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

    bindings, width, active = wave_bridge_emit._blocked_tensor_dim_bindings(
        builder,
        value_plan,
        FakeW(),
        "ttg.async_copy_global_to_local source",
    )

    assert width == 64
    assert active is not None
    dim0 = str(builder.index_exprs[0][0])
    dim1 = str(builder.index_exprs[1][0])
    assert "mod(mod(tlx_tensor_42_thread,64),2)" in dim0
    assert "(2*mod(floor((tlx_tensor_42_thread/64)),4))" in dim0
    assert "mod(floor((mod(tlx_tensor_42_thread,64)/2)),32)" in dim1
    assert builder.index_casts == []
    assert builder.casts == []
    assert [expr[2].type.element_type for expr in builder.index_exprs[:2]] == [
        "index",
        "index",
    ]
    assert [expr[2].type.width for expr in builder.index_exprs[:2]] == [64, 64]
    assert {str(symbol) for symbol in bindings} == {"tlx_dim0", "tlx_dim1"}


def test_tlx_wave_bridge_blocked_layout_repeated_components_cover_extent():
    class FakeBlockedEncoding:
        def is_blocked_encoding(self):
            return True

        def get_blocked_size_per_thread(self):
            return (1, 1)

        def get_blocked_threads_per_warp(self):
            return (2, 32)

        def get_blocked_warps_per_cta(self):
            return (4, 1)

        def get_blocked_order(self):
            return (1, 0)

    value_plan = SimpleNamespace(
        value_id=44,
        type_kind="tensor",
        type="tensor<32x32x!tt.ptr<f32>>",
        shape=(32, 32),
        encoding=(
            "#ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], "
            "warpsPerCTA = [4, 1], order = [1, 0]}>"
        ),
        encoding_attr=FakeBlockedEncoding(),
    )
    layout = wave_bridge_emit._blocked_tensor_layout_info(
        value_plan,
        "ttg.async_copy_global_to_local source",
        "async copy address lowering",
    )

    assert (
        wave_bridge_emit._blocked_layout_component_count(
            value_plan,
            "ttg.async_copy_global_to_local source",
            "async copy address lowering",
        )
        == 4
    )
    assert wave_bridge_emit._blocked_layout_static_coord(
        layout, value_plan.shape, thread=0, component=0
    ) == (0, 0)
    assert wave_bridge_emit._blocked_layout_static_coord(
        layout, value_plan.shape, thread=0, component=3
    ) == (24, 0)


def test_tlx_wave_mfma_tile_store_coords_include_warp_offsets():
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

    class FakeW:
        class FragmentType:
            def __init__(self, typ):
                self.registers = 4
                self.wave_size = 64

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

        @staticmethod
        def simd_type(element_type, width):
            return FakeSimdType(width, element_type)

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []
            self.cmpis = []
            self.selects = []

        def workitem_id(self, axis=0, width=64):
            return FakeValue(f"thread{axis}", FakeSimdType(width, "index"))

        def lane_id(self, width=64):
            return FakeValue("lane", FakeSimdType(width, "index"))

        def index_expr(self, expr, bindings=None, result_type=None):
            value = FakeValue(
                f"idx{len(self.index_exprs)}",
                result_type or FakeSimdType(64, "index"),
            )
            self.index_exprs.append((expr, bindings or {}, value))
            return value

        def constant(self, typ, value):
            return FakeValue(f"const{value}", typ)

        def splat(self, value, width):
            return FakeValue(f"splat{value.name}", FakeSimdType(width, value.type))

        def cmpi(self, predicate, lhs, rhs):
            value = FakeValue(f"{predicate}{len(self.cmpis)}", FakeSimdType(64, "i1"))
            self.cmpis.append((predicate, lhs, rhs, value))
            return value

        def select(self, condition, true_value, false_value):
            value = FakeValue(f"select{len(self.selects)}", true_value.type)
            self.selects.append((condition, true_value, false_value, value))
            return value

    value_plan = SimpleNamespace(value_id=77, shape=(128, 256))
    fragment = FakeValue("frag", typ="fragment")
    builder = FakeBuilder()

    bindings, width, active = wave_bridge_emit._fragment_store_tile_dim_bindings(
        builder,
        value_plan,
        fragment,
        tile_offsets=(32, 0),
        component=0,
        w=FakeW(),
    )

    assert width == 64
    assert active is not None
    dim0 = str(builder.index_exprs[0][0])
    assert "32+" in dim0
    assert "floor((tlx_store_77_32_0_thread/64))" in dim0
    assert "(4*mod(floor((floor((tlx_store_77_32_0_thread/64))/1)),4))" in dim0
    assert {str(symbol) for symbol in bindings} == {"tlx_dim0", "tlx_dim1"}

    class FakeW32(FakeW):
        class FragmentType:
            def __init__(self, typ):
                self.registers = 16
                self.wave_size = 64

    builder = FakeBuilder()
    bindings, width, active = wave_bridge_emit._fragment_store_tile_dim_bindings(
        builder,
        value_plan,
        fragment,
        tile_offsets=(32, 0),
        component=4,
        w=FakeW32(),
    )

    assert width == 64
    assert active is not None
    dim0 = str(builder.index_exprs[0][0])
    dim1 = str(builder.index_exprs[1][0])
    assert "32+mod(mod(tlx_store_77_32_0_thread,64),32)" in dim0
    assert "4*floor((mod(tlx_store_77_32_0_thread,64)/32))" in dim1
    assert "+8" in dim1
    assert {str(symbol) for symbol in bindings} == {"tlx_dim0", "tlx_dim1"}


def test_tlx_wave_mfma32_single_fragment_store_uses_mfma_coords():
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

    class FakeBlockedEncoding:
        def is_blocked_encoding(self):
            return True

        def get_blocked_size_per_thread(self):
            return (1, 4)

        def get_blocked_threads_per_warp(self):
            return (8, 8)

        def get_blocked_warps_per_cta(self):
            return (4, 1)

        def get_blocked_order(self):
            return (1, 0)

    class FakeSimdType:
        def __init__(self, width, element_type="index"):
            self.width = width
            self.element_type = element_type

    class FakeValue:
        def __init__(self, name, typ=None):
            self.name = name
            self.type = typ

    class FakeW:
        class FragmentType:
            def __init__(self, typ):
                self.registers = 16
                self.wave_size = 64

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

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []

        def workitem_id(self, axis=0, width=64):
            return FakeValue(f"thread{axis}", FakeSimdType(width, "index"))

        def index_expr(self, expr, bindings=None, result_type=None):
            value = FakeValue(
                f"idx{len(self.index_exprs)}",
                result_type or FakeSimdType(64, "index"),
            )
            self.index_exprs.append((expr, bindings or {}, value))
            return value

    value_plan = SimpleNamespace(
        value_id=88,
        type_kind="tensor",
        type="tensor<32x32xf32>",
        element_type="f32",
        shape=(32, 32),
        encoding=(
            "#ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], "
            "warpsPerCTA = [4, 1], order = [1, 0]}>"
        ),
        encoding_attr=FakeBlockedEncoding(),
    )
    builder = FakeBuilder()

    bindings, width = wave_bridge_emit._store_dim_bindings(
        builder,
        value_plan,
        FakeValue("frag", typ="fragment"),
        FakeW(),
        component=4,
    )

    assert width == 64
    dim0 = str(builder.index_exprs[0][0])
    dim1 = str(builder.index_exprs[1][0])
    assert "mod(mod(mod(tlx_store_88_thread,64),32),32)" in dim0
    assert "4*floor((mod(tlx_store_88_thread,64)/32))" in dim1
    assert "+8" in dim1
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

        class CastKind:
            IntConvert = "IntConvert"

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

        @staticmethod
        def simd_type(element_type, width):
            return FakeSimdType(width, element_type)

    class FakeBuilder:
        def __init__(self):
            self.index_exprs = []
            self.index_casts = []
            self.casts = []

        def workitem_id(self, axis=0, width=64):
            return FakeValue(f"thread{axis}", FakeSimdType(width, "i32"))

        def index_expr(self, expr, bindings=None, result_type=None):
            width = 1
            if bindings:
                widths = [
                    binding.type.width
                    for binding in bindings.values()
                    if isinstance(binding.type, FakeSimdType)
                ]
                width = widths[0] if widths else width
            value = FakeValue(
                f"idx{len(self.index_exprs)}",
                result_type or FakeSimdType(width),
            )
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

        def cast(self, value, typ, kind):
            result = FakeValue(f"wavecast{len(self.casts)}", typ)
            self.casts.append((value, typ, kind, result))
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

    bindings, width, active = wave_bridge_emit._blocked_tensor_dim_bindings(
        builder,
        value_plan,
        FakeW(),
        "ttg.async_copy_global_to_local source",
        component=1,
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
            result = FakeValue(
                f"splat{len(self.splats)}", FakeSimdType(width, value.type)
            )
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
    assert builder.index_casts == []
    assert builder.cmpis[0][0] == "ult"
    assert builder.cmpis[0][1] is builder.splats[0][2]
    assert builder.cmpis[0][2] is builder.splats[1][2]
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
    shared = (
        "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>"
    )
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
    plan = wave_bridge._build_bridge_plan(mod, kernel)
    assert (
        len(
            [
                address
                for address in plan.addresses
                if address.op == "ttg.async_copy_global_to_local"
            ]
        )
        == 1
    )
    del ctx


def test_tlx_wave_async_copy_other_bool_constant_uses_copied_element_type():
    class FakeValue:
        def __init__(self, typ, value=None):
            self.type = typ
            self.value = value

    class FakeBuilder:
        def __init__(self):
            self.constants = []
            self.splats = []

        def constant(self, typ, value):
            result = FakeValue(typ, value)
            self.constants.append((typ, value, result))
            return result

        def splat(self, value, element_type=None, width=None):
            result = FakeValue(f"simd<{element_type},{width}>", value)
            self.splats.append((value, element_type, width, result))
            return result

    class FakeW:
        def i1(self):
            return "i1"

    address_plan = wave_bridge_plan._ValuePlan(
        0,
        "value",
        "tt.addptr",
        0,
        "tensor<16x16x!tt.ptr<i1>>",
        "tensor",
        (16, 16),
        "!tt.ptr<i1>",
        1,
        "i1",
        "#blocked",
        None,
        None,
        None,
        None,
        None,
        (),
        "varying",
    )
    address = SimpleNamespace(element_type="i1", element_byte_width=1)
    other_plan = wave_bridge_emit._async_copy_other_value_plan(address_plan, address)
    state = {
        "values": {
            1: SimpleNamespace(
                producer="arith.constant",
                const_value=False,
                type="tensor<16x16xi1>",
            )
        },
        "wave_values": {
            1: wave_bridge_emit._WaveValue(
                "mask_expr", wave_bridge_emit._MaskConst(False)
            )
        },
    }
    builder = FakeBuilder()

    value = wave_bridge_emit._load_other_value(
        builder,
        state,
        1,
        other_plan,
        64,
        FakeW(),
        "ttg.async_copy_global_to_local other",
    )

    assert other_plan.element_type == "i1"
    assert builder.constants[0][:2] == ("i1", 0)
    assert builder.splats[0][1:] == ("i1", 64, value)


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

    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, one_func, target="hip:gfx942")
    wave = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options(arch="gfx942")
    )
    assert "waveamdmachine.target = \"amdgcn-amd-amdhsa--gfx942\"" in wave
    assert metadata["tlx_wave_arch"] == "gfx942"
    assert metadata["tlx_wave_ttgir_target"] == "hip:gfx942"
    del ctx

    with pytest.raises(ValueError, match="expected TTGIR target hip:gfx950"):
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


def test_tlx_wave_bridge_lowers_if_without_results(tmp_path):
    region_func = """
  tt.func public @region_kernel(%flag: i1) attributes {noinline = false} {
    scf.if %flag {
      %one = arith.constant 1 : i32
    }
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, region_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "scf.if" in wave
    del ctx


def test_tlx_wave_bridge_lowers_scalar_compare_if_condition(tmp_path):
    region_func = """
  tt.func public @region_kernel(%lhs: i32, %rhs: i32) attributes {noinline = false} {
    %flag = arith.cmpi slt, %lhs, %rhs : i32
    scf.if %flag {
      %one = arith.constant 1 : i32
    }
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, region_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "arith.cmpi slt" in wave
    assert "scf.if" in wave
    del ctx


def test_tlx_wave_lifts_cf_diamond_before_bridge(tmp_path):
    cf_func = """
  tt.func public @cf_diamond(%arg0: i32, %flag: i1) attributes {noinline = false} {
    cf.cond_br %flag, ^bb1(%arg0 : i32), ^bb2(%arg0 : i32)
  ^bb1(%x: i32):
    %one = arith.constant 1 : i32
    %then = arith.addi %x, %one : i32
    cf.br ^bb3(%then : i32)
  ^bb2(%y: i32):
    %two = arith.constant 2 : i32
    %else = arith.addi %y, %two : i32
    cf.br ^bb3(%else : i32)
  ^bb3(%result: i32):
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, cf_func)
    changed = passes.convert.triton_lift_cf_to_scf(mod)
    ttgir = str(mod)

    assert changed
    assert "cf.cond_br" not in ttgir
    assert "cf.br" not in ttgir
    assert "scf.if" in ttgir

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert "cf.cond_br" not in wave
    assert "cf.br" not in wave
    assert "scf.if" in wave
    del ctx


def test_tlx_wave_bridge_rejects_nested_side_effects_under_if(tmp_path):
    region_func = """
  tt.func public @if_nested_for_store(%flag: i1, %arg0: !tt.ptr<i32>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %base = tt.splat %arg0 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>, tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %value = arith.constant dense<0> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    scf.if %flag {
      scf.for %iv = %c0 to %c1 step %c1 {
        tt.store %ptr, %value : tensor<64x!tt.ptr<i32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
      }
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, region_func)

    with pytest.raises(ValueError, match="side effects inside scf\\.if.*scf\\.for"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


@pytest.mark.parametrize(
    ("local_func", "op_name"),
    [
        (
            """
  tt.func public @if_yield_kernel(%flag: i1) attributes {noinline = false} {
    %result = scf.if %flag -> (i32) {
      %one = arith.constant 1 : i32
      scf.yield %one : i32
    } else {
      %two = arith.constant 2 : i32
      scf.yield %two : i32
    }
    tt.return
  }
""",
            "scf.if",
        ),
        (
            """
  tt.func public @for_iter_args_kernel() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant 0 : i32
    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%carried = %init) -> (i32) {
      %one = arith.constant 1 : i32
      %next = arith.addi %carried, %one : i32
      scf.yield %next : i32
    }
    tt.return
  }
""",
            "scf.for",
        ),
    ],
)
def test_tlx_wave_bridge_lowers_value_yielding_control_flow(
    tmp_path, local_func, op_name
):
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    wave = wave_bridge.stop_before_wave_lowering(mod, metadata, _wave_bridge_options())

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert op_name in wave
    assert "scf.yield" in wave
    del ctx


def test_tlx_wave_bridge_rejects_unsupported_loop_carried_tensor(tmp_path):
    local_func = """
  tt.func public @for_tensor_iter_arg_kernel() attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = arith.constant dense<0> : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%carried = %init) -> (tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>) {
      scf.yield %carried : tensor<64xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func)

    with pytest.raises(ValueError, match="unsupported loop-carried"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
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

    with pytest.raises(ValueError, match="supports only matching f16 x f16 or bf16 x bf16"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_bridge_lowers_local_store_of_dot_fragment(tmp_path):
    dot_func = """
  tt.func public @dot_with_local_store() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %out_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    ttg.local_store %dot, %out_alloc : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> !ttg.memdesc<32x32xf32, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_mmas"] == 1
    assert metadata["tlx_wave_num_wave_barriers"] >= 1
    assert "wave.store" in wave_artifact
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
    assert "f16 x f16 or bf16 x bf16" in message
    assert "#ttg.dot_op" in message
    assert "opIdx = 0" in message
    del ctx


def test_tlx_wave_bridge_lowers_bf16_dot_local_load(tmp_path):
    dot_func = """
  tt.func public @bf16_dot_local_load() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xbf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xbf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xbf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xbf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_waveamd_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 2
    assert metadata["tlx_wave_num_fragment_packs"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 1
    assert metadata["tlx_wave_num_mmas"] == 1
    assert (
        wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_BF16_MMA_KIND}"')
        == 1
    )
    assert "waveamdmachine.mfma_f32_16x16x32_bf16" in machine
    del ctx


def test_tlx_wave_bridge_rejects_gfx942_mfma_wave32_gap(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @dot_local_load_mfma_gfx942() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        dot_func,
        target="hip:gfx942",
        preamble=preamble,
    )

    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(
            mod, {}, _wave_bridge_options(arch="gfx942")
        )
    message = str(exc_info.value)
    assert "gfx942/CDNA3 MFMA" in message
    assert "wave32" in message
    assert "wave64" in message
    del ctx


def test_tlx_wave_bridge_lowers_gfx950_mfma32_swizzled_fragment_load(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @local_load_mfma32_swizzled() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, local_func, preamble=preamble)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 2
    assert metadata["tlx_wave_num_fragment_packs"] == 2
    assert metadata["tlx_wave_num_fragment_fills"] == 0
    assert metadata["tlx_wave_num_mmas"] == 0
    assert wave_artifact.count("waveamd.fragment_pack") == 2
    assert "waveamd.mma" not in wave_artifact
    del ctx


def test_tlx_wave_bridge_lowers_gfx950_32x32x16_mfma_layout(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    dot_func = """
  tt.func public @dot_local_load_mfma32() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func, preamble=preamble)

    metadata = {}
    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )
    machine = _run_waveamd_to_machine(wave_artifact)

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 4
    assert metadata["tlx_wave_num_fragment_fills"] == 1
    assert metadata["tlx_wave_num_mmas"] == 2
    assert (
        wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA32_KIND}"')
        == 2
    )
    assert "waveamdmachine.mfma_f32_32x32x16_f16" in machine
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
    assert "expected f16/bf16 dot operand" in message
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


def test_tlx_wave_bridge_rejects_noncontiguous_swizzled_fragment_load(tmp_path):
    dot_func = """
  tt.func public @dot_kernel(%c: !tt.ptr<f32>) attributes {noinline = false} {
    %slot = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %a_view = ttg.memdesc_index %a_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_view = ttg.memdesc_index %b_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %lhs = ttg.local_load %a_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.local_load %b_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)
    with pytest.raises(ValueError, match="physically contiguous|byte offset"):
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    del ctx


def test_tlx_wave_bridge_lowers_blocked_dot_operand_layout_conversion(tmp_path):
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
    %out = ttg.convert_layout %dot : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %zero = arith.constant dense<0> : tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_base = tt.splat %c : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %out_ptr = tt.addptr %out_base, %zero : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<32x32xi32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %out_ptr, %out : tensor<32x32x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    metadata = {}
    mod, ctx = _parse_ttgir(tmp_path, dot_func)

    wave_artifact = wave_bridge.stop_before_wave_lowering(
        mod, metadata, _wave_bridge_options()
    )

    assert metadata["tlx_wave_status"] == "emitted_wave_ttgir_op_lowering"
    assert metadata["tlx_wave_num_wave_local_loads"] == 4
    assert metadata["tlx_wave_num_fragment_packs"] == 4
    assert metadata["tlx_wave_num_fragment_fills"] == 1
    assert metadata["tlx_wave_num_mmas"] == 1
    assert wave_artifact.count(f'waveamd.mma "{wave_bridge._GFX950_F16_MMA_KIND}"') == 1
    assert "wave.store" in wave_artifact
    del ctx


def test_tlx_wave_bridge_rejects_unsupported_dot_operand_layout_conversion(tmp_path):
    dot_func = """
  tt.func public @dot_kernel(%c: !tt.ptr<f32>) attributes {noinline = false} {
    %slot = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %a_view = ttg.memdesc_index %a_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %b_view = ttg.memdesc_index %b_alloc[%slot] : !ttg.memdesc<1x32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %lhs_physical = ttg.local_load %a_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs_physical = ttg.local_load %b_view : !ttg.memdesc<32x32xf16, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %lhs = ttg.convert_layout %lhs_physical : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %rhs = ttg.convert_layout %rhs_physical : tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<32x32xf32, #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, dot_func)
    with pytest.raises(ValueError) as exc_info:
        wave_bridge.stop_before_wave_lowering(mod, {}, _wave_bridge_options())
    message = str(exc_info.value)
    assert "ttg.convert_layout to a dot operand fragment" in message
    assert "source parent" in message
    assert "result parent" in message
    assert "sizePerThread=(2, 2)" in message
    assert "sizePerThread=(1, 2)" in message
    assert "supported layouts" in message
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
