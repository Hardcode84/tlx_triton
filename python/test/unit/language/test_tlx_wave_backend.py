import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile


pytestmark = pytest.mark.skipif("tlx_wave" not in backends, reason="tlx_wave backend is not installed")

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

    buffers = tlx.local_alloc((BLOCK_SIZE, ), tl.float32, 1)
    tile = tlx.local_view(buffers, 0)
    values = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tlx.local_store(tile, values)
    out = tlx.local_load(tile)
    tl.store(out_ptr + offs, out, mask=mask)


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


def test_tlx_wave_scaffold_stops_before_wave_lowering():
    src = ASTSource(
        fn=_tlx_wave_local_kernel,
        signature={"in_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32"},
        constexprs={"BLOCK_SIZE": 64},
    )

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.target.backend == "tlx_wave"
    assert compiled.metadata.arch == "gfx950"
    assert compiled.metadata.tlx_wave_status == "stopped_before_wave_lowering"
    assert "ttgir" in compiled.asm
    assert "wave" in compiled.asm

    wave_artifact = _asm_text(compiled, "wave")
    assert "tt.func" in wave_artifact
    assert "local_alloc" in wave_artifact


def test_tlx_wave_gemm_cutoff_preserves_async_gemm_shape():
    src = ASTSource(
        fn=_tlx_wave_gemm_cutoff_kernel,
        signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32", "M": "i32", "N": "i32"},
        constexprs={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32, "K_ITERS": 2, "NUM_BUFFERS": 2},
    )

    compiled = triton_compile(src, target=GFX950_WAVE)

    assert compiled.metadata.target == GFX950_WAVE
    assert compiled.metadata.num_ctas == 1
    assert compiled.metadata.warp_size == 64
    assert compiled.metadata.tlx_wave_status == "stopped_before_wave_lowering"
    assert "wave" in compiled.asm

    wave_artifact = _asm_text(compiled, "wave")
    assert wave_artifact.count("ttg.local_alloc") >= 2
    assert "ttg.memdesc_index" in wave_artifact
    assert wave_artifact.count("ttg.async_copy_global_to_local") >= 4
    assert wave_artifact.count("ttg.async_commit_group") >= 2
    assert wave_artifact.count("ttg.async_wait") >= 2
    assert wave_artifact.count("ttg.local_load") >= 4
    assert wave_artifact.count("tt.dot") >= 2
    assert "tt.store" in wave_artifact
    assert "amdg." not in wave_artifact
