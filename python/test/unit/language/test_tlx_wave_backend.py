import pytest

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile


pytestmark = pytest.mark.skipif("tlx_wave" not in backends, reason="tlx_wave backend is not installed")

GFX950_WAVE = GPUTarget("tlx_wave", "gfx950", 64)


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

    wave_artifact = compiled.asm["wave"]
    if isinstance(wave_artifact, bytes):
        wave_artifact = wave_artifact.decode("utf-8")
    assert "tt.func" in wave_artifact
    assert "local_alloc" in wave_artifact
