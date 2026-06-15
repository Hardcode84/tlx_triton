import importlib.util
from pathlib import Path

import pytest
import torch


def _load_gemm_wp_module():
    path = Path(__file__).with_name("amd-gemm-warp-pipeline_test.py")
    spec = importlib.util.spec_from_file_location("tlx_wave_gemm_wp_tutorial", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTensor:
    def __init__(self, shape, strides):
        self.shape = shape
        self._strides = strides

    def stride(self, dim):
        return self._strides[dim]


def _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch, m=32, n=32, k=128):
    import triton
    from triton import knobs
    from triton.backends import backends
    from triton.runtime.jit import MockTensor

    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "tlx_wave")

    if "tlx_wave" not in backends:
        pytest.skip("tlx_wave backend is not installed")

    with knobs.cache.scope(), knobs.runtime.scope():
        knobs.cache.dir = str(tmp_path / "triton-cache")
        knobs.runtime.override_arch = "gfx950"
        triton.runtime.driver._default = None
        triton.runtime.driver._active = None
        try:
            target = triton.runtime.driver.active.get_current_target()
        except RuntimeError as exc:
            pytest.skip(f"tlx_wave backend is not active: {exc}")
        if target.backend != "tlx_wave" or target.arch != "gfx950":
            pytest.skip(f"requires tlx_wave:gfx950, got {target}")

        tutorial = _load_gemm_wp_module()

        block_m = block_n = 32
        block_k = 32
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

        a = MockTensor(torch.float16, [m, k])
        b = MockTensor(torch.float16, [k, n])
        c = MockTensor(torch.float32, [m, n])
        compiled = tutorial.gemm_wp.warmup(
            a,
            b,
            c,
            m,
            n,
            k,
            a.stride()[0],
            a.stride()[1],
            b.stride()[0],
            b.stride()[1],
            c.stride()[0],
            c.stride()[1],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=16,
            NUM_BUFFERS=2,
            NUM_XCDS=tutorial.NUM_XCDS,
            XCD_CHUNK=4,
            num_warps=4,
            num_stages=1,
            waves_per_eu=0,
            matrix_instr_nonkdim=16,
            grid=grid,
        )
    return compiled


def test_gemm_wp_tlx_wave_warmup_emits_wave_handoff(monkeypatch, tmp_path):
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch)

    wave = compiled.asm["wave"]
    if isinstance(wave, bytes):
        wave = wave.decode()
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies >= 4
    assert compiled.metadata.tlx_wave_num_dma_load_lds == compiled.metadata.tlx_wave_num_async_copies
    assert compiled.metadata.tlx_wave_num_async_waits >= 2
    assert compiled.metadata.tlx_wave_num_mmas > 1
    assert "scf.for" in wave
    assert "waveamd.dma_load_lds" in wave
    assert "waveamd.mma" in wave
    assert "waveamdmachine.target" in wave


def test_gemm_wp_tlx_wave_warmup_handles_edge_tiles(monkeypatch, tmp_path):
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch, m=48, n=48, k=80)

    wave = compiled.asm["wave"]
    if isinstance(wave, bytes):
        wave = wave.decode()
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_dma_load_lds == compiled.metadata.tlx_wave_num_async_copies
    assert "waveamd.dma_load_lds" in wave
    assert "ttg.async_copy_global_to_local" not in wave


def test_gemm_wp_run_rejects_non_unit_inner_strides():
    tutorial = _load_gemm_wp_module()
    a = _FakeTensor((32, 80), (160, 2))
    b = _FakeTensor((80, 48), (48, 1))
    c = _FakeTensor((32, 48), (48, 1))

    with pytest.raises(ValueError, match="unit inner strides"):
        tutorial.run(a, b, c, 32, 32, 32, 2, 4, 16)


@pytest.mark.parametrize("shape", [(32, 33, 80), (32, 48, 81)])
def test_gemm_wp_run_rejects_unaligned_dma_packet_edges(shape):
    tutorial = _load_gemm_wp_module()
    m, n, k = shape
    a = _FakeTensor((m, k), (k, 1))
    b = _FakeTensor((k, n), (n, 1))
    c = _FakeTensor((m, n), (n, 1))

    with pytest.raises(ValueError, match="N/K.*divisible by 16"):
        tutorial.run(a, b, c, 32, 32, 32, 2, 4, 16)


def test_gemm_wp_run_allows_packet_aligned_edge_shape():
    tutorial = _load_gemm_wp_module()
    a = _FakeTensor((48, 80), (80, 1))
    b = _FakeTensor((80, 48), (48, 1))

    tutorial._validate_dma_packet_shape(a, b, 48, 80, 32, 32)
