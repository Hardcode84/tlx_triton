import importlib.util
import subprocess
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


def _warmup_gemm_wp_tlx_wave(
    tmp_path,
    monkeypatch,
    m=32,
    n=32,
    k=128,
    block_m=32,
    block_n=32,
    block_k=32,
    num_warps=4,
    group_m=16,
    num_buffers=2,
    matrix_instr_nonkdim=0,
    kpack=1,
    a_strides=None,
    b_strides=None,
    c_strides=None,
):
    import triton
    from triton import knobs
    from triton.backends import backends
    from triton.runtime.jit import MockTensor

    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "tlx_wave")
    wave_opt = (
        Path(__file__).parents[2] / "wave" / "build" / "wave-build" / "bin" / "wave-opt"
    )
    if wave_opt.exists():
        monkeypatch.setenv("TRITON_WAVE_OPT", str(wave_opt))

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

        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

        a = MockTensor(torch.float16, [m, k])
        b = MockTensor(torch.float16, [k, n])
        c = MockTensor(torch.float32, [m, n])
        a_strides = a.stride() if a_strides is None else a_strides
        b_strides = b.stride() if b_strides is None else b_strides
        c_strides = c.stride() if c_strides is None else c_strides
        compiled = tutorial.gemm_wp.warmup(
            a,
            b,
            c,
            m,
            n,
            k,
            a_strides[0],
            a_strides[1],
            b_strides[0],
            b_strides[1],
            c_strides[0],
            c_strides[1],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            NUM_BUFFERS=num_buffers,
            NUM_XCDS=tutorial.NUM_XCDS,
            XCD_CHUNK=4,
            num_warps=num_warps,
            num_stages=1,
            waves_per_eu=0,
            matrix_instr_nonkdim=matrix_instr_nonkdim,
            kpack=kpack,
            grid=grid,
        )
    return compiled


def _asm_text(compiled, artifact):
    text = compiled.asm[artifact]
    if isinstance(text, bytes):
        text = text.decode()
    return text


def _wave_text(compiled):
    return _asm_text(compiled, "wave")


def _assert_mfma32_kind(wave):
    custom = 'waveamd.mma "mfma.f32.32x32x16.f16"'
    generic = 'kind = "mfma.f32.32x32x16.f16"'
    assert custom in wave or generic in wave


def _run_wave_promote_buffer_to_machine(wave_artifact):
    wave_opt = (
        Path(__file__).parents[2] / "wave" / "build" / "wave-build" / "bin" / "wave-opt"
    )
    if not wave_opt.exists():
        pytest.skip("wave-opt is not built")
    result = subprocess.run(
        [
            str(wave_opt),
            "-",
            "--wave-expand-integer-div-rem",
            "--canonicalize",
            "--cse",
            "--wave-simplify-index-exprs",
            "--canonicalize",
            "--cse",
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


def test_gemm_wp_tlx_wave_warmup_emits_wave_handoff(monkeypatch, tmp_path):
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch)

    ttgir = _asm_text(compiled, "ttgir")
    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies >= 4
    assert (
        compiled.metadata.tlx_wave_num_dma_load_lds
        >= compiled.metadata.tlx_wave_num_async_copies
    )
    assert compiled.metadata.tlx_wave_num_async_waits >= 2
    assert compiled.metadata.tlx_wave_num_mmas > 1
    assert "scf.for" in wave
    assert "waveamd.dma_load_lds" in wave
    assert "waveamd.transpose_load" in wave
    assert "bytes = 4" in wave
    _assert_mfma32_kind(wave)
    assert "waveamdmachine.target" in wave
    assert (
        "#ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>"
        in ttgir
    )
    assert (
        "#ttg.padded_shared<[512:+32] {order = [1, 0], shape = [32, 32]}"
        in ttgir
    )
    assert "#ttg.linear" in ttgir
    assert "{contiguity = 2 : i32}" in ttgir


def test_gemm_wp_tlx_wave_epilogue_promotes_packed_store_to_buffer(
    monkeypatch, tmp_path
):
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch)

    wave = _wave_text(compiled)
    machine = _run_wave_promote_buffer_to_machine(wave)

    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert wave.count("wave.pack") >= 4
    assert (
        '#wave.pred<"x >= 0">, #wave.pred<"-536870907 + x <= 0">'
        in wave
    )
    assert '#wave.pred<"x >= 0">, #wave.pred<"-2147483647 + x <= 0">' in wave
    assert "floor(1/8*tlx_pow2_divsi" in wave
    assert "Mod(tlx_pow2_remsi" in wave
    assert len(machine.splitlines()) < 32_000
    assert machine.count("waveamdmachine.tuple_to_elements") < 8_000
    assert machine.count("waveamdmachine.s_cselect_b32") < 1_200
    assert machine.count("waveamdmachine.buffer_store_tuple_b32") == 4
    assert "waveamdmachine.global_store_b32_addr64" not in machine
    assert "waveamdmachine.global_store_b128_addr64" not in machine


def test_gemm_wp_tlx_wave_warmup_normalizes_deprecated_gfx950_kpack(
    monkeypatch, tmp_path
):
    with pytest.warns(UserWarning, match="kpack is deprecated"):
        compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch, kpack=2)

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert "waveamd.mma" in wave
    _assert_mfma32_kind(wave)


def test_gemm_wp_tlx_wave_warmup_handles_edge_tiles(monkeypatch, tmp_path):
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch, m=48, n=48, k=80)

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert (
        compiled.metadata.tlx_wave_num_dma_load_lds
        >= compiled.metadata.tlx_wave_num_async_copies
    )
    assert "waveamd.dma_load_lds" in wave
    assert "ttg.async_copy_global_to_local" not in wave


@pytest.mark.parametrize("block_m,block_n", [(128, 256), (256, 128), (256, 256)])
def test_gemm_wp_tlx_wave_warmup_lowers_full_mfma_layout(
    monkeypatch, tmp_path, block_m, block_n
):
    compiled = _warmup_gemm_wp_tlx_wave(
        tmp_path,
        monkeypatch,
        m=4096,
        n=4096,
        k=4096,
        block_m=block_m,
        block_n=block_n,
        block_k=32,
        num_warps=8,
    )

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_mmas >= 32
    assert compiled.metadata.tlx_wave_num_fragment_fills >= 32
    _assert_mfma32_kind(wave)


def test_gemm_wp_tlx_wave_warmup_lowers_8_warp_32x32_layout(monkeypatch, tmp_path):
    compiled = _warmup_gemm_wp_tlx_wave(
        tmp_path,
        monkeypatch,
        m=32,
        n=32,
        k=128,
        block_m=32,
        block_n=32,
        block_k=32,
        num_warps=8,
    )

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_mmas > 1
    _assert_mfma32_kind(wave)


def test_gemm_wp_validation_allows_non_unit_inner_strides():
    tutorial = _load_gemm_wp_module()
    a = _FakeTensor((32, 80), (160, 2))
    b = _FakeTensor((80, 48), (48, 1))

    tutorial._validate_dma_packet_shape(a, b, 48, 80, 32, 32, 32, 2, 16)


@pytest.mark.parametrize(
    "stride_override",
    [
        {"a_strides": (256, 2)},
        {"b_strides": (64, 2)},
    ],
)
def test_gemm_wp_tlx_wave_warmup_falls_back_for_non_unit_inner_stride(
    monkeypatch, tmp_path, stride_override
):
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch, **stride_override)

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies >= 4
    assert "wave.load" in wave
    assert "wave.store" in wave
    assert "ttg.async_copy_global_to_local" not in wave


@pytest.mark.parametrize("shape", [(32, 33, 80), (32, 48, 81)])
def test_gemm_wp_tlx_wave_warmup_falls_back_for_unaligned_packet_edges(
    monkeypatch, tmp_path, shape
):
    m, n, k = shape
    compiled = _warmup_gemm_wp_tlx_wave(tmp_path, monkeypatch, m=m, n=n, k=k)

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert compiled.metadata.tlx_wave_num_async_copies >= 4
    assert "wave.load" in wave
    assert "wave.store" in wave
    assert "ttg.async_copy_global_to_local" not in wave


def test_gemm_wp_tlx_wave_warmup_lowers_non_power_of_two_group_divisors(
    monkeypatch, tmp_path
):
    compiled = _warmup_gemm_wp_tlx_wave(
        tmp_path,
        monkeypatch,
        m=224,
        n=80,
        k=80,
        group_m=4,
    )

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert "waveamd.mma" in wave
    assert "(x & -1 + x) == 0" not in wave


def test_gemm_wp_validation_rejects_too_few_k_tiles():
    tutorial = _load_gemm_wp_module()
    a = _FakeTensor((32, 32), (32, 1))
    b = _FakeTensor((32, 32), (32, 1))

    with pytest.raises(ValueError, match="ceil\\(K / BLOCK_K\\).*NUM_BUFFERS"):
        tutorial._validate_dma_packet_shape(a, b, 32, 32, 32, 32, 32, 2, 16)


def test_gemm_wp_tlx_wave_warmup_lowers_non_power_of_two_tail_group_size(
    monkeypatch, tmp_path
):
    compiled = _warmup_gemm_wp_tlx_wave(
        tmp_path,
        monkeypatch,
        m=224,
        n=64,
        k=80,
        group_m=4,
    )

    wave = _wave_text(compiled)
    assert compiled.metadata.tlx_wave_status == "emitted_wave_ttgir_op_lowering"
    assert "waveamd.mma" in wave
    assert "(x & -1 + x) == 0" not in wave


def test_gemm_wp_run_allows_packet_aligned_edge_shape():
    tutorial = _load_gemm_wp_module()
    a = _FakeTensor((48, 80), (80, 1))
    b = _FakeTensor((80, 48), (48, 1))

    tutorial._validate_dma_packet_shape(a, b, 48, 80, 32, 32, 32, 2, 16)
