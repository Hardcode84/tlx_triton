"""Unit tests for triton.tools.tlx_benchmark_gen.

Tests cover the argument-capture serialization, grid capture, and standalone
test-script generation logic.  All tests are CPU-only unless marked with
@pytest.mark.skipif (GPU-dependent tests are gated on CUDA availability).
"""

import json
import os
from collections import OrderedDict

import pytest
import torch

from triton.tools.tlx_benchmark_gen import (
    _dtype_str,
    _ensure_dump_dir,
    capture_grid,
    capture_kernel_args,
    generate_standalone_test,
)

# ---------------------------------------------------------------------------
# _dtype_str
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (torch.bfloat16, "bfloat16"),
        (torch.float32, "float32"),
        (torch.float16, "float16"),
        (torch.int32, "int32"),
        (torch.int8, "int8"),
        (torch.bool, "bool"),
    ],
)
def test_dtype_str(dtype, expected):
    assert _dtype_str(dtype) == expected


# ---------------------------------------------------------------------------
# _ensure_dump_dir
# ---------------------------------------------------------------------------


def test_ensure_dump_dir_creates_dir(monkeypatch):
    monkeypatch.delenv("TRITON_TLX_DUMP_DIR", raising=False)
    dump_dir = _ensure_dump_dir()
    try:
        assert os.path.isdir(dump_dir)
        assert os.environ["TRITON_TLX_DUMP_DIR"] == dump_dir
    finally:
        monkeypatch.delenv("TRITON_TLX_DUMP_DIR", raising=False)
        os.rmdir(dump_dir)


def test_ensure_dump_dir_reuses_existing(monkeypatch, tmp_path):
    existing = str(tmp_path)
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", existing)
    assert _ensure_dump_dir() == existing


# ---------------------------------------------------------------------------
# capture_kernel_args — scalars
# ---------------------------------------------------------------------------


def test_capture_kernel_args_scalars(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    bound_args = OrderedDict([("alpha", 0.5), ("count", 42), ("flag", True)])
    signature = {"alpha": "fp32", "count": "i32", "flag": "i1"}
    constexprs = {}

    capture_kernel_args(bound_args, signature, constexprs)

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    args = meta["args"]
    assert len(args) == 3

    assert args[0]["name"] == "alpha"
    assert args[0]["kind"] == "scalar"
    assert args[0]["scalar_type"] == "float"
    assert args[0]["value"] == 0.5

    assert args[1]["name"] == "count"
    assert args[1]["kind"] == "scalar"
    assert args[1]["scalar_type"] == "int"
    assert args[1]["value"] == 42

    # bool must come before int in isinstance checks
    assert args[2]["name"] == "flag"
    assert args[2]["kind"] == "scalar"
    assert args[2]["scalar_type"] == "bool"
    assert args[2]["value"] is True


# ---------------------------------------------------------------------------
# capture_kernel_args — tensors
# ---------------------------------------------------------------------------


def test_capture_kernel_args_tensors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    t = torch.randn(4, 48, 1024, dtype=torch.float32)
    bound_args = OrderedDict([("M", t)])
    signature = {"M": "*fp32"}
    constexprs = {}

    capture_kernel_args(bound_args, signature, constexprs)

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    entry = meta["args"][0]
    assert entry["kind"] == "tensor"
    assert entry["shape"] == [4, 48, 1024]
    assert entry["dtype"] == "float32"
    assert entry["strides"] == list(t.stride())


# ---------------------------------------------------------------------------
# capture_kernel_args — TensorDescriptors
# ---------------------------------------------------------------------------


def test_capture_kernel_args_tensor_descriptors(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    # TensorDescriptor requires 16-byte aligned base pointer and strides.
    # On CPU tensors, data_ptr() alignment depends on the allocator, so we
    # directly write the expected JSON structure and verify it round-trips
    # correctly (testing the serialization format, not the isinstance path).
    base = torch.randn(4, 128, dtype=torch.bfloat16)

    import triton.tools.tlx_benchmark_gen as tbg
    dump_dir = tbg._ensure_dump_dir()
    meta = {
        "args": [{
            "name": "desc_q",
            "sig_type": "tensordesc<bf16[64, 128]>",
            "kind": "tensor_descriptor",
            "base_shape": list(base.shape),
            "base_dtype": tbg._dtype_str(base.dtype),
            "base_strides": list(base.stride()),
            "desc_shape": [512, 128],
            "desc_strides": [128, 1],
            "block_shape": [64, 128],
        }],
        "constexprs": {},
    }
    json_path = os.path.join(dump_dir, "_kernel_args.json")
    with open(json_path, "w") as f:
        json.dump(meta, f)

    with open(json_path) as f:
        loaded = json.load(f)

    entry = loaded["args"][0]
    assert entry["kind"] == "tensor_descriptor"
    assert entry["base_shape"] == [4, 128]
    assert entry["base_dtype"] == "bfloat16"
    assert entry["desc_shape"] == [512, 128]
    assert entry["desc_strides"] == [128, 1]
    assert entry["block_shape"] == [64, 128]


# ---------------------------------------------------------------------------
# capture_kernel_args — constexprs
# ---------------------------------------------------------------------------


def test_capture_kernel_args_constexprs(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    bound_args = OrderedDict([("x", 1.0), ("N", 1024), ("BLOCK_M", 256), ("FP8", False)])
    signature = {"x": "fp32", "N": "i32", "BLOCK_M": "constexpr", "FP8": "constexpr"}
    # constexprs maps (index,) -> value for constexpr params
    constexprs = {(2, ): 256, (3, ): False}

    capture_kernel_args(bound_args, signature, constexprs)

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    # x and N should be scalars, BLOCK_M and FP8 should be constexprs
    args = meta["args"]
    assert args[0]["kind"] == "scalar"
    assert args[1]["kind"] == "scalar"
    assert args[2]["kind"] == "constexpr"
    assert args[2]["name"] == "BLOCK_M"
    assert args[2]["value"] == 256
    assert args[3]["kind"] == "constexpr"
    assert args[3]["name"] == "FP8"
    assert args[3]["value"] is False

    # Top-level constexprs map should be populated
    assert meta["constexprs"]["BLOCK_M"] == 256
    assert meta["constexprs"]["FP8"] is False


# ---------------------------------------------------------------------------
# capture_grid
# ---------------------------------------------------------------------------


def test_capture_grid(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))
    # Write initial JSON
    with open(tmp_path / "_kernel_args.json", "w") as f:
        json.dump({"args": [], "constexprs": {}}, f)

    capture_grid((4, 192, 1))

    with open(tmp_path / "_kernel_args.json") as f:
        meta = json.load(f)

    assert meta["grid"] == [4, 192, 1]


def test_capture_grid_noop_without_dir(monkeypatch):
    monkeypatch.delenv("TRITON_TLX_DUMP_DIR", raising=False)
    # Should not raise
    capture_grid((1, 1, 1))


# ---------------------------------------------------------------------------
# generate_standalone_test — without source
# ---------------------------------------------------------------------------


def test_generate_standalone_test_no_source(tmp_path):
    """Test generation when no _source.py exists (TLX kernel only)."""
    kernel_name = "_my_kernel"
    meta = {
        "args": [
            {
                "name": "x_ptr", "kind": "tensor", "dtype": "float32", "shape": [1024], "strides": [1], "sig_type":
                "*fp32"
            },
            {"name": "N", "kind": "scalar", "scalar_type": "int", "value": 1024, "sig_type": "i32"},
            {"name": "BLOCK", "kind": "constexpr", "scalar_type": "int", "value": 256, "sig_type": "constexpr"},
        ],
        "constexprs": {"BLOCK": 256},
        "grid": [4, 1, 1],
    }
    with open(tmp_path / "_kernel_args.json", "w") as f:
        json.dump(meta, f)

    generate_standalone_test(str(tmp_path), kernel_name)

    test_path = tmp_path / "_test_standalone.py"
    assert test_path.exists()

    content = test_path.read_text()

    # Should import the kernel
    assert "from _my_kernel_kernel import _my_kernel" in content
    # Should have benchmark function
    assert "def benchmark():" in content
    # Should create tensors from JSON via dtype-aware helper
    assert "_make_tensor" in content
    # Should call do_bench
    assert "triton.testing.do_bench" in content
    # Should NOT have source module loading (no _source.py)
    assert "_load_source_module" not in content
    # Should NOT have source kernel benchmark section (no _load_source_module call)
    assert "src_kernel" not in content
    assert "ms_src" not in content
    # The generated script should be valid Python syntax
    compile(content, str(test_path), "exec")


# ---------------------------------------------------------------------------
# generate_standalone_test — with source
# ---------------------------------------------------------------------------


def test_generate_standalone_test_with_source(tmp_path):
    """Test generation when _source.py exists (both TLX and source kernel)."""
    kernel_name = "_attn_fwd"
    meta = {
        "args": [
            {"name": "sm_scale", "kind": "scalar", "scalar_type": "float", "value": 0.088, "sig_type": "fp32"},
            {
                "name": "M", "kind": "tensor", "dtype": "float32", "shape": [4, 48, 1024], "strides": [49152, 1024, 1],
                "sig_type": "*fp32"
            },
            {"name": "Z", "kind": "scalar", "scalar_type": "int", "value": 4, "sig_type": "i32"},
            {"name": "H", "kind": "scalar", "scalar_type": "int", "value": 48, "sig_type": "i32"},
            {
                "name": "desc_q", "kind": "tensor_descriptor", "base_shape": [4, 48, 1024, 128], "base_dtype":
                "bfloat16", "base_strides": [6291456, 131072, 128, 1], "desc_shape": [196608, 128], "desc_strides":
                [128, 1], "block_shape": [128, 128], "sig_type": "tensordesc<bf16[128,128]>"
            },
            {"name": "N_CTX", "kind": "scalar", "scalar_type": "int", "value": 1024, "sig_type": "i32"},
            {"name": "HEAD_DIM", "kind": "constexpr", "scalar_type": "int", "value": 128, "sig_type": "constexpr"},
            {"name": "BLOCK_M", "kind": "constexpr", "scalar_type": "int", "value": 256, "sig_type": "constexpr"},
            {"name": "FP8_OUTPUT", "kind": "constexpr", "scalar_type": "bool", "value": False, "sig_type": "constexpr"},
            {"name": "STAGE", "kind": "constexpr", "scalar_type": "int", "value": 1, "sig_type": "constexpr"},
        ],
        "constexprs": {
            "HEAD_DIM": 128,
            "BLOCK_M": 256,
            "FP8_OUTPUT": False,
            "STAGE": 1,
        },
        "grid": [4, 192, 1],
    }
    with open(tmp_path / "_kernel_args.json", "w") as f:
        json.dump(meta, f)

    # Create a dummy source file
    (tmp_path / "_attn_fwd_source.py").write_text("# dummy source\n")

    generate_standalone_test(str(tmp_path), kernel_name)

    test_path = tmp_path / "_test_standalone.py"
    assert test_path.exists()

    content = test_path.read_text()

    # Should import the kernel
    assert "from _attn_fwd_kernel import _attn_fwd" in content
    # Should have source module loading
    assert "_load_source_module" in content
    # Should have both TLX and source benchmarks
    assert "TLX kernel" in content
    assert "Source kernel" in content
    # Should compute TFLOPS from descriptor shapes
    assert "desc_base_shapes" in content
    assert "tflops_tlx" in content
    assert "tflops_src" in content
    # Should filter autotuner-managed constexprs for source kernel
    assert "BLOCK_" in content
    assert "user_kwargs" in content
    # Constexprs should NOT be passed to TLX kernel
    assert "*kernel_args)" in content
    # The generated script should be valid Python syntax
    compile(content, str(test_path), "exec")


# ---------------------------------------------------------------------------
# generate_standalone_test — missing JSON
# ---------------------------------------------------------------------------


def test_generate_standalone_test_missing_json(tmp_path):
    """generate_standalone_test should gracefully handle missing JSON."""
    generate_standalone_test(str(tmp_path), "_missing_kernel")
    # No test file should be created
    assert not (tmp_path / "_test_standalone.py").exists()


# ---------------------------------------------------------------------------
# E2E: capture_kernel_args + capture_grid + generate_standalone_test
# ---------------------------------------------------------------------------


def test_e2e_capture_and_generate(monkeypatch, tmp_path):
    """End-to-end test: capture args → capture grid → generate test."""
    monkeypatch.setenv("TRITON_TLX_DUMP_DIR", str(tmp_path))

    # Simulate the JIT capturing args for a kernel with mixed arg types
    t1 = torch.randn(4, 48, 1024, dtype=torch.float32)
    bound_args = OrderedDict([
        ("scale", 0.5),
        ("M", t1),
        ("batch", 4),
        ("heads", 48),
        ("BLOCK_SIZE", 256),
        ("USE_FP8", False),
    ])
    signature = {
        "scale": "fp32",
        "M": "*fp32",
        "batch": "i32",
        "heads": "i32",
        "BLOCK_SIZE": "constexpr",
        "USE_FP8": "constexpr",
    }
    constexprs = {(4, ): 256, (5, ): False}

    # Phase 1: capture args (happens before _do_compile in jit.py)
    capture_kernel_args(bound_args, signature, constexprs)
    json_path = tmp_path / "_kernel_args.json"
    assert json_path.exists()

    with open(json_path) as f:
        meta = json.load(f)
    assert len(meta["args"]) == 6
    assert "grid" not in meta  # grid not captured yet

    # Phase 2: capture grid (happens after grid evaluation in jit.py)
    capture_grid((4, 192, 1))
    with open(json_path) as f:
        meta = json.load(f)
    assert meta["grid"] == [4, 192, 1]

    # Phase 3: generate standalone test (happens in make_llir)
    kernel_name = "_my_kernel"
    generate_standalone_test(str(tmp_path), kernel_name)

    test_path = tmp_path / "_test_standalone.py"
    assert test_path.exists()

    content = test_path.read_text()
    # Verify the generated script is syntactically valid
    compile(content, str(test_path), "exec")
    # Verify it reads the JSON
    assert "_kernel_args.json" in content
    # Verify it creates the kernel call
    assert "_my_kernel[grid](*kernel_args)" in content


@pytest.fixture
def mxfp_collector(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    directory = Path(__file__).resolve().parents[4] / "third_party/tlx/tutorials/amd_mxfp_gemm_gfx1250"

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "bench", load("mxfp_bench", "bench.py"))
    return load("mxfp_collector", "collect_traces.py")


@pytest.fixture
def mxfp_sensors(mxfp_collector, tmp_path):
    device = tmp_path / "pci/0000:66:00.0"
    hwmon = device / "hwmon/hwmon42"
    hwmon.mkdir(parents=True)
    for name, value in {
            "name": "amdgpu", "power1_input": "400000000", "power1_cap": "750000000", "power1_label": "PPT",
            "freq1_input": "1900000000", "freq1_label": "sclk", "temp2_input": "57000", "temp2_label": "junction"
    }.items():
        (hwmon / name).write_text(value + "\n")
    (device / "power_dpm_force_performance_level").write_text("auto\n")
    return mxfp_collector.Sensors(device.name, pci_root=device.parent)


def test_mxfp_trace_cases_preserve_benchmark_defaults(mxfp_collector):
    cases = mxfp_collector.make_cases(mxfp_collector.parser().parse_args([]))
    assert [(case["kernel"], case["shape"][2]) for case in cases] == [("persistent", 8192), ("streamed_operands", 8192),
                                                                      ("persistent", 4096), ("streamed_operands", 4096)]
    baseline, streamed = cases[:2]
    assert streamed["application"] == baseline["application"][:baseline["application"].index("--num_programs")] + [
        "--streamed_operands"
    ] + baseline["application"][baseline["application"].index("--num_programs"):]
    for case in cases:
        config = case["config"]
        assert config["benchmark_num_iters"] == 256
        assert (config["block_m"], config["block_n"], config["block_k"]) == (256, 256, 128)
        assert config["num_programs"] == 256
        assert config["cluster_size"] == config["cluster_barrier_interval"] == 4


def test_mxfp_trace_device_selection_avoids_double_visibility(mxfp_collector, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "3,4")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    inherited = mxfp_collector.gpu_environment(None)
    assert inherited["ROCR_VISIBLE_DEVICES"] == "3,4"
    assert inherited["HIP_VISIBLE_DEVICES"] == "1"
    selected = mxfp_collector.gpu_environment("6")
    assert selected["ROCR_VISIBLE_DEVICES"] == "6"
    assert "HIP_VISIBLE_DEVICES" not in selected and "CUDA_VISIBLE_DEVICES" not in selected
    assert os.environ["ROCR_VISIBLE_DEVICES"] == "3,4"


def test_mxfp_trace_sensors_record_units_and_identity(mxfp_sensors):
    sample = mxfp_sensors.read()
    assert sample["power_w"] == 400
    assert sample["hwmon42_power1_cap_w"] == 750
    assert sample["hwmon42_freq1_input_mhz"] == 1900
    assert sample["hwmon42_temp2_input_c"] == 57
    assert sample["perf_level"] == "auto"
    assert not sample["errors"]
    assert sample["read_end_monotonic_ns"] >= sample["monotonic_ns"]
    assert mxfp_sensors.fields["hwmon42_freq1_input_mhz"]["label"] == "sclk"
    assert "0000:66:00.0" in mxfp_sensors.fields[mxfp_sensors.primary_power]["path"]


def test_mxfp_trace_power_falls_back_to_readable_average(mxfp_collector, mxfp_sensors):
    hwmon = mxfp_sensors.device / "hwmon/hwmon42"
    (hwmon / "power1_input").write_text("N/A\n")
    (hwmon / "power1_average").write_text("375000000\n")
    sensors = mxfp_collector.Sensors(mxfp_sensors.device.name, pci_root=mxfp_sensors.device.parent)
    assert sensors.read()["power_w"] == 375
    (hwmon / "power1_average").write_text("nan\n")
    with pytest.raises(mxfp_collector.CollectionError, match="cannot read device power"):
        mxfp_collector.Sensors(mxfp_sensors.device.name, pci_root=mxfp_sensors.device.parent)


@pytest.fixture
def mxfp_gpu_metrics_table():
    import struct

    def make(watts=2500, revision=8, kind=2, unit=3, attr_id=3, instances=1):
        if revision != 9:
            return struct.pack("<HBB4H", 12, 1, revision, 50, 60, 70, watts)
        # The v1.9 wire format packs each encoding immediately before its values.
        # Place array attributes on both sides to exercise variable offsets.
        before = struct.pack("<Q3H", (2 << 24) | (2 << 20) | 3, 50, 60, 70)
        power = struct.pack("<Q", (unit << 24) | (kind << 20) | (attr_id << 10) | instances)
        power += struct.pack("<" + "BbHhIiQq"[kind] * instances, *([watts] * instances))
        after = struct.pack("<Q2Q", (6 << 20) | (7 << 10) | 2, 123456789, 987654321)
        body = before + power + after
        return struct.pack("<HBBi", 8 + len(body), 1, revision, 3) + body

    return make


@pytest.mark.parametrize("revision", [4, 5, 6, 7, 8, 9])
def test_mxfp_trace_gpu_metrics_reads_socket_power(mxfp_collector, mxfp_gpu_metrics_table, revision):
    assert mxfp_collector.gpu_metrics_power(mxfp_gpu_metrics_table(revision=revision)) == (2500, f"1.{revision}")


@pytest.mark.parametrize("kind", [3, 4, 5, 6, 7])
def test_mxfp_trace_gpu_metrics_dynamic_power_types(mxfp_collector, mxfp_gpu_metrics_table, kind):
    assert mxfp_collector.gpu_metrics_power(mxfp_gpu_metrics_table(revision=9, kind=kind)) == (2500, "1.9")


@pytest.mark.parametrize("condition", [
    "header",
    "size",
    "version",
    "unavailable_fixed",
    "unavailable_dynamic",
    "unit",
    "instances",
    "missing_power",
    "negative",
    "count",
    "truncated_attribute",
    "trailing_bytes",
])
def test_mxfp_trace_gpu_metrics_rejects_invalid_tables(mxfp_collector, mxfp_gpu_metrics_table, condition):
    import struct

    fixed = mxfp_gpu_metrics_table()
    dynamic = mxfp_gpu_metrics_table(revision=9)
    tables = {
        "header": fixed[:3],
        "size": fixed[:-1],
        "version": mxfp_gpu_metrics_table(revision=10),
        "unavailable_fixed": mxfp_gpu_metrics_table(watts=0xffff),
        "unavailable_dynamic": mxfp_gpu_metrics_table(watts=0xffff, revision=9),
        "unit": mxfp_gpu_metrics_table(revision=9, unit=2),
        "instances": mxfp_gpu_metrics_table(revision=9, instances=2),
        "missing_power": mxfp_gpu_metrics_table(revision=9, attr_id=4),
        "negative": mxfp_gpu_metrics_table(revision=9, kind=3, watts=-2),
        "count": dynamic[:4] + struct.pack("<i", 1000) + dynamic[8:],
        "truncated_attribute": struct.pack("<H",
                                           len(dynamic) - 1) + dynamic[2:-1],
        "trailing_bytes": struct.pack("<H",
                                      len(dynamic) + 1) + dynamic[2:] + b"\0",
    }
    with pytest.raises(ValueError):
        mxfp_collector.gpu_metrics_power(tables[condition])


def test_mxfp_trace_gpu_metrics_survives_hwmon_failure_under_load(mxfp_collector, mxfp_sensors, mxfp_gpu_metrics_table,
                                                                  monkeypatch, tmp_path):
    from pathlib import Path

    metrics = mxfp_sensors.device / "gpu_metrics"
    metrics.write_bytes(mxfp_gpu_metrics_table(watts=400))
    sensors = mxfp_collector.Sensors(mxfp_sensors.device.name, pci_root=mxfp_sensors.device.parent)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        if path.name == "power1_input":
            raise OSError(0, "Error")
        return original_read(path, *args, **kwargs)

    # Idle probing succeeded; only the hwmon power path fails during measurement.
    monkeypatch.setattr(Path, "read_text", read)
    metrics.write_bytes(mxfp_gpu_metrics_table(watts=2500))
    samples = [sensors.read(), sensors.read()]
    for sample in samples:
        assert sample["power_w"] == 2500
        assert sample["power_source"] == "gpu_metrics_curr_socket_power_w"
        assert sample["gpu_metrics_version"] == "1.8"
        assert sample["errors"]["hwmon42_power1_input_w"] == "[Errno 0] Error"
    mxfp_collector.write_json(
        tmp_path / "workload.json",
        dict(phases=[
            dict(name="measure", start=dict(monotonic_ns=samples[0]["monotonic_ns"]), end=dict(
                monotonic_ns=samples[-1]["read_end_monotonic_ns"]))
        ]))
    (tmp_path / "telemetry.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    measured, = mxfp_collector.phase_summary(tmp_path, sensors)
    assert measured["metrics"]["power_w"] == dict(count=2, mean=2500, min=2500, max=2500)
    assert measured["power_sources"] == {"gpu_metrics_curr_socket_power_w": 2}
    diagnostics = json.loads((tmp_path / "telemetry_diagnostics.json").read_text())
    assert diagnostics["sensor_errors"] == {"hwmon42_power1_input_w: [Errno 0] Error": 2}
    assert diagnostics["gpu_metrics_versions"] == {"1.8": 2}


def test_mxfp_trace_power_fallback_tracks_source_without_stale_samples(mxfp_collector, mxfp_sensors,
                                                                       mxfp_gpu_metrics_table):
    metrics = mxfp_sensors.device / "gpu_metrics"
    metrics.write_bytes(mxfp_gpu_metrics_table(watts=410))
    hwmon = mxfp_sensors.device / "hwmon/hwmon42"
    (hwmon / "power1_average").write_text("375000000\n")
    sensors = mxfp_collector.Sensors(mxfp_sensors.device.name, pci_root=mxfp_sensors.device.parent)
    assert sensors.read()["power_source"] == "gpu_metrics_curr_socket_power_w"
    metrics.write_bytes(mxfp_gpu_metrics_table(watts=0xffff))
    sample = sensors.read()
    assert (sample["power_source"], sample["power_w"]) == ("hwmon42_power1_input_w", 400)
    (hwmon / "power1_input").write_text("N/A\n")
    sample = sensors.read()
    assert (sample["power_source"], sample["power_w"]) == ("hwmon42_power1_average_w", 375)
    (hwmon / "power1_average").write_text("N/A\n")
    sample = sensors.read()
    assert sample["power_source"] is None and sample["power_w"] is None
    assert sample["hwmon42_power1_cap_w"] == 750
    metrics.write_bytes(mxfp_gpu_metrics_table(watts=2600))
    sample = sensors.read()
    assert (sample["power_source"], sample["power_w"]) == ("gpu_metrics_curr_socket_power_w", 2600)


def test_mxfp_trace_gpu_metrics_power_without_hwmon(mxfp_collector, mxfp_gpu_metrics_table, tmp_path):
    device = tmp_path / "0000:66:00.0"
    device.mkdir()
    (device / "gpu_metrics").write_bytes(mxfp_gpu_metrics_table(revision=9))
    sensors = mxfp_collector.Sensors(device.name, pci_root=tmp_path)
    assert sensors.primary_power == "gpu_metrics_curr_socket_power_w"
    assert sensors.read()["power_w"] == 2500


def test_mxfp_trace_power_summary_excludes_setup_and_boundary_samples(mxfp_collector, mxfp_sensors, tmp_path):
    mxfp_collector.write_json(
        tmp_path / "workload.json",
        dict(phases=[dict(name="measure", start=dict(monotonic_ns=100), end=dict(monotonic_ns=300))]))
    samples = [
        dict(monotonic_ns=begin, read_end_monotonic_ns=end, power_w=power)
        for begin, end, power in [(0, 10, 50), (90, 110, 200), (120, 130, 400), (250, 260, 450), (280, 320,
                                                                                                  200), (350, 360, 50)]
    ]
    (tmp_path / "telemetry.jsonl").write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    summary = mxfp_collector.phase_summary(tmp_path, mxfp_sensors)[0]
    assert summary["samples"] == 2
    assert summary["metrics"]["power_w"] == dict(count=2, mean=425, min=400, max=450)


@pytest.mark.parametrize("values", [[], [None], [400]])
def test_mxfp_trace_missing_power_is_a_failure(mxfp_collector, mxfp_sensors, tmp_path, values):
    mxfp_collector.write_json(
        tmp_path / "workload.json",
        dict(phases=[dict(name="measure", start=dict(monotonic_ns=100), end=dict(monotonic_ns=300))]))
    (tmp_path / "telemetry.jsonl").write_text("".join(
        json.dumps(dict(monotonic_ns=150, read_end_monotonic_ns=160, power_w=value)) + "\n" for value in values))
    with pytest.raises(mxfp_collector.CollectionError, match="fewer than two valid power") as error:
        mxfp_collector.phase_summary(tmp_path, mxfp_sensors)
    summary = json.loads((tmp_path / "telemetry_summary.json").read_text())[0]
    diagnostics = json.loads((tmp_path / "telemetry_diagnostics.json").read_text())
    assert summary["samples"] == diagnostics["phases"][0]["complete_read_batches"] == len(values)
    assert diagnostics["phases"][0]["valid_power_samples"] == sum(value is not None for value in values)
    assert str(tmp_path / "telemetry_diagnostics.json") in str(error.value)


@pytest.mark.parametrize("readings,complete,overlap,errors", [
    ([(90, 110, 400), (280, 320, 410)], 0, 2, {}),
    ([(120, 130, None), (140, 150, None)], 2, 2, {"power: Input/output error": 2}),
    ([(0, 10, 400), (310, 320, 410)], 0, 0, {}),
    ([], 0, 0, {}),
])
def test_mxfp_trace_diagnostics_distinguish_sensor_errors_and_phase_coverage(mxfp_collector, tmp_path, readings,
                                                                             complete, overlap, errors):
    workload = dict(phases=[
        dict(name="measure", start=dict(monotonic_ns=100), end=dict(monotonic_ns=300)),
        dict(name="unfinished", start=dict(monotonic_ns=400))
    ])
    samples = [
        dict(monotonic_ns=begin, read_end_monotonic_ns=end, power_w=power,
             errors={"power": "Input/output error"} if power is None else {}) for begin, end, power in readings
    ]
    report = mxfp_collector.telemetry_diagnostics(tmp_path, workload, samples)
    assert report["sample_count"] == len(readings)
    assert report["samples_with_power"] == sum(power is not None for _, _, power in readings)
    assert report["sensor_errors"] == errors
    measure = report["phases"][0]
    assert measure["duration_seconds"] == pytest.approx(200 / 1e9)
    assert measure["complete_read_batches"] == complete
    assert measure["valid_power_samples"] == 0
    assert measure["overlapping_read_batches"] == overlap
    assert measure["overlapping_power_samples"] == (0 if errors else overlap)
    assert report["phases"][1] == dict(name="unfinished", status="unfinished")
    if readings:
        assert report["first_read_monotonic_ns"] == readings[0][0]
        assert report["last_read_monotonic_ns"] == readings[-1][1]
        assert report["read_duration_ms"]["max"] == pytest.approx(max(end - begin for begin, end, _ in readings) / 1e6)
    else:
        assert report["first_read_monotonic_ns"] is None
        assert report["last_read_monotonic_ns"] is None
        assert report["read_duration_ms"] is None


@pytest.mark.parametrize("inspect_stage", [False, True])
def test_mxfp_trace_inspection_preserves_existing_capture_without_device_access(mxfp_collector, monkeypatch, tmp_path,
                                                                                capsys, inspect_stage):
    stage = tmp_path / "case/power"
    stage.mkdir(parents=True)
    # Original captures predate telemetry_diagnostics.json and must remain readable.
    mxfp_collector.write_json(
        stage / "workload.json",
        dict(phases=[dict(name="measure", start=dict(monotonic_ns=100), end=dict(monotonic_ns=300))]))
    sample = dict(monotonic_ns=150, read_end_monotonic_ns=160, power_w=None,
                  errors={"hwmon42_power1_input_w": "Input/output error"})
    (stage / "telemetry.jsonl").write_text(json.dumps(sample) + "\n")

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

    before = snapshot()

    def unexpected_probe(*args, **kwargs):
        pytest.fail("inspection must not probe a device, load a profiler, or start a workload")

    for name in ("device_info", "command_record", "load_att_helper", "make_cases", "Sensors", "collect"):
        monkeypatch.setattr(mxfp_collector, name, unexpected_probe)
    assert mxfp_collector.main(["--inspect", str(stage if inspect_stage else tmp_path)]) == 0
    report, = json.loads(capsys.readouterr().out)
    assert report["directory"] == str(stage)
    assert report["sensor_errors"] == {"hwmon42_power1_input_w: Input/output error": 1}
    assert report["phases"][0]["complete_read_batches"] == 1
    assert report["phases"][0]["valid_power_samples"] == 0
    assert snapshot() == before


def test_mxfp_trace_existing_output_is_preserved(mxfp_collector, monkeypatch, tmp_path):
    previous = tmp_path / "previous.att"
    previous.write_bytes(b"previous trace")
    args = mxfp_collector.parser().parse_args(["--output", str(tmp_path)])

    def unexpected_probe(*args, **kwargs):
        pytest.fail("should reject existing output before probing or starting a workload")

    monkeypatch.setattr(mxfp_collector, "command_record", unexpected_probe)
    with pytest.raises(mxfp_collector.CollectionError, match="refusing to overwrite"):
        mxfp_collector.collect(args, [])
    assert previous.read_bytes() == b"previous trace"


def test_mxfp_trace_workload_numbers_only_target_launches(mxfp_collector, monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    actions = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    driver = SimpleNamespace(get_empty_cache_for_benchmark=lambda: "cache",
                             clear_cache=lambda _: actions.append("clear"))
    monkeypatch.setitem(sys.modules, "triton",
                        SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=driver))))
    monkeypatch.setattr(mxfp_collector, "device_info", lambda: dict(arch="gfx1250", bdf="0000:66:00.0"))
    spec = tmp_path / "spec.json"
    mxfp_collector.write_json(spec, dict(mode="att", device_bdf="0000:66:00.0", warmup_dispatches=3, benchmark_ms=256))

    def launch():
        actions.append("launch")
        return SimpleNamespace(name="test_kernel", metadata=SimpleNamespace(_asdict=lambda: {}),
                               asm=dict(amdgcn="test assembly", hsaco=b"compiled code"))

    mxfp_collector.profile_workload(launch, spec)
    assert actions == ["launch", "clear", "launch", "clear", "launch", "clear", "launch"]
    report = json.loads((tmp_path / "workload.json").read_text())
    assert report["selected_matching_dispatch"] == 4
    assert report["kernel_name"] == "test_kernel"
    assert [phase["name"] for phase in report["phases"]] == ["compile", "warmup", "selected_dispatch"]
    assert (tmp_path / "kernel.hsaco").read_bytes() == b"compiled code"


@pytest.mark.parametrize("filtered", [False, True])
def test_mxfp_trace_selected_dispatch_matches_decoded_wave(mxfp_collector, tmp_path, filtered):
    rows = "7,target,10,20\n" if filtered else "2,target,1,2\n4,other,2,3\n5,target,5,6\n7,target,10,20\n"
    (tmp_path / "trace_kernel_trace.csv").write_text("Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp\n" + rows)
    decoded = tmp_path / "ui_output_agent_0_dispatch_7"
    decoded.mkdir()
    result = mxfp_collector.check_dispatch(tmp_path, "target", 3)
    assert result["kernel_trace_row"]["dispatch_id"] == "7"
    decoded.rename(tmp_path / "ui_output_agent_0_dispatch_2")
    with pytest.raises(mxfp_collector.CollectionError, match="does not match"):
        mxfp_collector.check_dispatch(tmp_path, "target", 3)


def test_mxfp_trace_failed_process_retains_logs(mxfp_collector, mxfp_sensors, tmp_path):
    import sys

    with pytest.raises(mxfp_collector.CollectionError, match="workload exited 7"):
        mxfp_collector.run_recorded([sys.executable, "-c", "print('workload failed'); raise SystemExit(7)"], tmp_path,
                                    os.environ.copy(), mxfp_sensors, 0.01, 5)
    assert "workload failed" in (tmp_path / "stdout.log").read_text()
    result = json.loads((tmp_path / "process.json").read_text())
    assert result["returncode"] == 7
    assert result["end"]["monotonic_ns"] >= result["start"]["monotonic_ns"]
    assert (tmp_path / "telemetry.csv").is_file()


def test_mxfp_trace_timeout_reaps_child(mxfp_collector, mxfp_sensors, tmp_path):
    import subprocess
    import sys

    command = [
        sys.executable, "-c",
        "import os, signal; from pathlib import Path; Path('pid').write_text(str(os.getpid())); signal.pause()"
    ]
    with pytest.raises(subprocess.TimeoutExpired):
        mxfp_collector.run_recorded(command, tmp_path, os.environ.copy(), mxfp_sensors, 0.01, 0.3)
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert "end" in json.loads((tmp_path / "process.json").read_text())


@pytest.mark.parametrize("power_source", ["hwmon", "metrics_8", "metrics_9", "average"])
def test_mxfp_trace_collection_packages_timing_power_and_decoded_bundle(mxfp_collector, mxfp_sensors,
                                                                        mxfp_gpu_metrics_table, monkeypatch, tmp_path,
                                                                        power_source):
    import hashlib
    import sys
    import tarfile
    import textwrap
    from pathlib import Path

    failed_input = tmp_path / "input failed during warmup"
    hwmon = mxfp_sensors.device / "hwmon/hwmon42"
    if power_source.startswith("metrics_"):
        (mxfp_sensors.device / "gpu_metrics").write_bytes(
            mxfp_gpu_metrics_table(watts=425, revision=int(power_source[-1])))
        expected_source, expected_power = "gpu_metrics_curr_socket_power_w", 425
    elif power_source == "average":
        (hwmon / "power1_average").write_text("375000000\n")
        expected_source, expected_power = "hwmon42_power1_average_w", 375
    else:
        expected_source, expected_power = "hwmon42_power1_input_w", 400
    sensors = mxfp_collector.Sensors(mxfp_sensors.device.name, pci_root=mxfp_sensors.device.parent)
    read_text = Path.read_text

    def read(path, *args, **kwargs):
        if power_source != "hwmon" and path == hwmon / "power1_input" and failed_input.exists():
            raise OSError(0, "Error")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    # Run the collector's actual phase/warmup logic in external CPU processes.
    # GPU launches, benchmark timing, and decoded profiler output are fixtures.
    worker = tmp_path / "fixture worker.py"
    worker.write_text(
        textwrap.dedent('''\
        import importlib.util, json, sys, time
        from pathlib import Path
        from types import SimpleNamespace
        spec_path = Path(sys.argv[sys.argv.index('--profile_workload') + 1])
        spec = json.loads(spec_path.read_text())
        root = spec_path.parent
        module_spec = importlib.util.spec_from_file_location('collector', sys.argv[1])
        collector = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(collector)
        collector.device_info = lambda: dict(arch='gfx1250', bdf=spec['device_bdf'])
        sys.modules['torch'] = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        driver = SimpleNamespace(get_empty_cache_for_benchmark=lambda: None, clear_cache=lambda _: None)
        kernel_name = sys.argv[2]
        launch_count = 0

        def launch():
            global launch_count
            launch_count += 1
            return SimpleNamespace(name=kernel_name, metadata=SimpleNamespace(_asdict=lambda: {}),
                                   asm=dict(amdgcn='fixture assembly ' + kernel_name, hsaco=b'fixture code'))

        def do_bench(fn, warmup, rep):
            assert warmup == 30 and rep == spec['benchmark_ms'] == 256
            Path(sys.argv[3]).touch()
            fn()
            time.sleep(0.025)
            return 0.2

        sys.modules['triton'] = SimpleNamespace(runtime=SimpleNamespace(driver=SimpleNamespace(active=driver)),
                                               testing=SimpleNamespace(do_bench=do_bench))
        collector.profile_workload(launch, spec_path)
        (root / 'launch_count.json').write_text(json.dumps(launch_count))
    '''))
    profiler = tmp_path / "fixture profiler"
    profiler.write_text(
        textwrap.dedent('''\
        #!/usr/bin/env python3
        import json, sqlite3, subprocess, sys
        from pathlib import Path
        if '--help' in sys.argv:
            print('--att --rocm-root')
            raise SystemExit(0)
        subprocess.run(sys.argv[sys.argv.index('--') + 1:], check=True)
        root = Path(sys.argv[sys.argv.index('-d') + 1])
        kernel = sys.argv[sys.argv.index('--kernel-include-regex') + 1]
        selected, = json.loads(sys.argv[sys.argv.index('--kernel-iteration-range') + 1])
        spec_path = Path(sys.argv[sys.argv.index('--profile_workload') + 1])
        count = json.loads((spec_path.parent / 'launch_count.json').read_text())
        report = json.loads((spec_path.parent / 'workload.json').read_text())
        assert count == report['selected_matching_dispatch'] == selected == 1025
        dispatch_id = 2 * selected
        # Match rocprofv3's default: the database alone, without kernel CSVs.
        formats = ['rocpd']
        if '--output-format' in sys.argv:
            formats = []
            for value in sys.argv[sys.argv.index('--output-format') + 1:]:
                if value.startswith('-'):
                    break
                formats.append(value)
        if 'rocpd' in formats:
            with sqlite3.connect(root / 'fixture_results.db') as db:
                db.execute('CREATE TABLE fixture (value INTEGER)')
        (root / f'fixture_shader_engine_0_{dispatch_id}.att').write_bytes(b'fixture trace')
        (root / 'fixture_gfx1250_code_object_id_1.out').write_bytes(b'fixture code')
        (root / f'stats_ui_output_agent_0_dispatch_{dispatch_id}.csv').write_text('field\\nvalue\\n')
        rows = ''.join(f'{2*i+1},cache_clear\\n{2*i+2},{kernel}\\n' for i in range(selected))
        if 'csv' in formats:
            (root / 'fixture_kernel_trace.csv').write_text('Dispatch_Id,Kernel_Name\\n' + rows)
        ui = root / f'ui_output_agent_0_dispatch_{dispatch_id}'
        ui.mkdir()
        for name in ('code.json', 'filenames.json', 'occupancy.json', 'wstates0.json', 'se0_sm0_sl0_wv0.json'):
            (ui / name).write_text(json.dumps({'fixture': True}))
    '''))
    profiler.chmod(0o755)
    decoder = tmp_path / "fixture decoder"
    decoder.mkdir()
    (decoder / "librocprof-trace-decoder.so").touch()
    root = tmp_path / "collection with spaces"
    args = mxfp_collector.parser().parse_args([
        "--output",
        str(root), "--profiler",
        str(profiler), "--decoder-dir",
        str(decoder), "--sample-ms", "2", "--package", "--warmup-seconds", "0.075", "--duration-seconds", "0.075"
    ])
    cases = mxfp_collector.make_cases(args)
    assert len(cases) == 4
    for case in cases:
        case["application"] = [
            sys.executable,
            str(worker), mxfp_collector.__file__, mxfp_collector.KERNEL_NAMES[case["kernel"]],
            str(failed_input)
        ]
    monkeypatch.setattr(mxfp_collector, "Sensors", lambda bdf: sensors)

    def record(command, **kwargs):
        output = dict(bdf="0000:66:00.0", arch="gfx1250") if "--device-info" in command else {}
        return dict(command=command, returncode=0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(mxfp_collector, "command_record", record)
    mxfp_collector.collect(args, cases)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["power_sensor_priority"] == sensors.power_candidates
    assert len(manifest["cases"]) == 4
    for case in manifest["cases"]:
        measured = next(p for p in case["runs"]["power"]["telemetry"] if p["name"] == "measure")
        assert measured["metrics"]["power_w"]["mean"] == expected_power
        assert measured["power_sources"] == {expected_source: measured["samples"]}
        assert measured["samples"] >= 2
        workload = case["runs"]["power"]["workload"]
        assert [p["name"] for p in workload["phases"]] == ["compile", "warmup", "measure"]
        assert len(workload["benchmark_samples"]) >= 2
        assert workload["median_ms"] == 0.2
        assert case["runs"]["att"]["validation"]["valid"]
        assert case["runs"]["att"]["dispatch"]["kernel_trace_row"]["dispatch_id"] == "2050"
        assert case["runs"]["att"]["dispatch"]["matching_dispatch"] == 1025
        if power_source != "hwmon":
            diagnostics = json.loads((Path(case["root"]) / "power/telemetry_diagnostics.json").read_text())
            assert diagnostics["sensor_errors"]["hwmon42_power1_input_w: [Errno 0] Error"] > 0
    assert (root / "summary.csv").is_file()
    assert expected_source in (root / "summary.csv").read_text()
    archive = root.with_name(root.name + ".tar.gz")
    packaged = json.loads(root.with_name(root.name + ".package.json").read_text())
    assert packaged["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    assert sum(name.endswith("/power/telemetry.csv") for name in names) == 4
    assert sum(name.endswith("/att/trace/fixture_shader_engine_0_2050.att") for name in names) == 4
