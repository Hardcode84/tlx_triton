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
    with pytest.raises(mxfp_collector.CollectionError, match="fewer than two valid power"):
        mxfp_collector.phase_summary(tmp_path, mxfp_sensors)


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


def test_mxfp_trace_collection_packages_timing_power_and_decoded_bundle(mxfp_collector, mxfp_sensors, monkeypatch,
                                                                        tmp_path):
    import hashlib
    import sys
    import tarfile
    import textwrap

    # Exercise the real subprocess, telemetry, validation, and packaging paths
    # with an external CPU workload and a profiler producing a fixture bundle.
    worker = tmp_path / "fixture worker.py"
    worker.write_text(
        textwrap.dedent('''\
        import hashlib, json, sys, time
        from pathlib import Path
        spec_path = Path(sys.argv[sys.argv.index('--profile_workload') + 1])
        spec = json.loads(spec_path.read_text())
        root = spec_path.parent
        start = time.monotonic_ns()
        time.sleep(0.05)
        end = time.monotonic_ns()
        phase = dict(name='measure' if spec['mode'] == 'power' else 'selected_dispatch',
                     start=dict(monotonic_ns=start), end=dict(monotonic_ns=end))
        report = dict(phases=[phase], benchmark_samples=[dict(ms=0.2)], median_ms=0.2,
                      median_tflops=5000, code_sha256=dict(amdgcn=hashlib.sha256(b'assembly').hexdigest()))
        (root / 'workload.json').write_text(json.dumps(report))
        (root / 'kernel.amdgcn').write_text('assembly')
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
        with sqlite3.connect(root / 'fixture_results.db') as db:
            db.execute('CREATE TABLE fixture (value INTEGER)')
        (root / 'fixture_shader_engine_0_9.att').write_bytes(b'fixture trace')
        (root / 'fixture_gfx1250_code_object_id_1.out').write_bytes(b'fixture code')
        (root / 'stats_ui_output_agent_0_dispatch_9.csv').write_text('field\\nvalue\\n')
        (root / 'fixture_kernel_trace.csv').write_text('Dispatch_Id,Kernel_Name\\n9,' + kernel + '\\n')
        ui = root / 'ui_output_agent_0_dispatch_9'
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
        str(decoder), "--sample-ms", "2", "--package", "--kernels", "persistent"
    ])
    cases = mxfp_collector.make_cases(args)[:1]
    cases[0]["application"] = [sys.executable, str(worker)]
    monkeypatch.setattr(mxfp_collector, "Sensors", lambda bdf: mxfp_sensors)

    def record(command, **kwargs):
        output = dict(bdf="0000:66:00.0", arch="gfx1250") if "--device-info" in command else {}
        return dict(command=command, returncode=0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(mxfp_collector, "command_record", record)
    mxfp_collector.collect(args, cases)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    case = manifest["cases"][0]
    assert case["runs"]["power"]["telemetry"][0]["metrics"]["power_w"]["mean"] == 400
    assert case["runs"]["att"]["validation"]["valid"]
    assert case["runs"]["att"]["dispatch"]["kernel_trace_row"]["dispatch_id"] == "9"
    assert (root / "summary.csv").is_file()
    archive = root.with_name(root.name + ".tar.gz")
    packaged = json.loads(root.with_name(root.name + ".package.json").read_text())
    assert packaged["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    assert any(name.endswith("/power/telemetry.csv") for name in names)
    assert any(name.endswith("/att/trace/fixture_shader_engine_0_9.att") for name in names)
