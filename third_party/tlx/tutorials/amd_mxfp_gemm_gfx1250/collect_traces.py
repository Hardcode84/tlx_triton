"""Collect real-device MXFP instruction traces, benchmark timings, and power.

By default compare persistent and streamed A8W8 at both bench.py shapes.
Options after -- are passed to bench.py's argument resolver. Run under the
machine's GPU lock; this script reads device controls but never changes them.
"""

import argparse
import contextlib
import csv
import ctypes
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ATT_SCRIPT = HERE.parent.parent / "tools/agents/skills/amd-att-trace/scripts/att_trace.py"
VISIBILITY_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
KERNEL_NAMES = {"persistent": "mxgemm_tdm_persistent_kernel", "streamed_operands": "mxgemm_tdm_streamed_kernel"}


class CollectionError(RuntimeError):
    pass


def stamp():
    return dict(unix_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                monotonic_raw_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")


def gpu_environment(gpu):
    environment = os.environ.copy()
    if gpu is not None:
        # HIP indices are relative to ROCR visibility, not physical indices.
        for name in VISIBILITY_VARS:
            environment.pop(name, None)
        environment["ROCR_VISIBLE_DEVICES"] = gpu
    return environment


def device_info():
    import torch
    import triton

    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        raise CollectionError(f"expected a HIP device, got {target}")
    # Use the HIP library already loaded by PyTorch. Resolving an unversioned
    # soname again can load a different installation from LD_LIBRARY_PATH.
    hip_paths = {
        parts[5].strip()
        for line in Path("/proc/self/maps").read_text().splitlines()
        if len(parts := line.split(maxsplit=5)) == 6 and re.fullmatch(r"libamdhip64\.so(?:\.\d+)*",
                                                                      Path(parts[5].strip()).name)
    }
    if len(hip_paths) != 1:
        raise CollectionError(f"expected one loaded HIP runtime, found {sorted(hip_paths)}")
    hip_path = hip_paths.pop()
    hip = ctypes.CDLL(hip_path, mode=os.RTLD_NOLOAD)
    hip.hipDeviceGetPCIBusId.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    bdf = ctypes.create_string_buffer(64)
    status = hip.hipDeviceGetPCIBusId(bdf, len(bdf), 0)
    if status:
        raise CollectionError(f"hipDeviceGetPCIBusId failed: {status}")
    domain, bus, function = bdf.value.decode().lower().split(":")
    pci_address = f"{int(domain, 16):04x}:{bus}:{function}"
    pci_device = Path("/sys/bus/pci/devices") / pci_address
    if not pci_device.is_dir():
        raise CollectionError(f"HIP device {pci_address} is absent from host PCI devices; "
                              "use a real-device ROCm runtime with host sysfs access")
    return dict(bdf=pci_address, name=properties.name, arch=properties.gcnArchName, target=str(target),
                compute_units=properties.multi_processor_count, total_memory=properties.total_memory,
                torch=torch.__version__, hip=torch.version.hip, triton=triton.__version__, triton_path=triton.__file__,
                hip_library=hip_path, python=sys.executable, timestamp=stamp())


def command_record(command, *, environment=None, timeout=30):
    start = stamp()
    try:
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout, check=False)
        return dict(command=command, start=start, end=stamp(), returncode=result.returncode, stdout=result.stdout,
                    stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(command=command, start=start, end=stamp(), returncode=None, error=str(error))


def load_att_helper():
    spec = importlib.util.spec_from_file_location("mxfp_att_trace", ATT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Sensors:
    """Read the physical device selected by HIP, using documented hwmon units."""

    def __init__(self, bdf, pci_root=Path("/sys/bus/pci/devices")):
        if not re.fullmatch(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf):
            raise CollectionError(f"invalid HIP PCI address: {bdf!r}")
        self.device = pci_root / bdf
        self.fields = {}
        self.primary_power = None
        for hwmon in sorted(self.device.glob("hwmon/hwmon*")):
            if (hwmon / "name").read_text().strip() != "amdgpu":
                continue
            for path in sorted(hwmon.iterdir()):
                match = re.fullmatch(r"(power\d+)_(input|average|cap)", path.name)
                if match:
                    unit, divisor, label = "w", 1e6, match[1]
                elif re.fullmatch(r"temp\d+_input", path.name):
                    unit, divisor, label = "c", 1e3, path.name.split("_")[0]
                elif re.fullmatch(r"freq\d+_input", path.name):
                    unit, divisor, label = "mhz", 1e6, path.name.split("_")[0]
                elif re.fullmatch(r"energy\d+_input", path.name):
                    unit, divisor, label = "j", 1e6, path.name.split("_")[0]
                else:
                    continue
                field = f"{hwmon.name}_{path.name}_{unit}"
                label_path = hwmon / (label + "_label")
                self.fields[field] = dict(path=str(path), divisor=divisor, unit=unit,
                                          label=label_path.read_text().strip() if label_path.exists() else label)
                if path.name == "power1_input" or (path.name == "power1_average" and self.primary_power is None):
                    self.primary_power = field
        for name in ("gpu_busy_percent", "mem_busy_percent"):
            path = self.device / name
            if path.exists():
                self.fields[name] = dict(path=str(path), divisor=1, unit="percent", label=name)
        if self.primary_power is None:
            raise CollectionError(f"no hwmon power1_input or power1_average for {bdf}; device power is required")
        sample = self.read()
        candidates = sorted((name for name in self.fields if "_power1_input_" in name or "_power1_average_" in name),
                            key=lambda name: "_input_" not in name)
        self.primary_power = next((name for name in candidates if sample[name] is not None), None)
        if self.primary_power is None:
            raise CollectionError(f"cannot read device power for {bdf}: {sample['errors']}")

    def read(self):
        sample = stamp()
        sample["errors"] = {}
        for field, source in self.fields.items():
            try:
                value = float(Path(source["path"]).read_text().strip()) / source["divisor"]
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid sensor value {value}")
                sample[field] = value
            except (OSError, ValueError) as error:
                sample[field] = None
                sample["errors"][field] = str(error)
        sample["power_w"] = sample[self.primary_power]
        try:
            sample["perf_level"] = (self.device / "power_dpm_force_performance_level").read_text().strip()
        except OSError as error:
            sample["perf_level"] = None
            sample["errors"]["perf_level"] = str(error)
        sample["read_end_monotonic_ns"] = time.monotonic_ns()
        return sample

    def controls(self):
        result = {}
        for name in ("power_dpm_force_performance_level", "pp_dpm_sclk", "pp_dpm_mclk", "pp_power_profile_mode",
                     "current_compute_partition", "current_memory_partition"):
            try:
                result[name] = (self.device / name).read_text().strip()
            except OSError as error:
                result[name] = dict(error=str(error))
        return result


class Sampler:

    def __init__(self, sensors, output, interval):
        self.sensors, self.output, self.interval = sensors, output, interval
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        try:
            with (self.output / "telemetry.jsonl").open("x") as raw, (self.output / "telemetry.csv").open(
                    "x", newline="") as csv_file:
                fields = [
                    "unix_ns", "monotonic_ns", "monotonic_raw_ns", "read_end_monotonic_ns", "power_w",
                    *self.sensors.fields, "perf_level", "errors"
                ]
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    sample = self.sensors.read()
                    raw.write(json.dumps(sample) + "\n")
                    writer.writerow({**sample, "errors": json.dumps(sample["errors"])})
                    raw.flush()
                    csv_file.flush()
                    self.stop_event.wait(max(0, self.interval - (time.monotonic() - started)))
        except Exception as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exception):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if not exception[0]:
            if self.thread.is_alive():
                raise CollectionError("device telemetry reader did not stop")
            if self.error:
                raise CollectionError(f"device telemetry failed: {self.error}")


def run_recorded(command, directory, environment, sensors, interval, timeout):
    result = dict(command=command, command_display=shlex.join(command), start=stamp(), cwd=str(directory))
    write_json(directory / "process.json", result)
    process = None
    try:
        with Sampler(sensors, directory, interval), (directory / "stdout.log").open("x") as log:
            process = subprocess.Popen(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            result["returncode"] = process.wait(timeout=timeout)
    finally:
        if process is not None and process.poll() is None:
            # Only terminate the process group created for this collection.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        result["end"] = stamp()
        write_json(directory / "process.json", result)
    if result["returncode"]:
        raise CollectionError(f"workload exited {result['returncode']}; see {directory / 'stdout.log'}")


def phase_summary(directory, sensors):
    workload = json.loads((directory / "workload.json").read_text())
    samples = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    summaries = []
    for phase in workload["phases"]:
        begin, end = phase["start"]["monotonic_ns"], phase["end"]["monotonic_ns"]
        selected = [s for s in samples if begin <= s["monotonic_ns"] and s["read_end_monotonic_ns"] <= end]
        summary = dict(name=phase["name"], seconds=(end - begin) / 1e9, samples=len(selected), metrics={})
        for field in ("power_w", *sensors.fields):
            values = [s[field] for s in selected if s.get(field) is not None]
            summary["metrics"][field] = (dict(count=len(values), mean=statistics.mean(values), min=min(values),
                                              max=max(values)) if values else None)
        summaries.append(summary)
        if phase["name"] == "measure" and (summary["metrics"]["power_w"] is None
                                           or summary["metrics"]["power_w"]["count"] < 2):
            raise CollectionError(f"fewer than two valid power samples during measurement: {directory}")
    write_json(directory / "telemetry_summary.json", summaries)
    return summaries


def profile_workload(launch, spec_path):
    """Called by the tutorial after allocating its ordinary benchmark inputs."""
    import torch
    import triton

    spec_path = Path(spec_path)
    spec = json.loads(spec_path.read_text())
    root = spec_path.parent
    info = device_info()
    if info["arch"].split(":")[0] != "gfx1250" or info["bdf"] != spec["device_bdf"]:
        raise CollectionError(f"workload device differs from selected gfx1250 device: {info}")
    report = dict(device=info, phases=[], benchmark_samples=[], cache_policy="bench.py eager: flush before timing",
                  timing_budget_ms=spec["benchmark_ms"], correctness_checked=False)

    @contextlib.contextmanager
    def phase(name):
        torch.cuda.synchronize()
        entry = dict(name=name, start=stamp())
        report["phases"].append(entry)
        write_json(root / "workload.json", report)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            entry["end"] = stamp()
            write_json(root / "workload.json", report)

    with phase("compile"):
        kernel = launch()  # Matching dispatch 1, also when compilation is cached.
    report["kernel_name"] = kernel.name
    report["kernel_metadata"] = kernel.metadata._asdict()
    report["code_sha256"] = {}
    for kind in ("amdgcn", "hsaco", "llir", "ttgir"):
        if kind in kernel.asm:
            data = kernel.asm[kind]
            data = data.encode() if isinstance(data, str) else data
            (root / f"kernel.{kind}").write_bytes(data)
            report["code_sha256"][kind] = hashlib.sha256(data).hexdigest()

    if spec["mode"] == "att":
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
        with phase("warmup"):
            for _ in range(spec["warmup_dispatches"] - 1):
                triton.runtime.driver.active.clear_cache(cache)
                launch()
        report["selected_matching_dispatch"] = spec["warmup_dispatches"] + 1
        with phase("selected_dispatch"):
            triton.runtime.driver.active.clear_cache(cache)
            launch()
    else:
        for name, duration in (("warmup", spec["warmup_seconds"]), ("measure", spec["duration_seconds"])):
            with phase(name):
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    start = stamp()
                    ms = triton.testing.do_bench(launch, warmup=30, rep=spec["benchmark_ms"])
                    if name == "measure":
                        report["benchmark_samples"].append(dict(start=start, end=stamp(), ms=ms))
                        print(f"execution time: {ms:.8f} ms", flush=True)
        values = [sample["ms"] for sample in report["benchmark_samples"]]
        report["median_ms"] = statistics.median(values)
        report["median_tflops"] = 2 * math.prod(spec["shape"]) / report["median_ms"] / 1e9
    write_json(root / "workload.json", report)


def make_cases(args):
    if __package__:
        from . import bench
    else:
        import bench

    forwarded = args.benchmark_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if not any(arg.split("=")[0] in ("--variant", "--dtype-b") for arg in forwarded):
        forwarded = ["--variant", "mx8xmx8", *forwarded]
    kernel_options = {
        "--streamed-operands", "--first-use-prefetch", "--output-tail-reuse", "--warp-pipeline", "--operand-pipeline",
        "--register-pipeline", "--no-persistent"
    }
    if any(arg.split("=")[0] in kernel_options for arg in forwarded):
        raise CollectionError("select the profiled schedule using --kernels before --")
    cases = []
    for kernel in args.kernels:
        options = [*forwarded, *(["--streamed-operands"] if kernel == "streamed_operands" else [])]
        parsed, shapes, variants = bench.parse_benchmark_args(options)
        if parsed.benchmark_mode != "eager" or parsed.csv or parsed.output_dir or parsed.dry_run:
            raise CollectionError("collection requires eager timing; use the collector's output and dry-run options")
        for shape in shapes:
            for variant, dtype_b, config in variants:
                cases.append(
                    dict(kernel=kernel, variant=variant, shape=shape, config=vars(config),
                         application=bench._command(config, shape, dtype_b)))
    # Keep each baseline beside its candidate, rather than sweeping one kernel first.
    return sorted(cases, key=lambda case: (shapes.index(case["shape"]), case["variant"]))


def att_command(args, capability, root, case, application):
    return [
        sys.executable,
        str(ATT_SCRIPT), "collect", "--output",
        str(root / "trace"), "--name", "mxfp", "--profiler", capability["profiler"], "--decoder-dir",
        capability["decoder_directory"], "--cwd",
        str(root), "--kernel-regex", case["kernel_regex"], "--dispatch",
        str(args.warmup_dispatches + 1), "--target-cu",
        str(args.target_cu), "--shader-engine-mask", args.shader_engine_mask, "--simd-select", args.simd_select,
        "--activity",
        str(args.activity), "--", *application
    ]


def check_dispatch(trace_root, kernel_regex, selected):
    rows = []
    for path in trace_root.rglob("*kernel_trace.csv"):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns = {key: key.strip().lower().replace(" ", "_") for key in (reader.fieldnames or [])}
            if not {"kernel_name", "dispatch_id"}.issubset(columns.values()):
                raise CollectionError(f"unrecognized kernel trace columns in {path}: {reader.fieldnames}")
            for row in reader:
                row = {columns[key]: value for key, value in row.items() if key in columns}
                if re.search(kernel_regex, row.get("kernel_name", "")):
                    rows.append(row)
    rows.sort(key=lambda row: int(row["dispatch_id"]))
    if len(rows) == 1:
        expected = rows[0]
        selection = "profiler filtered kernel trace to the selected matching dispatch"
    elif len(rows) == selected:
        expected = rows[selected - 1]
        selection = "one-based matching dispatch verified against all warmup dispatches"
    else:
        raise CollectionError(f"cannot verify matching dispatch {selected}: found {len(rows)} kernel trace rows")
    decoded = list(trace_root.rglob("ui_output_agent_*_dispatch_*"))
    if not decoded or any(int(path.name.rsplit("_", 1)[1]) != int(expected["dispatch_id"]) for path in decoded):
        raise CollectionError("decoded ATT dispatch does not match the selected kernel trace row")
    return dict(selection=selection, matching_dispatch=selected, kernel_trace_row=expected)


def write_summary(root, cases):
    rows = []
    for case in cases:
        if "power" not in case["runs"]:
            continue
        workload = case["runs"]["power"]["workload"]
        measured = next(phase for phase in case["runs"]["power"]["telemetry"] if phase["name"] == "measure")
        power = measured["metrics"]["power_w"]
        m, n, k = case["shape"]
        rows.append(
            dict(kernel=case["kernel"], variant=case["variant"], M=m, N=n, K=k, median_ms=workload["median_ms"],
                 median_tflops=workload["median_tflops"], benchmark_batches=len(workload["benchmark_samples"]),
                 measurement_seconds=measured["seconds"], power_sample_mean_w=power["mean"], power_min_w=power["min"],
                 power_max_w=power["max"], power_samples=power["count"], run_dir=case["root"]))
    if rows:
        with (root / "summary.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def collect(args, cases):
    environment = gpu_environment(args.gpu)
    root = args.output.expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise CollectionError(f"refusing to overwrite non-empty output: {root}")
    probe = command_record([sys.executable, str(Path(__file__).resolve()), "--device-info"], environment=environment)
    if probe["returncode"] != 0:
        raise CollectionError(f"device probe failed: {probe}")
    device = json.loads(probe["stdout"])
    if device["arch"].split(":")[0] != "gfx1250":
        raise CollectionError(f"selected device is {device['arch']} at {device['bdf']}; gfx1250 is required")
    sensors = Sensors(device["bdf"])
    initial = sensors.read()
    if initial.get("gpu_busy_percent", 0) is not None and initial.get("gpu_busy_percent", 0) > 5:
        raise CollectionError("selected GPU is busy; acquire the machine's GPU lock and select an idle device")
    helper = load_att_helper() if args.att else None
    capability = helper.probe(args.profiler, args.decoder_dir) if helper else None
    if args.profiler and capability and Path(args.profiler).resolve() != Path(capability["profiler"]):
        raise CollectionError(f"requested profiler is not usable: {args.profiler}")
    if args.decoder_dir and capability and Path(args.decoder_dir).resolve() != Path(capability["decoder_directory"]):
        raise CollectionError(f"requested decoder is not usable: {args.decoder_dir}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        status="running", start=stamp(), hostname=socket.gethostname(), device=device, capability=capability,
        collector_command=sys.argv, parameters=vars(args), sensors=sensors.fields,
        primary_power_sensor=sensors.primary_power, controls_before=sensors.controls(), environment={
            key: environment[key]
            for key in (*VISIBILITY_VARS, "HSA_OVERRIDE_GFX_VERSION", "LD_LIBRARY_PATH", "LD_PRELOAD", "LLVM_SYSPATH",
                        "TRITON_AMDGCN_ASSEMBLER_PATH")
            if key in environment
        }, cases=[])
    write_json(root / "manifest.json", manifest)
    try:
        source_root = root / "source"
        source_root.mkdir()
        for source in HERE.glob("*.py"):
            shutil.copy2(source, source_root / source.name)
        if args.att:
            shutil.copy2(ATT_SCRIPT, source_root / ATT_SCRIPT.name)
        manifest["git"] = command_record(["git", "-C", str(HERE), "rev-parse", "HEAD"])
        manifest["git_status"] = command_record(["git", "-C", str(HERE), "status", "--short"])
        smi = shutil.which("amd-smi")
        if smi:
            for kind in ("static", "metric", "process"):
                write_json(root / f"device_{kind}_before.json",
                           command_record([smi, kind, "--gpu", device["bdf"], "--json"]))
        for index, case in enumerate(cases, 1):
            label = f"{index:02d}-{case['kernel']}-{case['variant']}-{'x'.join(map(str, case['shape']))}"
            print(f"[{index}/{len(cases)}] {label}", flush=True)
            case_root = root / label
            case_root.mkdir()
            case = {**case, "root": str(case_root), "kernel_regex": KERNEL_NAMES[case["kernel"]], "runs": {}}
            manifest["cases"].append(case)
            for mode in (("power", "att") if args.att else ("power", )):
                stage = case_root / mode
                stage.mkdir()
                spec = dict(mode=mode, shape=case["shape"], device_bdf=device["bdf"],
                            benchmark_ms=case["config"]["benchmark_num_iters"], warmup_seconds=args.warmup_seconds,
                            duration_seconds=args.duration_seconds, warmup_dispatches=args.warmup_dispatches)
                write_json(stage / "workload_spec.json", spec)
                application = [*case["application"], "--profile_workload", str(stage / "workload_spec.json")]
                command = att_command(args, capability, stage, case, application) if mode == "att" else application
                write_json(root / "manifest.json", manifest)
                if smi:
                    write_json(stage / "device_metric_before.json",
                               command_record([smi, "metric", "--gpu", device["bdf"], "--json"]))
                run_recorded(command, stage, environment, sensors, args.sample_ms / 1000, args.timeout)
                if smi:
                    write_json(stage / "device_metric_after.json",
                               command_record([smi, "metric", "--gpu", device["bdf"], "--json"]))
                case["runs"][mode] = dict(telemetry=phase_summary(stage, sensors), workload=json.loads(
                    (stage / "workload.json").read_text()))
                if mode == "att":
                    case["runs"][mode]["validation"] = helper.validate_bundle(stage / "trace")
                    case["runs"][mode]["dispatch"] = check_dispatch(stage / "trace", case["kernel_regex"],
                                                                    args.warmup_dispatches + 1)
                    if (case["runs"]["power"]["workload"]["code_sha256"]["amdgcn"]
                            != case["runs"]["att"]["workload"]["code_sha256"]["amdgcn"]):
                        raise CollectionError("power and ATT runs compiled different kernel assembly")
                write_json(root / "manifest.json", manifest)
            print(f"  {case['runs']['power']['workload']['median_ms']:.6f} ms; {case_root}", flush=True)
            write_summary(root, manifest["cases"])
        if smi:
            write_json(root / "device_metric_after.json",
                       command_record([smi, "metric", "--gpu", device["bdf"], "--json"]))
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"], manifest["error"] = "failed", str(error)
        raise
    finally:
        manifest["end"], manifest["controls_after"] = stamp(), sensors.controls()
        write_json(root / "manifest.json", manifest)
    if args.package:
        packaged = helper.package_bundle(root, None)
        write_json(root.with_name(root.name + ".package.json"), packaged)
        print(f"Archive: {packaged['archive']}\nSHA-256: {packaged['sha256']}")
    print(f"Collection: {root}")


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--output", type=Path, default=Path("mxfp-hw-" + time.strftime("%Y%m%d-%H%M%S")))
    result.add_argument("--gpu", help="physical ROCr index or GPU UUID; otherwise preserve existing visibility")
    result.add_argument("--kernels", nargs="+", choices=tuple(KERNEL_NAMES), default=list(KERNEL_NAMES))
    result.add_argument("--warmup-seconds", type=float, default=10,
                        help="repeat the eager benchmark before measurement")
    result.add_argument("--duration-seconds", type=float, default=5,
                        help="repeat 256 ms benchmark batches for this long")
    result.add_argument("--warmup-dispatches", type=int, default=1024,
                        help="ATT warmup launch count, including the first compiling launch; trace the next launch")
    result.add_argument("--sample-ms", type=float, default=100, help="hwmon polling interval; not sensor update rate")
    result.add_argument("--att", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--profiler", help="ATT-capable rocprofv3 executable from the target's ROCm installation")
    result.add_argument("--decoder-dir", help="matching directory containing librocprof-trace-decoder.so")
    result.add_argument("--target-cu", type=int, default=0)
    result.add_argument("--shader-engine-mask", default="0x1")
    result.add_argument("--simd-select", default="0xF")
    result.add_argument("--activity", type=int, default=0, help="ATT activity setting; 0 uses profiler defaults")
    result.add_argument("--timeout", type=float, default=600, help="maximum seconds per subprocess")
    result.add_argument("--package", action="store_true",
                        help="validate and archive the entire collection with SHA-256")
    result.add_argument("--dry-run", action="store_true", help="show cases and selection without accessing a GPU")
    result.add_argument("--device-info", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("benchmark_args", nargs=argparse.REMAINDER, help="bench.py options after --")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.device_info:
            print(json.dumps(device_info()))
            return 0
        if (not all(
                math.isfinite(value) and value > 0
                for value in (args.warmup_seconds, args.duration_seconds, args.sample_ms, args.timeout))
                or args.warmup_dispatches < 1 or args.target_cu < 0 or args.activity < 0):
            cli.error("durations, sampling interval and warmup count must be positive; CU/activity must be nonnegative")
        if args.package and not args.att:
            cli.error("--package requires ATT bundle validation")
        cases = make_cases(args)
        if args.att:
            print(f"ATT selection: matching dispatch {args.warmup_dispatches + 1} (one-based), "
                  f"after {args.warmup_dispatches} warmup launches.")
        if args.dry_run:
            print(json.dumps(dict(output=str(args.output), cases=cases), indent=2, default=str))
            return 0
        collect(args, cases)
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("collection interrupted; partial artifacts were retained", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(main())
