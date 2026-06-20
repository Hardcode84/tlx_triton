"""Wave tool discovery and verification helpers shared by bridge paths."""

import os
import subprocess
import sys
from pathlib import Path


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _cmake_build_dirs():
    try:
        from build_helpers import get_cmake_dir

        yield Path(get_cmake_dir())
    except Exception:
        pass

    build_root = _repo_root() / "build"
    if build_root.is_dir():
        yield from sorted(path for path in build_root.glob("cmake.*") if path.is_dir())


def _wave_build_dirs():
    for build_dir in _cmake_build_dirs():
        yield build_dir
        yield build_dir / "third_party" / "tlx_wave" / "wave"
        yield build_dir / "third_party" / "wave"
    yield _repo_root() / "third_party" / "wave" / "build" / "wave-build"
    yield _repo_root() / "third_party" / "wave" / "build"


def _candidate_wave_python_paths():
    override = os.environ.get("TRITON_WAVE_PYTHONPATH")
    if override:
        for entry in override.split(os.pathsep):
            if entry:
                yield Path(entry)

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "python_packages" / "wave_mlir"


def _candidate_wave_tool_paths(tool_name, override=None):
    if override:
        yield Path(override)

    tools_dir = os.environ.get("TRITON_WAVE_TOOLS_DIR")
    if tools_dir:
        yield Path(tools_dir) / tool_name

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "bin" / tool_name


def _candidate_wave_opt_paths():
    yield from _candidate_wave_tool_paths(
        "wave-opt",
        override=os.environ.get("TRITON_WAVE_OPT"),
    )


def _existing_paths(candidates):
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            yield path


def _load_wave_dsl():
    for path in _existing_paths(_candidate_wave_python_paths()):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        from mlir.dialects import wave_dsl as w
    except Exception as exc:
        candidates = "\n  ".join(str(path) for path in _candidate_wave_python_paths())
        raise RuntimeError(
            "tlx_wave requires Wave MLIR Python bindings from the third_party/wave submodule build. "
            "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave, or run "
            "`python third_party/wave/build_tools/build_llvm.py --python-bindings` followed by "
            "`cmake -S third_party/wave -B third_party/wave/build/wave-build "
            "-G Ninja -DWAVE_ENABLE_PYTHON_BINDINGS=ON` and "
            "`cmake --build third_party/wave/build/wave-build`. "
            f"Unable to import mlir.dialects.wave_dsl: {type(exc).__name__}: {exc}. "
            f"Checked Wave Python package candidates:\n  {candidates}"
        ) from exc
    return w


def _wave_tool(tool_name, override_env=None):
    override = os.environ.get(override_env) if override_env else None
    for path in _existing_paths(_candidate_wave_tool_paths(tool_name, override)):
        if os.access(path, os.X_OK):
            return str(path)
    candidates = "\n  ".join(
        str(path) for path in _candidate_wave_tool_paths(tool_name, override)
    )
    raise RuntimeError(
        f"tlx_wave requires {tool_name} from the third_party/wave submodule build. "
        "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave, or run the standalone "
        "third_party/wave build documented in third_party/wave/README.md. "
        f"Checked {tool_name} candidates:\n  {candidates}"
    )


def _wave_opt():
    return _wave_tool("wave-opt", override_env="TRITON_WAVE_OPT")


def _verify_wave_module(wave_text, wave_opt):
    result = subprocess.run(
        [wave_opt, "-", "--verify-diagnostics"],
        input=wave_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"tlx_wave generated Wave module failed wave-opt verification: {detail}"
        )
