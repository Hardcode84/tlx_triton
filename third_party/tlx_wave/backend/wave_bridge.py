import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _KernelArg:
    index: int
    name: str
    signature_type: str
    ttgir_type: str
    wave_type: str
    kind: str


@dataclass(frozen=True)
class _Kernel:
    name: str
    args: tuple[_KernelArg, ...]
    noinline: bool | None


@dataclass(frozen=True)
class _ModuleAttrs:
    target: str
    num_ctas: int
    num_warps: int
    threads_per_warp: int
    has_explicit_local_mem_access: bool


def _entry_name(mod):
    try:
        return mod.get_entry_func_name()
    except Exception:
        return "tlx_wave_kernel"


def _required_int_attr(op, name):
    value = op.get_int_attr(name)
    if value is None:
        raise ValueError(f"tlx_wave bridge expected TTGIR module attribute {name}")
    return value


def _module_attrs(mod):
    op = mod.get_operation()
    target = op.get_str_attr("ttg.target")
    if target is None:
        raise ValueError("tlx_wave bridge expected TTGIR module attribute ttg.target")
    num_ctas = _required_int_attr(op, "ttg.num-ctas")
    num_warps = _required_int_attr(op, "ttg.num-warps")
    threads_per_warp = _required_int_attr(op, "ttg.threads-per-warp")
    has_explicit_local_mem_access = bool(op.get_bool_attr("tlx.has_explicit_local_mem_access"))
    return _ModuleAttrs(target, num_ctas, num_warps, threads_per_warp, has_explicit_local_mem_access)


def _wave_arg_type(name, signature_type):
    if signature_type.startswith("*"):
        element_type = signature_type[1:]
        if not element_type:
            raise ValueError(f"tlx_wave bridge does not yet support pointer argument %{name} with empty element type")
        return f"!wave.ptr<#wave.global, {element_type}>", "pointer"
    if signature_type in {"i1", "i8", "i16", "i32", "i64", "index", "f16", "bf16", "f32"}:
        return signature_type, "scalar"
    raise ValueError(f"tlx_wave bridge does not yet support kernel argument %{name} with type {signature_type}")


def _public_tt_func_ops(mod):
    funcs = []

    def visit(op):
        if op.get_name() == "tt.func" and op.get_str_attr("sym_visibility") == "public":
            funcs.append(op)
        return True

    mod.walk(visit)
    return tuple(funcs)


def _kernel_from_module(mod):
    funcs = _public_tt_func_ops(mod)
    if len(funcs) != 1:
        names = ", ".join(func.get_str_attr("sym_name") or "<unnamed>" for func in funcs) or "none"
        raise ValueError(f"tlx_wave bridge supports exactly one public tt.func kernel, found {len(funcs)} ({names})")

    op = funcs[0]
    name = op.get_str_attr("sym_name") or _entry_name(mod)
    fn = mod.get_function(name)
    signature = mod.get_function_signature(fn)
    if len(signature) != fn.get_num_args():
        raise ValueError(
            "tlx_wave bridge expected function signature length to match argument count, "
            f"got {len(signature)} signature entries for {fn.get_num_args()} args"
        )

    args = []
    for index, signature_type in enumerate(signature):
        name_for_arg = f"arg{index}"
        ttgir_type = str(fn.args(index).get_type())
        wave_type, kind = _wave_arg_type(name_for_arg, signature_type)
        args.append(_KernelArg(index, name_for_arg, signature_type, ttgir_type, wave_type, kind))
    return _Kernel(name, tuple(args), op.get_bool_attr("noinline"))


def _validate_target(options, attrs):
    if options.arch != "gfx950":
        raise ValueError(f"tlx_wave bridge only supports gfx950 Wave skeletons, got {options.arch}")
    if options.warp_size != 64:
        raise ValueError(f"tlx_wave bridge only supports wave64 inputs, got warp_size={options.warp_size}")
    if attrs.target != "hip:gfx950":
        raise ValueError(f"tlx_wave bridge only supports TTGIR target hip:gfx950, got {attrs.target}")
    if attrs.threads_per_warp != 64:
        raise ValueError(
            "tlx_wave bridge only supports wave64 TTGIR, "
            f'got "ttg.threads-per-warp" = {attrs.threads_per_warp}'
        )


def _target_triple(attrs):
    return attrs.target.replace("hip:", "amdgcn-amd-amdhsa--")


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
        yield build_dir / "third_party" / "tlx_wave" / "wave"
        yield build_dir / "third_party" / "wave"
    yield _repo_root() / "third_party" / "wave" / "build"


def _candidate_wave_python_paths():
    override = os.environ.get("TRITON_WAVE_PYTHONPATH")
    if override:
        for entry in override.split(os.pathsep):
            if entry:
                yield Path(entry)

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "python_packages" / "wave_mlir"


def _candidate_wave_opt_paths():
    override = os.environ.get("TRITON_WAVE_OPT")
    if override:
        yield Path(override)

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "bin" / "wave-opt"


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
            "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave and WAVE_ENABLE_PYTHON_BINDINGS=ON; "
            "the underlying MLIR install must have MLIR_ENABLE_BINDINGS_PYTHON=ON. "
            f"Unable to import mlir.dialects.wave_dsl: {type(exc).__name__}: {exc}. "
            f"Checked Wave Python package candidates:\n  {candidates}"
        ) from exc
    return w


def _wave_opt():
    for path in _existing_paths(_candidate_wave_opt_paths()):
        if os.access(path, os.X_OK):
            return str(path)
    candidates = "\n  ".join(str(path) for path in _candidate_wave_opt_paths())
    raise RuntimeError(
        "tlx_wave requires wave-opt from the third_party/wave submodule build. "
        "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave so the Wave tools are built. "
        f"Checked wave-opt candidates:\n  {candidates}"
    )


def _binding_type(signature_type, w):
    if signature_type.startswith("*"):
        return w.ptr_type(_binding_type(signature_type[1:], w))
    if signature_type == "i1":
        return w.i1()
    if signature_type == "i8":
        return w.i8()
    if signature_type == "i16":
        return w.IntegerType.get_signless(16)
    if signature_type == "i32":
        return w.i32()
    if signature_type == "i64":
        return w.i64()
    if signature_type == "index":
        return w.index_type()
    if signature_type == "f16":
        return w.f16()
    if signature_type == "bf16":
        return w.bf16()
    if signature_type == "f32":
        return w.f32()
    raise ValueError(f"tlx_wave bridge does not yet support Wave binding type {signature_type}")


def _binding_i32_attr(w, value):
    return w.IntegerAttr.get(w.i32(), int(value))


def _binding_bool_attr(w, value):
    return w.IntegerAttr.get(w.i1(), int(bool(value)))


def _binding_attrs(w, attrs):
    return {
        "tlx_wave.bridge.stage": w.StringAttr.get("module-function-skeleton"),
        "tlx_wave.source_op": w.StringAttr.get("tt.func"),
        "tlx_wave.num_pointer_args": _binding_i32_attr(w, attrs["pointer_count"]),
        "tlx_wave.num_scalar_args": _binding_i32_attr(w, attrs["scalar_count"]),
        "tlx_wave.wave_size": _binding_i32_attr(w, attrs["wave_size"]),
        "tlx_wave.num_warps": _binding_i32_attr(w, attrs["num_warps"]),
    }


def _emit_wave_skeleton_with_bindings(kernel, attrs):
    w = _load_wave_dsl()
    target_triple = _target_triple(attrs)
    pointer_count = sum(arg.kind == "pointer" for arg in kernel.args)
    scalar_count = sum(arg.kind == "scalar" for arg in kernel.args)
    with w.module() as module_builder:
        func_attrs = _binding_attrs(
            w, {
                "pointer_count": pointer_count,
                "scalar_count": scalar_count,
                "wave_size": attrs.threads_per_warp,
                "num_warps": attrs.num_warps,
            })
        if kernel.noinline is not None:
            func_attrs["tlx_wave.ttgir.noinline"] = _binding_bool_attr(w, kernel.noinline)

        arg_types = [_binding_type(arg.signature_type, w) for arg in kernel.args]
        module_builder.module.operation.attributes["waveamdmachine.target"] = w.StringAttr.get(target_triple)
        module_builder.module.operation.attributes["tlx_wave.source_target"] = w.StringAttr.get(attrs.target)
        module_builder.module.operation.attributes["tlx_wave.num_ctas"] = _binding_i32_attr(w, attrs.num_ctas)
        module_builder.module.operation.attributes["tlx_wave.num_warps"] = _binding_i32_attr(w, attrs.num_warps)
        module_builder.module.operation.attributes["tlx_wave.threads_per_warp"] = _binding_i32_attr(
            w, attrs.threads_per_warp)
        module_builder.module.operation.attributes["tlx_wave.has_explicit_local_mem_access"] = _binding_bool_attr(
            w, attrs.has_explicit_local_mem_access)
        with module_builder.function(kernel.name, arg_types, kernel=True, attrs=func_attrs):
            pass
        return str(module_builder.module)


def _emit_wave_skeleton(kernel, attrs):
    return _emit_wave_skeleton_with_bindings(kernel, attrs), "wave-dsl"


def _verify_wave_skeleton(wave_text, wave_opt):
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
        raise RuntimeError(f"tlx_wave generated Wave skeleton failed wave-opt verification: {detail}")


def stop_before_wave_lowering(mod, metadata, options):
    """Emit the first Wave/WaveAMD module/function skeleton from cutoff TTGIR.

    This stage still does not lower TTGIR operations. It parses the handoff
    module enough to preserve the public kernel ABI and launch metadata, then
    emits a Wave textual MLIR shell for later bridge stages to fill.
    """
    attrs = _module_attrs(mod)
    _validate_target(options, attrs)
    kernel = _kernel_from_module(mod)

    metadata["name"] = kernel.name or _entry_name(mod)
    metadata["shared"] = 0
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = "emitted_wave_skeleton"
    metadata["tlx_wave_arch"] = options.arch
    metadata["tlx_wave_ttgir_target"] = attrs.target
    metadata["tlx_wave_num_warps"] = attrs.num_warps
    metadata["tlx_wave_threads_per_warp"] = attrs.threads_per_warp
    metadata["tlx_wave_num_ctas"] = attrs.num_ctas
    metadata["tlx_wave_num_kernel_args"] = len(kernel.args)
    metadata["tlx_wave_num_pointer_args"] = sum(arg.kind == "pointer" for arg in kernel.args)
    metadata["tlx_wave_num_scalar_args"] = sum(arg.kind == "scalar" for arg in kernel.args)
    wave_text, builder = _emit_wave_skeleton(kernel, attrs)
    wave_opt = _wave_opt()
    _verify_wave_skeleton(wave_text, wave_opt)
    metadata["tlx_wave_wave_builder"] = builder
    metadata["tlx_wave_wave_opt"] = wave_opt
    return wave_text
