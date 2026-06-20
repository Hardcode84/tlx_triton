import json

from .wave_bridge_conversion import _convert_imported_ttgir
from .wave_bridge_rewriters import (
    _AsyncCommitGroupValue,
    _AsyncCopyValue,
    _BufferStoreEffect,
    _IfEffect,
    _LocalLoadValue,
    _basic_rewriter_registry,
)
from .wave_bridge_structural_emit import (
    _UnsupportedStructuralEmission,
    _can_emit_structural_wave_module,
    _emit_structural_wave_module,
)
from .wave_bridge_text_emit import _emit_new_wave_module
from .wave_bridge_tools import _verify_wave_module, _wave_opt


_NEW_BRIDGE_SUPPORTED_OPS = frozenset(
    {
        "amdg.buffer_load_to_local",
        "amdg.buffer_store",
        "arith.addi",
        "arith.andi",
        "arith.cmpi",
        "arith.constant",
        "arith.divsi",
        "arith.divui",
        "arith.minsi",
        "arith.muli",
        "arith.ori",
        "arith.remsi",
        "arith.remui",
        "arith.subi",
        "arith.truncf",
        "arith.xori",
        "llvm.intr.assume",
        "rocdl.sched.barrier",
        "rocdl.sched.group.barrier",
        "scf.for",
        "scf.if",
        "scf.yield",
        "tt.addptr",
        "tt.broadcast",
        "tt.dot",
        "tt.expand_dims",
        "tt.get_program_id",
        "tt.load",
        "tt.make_range",
        "tt.return",
        "tt.splat",
        "tt.store",
        "ttg.async_commit_group",
        "ttg.async_copy_global_to_local",
        "ttg.async_wait",
        "ttg.convert_layout",
        "ttg.local_alloc",
        "ttg.local_load",
        "ttg.local_store",
        "ttg.memdesc_index",
    }
)

_NEW_BRIDGE_STRUCTURAL_OPS = frozenset({"builtin.module", "tt.func"})


def stop_before_wave_lowering(mod, metadata, options):
    """Emit the first Wave/WaveAMD module from cutoff TTGIR.

    This stage preserves the public kernel ABI and lowers the supported
    TTGIR graph prefix into Wave/WaveAMD memory-token operations. For the
    supported dot/MMA op path it also lowers dot-operand local loads and
    f16/bf16 to f32 tt.dot SSA flow into WaveAMD MFMA fragments.
    """
    if _module_uses_new_bridge_required_ops(mod):
        return _stop_before_wave_lowering_new_bridge(mod, metadata, options)
    raise ValueError(
        "tlx_wave new bridge does not support this TTGIR module yet: "
        + ", ".join(_module_op_names(mod))
    )


def _module_uses_new_bridge_required_ops(mod):
    op_names = set()
    found_simple_blocker = False

    def visit(op):
        nonlocal found_simple_blocker
        name = op.get_name()
        if name not in _NEW_BRIDGE_STRUCTURAL_OPS:
            op_names.add(name)
        if name in {"tt.load", "tt.store"} and _has_nondefault_cache_modifier(op):
            found_simple_blocker = True
            return True
        return True

    mod.walk(visit)
    if found_simple_blocker:
        return False
    if op_names == {"tt.return"}:
        return True
    return bool(op_names) and op_names <= _NEW_BRIDGE_SUPPORTED_OPS


def _module_op_names(mod):
    op_names = set()

    def visit(op):
        name = op.get_name()
        if name not in _NEW_BRIDGE_STRUCTURAL_OPS:
            op_names.add(name)
        return True

    mod.walk(visit)
    return tuple(sorted(op_names))


def _has_nondefault_cache_modifier(op):
    attrs = dict(op.get_attrs())
    cache = attrs.get("cacheModifier")
    if cache is None:
        return False
    return str(cache).strip() not in {"", "none"}

def _stop_before_wave_lowering_new_bridge(mod, metadata, options):
    run = _convert_imported_ttgir(mod, _basic_rewriter_registry())
    program = run.program
    _validate_new_bridge_options(program, options)
    emit_api = "text"
    if _can_emit_structural_wave_module(run):
        try:
            structural = _emit_structural_wave_module(run)
            wave_text = structural.text
            emit_api = "structural-python"
        except _UnsupportedStructuralEmission:
            wave_text = _emit_new_wave_module(run)
    else:
        wave_text = _emit_new_wave_module(run)

    metadata["name"] = program.kernel_name or "kernel"
    metadata["shared"] = _wave_lds_size(wave_text)
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = "emitted_wave_ttgir_op_lowering"
    metadata["tlx_wave_bridge_stage"] = "ttgir-op-lowering"
    metadata["tlx_wave_arch"] = options.arch
    metadata["tlx_wave_ttgir_target"] = program.target
    metadata["tlx_wave_num_warps"] = program.num_warps
    metadata["tlx_wave_threads_per_warp"] = program.threads_per_warp
    metadata["tlx_wave_num_ctas"] = program.num_ctas
    metadata["tlx_wave_num_kernel_args"] = len(program.kernel_args)
    metadata["tlx_wave_num_pointer_args"] = sum(
        program.values[value_id].type.kind == "pointer" for value_id in program.kernel_args
    )
    metadata["tlx_wave_num_scalar_args"] = sum(
        program.values[value_id].type.kind == "scalar" for value_id in program.kernel_args
    )
    metadata["tlx_wave_plan_kind"] = "new-python-rewrite"
    metadata["tlx_wave_plan_num_ops"] = len(program.ops)
    metadata["tlx_wave_plan_num_values"] = len(program.values)
    metadata["tlx_wave_plan_num_addresses"] = 0
    metadata["tlx_wave_plan_num_memdescs"] = sum(
        value.type.kind == "memdesc" for value in program.values.values()
    )
    metadata["tlx_wave_plan_num_layouts"] = 0
    metadata["tlx_wave_plan_num_layout_constraints"] = 0
    metadata["tlx_wave_plan_num_storage_aliases"] = 0
    metadata["tlx_wave_plan_num_tokens"] = len(program.token_graph.nodes)
    metadata["tlx_wave_plan_json"] = json.dumps(
        {
            "kind": "new-python-rewrite",
            "ops": len(program.ops),
            "tokens": len(program.token_graph.nodes),
        },
        sort_keys=True,
    )
    metadata["tlx_wave_lds_size_bytes"] = metadata["shared"]
    converted_values = tuple(_iter_converted_values(run.result.values.values())) + tuple(
        _iter_effect_values(run.result.effects)
    )
    converted_effects = tuple(_iter_converted_effects(run.result.effects))
    metadata["tlx_wave_num_async_copies"] = sum(
        isinstance(value, _AsyncCopyValue) for value in converted_values
    )
    metadata["tlx_wave_num_dma_load_lds"] = wave_text.count("waveamd.dma_load_lds")
    metadata["tlx_wave_num_async_commit_groups"] = sum(
        isinstance(value, _AsyncCommitGroupValue)
        for value in converted_values
    )
    metadata["tlx_wave_num_async_waits"] = wave_text.count("wave.wait")
    metadata["tlx_wave_num_wave_joins"] = wave_text.count("wave.join")
    metadata["tlx_wave_num_wave_barriers"] = wave_text.count("wave.barrier")
    metadata["tlx_wave_num_wave_local_loads"] = _wave_local_load_count(wave_text)
    metadata["tlx_wave_num_fragment_packs"] = wave_text.count("waveamd.fragment_pack")
    metadata["tlx_wave_num_fragment_fills"] = wave_text.count("waveamd.fragment_fill")
    metadata["tlx_wave_num_mmas"] = wave_text.count("waveamd.mma")
    metadata["tlx_wave_num_buffer_stores"] = sum(
        isinstance(effect, _BufferStoreEffect) for effect in converted_effects
    )
    wave_opt = _wave_opt()
    _verify_wave_module(wave_text, wave_opt)
    metadata["tlx_wave_wave_builder"] = "new-python-rewrite"
    metadata["tlx_wave_emit_api"] = emit_api
    metadata["tlx_wave_wave_opt"] = wave_opt
    return wave_text


def _validate_new_bridge_options(program, options):
    arch = getattr(options, "arch", None)
    if arch not in {"gfx942", "gfx950"}:
        raise ValueError(
            f"tlx_wave bridge only supports gfx942/gfx950, got {arch}"
        )
    expected_target = f"hip:{arch}"
    if program.target != expected_target:
        raise ValueError(
            f"tlx_wave bridge expected TTGIR target {expected_target}, got {program.target}"
        )
    option_warp_size = getattr(options, "warp_size", None)
    if option_warp_size is not None and int(option_warp_size) != 64:
        raise ValueError(
            f"tlx_wave bridge only supports wave64, got warp_size={option_warp_size}"
        )
    if program.threads_per_warp is not None and int(program.threads_per_warp) != 64:
        raise ValueError(
            "tlx_wave bridge only supports wave64, got "
            f"threads_per_warp={program.threads_per_warp}"
        )


def _iter_converted_values(values):
    for value in values:
        yield value
        for attr_name in ("then_values", "else_values", "body_values"):
            nested = getattr(value, attr_name, None)
            if nested:
                yield from _iter_converted_values(nested.values())


def _iter_converted_effects(effects):
    for effect in effects:
        yield effect
        if isinstance(effect, _IfEffect):
            yield from _iter_converted_effects(effect.then_effects)
            yield from _iter_converted_effects(effect.else_effects)


def _iter_effect_values(effects):
    for effect in effects:
        if isinstance(effect, _IfEffect):
            yield from _iter_converted_values(effect.then_values.values())
            yield from _iter_converted_values(effect.else_values.values())
            yield from _iter_effect_values(effect.then_effects)
            yield from _iter_effect_values(effect.else_effects)


def _wave_lds_size(wave_text):
    marker = "wave.lds_size = "
    for line in wave_text.splitlines():
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        return int(tail.split(":", 1)[0].strip())
    return 0


def _wave_local_load_count(wave_text):
    return sum(
        1
        for line in wave_text.splitlines()
        if "wave.load" in line and "#wave.shared" in line
    ) + wave_text.count("waveamd.transpose_load")
