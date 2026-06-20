"""Wave text emission for the new TLX Wave conversion path."""

from dataclasses import dataclass, field
from math import gcd
import re

from .wave_bridge_conversion import (
    _ConvertedInputValue,
    _ConversionRun,
    _amd_mfma_encoding_info,
    _attr_bool,
    _attr_value,
    _blocked_encoding_info as _conversion_blocked_encoding_info,
    _convert_imported_ttgir,
    _dot_operand_encoding_info,
    _mma_shape_for_parent,
    _tile_rep_count,
)
from .wave_bridge_rewriters import (
    _AssumeEffect,
    _AsyncCommitGroupValue,
    _AsyncCopyValue,
    _AsyncWaitEffect,
    _AsyncWaitValue,
    _BinaryValue,
    _BufferStoreEffect,
    _CastValue,
    _CompareValue,
    _ConstantValue,
    _DotValue,
    _ForEffect,
    _ForValue,
    _ForwardValue,
    _IfEffect,
    _IfValue,
    _LoadValue,
    _LocalAllocValue,
    _LocalLoadValue,
    _LocalStoreEffect,
    _MaskAndValue,
    _MemdescIndexValue,
    _MinValue,
    _PointerAddValue,
    _ProgramIdValue,
    _RangeValue,
    _ReturnEffect,
    _StoreEffect,
    _UnaryTensorValue,
    _basic_rewriter_registry,
)


@dataclass
class _TextEmitterState:
    lines: list[str]
    names: dict[int, str]
    types: dict[int, str]
    converted_values: dict[int, object]
    source_ops: tuple[object, ...]
    source_values: dict[int, object]
    token_graph: object
    lds_offsets: dict[int, int]
    deferred_coordinate_values: frozenset[int] = frozenset()
    assume_condition_values: frozenset[int] = frozenset()
    nonnegative_values: frozenset[int] = frozenset()
    positive_values: frozenset[int] = frozenset()
    coordinate_cache: dict[tuple[int, tuple[tuple[str, str], ...]], tuple[str, str]] = (
        field(default_factory=dict)
    )
    fragment_coord_root_cache: dict[
        tuple[int, int], tuple[tuple[tuple[str, str], ...], ...]
    ] = field(default_factory=dict)
    buffer_cache: dict[int, tuple[str, str]] = field(default_factory=dict)
    next_id: int = 0
    mem_token: str | None = None
    pending_read_tokens: tuple[str, ...] = ()
    indent: int = 4

    def fresh(self):
        name = f"%n{self.next_id}"
        self.next_id += 1
        return name

    def emit(self, line):
        self.lines.append(f"{' ' * self.indent}{line}")


@dataclass(frozen=True)
class _FragmentTypeInfo:
    role: int
    element_type: str
    rows: int
    cols: int
    wave_size: int
    registers: int


@dataclass(frozen=True)
class _BlockedEncodingInfo:
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]


@dataclass(frozen=True)
class _ThreadLayoutInfo:
    width: int
    cta_threads: int


@dataclass(frozen=True)
class _SwizzledSharedEncodingInfo:
    vec: int
    per_phase: int
    max_phase: int
    order: tuple[int, ...]


@dataclass(frozen=True)
class _CoordGroupFact:
    divisibility: int
    delta_min: int
    delta_max: int


def _emit_imported_ttgir_new_wave(mod, registry=None):
    registry = _basic_rewriter_registry() if registry is None else registry
    return _emit_new_wave_module(_convert_imported_ttgir(mod, registry))


def _deferred_coordinate_value_ids(program, result):
    fragment_store_effects = tuple(
        effect
        for effect in result.effects
        if isinstance(effect, (_StoreEffect, _BufferStoreEffect))
        and _is_fragment_physical_value(result.values, effect.value_id)
    )
    dma_copy_values = tuple(
        value
        for value in result.values.values()
        if isinstance(value, _AsyncCopyValue)
        and value.mask_value_id is None
        and value.source_address_value_id is not None
        and value.memdesc_value_id is not None
    )
    if not fragment_store_effects and not dma_copy_values:
        return frozenset()

    initial = set()
    for effect in fragment_store_effects:
        if isinstance(effect, _BufferStoreEffect):
            _collect_coordinate_dependencies(
                program, result.values, effect.offset_id, initial
            )
        else:
            _collect_coordinate_dependencies(
                program, result.values, effect.pointer_id, initial
            )
        if effect.mask_id is not None:
            _collect_coordinate_dependencies(program, result.values, effect.mask_id, initial)
    for value in dma_copy_values:
        if value.source_offset_value_id is not None:
            _collect_coordinate_dependencies(
                program, result.values, value.source_offset_value_id, initial
            )
        elif value.source_address_value_id is not None:
            _collect_coordinate_dependencies(
                program, result.values, value.source_address_value_id, initial
            )

    consumers = _value_consumers(result)
    fragment_store_ids = {id(effect) for effect in fragment_store_effects}
    dma_copy_ids = {value.value_id for value in dma_copy_values}
    deferred = set()
    for value_id in initial:
        keep_deferred = True
        for kind, consumer in consumers.get(value_id, ()):
            if kind == "value" and consumer in initial:
                continue
            if kind == "value" and consumer in dma_copy_ids:
                continue
            if kind == "effect" and consumer in fragment_store_ids:
                continue
            keep_deferred = False
            break
        if keep_deferred:
            deferred.add(value_id)
    return frozenset(deferred)


def _assumed_nonnegative_value_ids(result):
    value_ids = set()
    for effect in result.effects:
        if not isinstance(effect, _AssumeEffect):
            continue
        for value_id, predicate_text in _assume_facts(effect.predicate):
            if predicate_text in {"x >= 0", "x > 0"}:
                value_ids.add(value_id)
    return frozenset(value_ids)


def _assumed_positive_value_ids(result):
    value_ids = set()
    for effect in result.effects:
        if not isinstance(effect, _AssumeEffect):
            continue
        for value_id, predicate_text in _assume_facts(effect.predicate):
            if predicate_text == "x > 0":
                value_ids.add(value_id)
    return frozenset(value_ids)


def _assume_condition_value_ids(result):
    assume_effects = tuple(
        effect for effect in result.effects if isinstance(effect, _AssumeEffect)
    )
    if not assume_effects:
        return frozenset()
    candidates = set()
    for effect in assume_effects:
        _collect_assume_condition_dependencies(
            result.values,
            effect.predicate_id,
            candidates,
            set(),
        )
    assume_effect_ids = {id(effect) for effect in assume_effects}
    consumers = _value_consumers(result)
    skipped = set()
    changed = True
    while changed:
        changed = False
        for value_id in candidates:
            if value_id in skipped:
                continue
            value_consumers = consumers.get(value_id, ())
            if value_consumers and all(
                (kind == "effect" and consumer in assume_effect_ids)
                or (kind == "value" and consumer in skipped)
                for kind, consumer in value_consumers
            ):
                skipped.add(value_id)
                changed = True
    return frozenset(skipped)


def _collect_assume_condition_dependencies(converted_values, value_id, out, seen):
    if value_id in seen:
        return
    seen.add(value_id)
    out.add(value_id)
    value = converted_values.get(value_id)
    if value is None:
        return
    for operand_id in _record_operand_ids(value):
        _collect_assume_condition_dependencies(converted_values, operand_id, out, seen)


def _collect_coordinate_dependencies(program, converted_values, value_id, out):
    source_value = program.values.get(value_id)
    if source_value is not None and source_value.type.kind == "tensor":
        out.add(value_id)
    value = converted_values.get(value_id)
    if value is None:
        return
    for operand_id in _record_operand_ids(value):
        _collect_coordinate_dependencies(program, converted_values, operand_id, out)


def _value_consumers(result):
    consumers = {}
    _add_value_consumers(consumers, result.values)
    _add_effect_consumers(consumers, result.effects)
    return consumers


def _add_value_consumers(consumers, values):
    for result_id, value in values.items():
        for operand_id in _record_operand_ids(value):
            consumers.setdefault(operand_id, set()).add(("value", result_id))
        if isinstance(value, _IfValue):
            _add_value_consumers(consumers, value.then_values)
            _add_value_consumers(consumers, value.else_values)
        elif isinstance(value, _ForValue):
            _add_value_consumers(consumers, value.body_values)
            _add_effect_consumers(consumers, value.body_effects)


def _add_effect_consumers(consumers, effects):
    for effect in effects:
        for operand_id in _effect_operand_ids(effect):
            consumers.setdefault(operand_id, set()).add(("effect", id(effect)))
        if isinstance(effect, _ForEffect):
            _add_value_consumers(consumers, effect.body_values)
            _add_effect_consumers(consumers, effect.body_effects)
        elif isinstance(effect, _IfEffect):
            _add_value_consumers(consumers, effect.then_values)
            _add_value_consumers(consumers, effect.else_values)
            _add_effect_consumers(consumers, effect.then_effects)
            _add_effect_consumers(consumers, effect.else_effects)


def _record_operand_ids(value):
    if isinstance(value, (_UnaryTensorValue, _ForwardValue, _CastValue)):
        return (value.operand_id,)
    if isinstance(value, (_BinaryValue, _CompareValue, _MaskAndValue, _MinValue)):
        return (value.lhs_id, value.rhs_id)
    if isinstance(value, _PointerAddValue):
        return (value.base_id, value.offset_id)
    if isinstance(value, _DotValue):
        return (value.lhs_id, value.rhs_id, value.acc_id)
    if isinstance(value, _IfValue):
        return (value.condition_id,) + value.then_yield_ids + value.else_yield_ids
    if isinstance(value, _ForValue):
        return (
            value.lower_id,
            value.upper_id,
            value.step_id,
        ) + value.iter_arg_ids + value.body_yield_ids
    if isinstance(value, _MemdescIndexValue):
        return (value.memdesc_id, value.index_id)
    if isinstance(value, _LoadValue):
        ids = [value.pointer_id]
        if value.mask_id is not None:
            ids.append(value.mask_id)
        if value.other_id is not None:
            ids.append(value.other_id)
        return tuple(ids)
    if isinstance(value, _LocalLoadValue):
        return (value.memdesc_id,) if value.token_id is None else (value.memdesc_id, value.token_id)
    if isinstance(value, _AsyncCopyValue):
        ids = []
        for operand_id in (
            value.source_address_value_id,
            value.source_offset_value_id,
            value.memdesc_value_id,
            value.mask_value_id,
            value.other_value_id,
        ):
            if operand_id is not None:
                ids.append(operand_id)
        return tuple(ids)
    if isinstance(value, _AsyncCommitGroupValue):
        return value.member_token_ids
    return ()


def _effect_operand_ids(effect):
    if isinstance(effect, _StoreEffect):
        ids = [effect.pointer_id, effect.value_id]
        if effect.mask_id is not None:
            ids.append(effect.mask_id)
        return tuple(ids)
    if isinstance(effect, _BufferStoreEffect):
        ids = [effect.value_id, effect.base_id, effect.offset_id]
        if effect.mask_id is not None:
            ids.append(effect.mask_id)
        return tuple(ids)
    if isinstance(effect, _LocalStoreEffect):
        return (effect.value_id, effect.memdesc_id) if effect.token_id is None else (
            effect.value_id,
            effect.memdesc_id,
            effect.token_id,
        )
    if isinstance(effect, _AssumeEffect):
        return (effect.predicate_id,)
    if isinstance(effect, _ForEffect):
        return (
            effect.lower_id,
            effect.upper_id,
            effect.step_id,
        ) + effect.iter_arg_ids + effect.body_yield_ids
    if isinstance(effect, _IfEffect):
        return (effect.condition_id,)
    if isinstance(effect, _AsyncWaitEffect):
        return effect.input_token_ids
    if isinstance(effect, _ReturnEffect):
        return effect.operand_ids
    return ()


def _is_fragment_physical_value(converted_values, value_id, seen=frozenset()):
    if value_id in seen:
        return False
    value = converted_values.get(value_id)
    if value is None:
        return False
    converted_type = getattr(value, "converted_type", None)
    if converted_type is not None and converted_type.kind == "fragment":
        return True
    if isinstance(value, (_ForwardValue, _CastValue)):
        return _is_fragment_physical_value(
            converted_values,
            value.operand_id,
            seen | {value_id},
        )
    return False


def _emit_new_wave_module(run: _ConversionRun):
    program = run.program
    result = run.result
    lds_offsets, lds_size = _compute_lds_layout(program)
    deferred_coordinate_values = _deferred_coordinate_value_ids(program, result)
    assume_condition_values = _assume_condition_value_ids(result)
    nonnegative_values = _assumed_nonnegative_value_ids(result)
    positive_values = _assumed_positive_value_ids(result)
    state = _TextEmitterState(
        lines=[],
        names={},
        types={},
        converted_values=result.values,
        source_ops=program.ops,
        source_values=program.values,
        token_graph=program.token_graph,
        lds_offsets=lds_offsets,
        deferred_coordinate_values=deferred_coordinate_values,
        assume_condition_values=assume_condition_values,
        nonnegative_values=nonnegative_values,
        positive_values=positive_values,
    )
    arg_decls = []
    for arg_index, value_id in enumerate(program.kernel_args):
        value = result.values[value_id]
        if not isinstance(value, _ConvertedInputValue):
            raise ValueError(
                "tlx_wave new emitter expected converted input value for "
                f"kernel arg {arg_index}"
            )
        arg_name = f"%arg{arg_index}"
        state.names[value_id] = _emit_input_binding(state, arg_name, value)
        state.types[value_id] = value.converted_type.wave_type
        arg_decls.append(f"{arg_name}: {value.converted_type.wave_type}")

    effects_by_op_index = {}
    trailing_effects = []
    for effect in result.effects:
        op_index = getattr(effect, "op_index", -1)
        if op_index < 0:
            trailing_effects.append(effect)
        else:
            effects_by_op_index.setdefault(op_index, []).append(effect)

    for op in program.ops:
        for result_id in op.results:
            if (
                result_id in state.deferred_coordinate_values
                or result_id in state.assume_condition_values
            ):
                continue
            value = result.values.get(result_id)
            if value is None:
                continue
            _emit_value(state, value)
        for effect in effects_by_op_index.pop(op.index, ()):
            _emit_effect(state, effect)

    for op_index in sorted(effects_by_op_index):
        for effect in effects_by_op_index[op_index]:
            _emit_effect(state, effect)
    for effect in trailing_effects:
        _emit_effect(state, effect)

    if not any(line.strip().startswith("return") for line in state.lines):
        state.emit("return")

    kernel_name = program.kernel_name or "kernel"
    target = program.target or "hip:gfx950"
    target_triple = target.replace("hip:", "amdgcn-amd-amdhsa--")
    num_ctas = _int_or(program.num_ctas, 1)
    num_warps = _int_or(program.num_warps, 1)
    wave_size = _int_or(program.threads_per_warp, 64)
    has_explicit_local_mem_access = (
        "true" if program.has_explicit_local_mem_access else "false"
    )
    noinline = "true" if program.noinline else "false"
    pointer_count = sum(
        1 for value_id in program.kernel_args if _input_kind(result.values[value_id]) == "pointer"
    )
    scalar_count = sum(
        1 for value_id in program.kernel_args if _input_kind(result.values[value_id]) == "scalar"
    )
    body = "\n".join(state.lines)
    args = ", ".join(arg_decls)
    lds_attr = f", wave.lds_size = {lds_size} : i64" if lds_size else ""
    return f"""module attributes {{gpu.container_module, tlx_wave.has_explicit_local_mem_access = {has_explicit_local_mem_access}, tlx_wave.new_bridge = true, tlx_wave.num_ctas = {num_ctas} : i32, tlx_wave.num_warps = {num_warps} : i32, tlx_wave.source_target = \"{target}\", tlx_wave.threads_per_warp = {wave_size} : i32, waveamdmachine.target = \"{target_triple}\"}} {{
  func.func @{kernel_name}({args}) attributes {{tlx_wave.bridge.stage = \"new-python-rewrite\", tlx_wave.num_pointer_args = {pointer_count} : i32, tlx_wave.num_scalar_args = {scalar_count} : i32, tlx_wave.num_warps = {num_warps} : i32, tlx_wave.source_op = \"tt.func\", tlx_wave.ttgir.noinline = {noinline}, tlx_wave.wave_size = {wave_size} : i32, wave.kernel{lds_attr}}} {{
{body}
  }}
}}
"""


def _emit_value(state, value):
    if value.value_id in state.names:
        return
    if isinstance(value, _ConstantValue):
        _emit_constant(state, value)
    elif isinstance(value, _RangeValue):
        _emit_range(state, value)
    elif isinstance(value, _ProgramIdValue):
        _emit_program_id(state, value)
    elif isinstance(value, _UnaryTensorValue):
        _emit_unary_tensor(state, value)
    elif isinstance(value, _BinaryValue):
        _emit_binary(state, value)
    elif isinstance(value, _CompareValue):
        _emit_compare(state, value)
    elif isinstance(value, _MaskAndValue):
        _emit_mask_and(state, value)
    elif isinstance(value, _MinValue):
        _emit_min(state, value)
    elif isinstance(value, _PointerAddValue):
        _emit_pointer_add(state, value)
    elif isinstance(value, _DotValue):
        _emit_dot(state, value)
    elif isinstance(value, _ForwardValue):
        _emit_forward(state, value)
    elif isinstance(value, _IfValue):
        _emit_if(state, value)
    elif isinstance(value, _ForValue):
        _emit_for(state, value)
    elif isinstance(value, _CastValue):
        _emit_cast(state, value)
    elif isinstance(value, _LoadValue):
        _emit_load(state, value)
    elif isinstance(value, _AsyncWaitValue):
        _emit_async_wait_value(state, value)
    elif isinstance(value, _LocalAllocValue):
        _emit_local_alloc(state, value)
    elif isinstance(value, _MemdescIndexValue):
        _emit_memdesc_index(state, value)
    elif isinstance(value, _LocalLoadValue):
        _emit_local_load(state, value)
    elif isinstance(value, _AsyncCopyValue):
        _emit_async_copy(state, value)
    elif isinstance(value, _AsyncCommitGroupValue):
        _emit_async_commit_group(state, value)
    else:
        raise ValueError(
            "tlx_wave new emitter cannot emit converted value "
            f"{type(value).__name__} for value {value.value_id}"
        )


def _emit_input_binding(state, arg_name, value):
    wave_type = value.converted_type.wave_type
    signed_bounds = _signed_int_bounds(wave_type)
    if signed_bounds is None:
        return arg_name
    lower, upper = signed_bounds
    assumed = state.fresh()
    state.emit(
        f"{assumed} = wave.assume {arg_name} as \"x\" "
        f"[#wave.pred<\"{abs(lower)} + x >= 0\">, "
        f"#wave.pred<\"-{upper} + x <= 0\">] "
        f": {wave_type}"
    )
    return assumed


def _signed_int_bounds(wave_type):
    if wave_type == "i8":
        return -(1 << 7), (1 << 7) - 1
    if wave_type == "i16":
        return -(1 << 15), (1 << 15) - 1
    if wave_type == "i32":
        return -(1 << 31), (1 << 31) - 1
    return None


def _emit_constant(state, value):
    if value.converted_type.kind == "fragment":
        _emit_fragment_constant(state, value)
        return
    if _source_type(state, value.value_id).kind == "tensor":
        _emit_tensor_constant(state, value)
        return
    name = state.fresh()
    literal = _constant_literal_text(value)
    state.emit(f"{name} = arith.constant {literal}")
    state.names[value.value_id] = name
    state.types[value.value_id] = value.converted_type.wave_type


def _emit_tensor_constant(state, value):
    literal = _constant_scalar_literal(value)
    if literal is None:
        raise ValueError(
            "tlx_wave new emitter supports tensor constants only for splat "
            f"values, got {value.raw_literal}"
        )
    source_type = _source_type(state, value.value_id)
    result_type = value.converted_type.wave_type
    components = []
    if source_type.element_type == "i1":
        text = "true" if bool(literal) else "false"
        for _ in range(value.converted_type.component_count):
            name = state.fresh()
            state.emit(f"{name} = wave.constant {text} -> {result_type}")
            components.append(name)
    else:
        scalar_type = source_type.element_type or _simd_element_type(result_type)
        const = state.fresh()
        state.emit(f"{const} = arith.constant {_format_literal(literal)} : {scalar_type}")
        for _ in range(value.converted_type.component_count):
            splat = state.fresh()
            state.emit(f"{splat} = wave.splat {const} : {scalar_type} -> {result_type}")
            components.append(splat)
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _emit_fragment_constant(state, value):
    if not _constant_is_zero(value):
        raise ValueError(
            "tlx_wave new emitter supports fragment constants only for zero "
            f"fill, got {value.raw_literal}"
        )
    zero = state.fresh()
    state.emit(f"{zero} = arith.constant 0 : i32")
    names = []
    for _ in range(value.converted_type.component_count):
        frag = state.fresh()
        state.emit(
            f"{frag} = waveamd.fragment_fill {zero} : i32 -> "
            f"{value.converted_type.wave_type}"
        )
        names.append(frag)
    state.names[value.value_id] = names[0] if len(names) == 1 else tuple(names)
    state.types[value.value_id] = value.converted_type.wave_type


def _constant_is_zero(value):
    if value.literal in (0, 0.0):
        return True
    raw = "" if value.raw_literal is None else value.raw_literal.strip()
    return raw.startswith("dense<0") or raw.startswith("dense<0.0")


def _emit_range(state, value):
    wave_type = value.converted_type.wave_type
    width = _simd_width(wave_type)
    component_count = value.converted_type.component_count
    if value.end - value.start > width * component_count:
        raise ValueError(
            "tlx_wave new emitter cannot fit tt.make_range in emitted SIMD "
            "component coverage, got "
            f"start={value.start}, end={value.end}, components={component_count}, "
            f"type={wave_type}"
        )
    base = state.fresh()
    state.emit(f"{base} = wave.workitem_id 0 : {wave_type}")
    components = []
    for component in range(component_count):
        start = int(value.start) + int(component) * int(width)
        if not start:
            components.append(base)
            continue
        const = state.fresh()
        splat = state.fresh()
        result = state.fresh()
        scalar_type = _simd_element_type(wave_type)
        state.emit(f"{const} = arith.constant {start} : {scalar_type}")
        state.emit(f"{splat} = wave.splat {const} : {scalar_type} -> {wave_type}")
        state.emit(
            f"{result} = wave.binary addi {base}, {splat} : "
            f"{wave_type}, {wave_type} -> {wave_type}"
        )
        components.append(result)
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = wave_type


def _emit_program_id(state, value):
    raw = state.fresh()
    assumed = state.fresh()
    result_type = value.converted_type.wave_type
    state.emit(f"{raw} = wave.workgroup_id {value.axis}")
    state.emit(
        f"{assumed} = wave.assume {raw} as \"x\" "
        "[#wave.pred<\"x >= 0\">, #wave.pred<\"-2147483647 + x <= 0\">] "
        f": {result_type}"
    )
    state.names[value.value_id] = assumed
    state.types[value.value_id] = result_type


def _emit_unary_tensor(state, value):
    operand_name = _require_name(state, value.operand_id)
    operand_type = _require_type(state, value.operand_id)
    result_type = value.converted_type.wave_type
    result_count = value.converted_type.component_count
    operand_components = _as_components(operand_name)
    if operand_type == "i1" and result_type.startswith("!wave.mask<"):
        components = [
            _emit_scalar_i1_to_mask(state, operand_name, result_type)
            for _ in range(result_count)
        ]
        state.names[value.value_id] = _pack_components(components)
        state.types[value.value_id] = result_type
        return
    if operand_type == result_type and len(operand_components) == result_count:
        state.names[value.value_id] = operand_name
        state.types[value.value_id] = result_type
        return
    if not _is_wave_sequence_type(operand_type):
        components = []
        for _ in range(result_count):
            name = state.fresh()
            state.emit(
                f"{name} = wave.splat {operand_name} : {operand_type} -> {result_type}"
            )
            components.append(name)
        state.names[value.value_id] = _pack_components(components)
        state.types[value.value_id] = result_type
        return
    components = _remap_component_tuple(
        operand_components,
        result_count,
        value.op_name,
    )
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _emit_scalar_i1_to_mask(state, condition, result_type):
    true_mask = state.fresh()
    false_mask = state.fresh()
    result = state.fresh()
    state.emit(f"{true_mask} = wave.constant true -> {result_type}")
    state.emit(f"{false_mask} = wave.constant false -> {result_type}")
    state.emit(
        f"{result} = wave.select {condition}, {true_mask}, {false_mask} : "
        f"{result_type}"
    )
    return result


def _emit_binary(state, value):
    lhs_type = _require_type(state, value.lhs_id)
    rhs_type = _require_type(state, value.rhs_id)
    result_type = value.converted_type.wave_type
    result_count = value.converted_type.component_count
    lhs_components = _value_components_for_count(
        state, value.lhs_id, result_count, result_type, "binary lhs"
    )
    rhs_components = _value_components_for_count(
        state, value.rhs_id, result_count, result_type, "binary rhs"
    )
    if _try_emit_pow2_signed_div_rem(state, value, result_type, result_count):
        return
    kind = _effective_binary_kind(state, value, result_type)
    components = []
    for lhs, rhs in zip(lhs_components, rhs_components):
        name = state.fresh()
        state.emit(
            f"{name} = wave.binary {kind} {lhs}, {rhs} : "
            f"{result_type}, {result_type} -> {result_type}"
        )
        components.append(name)
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _effective_binary_kind(state, value, result_type):
    if value.kind not in {"divsi", "remsi"}:
        return value.kind
    if not _is_index_or_integer_scalar(result_type):
        return value.kind
    if not _scalar_nonnegative(state, value.lhs_id):
        return value.kind
    if not _scalar_positive(state, value.rhs_id):
        return value.kind
    rhs_const = _constant_int_value(state, value.rhs_id)
    if rhs_const is not None and _is_positive_power_of_two(rhs_const):
        return value.kind
    return {"divsi": "divui", "remsi": "remui"}[value.kind]


def _try_emit_pow2_signed_div_rem(state, value, result_type, result_count):
    if value.kind not in {"divsi", "remsi"}:
        return False
    if int(result_count) != 1 or not _is_index_or_integer_scalar(result_type):
        return False
    divisor = _constant_int_value(state, value.rhs_id)
    if not _is_positive_power_of_two(divisor):
        return False
    if not _scalar_nonnegative(state, value.lhs_id):
        return False
    lhs = _require_name(state, value.lhs_id)
    lhs_type = _require_type(state, value.lhs_id)
    lhs_lower = _scalar_lower_bound(state, value.lhs_id)
    if lhs_lower is not None and lhs_lower >= 0:
        assumed_lhs = state.fresh()
        lower_pred = "x >= 0" if lhs_lower == 0 else f"-{int(lhs_lower)} + x >= 0"
        state.emit(
            f"{assumed_lhs} = wave.assume {lhs} as \"x\" "
            f"[#wave.pred<\"{lower_pred}\">, "
            f"#wave.pred<\"-2147483647 + x <= 0\">] : {lhs_type}"
        )
        lhs = assumed_lhs
    symbol = f"tlx_pow2_{value.kind}_{value.lhs_id}_lhs"
    if value.kind == "divsi":
        expr = f"floor(1/{int(divisor)}*{symbol})"
    else:
        expr = f"Mod({symbol}, {int(divisor)})"
    index_name = state.fresh()
    state.emit(
        f"{index_name} = wave.index_expr <\"{expr}\"> "
        f"[\"{symbol}\"]({lhs}) : ({lhs_type}) -> index"
    )
    lower = _scalar_lower_bound(state, value.value_id)
    if lower is not None and lower >= 0:
        assumed = state.fresh()
        lower_pred = "x >= 0" if lower == 0 else f"-{int(lower)} + x >= 0"
        state.emit(
            f"{assumed} = wave.assume {index_name} as \"x\" "
            f"[#wave.pred<\"{lower_pred}\">, "
            f"#wave.pred<\"-2147483647 + x <= 0\">] : index"
        )
        index_name = assumed
    name = index_name
    if result_type != "index":
        name = state.fresh()
        state.emit(
            f"{name} = {_emit_scalar_intconvert_text(index_name, 'index', result_type)}"
        )
    state.types[value.value_id] = result_type
    state.names[value.value_id] = name
    return True


def _is_positive_power_of_two(value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        return False
    return value & (value - 1) == 0


def _emit_compare(state, value):
    lhs_type = _require_type(state, value.lhs_id)
    rhs_type = _require_type(state, value.rhs_id)
    result_type = value.converted_type.wave_type
    if result_type == "i1":
        lhs = _require_name(state, value.lhs_id)
        rhs = _require_name(state, value.rhs_id)
        if lhs_type != rhs_type:
            raise ValueError(
                "tlx_wave new emitter cannot compare scalar values with "
                f"different types: {lhs_type}, {rhs_type}"
            )
        name = state.fresh()
        state.emit(f"{name} = arith.cmpi {value.predicate}, {lhs}, {rhs} : {lhs_type}")
        state.names[value.value_id] = name
    else:
        result_count = value.converted_type.component_count
        lhs_components = _value_components_for_count(
            state, value.lhs_id, result_count, lhs_type, "compare lhs"
        )
        rhs_components = _value_components_for_count(
            state, value.rhs_id, result_count, rhs_type, "compare rhs"
        )
        components = []
        for lhs, rhs in zip(lhs_components, rhs_components):
            name = state.fresh()
            state.emit(
                f"{name} = wave.cmpi {value.predicate} {lhs}, {rhs} : "
                f"{lhs_type}, {rhs_type} -> {result_type}"
            )
            components.append(name)
        state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _emit_mask_and(state, value):
    lhs_type = _require_type(state, value.lhs_id)
    rhs_type = _require_type(state, value.rhs_id)
    result_type = value.converted_type.wave_type
    if lhs_type != result_type or rhs_type != result_type:
        raise ValueError(
            "tlx_wave new emitter cannot combine masks with incompatible types: "
            f"{lhs_type}, {rhs_type} -> {result_type}"
        )
    result_count = value.converted_type.component_count
    lhs_components = _value_components_for_count(
        state, value.lhs_id, result_count, result_type, "mask lhs"
    )
    rhs_components = _value_components_for_count(
        state, value.rhs_id, result_count, result_type, "mask rhs"
    )
    components = []
    for lhs, rhs in zip(lhs_components, rhs_components):
        false_mask = state.fresh()
        name = state.fresh()
        state.emit(f"{false_mask} = wave.constant false -> {result_type}")
        state.emit(
            f"{name} = wave.select {lhs}, {rhs}, {false_mask} : "
            f"{result_type}, {result_type}"
        )
        components.append(name)
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _emit_min(state, value):
    lhs_type = _require_type(state, value.lhs_id)
    rhs_type = _require_type(state, value.rhs_id)
    result_type = value.converted_type.wave_type
    if lhs_type != result_type or rhs_type != result_type:
        raise ValueError(
            "tlx_wave new emitter cannot lower min with incompatible types: "
            f"{lhs_type}, {rhs_type} -> {result_type}"
        )
    mask_type = _cmp_result_type(result_type)
    result_count = value.converted_type.component_count
    lhs_components = _value_components_for_count(
        state, value.lhs_id, result_count, result_type, "min lhs"
    )
    rhs_components = _value_components_for_count(
        state, value.rhs_id, result_count, result_type, "min rhs"
    )
    components = []
    for lhs, rhs in zip(lhs_components, rhs_components):
        pred = state.fresh()
        name = state.fresh()
        if mask_type == "i1":
            state.emit(
                f"{pred} = arith.cmpi {value.predicate}, {lhs}, {rhs} : {lhs_type}"
            )
        else:
            state.emit(
                f"{pred} = wave.cmpi {value.predicate} {lhs}, {rhs} : "
                f"{lhs_type}, {rhs_type} -> {mask_type}"
            )
        if mask_type == "i1":
            state.emit(f"{name} = wave.select {pred}, {lhs}, {rhs} : {result_type}")
        else:
            state.emit(
                f"{name} = wave.select {pred}, {lhs}, {rhs} : "
                f"{mask_type}, {result_type}"
            )
        components.append(name)
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _emit_pointer_add(state, value):
    base_id = value.base_id
    base_type = _require_type(state, base_id)
    if isinstance(value.base, _UnaryTensorValue) and base_type.startswith("!wave.simd<"):
        scalar_base_id = value.base.operand_id
        scalar_base_type = _require_type(state, scalar_base_id)
        if scalar_base_type.startswith("!wave.ptr<"):
            base_id = scalar_base_id
            base_type = scalar_base_type
    result_type = value.converted_type.wave_type
    result_count = value.converted_type.component_count
    base_components = _value_components_for_count(
        state, base_id, result_count, base_type, "ptr_add base"
    )
    offset_components, offset_type = _index_offset_components(
        state, value.offset_id, result_count
    )
    components = []
    for base, offset in zip(base_components, offset_components):
        name = state.fresh()
        state.emit(
            f"{name} = wave.ptr_add {base}, {offset} : "
            f"{base_type}, {offset_type} -> {result_type}"
        )
        components.append(name)
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type


def _emit_dot(state, value):
    if value.value_id in state.names:
        return
    _emit_value(state, value.lhs)
    _emit_value(state, value.rhs)
    _emit_value(state, value.acc)
    lhs_fragments = _fragment_values(
        state, value.lhs_id, value.lhs_tile_shape, "tt.dot lhs"
    )
    rhs_fragments = _fragment_values(
        state, value.rhs_id, value.rhs_tile_shape, "tt.dot rhs"
    )
    acc_fragments = _fragment_values(
        state, value.acc_id, value.result_tile_shape, "tt.dot accumulator"
    )
    lhs_type = _require_type(state, value.lhs_id)
    rhs_type = _require_type(state, value.rhs_id)
    acc_type = _require_type(state, value.acc_id)
    result_type = value.converted_type.wave_type
    results = []
    for row in range(value.result_tile_shape[0]):
        for col in range(value.result_tile_shape[1]):
            acc = acc_fragments[_tile_index(value.result_tile_shape, row, col)]
            for k_tile in range(value.lhs_tile_shape[1]):
                lhs = lhs_fragments[_tile_index(value.lhs_tile_shape, row, k_tile)]
                rhs = rhs_fragments[_tile_index(value.rhs_tile_shape, k_tile, col)]
                dot = state.fresh()
                state.emit(
                    f"{dot} = waveamd.mma \"{value.mma_kind}\" {lhs}, {rhs}, {acc} : "
                    f"{lhs_type}, {rhs_type}, {acc_type} -> {result_type}"
                )
                acc = dot
            results.append(acc)
    state.names[value.value_id] = results[0] if len(results) == 1 else tuple(results)
    state.types[value.value_id] = result_type


def _fragment_values(state, value_id, tile_shape, context):
    value = _require_name(state, value_id)
    expected = int(tile_shape[0]) * int(tile_shape[1])
    if isinstance(value, tuple):
        if len(value) != expected:
            raise ValueError(
                f"tlx_wave new emitter expected {expected} fragments for {context}, "
                f"got {len(value)}"
            )
        return value
    if expected != 1:
        raise ValueError(
            f"tlx_wave new emitter expected fragment tuple for {context}, got scalar"
        )
    return (value,)


def _tile_index(tile_shape, row, col):
    return int(row) * int(tile_shape[1]) + int(col)


def _emit_forward(state, value):
    operand_name = _require_name(state, value.operand_id)
    operand_type = _require_type(state, value.operand_id)
    result_type = value.converted_type.wave_type
    if value.op_name == "ttg.convert_layout" and _is_fragment_wave_type(operand_type):
        state.names[value.value_id] = operand_name
        state.types[value.value_id] = operand_type
        return
    if operand_type != result_type:
        raise ValueError(
            "tlx_wave new emitter cannot forward value with changed Wave type: "
            f"{value.op_name} {operand_type} -> {result_type}"
        )
    state.names[value.value_id] = operand_name
    state.types[value.value_id] = result_type


def _emit_if(state, value):
    if all(result_id in state.names for result_id in value.result_ids):
        return
    condition = _require_name(state, value.condition_id)
    condition_type = _require_type(state, value.condition_id)
    if condition_type != "i1":
        raise ValueError(
            "tlx_wave new emitter cannot lower scf.if with non-scalar condition "
            f"{condition_type}"
        )
    result_types = tuple(
        value.converted_type.wave_type
        if result_id == value.value_id
        else _source_wave_type_for_value(state, result_id)
        for result_id in value.result_ids
    )
    if len(value.result_ids) == 1:
        result_base = state.fresh()
        result_names = (result_base,)
        prefix = result_base
    else:
        result_base = state.fresh()
        result_names = tuple(f"{result_base}#{index}" for index in range(len(value.result_ids)))
        prefix = f"{result_base}:{len(value.result_ids)}"
    state.emit(f"{prefix} = scf.if {condition} -> ({', '.join(result_types)}) {{")
    then_state = _child_state(state)
    _emit_region_body(then_state, value.then_op_indices, value.then_values)
    _emit_region_yield(then_state, value.then_yield_ids)
    state.next_id = then_state.next_id
    state.emit("} else {")
    else_state = _child_state(state)
    _emit_region_body(else_state, value.else_op_indices, value.else_values)
    _emit_region_yield(else_state, value.else_yield_ids)
    state.next_id = else_state.next_id
    state.emit("}")
    for result_id, name, result_type in zip(value.result_ids, result_names, result_types):
        state.names[result_id] = name
        state.types[result_id] = result_type


def _emit_if_effect(state, effect):
    condition = _require_name(state, effect.condition_id)
    condition_type = _require_type(state, effect.condition_id)
    if condition_type != "i1":
        raise ValueError(
            "tlx_wave new emitter cannot lower scf.if with non-scalar condition "
            f"{condition_type}"
        )
    state.emit(f"scf.if {condition} {{")
    then_state = _child_state(state)
    _emit_region_body(
        then_state,
        effect.then_op_indices,
        effect.then_values,
        effect.then_effects,
    )
    state.next_id = then_state.next_id
    if effect.else_op_indices or effect.else_effects:
        state.emit("} else {")
        else_state = _child_state(state)
        _emit_region_body(
            else_state,
            effect.else_op_indices,
            effect.else_values,
            effect.else_effects,
        )
        state.next_id = else_state.next_id
    state.emit("}")


def _emit_for(state, value):
    if all(result_id in state.names for result_id in value.result_ids):
        return
    _emit_for_loop(state, value)


def _emit_for_effect(state, effect):
    _emit_for_loop(state, effect)


def _emit_for_loop(state, loop):
    lower = _require_name(state, loop.lower_id)
    upper = _require_name(state, loop.upper_id)
    step = _require_name(state, loop.step_id)
    lower_type = _require_type(state, loop.lower_id)
    upper_type = _require_type(state, loop.upper_id)
    step_type = _require_type(state, loop.step_id)
    if lower_type != upper_type or lower_type != step_type:
        raise ValueError(
            "tlx_wave new emitter expected scf.for bounds to have one type, "
            f"got {lower_type}, {upper_type}, {step_type}"
        )
    lower_bound = _scf_for_index_bound(state, lower, lower_type)
    upper_bound = _scf_for_index_bound(state, upper, upper_type)
    step_bound = _scf_for_index_bound(state, step, step_type)
    body_has_memory_effects = bool(loop.body_effects)
    explicit_result_ids = getattr(loop, "result_ids", ())
    explicit_result_types = tuple(loop.result_types)
    explicit_result_type_texts = tuple(
        result_type.wave_type
        for result_type in explicit_result_types
        for _ in range(_loop_value_component_count(result_type))
    )
    loop_result_types = (
        ("!wave.mem.token",) if body_has_memory_effects else ()
    ) + explicit_result_type_texts
    loop_result_names, prefix = _loop_result_names(state, len(loop_result_types))
    induction_name = state.fresh()
    iter_bindings = []
    body_arg_bindings = []
    if body_has_memory_effects:
        token = _ordered_mem_token(state)
        token_arg = state.fresh()
        iter_bindings.append(f"{token_arg} = {token}")
        body_arg_bindings.append((None, token_arg, "!wave.mem.token"))
    for iter_arg_id, body_arg_id, result_type in zip(
        loop.iter_arg_ids, loop.body_arg_ids[1:], explicit_result_types
    ):
        count = _loop_value_component_count(result_type)
        init_components = _value_components_for_count(
            state,
            iter_arg_id,
            count,
            result_type.wave_type,
            "scf.for iter_arg",
        )
        body_args = []
        for init_name in init_components:
            body_arg = state.fresh()
            iter_bindings.append(f"{body_arg} = {init_name}")
            body_args.append(body_arg)
        body_arg_bindings.append(
            (body_arg_id, _pack_components(body_args), result_type.wave_type)
        )
    iter_text = ""
    if iter_bindings:
        iter_text = (
            f" iter_args({', '.join(iter_bindings)}) -> "
            f"({', '.join(loop_result_types)})"
        )
    state.emit(
        f"{prefix}scf.for {induction_name} = {lower_bound} to {upper_bound} "
        f"step {step_bound}{iter_text} {{"
    )
    body_state = _child_state(state)
    body_iv_type = _source_wave_type_for_value(state, loop.body_arg_ids[0])
    if body_iv_type != "index":
        body_iv = body_state.fresh()
        body_state.emit(
            f"{body_iv} = {_emit_scalar_intconvert_text(induction_name, 'index', body_iv_type)}"
        )
    else:
        body_iv = induction_name
    body_state.names[loop.body_arg_ids[0]] = body_iv
    body_state.types[loop.body_arg_ids[0]] = body_iv_type
    if body_has_memory_effects:
        body_state.mem_token = body_arg_bindings[0][1]
        explicit_body_arg_bindings = body_arg_bindings[1:]
    else:
        explicit_body_arg_bindings = body_arg_bindings
    for body_arg_id, name, value_type in explicit_body_arg_bindings:
        body_state.names[body_arg_id] = name
        body_state.types[body_arg_id] = value_type
    _emit_region_body(
        body_state,
        loop.body_op_indices,
        loop.body_values,
        loop.body_effects,
    )
    _emit_for_region_yield(body_state, loop.body_yield_ids, body_has_memory_effects)
    state.next_id = body_state.next_id
    state.emit("}")
    if body_has_memory_effects:
        state.mem_token = loop_result_names[0]
        explicit_loop_result_names = loop_result_names[1:]
    else:
        explicit_loop_result_names = loop_result_names
    offset = 0
    for result_id, result_type in zip(explicit_result_ids, explicit_result_types):
        count = _loop_value_component_count(result_type)
        names = explicit_loop_result_names[offset : offset + count]
        offset += count
        state.names[result_id] = _pack_components(names)
        state.types[result_id] = result_type.wave_type


def _loop_result_names(state, result_count):
    if result_count == 0:
        return (), ""
    result_base = state.fresh()
    if result_count == 1:
        return (result_base,), f"{result_base} = "
    return (
        tuple(f"{result_base}#{index}" for index in range(result_count)),
        f"{result_base}:{result_count} = ",
    )


def _scf_for_index_bound(state, value, value_type):
    if value_type == "index":
        return value
    if not _is_index_or_integer_scalar(value_type):
        raise ValueError(
            f"tlx_wave new emitter cannot use {value_type} as an scf.for bound"
        )
    cast = state.fresh()
    state.emit(
        f"{cast} = wave.index_expr <\"x\"> [\"x\"]({value}) : "
        f"({value_type}) -> index"
    )
    return cast


def _loop_value_component_count(converted_type):
    return max(1, int(getattr(converted_type, "component_count", 1)))


def _emit_for_region_yield(state, yield_ids, has_memory_result):
    operands = []
    types = []
    if has_memory_result:
        token = _ordered_mem_token(state)
        operands.append(token)
        types.append("!wave.mem.token")
    for value_id in yield_ids:
        value_name = _require_name(state, value_id)
        value_type = _require_type(state, value_id)
        components = _as_components(value_name)
        operands.extend(components)
        types.extend(value_type for _ in components)
    state.emit(f"scf.yield {', '.join(operands)} : {', '.join(types)}")


def _child_state(state):
    return _TextEmitterState(
        lines=state.lines,
        names=dict(state.names),
        types=dict(state.types),
        converted_values=state.converted_values,
        source_ops=state.source_ops,
        source_values=state.source_values,
        token_graph=state.token_graph,
        lds_offsets=state.lds_offsets,
        deferred_coordinate_values=state.deferred_coordinate_values,
        assume_condition_values=state.assume_condition_values,
        nonnegative_values=state.nonnegative_values,
        positive_values=state.positive_values,
        coordinate_cache=state.coordinate_cache,
        fragment_coord_root_cache=state.fragment_coord_root_cache,
        buffer_cache=state.buffer_cache,
        next_id=state.next_id,
        mem_token=state.mem_token,
        pending_read_tokens=state.pending_read_tokens,
        indent=state.indent + 2,
    )


def _emit_region_body(state, op_indices, converted_values, effects=()):
    effects_by_op_index = {}
    for effect in effects:
        op_index = getattr(effect, "op_index", -1)
        effects_by_op_index.setdefault(op_index, []).append(effect)
    for op_index in op_indices:
        op = state.source_ops[op_index]
        if op.name == "scf.yield":
            continue
        for result_id in op.results:
            if (
                result_id in state.deferred_coordinate_values
                or result_id in state.assume_condition_values
            ):
                continue
            value = converted_values.get(result_id)
            if value is None:
                continue
            _emit_value(state, value)
        for effect in effects_by_op_index.pop(op.index, ()):
            _emit_effect(state, effect)
    for op_index in sorted(effects_by_op_index):
        for effect in effects_by_op_index[op_index]:
            _emit_effect(state, effect)


def _emit_region_yield(state, yield_ids):
    operands = ", ".join(_require_name(state, value_id) for value_id in yield_ids)
    types = ", ".join(_require_type(state, value_id) for value_id in yield_ids)
    state.emit(f"scf.yield {operands} : {types}")


def _source_wave_type_for_value(state, value_id):
    source_type = state.source_values[value_id].type
    if source_type.kind == "scalar":
        return source_type.raw
    if source_type.kind == "tensor":
        if source_type.element_type == "i1":
            return f"!wave.mask<{_product(source_type.shape)}>"
        return f"!wave.simd<{source_type.element_type}, {_product(source_type.shape)}>"
    if source_type.kind == "pointer":
        address_space = "wave.shared" if source_type.address_space == 3 else "wave.global"
        return f"!wave.ptr<#{address_space}, {source_type.pointee_type or 'i8'}>"
    if source_type.kind == "token":
        return "!wave.mem.token"
    if source_type.kind == "memdesc":
        return _shared_pointer_type_for_source(state, value_id)
    return source_type.raw


def _emit_cast(state, value):
    operand_name = _require_name(state, value.operand_id)
    operand_type = _require_type(state, value.operand_id)
    result_type = value.converted_type.wave_type
    if _is_fragment_wave_type(operand_type) and value.kind == "fpconvert":
        state.names[value.value_id] = operand_name
        state.types[value.value_id] = operand_type
        return
    if operand_type == result_type:
        state.names[value.value_id] = operand_name
        state.types[value.value_id] = result_type
        return
    name = state.fresh()
    state.emit(
        f"{name} = wave.cast {value.kind} {operand_name} : "
        f"{operand_type} -> {result_type}"
    )
    state.names[value.value_id] = name
    state.types[value.value_id] = result_type


def _emit_local_alloc(state, value):
    source_type = _source_type(state, value.value_id)
    element_type = source_type.element_type or "i8"
    offset = state.lds_offsets.get(value.value_id, 0)
    offset_attr = "" if offset == 0 else f" {{offset = {offset} : i64}}"
    wave_type = f"!wave.ptr<#wave.shared, {element_type}>"
    name = state.fresh()
    state.emit(f"{name} = wave.lds_base{offset_attr} : {wave_type}")
    state.names[value.value_id] = name
    state.types[value.value_id] = wave_type


def _emit_memdesc_index(state, value):
    base = _require_name(state, value.memdesc_id)
    base_type = _require_type(state, value.memdesc_id)
    result_type = _shared_pointer_type_for_source(state, value.value_id)
    if base_type != result_type:
        raise ValueError(
            "tlx_wave new emitter cannot index memdesc with changed pointer type: "
            f"{base_type} -> {result_type}"
        )
    elements_per_slot = _memdesc_index_elements_per_slot(state, value)
    static_index = _constant_int_record(value.index)
    if static_index is not None:
        element_offset = int(static_index) * int(elements_per_slot)
        offset_name = None
        offset_type = "i32"
    else:
        index_name = _require_name(state, value.index_id)
        index_type = _require_type(state, value.index_id)
        if index_type.startswith("!wave.simd<") or index_type.startswith("!wave.mask<"):
            raise ValueError(
                "tlx_wave new emitter cannot lower lane-varying memdesc_index "
                f"slot of type {index_type}"
            )
        offset_name = index_name
        offset_type = index_type
        element_offset = 0
        if int(elements_per_slot) != 1:
            slot_stride = state.fresh()
            scaled = state.fresh()
            state.emit(f"{slot_stride} = arith.constant {int(elements_per_slot)} : {index_type}")
            state.emit(
                f"{scaled} = wave.binary muli {index_name}, {slot_stride} : "
                f"{index_type}, {index_type} -> {index_type}"
            )
            offset_name = scaled
    if element_offset == 0:
        if offset_name is None:
            state.names[value.value_id] = base
            state.types[value.value_id] = result_type
            return
    else:
        offset_name = state.fresh()
        state.emit(f"{offset_name} = arith.constant {element_offset} : {offset_type}")
    ptr = state.fresh()
    state.emit(
        f"{ptr} = wave.ptr_add {base}, {offset_name} : "
        f"{base_type}, {offset_type} -> {result_type}"
    )
    state.names[value.value_id] = ptr
    state.types[value.value_id] = result_type


def _emit_local_load(state, value):
    if value.converted_type.kind == "fragment":
        _emit_fragment_local_load(state, value)
        return
    result_type = value.converted_type.wave_type
    ptr_components, ptr_type = _local_linear_pointers(
        state,
        value.memdesc_id,
        value.value_id,
        result_type,
        value.converted_type.component_count,
    )
    input_token = _memory_token_operand(state, value.token_id)
    components = []
    load_tokens = []
    for ptr in ptr_components:
        pair = state.fresh()
        state.emit(
            f"{pair}:2 = wave.load {ptr} after {input_token} : "
            f"({ptr_type}, !wave.mem.token) -> ({result_type}, !wave.mem.token)"
        )
        components.append(f"{pair}#0")
        load_tokens.append(f"{pair}#1")
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type
    _remember_read_tokens(state, load_tokens)


def _remember_read_tokens(state, tokens):
    state.pending_read_tokens = tuple(
        dict.fromkeys((*state.pending_read_tokens, *(token for token in tokens if token)))
    )


def _join_memory_tokens(state, tokens, fallback_token):
    unique_tokens = tuple(dict.fromkeys(token for token in tokens if token))
    if not unique_tokens:
        return fallback_token
    if len(unique_tokens) == 1:
        return unique_tokens[0]
    name = state.fresh()
    deps = ", ".join(unique_tokens)
    dep_types = ", ".join("!wave.mem.token" for _ in unique_tokens)
    state.emit(f"{name} = wave.join {deps} : {dep_types} -> !wave.mem.token")
    return name


def _emit_fragment_local_load(state, value):
    source_type = _source_type(state, value.value_id)
    memdesc_type = _source_type(state, value.memdesc_id)
    padded = _padded_shared_memdesc_info(memdesc_type)
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if (
        padded is None
        and not _is_identity_shared_memdesc(memdesc_type)
        and swizzled is None
    ):
        raise ValueError(
            "tlx_wave new emitter cannot lower fragment ttg.local_load from "
            "non-identity shared layout yet; source encoding="
            f"{source_type.encoding}, memdesc encoding={memdesc_type.encoding}"
        )
    dot = _dot_operand_encoding_info(source_type)
    if dot is None:
        raise ValueError(
            "tlx_wave new emitter supports fragment local_load only for "
            f"#ttg.dot_op tensors, got {source_type.encoding}"
        )
    mma = _mma_shape_for_parent(dot.parent, "ttg.local_load")
    if dot.op_idx == 0:
        tile_shape = (
            _tile_rep_count(source_type.shape[0], mma.output_tile_shape[0]),
            _tile_rep_count(source_type.shape[1], mma.k_dim),
        )
    else:
        tile_shape = (
            _tile_rep_count(source_type.shape[0], mma.k_dim),
            _tile_rep_count(source_type.shape[1], mma.output_tile_shape[1]),
        )
    base = _require_name(state, value.memdesc_id)
    base_type = _require_type(state, value.memdesc_id)
    i32_base, i32_base_type = _ptr_cast(state, base, base_type, "i32")
    input_token = _memory_token_operand(state, value.token_id)
    names = []
    load_tokens = []
    for row in range(tile_shape[0]):
        for col in range(tile_shape[1]):
            tile_offsets = _dot_operand_tile_offsets(dot, mma, row, col)
            if _can_emit_b16_transpose_fragment_load(
                source_type, memdesc_type, dot, mma
            ):
                fragment, load_token = _emit_b16_transpose_fragment_load_tile(
                    state,
                    base,
                    base_type,
                    source_type,
                    memdesc_type,
                    value.converted_type.wave_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            elif padded is not None:
                fragment, load_token = _emit_padded_fragment_load_tile(
                    state,
                    i32_base,
                    i32_base_type,
                    source_type,
                    memdesc_type,
                    padded,
                    value.converted_type.wave_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            elif _is_identity_shared_memdesc(memdesc_type):
                fragment, load_token = _emit_dense_fragment_load_tile(
                    state,
                    i32_base,
                    i32_base_type,
                    source_type,
                    value.converted_type.wave_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            else:
                fragment, load_token = _emit_swizzled_fragment_load_tile(
                    state,
                    i32_base,
                    i32_base_type,
                    source_type,
                    memdesc_type,
                    swizzled,
                    value.converted_type.wave_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            names.append(fragment)
            load_tokens.append(load_token)
    expected = value.converted_type.component_count
    if len(names) != expected:
        raise ValueError(
            "tlx_wave new emitter internal fragment tile mismatch: "
            f"loaded {len(names)}, expected {expected}"
        )
    state.names[value.value_id] = names[0] if len(names) == 1 else tuple(names)
    state.types[value.value_id] = value.converted_type.wave_type
    _remember_read_tokens(state, load_tokens)


def _emit_dense_fragment_load_tile(
    state,
    base,
    base_type,
    source_type,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    tile_base = _dense_tile_base_dwords(source_type, tile_offsets)
    lane = state.fresh()
    offset = state.fresh()
    ptr = state.fresh()
    pair = state.fresh()
    frag = state.fresh()
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    ptr_type = f"!wave.simd<!wave.ptr<#wave.shared, i32>, {wave_size}>"
    load_type = f"!wave.simd<vector<{registers}xi32>, {wave_size}>"
    state.emit(f"{lane} = wave.workitem_id 0 : {simd_i32}")
    if tile_base:
        base_const = state.fresh()
        base_splat = state.fresh()
        reg_const = state.fresh()
        reg_splat = state.fresh()
        lane_scaled = state.fresh()
        state.emit(f"{base_const} = arith.constant {tile_base} : i32")
        state.emit(f"{base_splat} = wave.splat {base_const} : i32 -> {simd_i32}")
        state.emit(f"{reg_const} = arith.constant {registers} : i32")
        state.emit(f"{reg_splat} = wave.splat {reg_const} : i32 -> {simd_i32}")
        state.emit(
            f"{lane_scaled} = wave.binary muli {lane}, {reg_splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        state.emit(
            f"{offset} = wave.binary addi {base_splat}, {lane_scaled} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
    else:
        reg_const = state.fresh()
        reg_splat = state.fresh()
        state.emit(f"{reg_const} = arith.constant {registers} : i32")
        state.emit(f"{reg_splat} = wave.splat {reg_const} : i32 -> {simd_i32}")
        state.emit(
            f"{offset} = wave.binary muli {lane}, {reg_splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
    offset = _emit_i32_simd_range_assume(
        state,
        offset,
        simd_i32,
        _fragment_load_dword_offset_upper(source_type, registers),
    )
    state.emit(f"{ptr} = wave.ptr_add {base}, {offset} : {base_type}, {simd_i32} -> {ptr_type}")
    state.emit(
        f"{pair}:2 = wave.load {ptr} after {token} : "
        f"({ptr_type}, !wave.mem.token) -> ({load_type}, !wave.mem.token)"
    )
    state.emit(f"{frag} = waveamd.fragment_pack {pair}#0 : {load_type} -> {fragment_type}")
    return frag, f"{pair}#1"


def _emit_padded_fragment_load_tile(
    state,
    base,
    base_type,
    source_type,
    memdesc_type,
    padded,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    dword_offset = _emit_padded_fragment_dword_offset(
        state,
        source_type,
        memdesc_type,
        padded,
        tile_offsets,
        registers,
        wave_size,
    )
    ptr = state.fresh()
    pair = state.fresh()
    frag = state.fresh()
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    ptr_type = f"!wave.simd<!wave.ptr<#wave.shared, i32>, {wave_size}>"
    load_type = f"!wave.simd<vector<{registers}xi32>, {wave_size}>"
    state.emit(f"{ptr} = wave.ptr_add {base}, {dword_offset} : {base_type}, {simd_i32} -> {ptr_type}")
    state.emit(
        f"{pair}:2 = wave.load {ptr} after {token} : "
        f"({ptr_type}, !wave.mem.token) -> ({load_type}, !wave.mem.token)"
    )
    state.emit(f"{frag} = waveamd.fragment_pack {pair}#0 : {load_type} -> {fragment_type}")
    return frag, f"{pair}#1"


def _can_emit_b16_transpose_fragment_load(source_type, memdesc_type, dot, mma):
    if dot.op_idx != 1:
        return False
    if tuple(mma.instr_shape) != (32, 32, 16):
        return False
    if int(mma.wave_size) != 64 or int(mma.operand_registers) != 4:
        return False
    if source_type.element_type not in {"f16", "bf16"}:
        return False
    if source_type.element_byte_width != 2 or memdesc_type.element_byte_width != 2:
        return False
    if memdesc_type.element_type != source_type.element_type:
        return False
    if _padded_shared_memdesc_info(memdesc_type) is not None:
        return True
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    return (
        swizzled is not None
        and int(swizzled.vec) == 8
        and int(swizzled.per_phase) == 4
        and int(swizzled.max_phase) == 4
        and tuple(swizzled.order) == (1, 0)
    )


def _emit_b16_transpose_fragment_load_tile(
    state,
    base,
    base_type,
    source_type,
    memdesc_type,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    elements_per_lane = int(registers) * (4 // int(memdesc_type.element_byte_width))
    if elements_per_lane != 8:
        raise ValueError(
            "tlx_wave new emitter cannot lower b16 transpose fragment load "
            f"with {elements_per_lane} elements per lane"
        )
    _validate_transpose_fragment_packets(
        source_type,
        memdesc_type,
        tile_offsets,
        elements_per_lane,
        int(memdesc_type.element_byte_width),
        int(wave_size),
    )
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    ptr_type = f"!wave.simd<!wave.ptr<#wave.shared, {source_type.element_type}>, {wave_size}>"
    load_type = f"!wave.simd<vector<4x{source_type.element_type}>, {wave_size}>"
    element_type = f"!wave.simd<{source_type.element_type}, {wave_size}>"
    components = []
    for chunk in range(2):
        chunk_offsets = _transpose_fragment_chunk_tile_offsets(tile_offsets, chunk)
        offset, offset_type = _emit_fragment_element_offset(
            state,
            source_type,
            memdesc_type,
            chunk_offsets,
            elements_per_lane,
            wave_size,
            access_elements=4,
        )
        ptr = state.fresh()
        pair = state.fresh()
        state.emit(
            f"{ptr} = wave.ptr_add {base}, {offset} : "
            f"{base_type}, {offset_type} -> {ptr_type}"
        )
        state.emit(
            f"{pair}:2 = waveamd.transpose_load {ptr} after {token} : "
            f"({ptr_type}, !wave.mem.token) -> ({load_type}, !wave.mem.token)"
        )
        token = f"{pair}#1"
        for component in range(4):
            extracted = state.fresh()
            state.emit(
                f"{extracted} = wave.extract {pair}#0[{component}] : "
                f"{load_type} -> {element_type}"
            )
            components.append(extracted)
    packed_type = f"!wave.simd<vector<8x{source_type.element_type}>, {wave_size}>"
    packed = state.fresh()
    component_types = ", ".join(element_type for _ in components)
    state.emit(
        f"{packed} = wave.pack {', '.join(components)} : "
        f"{component_types} -> {packed_type}"
    )
    frag = state.fresh()
    state.emit(f"{frag} = waveamd.fragment_pack {packed} : {packed_type} -> {fragment_type}")
    return frag, token


def _transpose_fragment_chunk_tile_offsets(tile_offsets, chunk):
    offsets = list(tile_offsets)
    offsets[-1] += 4 * int(chunk)
    return tuple(offsets)


def _emit_fragment_element_offset(
    state,
    source_type,
    memdesc_type,
    tile_offsets,
    elements_per_lane,
    wave_size,
    *,
    access_elements,
):
    offset_type = f"!wave.simd<i32, {wave_size}>"
    padded = _padded_shared_memdesc_info(memdesc_type)
    if padded is not None:
        offset = _emit_padded_fragment_element_offset(
            state,
            source_type,
            memdesc_type,
            padded,
            tile_offsets,
            elements_per_lane,
            wave_size,
        )
    else:
        swizzled = _swizzled_shared_memdesc_info(memdesc_type)
        if swizzled is not None and not _is_identity_shared_memdesc(memdesc_type):
            offset, offset_type = _emit_swizzled_fragment_element_offset(
                state,
                source_type,
                memdesc_type,
                swizzled,
                tile_offsets,
                elements_per_lane,
                wave_size,
            )
        else:
            offset = _emit_dense_fragment_element_offset(
                state, source_type, tile_offsets, elements_per_lane, wave_size
            )
    upper = _fragment_load_element_offset_upper(memdesc_type, access_elements)
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    if offset_type == simd_i32:
        offset = _emit_i32_simd_range_assume(state, offset, simd_i32, upper)
    return offset, offset_type


def _emit_dense_fragment_element_offset(
    state,
    source_type,
    tile_offsets,
    elements_per_lane,
    wave_size,
):
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    return _emit_lane_scaled_offset(
        state, int(wave_size), int(elements_per_lane), tile_base
    )


def _emit_padded_fragment_element_offset(
    state,
    source_type,
    memdesc_type,
    padded,
    tile_offsets,
    elements_per_lane,
    wave_size,
):
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    lane = _emit_assumed_workitem_id(state, int(wave_size))
    logical = _emit_simd_binary_const(
        state, "muli", lane, simd_i32, int(elements_per_lane)
    )
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    if tile_base:
        logical = _emit_simd_add_const(state, logical, simd_i32, tile_base)
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        quotient = _emit_simd_binary_const(
            state, "divui", logical, simd_i32, int(interval)
        )
        pad = _emit_simd_binary_const(
            state, "muli", quotient, simd_i32, int(padding)
        )
        encoded = _emit_simd_binary(state, "addi", encoded, pad, simd_i32)
    return encoded


def _emit_swizzled_fragment_element_offset(
    state,
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    elements_per_lane,
    wave_size,
):
    return _emit_swizzled_fragment_index_offset(
        state,
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        elements_per_lane,
        wave_size,
        elements_per_offset_unit=1,
    )


def _emit_swizzled_fragment_index_offset(
    state,
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    elements_per_lane,
    wave_size,
    *,
    elements_per_offset_unit,
):
    if len(memdesc_type.shape) != 2:
        raise ValueError(
            "tlx_wave new emitter supports swizzled fragment loads only for "
            f"rank-2 shared memdescs, got shape={memdesc_type.shape}"
        )
    if len(source_type.shape) != 2:
        raise ValueError(
            "tlx_wave new emitter supports swizzled fragment loads only for "
            f"rank-2 source tensors, got shape={source_type.shape}"
        )
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    simd_index = f"!wave.simd<index, {wave_size}>"
    lane = _emit_assumed_workitem_id(state, int(wave_size))
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    logical = f"{int(elements_per_lane)}*lid"
    if tile_base:
        logical = f"{int(tile_base)} + {logical}"
    cols = int(source_type.shape[-1])
    unit = int(elements_per_offset_unit)
    vec = int(swizzled.vec)
    cols = int(memdesc_type.shape[-1])
    if int(elements_per_lane) % vec == 0 and cols % vec == 0:
        low_col = int(tile_base) % vec
        if unit != 1 and (cols % unit or vec % unit or low_col % unit):
            raise ValueError(
                "tlx_wave new emitter cannot divide swizzled fragment offset "
                f"by {unit}: cols={cols}, vec={vec}, low_col={low_col}"
            )
        groups_per_row = cols // vec
        logical_group = _linear_lid_expr(tile_base // vec, elements_per_lane // vec)
        row = f"floor(1/{groups_per_row}*({logical_group}))"
        col_group = f"Mod({logical_group}, {groups_per_row})"
        phase = (
            f"Mod(floor(1/{int(swizzled.per_phase)}*({row})), "
            f"{int(swizzled.max_phase)})"
        )
        if unit == 1:
            expr = f"{cols}*({row}) + {vec}*xor({col_group}, {phase})"
            if low_col:
                expr = f"{expr} + {low_col}"
        else:
            expr = f"{cols // unit}*({row}) + {vec // unit}*xor({col_group}, {phase})"
            if low_col:
                expr = f"{expr} + {low_col // unit}"
    else:
        row = f"floor(1/{cols}*({logical}))"
        col = f"Mod({logical}, {cols})"
        phase = (
            f"Mod(floor(1/{int(swizzled.per_phase)}*({row})), "
            f"{int(swizzled.max_phase)})"
        )
        col_group = f"floor(1/{int(swizzled.vec)}*({col}))"
        swizzled_col = (
            f"{vec}*xor({col_group}, {phase}) + "
            f"Mod({col}, {vec})"
        )
        element = f"{cols}*({row}) + {swizzled_col}"
        if unit != 1:
            expr = f"floor(1/{unit}*({element}))"
        else:
            expr = element
    offset = state.fresh()
    state.emit(
        f"{offset} = wave.index_expr <\"{expr}\"> "
        f"assuming [#wave.pred<\"lid >= 0\">, "
        f"#wave.pred<\"-{int(wave_size) - 1} + lid <= 0\">] "
        f"[\"lid\"]({lane}) : ({simd_i32}) -> {simd_index}"
    )
    return offset, simd_index


def _linear_lid_expr(constant, scale):
    constant = int(constant)
    scale = int(scale)
    if scale == 0:
        return str(constant)
    term = "lid" if scale == 1 else f"{scale}*lid"
    if constant == 0:
        return term
    return f"{constant} + {term}"


def _emit_swizzled_element_from_coords(state, row, col, simd_i32, memdesc_type, swizzled):
    row_phase = _emit_simd_binary_const(
        state, "divui", row, simd_i32, int(swizzled.per_phase)
    )
    phase = _emit_simd_binary_const(
        state, "remui", row_phase, simd_i32, int(swizzled.max_phase)
    )
    col_group = _emit_simd_binary_const(
        state, "divui", col, simd_i32, int(swizzled.vec)
    )
    swizzled_group = _emit_simd_binary(
        state, "xori", col_group, phase, simd_i32
    )
    swizzled_base = _emit_simd_binary_const(
        state, "muli", swizzled_group, simd_i32, int(swizzled.vec)
    )
    col_in_vec = _emit_simd_binary_const(
        state, "remui", col, simd_i32, int(swizzled.vec)
    )
    swizzled_col = _emit_simd_binary(
        state, "addi", swizzled_base, col_in_vec, simd_i32
    )
    row_scaled = _emit_simd_binary_const(
        state, "muli", row, simd_i32, int(memdesc_type.shape[-1])
    )
    return _emit_simd_binary(state, "addi", row_scaled, swizzled_col, simd_i32)


def _emit_swizzled_fragment_load_tile(
    state,
    base,
    base_type,
    source_type,
    memdesc_type,
    swizzled,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    dword_offset, dword_offset_type = _emit_swizzled_fragment_dword_offset(
        state,
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        registers,
        wave_size,
    )
    ptr = state.fresh()
    pair = state.fresh()
    frag = state.fresh()
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    ptr_type = f"!wave.simd<!wave.ptr<#wave.shared, i32>, {wave_size}>"
    load_type = f"!wave.simd<vector<{registers}xi32>, {wave_size}>"
    state.emit(
        f"{ptr} = wave.ptr_add {base}, {dword_offset} : "
        f"{base_type}, {dword_offset_type} -> {ptr_type}"
    )
    state.emit(
        f"{pair}:2 = wave.load {ptr} after {token} : "
        f"({ptr_type}, !wave.mem.token) -> ({load_type}, !wave.mem.token)"
    )
    state.emit(f"{frag} = waveamd.fragment_pack {pair}#0 : {load_type} -> {fragment_type}")
    return frag, f"{pair}#1"


def _emit_swizzled_fragment_dword_offset(
    state,
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    registers,
    wave_size,
):
    byte_width = memdesc_type.element_byte_width or source_type.element_byte_width
    if byte_width is None:
        raise ValueError(
            "tlx_wave new emitter cannot lower swizzled fragment load without "
            "known element byte width"
        )
    elements_per_lane = int(registers) * (4 // int(byte_width))
    _validate_swizzled_fragment_packet(
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        elements_per_lane,
        int(byte_width),
        int(wave_size),
    )
    elements_per_dword = 4 // int(byte_width)
    return _emit_swizzled_fragment_index_offset(
        state,
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        elements_per_lane,
        wave_size,
        elements_per_offset_unit=elements_per_dword,
    )


def _emit_padded_fragment_dword_offset(
    state,
    source_type,
    memdesc_type,
    padded,
    tile_offsets,
    registers,
    wave_size,
):
    byte_width = memdesc_type.element_byte_width or source_type.element_byte_width
    if byte_width is None:
        raise ValueError(
            "tlx_wave new emitter cannot lower padded fragment load without "
            "known element byte width"
        )
    elements_per_lane = int(registers) * (4 // int(byte_width))
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    _validate_padded_fragment_packet(padded, tile_base, elements_per_lane, int(byte_width))
    simd_i32 = f"!wave.simd<i32, {wave_size}>"
    affine = _padded_fragment_affine_dword_offset(
        padded,
        tile_base,
        elements_per_lane,
        int(byte_width),
        int(wave_size),
    )
    upper = _fragment_load_dword_offset_upper(memdesc_type, registers)
    if affine is not None:
        offset = _emit_lane_scaled_offset(
            state, int(wave_size), affine[0], affine[1]
        )
        return _emit_i32_simd_range_assume(state, offset, simd_i32, upper)
    lane = state.fresh()
    reg_const = state.fresh()
    reg_splat = state.fresh()
    logical = state.fresh()
    state.emit(f"{lane} = wave.workitem_id 0 : {simd_i32}")
    state.emit(f"{reg_const} = arith.constant {elements_per_lane} : i32")
    state.emit(f"{reg_splat} = wave.splat {reg_const} : i32 -> {simd_i32}")
    state.emit(
        f"{logical} = wave.binary muli {lane}, {reg_splat} : "
        f"{simd_i32}, {simd_i32} -> {simd_i32}"
    )
    if tile_base:
        base_const = state.fresh()
        base_splat = state.fresh()
        shifted = state.fresh()
        state.emit(f"{base_const} = arith.constant {tile_base} : i32")
        state.emit(f"{base_splat} = wave.splat {base_const} : i32 -> {simd_i32}")
        state.emit(
            f"{shifted} = wave.binary addi {logical}, {base_splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        logical = shifted
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        div_const = state.fresh()
        div_splat = state.fresh()
        quotient = state.fresh()
        pad_const = state.fresh()
        pad_splat = state.fresh()
        pad = state.fresh()
        with_pad = state.fresh()
        state.emit(f"{div_const} = arith.constant {int(interval)} : i32")
        state.emit(f"{div_splat} = wave.splat {div_const} : i32 -> {simd_i32}")
        state.emit(
            f"{quotient} = wave.binary divui {logical}, {div_splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        state.emit(f"{pad_const} = arith.constant {int(padding)} : i32")
        state.emit(f"{pad_splat} = wave.splat {pad_const} : i32 -> {simd_i32}")
        state.emit(
            f"{pad} = wave.binary muli {quotient}, {pad_splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        state.emit(
            f"{with_pad} = wave.binary addi {encoded}, {pad} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        encoded = with_pad
    elements_per_dword = 4 // int(byte_width)
    if elements_per_dword == 1:
        return encoded
    div_const = state.fresh()
    div_splat = state.fresh()
    dwords = state.fresh()
    state.emit(f"{div_const} = arith.constant {elements_per_dword} : i32")
    state.emit(f"{div_splat} = wave.splat {div_const} : i32 -> {simd_i32}")
    state.emit(
        f"{dwords} = wave.binary divui {encoded}, {div_splat} : "
        f"{simd_i32}, {simd_i32} -> {simd_i32}"
    )
    return _emit_i32_simd_range_assume(state, dwords, simd_i32, upper)


def _fragment_load_dword_offset_upper(source_type, registers):
    total_bytes = _local_alloc_size_bytes(source_type)
    access_bytes = int(registers) * 4
    upper_bytes = total_bytes - access_bytes
    if upper_bytes < 0:
        raise ValueError(
            "tlx_wave new emitter cannot bound fragment load offset: "
            f"{access_bytes} byte load exceeds {total_bytes} byte allocation"
        )
    return upper_bytes // 4


def _fragment_load_element_offset_upper(source_type, access_elements):
    if source_type.element_byte_width is None:
        raise ValueError(
            "tlx_wave new emitter cannot bound fragment load element offset: "
            "unknown element byte width"
        )
    total_elements = _local_alloc_size_bytes(source_type) // int(
        source_type.element_byte_width
    )
    upper = total_elements - int(access_elements)
    if upper < 0:
        raise ValueError(
            "tlx_wave new emitter cannot bound fragment load element offset: "
            f"{access_elements} element load exceeds allocation"
        )
    return upper


def _emit_i32_simd_range_assume(state, value, value_type, upper):
    if not value_type.startswith("!wave.simd<i32,"):
        return value
    assumed = state.fresh()
    state.emit(
        f"{assumed} = wave.assume {value} as \"x\" "
        f"[#wave.pred<\"x >= 0\">, #wave.pred<\"-{int(upper)} + x <= 0\">] "
        f": {value_type}"
    )
    return assumed


def _padded_fragment_affine_dword_offset(
    padded,
    tile_base,
    elements_per_lane,
    byte_width,
    wave_size,
):
    elements_per_dword = 4 // int(byte_width)
    max_lane_elements = (int(wave_size) - 1) * int(elements_per_lane)
    pad_elements = 0
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        interval = int(interval)
        padding = int(padding)
        if int(tile_base) % interval + max_lane_elements >= interval:
            return None
        pad_elements += (int(tile_base) // interval) * padding
    constant_elements = int(tile_base) + pad_elements
    if int(elements_per_lane) % elements_per_dword:
        return None
    if constant_elements % elements_per_dword:
        return None
    return (
        int(elements_per_lane) // elements_per_dword,
        constant_elements // elements_per_dword,
    )


def _emit_lane_scaled_offset(state, wave_size, scale, constant):
    simd_i32 = f"!wave.simd<i32, {int(wave_size)}>"
    offset = _emit_assumed_workitem_id(state, int(wave_size))
    if int(scale) != 1:
        scale_const = state.fresh()
        scale_splat = state.fresh()
        scaled = state.fresh()
        state.emit(f"{scale_const} = arith.constant {int(scale)} : i32")
        state.emit(f"{scale_splat} = wave.splat {scale_const} : i32 -> {simd_i32}")
        state.emit(
            f"{scaled} = wave.binary muli {offset}, {scale_splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        offset = scaled
    if int(constant) != 0:
        const = state.fresh()
        splat = state.fresh()
        shifted = state.fresh()
        state.emit(f"{const} = arith.constant {int(constant)} : i32")
        state.emit(f"{splat} = wave.splat {const} : i32 -> {simd_i32}")
        state.emit(
            f"{shifted} = wave.binary addi {offset}, {splat} : "
            f"{simd_i32}, {simd_i32} -> {simd_i32}"
        )
        offset = shifted
    return offset


def _validate_padded_fragment_packet(padded, tile_base, elements_per_lane, byte_width):
    if 4 % int(byte_width):
        raise ValueError(
            "tlx_wave new emitter supports padded fragment loads only when "
            f"element byte width divides a dword, got {byte_width}"
        )
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        if int(interval) % int(elements_per_lane):
            raise ValueError(
                "tlx_wave new emitter cannot lower padded fragment load whose "
                f"{elements_per_lane}-element packets can cross padding interval "
                f"{interval}"
            )
        if int(tile_base) % int(elements_per_lane):
            raise ValueError(
                "tlx_wave new emitter cannot lower padded fragment load with "
                f"tile base {tile_base} not aligned to packet width "
                f"{elements_per_lane}"
            )
        if int(padding) % (4 // int(byte_width)):
            raise ValueError(
                "tlx_wave new emitter cannot lower padded fragment load with "
                f"padding {padding} not aligned to i32 load units"
            )


def _validate_swizzled_fragment_packet(
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    elements_per_lane,
    byte_width,
    wave_size,
):
    if 4 % int(byte_width):
        raise ValueError(
            "tlx_wave new emitter supports swizzled fragment loads only when "
            f"element byte width divides a dword, got {byte_width}"
        )
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    for lane in range(int(wave_size)):
        first = None
        for element in range(int(elements_per_lane)):
            logical = tile_base + lane * int(elements_per_lane) + element
            coords = _row_major_coords_static(source_type.shape, logical)
            for dim, coord in enumerate(coords):
                if coord < 0 or coord >= int(memdesc_type.shape[dim]):
                    raise ValueError(
                        "tlx_wave new emitter cannot lower swizzled fragment "
                        f"load: coordinate {coords} exceeds memdesc shape "
                        f"{memdesc_type.shape}"
                    )
            byte_offset = _swizzled_shared_static_byte_offset(
                memdesc_type,
                coords,
                swizzled,
                "ttg.local_load fragment source",
            )
            if first is None:
                first = byte_offset
                if first % 4:
                    raise ValueError(
                        "tlx_wave new emitter cannot lower swizzled fragment "
                        f"load: packet base byte offset {first} is not "
                        "dword-aligned"
                    )
                continue
            expected = first + element * int(byte_width)
            if byte_offset != expected:
                raise ValueError(
                    "tlx_wave new emitter cannot lower swizzled fragment load: "
                    f"lane {lane} packet is not physically contiguous"
                )


def _validate_transpose_fragment_packets(
    source_type,
    memdesc_type,
    tile_offsets,
    elements_per_lane,
    byte_width,
    wave_size,
):
    for chunk in range(2):
        chunk_offsets = _transpose_fragment_chunk_tile_offsets(tile_offsets, chunk)
        tile_base = _dense_tile_base_elements(source_type, chunk_offsets)
        for lane in range(int(wave_size)):
            first = None
            for element in range(4):
                logical = tile_base + lane * int(elements_per_lane) + element
                coords = _row_major_coords_static(source_type.shape, logical)
                for dim, coord in enumerate(coords):
                    if coord < 0 or coord >= int(memdesc_type.shape[dim]):
                        raise ValueError(
                            "tlx_wave new emitter cannot lower transpose "
                            f"fragment load: coordinate {coords} exceeds "
                            f"memdesc shape {memdesc_type.shape}"
                        )
                byte_offset = _memdesc_static_byte_offset(
                    memdesc_type,
                    coords,
                    "ttg.local_load transpose fragment source",
                )
                if first is None:
                    first = byte_offset
                    if first % 8:
                        raise ValueError(
                            "tlx_wave new emitter cannot lower transpose "
                            f"fragment load: packet base byte offset {first} "
                            "is not 8-byte aligned"
                        )
                    continue
                expected = first + element * int(byte_width)
                if byte_offset != expected:
                    raise ValueError(
                        "tlx_wave new emitter cannot lower transpose fragment "
                        f"load: lane {lane} chunk {chunk} packet is not "
                        "physically contiguous"
                    )


def _dot_operand_tile_offsets(info, mma, row, col):
    if info.op_idx == 0:
        return (
            int(row) * int(mma.output_tile_shape[0]),
            int(col) * int(mma.k_dim),
        )
    return (
        int(row) * int(mma.k_dim),
        int(col) * int(mma.output_tile_shape[1]),
    )


def _dense_tile_base_dwords(source_type, tile_offsets):
    if source_type.element_byte_width is None:
        raise ValueError(
            "tlx_wave new emitter cannot compute fragment tile offset for "
            f"{source_type.raw}: unknown element byte width"
        )
    element_offset = 0
    stride = 1
    for dim in reversed(range(len(source_type.shape))):
        element_offset += int(tile_offsets[dim]) * stride
        stride *= int(source_type.shape[dim])
    byte_offset = element_offset * int(source_type.element_byte_width)
    if byte_offset % 4:
        raise ValueError(
            "tlx_wave new emitter cannot compute fragment tile offset: "
            f"{byte_offset} byte offset is not i32 aligned"
        )
    return byte_offset // 4


def _dense_tile_base_elements(source_type, tile_offsets):
    element_offset = 0
    stride = 1
    for dim in reversed(range(len(source_type.shape))):
        element_offset += int(tile_offsets[dim]) * stride
        stride *= int(source_type.shape[dim])
    return element_offset


def _ptr_cast(state, name, source_type, element_type):
    result_type = _ptr_type_with_element(source_type, element_type)
    if source_type == result_type:
        return name, source_type
    result = state.fresh()
    state.emit(f"{result} = wave.ptr_cast {name} : {source_type} -> {result_type}")
    return result, result_type


def _ptr_type_with_element(pointer_type, element_type):
    if not pointer_type.startswith("!wave.ptr<#wave."):
        raise ValueError(f"tlx_wave new emitter expected pointer type, got {pointer_type}")
    space = pointer_type.split(",", 1)[0].removeprefix("!wave.ptr<")
    return f"!wave.ptr<{space}, {element_type}>"


def _is_identity_shared_memdesc(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_swizzled_shared_encoding"):
        return False
    order = tuple(int(value) for value in _attr_value(attr, "get_swizzled_shared_order"))
    return (
        int(_attr_value(attr, "get_swizzled_shared_vec")) == 1
        and int(_attr_value(attr, "get_swizzled_shared_per_phase")) == 1
        and int(_attr_value(attr, "get_swizzled_shared_max_phase")) == 1
        and order == tuple(reversed(range(len(source_type.shape))))
    )


def _swizzled_shared_memdesc_info(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_swizzled_shared_encoding"):
        return None
    info = _SwizzledSharedEncodingInfo(
        int(_attr_value(attr, "get_swizzled_shared_vec")),
        int(_attr_value(attr, "get_swizzled_shared_per_phase")),
        int(_attr_value(attr, "get_swizzled_shared_max_phase")),
        tuple(int(value) for value in _attr_value(attr, "get_swizzled_shared_order")),
    )
    if _is_identity_shared_memdesc(source_type):
        return info
    _validate_swizzled_shared_memdesc(source_type, info, "swizzled shared memdesc")
    return info


def _validate_swizzled_shared_memdesc(memdesc_type, swizzled, context):
    if len(memdesc_type.shape) < 2 or swizzled.order != (1, 0):
        raise ValueError(
            f"tlx_wave new emitter cannot lower {context}: unsupported "
            f"shape={memdesc_type.shape}, order={swizzled.order}; only rank-2 "
            "order=[1, 0] swizzles are supported"
        )
    if swizzled.vec <= 0 or swizzled.per_phase <= 0 or swizzled.max_phase <= 0:
        raise ValueError(
            f"tlx_wave new emitter cannot lower {context}: invalid swizzle "
            f"parameters vec={swizzled.vec}, perPhase={swizzled.per_phase}, "
            f"maxPhase={swizzled.max_phase}"
        )
    if int(memdesc_type.shape[-1]) % int(swizzled.vec):
        raise ValueError(
            f"tlx_wave new emitter cannot lower {context}: columns "
            f"{memdesc_type.shape[-1]} are not divisible by vec={swizzled.vec}"
        )


def _padded_shared_memdesc_info(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_padded_shared_encoding"):
        return None
    return {
        "intervals": tuple(
            int(value) for value in _attr_value(attr, "get_padded_shared_intervals")
        ),
        "paddings": tuple(
            int(value) for value in _attr_value(attr, "get_padded_shared_paddings")
        ),
        "order": tuple(
            int(value) for value in _attr_value(attr, "get_padded_shared_order")
        ),
    }


def _exact_div(lhs, rhs, context):
    lhs = int(lhs)
    rhs = int(rhs)
    if rhs == 0 or lhs % rhs:
        raise ValueError(
            f"tlx_wave new emitter expected {context}={lhs} to be a multiple of {rhs}"
        )
    return lhs // rhs


def _emit_load(state, value):
    if value.mask_id is not None:
        _emit_masked_load(state, value)
        return
    if value.other_id is not None:
        raise ValueError("tlx_wave new emitter does not yet emit unmasked tt.load with other")
    ptr = _require_name(state, value.pointer_id)
    ptr_type = _require_type(state, value.pointer_id)
    token = _ensure_mem_token(state)
    result_type = value.converted_type.wave_type
    ptr_components = _remap_component_tuple(
        _as_components(ptr),
        value.converted_type.component_count,
        "load pointer",
    )
    components = []
    for ptr_component in ptr_components:
        pair = state.fresh()
        state.emit(
            f"{pair}:2 = wave.load {ptr_component} after {token} : "
            f"({ptr_type}, !wave.mem.token) -> ({result_type}, !wave.mem.token)"
        )
        components.append(f"{pair}#0")
        token = f"{pair}#1"
    state.names[value.value_id] = _pack_components(components)
    state.types[value.value_id] = result_type
    state.mem_token = token


def _emit_async_copy(state, value):
    try:
        _emit_async_copy_dma(state, value)
    except ValueError:
        _emit_async_copy_fallback(state, value)


def _emit_async_copy_dma(state, value):
    lowering = _select_async_copy_dma_lowering(state, value)
    token = _ordered_mem_token(state)
    base, base_type = _ptr_cast(
        state,
        _require_name(state, value.memdesc_value_id),
        _require_type(state, value.memdesc_value_id),
        "i32",
    )
    for component in range(lowering["component_count"]):
        coords = _emit_dma_packet_start_coords(state, lowering, component)
        source, source_type = _emit_dma_source_pointer(
            state, value, coords, lowering["packet_bytes"]
        )
        destination = _emit_dma_destination_pointer(state, base, base_type, lowering, component)
        mask_name = None
        mask_type = None
        if lowering["mask_value_id"] is not None:
            mask_name, mask_type = _materialize_coordinate_value(
                state, lowering["mask_value_id"], coords
            )
        token = _emit_dma_load_lds_text(
            state,
            source,
            source_type,
            destination,
            mask_name,
            mask_type,
            token,
            lowering["packet_bytes"],
        )
    state.names[value.value_id] = token
    state.types[value.value_id] = "!wave.mem.token"
    state.mem_token = token


def _select_async_copy_dma_lowering(state, value):
    if value.source_address_value_id is None or value.memdesc_value_id is None:
        raise ValueError("tlx_wave new emitter cannot DMA async copy without endpoints")
    if value.other_value_id is not None:
        raise ValueError("tlx_wave new emitter cannot DMA async copy with other value")
    source_type, element_type, element_byte_width = _async_copy_source_tensor_type(
        state, value
    )
    memdesc_type = _source_type(state, value.memdesc_value_id)
    if memdesc_type.kind != "memdesc":
        raise ValueError("tlx_wave new emitter expected DMA destination memdesc")
    if tuple(source_type.shape) != tuple(memdesc_type.shape):
        raise ValueError(
            "tlx_wave new emitter cannot DMA async copy with mismatched "
            f"source/destination shapes: {source_type.shape} vs {memdesc_type.shape}"
        )
    if element_type != memdesc_type.element_type:
        raise ValueError(
            "tlx_wave new emitter cannot DMA async copy with mismatched "
            f"element types: {element_type} vs {memdesc_type.element_type}"
        )
    if element_byte_width is None or element_byte_width <= 0:
        raise ValueError("tlx_wave new emitter cannot DMA async copy without element width")
    layout = _async_dma_thread_layout_info(source_type, "async copy DMA source")
    width = layout.width
    cta_threads = layout.cta_threads
    if width <= 0 or cta_threads <= 0:
        raise ValueError("tlx_wave new emitter cannot DMA async copy with empty layout")
    packet_errors = []
    for packet_bytes in _dma_packet_byte_candidates(element_type, element_byte_width):
        try:
            packet_elements = _dma_packet_elements(element_byte_width, packet_bytes)
            _require_dma_source_packet_contiguous(state, value, source_type, packet_elements)
            if _use_buffer_dma_source(state, value):
                _require_dma_source_offset_nonnegative(state, value, source_type)
            destination_remap = _dma_destination_remap(
                memdesc_type,
                packet_elements,
                packet_bytes,
            )
            if destination_remap is None:
                _require_dma_destination_packets(
                    memdesc_type,
                    packet_elements,
                    packet_bytes,
                    width,
                    "async copy DMA destination",
                )
            if value.mask_value_id is not None:
                _require_dma_packet_mask_uniform(
                    state,
                    value.mask_value_id,
                    source_type.shape,
                    packet_elements,
                )
            total_elements = _product(source_type.shape)
            if total_elements % packet_elements:
                raise ValueError("tensor element count is not packet-divisible")
            total_packets = total_elements // packet_elements
            packet_stride = cta_threads if total_packets >= cta_threads else width
            if total_packets % packet_stride:
                raise ValueError(
                    "packet count is not a full wave/CTA-thread multiple for DMA"
                )
            return {
                "source_shape": tuple(int(dim) for dim in source_type.shape),
                "packet_bytes": int(packet_bytes),
                "packet_elements": int(packet_elements),
                "component_count": total_packets // packet_stride,
                "width": int(width),
                "cta_threads": int(packet_stride),
                "memdesc_type": memdesc_type,
                "mask_value_id": value.mask_value_id,
                "destination_remap": destination_remap,
            }
        except ValueError as exc:
            packet_errors.append(exc)
    if packet_errors:
        raise packet_errors[-1]
    raise ValueError("tlx_wave new emitter found no legal async DMA packet size")


def _require_dma_packet_mask_uniform(state, mask_value_id, shape, packet_elements):
    if _mask_uniform_for_coord_group(
        state,
        mask_value_id,
        _dma_packet_coord_group_facts(shape, packet_elements),
    ):
        return
    raise ValueError(
        "tlx_wave new emitter cannot prove async DMA mask is uniform over "
        f"{packet_elements}-element packets"
    )


def _dma_packet_coord_group_facts(shape, packet_elements):
    inner_dim = len(tuple(shape)) - 1
    return tuple(
        _CoordGroupFact(int(packet_elements), 0, int(packet_elements) - 1)
        if dim == inner_dim
        else _CoordGroupFact(1, 0, 0)
        for dim in range(len(tuple(shape)))
    )


def _emit_dma_load_lds_text(
    state,
    source,
    source_type,
    destination,
    mask,
    mask_type,
    token,
    packet_bytes,
):
    if mask is None:
        name = state.fresh()
        state.emit(
            f"{name} = waveamd.dma_load_lds {source} -> {destination} after {token} "
            f"{{bytes = {packet_bytes} : i64}} : "
            f"({source_type}, !wave.ptr<#wave.shared, i32>, !wave.mem.token) "
            "-> !wave.mem.token"
        )
        return name
    where = state.fresh()
    inner = state.fresh()
    state.emit(f"{where} = wave.where {mask} {{")
    state.emit(
        f"  {inner} = waveamd.dma_load_lds {source} -> {destination} after {token} "
        f"{{bytes = {packet_bytes} : i64}} : "
        f"({source_type}, !wave.ptr<#wave.shared, i32>, !wave.mem.token) "
        "-> !wave.mem.token"
    )
    state.emit(f"  wave.yield {inner} : !wave.mem.token")
    state.emit("} otherwise {")
    state.emit(f"  wave.yield {token} : !wave.mem.token")
    state.emit(f"}} : {mask_type} -> !wave.mem.token")
    return where


def _async_copy_source_tensor_type(state, value):
    if value.source_offset_value_id is not None:
        source_type = _source_type(state, value.source_offset_value_id)
        pointer_type = _source_type(state, value.source_address_value_id)
        return source_type, pointer_type.pointee_type, pointer_type.element_byte_width
    source_type = _source_type(state, value.source_address_value_id)
    return source_type, source_type.pointee_type, source_type.element_byte_width


def _dma_packet_byte_candidates(element_type, element_byte_width):
    if element_byte_width == 2 and element_type in {"f16", "bf16"}:
        return (16, 4)
    if element_byte_width in (1, 2, 4):
        return (4,)
    if element_byte_width == 16:
        return (16,)
    return ()


def _dma_packet_elements(element_byte_width, packet_bytes):
    if packet_bytes % element_byte_width:
        raise ValueError("DMA packet size is not divisible by element width")
    return packet_bytes // element_byte_width


def _blocked_encoding_info(source_type, context):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_blocked_encoding"):
        raise ValueError(
            f"tlx_wave new emitter expected blocked encoding for {context}, "
            f"got {source_type.encoding}"
        )
    return _BlockedEncodingInfo(
        tuple(int(value) for value in _attr_value(attr, "get_blocked_size_per_thread")),
        tuple(int(value) for value in _attr_value(attr, "get_blocked_threads_per_warp")),
        tuple(int(value) for value in _attr_value(attr, "get_blocked_warps_per_cta")),
        tuple(int(value) for value in _attr_value(attr, "get_blocked_order")),
    )


def _async_dma_thread_layout_info(source_type, context):
    attr = source_type.encoding_attr
    if attr is not None and _attr_bool(attr, "is_blocked_encoding"):
        blocked = _blocked_encoding_info(source_type, context)
        width = _product(blocked.threads_per_warp)
        return _ThreadLayoutInfo(width, width * _product(blocked.warps_per_cta))
    if attr is not None and _attr_bool(attr, "is_linear_encoding"):
        block_bases = tuple(_linear_basis_vector(basis) for basis in _attr_value(
            attr, "get_linear_block_bases"
        ))
        if block_bases:
            raise ValueError(
                f"tlx_wave new emitter cannot DMA {context}: linear block "
                f"bases are not supported ({block_bases})"
            )
        lane_bases = tuple(_linear_basis_vector(basis) for basis in _attr_value(
            attr, "get_linear_lane_bases"
        ))
        warp_bases = tuple(_linear_basis_vector(basis) for basis in _attr_value(
            attr, "get_linear_warp_bases"
        ))
        width = 1 << len(lane_bases)
        cta_threads = width * (1 << len(warp_bases))
        return _ThreadLayoutInfo(width, cta_threads)
    raise ValueError(
        f"tlx_wave new emitter expected blocked or linear encoding for {context}, "
        f"got {source_type.encoding}"
    )


def _linear_basis_vector(basis):
    return tuple(int(value) for value in basis)


def _require_dma_source_packet_contiguous(state, value, source_type, packet_elements):
    if packet_elements <= 1:
        return
    inner_dim = len(source_type.shape) - 1
    coord_coeffs = tuple(1 if dim == inner_dim else 0 for dim in range(len(source_type.shape)))
    if value.source_offset_value_id is not None:
        coeff = _coordinate_inner_coeff(
            state, value.source_offset_value_id, coord_coeffs, packet_elements
        )
    else:
        coeff = _pointer_inner_coeff(
            state, value.source_address_value_id, coord_coeffs, packet_elements
        )
    if coeff is None or coeff[0] != 1:
        raise ValueError(
            "tlx_wave new emitter cannot prove async DMA source packets are "
            "contiguous along the innermost dimension"
        )


def _require_dma_source_offset_nonnegative(state, value, source_type):
    coord_nonnegative = tuple(True for _ in source_type.shape)
    if value.source_offset_value_id is not None:
        proven = _coordinate_nonnegative(
            state, value.source_offset_value_id, coord_nonnegative
        )
    else:
        proven = _pointer_offset_nonnegative(
            state, value.source_address_value_id, coord_nonnegative
        )
    if not proven:
        raise ValueError(
            "tlx_wave new emitter cannot prove async DMA source offsets are "
            "nonnegative"
        )


def _emit_dma_packet_start_coords(state, lowering, component):
    if lowering.get("destination_remap") is not None:
        return _emit_remapped_dma_packet_start_coords(state, lowering, component)
    width = lowering["width"]
    simd_i32 = f"!wave.simd<i32, {width}>"
    lane = _emit_assumed_workitem_id(state, width)
    packet_index = _emit_simd_add_const(
        state,
        lane,
        simd_i32,
        int(component) * int(lowering["cta_threads"]),
    )
    logical = _emit_simd_binary_const(
        state,
        "muli",
        packet_index,
        simd_i32,
        lowering["packet_elements"],
    )
    return tuple(
        (coord, simd_i32)
        for coord in _emit_row_major_coords_from_linear(
            state, logical, simd_i32, lowering["source_shape"]
        )
    )


def _emit_remapped_dma_packet_start_coords(state, lowering, component):
    remap = lowering["destination_remap"]
    if remap["kind"] != "swizzled_physical_packets":
        raise ValueError(
            "tlx_wave new emitter cannot emit unknown DMA destination remap "
            f"{remap['kind']}"
        )
    width = lowering["width"]
    simd_i32 = f"!wave.simd<i32, {width}>"
    lane = _emit_assumed_workitem_id(state, width)
    packet = _emit_simd_add_const(
        state,
        lane,
        simd_i32,
        int(component) * int(lowering["cta_threads"]),
    )
    groups_per_row = int(remap["groups_per_row"])
    row = _emit_simd_binary_const(state, "divui", packet, simd_i32, groups_per_row)
    physical_col_group = _emit_simd_binary_const(
        state, "remui", packet, simd_i32, groups_per_row
    )
    row_phase_outer = _emit_simd_binary_const(
        state, "divui", row, simd_i32, int(remap["per_phase"])
    )
    phase = _emit_simd_binary_const(
        state, "remui", row_phase_outer, simd_i32, int(remap["max_phase"])
    )
    logical_col_group = _emit_simd_binary(
        state, "xori", physical_col_group, phase, simd_i32
    )
    col = _emit_simd_binary_const(
        state, "muli", logical_col_group, simd_i32, int(remap["vec"])
    )
    return ((row, simd_i32), (col, simd_i32))


def _emit_assumed_workitem_id(state, width):
    simd_i32 = f"!wave.simd<i32, {int(width)}>"
    raw = state.fresh()
    assumed = state.fresh()
    state.emit(f"{raw} = wave.workitem_id 0 : {simd_i32}")
    state.emit(
        f"{assumed} = wave.assume {raw} as \"x\" "
        f"[#wave.pred<\"x >= 0\">, #wave.pred<\"-{int(width) - 1} + x <= 0\">] "
        f": {simd_i32}"
    )
    return assumed


def _emit_row_major_coords_from_linear(state, linear, value_type, shape):
    remaining = linear
    coords = [None for _ in shape]
    for dim in reversed(range(len(shape))):
        extent = int(shape[dim])
        if dim == 0:
            coords[dim] = remaining
            continue
        coords[dim] = _emit_simd_binary_const(state, "remui", remaining, value_type, extent)
        remaining = _emit_simd_binary_const(state, "divui", remaining, value_type, extent)
    return tuple(coords)


def _emit_dma_source_pointer(state, value, coords, packet_bytes):
    if value.source_offset_value_id is not None:
        offset, offset_type = _materialize_coordinate_value(
            state, value.source_offset_value_id, coords
        )
        offset, offset_type = _ensure_coordinate_offset_type(
            state, offset, offset_type, _coord_width(coords)
        )
        if _use_buffer_dma_source(state, value):
            offset, offset_type = _assume_buffer_offset_bounds(
                state,
                value.source_address_value_id,
                offset,
                offset_type,
                packet_bytes,
                "amdg.buffer_load_to_local source",
            )
            base, base_type = _buffer_pointer_for_base(
                state, value.source_address_value_id
            )
            result_type = _buffer_simd_pointer_type(base_type, offset_type)
        else:
            base_id = _pointer_scalar_base_id(state, value.source_address_value_id)
            offset = _emit_global_dma_source_offset_assume(
                state,
                base_id,
                offset,
                offset_type,
                packet_bytes,
            )
            base = _require_name(state, value.source_address_value_id)
            base_type = _require_type(state, value.source_address_value_id)
            result_type = _global_simd_pointer_type(base_type, offset_type)
        source = state.fresh()
        state.emit(
            f"{source} = wave.ptr_add {base}, {offset} : "
            f"{base_type}, {offset_type} -> {result_type}"
        )
        return source, result_type

    base, base_type, offset, offset_type = _materialize_pointer_parts_at_coords(
        state, value.source_address_value_id, coords
    )
    if offset is None:
        if not base_type.startswith("!wave.simd<"):
            raise ValueError("tlx_wave new emitter cannot DMA scalar source pointer tensor")
        return base, base_type
    offset, offset_type = _ensure_coordinate_offset_type(
        state, offset, offset_type, _coord_width(coords)
    )
    base_id = _pointer_scalar_base_id(state, value.source_address_value_id)
    offset = _emit_global_dma_source_offset_assume(
        state,
        base_id,
        offset,
        offset_type,
        packet_bytes,
    )
    result_type = _global_simd_pointer_type(base_type, offset_type)
    source = state.fresh()
    state.emit(
        f"{source} = wave.ptr_add {base}, {offset} : "
        f"{base_type}, {offset_type} -> {result_type}"
    )
    return source, result_type


def _use_buffer_dma_source(state, value):
    if value.op_name != "amdg.buffer_load_to_local":
        return False
    if value.source_address_value_id is None or value.source_offset_value_id is None:
        return False
    source_type = _source_type(state, value.source_address_value_id)
    return (
        source_type.kind == "pointer"
        and source_type.pointer_range is not None
        and int(source_type.pointer_range) > 0
        and int(source_type.pointer_range) <= 32
    )


def _emit_dma_destination_pointer(state, base, base_type, lowering, component):
    byte_offset = _dma_destination_component_byte_offset(lowering, component)
    if byte_offset % 4:
        raise ValueError("tlx_wave new emitter DMA destination is not dword aligned")
    dword_offset = byte_offset // 4
    if dword_offset == 0:
        return base
    const = state.fresh()
    ptr = state.fresh()
    state.emit(f"{const} = arith.constant {dword_offset} : i32")
    state.emit(
        f"{ptr} = wave.ptr_add {base}, {const} : "
        f"{base_type}, i32 -> {base_type}"
    )
    return ptr


def _dma_destination_component_byte_offset(lowering, component):
    packet_index = int(component) * int(lowering["cta_threads"])
    if lowering.get("destination_remap") is not None:
        return packet_index * int(lowering["packet_bytes"])
    return _dma_packet_static_byte_offset(
        lowering["memdesc_type"],
        lowering["packet_elements"],
        packet_index,
        "async copy DMA destination",
    )


def _emit_async_copy_fallback(state, value):
    if value.source_address_value_id is None or value.memdesc_value_id is None:
        raise ValueError("tlx_wave new emitter cannot emit async copy without source/destination")
    _emit_deferred_value_tree(state, value.source_address_value_id)
    _emit_deferred_value_tree(state, value.source_offset_value_id)
    _emit_deferred_value_tree(state, value.memdesc_value_id)
    _emit_deferred_value_tree(state, value.mask_value_id)
    _emit_deferred_value_tree(state, value.other_value_id)
    source_ptrs, source_ptr_type = _async_copy_source_pointers(state, value)
    dest_ptrs, dest_ptr_type = _async_copy_destination_pointers(
        state, value, len(source_ptrs)
    )
    mask_components, mask_type = _async_copy_mask_components(
        state, value, len(source_ptrs)
    )
    token = _ordered_mem_token(state)
    element_type = _simd_pointer_element_type(source_ptr_type)
    value_type = f"!wave.simd<{element_type}, {_simd_width(source_ptr_type)}>"
    other_components = None
    if value.other_value_id is not None:
        other, other_type = _simd_value_for_type(state, value.other_value_id, value_type)
        if other_type != value_type:
            raise ValueError(
                "tlx_wave new emitter expected async copy other type to match "
                f"loaded value type, got {other_type} -> {value_type}"
            )
        other_components = _remap_component_tuple(
            _as_components(other), len(source_ptrs), "async copy other"
        )
    for index, (source_ptr, dest_ptr) in enumerate(zip(source_ptrs, dest_ptrs)):
        mask = None if mask_components is None else mask_components[index]
        other = None if other_components is None else other_components[index]
        token = _emit_async_copy_fallback_component(
            state,
            source_ptr,
            source_ptr_type,
            dest_ptr,
            dest_ptr_type,
            mask,
            mask_type,
            other,
            token,
            value_type,
        )
    state.names[value.value_id] = token
    state.types[value.value_id] = "!wave.mem.token"
    state.mem_token = token


def _async_copy_mask_components(state, value, count):
    if value.mask_value_id is None:
        return None, None
    mask = _require_name(state, value.mask_value_id)
    mask_type = _require_type(state, value.mask_value_id)
    return _remap_component_tuple(
        _as_components(mask), count, "async copy mask"
    ), mask_type


def _emit_async_copy_fallback_component(
    state,
    source_ptr,
    source_ptr_type,
    dest_ptr,
    dest_ptr_type,
    mask,
    mask_type,
    other,
    token,
    value_type,
):
    if mask is None:
        pair = state.fresh()
        store_token = state.fresh()
        state.emit(
            f"{pair}:2 = wave.load {source_ptr} after {token} : "
            f"({source_ptr_type}, !wave.mem.token) -> ({value_type}, !wave.mem.token)"
        )
        state.emit(
            f"{store_token} = wave.store {pair}#0 -> {dest_ptr} after {pair}#1 : "
            f"({value_type}, {dest_ptr_type}, !wave.mem.token) -> !wave.mem.token"
        )
        return store_token
    where = state.fresh()
    pair = state.fresh()
    store_token = state.fresh()
    state.emit(f"{where} = wave.where {mask} {{")
    state.emit(
        f"  {pair}:2 = wave.load {source_ptr} after {token} : "
        f"({source_ptr_type}, !wave.mem.token) -> ({value_type}, !wave.mem.token)"
    )
    state.emit(
        f"  {store_token} = wave.store {pair}#0 -> {dest_ptr} after {pair}#1 : "
        f"({value_type}, {dest_ptr_type}, !wave.mem.token) -> !wave.mem.token"
    )
    state.emit(f"  wave.yield {store_token} : !wave.mem.token")
    state.emit("} otherwise {")
    if other is None:
        state.emit(f"  wave.yield {token} : !wave.mem.token")
    else:
        other_store = state.fresh()
        state.emit(
            f"  {other_store} = wave.store {other} -> {dest_ptr} after {token} : "
            f"({value_type}, {dest_ptr_type}, !wave.mem.token) -> !wave.mem.token"
        )
        state.emit(f"  wave.yield {other_store} : !wave.mem.token")
    state.emit(f"}} : {mask_type} -> !wave.mem.token")
    return where


def _emit_deferred_value_tree(state, value_id):
    if value_id is None or value_id in state.names:
        return
    value = state.converted_values.get(value_id)
    if value is None:
        return
    for operand_id in _record_operand_ids(value):
        _emit_deferred_value_tree(state, operand_id)
    _emit_value(state, value)


def _coordinate_inner_coeff(state, value_id, coord_coeffs, packet_elements=None):
    source_type = _source_type(state, value_id)
    value = state.converted_values.get(value_id)
    if value is None:
        return None
    if source_type.kind != "tensor":
        return _scalar_inner_coeff(value)
    if isinstance(value, _ConstantValue):
        return 0, _constant_int_literal(value)
    if isinstance(value, _RangeValue):
        if len(coord_coeffs) != 1:
            return None
        return int(coord_coeffs[0]), int(value.start)
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return _coordinate_inner_coeff(state, value.operand_id, (), packet_elements)
        if value.op_name == "tt.broadcast":
            operand_coeffs = _broadcast_operand_coord_coeffs(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_coeffs,
            )
            if operand_coeffs is None:
                return None
            return _coordinate_inner_coeff(
                state, value.operand_id, operand_coeffs, packet_elements
            )
        return None
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _coordinate_inner_coeff(
                state,
                value.operand_id,
                coord_coeffs[:axis] + coord_coeffs[axis + 1 :],
                packet_elements,
            )
        if value.op_name == "ttg.convert_layout":
            return _coordinate_inner_coeff(
                state, value.operand_id, coord_coeffs, packet_elements
            )
        return None
    if isinstance(value, _CastValue):
        return _coordinate_inner_coeff(
            state, value.operand_id, coord_coeffs, packet_elements
        )
    if isinstance(value, _BinaryValue):
        if value.kind in {"remsi", "remui"}:
            rem = _coordinate_rem_inner_coeff(
                state, value, coord_coeffs, packet_elements
            )
            if rem is not None:
                return rem
        lhs = _coordinate_inner_coeff(state, value.lhs_id, coord_coeffs, packet_elements)
        rhs = _coordinate_inner_coeff(state, value.rhs_id, coord_coeffs, packet_elements)
        return _combine_inner_coeffs(value.kind, lhs, rhs)
    return None


def _pointer_inner_coeff(state, value_id, coord_coeffs, packet_elements=None):
    value = state.converted_values.get(value_id)
    source_type = _source_type(state, value_id)
    if isinstance(value, _PointerAddValue):
        base = _pointer_inner_coeff(state, value.base_id, coord_coeffs, packet_elements)
        offset = _coordinate_inner_coeff(
            state, value.offset_id, coord_coeffs, packet_elements
        )
        return _combine_inner_coeffs("addi", base, offset)
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return 0, None
        if value.op_name == "tt.broadcast":
            operand_coeffs = _broadcast_operand_coord_coeffs(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_coeffs,
            )
            if operand_coeffs is None:
                return None
            return _pointer_inner_coeff(
                state, value.operand_id, operand_coeffs, packet_elements
            )
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _pointer_inner_coeff(
                state,
                value.operand_id,
                coord_coeffs[:axis] + coord_coeffs[axis + 1 :],
                packet_elements,
            )
        if value.op_name == "ttg.convert_layout":
            return _pointer_inner_coeff(
                state, value.operand_id, coord_coeffs, packet_elements
            )
    return None


def _coordinate_rem_inner_coeff(state, value, coord_coeffs, packet_elements):
    lhs = _coordinate_inner_coeff(state, value.lhs_id, coord_coeffs, packet_elements)
    rhs = _coordinate_inner_coeff(state, value.rhs_id, coord_coeffs, packet_elements)
    if lhs is None or rhs is None:
        return None
    lhs_coeff, _lhs_const = lhs
    rhs_coeff, _rhs_const = rhs
    if rhs_coeff != 0:
        return None
    if lhs_coeff == 0:
        return 0, None
    if lhs_coeff != 1 or packet_elements is None or int(packet_elements) <= 1:
        return None
    coord_facts = _packet_coord_facts_for_inner_coeffs(coord_coeffs, packet_elements)
    if coord_facts is None:
        return None
    lhs_fact = _value_div_delta_for_coord_group(state, value.lhs_id, coord_facts)
    rhs_fact = _value_div_delta_for_coord_group(state, value.rhs_id, coord_facts)
    if lhs_fact is None or rhs_fact is None:
        return None
    packet_elements = int(packet_elements)
    if (
        lhs_fact.delta_min == 0
        and lhs_fact.delta_max == packet_elements - 1
        and rhs_fact.delta_min == 0
        and rhs_fact.delta_max == 0
        and _divisibility_covers(lhs_fact.divisibility, packet_elements)
        and _divisibility_covers(rhs_fact.divisibility, packet_elements)
    ):
        return 1, None
    return None


def _packet_coord_facts_for_inner_coeffs(coord_coeffs, packet_elements):
    facts = []
    seen_varying_dim = False
    for coeff in coord_coeffs:
        coeff = int(coeff)
        if coeff == 0:
            facts.append(_CoordGroupFact(1, 0, 0))
        elif coeff == 1 and not seen_varying_dim:
            facts.append(_CoordGroupFact(int(packet_elements), 0, int(packet_elements) - 1))
            seen_varying_dim = True
        else:
            return None
    return tuple(facts)


def _coordinate_nonnegative(state, value_id, coord_nonnegative):
    source_type = _source_type(state, value_id)
    value = state.converted_values.get(value_id)
    if value is None:
        return False
    if source_type.kind != "tensor":
        return _scalar_nonnegative(state, value_id)
    if value_id in state.nonnegative_values:
        return True
    if isinstance(value, _ConstantValue):
        literal = _constant_scalar_literal(value)
        return isinstance(literal, int) and not isinstance(literal, bool) and literal >= 0
    if isinstance(value, _RangeValue):
        return len(coord_nonnegative) == 1 and bool(coord_nonnegative[0]) and value.start >= 0
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return _scalar_nonnegative(state, value.operand_id)
        if value.op_name == "tt.broadcast":
            operand_nonnegative = _broadcast_operand_coord_nonnegative(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_nonnegative,
            )
            return (
                operand_nonnegative is not None
                and _coordinate_nonnegative(state, value.operand_id, operand_nonnegative)
            )
        return False
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _coordinate_nonnegative(
                state,
                value.operand_id,
                coord_nonnegative[:axis] + coord_nonnegative[axis + 1 :],
            )
        if value.op_name == "ttg.convert_layout":
            return _coordinate_nonnegative(state, value.operand_id, coord_nonnegative)
        return False
    if isinstance(value, _CastValue):
        return _coordinate_nonnegative(state, value.operand_id, coord_nonnegative)
    if isinstance(value, _BinaryValue):
        return _binary_nonnegative(
            value.kind,
            _coordinate_nonnegative(state, value.lhs_id, coord_nonnegative),
            _coordinate_nonnegative(state, value.rhs_id, coord_nonnegative),
            _constant_int_value(state, value.lhs_id),
            _constant_int_value(state, value.rhs_id),
        )
    if isinstance(value, _MinValue):
        return _coordinate_nonnegative(
            state, value.lhs_id, coord_nonnegative
        ) and _coordinate_nonnegative(state, value.rhs_id, coord_nonnegative)
    return False


def _pointer_offset_nonnegative(state, value_id, coord_nonnegative):
    value = state.converted_values.get(value_id)
    source_type = _source_type(state, value_id)
    if isinstance(value, _PointerAddValue):
        return _pointer_offset_nonnegative(
            state, value.base_id, coord_nonnegative
        ) and _coordinate_nonnegative(state, value.offset_id, coord_nonnegative)
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return True
        if value.op_name == "tt.broadcast":
            operand_nonnegative = _broadcast_operand_coord_nonnegative(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_nonnegative,
            )
            return (
                operand_nonnegative is not None
                and _pointer_offset_nonnegative(
                    state, value.operand_id, operand_nonnegative
                )
            )
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _pointer_offset_nonnegative(
                state,
                value.operand_id,
                coord_nonnegative[:axis] + coord_nonnegative[axis + 1 :],
            )
        if value.op_name == "ttg.convert_layout":
            return _pointer_offset_nonnegative(state, value.operand_id, coord_nonnegative)
    return False


def _scalar_nonnegative(state, value_id):
    if value_id in state.nonnegative_values:
        return True
    value = state.converted_values.get(value_id)
    if isinstance(value, _ConstantValue):
        literal = _constant_scalar_literal(value)
        return isinstance(literal, int) and not isinstance(literal, bool) and literal >= 0
    if isinstance(value, _ProgramIdValue):
        return True
    if isinstance(value, _CastValue):
        return _scalar_nonnegative(state, value.operand_id)
    if isinstance(value, _BinaryValue):
        return _binary_nonnegative(
            value.kind,
            _scalar_nonnegative(state, value.lhs_id),
            _scalar_nonnegative(state, value.rhs_id),
            _constant_int_value(state, value.lhs_id),
            _constant_int_value(state, value.rhs_id),
        )
    if isinstance(value, _MinValue):
        return _scalar_nonnegative(state, value.lhs_id) and _scalar_nonnegative(
            state, value.rhs_id
        )
    return False


def _scalar_positive(state, value_id):
    lower = _scalar_lower_bound(state, value_id)
    if lower is not None:
        return lower > 0
    if value_id in state.positive_values:
        return True
    value = state.converted_values.get(value_id)
    if isinstance(value, _ConstantValue):
        literal = _constant_scalar_literal(value)
        return isinstance(literal, int) and not isinstance(literal, bool) and literal > 0
    if isinstance(value, _CastValue):
        return _scalar_positive(state, value.operand_id)
    if isinstance(value, _BinaryValue):
        return _binary_positive(
            value.kind,
            _scalar_nonnegative(state, value.lhs_id),
            _scalar_nonnegative(state, value.rhs_id),
            _scalar_positive(state, value.lhs_id),
            _scalar_positive(state, value.rhs_id),
            _constant_int_value(state, value.lhs_id),
            _constant_int_value(state, value.rhs_id),
        )
    if isinstance(value, _MinValue):
        return _scalar_positive(state, value.lhs_id) and _scalar_positive(
            state, value.rhs_id
        )
    return False


def _scalar_lower_bound(state, value_id, seen=frozenset()):
    if value_id in seen:
        return None
    seen = seen | {value_id}
    value = state.converted_values.get(value_id)
    if isinstance(value, _ConstantValue):
        literal = _constant_scalar_literal(value)
        if isinstance(literal, int) and not isinstance(literal, bool):
            return int(literal)
        return None
    if isinstance(value, _ProgramIdValue):
        return 0
    if isinstance(value, _CastValue):
        return _scalar_lower_bound(state, value.operand_id, seen)
    if isinstance(value, _BinaryValue):
        lhs_lower = _scalar_lower_bound(state, value.lhs_id, seen)
        rhs_lower = _scalar_lower_bound(state, value.rhs_id, seen)
        rhs_const = _constant_int_value(state, value.rhs_id)
        if value.kind == "addi" and lhs_lower is not None and rhs_lower is not None:
            return lhs_lower + rhs_lower
        if value.kind == "subi" and lhs_lower is not None and rhs_const is not None:
            return lhs_lower - rhs_const
        if (
            value.kind == "muli"
            and lhs_lower is not None
            and rhs_lower is not None
            and lhs_lower >= 0
            and rhs_lower >= 0
        ):
            return lhs_lower * rhs_lower
        if value.kind in {"divsi", "divui"} and lhs_lower is not None:
            if lhs_lower >= 0 and rhs_const is not None and rhs_const > 0:
                return lhs_lower // rhs_const
        if value.kind in {"remsi", "remui"} and lhs_lower is not None:
            if lhs_lower >= 0 and _scalar_positive(state, value.rhs_id):
                return 0
    if isinstance(value, _MinValue):
        lhs_lower = _scalar_lower_bound(state, value.lhs_id, seen)
        rhs_lower = _scalar_lower_bound(state, value.rhs_id, seen)
        if lhs_lower is not None and rhs_lower is not None:
            return min(lhs_lower, rhs_lower)
    if value_id in state.positive_values:
        return 1
    if value_id in state.nonnegative_values:
        return 0
    return None


def _binary_nonnegative(kind, lhs_nonnegative, rhs_nonnegative, lhs_const, rhs_const):
    if kind == "addi":
        return lhs_nonnegative and rhs_nonnegative
    if kind == "muli":
        return lhs_nonnegative and rhs_nonnegative
    if kind == "divsi":
        return lhs_nonnegative and rhs_const is not None and rhs_const > 0
    if kind == "remsi":
        return lhs_nonnegative
    if kind == "remui":
        return True
    if kind == "subi":
        return rhs_const is not None and rhs_const <= 0 and lhs_nonnegative
    return False


def _binary_positive(
    kind,
    lhs_nonnegative,
    rhs_nonnegative,
    lhs_positive,
    rhs_positive,
    lhs_const,
    rhs_const,
):
    if kind == "addi":
        return (lhs_positive and rhs_nonnegative) or (
            lhs_nonnegative and rhs_positive
        )
    if kind == "muli":
        return lhs_positive and rhs_positive
    if kind == "subi":
        if rhs_const is None:
            return False
        return (lhs_positive and rhs_const <= 0) or (
            lhs_nonnegative and rhs_const < 0
        )
    return False


def _constant_int_value(state, value_id):
    value = state.converted_values.get(value_id)
    if not isinstance(value, _ConstantValue):
        return None
    return _constant_int_literal(value)


def _broadcast_operand_coord_nonnegative(operand_shape, result_shape, coord_nonnegative):
    if len(operand_shape) > len(result_shape):
        return None
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, nonnegative in zip(
        operand_shape, result_shape[leading:], coord_nonnegative[leading:]
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(bool(nonnegative))
        elif int(operand_dim) == 1:
            mapped.append(True)
        else:
            return None
    return tuple(mapped)


def _emit_nonnegative_i32_simd_assume(state, value, value_type):
    if not value_type.startswith("!wave.simd<i32,"):
        return value
    assumed = state.fresh()
    state.emit(
        f"{assumed} = wave.assume {value} as \"x\" "
        "[#wave.pred<\"x >= 0\">, #wave.pred<\"-2147483647 + x <= 0\">] "
        f": {value_type}"
    )
    return assumed


def _emit_global_dma_source_offset_assume(
    state,
    base_id,
    offset,
    offset_type,
    access_byte_width,
):
    if not offset_type.startswith("!wave.simd<i32,"):
        return offset
    lower, upper = _global_dma_source_offset_bounds(
        state,
        base_id,
        access_byte_width,
    )
    assumed = state.fresh()
    state.emit(
        f"{assumed} = wave.assume {offset} as \"x\" "
        f"[#wave.pred<\"{_lower_bound_predicate(lower)}\">, "
        f"#wave.pred<\"{_upper_bound_predicate(upper)}\">] : {offset_type}"
    )
    return assumed


def _global_dma_source_offset_bounds(state, base_id, access_byte_width):
    source_type = _source_type(state, base_id)
    bounds = _small_pointer_element_offset_bounds(
        source_type.pointer_range,
        source_type.element_byte_width,
        access_byte_width,
    )
    if bounds is not None:
        return bounds
    return -(1 << 31), (1 << 31) - 1


def _lower_bound_predicate(lower):
    lower = int(lower)
    if lower == 0:
        return "x >= 0"
    if lower > 0:
        return f"-{lower} + x >= 0"
    return f"{abs(lower)} + x >= 0"


def _upper_bound_predicate(upper):
    upper = int(upper)
    if upper == 0:
        return "x <= 0"
    if upper > 0:
        return f"-{upper} + x <= 0"
    return f"{abs(upper)} + x <= 0"


def _scalar_inner_coeff(value):
    if isinstance(value, _ConstantValue):
        return 0, _constant_int_literal(value)
    if isinstance(value, _CastValue):
        return 0, None
    if isinstance(value, _BinaryValue):
        lhs = _scalar_inner_coeff(value.lhs)
        rhs = _scalar_inner_coeff(value.rhs)
        combined = _combine_inner_coeffs(value.kind, lhs, rhs)
        return (0, None) if combined is None else combined
    return 0, None


def _combine_inner_coeffs(kind, lhs, rhs):
    if lhs is None or rhs is None:
        return None
    lhs_coeff, lhs_const = lhs
    rhs_coeff, rhs_const = rhs
    if kind == "addi":
        const = None if lhs_const is None or rhs_const is None else lhs_const + rhs_const
        return lhs_coeff + rhs_coeff, const
    if kind == "subi":
        const = None if lhs_const is None or rhs_const is None else lhs_const - rhs_const
        return lhs_coeff - rhs_coeff, const
    if kind == "muli":
        if lhs_const is not None:
            const = None if rhs_const is None else lhs_const * rhs_const
            return lhs_const * rhs_coeff, const
        if rhs_const is not None:
            const = None if lhs_const is None else lhs_const * rhs_const
            return rhs_const * lhs_coeff, const
        if lhs_coeff == 0 and rhs_coeff == 0:
            return 0, None
        return None
    return None


def _constant_int_literal(value):
    literal = _constant_scalar_literal(value)
    if isinstance(literal, bool) or not isinstance(literal, int):
        return None
    return literal


def _broadcast_operand_coord_coeffs(operand_shape, result_shape, coord_coeffs):
    if len(operand_shape) > len(result_shape):
        return None
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, coeff in zip(
        operand_shape, result_shape[leading:], coord_coeffs[leading:]
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(int(coeff))
        elif int(operand_dim) == 1:
            mapped.append(0)
        else:
            return None
    return tuple(mapped)


def _require_dma_destination_packets(
    memdesc_type,
    packet_elements,
    packet_bytes,
    width,
    context,
):
    total_elements = _product(memdesc_type.shape)
    if total_elements % int(packet_elements):
        raise ValueError(
            f"tlx_wave new emitter cannot DMA {context}: tensor element count "
            f"{total_elements} is not divisible by packet width {packet_elements}"
        )
    total_packets = total_elements // int(packet_elements)
    packet_offsets = []
    for packet_index in range(total_packets):
        coords = _dma_packet_start_coords_static(
            memdesc_type.shape, packet_elements, packet_index
        )
        packet_offsets.append(
            _require_dma_packet_contiguous_from_coords(
                memdesc_type,
                coords,
                packet_elements,
                packet_bytes,
                context,
            )
        )
    for chunk_start in range(0, total_packets, int(width)):
        chunk = packet_offsets[chunk_start : chunk_start + int(width)]
        if len(chunk) != int(width):
            raise ValueError(
                f"tlx_wave new emitter cannot DMA {context}: final packet "
                "chunk is not a full wave"
            )
        first = chunk[0]
        for lane, byte_offset in enumerate(chunk[1:], start=1):
            expected = first + lane * int(packet_bytes)
            if byte_offset != expected:
                raise ValueError(
                    f"tlx_wave new emitter cannot DMA {context}: packet starts "
                    f"are not whole-wave contiguous; lane {lane} starts at "
                    f"{byte_offset}, expected {expected}"
                )


def _dma_destination_remap(memdesc_type, packet_elements, packet_bytes):
    if _is_identity_shared_memdesc(memdesc_type):
        return None
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if swizzled is None:
        return None
    if len(tuple(memdesc_type.shape)) != 2:
        return None
    if tuple(swizzled.order) != (1, 0):
        return None
    if int(packet_elements) != int(swizzled.vec):
        return None
    if memdesc_type.element_byte_width is None:
        return None
    if int(packet_bytes) != int(swizzled.vec) * int(memdesc_type.element_byte_width):
        return None
    rows, cols = (int(dim) for dim in memdesc_type.shape)
    if rows <= 0 or cols <= 0 or cols % int(swizzled.vec):
        return None
    return {
        "kind": "swizzled_physical_packets",
        "vec": int(swizzled.vec),
        "per_phase": int(swizzled.per_phase),
        "max_phase": int(swizzled.max_phase),
        "groups_per_row": cols // int(swizzled.vec),
    }


def _dma_packet_start_coords_static(shape, packet_elements, packet_index):
    return _row_major_coords_static(
        tuple(int(dim) for dim in shape),
        int(packet_index) * int(packet_elements),
    )


def _require_dma_packet_contiguous_from_coords(
    memdesc_type,
    coords,
    packet_elements,
    packet_bytes,
    context,
):
    first = None
    inner_dim = len(memdesc_type.shape) - 1
    for element in range(int(packet_elements)):
        packet_coords = list(coords)
        packet_coords[inner_dim] += element
        if packet_coords[inner_dim] >= int(memdesc_type.shape[inner_dim]):
            raise ValueError(
                f"tlx_wave new emitter cannot DMA {context}: packet {coords} "
                "crosses the innermost dimension"
            )
        byte_offset = _memdesc_static_byte_offset(
            memdesc_type, tuple(packet_coords), context
        )
        if first is None:
            first = byte_offset
            if first % 4:
                raise ValueError(
                    f"tlx_wave new emitter cannot DMA {context}: packet starts "
                    f"at unaligned byte offset {first}"
                )
            continue
        expected = first + element * int(memdesc_type.element_byte_width)
        if byte_offset != expected:
            raise ValueError(
                f"tlx_wave new emitter cannot DMA {context}: packet {coords} "
                "is not physically contiguous"
            )
    if int(packet_elements) * int(memdesc_type.element_byte_width) != int(packet_bytes):
        raise ValueError(
            f"tlx_wave new emitter cannot DMA {context}: packet byte width "
            "does not match element width"
        )
    return first


def _dma_packet_static_byte_offset(memdesc_type, packet_elements, packet_index, context):
    coords = _dma_packet_start_coords_static(
        memdesc_type.shape, packet_elements, packet_index
    )
    return _memdesc_static_byte_offset(memdesc_type, coords, context)


def _memdesc_static_byte_offset(memdesc_type, coords, context):
    if memdesc_type.element_byte_width is None:
        raise ValueError(f"tlx_wave new emitter cannot address {context}: unknown width")
    if _is_identity_shared_memdesc(memdesc_type):
        return _row_major_linear_static(memdesc_type.shape, coords) * int(
            memdesc_type.element_byte_width
        )
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if swizzled is not None:
        return _swizzled_shared_static_byte_offset(
            memdesc_type, coords, swizzled, context
        )
    padded = _padded_shared_memdesc_info(memdesc_type)
    if padded is None:
        raise ValueError(
            f"tlx_wave new emitter cannot DMA {context}: unsupported shared layout "
            f"{memdesc_type.encoding}"
        )
    if padded["order"] != tuple(reversed(range(len(memdesc_type.shape)))):
        raise ValueError(
            f"tlx_wave new emitter cannot DMA {context}: unsupported padded order "
            f"{padded['order']}"
        )
    logical = _row_major_linear_static(memdesc_type.shape, coords)
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        encoded += (logical // int(interval)) * int(padding)
    return encoded * int(memdesc_type.element_byte_width)


def _swizzled_shared_static_byte_offset(memdesc_type, coords, swizzled, context):
    _validate_swizzled_shared_memdesc(memdesc_type, swizzled, context)
    if len(memdesc_type.shape) != 2:
        raise ValueError(
            f"tlx_wave new emitter cannot address {context}: only rank-2 "
            f"swizzled shared memdescs are supported, got {memdesc_type.shape}"
        )
    row = int(coords[-2])
    col = int(coords[-1])
    rows = int(memdesc_type.shape[-2])
    cols = int(memdesc_type.shape[-1])
    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise ValueError(
            f"tlx_wave new emitter cannot address {context}: coordinate "
            f"{coords} exceeds shape {memdesc_type.shape}"
        )
    phase = (row // int(swizzled.per_phase)) % int(swizzled.max_phase)
    col_group = col // int(swizzled.vec)
    swizzled_col = (col_group ^ phase) * int(swizzled.vec) + (
        col % int(swizzled.vec)
    )
    return (row * cols + swizzled_col) * int(memdesc_type.element_byte_width)


def _row_major_coords_static(shape, linear):
    coords = [0 for _ in shape]
    remaining = int(linear)
    for dim in reversed(range(len(shape))):
        if dim == 0:
            coords[dim] = remaining
        else:
            extent = int(shape[dim])
            coords[dim] = remaining % extent
            remaining //= extent
    return tuple(coords)


def _row_major_linear_static(shape, coords):
    linear = 0
    stride = 1
    for dim in reversed(range(len(shape))):
        linear += int(coords[dim]) * stride
        stride *= int(shape[dim])
    return linear


def _async_copy_source_pointers(state, value):
    source_type = _require_type(state, value.source_address_value_id)
    if value.source_offset_value_id is None:
        source = _require_name(state, value.source_address_value_id)
        if not source_type.startswith("!wave.simd<"):
            raise ValueError(
                "tlx_wave new emitter expected async source pointer tensor, "
                f"got {source_type}"
            )
        return _as_components(source), source_type
    offsets, offset_type = _index_offset_components(
        state,
        value.source_offset_value_id,
        _component_count_for_value(state, value.source_offset_value_id),
    )
    base = _require_name(state, value.source_address_value_id)
    if isinstance(base, tuple):
        raise ValueError(
            "tlx_wave new emitter expected uniform async source base pointer, "
            f"got component tuple for value {value.source_address_value_id}"
        )
    result_type = _global_simd_pointer_type(source_type, offset_type)
    ptrs = []
    for offset in offsets:
        ptr = state.fresh()
        state.emit(
            f"{ptr} = wave.ptr_add {base}, {offset} : "
            f"{source_type}, {offset_type} -> {result_type}"
        )
        ptrs.append(ptr)
    return tuple(ptrs), result_type


def _async_copy_destination_pointers(state, value, count):
    if value.source_offset_value_id is not None:
        offset_id = value.source_offset_value_id
    elif isinstance(value.source_address, _PointerAddValue):
        offset_id = value.source_address.offset_id
    else:
        raise ValueError(
            "tlx_wave new emitter supports async copy fallback only from "
            "tt.addptr source addresses or buffer-style source offsets"
        )
    base = _require_name(state, value.memdesc_value_id)
    base_type = _require_type(state, value.memdesc_value_id)
    offsets, offset_type = _index_offset_components(state, offset_id, count)
    result_type = _shared_simd_pointer_type(base_type, offset_type)
    ptrs = []
    for offset in offsets:
        name = state.fresh()
        state.emit(
            f"{name} = wave.ptr_add {base}, {offset} : "
            f"{base_type}, {offset_type} -> {result_type}"
        )
        ptrs.append(name)
    return tuple(ptrs), result_type


def _component_count_for_value(state, value_id):
    return len(_as_components(_require_name(state, value_id)))


def _global_simd_pointer_type(base_type, offset_type):
    element_type = _pointer_element_type(base_type)
    if offset_type.startswith("!wave.simd<"):
        return f"!wave.simd<!wave.ptr<#wave.global, {element_type}>, {_simd_width(offset_type)}>"
    return f"!wave.ptr<#wave.global, {element_type}>"


def _buffer_simd_pointer_type(base_type, offset_type):
    element_type = _pointer_element_type(base_type)
    if offset_type.startswith("!wave.simd<"):
        return f"!wave.simd<!wave.ptr<#waveamd.buffer, {element_type}>, {_simd_width(offset_type)}>"
    return f"!wave.ptr<#waveamd.buffer, {element_type}>"


def _async_copy_destination_pointer(state, source_address, memdesc_value_id):
    value = _AsyncCopyValue(
        -1,
        "ttg.async_copy_global_to_local",
        None,
        None,
        memdesc_value_id,
        None,
        None,
        source_address,
        None,
        None,
        None,
        None,
        None,
    )
    ptrs, ptr_type = _async_copy_destination_pointers(state, value, 1)
    return ptrs[0], ptr_type


def _emit_async_commit_group(state, value):
    member_names = tuple(
        _require_name(state, token_id) for token_id in value.member_token_ids
    )
    if not member_names:
        token = _ordered_mem_token(state)
        state.names[value.value_id] = token
        state.types[value.value_id] = "!wave.mem.token"
        return
    name = state.fresh()
    deps = ", ".join(member_names)
    dep_types = ", ".join("!wave.mem.token" for _ in member_names)
    state.emit(f"{name} = wave.join {deps} : {dep_types} -> !wave.mem.token")
    state.names[value.value_id] = name
    state.types[value.value_id] = "!wave.mem.token"
    state.mem_token = name


def _emit_async_wait(state, effect):
    _emit_async_wait_token(state, effect)


def _emit_async_wait_value(state, value):
    token = _emit_async_wait_token(state, value)
    state.names[value.value_id] = token
    state.types[value.value_id] = "!wave.mem.token"


def _emit_async_wait_token(state, effect):
    pending_read_token = _ordered_mem_token(state) if state.pending_read_tokens else None
    if effect.input_token_ids:
        deps = tuple(_require_name(state, token_id) for token_id in effect.input_token_ids)
    else:
        deps = tuple(
            _count_wait_group_dependency(state, group.token_value_id)
            for group_index in effect.waited_group_indices
            for group in (state.token_graph.groups[group_index],)
            if group.token_value_id is not None
        )
        deps = tuple(dict.fromkeys(dep for dep in deps if dep is not None))
    if pending_read_token is not None:
        deps = tuple(dict.fromkeys((*deps, pending_read_token)))
    if not deps:
        return _ensure_mem_token(state)
    dep_text = ", ".join(deps)
    dep_types = ", ".join("!wave.mem.token" for _ in deps)
    state.emit(f"wave.wait {dep_text} : {dep_types}")
    barrier = state.fresh()
    state.emit(f"{barrier} = wave.barrier {dep_text} : ({dep_types}) -> !wave.mem.token")
    state.mem_token = barrier
    return barrier


def _count_wait_group_dependency(state, token_value_id):
    if token_value_id in state.names:
        return state.names[token_value_id]
    return state.mem_token


def _emit_masked_load(state, value):
    ptr = _require_name(state, value.pointer_id)
    ptr_type = _require_type(state, value.pointer_id)
    mask = _require_name(state, value.mask_id)
    mask_type = _require_type(state, value.mask_id)
    token = _ensure_mem_token(state)
    result_type = value.converted_type.wave_type
    if value.other_id is None:
        other = _zero_simd_value(state, result_type)
    else:
        other, other_type = _simd_value_for_type(state, value.other_id, result_type)
        if other_type != result_type:
            raise ValueError(
                "tlx_wave new emitter expected masked tt.load other type to match "
                f"result type, got {other_type} -> {result_type}"
            )
    pair = state.fresh()
    inner_value = state.fresh()
    inner_token = state.fresh()
    state.emit(f"{pair}:2 = wave.where {mask} {{")
    state.emit(
        f"  {inner_value}, {inner_token} = wave.load {ptr} after {token} : "
        f"({ptr_type}, !wave.mem.token) -> ({result_type}, !wave.mem.token)"
    )
    state.emit(f"  wave.yield {inner_value}, {inner_token} : {result_type}, !wave.mem.token")
    state.emit("} otherwise {")
    state.emit(f"  wave.yield {other}, {token} : {result_type}, !wave.mem.token")
    state.emit(f"}} : {mask_type} -> {result_type}, !wave.mem.token")
    state.names[value.value_id] = f"{pair}#0"
    state.types[value.value_id] = result_type
    state.mem_token = f"{pair}#1"


def _simd_value_for_type(state, value_id, result_type):
    name = _require_name(state, value_id)
    value_type = _require_type(state, value_id)
    if value_type == result_type:
        return name, value_type
    if value_type.startswith("!wave.simd<"):
        return name, value_type
    element_type = _simd_element_type(result_type)
    if value_type != element_type:
        return name, value_type
    splat = state.fresh()
    state.emit(f"{splat} = wave.splat {name} : {value_type} -> {result_type}")
    return splat, result_type


def _zero_simd_value(state, result_type):
    element_type = _simd_element_type(result_type)
    const = state.fresh()
    splat = state.fresh()
    state.emit(f"{const} = arith.constant {_zero_literal(element_type)} : {element_type}")
    state.emit(f"{splat} = wave.splat {const} : {element_type} -> {result_type}")
    return splat


def _zero_literal(element_type):
    if element_type in {"f16", "bf16", "f32", "f64"}:
        return "0.000000e+00"
    return "0"


def _emit_fragment_store(state, effect):
    logical_type = _source_type(state, effect.value_id)
    if logical_type.kind != "tensor" or len(logical_type.shape) != 2:
        raise ValueError(
            "tlx_wave new emitter supports fragment stores only for rank-2 "
            f"tensor values, got {logical_type.raw}"
        )
    if logical_type.element_type not in {"f32", "f16"}:
        raise ValueError(
            "tlx_wave new emitter supports fragment stores only for f32/f16 "
            f"tensor values, got {logical_type.raw}"
        )
    physical_type = _require_type(state, effect.value_id)
    frag_info = _fragment_type_info(physical_type)
    if frag_info.role != 2:
        raise ValueError(
            "tlx_wave new emitter supports stores only from accumulator "
            f"fragments, got {physical_type}"
        )
    fragments = _as_components(_require_name(state, effect.value_id))
    mma = _fragment_store_mma_shape(state, effect.value_id)
    tile_rows = _tile_rep_count(logical_type.shape[0], mma.output_tile_shape[0])
    tile_cols = _tile_rep_count(logical_type.shape[1], mma.output_tile_shape[1])
    if len(fragments) != tile_rows * tile_cols:
        raise ValueError(
            "tlx_wave new emitter fragment store tile mismatch: got "
            f"{len(fragments)} fragment(s), expected {tile_rows * tile_cols}"
        )
    token = _ordered_mem_token(state)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            fragment = fragments[_tile_index((tile_rows, tile_cols), tile_row, tile_col)]
            store_vector, store_vector_type = _emit_fragment_store_vector(
                state, fragment, physical_type, logical_type, frag_info
            )
            tile_offsets = (
                tile_row * mma.output_tile_shape[0],
                tile_col * mma.output_tile_shape[1],
            )
            component_coords = _emit_fragment_component_coords(state, frag_info, tile_offsets)
            component = 0
            while component < len(component_coords):
                packed = _try_emit_buffer_fragment_vector_store(
                    state,
                    effect,
                    logical_type,
                    frag_info,
                    tile_offsets,
                    component_coords,
                    store_vector,
                    store_vector_type,
                    token,
                    component,
                )
                if packed is not None:
                    token, packed_count = packed
                    component += packed_count
                    continue
                coords = component_coords[component]
                value_name, value_type = _emit_fragment_store_component_value(
                    state,
                    store_vector,
                    store_vector_type,
                    logical_type,
                    frag_info,
                    component,
                )
                mask_name = None
                mask_type = None
                if effect.mask_id is not None:
                    mask_name, mask_type = _materialize_coordinate_value(
                        state, effect.mask_id, coords
                    )
                if isinstance(effect, _BufferStoreEffect) and mask_name is not None:
                    token = _emit_masked_buffer_component_store_text(
                        state,
                        effect,
                        coords,
                        value_name,
                        value_type,
                        mask_name,
                        mask_type,
                        token,
                    )
                    component += 1
                    continue
                ptr_name, ptr_type = _materialize_fragment_store_pointer(
                    state, effect, coords, value_type
                )
                token = _emit_component_store_text(
                    state,
                    value_name,
                    value_type,
                    ptr_name,
                    ptr_type,
                    mask_name,
                    mask_type,
                    token,
                )
                component += 1
    state.mem_token = token


def _try_emit_buffer_fragment_vector_store(
    state,
    effect,
    logical_type,
    frag,
    tile_offsets,
    component_coords,
    store_vector,
    store_vector_type,
    token,
    component,
):
    if not isinstance(effect, (_BufferStoreEffect, _StoreEffect)):
        return None
    count = _fragment_vector_store_count(
        effect, logical_type, frag, component, len(component_coords)
    )
    if count <= 1:
        return None
    coord_facts = _fragment_store_coord_group_facts(
        frag, tile_offsets, component, count
    )
    if isinstance(effect, _StoreEffect) and not _store_pointer_contiguous_for_coord_group(
        state, effect, coord_facts, count
    ):
        return None
    if effect.mask_id is not None and not _mask_uniform_for_coord_group(
        state, effect.mask_id, coord_facts
    ):
        return None
    mask_name = None
    mask_type = None
    first_coords = component_coords[component]
    value_name, value_type = _emit_fragment_store_vector_group(
        state,
        store_vector,
        store_vector_type,
        logical_type,
        frag,
        component,
        count,
    )
    if effect.mask_id is not None:
        mask_name, mask_type = _materialize_coordinate_value(
            state, effect.mask_id, first_coords
        )
        if isinstance(effect, _BufferStoreEffect):
            token = _emit_masked_buffer_component_store_text(
                state,
                effect,
                first_coords,
                value_name,
                value_type,
                mask_name,
                mask_type,
                token,
            )
        else:
            ptr_name, ptr_type = _materialize_fragment_store_pointer(
                state, effect, first_coords, value_type
            )
            token = _emit_component_store_text(
                state,
                value_name,
                value_type,
                ptr_name,
                ptr_type,
                mask_name,
                mask_type,
                token,
            )
        return token, count
    ptr_name, ptr_type = _materialize_fragment_store_pointer(
        state, effect, first_coords, value_type
    )
    token = _emit_component_store_text(
        state, value_name, value_type, ptr_name, ptr_type, mask_name, mask_type, token
    )
    return token, count


def _fragment_vector_store_count(
    effect, logical_type, frag, component, component_count
):
    if logical_type.kind != "tensor" or logical_type.element_byte_width is None:
        return 1
    if isinstance(effect, _BufferStoreEffect):
        if effect.contiguity is None or int(effect.contiguity) <= 1:
            return 1
        count = int(effect.contiguity)
    else:
        count = min(
            16 // int(logical_type.element_byte_width),
            int(frag.registers),
        )
        if count <= 1:
            return 1
    if int(component) % count:
        return 1
    if int(component) + count > min(int(frag.registers), int(component_count)):
        return 1
    if count * int(logical_type.element_byte_width) > 16:
        return 1
    return count


def _emit_fragment_store_vector_group(
    state,
    store_vector,
    store_vector_type,
    logical_type,
    frag,
    component,
    count,
):
    if int(component) == 0 and int(count) == int(frag.registers):
        return store_vector, store_vector_type
    values = []
    value_types = []
    for index in range(int(component), int(component) + int(count)):
        value, value_type = _emit_fragment_store_component_value(
            state, store_vector, store_vector_type, logical_type, frag, index
        )
        values.append(value)
        value_types.append(value_type)
    packed_type = (
        f"!wave.simd<vector<{int(count)}x{logical_type.element_type}>, "
        f"{frag.wave_size}>"
    )
    packed = state.fresh()
    state.emit(
        f"{packed} = wave.pack {', '.join(values)} : "
        f"{', '.join(value_types)} -> {packed_type}"
    )
    return packed, packed_type


def _store_pointer_contiguous_for_coord_group(state, effect, coord_facts, count):
    fact = _pointer_div_delta_for_coord_group(state, effect.pointer_id, coord_facts)
    if fact is None:
        return False
    return fact.delta_max - fact.delta_min == int(count) - 1


def _pointer_div_delta_for_coord_group(state, value_id, coord_facts):
    value = state.converted_values.get(value_id)
    if isinstance(value, _PointerAddValue):
        base = _pointer_div_delta_for_coord_group(state, value.base_id, coord_facts)
        offset = _value_div_delta_for_coord_group(state, value.offset_id, coord_facts)
        if base is None or offset is None:
            return None
        return _CoordGroupFact(
            _gcd_divisibility(base.divisibility, offset.divisibility),
            base.delta_min + offset.delta_min,
            base.delta_max + offset.delta_max,
        )
    if isinstance(value, _UnaryTensorValue) and value.op_name == "tt.splat":
        return _CoordGroupFact(0, 0, 0)
    if isinstance(value, _ForwardValue) and value.op_name == "ttg.convert_layout":
        return _pointer_div_delta_for_coord_group(state, value.operand_id, coord_facts)
    return None


def _fragment_store_coord_group_facts(frag, tile_offsets, component, count):
    components = tuple(range(int(component), int(component) + int(count)))
    if int(frag.registers) == 4:
        row_deltas = tuple(comp // 2 for comp in components)
        col_deltas = tuple(comp % 2 for comp in components)
        return (
            _coord_group_fact(tile_offsets[0], 2, row_deltas),
            _coord_group_fact(tile_offsets[1], 2, col_deltas),
        )
    if int(frag.registers) == 16:
        row_deltas = tuple(0 for _ in components)
        col_deltas = tuple((comp % 4) + 8 * (comp // 4) for comp in components)
        return (
            _coord_group_fact(tile_offsets[0], 1, row_deltas),
            _coord_group_fact(tile_offsets[1], 4, col_deltas),
        )
    return tuple(
        _CoordGroupFact(1, 0, 0) for _ in range(len(tuple(tile_offsets)))
    )


def _coord_group_fact(tile_offset, base_divisibility, deltas):
    deltas = tuple(int(delta) for delta in deltas)
    base_div = gcd(abs(int(tile_offset)), abs(int(base_divisibility)))
    if base_div == 0:
        base_div = abs(int(base_divisibility)) or 1
    return _CoordGroupFact(base_div, min(deltas), max(deltas))


def _mask_uniform_for_coord_group(state, mask_id, coord_facts):
    value = state.converted_values.get(mask_id)
    if isinstance(value, _ConstantValue):
        return isinstance(_constant_scalar_literal(value), bool)
    if isinstance(value, _ForwardValue):
        if value.op_name == "ttg.convert_layout":
            return _mask_uniform_for_coord_group(state, value.operand_id, coord_facts)
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
            )
            return _mask_uniform_for_coord_group(
                state, value.operand_id, coord_facts[:axis] + coord_facts[axis + 1 :]
            )
    if isinstance(value, _UnaryTensorValue) and value.op_name == "tt.broadcast":
        operand_facts = _broadcast_operand_coord_facts(
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
            coord_facts,
        )
        return _mask_uniform_for_coord_group(state, value.operand_id, operand_facts)
    if isinstance(value, _MaskAndValue):
        return _mask_uniform_for_coord_group(
            state, value.lhs_id, coord_facts
        ) and _mask_uniform_for_coord_group(state, value.rhs_id, coord_facts)
    if isinstance(value, _CompareValue):
        return _compare_uniform_for_coord_group(state, value, coord_facts)
    return False


def _compare_uniform_for_coord_group(state, value, coord_facts):
    lhs = _value_div_delta_for_coord_group(state, value.lhs_id, coord_facts)
    rhs = _value_div_delta_for_coord_group(state, value.rhs_id, coord_facts)
    if lhs is None or rhs is None:
        return False
    if value.predicate in {"slt", "sle"}:
        return _ordered_compare_uniform(lhs, rhs)
    if value.predicate in {"sgt", "sge"}:
        return _ordered_compare_uniform(rhs, lhs)
    if value.predicate in {"eq", "ne"}:
        return lhs.delta_min == lhs.delta_max == rhs.delta_min == rhs.delta_max == 0
    return False


def _ordered_compare_uniform(varying, bound):
    if bound.delta_min != 0 or bound.delta_max != 0:
        return False
    if varying.delta_min < 0:
        return False
    span = varying.delta_max - varying.delta_min
    if span == 0:
        return True
    alignment = int(span) + 1
    return _divisibility_covers(varying.divisibility, alignment) and _divisibility_covers(
        bound.divisibility, alignment
    )


def _value_div_delta_for_coord_group(state, value_id, coord_facts):
    source_type = _source_type(state, value_id)
    value = state.converted_values.get(value_id)
    if isinstance(value, _ConstantValue):
        literal = _constant_int_record(value)
        if literal is None:
            return None
        return _CoordGroupFact(abs(int(literal)), 0, 0)
    if isinstance(value, _RangeValue):
        if len(coord_facts) != 1:
            return None
        coord = coord_facts[0]
        return _CoordGroupFact(
            _gcd_divisibility(coord.divisibility, abs(int(value.start))),
            coord.delta_min,
            coord.delta_max,
        )
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return _value_div_delta_for_coord_group(state, value.operand_id, ())
        if value.op_name == "tt.broadcast":
            operand_facts = _broadcast_operand_coord_facts(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
                coord_facts,
            )
            return _value_div_delta_for_coord_group(state, value.operand_id, operand_facts)
    if isinstance(value, _ForwardValue):
        if value.op_name == "ttg.convert_layout":
            return _value_div_delta_for_coord_group(state, value.operand_id, coord_facts)
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
            )
            return _value_div_delta_for_coord_group(
                state, value.operand_id, coord_facts[:axis] + coord_facts[axis + 1 :]
            )
    if isinstance(value, _BinaryValue):
        result = _binary_div_delta_for_coord_group(state, value, coord_facts)
        if result is not None:
            return result
        if source_type.kind == "scalar":
            return _CoordGroupFact(_source_divisibility(source_type), 0, 0)
        return None
    if isinstance(value, _CastValue):
        return _value_div_delta_for_coord_group(state, value.operand_id, coord_facts)
    if source_type.kind == "scalar":
        return _CoordGroupFact(_source_divisibility(source_type), 0, 0)
    return None


def _binary_div_delta_for_coord_group(state, value, coord_facts):
    lhs = _value_div_delta_for_coord_group(state, value.lhs_id, coord_facts)
    rhs = _value_div_delta_for_coord_group(state, value.rhs_id, coord_facts)
    if lhs is None or rhs is None:
        return None
    if value.kind == "addi":
        return _CoordGroupFact(
            _gcd_divisibility(lhs.divisibility, rhs.divisibility),
            lhs.delta_min + rhs.delta_min,
            lhs.delta_max + rhs.delta_max,
        )
    if value.kind == "subi":
        return _CoordGroupFact(
            _gcd_divisibility(lhs.divisibility, rhs.divisibility),
            lhs.delta_min - rhs.delta_max,
            lhs.delta_max - rhs.delta_min,
        )
    if value.kind == "muli":
        lhs_const = _constant_int_for_value(state, value.lhs_id)
        rhs_const = _constant_int_for_value(state, value.rhs_id)
        if lhs_const is not None:
            return _scale_coord_group_fact(rhs, lhs_const)
        if rhs_const is not None:
            return _scale_coord_group_fact(lhs, rhs_const)
    return None


def _scale_coord_group_fact(fact, scale):
    scale = int(scale)
    lo = fact.delta_min * scale
    hi = fact.delta_max * scale
    return _CoordGroupFact(
        _scale_divisibility(fact.divisibility, scale),
        min(lo, hi),
        max(lo, hi),
    )


def _broadcast_operand_coord_facts(operand_shape, result_shape, coord_facts):
    if len(operand_shape) > len(result_shape):
        return ()
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, fact in zip(
        operand_shape, result_shape[leading:], coord_facts[leading:]
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(fact)
        elif int(operand_dim) == 1:
            mapped.append(_CoordGroupFact(0, 0, 0))
        else:
            return ()
    return tuple(mapped)


def _constant_int_for_value(state, value_id):
    value = state.converted_values.get(value_id)
    if not isinstance(value, _ConstantValue):
        return None
    return _constant_int_record(value)


def _source_divisibility(source_type):
    if source_type.divisibility is None or int(source_type.divisibility) <= 0:
        return 1
    return int(source_type.divisibility)


def _gcd_divisibility(lhs, rhs):
    lhs = abs(int(lhs))
    rhs = abs(int(rhs))
    if lhs == 0:
        return rhs
    if rhs == 0:
        return lhs
    return gcd(lhs, rhs)


def _scale_divisibility(divisibility, scale):
    divisibility = abs(int(divisibility))
    scale = abs(int(scale))
    if divisibility == 0 or scale == 0:
        return 0
    return divisibility * scale


def _divisibility_covers(divisibility, alignment):
    alignment = abs(int(alignment))
    if alignment <= 1:
        return True
    divisibility = abs(int(divisibility))
    return divisibility == 0 or divisibility % alignment == 0


def _materialize_fragment_store_pointer(state, effect, coords, value_type):
    if isinstance(effect, _BufferStoreEffect):
        return _materialize_buffer_store_pointer(state, effect, coords, value_type)
    return _materialize_coordinate_value(state, effect.pointer_id, coords)


def _materialize_buffer_store_pointer(state, effect, coords, value_type):
    buffer, buffer_type = _buffer_pointer_for_base(state, effect.base_id)
    width = _coord_width(coords)
    offset, offset_type = _materialize_coordinate_value(state, effect.offset_id, coords)
    offset, offset_type = _ensure_coordinate_offset_type(state, offset, offset_type, width)
    offset, offset_type = _assume_buffer_store_offset_bounds(
        state, effect, offset, offset_type, value_type
    )
    source_type = _source_type(state, effect.base_id)
    ptr_type = f"!wave.simd<!wave.ptr<#waveamd.buffer, {source_type.pointee_type}>, {width}>"
    ptr = state.fresh()
    state.emit(
        f"{ptr} = wave.ptr_add {buffer}, {offset} : "
        f"{buffer_type}, {offset_type} -> {ptr_type}"
    )
    return ptr, ptr_type


def _materialize_masked_buffer_store_pointer(
    state,
    effect,
    coords,
    value_type,
    mask_name,
    mask_type,
):
    buffer, buffer_type = _buffer_pointer_for_base(state, effect.base_id)
    width = _coord_width(coords)
    offset, offset_type = _materialize_coordinate_value(state, effect.offset_id, coords)
    offset, offset_type = _ensure_coordinate_offset_type(state, offset, offset_type, width)
    access_byte_width = _store_access_byte_width(value_type)
    offset, offset_type = _assume_buffer_offset_bounds(
        state,
        effect.base_id,
        offset,
        offset_type,
        access_byte_width,
        "amdg.buffer_store offset",
    )
    upper = _buffer_offset_upper(
        state,
        effect.base_id,
        access_byte_width,
        "amdg.buffer_store offset",
    )
    # False lanes use a one-past-valid element offset so the bounded buffer
    # descriptor suppresses the store without an exec-mask region.
    offset_element_type = _simd_element_type(offset_type)
    oob_const = state.fresh()
    oob_splat = state.fresh()
    true_ptr = state.fresh()
    oob_ptr = state.fresh()
    selected_ptr = state.fresh()
    state.emit(f"{oob_const} = arith.constant {int(upper) + 1} : {offset_element_type}")
    state.emit(f"{oob_splat} = wave.splat {oob_const} : {offset_element_type} -> {offset_type}")
    source_type = _source_type(state, effect.base_id)
    ptr_type = f"!wave.simd<!wave.ptr<#waveamd.buffer, {source_type.pointee_type}>, {width}>"
    state.emit(
        f"{true_ptr} = wave.ptr_add {buffer}, {offset} : "
        f"{buffer_type}, {offset_type} -> {ptr_type}"
    )
    state.emit(
        f"{oob_ptr} = wave.ptr_add {buffer}, {oob_splat} : "
        f"{buffer_type}, {offset_type} -> {ptr_type}"
    )
    state.emit(
        f"{selected_ptr} = wave.select {mask_name}, {true_ptr}, {oob_ptr} : "
        f"{mask_type}, {ptr_type}"
    )
    return selected_ptr, ptr_type


def _assume_buffer_store_offset_bounds(state, effect, offset, offset_type, value_type):
    return _assume_buffer_offset_bounds(
        state,
        effect.base_id,
        offset,
        offset_type,
        _store_access_byte_width(value_type),
        "amdg.buffer_store offset",
    )


def _assume_buffer_offset_bounds(
    state,
    base_id,
    offset,
    offset_type,
    access_byte_width,
    context,
):
    upper = _buffer_offset_upper(state, base_id, access_byte_width, context)
    assumed = state.fresh()
    state.emit(
        f"{assumed} = wave.assume {offset} as \"x\" "
        f"[#wave.pred<\"x >= 0\">, #wave.pred<\"-{upper} + x <= 0\">] : "
        f"{offset_type}"
    )
    return assumed, offset_type


def _buffer_offset_upper(state, base_id, access_byte_width, context):
    source_type = _source_type(state, base_id)
    upper = _small_pointer_element_offset_upper(
        source_type.pointer_range,
        source_type.element_byte_width,
        access_byte_width,
    )
    if upper is None:
        raise ValueError(
            f"tlx_wave new emitter cannot prove {context} bounds for base "
            f"type {source_type.raw}"
        )
    return upper


def _small_pointer_element_offset_upper(
    pointer_range,
    element_byte_width,
    access_byte_width,
):
    bounds = _small_pointer_element_offset_bounds(
        pointer_range,
        element_byte_width,
        access_byte_width,
    )
    return None if bounds is None else bounds[1]


def _small_pointer_element_offset_bounds(
    pointer_range,
    element_byte_width,
    access_byte_width,
):
    if (
        pointer_range is None
        or pointer_range <= 0
        or pointer_range > 32
        or element_byte_width is None
        or element_byte_width <= 0
        or access_byte_width is None
        or access_byte_width <= 0
    ):
        return None
    byte_lower = -(1 << (int(pointer_range) - 1))
    byte_upper = (1 << (int(pointer_range) - 1)) - 1
    byte_upper -= int(access_byte_width)
    if byte_upper < 0:
        return None
    lower = byte_lower // int(element_byte_width)
    upper = byte_upper // int(element_byte_width)
    return lower, upper


def _store_access_byte_width(value_type):
    element_type = _simd_element_type(value_type)
    return _scalar_type_byte_width(element_type)


def _scalar_type_byte_width(element_type):
    if element_type in {"i1", "i8"}:
        return 1
    if element_type in {"i16", "f16", "bf16"}:
        return 2
    if element_type in {"i32", "f32", "index"}:
        return 4
    if element_type in {"i64", "f64"}:
        return 8
    vector = re.fullmatch(r"vector<\s*(\d+)\s*x\s*([^>]+)\s*>", element_type)
    if vector is not None:
        return int(vector.group(1)) * _scalar_type_byte_width(vector.group(2).strip())
    return None


def _buffer_pointer_for_base(state, base_id):
    cached = state.buffer_cache.get(base_id)
    if cached is not None:
        return cached
    source_type = _source_type(state, base_id)
    if source_type.kind != "pointer" or source_type.pointee_type is None:
        raise ValueError(
            "tlx_wave new emitter expected buffer memory base to be a "
            f"typed pointer, got {source_type.raw}"
        )
    if source_type.pointer_range is None or source_type.pointer_range <= 0:
        raise ValueError(
            "tlx_wave new emitter cannot lower buffer memory op without a "
            "positive tt.pointer_range on the base pointer"
        )
    if source_type.pointer_range > 32:
        raise ValueError(
            "tlx_wave new emitter cannot lower buffer memory op with "
            f"tt.pointer_range={source_type.pointer_range}; expected <= 32"
        )
    base = _require_name(state, base_id)
    if isinstance(base, tuple):
        raise ValueError(
            "tlx_wave new emitter expected buffer memory base pointer to be "
            f"uniform, got component tuple for value {base_id}"
        )
    base_type = _require_type(state, base_id)
    range_bytes = (1 << (int(source_type.pointer_range) - 1)) - 1
    range_const = state.fresh()
    buffer = state.fresh()
    buffer_type = f"!wave.ptr<#waveamd.buffer, {source_type.pointee_type}>"
    state.emit(f"{range_const} = arith.constant {range_bytes} : i32")
    state.emit(
        f"{buffer} = waveamd.make_buffer {base}, {range_const} : "
        f"{base_type}, i32 -> {buffer_type}"
    )
    state.buffer_cache[base_id] = (buffer, buffer_type)
    return buffer, buffer_type


def _emit_masked_buffer_component_store_text(
    state,
    effect,
    coords,
    value_name,
    value_type,
    mask_name,
    mask_type,
    token,
):
    ptr_name, ptr_type = _materialize_masked_buffer_store_pointer(
        state,
        effect,
        coords,
        value_type,
        mask_name,
        mask_type,
    )
    store = state.fresh()
    state.emit(
        f"{store} = wave.store {value_name} -> {ptr_name} after {token} : "
        f"({value_type}, {ptr_type}, !wave.mem.token) -> !wave.mem.token"
    )
    return store


def _fragment_store_mma_shape(state, value_id):
    source_type = _fragment_physical_source_type(state, value_id)
    parent = _amd_mfma_encoding_info(source_type.encoding_attr)
    if parent is None:
        parent = _conversion_blocked_encoding_info(source_type.encoding_attr)
    if parent is None:
        raise ValueError(
            "tlx_wave new emitter expected fragment store value to originate "
            f"from #ttg.amd_mfma or a supported blocked MFMA parent, got {source_type.raw}"
        )
    return _mma_shape_for_parent(parent, "fragment store")


def _fragment_physical_source_type(state, value_id):
    source_type = _source_type(state, value_id)
    if (
        _amd_mfma_encoding_info(source_type.encoding_attr) is not None
        or _conversion_blocked_encoding_info(source_type.encoding_attr) is not None
    ):
        return source_type
    value = state.converted_values.get(value_id)
    if isinstance(value, _ForwardValue) and value.op_name == "ttg.convert_layout":
        return _fragment_physical_source_type(state, value.operand_id)
    if isinstance(value, _CastValue):
        cast_type = _source_type(state, value.value_id)
        if (
            _amd_mfma_encoding_info(cast_type.encoding_attr) is not None
            or _conversion_blocked_encoding_info(cast_type.encoding_attr) is not None
        ):
            return cast_type
        return _fragment_physical_source_type(state, value.operand_id)
    return source_type


def _emit_fragment_store_vector(state, fragment, physical_type, logical_type, frag):
    element_type = "f32" if logical_type.element_type == "f16" else logical_type.element_type
    unpack_type = (
        f"!wave.simd<vector<{frag.registers}x{element_type}>, "
        f"{frag.wave_size}>"
    )
    unpack = state.fresh()
    state.emit(
        f"{unpack} = waveamd.fragment_unpack {fragment} : "
        f"{physical_type} -> {unpack_type}"
    )
    if element_type == logical_type.element_type:
        return unpack, unpack_type
    store_vector_type = (
        f"!wave.simd<vector<{frag.registers}x{logical_type.element_type}>, "
        f"{frag.wave_size}>"
    )
    store_vector = state.fresh()
    state.emit(
        f"{store_vector} = wave.cast fpconvert {unpack} : "
        f"{unpack_type} -> {store_vector_type}"
    )
    return store_vector, store_vector_type


def _emit_fragment_store_component_value(
    state,
    store_vector,
    store_vector_type,
    logical_type,
    frag,
    component,
):
    extracted_type = f"!wave.simd<{logical_type.element_type}, {frag.wave_size}>"
    extracted = state.fresh()
    state.emit(
        f"{extracted} = wave.extract {store_vector}[{component}] : "
        f"{store_vector_type} "
        f"-> {extracted_type}"
    )
    return extracted, extracted_type


def _emit_component_store_text(
    state,
    value_name,
    value_type,
    ptr_name,
    ptr_type,
    mask_name,
    mask_type,
    token,
):
    if mask_name is None:
        store = state.fresh()
        state.emit(
            f"{store} = wave.store {value_name} -> {ptr_name} after {token} : "
            f"({value_type}, {ptr_type}, !wave.mem.token) -> !wave.mem.token"
        )
        return store
    where = state.fresh()
    inner = state.fresh()
    state.emit(f"{where} = wave.where {mask_name} {{")
    state.emit(
        f"  {inner} = wave.store {value_name} -> {ptr_name} after {token} : "
        f"({value_type}, {ptr_type}, !wave.mem.token) -> !wave.mem.token"
    )
    state.emit(f"  wave.yield {inner} : !wave.mem.token")
    state.emit(f"}} : {mask_type} -> !wave.mem.token")
    return where


def _emit_fragment_component_coords(state, frag, tile_offsets):
    roots = _fragment_component_coord_roots(state, frag)
    simd_i32 = f"!wave.simd<i32, {frag.wave_size}>"
    return tuple(
        _offset_fragment_component_coord_pair(
            state,
            simd_i32,
            tile_offsets,
            root[0][0],
            root[1][0],
        )
        for root in roots
    )


def _fragment_component_coord_roots(state, frag):
    key = (int(frag.registers), int(frag.wave_size))
    use_cache = state.indent == 4
    if use_cache:
        cached = state.fragment_coord_root_cache.get(key)
        if cached is not None:
            return cached
    width = int(frag.wave_size)
    simd_i32 = f"!wave.simd<i32, {width}>"
    lane = _emit_assumed_workitem_id(state, width)
    roots = []
    if int(frag.registers) == 4:
        lane_col = _emit_simd_binary_const(state, "remui", lane, simd_i32, 16)
        lane_row = _emit_simd_binary_const(state, "divui", lane, simd_i32, 16)
        base_row = _emit_simd_binary_const(state, "muli", lane_row, simd_i32, 2)
        base_col = _emit_simd_binary_const(state, "muli", lane_col, simd_i32, 2)
        for component in range(int(frag.registers)):
            row = _emit_simd_add_const(state, base_row, simd_i32, component // 2)
            col = _emit_simd_add_const(state, base_col, simd_i32, component % 2)
            roots.append(((row, simd_i32), (col, simd_i32)))
    elif int(frag.registers) == 16:
        base_row = _emit_simd_binary_const(state, "remui", lane, simd_i32, 32)
        lane_hi = _emit_simd_binary_const(state, "divui", lane, simd_i32, 32)
        base_col = _emit_simd_binary_const(state, "muli", lane_hi, simd_i32, 4)
        for component in range(int(frag.registers)):
            col = _emit_simd_add_const(
                state,
                base_col,
                simd_i32,
                (component % 4) + 8 * (component // 4),
            )
            roots.append(((base_row, simd_i32), (col, simd_i32)))
    else:
        raise ValueError(
            "tlx_wave new emitter cannot map accumulator fragment with "
            f"{frag.registers} registers"
        )
    roots = tuple(roots)
    if use_cache:
        state.fragment_coord_root_cache[key] = roots
    return roots


def _offset_fragment_component_coord_pair(state, simd_i32, tile_offsets, row, col):
    row = _emit_simd_add_const(state, row, simd_i32, tile_offsets[0])
    col = _emit_simd_add_const(state, col, simd_i32, tile_offsets[1])
    return ((row, simd_i32), (col, simd_i32))


def _materialize_coordinate_value(state, value_id, coords):
    source_type = _source_type(state, value_id)
    if source_type.kind != "tensor":
        return _require_name(state, value_id), _require_type(state, value_id)
    cache_key = (int(value_id), tuple((name, value_type) for name, value_type in coords))
    use_cache = state.indent == 4
    if use_cache:
        cached = state.coordinate_cache.get(cache_key)
        if cached is not None:
            return cached
    value = state.converted_values[value_id]
    width = _coord_width(coords)
    if isinstance(value, _ConstantValue):
        result = _materialize_coordinate_constant(state, value, width)
    elif isinstance(value, _RangeValue):
        result = _materialize_coordinate_range(state, value, coords)
    elif isinstance(value, _UnaryTensorValue):
        result = _materialize_coordinate_unary_tensor(state, value, coords)
    elif isinstance(value, _ForwardValue):
        result = _materialize_coordinate_forward(state, value, coords)
    elif isinstance(value, _BinaryValue):
        result = _materialize_coordinate_binary(state, value, coords)
    elif isinstance(value, _CompareValue):
        result = _materialize_coordinate_compare(state, value, coords)
    elif isinstance(value, _MaskAndValue):
        result = _materialize_coordinate_mask_and(state, value, coords)
    elif isinstance(value, _MinValue):
        result = _materialize_coordinate_min(state, value, coords)
    elif isinstance(value, _PointerAddValue):
        result = _materialize_coordinate_pointer_add(state, value, coords)
    elif isinstance(value, _CastValue):
        name, value_type = _materialize_coordinate_value(state, value.operand_id, coords)
        result_type = _coordinate_wave_type(state, value.value_id, width)
        result = _emit_coord_cast(state, value.kind, name, value_type, result_type)
    else:
        raise ValueError(
            "tlx_wave new emitter cannot materialize tensor value "
            f"{value_id} at fragment coordinates from {type(value).__name__}"
        )
    if use_cache:
        state.coordinate_cache[cache_key] = result
    return result


def _materialize_coordinate_constant(state, value, width):
    source_type = _source_type(state, value.value_id)
    if source_type.element_type == "i1":
        result_type = f"!wave.mask<{width}>"
        literal = _constant_scalar_literal(value)
        if literal is None:
            raise ValueError(
                "tlx_wave new emitter cannot materialize non-scalar mask "
                f"constant {value.raw_literal}"
            )
        name = state.fresh()
        text = "true" if bool(literal) else "false"
        state.emit(f"{name} = wave.constant {text} -> {result_type}")
        return name, result_type
    scalar_type = source_type.element_type or value.converted_type.wave_type
    result_type = _coordinate_wave_type(state, value.value_id, width)
    literal = _constant_scalar_literal(value)
    if literal is None:
        raise ValueError(
            "tlx_wave new emitter cannot materialize non-scalar tensor "
            f"constant {value.raw_literal}"
        )
    const = state.fresh()
    splat = state.fresh()
    state.emit(f"{const} = arith.constant {_format_literal(literal)} : {scalar_type}")
    state.emit(f"{splat} = wave.splat {const} : {scalar_type} -> {result_type}")
    return splat, result_type


def _materialize_coordinate_range(state, value, coords):
    if len(coords) != 1:
        raise ValueError(
            "tlx_wave new emitter expected rank-1 tt.make_range coordinate, "
            f"got {len(coords)} coordinate(s)"
        )
    coord, coord_type = coords[0]
    result_type = _coordinate_wave_type(state, value.value_id, _simd_width(coord_type))
    if coord_type != result_type:
        coord, coord_type = _emit_coord_cast(state, "index_cast", coord, coord_type, result_type)
    return _emit_simd_add_const(state, coord, result_type, value.start), result_type


def _materialize_coordinate_unary_tensor(state, value, coords):
    if value.op_name == "tt.splat":
        operand_name, operand_type = _materialize_coordinate_value(
            state, value.operand_id, ()
        )
        result_type = _coordinate_wave_type(state, value.value_id, _coord_width(coords))
        return _ensure_coordinate_type(state, operand_name, operand_type, result_type)
    if value.op_name == "tt.broadcast":
        operand_coords = _broadcast_operand_coords(
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
            coords,
        )
        return _materialize_coordinate_value(state, value.operand_id, operand_coords)
    raise ValueError(
        "tlx_wave new emitter cannot materialize unary tensor op "
        f"{value.op_name} at fragment coordinates"
    )


def _materialize_coordinate_forward(state, value, coords):
    if value.op_name == "tt.expand_dims":
        axis = _expand_dims_axis(
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
        )
        return _materialize_coordinate_value(
            state,
            value.operand_id,
            coords[:axis] + coords[axis + 1 :],
        )
    if value.op_name == "ttg.convert_layout":
        return _materialize_coordinate_value(state, value.operand_id, coords)
    raise ValueError(
        "tlx_wave new emitter cannot materialize forward op "
        f"{value.op_name} at fragment coordinates"
    )


def _materialize_coordinate_binary(state, value, coords):
    width = _coord_width(coords)
    result_type = _coordinate_wave_type(state, value.value_id, width)
    lhs, lhs_type = _materialize_coordinate_value(state, value.lhs_id, coords)
    rhs, rhs_type = _materialize_coordinate_value(state, value.rhs_id, coords)
    lhs, lhs_type = _ensure_coordinate_type(state, lhs, lhs_type, result_type)
    rhs, rhs_type = _ensure_coordinate_type(state, rhs, rhs_type, result_type)
    result = state.fresh()
    state.emit(
        f"{result} = wave.binary {value.kind} {lhs}, {rhs} : "
        f"{lhs_type}, {rhs_type} -> {result_type}"
    )
    return result, result_type


def _materialize_coordinate_compare(state, value, coords):
    width = _coord_width(coords)
    lhs, lhs_type = _materialize_coordinate_value(state, value.lhs_id, coords)
    rhs, rhs_type = _materialize_coordinate_value(state, value.rhs_id, coords)
    target_type = lhs_type if lhs_type.startswith("!wave.simd<") else rhs_type
    if target_type.startswith("!wave.simd<"):
        lhs, lhs_type = _ensure_coordinate_type(state, lhs, lhs_type, target_type)
        rhs, rhs_type = _ensure_coordinate_type(state, rhs, rhs_type, target_type)
        result_type = f"!wave.mask<{width}>"
        result = state.fresh()
        state.emit(
            f"{result} = wave.cmpi {value.predicate} {lhs}, {rhs} : "
            f"{lhs_type}, {rhs_type} -> {result_type}"
        )
        return result, result_type
    result = state.fresh()
    state.emit(f"{result} = arith.cmpi {value.predicate}, {lhs}, {rhs} : {lhs_type}")
    return result, "i1"


def _materialize_coordinate_mask_and(state, value, coords):
    width = _coord_width(coords)
    result_type = f"!wave.mask<{width}>"
    lhs, lhs_type = _materialize_coordinate_value(state, value.lhs_id, coords)
    rhs, rhs_type = _materialize_coordinate_value(state, value.rhs_id, coords)
    lhs, lhs_type = _ensure_coordinate_type(state, lhs, lhs_type, result_type)
    rhs, rhs_type = _ensure_coordinate_type(state, rhs, rhs_type, result_type)
    false_mask = state.fresh()
    result = state.fresh()
    state.emit(f"{false_mask} = wave.constant false -> {result_type}")
    state.emit(
        f"{result} = wave.select {lhs}, {rhs}, {false_mask} : "
        f"{result_type}, {result_type}"
    )
    return result, result_type


def _materialize_coordinate_min(state, value, coords):
    width = _coord_width(coords)
    result_type = _coordinate_wave_type(state, value.value_id, width)
    lhs, lhs_type = _materialize_coordinate_value(state, value.lhs_id, coords)
    rhs, rhs_type = _materialize_coordinate_value(state, value.rhs_id, coords)
    lhs, lhs_type = _ensure_coordinate_type(state, lhs, lhs_type, result_type)
    rhs, rhs_type = _ensure_coordinate_type(state, rhs, rhs_type, result_type)
    mask_type = f"!wave.mask<{width}>"
    pred = state.fresh()
    result = state.fresh()
    state.emit(
        f"{pred} = wave.cmpi {value.predicate} {lhs}, {rhs} : "
        f"{lhs_type}, {rhs_type} -> {mask_type}"
    )
    state.emit(
        f"{result} = wave.select {pred}, {lhs}, {rhs} : "
        f"{mask_type}, {result_type}"
    )
    return result, result_type


def _materialize_coordinate_pointer_add(state, value, coords):
    width = _coord_width(coords)
    result_type = _coordinate_wave_type(state, value.value_id, width)
    base, base_type, offset, offset_type = _materialize_pointer_parts_at_coords(
        state, value.value_id, coords
    )
    if offset is None:
        return base, base_type
    result = state.fresh()
    state.emit(
        f"{result} = wave.ptr_add {base}, {offset} : "
        f"{base_type}, {offset_type} -> {result_type}"
    )
    return result, result_type


def _pointer_scalar_base_id(state, value_id):
    value = state.converted_values.get(value_id)
    if isinstance(value, _PointerAddValue):
        return _pointer_scalar_base_id(state, value.base_id)
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            operand_type = _source_type(state, value.operand_id)
            if operand_type.kind == "pointer":
                return value.operand_id
        if value.op_name == "tt.broadcast":
            return _pointer_scalar_base_id(state, value.operand_id)
    if isinstance(value, _ForwardValue) and value.op_name in {
        "tt.expand_dims",
        "ttg.convert_layout",
    }:
        return _pointer_scalar_base_id(state, value.operand_id)
    return value_id


def _materialize_pointer_parts_at_coords(state, value_id, coords):
    value = state.converted_values.get(value_id)
    if isinstance(value, _PointerAddValue):
        base, base_type, base_offset, base_offset_type = _materialize_pointer_parts_at_coords(
            state, value.base_id, coords
        )
        offset, offset_type = _materialize_coordinate_value(state, value.offset_id, coords)
        offset, offset_type = _ensure_coordinate_offset_type(
            state, offset, offset_type, _coord_width(coords)
        )
        if base_offset is not None:
            base_offset, base_offset_type = _ensure_coordinate_type(
                state, base_offset, base_offset_type, offset_type
            )
            combined = state.fresh()
            state.emit(
                f"{combined} = wave.binary addi {base_offset}, {offset} : "
                f"{base_offset_type}, {offset_type} -> {offset_type}"
            )
            offset = combined
            offset_type = offset_type
        return base, base_type, offset, offset_type
    if isinstance(value, _UnaryTensorValue) and value.op_name == "tt.splat":
        operand_type = _source_type(state, value.operand_id)
        if operand_type.kind == "pointer":
            return _require_name(state, value.operand_id), _require_type(state, value.operand_id), None, None
    if isinstance(value, _UnaryTensorValue) and value.op_name == "tt.broadcast":
        operand_coords = _broadcast_operand_coords(
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
            coords,
        )
        return _materialize_pointer_parts_at_coords(state, value.operand_id, operand_coords)
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
            )
            return _materialize_pointer_parts_at_coords(
                state,
                value.operand_id,
                coords[:axis] + coords[axis + 1 :],
            )
        if value.op_name == "ttg.convert_layout":
            return _materialize_pointer_parts_at_coords(state, value.operand_id, coords)
    ptr, ptr_type = _materialize_coordinate_value(state, value_id, coords)
    return ptr, ptr_type, None, None


def _ensure_coordinate_offset_type(state, name, value_type, width):
    if value_type.startswith("!wave.simd<"):
        return name, value_type
    target_type = f"!wave.simd<{value_type}, {width}>"
    return _ensure_coordinate_type(state, name, value_type, target_type)


def _ensure_coordinate_type(state, name, value_type, target_type):
    if value_type == target_type:
        return name, value_type
    if not _is_wave_sequence_type(value_type) and target_type.startswith("!wave.simd<"):
        splat = state.fresh()
        state.emit(f"{splat} = wave.splat {name} : {value_type} -> {target_type}")
        return splat, target_type
    if value_type == "i1" and target_type.startswith("!wave.mask<"):
        return _emit_scalar_i1_to_mask(state, name, target_type), target_type
    raise ValueError(
        "tlx_wave new emitter cannot adapt coordinate value type "
        f"{value_type} -> {target_type}"
    )


def _emit_coord_cast(state, kind, name, value_type, result_type):
    if value_type == result_type:
        return name, value_type
    result = state.fresh()
    state.emit(f"{result} = wave.cast {kind} {name} : {value_type} -> {result_type}")
    return result, result_type


def _coordinate_wave_type(state, value_id, width):
    source_type = _source_type(state, value_id)
    if source_type.kind == "tensor":
        if source_type.element_type == "i1":
            return f"!wave.mask<{width}>"
        if source_type.pointee_type is not None:
            address_space = "wave.shared" if source_type.address_space == 3 else "wave.global"
            element = f"!wave.ptr<#{address_space}, {source_type.pointee_type}>"
        else:
            element = source_type.element_type or "unknown"
        return f"!wave.simd<{element}, {width}>"
    return _require_type(state, value_id)


def _broadcast_operand_coords(operand_shape, result_shape, coords):
    if len(operand_shape) > len(result_shape):
        raise ValueError(
            "tlx_wave new emitter cannot broadcast from higher-rank operand "
            f"{operand_shape} to {result_shape}"
        )
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, coord in zip(
        operand_shape, result_shape[leading:], coords[leading:]
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(coord)
        elif int(operand_dim) == 1:
            mapped.append(_zero_coord_like(coord))
        else:
            raise ValueError(
                "tlx_wave new emitter cannot map broadcast coordinate from "
                f"{operand_shape} to {result_shape}"
            )
    return tuple(mapped)


def _expand_dims_axis(operand_shape, result_shape):
    candidates = []
    for axis in range(len(result_shape)):
        if int(result_shape[axis]) != 1:
            continue
        if tuple(result_shape[:axis] + result_shape[axis + 1 :]) == tuple(operand_shape):
            candidates.append(axis)
    if len(candidates) != 1:
        raise ValueError(
            "tlx_wave new emitter cannot derive unique expand_dims axis for "
            f"{operand_shape} -> {result_shape}"
        )
    return candidates[0]


def _zero_coord_like(coord):
    _name, coord_type = coord
    return ("0", coord_type)


def _coord_width(coords):
    if not coords:
        raise ValueError("tlx_wave new emitter needs coordinates for tensor value")
    return _simd_width(coords[0][1])


def _constant_scalar_literal(value):
    if isinstance(value.literal, (bool, int, float)):
        return value.literal
    raw = "" if value.raw_literal is None else value.raw_literal.strip()
    match = re.match(r"dense<([^>]+)>", raw)
    if match:
        inner = match.group(1).strip()
        if inner in {"true", "false"}:
            return inner == "true"
        try:
            return int(inner, 0)
        except ValueError:
            try:
                return float(inner)
            except ValueError:
                return None
    return None


def _fragment_type_info(wave_type):
    match = re.fullmatch(
        r"!waveamd\.fragment<\s*(\d+)\s*,\s*([^,]+)\s*,\s*(\d+)\s*,"
        r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>",
        wave_type,
    )
    if match is None:
        raise ValueError(f"tlx_wave new emitter expected fragment type, got {wave_type}")
    return _FragmentTypeInfo(
        int(match.group(1)),
        match.group(2).strip(),
        int(match.group(3)),
        int(match.group(4)),
        int(match.group(5)),
        int(match.group(6)),
    )


def _is_fragment_wave_type(wave_type):
    return isinstance(wave_type, str) and wave_type.startswith("!waveamd.fragment<")


def _emit_simd_add_const(state, value, value_type, constant):
    if int(constant) == 0:
        return value
    return _emit_simd_binary_const(state, "addi", value, value_type, constant)


def _emit_simd_binary_const(state, kind, value, value_type, constant):
    if kind == "divui" and _is_positive_power_of_two(int(constant)):
        if int(constant) == 1:
            return value
        kind = "shrui"
        constant = int(constant).bit_length() - 1
    elif kind == "remui" and _is_positive_power_of_two(int(constant)):
        if int(constant) == 1:
            const = state.fresh()
            splat = state.fresh()
            element_type = _simd_element_type(value_type)
            state.emit(f"{const} = arith.constant 0 : {element_type}")
            state.emit(f"{splat} = wave.splat {const} : {element_type} -> {value_type}")
            return splat
        kind = "andi"
        constant = int(constant) - 1
    const = state.fresh()
    splat = state.fresh()
    result = state.fresh()
    element_type = _simd_element_type(value_type)
    state.emit(f"{const} = arith.constant {int(constant)} : {element_type}")
    state.emit(f"{splat} = wave.splat {const} : {element_type} -> {value_type}")
    state.emit(
        f"{result} = wave.binary {kind} {value}, {splat} : "
        f"{value_type}, {value_type} -> {value_type}"
    )
    return result


def _emit_simd_binary(state, kind, lhs, rhs, value_type):
    result = state.fresh()
    state.emit(
        f"{result} = wave.binary {kind} {lhs}, {rhs} : "
        f"{value_type}, {value_type} -> {value_type}"
    )
    return result


def _emit_store(state, effect):
    value_type = _require_type(state, effect.value_id)
    if _is_fragment_wave_type(value_type):
        _emit_fragment_store(state, effect)
        return
    value_components, ptr_components, mask_components, ptr_type, mask_type = (
        _store_component_operands(state, effect)
    )
    token = _ordered_mem_token(state)
    for index, (value, ptr) in enumerate(zip(value_components, ptr_components)):
        mask = None if mask_components is None else mask_components[index]
        token = _emit_component_store_text(
            state,
            value,
            value_type,
            ptr,
            ptr_type,
            mask,
            mask_type,
            token,
        )
    state.mem_token = token


def _emit_buffer_store(state, effect):
    value_type = _require_type(state, effect.value_id)
    if _is_fragment_wave_type(value_type):
        _emit_fragment_store(state, effect)
        return
    raise ValueError(
        "tlx_wave new emitter supports amdg.buffer_store only for fragment "
        f"values so far, got {value_type}"
    )


def _emit_local_store(state, effect):
    if _is_fragment_wave_type(_require_type(state, effect.value_id)):
        _emit_fragment_local_store(state, effect)
        return
    value = _require_name(state, effect.value_id)
    value_type = _require_type(state, effect.value_id)
    value_components = _as_components(value)
    ptr_components, ptr_type = _local_linear_pointers(
        state,
        effect.memdesc_id,
        effect.value_id,
        value_type,
        len(value_components),
    )
    token = _ordered_memory_token_operand(state, effect.token_id)
    for value_component, ptr in zip(value_components, ptr_components):
        store = state.fresh()
        state.emit(
            f"{store} = wave.store {value_component} -> {ptr} after {token} : "
            f"({value_type}, {ptr_type}, !wave.mem.token) -> !wave.mem.token"
        )
        token = store
    barrier = state.fresh()
    state.emit(f"{barrier} = wave.barrier {token} : (!wave.mem.token) -> !wave.mem.token")
    state.mem_token = barrier


def _emit_fragment_local_store(state, effect):
    logical_type = _source_type(state, effect.value_id)
    memdesc_type = _source_type(state, effect.memdesc_id)
    if logical_type.kind != "tensor" or len(logical_type.shape) != 2:
        raise ValueError(
            "tlx_wave new emitter supports fragment local_store only for rank-2 "
            f"tensor values, got {logical_type.raw}"
        )
    if logical_type.element_type not in {"f32", "f16"}:
        raise ValueError(
            "tlx_wave new emitter supports fragment local_store only for f32/f16 "
            f"tensor values, got {logical_type.raw}"
        )
    base_element = memdesc_type.element_type
    if base_element != logical_type.element_type:
        raise ValueError(
            "tlx_wave new emitter cannot lower fragment local_store with "
            f"mismatched value/memdesc element types: {logical_type.element_type} "
            f"vs {base_element}"
        )
    physical_type = _require_type(state, effect.value_id)
    frag_info = _fragment_type_info(physical_type)
    if frag_info.role != 2:
        raise ValueError(
            "tlx_wave new emitter supports local stores only from accumulator "
            f"fragments, got {physical_type}"
        )
    fragments = _as_components(_require_name(state, effect.value_id))
    mma = _fragment_store_mma_shape(state, effect.value_id)
    tile_rows = _tile_rep_count(logical_type.shape[0], mma.output_tile_shape[0])
    tile_cols = _tile_rep_count(logical_type.shape[1], mma.output_tile_shape[1])
    if len(fragments) != tile_rows * tile_cols:
        raise ValueError(
            "tlx_wave new emitter fragment local_store tile mismatch: got "
            f"{len(fragments)} fragment(s), expected {tile_rows * tile_cols}"
        )
    token = _ordered_memory_token_operand(state, effect.token_id)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            fragment = fragments[_tile_index((tile_rows, tile_cols), tile_row, tile_col)]
            store_vector, store_vector_type = _emit_fragment_store_vector(
                state, fragment, physical_type, logical_type, frag_info
            )
            tile_offsets = (
                tile_row * mma.output_tile_shape[0],
                tile_col * mma.output_tile_shape[1],
            )
            component_coords = _emit_fragment_component_coords(state, frag_info, tile_offsets)
            for component, coords in enumerate(component_coords):
                value_name, value_type = _emit_fragment_store_component_value(
                    state,
                    store_vector,
                    store_vector_type,
                    logical_type,
                    frag_info,
                    component,
                )
                ptr_name, ptr_type = _materialize_fragment_local_store_pointer(
                    state,
                    effect.memdesc_id,
                    memdesc_type,
                    coords,
                    value_type,
                )
                token = _emit_component_store_text(
                    state,
                    value_name,
                    value_type,
                    ptr_name,
                    ptr_type,
                    None,
                    None,
                    token,
                )
    barrier = state.fresh()
    state.emit(f"{barrier} = wave.barrier {token} : (!wave.mem.token) -> !wave.mem.token")
    state.mem_token = barrier


def _emit_masked_store(state, effect):
    _emit_store(state, effect)


def _store_component_operands(state, effect):
    value_name = _require_name(state, effect.value_id)
    ptr_name = _require_name(state, effect.pointer_id)
    value_components = _as_components(value_name)
    ptr_components = _as_components(ptr_name)
    mask_components = None
    if effect.mask_id is not None:
        mask_components = _as_components(_require_name(state, effect.mask_id))
    component_count = max(
        len(value_components),
        len(ptr_components),
        0 if mask_components is None else len(mask_components),
    )
    value_components = _remap_component_tuple(
        value_components, component_count, "store value"
    )
    ptr_components = _remap_component_tuple(
        ptr_components, component_count, "store pointer"
    )
    if mask_components is not None:
        mask_components = _remap_component_tuple(
            mask_components, component_count, "store mask"
        )
    ptr_type = _require_type(state, effect.pointer_id)
    mask_type = None if effect.mask_id is None else _require_type(state, effect.mask_id)
    return value_components, ptr_components, mask_components, ptr_type, mask_type


def _emit_effect(state, effect):
    if isinstance(effect, _StoreEffect):
        _emit_store(state, effect)
    elif isinstance(effect, _BufferStoreEffect):
        _emit_buffer_store(state, effect)
    elif isinstance(effect, _LocalStoreEffect):
        _emit_local_store(state, effect)
    elif isinstance(effect, _ForEffect):
        _emit_for_effect(state, effect)
    elif isinstance(effect, _IfEffect):
        _emit_if_effect(state, effect)
    elif isinstance(effect, _AssumeEffect):
        _emit_assume_effect(state, effect)
    elif isinstance(effect, _AsyncWaitEffect):
        _emit_async_wait(state, effect)
    elif isinstance(effect, _ReturnEffect):
        _emit_return(state, effect)
    else:
        raise ValueError(
            "tlx_wave new emitter cannot emit converted effect "
            f"{type(effect).__name__}"
        )


def _emit_return(state, effect):
    if effect.operand_ids:
        operands = ", ".join(state.names[value_id] for value_id in effect.operand_ids)
        types = ", ".join(state.types[value_id] for value_id in effect.operand_ids)
        state.emit(f"return {operands} : {types}")
    else:
        state.emit("return")


def _emit_assume_effect(state, effect):
    facts = _assume_facts(effect.predicate)
    if not facts:
        return
    for value_id, predicate_text in facts:
        if value_id not in state.names and value_id in state.assume_condition_values:
            continue
        value_type = _require_type(state, value_id)
        if value_type.startswith("!wave.simd<") or value_type.startswith("!wave.mask<"):
            continue
        value_name = _require_name(state, value_id)
        assumed = state.fresh()
        state.emit(
            f"{assumed} = wave.assume {value_name} as \"x\" "
            f"[#wave.pred<\"{predicate_text}\">] : {value_type}"
        )
        state.names[value_id] = assumed
        state.types[value_id] = value_type


def _ensure_mem_token(state):
    if state.mem_token is None:
        state.mem_token = state.fresh()
        state.emit(f"{state.mem_token} = wave.token : !wave.mem.token")
    return state.mem_token


def _ordered_mem_token(state):
    base_token = _ensure_mem_token(state)
    if not state.pending_read_tokens:
        return base_token
    token = _join_memory_tokens(state, (*state.pending_read_tokens, base_token), base_token)
    state.pending_read_tokens = ()
    state.mem_token = token
    return token


def _memory_token_operand(state, token_id):
    if token_id is None:
        return _ensure_mem_token(state)
    return _require_name(state, token_id)


def _ordered_memory_token_operand(state, token_id):
    if token_id is None:
        return _ordered_mem_token(state)
    token = _require_name(state, token_id)
    if not state.pending_read_tokens:
        return token
    ordered = _join_memory_tokens(state, (*state.pending_read_tokens, token), token)
    state.pending_read_tokens = ()
    state.mem_token = ordered
    return ordered


def _pack_components(components):
    components = tuple(components)
    if len(components) == 1:
        return components[0]
    return components


def _as_components(value):
    if isinstance(value, tuple):
        return value
    return (value,)


def _is_wave_sequence_type(value_type):
    return (
        value_type.startswith("!wave.simd<")
        or value_type.startswith("!wave.mask<")
        or value_type.startswith("!waveamd.fragment<")
    )


def _remap_component_tuple(components, result_count, context):
    components = tuple(components)
    if len(components) == result_count:
        return components
    if len(components) == 1:
        return components * result_count
    if result_count % len(components):
        raise ValueError(
            "tlx_wave new emitter cannot remap component tuple for "
            f"{context}: source components={len(components)}, "
            f"result components={result_count}"
        )
    repeat = result_count // len(components)
    remapped = []
    for component in components:
        remapped.extend(component for _ in range(repeat))
    return tuple(remapped)


def _value_components_for_count(state, value_id, count, target_type, context):
    name = _require_name(state, value_id)
    value_type = _require_type(state, value_id)
    components = _as_components(name)
    if value_type == target_type:
        return _remap_component_tuple(components, count, context)
    if not _is_wave_sequence_type(value_type):
        if _is_wave_sequence_type(target_type):
            splats = []
            for _ in range(count):
                splat = state.fresh()
                state.emit(
                    f"{splat} = wave.splat {name} : {value_type} -> {target_type}"
                )
                splats.append(splat)
            return tuple(splats)
        if _can_index_cast_scalar(value_type, target_type):
            cast = state.fresh()
            state.emit(
                f"{cast} = {_emit_scalar_intconvert_text(name, value_type, target_type)}"
            )
            return (cast,) * count
        raise ValueError(
            "tlx_wave new emitter cannot adapt scalar component for "
            f"{context}: {value_type} -> {target_type}"
        )
    raise ValueError(
        "tlx_wave new emitter cannot adapt component tuple for "
        f"{context}: {value_type} -> {target_type}"
    )


def _can_index_cast_scalar(source_type, target_type):
    return _is_index_or_integer_scalar(source_type) and _is_index_or_integer_scalar(
        target_type
    )


def _emit_scalar_intconvert_text(value, source_type, target_type):
    policy = ""
    source_bits = _scalar_integer_bits(source_type)
    target_bits = _scalar_integer_bits(target_type)
    if target_type == "index" and source_bits is not None:
        policy = " policy {extension = #wave.cast_extension<zero>}"
    elif (
        source_bits is not None
        and target_bits is not None
        and source_bits < target_bits
    ):
        policy = " policy {extension = #wave.cast_extension<zero>}"
    return (
        f"wave.cast intconvert {value}{policy} : "
        f"{source_type} -> {target_type}"
    )


def _scalar_integer_bits(value_type):
    if value_type == "index":
        return None
    match = re.fullmatch(r"i(\d+)", value_type or "")
    if match is None:
        return None
    return int(match.group(1))


def _is_index_or_integer_scalar(value_type):
    return value_type == "index" or re.fullmatch(r"i\d+", value_type or "") is not None


def _index_offset_components(state, value_id, count):
    name = _require_name(state, value_id)
    value_type = _require_type(state, value_id)
    components = _as_components(name)
    if value_type.startswith("!wave.simd<index,"):
        return _remap_component_tuple(components, count, "index offset"), value_type
    if not value_type.startswith("!wave.simd<"):
        raise ValueError(
            "tlx_wave new emitter expected SIMD pointer offset, got "
            f"{value_type}"
        )
    index_type = _simd_index_type(value_type)
    converted_components = []
    for component in _remap_component_tuple(components, count, "index offset"):
        converted = state.fresh()
        state.emit(
            f"{converted} = wave.index_expr <\"x\"> [\"x\"]({component}) : "
            f"({value_type}) -> {index_type}"
        )
        converted_components.append(converted)
    return tuple(converted_components), index_type


def _index_offset(state, value_id):
    components, value_type = _index_offset_components(state, value_id, 1)
    return components[0], value_type


def _local_linear_pointers(state, memdesc_id, tensor_value_id, value_type, component_count):
    _validate_linear_tensor_components(state, tensor_value_id, value_type, component_count)
    base = _require_name(state, memdesc_id)
    base_type = _require_type(state, memdesc_id)
    base_element = _pointer_element_type(base_type)
    value_element = _simd_element_type(value_type)
    if base_element != value_element:
        raise ValueError(
            "tlx_wave new emitter cannot map local memory with mismatched "
            f"element types: {base_element} vs {value_element}"
        )
    offset_type = _simd_i32_type(value_type)
    ptr_type = _shared_simd_pointer_type(base_type, offset_type)
    lane = state.fresh()
    state.emit(f"{lane} = wave.workitem_id 0 : {offset_type}")
    width = _simd_width(offset_type)
    ptrs = []
    for component in range(int(component_count)):
        offset = lane
        if component:
            const = state.fresh()
            splat = state.fresh()
            shifted = state.fresh()
            state.emit(f"{const} = arith.constant {component * width} : i32")
            state.emit(f"{splat} = wave.splat {const} : i32 -> {offset_type}")
            state.emit(
                f"{shifted} = wave.binary addi {lane}, {splat} : "
                f"{offset_type}, {offset_type} -> {offset_type}"
            )
            offset = shifted
        ptr = state.fresh()
        state.emit(
            f"{ptr} = wave.ptr_add {base}, {offset} : "
            f"{base_type}, {offset_type} -> {ptr_type}"
        )
        ptrs.append(ptr)
    return tuple(ptrs), ptr_type


def _materialize_fragment_local_store_pointer(
    state,
    memdesc_id,
    memdesc_type,
    coords,
    value_type,
):
    base = _require_name(state, memdesc_id)
    base_type = _require_type(state, memdesc_id)
    base_element = _pointer_element_type(base_type)
    value_element = _simd_element_type(value_type)
    if base_element != value_element:
        raise ValueError(
            "tlx_wave new emitter cannot map fragment local_store with "
            f"mismatched pointer/value element types: {base_element} vs {value_element}"
        )
    row, row_type = coords[-2]
    col, col_type = coords[-1]
    if row_type != col_type:
        raise ValueError(
            "tlx_wave new emitter cannot map fragment local_store with mixed "
            f"coordinate types: {row_type} vs {col_type}"
        )
    offset, offset_type = _emit_memdesc_element_offset_from_coords(
        state,
        memdesc_type,
        row,
        col,
        row_type,
        "fragment local_store",
    )
    ptr_type = _shared_simd_pointer_type(base_type, offset_type)
    ptr = state.fresh()
    state.emit(
        f"{ptr} = wave.ptr_add {base}, {offset} : "
        f"{base_type}, {offset_type} -> {ptr_type}"
    )
    return ptr, ptr_type


def _emit_memdesc_element_offset_from_coords(
    state,
    memdesc_type,
    row,
    col,
    coord_type,
    context,
):
    if len(memdesc_type.shape) != 2:
        raise ValueError(
            f"tlx_wave new emitter supports {context} only for rank-2 "
            f"memdescs, got {memdesc_type.shape}"
        )
    if _is_identity_shared_memdesc(memdesc_type):
        return _emit_row_major_element_from_coords(
            state, row, col, coord_type, int(memdesc_type.shape[-1])
        ), coord_type
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if swizzled is not None:
        return (
            _emit_swizzled_element_from_coords(
                state, row, col, coord_type, memdesc_type, swizzled
            ),
            coord_type,
        )
    padded = _padded_shared_memdesc_info(memdesc_type)
    if padded is None:
        raise ValueError(
            f"tlx_wave new emitter cannot lower {context}: unsupported shared "
            f"layout {memdesc_type.encoding}"
        )
    if padded["order"] != tuple(reversed(range(len(memdesc_type.shape)))):
        raise ValueError(
            f"tlx_wave new emitter cannot lower {context}: unsupported padded "
            f"order {padded['order']}"
        )
    logical = _emit_row_major_element_from_coords(
        state, row, col, coord_type, int(memdesc_type.shape[-1])
    )
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        quotient = _emit_simd_binary_const(
            state, "divui", logical, coord_type, int(interval)
        )
        pad = _emit_simd_binary_const(
            state, "muli", quotient, coord_type, int(padding)
        )
        encoded = _emit_simd_binary(state, "addi", encoded, pad, coord_type)
    return encoded, coord_type


def _emit_row_major_element_from_coords(state, row, col, coord_type, cols):
    row_scaled = _emit_simd_binary_const(state, "muli", row, coord_type, int(cols))
    return _emit_simd_binary(state, "addi", row_scaled, col, coord_type)


def _shared_pointer_type_for_source(state, value_id):
    source_type = _source_type(state, value_id)
    element_type = source_type.element_type or "i8"
    return f"!wave.ptr<#wave.shared, {element_type}>"


def _memdesc_index_elements_per_slot(state, value):
    result_type = _source_type(state, value.value_id)
    elements_per_slot = 1
    for dim in result_type.alloc_shape or result_type.shape:
        elements_per_slot *= int(dim)
    return elements_per_slot


def _validate_linear_tensor_components(state, value_id, value_type, component_count):
    source_type = _source_type(state, value_id)
    if source_type.kind != "tensor":
        raise ValueError(
            "tlx_wave new emitter expected tensor value for local memory access, "
            f"got {source_type.raw}"
        )
    expected_width = _simd_width(value_type)
    element_count = 1
    for dim in source_type.shape:
        element_count *= int(dim)
    if element_count != expected_width * int(component_count):
        raise ValueError(
            "tlx_wave new emitter supports local load/store only for "
            "whole-component linear tensors; got "
            f"{source_type.raw} with {element_count} elements for {value_type}"
        )


def _simple_assume_fact(predicate):
    if isinstance(predicate, _ConstantValue):
        return None
    if not isinstance(predicate, _CompareValue):
        return None
    lhs_const = _constant_int_record(predicate.lhs)
    rhs_const = _constant_int_record(predicate.rhs)
    if rhs_const is not None:
        return _predicate_with_rhs_const(predicate.lhs_id, predicate.predicate, rhs_const)
    if lhs_const is not None:
        inverted = _invert_predicate(predicate.predicate)
        if inverted is None:
            return None
        return _predicate_with_rhs_const(predicate.rhs_id, inverted, lhs_const)
    return None


def _assume_facts(predicate):
    raw_facts = _raw_assume_facts(predicate)
    facts_by_id = {}
    for value_id, predicate_text in raw_facts:
        facts_by_id.setdefault(value_id, set()).add(predicate_text)
    filtered = []
    seen = set()
    for fact in raw_facts:
        value_id, predicate_text = fact
        if predicate_text == "x & (x - 1) == 0":
            predicates = facts_by_id.get(value_id, set())
            if "x > 0" not in predicates or "x <= 2147483647" not in predicates:
                continue
        if fact in seen:
            continue
        seen.add(fact)
        filtered.append(fact)
    return tuple(filtered)


def _raw_assume_facts(predicate):
    if isinstance(predicate, _MaskAndValue):
        return _raw_assume_facts(predicate.lhs) + _raw_assume_facts(predicate.rhs)
    if isinstance(predicate, _BinaryValue) and predicate.kind == "andi":
        return _raw_assume_facts(predicate.lhs) + _raw_assume_facts(predicate.rhs)
    pow2_fact = _power_of_two_assume_fact(predicate)
    if pow2_fact is not None:
        return (pow2_fact,)
    simple_fact = _simple_assume_fact(predicate)
    return () if simple_fact is None else (simple_fact,)


def _power_of_two_assume_fact(predicate):
    if not isinstance(predicate, _CompareValue) or predicate.predicate != "eq":
        return None
    lhs_const = _constant_int_record(predicate.lhs)
    rhs_const = _constant_int_record(predicate.rhs)
    if rhs_const == 0:
        bits = predicate.lhs
    elif lhs_const == 0:
        bits = predicate.rhs
    else:
        return None
    if not isinstance(bits, _BinaryValue) or bits.kind != "andi":
        return None
    lhs_base = _sub_one_base_id(bits.lhs)
    rhs_base = _sub_one_base_id(bits.rhs)
    if lhs_base is not None and lhs_base == bits.rhs_id:
        return lhs_base, "x & (x - 1) == 0"
    if rhs_base is not None and rhs_base == bits.lhs_id:
        return rhs_base, "x & (x - 1) == 0"
    return None


def _sub_one_base_id(value):
    if not isinstance(value, _BinaryValue) or value.kind != "subi":
        return None
    rhs_const = _constant_int_record(value.rhs)
    if rhs_const != 1:
        return None
    return value.lhs_id


def _predicate_with_rhs_const(value_id, predicate, constant):
    op = {
        "eq": "==",
        "ne": "!=",
        "slt": "<",
        "sle": "<=",
        "sgt": ">",
        "sge": ">=",
    }.get(predicate)
    if op is None:
        return None
    return value_id, f"x {op} {int(constant)}"


def _invert_predicate(predicate):
    return {
        "eq": "eq",
        "ne": "ne",
        "slt": "sgt",
        "sle": "sge",
        "sgt": "slt",
        "sge": "sle",
        "ult": "ugt",
        "ule": "uge",
        "ugt": "ult",
        "uge": "ule",
    }.get(predicate)


def _constant_int_record(value):
    if isinstance(value, _ConstantValue):
        literal = _constant_scalar_literal(value)
        if isinstance(literal, int) and not isinstance(literal, bool):
            return literal
    return None


def _constant_literal_text(value):
    raw = value.raw_literal
    if raw is None:
        scalar_type = value.converted_type.wave_type
        return f"{_format_literal(value.literal)} : {scalar_type}"
    text = raw.strip()
    if text in {"True", "False"}:
        text = text.lower()
    if ":" in text or text in {"true", "false"}:
        return text
    return f"{text} : {value.converted_type.wave_type}"


def _format_literal(literal):
    if literal is True:
        return "true"
    if literal is False:
        return "false"
    return str(literal)


def _require_name(state, value_id):
    _ensure_value_emitted(state, value_id)
    if value_id not in state.names:
        raise ValueError(
            f"tlx_wave new emitter cannot emit value {value_id}: operand has no SSA name"
        )
    return state.names[value_id]


def _require_type(state, value_id):
    _ensure_value_emitted(state, value_id)
    if value_id not in state.types:
        raise ValueError(
            f"tlx_wave new emitter cannot emit value {value_id}: operand has no Wave type"
        )
    return state.types[value_id]


def _ensure_value_emitted(state, value_id):
    if value_id in state.names and value_id in state.types:
        return
    value = state.converted_values.get(value_id)
    if value is None:
        return
    _emit_value(state, value)


def _input_kind(value):
    converted_type = value.converted_type
    return converted_type.kind


def _int_or(value, default):
    return default if value is None else int(value)


def _simd_width(wave_type):
    try:
        return int(wave_type.rsplit(",", 1)[1].split(">", 1)[0].strip())
    except (IndexError, ValueError) as exc:
        raise ValueError(f"tlx_wave new emitter expected SIMD type, got {wave_type}") from exc


def _simd_element_type(wave_type):
    prefix = "!wave.simd<"
    if not wave_type.startswith(prefix):
        raise ValueError(f"tlx_wave new emitter expected SIMD type, got {wave_type}")
    return wave_type[len(prefix) :].rsplit(",", 1)[0].strip()


def _simd_index_type(wave_type):
    return f"!wave.simd<index, {_simd_width(wave_type)}>"


def _simd_i32_type(wave_type):
    return f"!wave.simd<i32, {_simd_width(wave_type)}>"


def _cmp_result_type(value_type):
    if value_type.startswith("!wave.simd<"):
        return f"!wave.mask<{_simd_width(value_type)}>"
    return "i1"


def _compute_lds_layout(program):
    offsets = {}
    cursor = 0
    for op in program.ops:
        if op.name != "ttg.local_alloc" or not op.results:
            continue
        value_id = op.results[0]
        size = _local_alloc_size_bytes(program.values[value_id].type)
        offsets[value_id] = cursor
        cursor = _align_to(cursor + size, 16)
    return offsets, cursor


def _local_alloc_size_bytes(source_type):
    byte_width = source_type.element_byte_width
    if byte_width is None:
        raise ValueError(
            f"tlx_wave new emitter cannot size LDS allocation {source_type.raw}: "
            "unknown element byte width"
        )
    count = 1
    for dim in source_type.alloc_shape or source_type.shape:
        count *= int(dim)
    return count * int(byte_width)


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _align_to(value, alignment):
    if value == 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


def _source_type(state, value_id):
    return state.source_values[value_id].type


def _shared_simd_pointer_type(base_type, offset_type):
    element_type = _pointer_element_type(base_type)
    if offset_type.startswith("!wave.simd<"):
        return f"!wave.simd<!wave.ptr<#wave.shared, {element_type}>, {_simd_width(offset_type)}>"
    return f"!wave.ptr<#wave.shared, {element_type}>"


def _simd_pointer_element_type(wave_type):
    return _pointer_element_type(_simd_element_type(wave_type))


def _pointer_element_type(pointer_type):
    marker = "!wave.ptr<#wave"
    if not pointer_type.startswith(marker):
        raise ValueError(f"tlx_wave new emitter expected Wave pointer type, got {pointer_type}")
    try:
        return pointer_type.split(",", 1)[1].rsplit(">", 1)[0].strip()
    except IndexError as exc:
        raise ValueError(
            f"tlx_wave new emitter cannot parse Wave pointer type {pointer_type}"
        ) from exc
