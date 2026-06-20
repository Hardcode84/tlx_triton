"""Structural Wave emission for verified target programs."""

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

from .diagnostics import fail
from . import target_ir


STAGE = "emission"


@dataclass(frozen=True)
class EmittedWaveModule:
    text: str
    lds_size: int = 0


@dataclass
class _EmissionState:
    dsl: object
    ir: object
    builder: object
    target_program: target_ir.TargetProgram
    fact_program: object | None
    values: dict[int, object]


def emit_wave_module(target_program, fact_program=None):
    dsl, ir = _load_wave_dsl()
    kernel = target_program.kernel
    lds_size = _target_lds_size(target_program)
    with dsl.ModuleBuilder() as module_builder:
        _set_module_attrs(module_builder, dsl, ir, kernel)
        arg_types = [
            _wave_type(dsl, target_program.values[target_value_id].type)
            for target_value_id in kernel.arg_target_ids
        ]
        with module_builder.gpu_module("kernels") as gpu_module:
            with gpu_module.kernel(
                kernel.name,
                arg_types,
                lds_size=lds_size or None,
                attrs=_function_attrs(dsl, ir, kernel),
            ) as builder:
                state = _EmissionState(
                    dsl,
                    ir,
                    builder,
                    target_program,
                    fact_program,
                    {},
                )
                for target_value_id, arg in zip(kernel.arg_target_ids, builder.args):
                    state.values[target_value_id] = arg
                for op in target_program.ops:
                    _emit_target_op(state, op)
        return EmittedWaveModule(str(module_builder), lds_size)


def _emit_target_op(state, op):
    if op.kind == "constant":
        _emit_constant(state, op)
        return
    if op.kind == "binary":
        _emit_binary(state, op)
        return
    if op.kind == "cmpi":
        _emit_cmpi(state, op)
        return
    if op.kind == "minsi":
        _emit_minsi(state, op)
        return
    if op.kind == "assume":
        _emit_assume(state, op)
        return
    if op.kind == "make_range":
        _emit_make_range(state, op)
        return
    if op.kind == "splat":
        _emit_splat(state, op)
        return
    if op.kind == "broadcast":
        _emit_broadcast(state, op)
        return
    if op.kind == "addptr":
        _emit_addptr(state, op)
        return
    if op.kind == "expand_dims":
        _emit_expand_dims(state, op)
        return
    if op.kind == "program_id":
        _emit_program_id(state, op)
        return
    if op.kind == "select":
        _emit_select(state, op)
        return
    if op.kind == "local_alloc":
        _emit_local_alloc(state, op)
        return
    if op.kind == "memdesc_index":
        _emit_memdesc_index(state, op)
        return
    if op.kind == "buffer_load_to_local":
        _emit_buffer_load_to_local(state, op)
        return
    if op.kind == "local_load_fragment":
        _emit_local_load_fragment(state, op)
        return
    if op.kind == "fragment_fill":
        _emit_fragment_fill(state, op)
        return
    if op.kind == "mma":
        _emit_mma(state, op)
        return
    if op.kind == "fragment_truncf":
        _emit_fragment_truncf(state, op)
        return
    if op.kind == "layout_convert":
        _emit_layout_convert(state, op)
        return
    if op.kind == "buffer_store":
        _emit_buffer_store(state, op)
        return
    if op.kind == "async_commit_group":
        _emit_async_commit_group(state, op)
        return
    if op.kind == "async_wait":
        _emit_async_wait(state, op)
        return
    if op.kind == "return":
        if op.operands:
            fail(
                "TLXW_EMIT_RETURN_VALUES",
                STAGE,
                "empty-return emission is supported first; return values are not",
                target_op_id=op.target_op_id,
            )
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_TARGET_OP",
        STAGE,
        f"no structural emission for target op {op.kind}",
        target_op_id=op.target_op_id,
    )


def _emit_constant(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    literal = attrs["value"]
    if target_type.representation == "scalar":
        state.values[result_id] = _scalar_constant(
            state,
            _scalar_type(state.dsl, target_type.element_type),
            target_type.element_type,
            literal,
            op,
        )
        return
    if target_type.representation in {"simd", "simd_tuple"}:
        value = _wave_constant(
            state,
            result_type,
            _scalar_type(state.dsl, target_type.element_type),
            target_type.element_type,
            literal,
            op,
        )
        state.values[result_id] = _pack_components(
            tuple(value for _ in range(_component_count(state, result_id)))
        )
        return
    if target_type.representation in {"mask", "mask_tuple"}:
        value = state.dsl.wave.ConstantOp(
            result_type,
            state.ir.Attribute.parse("true" if _literal_bool(literal, op) else "false"),
        ).result
        state.values[result_id] = _pack_components(
            tuple(value for _ in range(_component_count(state, result_id)))
        )
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_CONSTANT",
        STAGE,
        f"constant emission does not support {target_type.representation}",
        target_op_id=op.target_op_id,
        target_value_id=result_id,
    )


def _emit_binary(state, op):
    _materialize_fact_ids(state, op)
    attrs = target_ir.attrs_dict(op)
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    result_type = state.target_program.values[result_id].type
    count = _component_count(state, result_id)
    lhs_components, rhs_components = _broadcast_components((lhs, rhs), count, op)
    if result_type.representation in {"mask", "mask_tuple"}:
        if attrs["operation"] != "andi":
            fail(
                "TLXW_EMIT_UNSUPPORTED_MASK_BINARY",
                STAGE,
                f"unsupported mask binary operation {attrs['operation']}",
                target_op_id=op.target_op_id,
                target_value_id=result_id,
            )
        false_mask = _wave_mask_constant(
            state,
            _wave_type(state.dsl, result_type),
            False,
        )
        reused = []
        state.values[result_id] = _pack_components(
            tuple(
                _reuse_component_result(
                    reused,
                    (lhs_component, rhs_component, false_mask),
                    lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.select(
                        lhs_component,
                        rhs_component,
                        false_mask,
                    ),
                )
                for lhs_component, rhs_component in zip(lhs_components, rhs_components)
            )
        )
        return
    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.binary(
                    _binary_kind(state.dsl, attrs["operation"]),
                    lhs_component,
                    rhs_component,
                    nsw=bool(attrs.get("nsw", False)),
                    nuw=bool(attrs.get("nuw", False)),
                ),
            )
            for lhs_component, rhs_component in zip(lhs_components, rhs_components)
        )
    )


def _emit_cmpi(state, op):
    attrs = target_ir.attrs_dict(op)
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    lhs_components, rhs_components = _broadcast_components((lhs, rhs), count, op)
    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: _cmpi(
                    state,
                    attrs["predicate"],
                    lhs_component,
                    rhs_component,
                ),
            )
            for lhs_component, rhs_component in zip(lhs_components, rhs_components)
        )
    )


def _emit_minsi(state, op):
    lhs, rhs = _operand_values(state, op, 2)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    lhs_components, rhs_components = _broadcast_components((lhs, rhs), count, op)
    reused = []
    components = []
    for lhs_component, rhs_component in zip(lhs_components, rhs_components):
        components.append(
            _reuse_component_result(
                reused,
                (lhs_component, rhs_component),
                lambda lhs_component=lhs_component, rhs_component=rhs_component: state.builder.select(
                    _cmpi(state, "slt", lhs_component, rhs_component),
                    lhs_component,
                    rhs_component,
                ),
            )
        )
    state.values[result_id] = _pack_components(tuple(components))


def _emit_assume(state, op):
    _materialize_fact_ids(state, op)


def _materialize_fact_ids(state, op):
    if not op.fact_ids:
        return
    if state.fact_program is None:
        fail(
            "TLXW_EMIT_MISSING_FACT_PROGRAM",
            STAGE,
            "fact materialization requires verified fact records",
            target_op_id=op.target_op_id,
        )
    facts = {fact.fact_id: fact for fact in state.fact_program.facts}
    for fact_id in op.fact_ids:
        fact = facts.get(fact_id)
        if fact is None:
            fail(
                "TLXW_EMIT_UNKNOWN_FACT",
                STAGE,
                f"assume references missing fact {fact_id}",
                target_op_id=op.target_op_id,
                fact_id=fact_id,
            )
        target_ids = state.target_program.target_values_for_source(
            fact.subject_value_id
        )
        if len(target_ids) != 1:
            fail(
                "TLXW_EMIT_FACT_TARGET",
                STAGE,
                f"fact {fact_id} subject has target values {target_ids}",
                target_op_id=op.target_op_id,
                fact_id=fact_id,
            )
        target_value_id = target_ids[0]
        value = _require_value(state, target_value_id, op)
        assumptions = _range_assumptions(state.dsl, fact)
        if assumptions:
            state.values[target_value_id] = state.builder.assume(
                value,
                assumptions,
                name="x",
            )


def _emit_make_range(state, op):
    attrs = target_ir.attrs_dict(op)
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    width = int(target_type.lane_width or 64)
    element_type = _scalar_type(state.dsl, target_type.element_type)
    lane = state.builder.workitem_id(0, element_type, width)
    start = int(attrs["start"])
    components = []
    for component in range(_component_count(state, result_id)):
        component_start = start + component * width
        value = lane
        if component_start:
            start_value = state.builder.splat(
                state.builder.constant(element_type, component_start),
                element_type,
                width,
            )
            value = state.builder.binary(state.dsl.BinaryKind.AddI, lane, start_value)
        components.append(value)
    state.values[result_id] = _pack_components(tuple(components))


def _emit_splat(state, op):
    operand = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    target_type = state.target_program.values[result_id].type
    splat = state.builder.splat(
        operand,
        _splat_element_type(state.dsl, target_type),
        int(target_type.lane_width or 64),
    )
    state.values[result_id] = _pack_components(
        tuple(splat for _ in range(_component_count(state, result_id)))
    )


def _emit_broadcast(state, op):
    operand = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    target_count = _component_count(state, result_id)
    source_components = _as_components(operand)
    if target_count == len(source_components):
        state.values[result_id] = operand
        return
    if target_count % len(source_components) != 0:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BROADCAST",
            STAGE,
            "tt.broadcast requires the result component count to be a "
            "multiple of the source component count",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    repeat = target_count // len(source_components)
    state.values[result_id] = tuple(
        component
        for source_component in source_components
        for component in (source_component,) * repeat
    )


def _emit_addptr(state, op):
    base, offset = _operand_values(state, op, 2)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    base_components, offset_components = _broadcast_components((base, offset), count, op)
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    state.values[result_id] = _pack_components(
        tuple(
            state.builder.ptr_add(
                base_component,
                offset_component,
                result_type=result_type,
            )
            for base_component, offset_component in zip(base_components, offset_components)
        )
    )


def _emit_expand_dims(state, op):
    operand = _operand_values(state, op, 1)[0]
    result_id = _single_result(op)
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
    components = _as_components(operand)
    if any(str(component.type) != str(result_type) for component in components):
        fail(
            "TLXW_EMIT_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims changed the emitted Wave type; explicit remap is required",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    if len(components) != _component_count(state, result_id):
        fail(
            "TLXW_EMIT_UNSUPPORTED_REMAP",
            STAGE,
            "tt.expand_dims changed component count; explicit remap is required",
            target_op_id=op.target_op_id,
            target_value_id=result_id,
        )
    state.values[result_id] = operand


def _emit_program_id(state, op):
    attrs = target_ir.attrs_dict(op)
    state.values[_single_result(op)] = state.builder.workgroup_id(int(attrs["axis"]))


def _emit_select(state, op):
    condition, true_value, false_value = _operand_values(state, op, 3)
    result_id = _single_result(op)
    count = _component_count(state, result_id)
    cond_components, true_components, false_components = _broadcast_components(
        (condition, true_value, false_value),
        count,
        op,
    )
    reused = []
    state.values[result_id] = _pack_components(
        tuple(
            _reuse_component_result(
                reused,
                (condition_component, true_component, false_component),
                lambda condition_component=condition_component, true_component=true_component, false_component=false_component: state.builder.select(
                    condition_component,
                    true_component,
                    false_component,
                ),
            )
            for condition_component, true_component, false_component in zip(
                cond_components,
                true_components,
                false_components,
            )
        )
    )


def _emit_local_alloc(state, op):
    attrs = target_ir.attrs_dict(op)
    state.values[_single_result(op)] = state.builder.lds_base(
        _scalar_type(state.dsl, attrs["element_type"]),
        offset=int(attrs.get("byte_offset", 0)),
    )


def _emit_memdesc_index(state, op):
    attrs = target_ir.attrs_dict(op)
    base, index = _operand_values(state, op, 2)
    static_offset = attrs.get("static_lds_byte_offset")
    if static_offset is not None:
        result_id = _single_result(op)
        target_type = state.target_program.values[result_id].type
        state.values[result_id] = state.builder.lds_base(
            _scalar_type(state.dsl, target_type.element_type),
            offset=int(static_offset),
        )
        return
    if isinstance(index, tuple):
        fail(
            "TLXW_EMIT_UNSUPPORTED_MEMDESC_INDEX",
            STAGE,
            "ttg.memdesc_index requires a scalar slot index",
            target_op_id=op.target_op_id,
        )
    elements_per_slot = int(attrs["elements_per_slot"])
    offset = index
    if elements_per_slot != 1:
        stride = state.builder.constant(index.type, elements_per_slot)
        offset = state.builder.binary(state.dsl.BinaryKind.MulI, index, stride)
    state.values[_single_result(op)] = state.builder.ptr_add(
        base,
        offset,
        result_type=base.type,
    )


def _emit_buffer_load_to_local(state, op):
    attrs = target_ir.attrs_dict(op)
    if attrs["mode"] == "dma_packet_lds":
        scalar_count = int(attrs["source_scalar_count"])
        operands = _operand_values(state, op, 2 + scalar_count)
        dest_base, source_base = operands[:2]
        scalar_values = operands[2:]
        element_type = _scalar_type(state.dsl, attrs["element_type"])
        lane_width = int(attrs["lane_width"])
        range_bytes = state.builder.constant(state.dsl.i32(), int(attrs["range_bytes"]))
        buffer_base = state.builder.make_buffer(
            source_base,
            range_bytes,
            result_type=state.dsl.buffer_ptr_type(element_type),
        )
        state.values[_single_result(op)] = _emit_buffer_load_to_local_packet_dma(
            state,
            op,
            attrs,
            dest_base,
            buffer_base,
            scalar_values,
            element_type,
            lane_width,
        )
        return
    dest_base, source_base, offsets = _operand_values(state, op, 3)
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
    destination_offsets = tuple(int(value) for value in attrs["destination_component_offsets"])
    expected_components = int(attrs["component_count"])
    offset_components = _as_components(offsets)
    if len(offset_components) != expected_components:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local offset component count does not "
            "match target op attrs",
            target_op_id=op.target_op_id,
        )
    if len(destination_offsets) != expected_components:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local destination component offsets do "
            "not match target op attrs",
            target_op_id=op.target_op_id,
        )
    range_bytes = state.builder.constant(state.dsl.i32(), int(attrs["range_bytes"]))
    buffer_base = state.builder.make_buffer(
        source_base,
        range_bytes,
        result_type=state.dsl.buffer_ptr_type(element_type),
    )
    if attrs["mode"] == "dma_load_lds":
        state.values[_single_result(op)] = _emit_buffer_load_to_local_dma(
            state,
            op,
            attrs,
            dest_base,
            buffer_base,
            offset_components,
            destination_offsets,
            element_type,
            lane_width,
        )
        return
    if attrs["mode"] != "scalarized_load_store":
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"unsupported amdg.buffer_load_to_local mode {attrs['mode']}",
            target_op_id=op.target_op_id,
        )
    token = state.builder.token()
    lane = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    value_type = state.dsl.simd_type(element_type, lane_width)
    source_ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.buffer_address_space(),
        lane_width,
    )
    dest_ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.shared_address_space(),
        lane_width,
    )
    for offset_component, destination_base_offset in zip(
        offset_components,
        destination_offsets,
    ):
        source_ptr = state.builder.ptr_add(
            buffer_base,
            offset_component,
            result_type=source_ptr_type,
        )
        loaded, load_token = state.builder.load(source_ptr, value_type, after=token)
        dest_offset = lane
        if destination_base_offset:
            base_offset = state.builder.splat(
                state.builder.constant(state.dsl.i32(), destination_base_offset),
                state.dsl.i32(),
                lane_width,
            )
            dest_offset = state.builder.binary(
                state.dsl.BinaryKind.AddI,
                lane,
                base_offset,
            )
        dest_ptr = state.builder.ptr_add(
            dest_base,
            dest_offset,
            result_type=dest_ptr_type,
        )
        token = state.builder.store(loaded, dest_ptr, after=load_token)
    state.values[_single_result(op)] = token


def _emit_buffer_load_to_local_packet_dma(
    state,
    op,
    attrs,
    dest_base,
    buffer_base,
    scalar_values,
    element_type,
    lane_width,
):
    packet_bytes = int(attrs["packet_bytes"])
    packet_elements = int(attrs["packet_elements"])
    element_byte_width = int(attrs["element_byte_width"])
    component_count = int(attrs["component_count"])
    component_thread_count = int(attrs.get("component_thread_count", lane_width))
    shape = tuple(int(dim) for dim in attrs["source_shape"])
    destination_offsets = tuple(int(value) for value in attrs["destination_component_offsets"])
    if len(destination_offsets) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "amdg.buffer_load_to_local packet destination offsets do not "
            "match component count",
            target_op_id=op.target_op_id,
        )
    source_ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.buffer_address_space(),
        lane_width,
    )
    i32_shared = state.dsl.ptr_type(state.dsl.i32(), state.dsl.shared_address_space())
    dest_base_i32 = _ptr_cast(state, dest_base, i32_shared)
    lane = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    wave_first = None
    destination_wave_stride_dwords = int(attrs.get("destination_wave_stride_dwords", 0))
    if destination_wave_stride_dwords:
        wave_first = state.builder.read_first(lane)
        wave_first = _assume_value_range(
            state,
            wave_first,
            (0, max(0, int(component_thread_count) - 1)),
            op,
        )
    token = state.builder.token()
    for component, destination_offset in enumerate(destination_offsets):
        coords = _packet_coordinate_values(
            state,
            component,
            lane,
            component_thread_count,
            packet_elements,
            shape,
        )
        source_offset = _affine_offset_value(
            state,
            attrs["source_offset_terms"],
            coords,
            scalar_values,
            op,
        )
        source_offset = _assume_value_range(
            state,
            source_offset,
            attrs.get("source_offset_range"),
            op,
        )
        source_ptr = state.builder.ptr_add(
            buffer_base,
            source_offset,
            result_type=source_ptr_type,
        )
        dest_offset = _packet_destination_offset_value(
            state,
            int(destination_offset),
            element_byte_width,
            destination_wave_stride_dwords,
            wave_first,
            lane_width,
            op,
        )
        if dest_offset is None:
            dest_ptr = dest_base_i32
        else:
            dest_ptr = state.builder.ptr_add(
                dest_base_i32,
                dest_offset,
                result_type=i32_shared,
            )
        token = state.builder.dma_load_lds(
            source_ptr,
            dest_ptr,
            after=token,
            bytes=packet_bytes,
        )
    return token


def _emit_buffer_load_to_local_dma(
    state,
    op,
    attrs,
    dest_base,
    buffer_base,
    offset_components,
    destination_offsets,
    element_type,
    lane_width,
):
    packet_bytes = int(attrs["packet_bytes"])
    if packet_bytes <= 0:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local DMA requires a positive packet byte width",
            target_op_id=op.target_op_id,
        )
    token = state.builder.token()
    source_ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.buffer_address_space(),
        lane_width,
    )
    dest_ptr_type = state.dsl.ptr_type(element_type, state.dsl.shared_address_space())
    for offset_component, destination_base_offset in zip(
        offset_components,
        destination_offsets,
    ):
        source_ptr = state.builder.ptr_add(
            buffer_base,
            offset_component,
            result_type=source_ptr_type,
        )
        dest_ptr = dest_base
        if destination_base_offset:
            dest_ptr = state.builder.ptr_add(
                dest_base,
                state.builder.constant(state.dsl.i32(), destination_base_offset),
                result_type=dest_ptr_type,
            )
        token = state.builder.dma_load_lds(
            source_ptr,
            dest_ptr,
            after=token,
            bytes=packet_bytes,
        )
    return token


def _emit_async_commit_group(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    if tokens:
        token = state.builder.join(*tokens)
    else:
        token = state.builder.token()
    if op.results:
        state.values[_single_result(op)] = token


def _emit_async_wait(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    if tokens:
        state.builder.wait(*tokens)
        token = state.builder.barrier(*tokens)
    else:
        token = state.builder.token()
    if op.results:
        state.values[_single_result(op)] = token


def _emit_local_load_fragment(state, op):
    attrs = target_ir.attrs_dict(op)
    base = _operand_values(state, op, 1)[0]
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["registers"])
    expected_components = int(attrs["component_count"])
    warps_per_cta = tuple(int(value) for value in attrs.get("warps_per_cta", (1, 1)))
    wave_tile_axis = attrs.get("wave_tile_axis", "none")
    wave_tile_stride_dwords = int(attrs.get("wave_tile_stride_dwords", 0))
    load_mode = attrs.get("load_mode", "fragment_load")
    offset_attr = (
        "component_dword_offsets"
        if load_mode == "fragment_load"
        else "component_tile_offsets"
    )
    if len(attrs[offset_attr]) != expected_components:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "local_load_fragment component offsets do not match attrs",
            target_op_id=op.target_op_id,
    )
    i32_shared = state.dsl.ptr_type(state.dsl.i32(), state.dsl.shared_address_space())
    base_i32 = _ptr_cast(state, base, i32_shared)
    wi = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    simd_i32 = state.dsl.simd_type(state.dsl.i32(), lane_width)
    lane_i32 = _simd_binary_const(state, "remui", wi, lane_width, lane_width)
    fragment_type = state.dsl.fragment_type(
        int(attrs["role"]),
        element_type,
        int(attrs["rows"]),
        int(attrs["columns"]),
        lane_width,
        registers,
    )
    if load_mode == "b16_transpose":
        state.values[_single_result(op)] = _emit_b16_transpose_fragment_load(
            state,
            op,
            attrs,
            base,
            element_type,
            fragment_type,
            wi,
            lane_i32,
            simd_i32,
        )
        return
    if load_mode == "swizzled_fragment_load":
        state.values[_single_result(op)] = _emit_swizzled_fragment_load(
            state,
            op,
            attrs,
            base,
            fragment_type,
            wi,
            lane_i32,
            simd_i32,
        )
        return
    if load_mode != "fragment_load":
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported local_load_fragment mode {load_mode}",
            target_op_id=op.target_op_id,
        )
    component_offsets = tuple(int(value) for value in attrs["component_dword_offsets"])
    lane_stride = lane_i32
    if registers != 1:
        lane_stride = _simd_binary_const(
            state,
            "muli",
            lane_i32,
            registers,
            lane_width,
        )
    ptr_type = state.dsl.simd_ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
        lane_width,
    )
    wave_tile_offset = _wave_tile_offset_i32(
        state,
        wi,
        wave_tile_axis,
        warps_per_cta,
        wave_tile_stride_dwords,
        lane_width,
        op,
    )
    fragments = []
    for component_offset in component_offsets:
        offset = _add_optional_offset(state, lane_stride, wave_tile_offset)
        if component_offset:
            base_offset = state.builder.splat(
                state.builder.constant(state.dsl.i32(), component_offset),
                state.dsl.i32(),
                lane_width,
            )
            offset = state.builder.binary(
                state.dsl.BinaryKind.AddI,
                offset,
                base_offset,
            )
        ptr = state.builder.ptr_add(base_i32, offset, result_type=ptr_type)
        fragment, _token = state.builder.fragment_load(ptr, fragment_type)
        fragments.append(fragment)
    state.values[_single_result(op)] = _pack_components(tuple(fragments))


def _emit_b16_transpose_fragment_load(
    state,
    op,
    attrs,
    base,
    element_type,
    fragment_type,
    wi,
    lane,
    simd_i32,
):
    lane_width = int(attrs["lane_width"])
    base_type = state.dsl.ptr_type(element_type, state.dsl.shared_address_space())
    base = _ptr_cast(state, base, base_type)
    ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.shared_address_space(),
        lane_width,
    )
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(int(attrs["chunk_elements"]), element_type),
        width=lane_width,
    )
    component_type = state.dsl.simd_type(element_type, lane_width)
    wave_tile_offset = _wave_tile_offset_i32(
        state,
        wi,
        attrs.get("wave_tile_axis", "none"),
        tuple(int(value) for value in attrs.get("warps_per_cta", (1, 1))),
        int(attrs.get("wave_tile_stride_elements", 0)),
        lane_width,
        op,
    )
    lane_scaled = _simd_binary_const(
        state,
        "muli",
        lane,
        int(attrs["elements_per_lane"]),
        lane_width,
    )
    logical_base = _add_optional_offset(state, lane_scaled, wave_tile_offset)
    chunk_element_deltas = attrs.get("chunk_element_deltas")
    fragments = []
    for component_index, tile_offsets in enumerate(attrs["component_tile_offsets"]):
        token = None
        components = []
        component_base = None
        component_deltas = None
        if chunk_element_deltas is not None:
            component_deltas = tuple(
                int(value) for value in chunk_element_deltas[component_index]
            )
            component_base = _local_fragment_element_offset(
                state,
                attrs,
                logical_base,
                tuple(int(value) for value in tile_offsets),
                0,
                lane_width,
            )
        for chunk in range(int(attrs["chunks_per_component"])):
            if component_base is not None:
                offset = component_base
                delta = int(component_deltas[chunk])
                if delta:
                    offset = _simd_binary_const(
                        state,
                        "addi",
                        offset,
                        delta,
                        lane_width,
                    )
            else:
                offset = _local_fragment_element_offset(
                    state,
                    attrs,
                    logical_base,
                    tuple(int(value) for value in tile_offsets),
                    int(attrs["chunk_elements"]) * chunk,
                    lane_width,
                )
            ptr = state.builder.ptr_add(base, offset, result_type=ptr_type)
            loaded, token = state.builder.transpose_load(ptr, load_type, after=token)
            for component in range(int(attrs["chunk_elements"])):
                components.append(
                    state.dsl.wave.ExtractOp(
                        component_type,
                        loaded,
                        int(component),
                    ).result
                )
        packed_type = state.dsl.simd_type(
            state.dsl.vector_type(len(components), element_type),
            width=lane_width,
        )
        packed = state.dsl.wave.PackOp(packed_type, components).result
        fragments.append(state.builder.fragment_pack(packed, fragment_type))
    return _pack_components(tuple(fragments))


def _emit_swizzled_fragment_load(
    state,
    op,
    attrs,
    base,
    fragment_type,
    wi,
    lane,
    simd_i32,
):
    lane_width = int(attrs["lane_width"])
    i32_shared = state.dsl.ptr_type(state.dsl.i32(), state.dsl.shared_address_space())
    base_i32 = _ptr_cast(state, base, i32_shared)
    ptr_type = state.dsl.simd_ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
        lane_width,
    )
    wave_tile_offset = _wave_tile_offset_i32(
        state,
        wi,
        attrs.get("wave_tile_axis", "none"),
        tuple(int(value) for value in attrs.get("warps_per_cta", (1, 1))),
        int(attrs.get("wave_tile_stride_elements", 0)),
        lane_width,
        op,
    )
    lane_scaled = _simd_binary_const(
        state,
        "muli",
        lane,
        int(attrs["elements_per_lane"]),
        lane_width,
    )
    logical_base = _add_optional_offset(state, lane_scaled, wave_tile_offset)
    fragments = []
    for tile_offsets in attrs["component_tile_offsets"]:
        offset = _local_fragment_element_offset(
            state,
            attrs,
            logical_base,
            tuple(int(value) for value in tile_offsets),
            0,
            lane_width,
            elements_per_offset_unit=2,
        )
        ptr = state.builder.ptr_add(base_i32, offset, result_type=ptr_type)
        fragment, _token = state.builder.fragment_load(ptr, fragment_type)
        fragments.append(fragment)
    return _pack_components(tuple(fragments))


def _local_fragment_element_offset(
    state,
    attrs,
    logical_base,
    tile_offsets,
    extra_elements,
    lane_width,
    *,
    elements_per_offset_unit=1,
):
    tile_base = _dense_tile_base_elements(
        attrs.get("memdesc_shape", attrs["source_shape"]),
        tile_offsets,
    )
    logical = logical_base
    if tile_base or extra_elements:
        logical = _simd_binary_const(
            state,
            "addi",
            logical,
            int(tile_base) + int(extra_elements),
            lane_width,
        )
    layout_kind = attrs.get("shared_layout_kind", "dense")
    if layout_kind == "dense":
        encoded = logical
    elif layout_kind == "swizzled_shared":
        encoded = _swizzled_element_offset(state, attrs, logical, lane_width)
    elif layout_kind == "padded_shared":
        encoded = _padded_element_offset(state, attrs, logical, lane_width)
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported local_load shared layout {layout_kind}",
        )
    if int(elements_per_offset_unit) != 1:
        encoded = _simd_binary_const(
            state,
            "divui",
            encoded,
            int(elements_per_offset_unit),
            lane_width,
        )
    return encoded


def _wave_tile_offset_i32(
    state,
    wi,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    lane_width,
    op,
):
    if wave_tile_axis == "none" or not int(wave_tile_stride):
        return None
    if len(warps_per_cta) < 2 or int(warps_per_cta[1]) <= 0:
        fail(
            "TLXW_EMIT_LOCAL_LOAD_TILE_MAP",
            STAGE,
            "local_load_fragment requires a valid warps_per_cta mapping",
            target_op_id=op.target_op_id,
        )
    wave_id = _simd_binary_const(state, "divui", wi, int(lane_width), lane_width)
    if wave_tile_axis == "m":
        wave_coord = _simd_binary_const(
            state,
            "divui",
            wave_id,
            int(warps_per_cta[1]),
            lane_width,
        )
    elif wave_tile_axis == "n":
        wave_coord = _simd_binary_const(
            state,
            "remui",
            wave_id,
            int(warps_per_cta[1]),
            lane_width,
        )
    else:
        fail(
            "TLXW_EMIT_LOCAL_LOAD_TILE_MAP",
            STAGE,
            f"unsupported local_load_fragment wave axis {wave_tile_axis}",
            target_op_id=op.target_op_id,
        )
    return _simd_binary_const(
        state,
        "muli",
        wave_coord,
        int(wave_tile_stride),
        lane_width,
    )


def _swizzled_element_offset(state, attrs, logical, lane_width):
    cols = int(attrs.get("memdesc_shape", attrs["source_shape"])[-1])
    vec = int(attrs["swizzled_vec"])
    row = _simd_binary_const(state, "divui", logical, cols, lane_width)
    col = _simd_binary_const(state, "remui", logical, cols, lane_width)
    row_phase = _simd_binary_const(
        state,
        "divui",
        row,
        int(attrs["swizzled_per_phase"]),
        lane_width,
    )
    phase = _simd_binary_const(
        state,
        "remui",
        row_phase,
        int(attrs["swizzled_max_phase"]),
        lane_width,
    )
    col_group = _simd_binary_const(state, "divui", col, vec, lane_width)
    swizzled_group = state.builder.binary(
        state.dsl.BinaryKind.XOrI,
        col_group,
        phase,
    )
    swizzled_base = _simd_binary_const(state, "muli", swizzled_group, vec, lane_width)
    col_in_vec = _simd_binary_const(state, "remui", col, vec, lane_width)
    swizzled_col = state.builder.binary(
        state.dsl.BinaryKind.AddI,
        swizzled_base,
        col_in_vec,
    )
    row_scaled = _simd_binary_const(state, "muli", row, cols, lane_width)
    return state.builder.binary(state.dsl.BinaryKind.AddI, row_scaled, swizzled_col)


def _padded_element_offset(state, attrs, logical, lane_width):
    encoded = logical
    for interval, padding in zip(attrs.get("padded_intervals", ()), attrs.get("padded_paddings", ())):
        quotient = _simd_binary_const(state, "divui", logical, int(interval), lane_width)
        pad = _simd_binary_const(state, "muli", quotient, int(padding), lane_width)
        encoded = state.builder.binary(state.dsl.BinaryKind.AddI, encoded, pad)
    return encoded


def _simd_binary_const(state, operation, value, constant, lane_width):
    constant = int(constant)
    if operation == "divui" and constant == 1:
        return value
    if operation == "remui" and constant == 1:
        return state.builder.splat(
            state.builder.constant(state.dsl.i32(), 0),
            state.dsl.i32(),
            int(lane_width),
        )
    if operation == "divui" and _is_power_of_two(constant):
        operation_kind = state.dsl.BinaryKind.ShRUI
        constant = constant.bit_length() - 1
    elif operation == "remui" and _is_power_of_two(constant):
        operation_kind = state.dsl.BinaryKind.AndI
        constant -= 1
    else:
        operation_kind = _binary_kind(state.dsl, operation)
    rhs = state.builder.splat(
        state.builder.constant(state.dsl.i32(), constant),
        state.dsl.i32(),
        int(lane_width),
    )
    return state.builder.binary(operation_kind, value, rhs)


def _is_power_of_two(value):
    value = int(value)
    return value > 0 and (value & (value - 1)) == 0


def _dense_tile_base_elements(shape, tile_offsets):
    element_offset = 0
    stride = 1
    for dim in reversed(range(len(shape))):
        element_offset += int(tile_offsets[dim]) * stride
        stride *= int(shape[dim])
    return int(element_offset)


def _add_optional_offset(state, base, offset):
    if offset is None:
        return base
    return state.builder.binary(state.dsl.BinaryKind.AddI, base, offset)


def _emit_fragment_fill(state, op):
    attrs = target_ir.attrs_dict(op)
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["registers"])
    fragment_type = state.dsl.fragment_type(
        int(attrs["role"]),
        _scalar_type(state.dsl, attrs["element_type"]),
        int(attrs["rows"]),
        int(attrs["columns"]),
        lane_width,
        registers,
    )
    fill = state.builder.constant(state.dsl.i32(), int(attrs["fill_value"]))
    state.values[_single_result(op)] = _pack_components(
        tuple(
            state.builder.fragment_fill(fill, fragment_type)
            for _ in range(int(attrs["component_count"]))
        )
    )


def _emit_mma(state, op):
    attrs = target_ir.attrs_dict(op)
    lhs, rhs, acc = _operand_values(state, op, 3)
    lhs_components = _as_components(lhs)
    rhs_components = _as_components(rhs)
    acc_components = _as_components(acc)
    m_tiles = int(attrs["m_tiles"])
    n_tiles = int(attrs["n_tiles"])
    k_tiles = int(attrs.get("k_tiles", 1))
    if len(lhs_components) != m_tiles * k_tiles or len(rhs_components) != n_tiles * k_tiles:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "mma operand component counts do not match tile attrs",
            target_op_id=op.target_op_id,
        )
    if len(acc_components) != m_tiles * n_tiles:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "mma accumulator component count does not match tile attrs",
            target_op_id=op.target_op_id,
        )
    results = []
    for m_tile in range(m_tiles):
        for n_tile in range(n_tiles):
            index = m_tile * n_tiles + n_tile
            acc_value = acc_components[index]
            for k_tile in range(k_tiles):
                acc_value = state.builder.mma(
                    attrs["kind"],
                    lhs_components[m_tile * k_tiles + k_tile],
                    rhs_components[n_tile * k_tiles + k_tile],
                    acc_value,
                )
            results.append(acc_value)
    state.values[_single_result(op)] = _pack_components(tuple(results))


def _emit_fragment_truncf(state, op):
    attrs = target_ir.attrs_dict(op)
    (fragment_value,) = _operand_values(state, op, 1)
    fragments = _as_components(fragment_value)
    component_count = int(attrs["component_count"])
    if len(fragments) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "fragment_truncf component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    lane_width = int(attrs["lane_width"])
    registers = int(attrs["registers"])
    f32_regs = state.dsl.simd_type(
        state.dsl.vector_type(registers, state.dsl.f32()),
        width=lane_width,
    )
    f16_regs = state.dsl.simd_type(
        state.dsl.vector_type(registers, state.dsl.f16()),
        width=lane_width,
    )
    packed = []
    for fragment in fragments:
        regs = state.dsl.waveamd.FragmentUnpackOp(f32_regs, fragment).result
        packed.append(state.builder.fpconvert(regs, f16_regs))
    state.values[_single_result(op)] = _pack_components(tuple(packed))


def _emit_layout_convert(state, op):
    attrs = target_ir.attrs_dict(op)
    (value,) = _operand_values(state, op, 1)
    components = _as_components(value)
    mode = attrs["mode"]
    if mode == "alias":
        state.values[_single_result(op)] = value
        return
    if mode == "component_group_first":
        group_size = int(attrs["group_size"])
        result_count = int(attrs["result_component_count"])
        if len(components) != group_size * result_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "layout_convert component count does not match group attrs",
                target_op_id=op.target_op_id,
            )
        state.values[_single_result(op)] = _pack_components(
            tuple(components[index * group_size] for index in range(result_count))
        )
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_LAYOUT_CONVERT",
        STAGE,
        f"unsupported layout_convert mode {mode}",
        target_op_id=op.target_op_id,
    )


def _emit_buffer_store(state, op):
    attrs = target_ir.attrs_dict(op)
    operand_count = 4 if attrs["has_mask"] else 3
    operands = _operand_values(state, op, operand_count)
    value, source_base, offsets = operands[:3]
    masks = operands[3] if attrs["has_mask"] else None
    value_components = _as_components(value)
    offset_components = _as_components(offsets)
    mask_components = None if masks is None else _as_components(masks)
    component_count = int(attrs["component_count"])
    if len(value_components) != component_count or len(offset_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store value/offset component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if mask_components is not None and len(mask_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_store mask component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
    mask_mode = attrs.get("mask_mode", "exec_where" if attrs["has_mask"] else "none")
    range_bytes = state.builder.constant(state.dsl.i32(), int(attrs["range_bytes"]))
    buffer_base = state.builder.make_buffer(
        source_base,
        range_bytes,
        result_type=state.dsl.buffer_ptr_type(element_type),
    )
    ptr_type = state.dsl.simd_ptr_type(
        element_type,
        state.dsl.buffer_address_space(),
        lane_width,
    )
    for index, (value_component, offset_component) in enumerate(
        zip(value_components, offset_components)
    ):
        offset_component = _assume_value_range(
            state,
            offset_component,
            attrs.get("offset_range"),
            op,
        )
        ptr = state.builder.ptr_add(
            buffer_base,
            offset_component,
            result_type=ptr_type,
        )
        if mask_components is not None and mask_mode == "select_oob_offset":
            inactive_offset = state.builder.splat(
                state.builder.constant(state.dsl.i32(), int(attrs["inactive_offset"])),
                state.dsl.i32(),
                lane_width,
            )
            inactive_ptr = state.builder.ptr_add(
                buffer_base,
                inactive_offset,
                result_type=ptr_type,
            )
            ptr = state.builder.select(mask_components[index], ptr, inactive_ptr)
        if mask_components is None or mask_mode == "select_oob_offset":
            state.builder.store(value_component, ptr)
            continue
        if mask_mode != "exec_where":
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_STORE_MASK",
                STAGE,
                f"unsupported buffer_store mask mode {mask_mode}",
                target_op_id=op.target_op_id,
            )
        with state.builder.where(mask_components[index]):
            state.builder.store(value_component, ptr)


def _scalar_constant(state, scalar_type, element_type, literal, op):
    if element_type == "i1":
        return state.dsl.arith.ConstantOp(
            scalar_type,
            state.ir.IntegerAttr.get(scalar_type, int(_literal_bool(literal, op))),
        ).result
    _require_numeric_literal(literal, op)
    return state.builder.constant(scalar_type, literal)


def _wave_constant(state, result_type, scalar_type, element_type, literal, op):
    if element_type == "i1":
        attr = state.ir.IntegerAttr.get(scalar_type, int(_literal_bool(literal, op)))
    elif _is_float_element(element_type):
        _require_numeric_literal(literal, op)
        attr = state.ir.FloatAttr.get(scalar_type, float(literal))
    else:
        _require_numeric_literal(literal, op)
        attr = state.ir.IntegerAttr.get(scalar_type, int(literal))
    return state.dsl.wave.ConstantOp(result_type, attr).result


def _wave_mask_constant(state, result_type, value):
    return state.dsl.wave.ConstantOp(
        result_type,
        state.ir.Attribute.parse("true" if value else "false"),
    ).result


def _reuse_component_result(reused, operands, create):
    operands = tuple(operands)
    for existing_operands, value in reused:
        if len(existing_operands) == len(operands) and all(
            existing is operand
            for existing, operand in zip(existing_operands, operands)
        ):
            return value
    value = create()
    reused.append((operands, value))
    return value


def _literal_bool(literal, op):
    if isinstance(literal, bool):
        return literal
    if isinstance(literal, int) and literal in (0, 1):
        return bool(literal)
    if isinstance(literal, str) and literal.lower() in {"true", "false"}:
        return literal.lower() == "true"
    fail(
        "TLXW_EMIT_UNSUPPORTED_CONSTANT",
        STAGE,
        f"cannot emit {literal!r} as an i1/mask constant",
        target_op_id=op.target_op_id,
    )


def _require_numeric_literal(literal, op):
    if isinstance(literal, bool) or not isinstance(literal, (int, float)):
        fail(
            "TLXW_EMIT_UNSUPPORTED_CONSTANT",
            STAGE,
            f"cannot emit non-numeric constant literal {literal!r}",
            target_op_id=op.target_op_id,
        )


def _is_float_element(element_type):
    return element_type in {"f16", "bf16", "f32", "f64"}


def _packet_coordinate_values(
    state,
    component,
    lane,
    component_thread_count,
    packet_elements,
    shape,
):
    linear = _simd_binary_const(
        state,
        "muli",
        lane,
        int(packet_elements),
        int(state.dsl.SimdType(lane.type).width),
    )
    constant = int(component) * int(component_thread_count) * int(packet_elements)
    if constant:
        linear = _simd_binary_const(
            state,
            "addi",
            linear,
            constant,
            int(state.dsl.SimdType(lane.type).width),
        )
    lane_width = int(state.dsl.SimdType(lane.type).width)
    coords = []
    for dim, extent in enumerate(shape):
        stride = _product(shape[dim + 1 :])
        coord = _simd_binary_const(state, "divui", linear, int(stride), lane_width)
        if int(extent) != 1:
            coord = _simd_binary_const(state, "remui", coord, int(extent), lane_width)
        coords.append(coord)
    return tuple(coords)


def _packet_destination_offset_value(
    state,
    destination_offset,
    element_byte_width,
    destination_wave_stride_dwords,
    wave_first,
    lane_width,
    op,
):
    byte_offset = int(destination_offset) * int(element_byte_width)
    if byte_offset % 4:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "packet DMA destination offset must be dword aligned",
            target_op_id=op.target_op_id,
        )
    base_dwords = byte_offset // 4
    if not destination_wave_stride_dwords:
        if base_dwords == 0:
            return None
        return state.builder.constant(state.dsl.i32(), int(base_dwords))
    if wave_first is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "packet DMA destination wave offset is missing its uniform binding",
            target_op_id=op.target_op_id,
        )
    wave_id = _scalar_binary_const_i32(
        state,
        "divui",
        wave_first,
        int(lane_width),
    )
    offset = _scalar_binary_const_i32(
        state,
        "muli",
        wave_id,
        int(destination_wave_stride_dwords),
    )
    if base_dwords:
        offset = _scalar_binary_const_i32(state, "addi", offset, int(base_dwords))
    return offset


def _affine_offset_value(state, encoded_terms, coords, scalar_values, op):
    lane_width = int(state.dsl.SimdType(coords[0].type).width) if coords else 64
    scalar_components = tuple(
        _splat_i32_scalar(state, value, lane_width, op) for value in scalar_values
    )
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        lane_width,
    )
    for encoded in encoded_terms:
        term = _affine_term_i32(
            state,
            encoded,
            coords,
            scalar_components,
            lane_width,
            op,
        )
        result = state.builder.binary(state.dsl.BinaryKind.AddI, result, term)
    return result


def _affine_term_i32(state, encoded, coords, scalar_components, lane_width, op):
    kind, coefficient, dim, slots = encoded
    coefficient = int(coefficient)
    dim = int(dim)
    slots = tuple(int(slot) for slot in slots)
    if kind == "const":
        return state.builder.splat(
            state.builder.constant(state.dsl.i32(), coefficient),
            state.dsl.i32(),
            lane_width,
        )
    if kind == "dim":
        return _scale_simd_i32(
            state,
            coords[_require_dim_slot(dim, coords, op)],
            coefficient,
            lane_width,
        )
    if kind == "scalar":
        return _scale_simd_i32(
            state,
            scalar_components[_require_scalar_slot(slots, scalar_components, op)],
            coefficient,
            lane_width,
        )
    if kind == "dim_scalar":
        dim_value = coords[_require_dim_slot(dim, coords, op)]
        scalar_value = scalar_components[_require_scalar_slot(slots, scalar_components, op)]
        product = state.builder.binary(state.dsl.BinaryKind.MulI, dim_value, scalar_value)
        return _scale_simd_i32(state, product, coefficient, lane_width)
    if kind == "scalar_product":
        if len(slots) != 2:
            fail(
                "TLXW_EMIT_BAD_AFFINE_TERM",
                STAGE,
                "scalar_product affine term requires two scalar operands",
                target_op_id=op.target_op_id,
            )
        lhs = scalar_components[_require_scalar_slot((slots[0],), scalar_components, op)]
        rhs = scalar_components[_require_scalar_slot((slots[1],), scalar_components, op)]
        product = state.builder.binary(state.dsl.BinaryKind.MulI, lhs, rhs)
        return _scale_simd_i32(state, product, coefficient, lane_width)
    fail(
        "TLXW_EMIT_BAD_AFFINE_TERM",
        STAGE,
        f"unsupported affine term kind {kind}",
        target_op_id=op.target_op_id,
    )


def _scale_simd_i32(state, value, coefficient, lane_width):
    coefficient = int(coefficient)
    if coefficient == 1:
        return value
    return _simd_binary_const(
        state,
        "muli",
        value,
        coefficient,
        lane_width,
    )


def _splat_i32_scalar(state, value, lane_width, op):
    if str(value.type) != str(state.dsl.i32()):
        fail(
            "TLXW_EMIT_BAD_AFFINE_TERM",
            STAGE,
            f"affine scalar operand must be i32, got {value.type}",
            target_op_id=op.target_op_id,
        )
    return state.builder.splat(value, state.dsl.i32(), lane_width)


def _scalar_binary_const_i32(state, operation, value, constant):
    constant = int(constant)
    if operation == "divui" and constant == 1:
        return value
    if operation == "remui" and constant == 1:
        return state.builder.constant(state.dsl.i32(), 0)
    if operation == "divui" and _is_power_of_two(constant):
        operation_kind = state.dsl.BinaryKind.ShRUI
        constant = constant.bit_length() - 1
    elif operation == "remui" and _is_power_of_two(constant):
        operation_kind = state.dsl.BinaryKind.AndI
        constant -= 1
    else:
        operation_kind = _binary_kind(state.dsl, operation)
    rhs = state.builder.constant(state.dsl.i32(), constant)
    return state.builder.binary(operation_kind, value, rhs)


def _require_dim_slot(dim, coords, op):
    if dim < 0 or dim >= len(coords):
        fail(
            "TLXW_EMIT_BAD_AFFINE_TERM",
            STAGE,
            f"affine term references dimension {dim}",
            target_op_id=op.target_op_id,
        )
    return dim


def _require_scalar_slot(slots, scalar_symbols, op):
    if len(slots) != 1 or slots[0] < 0 or slots[0] >= len(scalar_symbols):
        fail(
            "TLXW_EMIT_BAD_AFFINE_TERM",
            STAGE,
            f"affine term references scalar slots {slots}",
            target_op_id=op.target_op_id,
        )
    return slots[0]


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _target_lds_size(target_program):
    size = 0
    for op in target_program.ops:
        if op.kind != "local_alloc":
            continue
        attrs = target_ir.attrs_dict(op)
        end = int(attrs.get("byte_offset", 0)) + int(attrs.get("allocation_bytes", 0))
        size = max(size, end)
    return _align_to(size, 16)


def _align_to(value, alignment):
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


def _assume_value_range(state, value, encoded_range, op):
    if encoded_range is None:
        return value
    if len(encoded_range) != 2:
        fail(
            "TLXW_EMIT_BAD_ASSUME_RANGE",
            STAGE,
            "encoded range must contain lower and upper bounds",
            target_op_id=op.target_op_id,
        )
    lower, upper = encoded_range
    x = state.dsl.sym("x")
    assumptions = []
    if lower is not None:
        assumptions.append(x >= int(lower))
    if upper is not None:
        assumptions.append(x <= int(upper))
    if not assumptions:
        return value
    return state.builder.assume(value, tuple(assumptions), name="x")


def _ptr_cast(state, value, result_type):
    if str(value.type) == str(result_type):
        return value
    return state.dsl.wave.PtrCastOp(result_type, value).result


def _operand_values(state, op, count):
    if len(op.operands) != count:
        fail(
            "TLXW_EMIT_OPERAND_COUNT",
            STAGE,
            f"target op {op.kind} expected {count} operands, got {len(op.operands)}",
            target_op_id=op.target_op_id,
        )
    return tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)


def _require_value(state, target_value_id, op):
    if target_value_id not in state.values:
        fail(
            "TLXW_EMIT_UNBOUND_VALUE",
            STAGE,
            f"target value {target_value_id} is not bound before {op.kind}",
            target_op_id=op.target_op_id,
            target_value_id=target_value_id,
        )
    return state.values[target_value_id]


def _single_result(op):
    if len(op.results) != 1:
        fail(
            "TLXW_EMIT_RESULT_COUNT",
            STAGE,
            f"target op {op.kind} expected one result, got {len(op.results)}",
            target_op_id=op.target_op_id,
        )
    return op.results[0]


def _component_count(state, target_value_id):
    return int(state.target_program.values[target_value_id].type.component_count)


def _as_components(value):
    return value if isinstance(value, tuple) else (value,)


def _pack_components(components):
    return components[0] if len(components) == 1 else tuple(components)


def _broadcast_components(values, count, op):
    return tuple(_broadcast_component(value, count, op) for value in values)


def _broadcast_component(value, count, op):
    components = _as_components(value)
    if len(components) == count:
        return components
    if len(components) == 1:
        return components * count
    fail(
        "TLXW_EMIT_COMPONENT_COUNT",
        STAGE,
        f"target op {op.kind} cannot broadcast {len(components)} components "
        f"to {count}",
        target_op_id=op.target_op_id,
    )


def _range_assumptions(dsl, fact):
    if fact.kind != "range":
        return ()
    x = dsl.sym("x")
    assumptions = []
    if fact.lower is not None:
        assumptions.append(x >= int(fact.lower))
    if fact.upper is not None:
        assumptions.append(x <= int(fact.upper))
    return tuple(assumptions)


def _wave_type(dsl, target_type):
    if target_type.representation == "scalar":
        return _scalar_type(dsl, target_type.element_type)
    if target_type.representation == "uniform_pointer":
        return dsl.ptr_type(_scalar_type(dsl, target_type.element_type))
    if target_type.representation in {"simd", "simd_tuple"}:
        return dsl.simd_type(
            _scalar_type(dsl, target_type.element_type),
            int(target_type.lane_width or 64),
        )
    if target_type.representation in {"mask", "mask_tuple"}:
        return dsl.mask_type(int(target_type.lane_width or 64))
    if target_type.representation in {"per_lane_pointer", "pointer_tuple"}:
        return dsl.simd_ptr_type(
            _scalar_type(dsl, target_type.element_type),
            dsl.global_address_space(),
            int(target_type.lane_width or 64),
        )
    fail(
        "TLXW_EMIT_UNSUPPORTED_TYPE",
        STAGE,
        f"cannot emit target type {target_type}",
    )


def _splat_element_type(dsl, target_type):
    if target_type.representation in {"per_lane_pointer", "pointer_tuple"}:
        return dsl.ptr_type(_scalar_type(dsl, target_type.element_type))
    return _scalar_type(dsl, target_type.element_type)


def _scalar_type(dsl, element_type):
    return {
        "i1": dsl.i1,
        "i8": dsl.i8,
        "i16": lambda: dsl.IntegerType.get_signless(16),
        "i32": dsl.i32,
        "i64": dsl.i64,
        "index": dsl.index_type,
        "f16": dsl.f16,
        "bf16": dsl.bf16,
        "f32": dsl.f32,
    }[element_type]()


def _binary_kind(dsl, operation):
    return {
        "addi": dsl.BinaryKind.AddI,
        "subi": dsl.BinaryKind.SubI,
        "muli": dsl.BinaryKind.MulI,
        "andi": dsl.BinaryKind.AndI,
        "ori": dsl.BinaryKind.OrI,
        "xori": dsl.BinaryKind.XOrI,
        "divui": dsl.BinaryKind.DivUI,
        "divsi": dsl.BinaryKind.DivSI,
        "remui": dsl.BinaryKind.RemUI,
        "remsi": dsl.BinaryKind.RemSI,
    }[operation]


def _is_simd_value(dsl, value):
    try:
        dsl.SimdType(value.type)
    except ValueError:
        return False
    return True


def _cmpi(state, predicate_name, lhs, rhs):
    predicate = state.dsl.CmpIPredicate[predicate_name]
    if _is_simd_value(state.dsl, lhs):
        return state.builder.cmpi(predicate, lhs, rhs)
    return state.dsl.arith.CmpIOp(predicate, lhs, rhs).result


def _set_module_attrs(module_builder, dsl, ir, kernel):
    attrs = module_builder.module.operation.attributes
    attrs["tlx_wave.new_converter"] = ir.Attribute.parse("true")
    attrs["tlx_wave.num_ctas"] = ir.IntegerAttr.get(dsl.i32(), int(kernel.num_ctas or 1))
    attrs["tlx_wave.num_warps"] = ir.IntegerAttr.get(dsl.i32(), int(kernel.num_warps or 1))
    attrs["tlx_wave.threads_per_warp"] = ir.IntegerAttr.get(
        dsl.i32(),
        int(kernel.threads_per_warp or 64),
    )
    if kernel.target:
        attrs["tlx_wave.source_target"] = ir.StringAttr.get(kernel.target)
        attrs["waveamdmachine.target"] = ir.StringAttr.get(
            kernel.target.replace("hip:", "amdgcn-amd-amdhsa--")
        )


def _function_attrs(dsl, ir, kernel):
    return {
        "tlx_wave.converter.stage": ir.StringAttr.get("structural-emission"),
        "tlx_wave.num_warps": ir.IntegerAttr.get(dsl.i32(), int(kernel.num_warps or 1)),
        "tlx_wave.wave_size": ir.IntegerAttr.get(
            dsl.i32(),
            int(kernel.threads_per_warp or 64),
        ),
        "tlx_wave.ttgir.noinline": ir.Attribute.parse(
            "true" if kernel.noinline else "false"
        ),
    }


def _load_wave_dsl():
    third_party = Path(__file__).resolve().parents[3]
    wave_python = (
        third_party
        / "wave"
        / "build"
        / "wave-build"
        / "python_packages"
        / "wave_mlir"
    )
    if not wave_python.exists():
        fail(
            "TLXW_EMIT_BINDINGS_UNAVAILABLE",
            STAGE,
            f"Wave MLIR Python package is missing at {wave_python}",
        )
    path = str(wave_python)
    if path not in sys.path:
        sys.path.insert(0, path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attribute builder for .* is already registered",
            category=RuntimeWarning,
        )
        try:
            from mlir import ir
            from mlir.dialects import wave_dsl as dsl
        except Exception as exc:
            fail(
                "TLXW_EMIT_BINDINGS_UNAVAILABLE",
                STAGE,
                f"cannot import Wave MLIR Python bindings: {type(exc).__name__}: {exc}",
            )
    return dsl, ir
