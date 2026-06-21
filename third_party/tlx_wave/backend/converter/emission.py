"""Structural Wave emission for verified target programs."""

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

from .diagnostics import fail
from . import domains
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
    scratch_token: object | None = None


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
                workgroup_size=_kernel_workgroup_size(kernel),
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
                _emit_region(state, 0)
        return EmittedWaveModule(str(module_builder), lds_size)


def _emit_region(state, region_id):
    try:
        region = state.target_program.regions[region_id]
    except IndexError:
        fail(
            "TLXW_EMIT_UNKNOWN_REGION",
            STAGE,
            f"unknown target region {region_id}",
        )
    for target_op_id in region.op_ids:
        try:
            op = state.target_program.ops[target_op_id]
        except IndexError:
            fail(
                "TLXW_EMIT_UNKNOWN_REGION_OP",
                STAGE,
                f"target region {region_id} references missing op {target_op_id}",
                target_op_id=target_op_id,
            )
        _emit_target_op(state, op)
    return tuple(
        _require_value(state, target_value_id, None)
        for target_value_id in region.yield_value_ids
    )


def _emit_target_op(state, op):
    emitter = _TARGET_EMITTERS.get(op.kind)
    if emitter is not None:
        emitter(state, op)
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_TARGET_OP",
        STAGE,
        f"no structural emission for target op {op.kind}",
        target_op_id=op.target_op_id,
    )


def _emit_return(state, op):
    del state
    if op.operands:
        fail(
            "TLXW_EMIT_RETURN_VALUES",
            STAGE,
            "empty-return emission is supported first; return values are not",
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


def _emit_float_binary(state, op):
    attrs = target_ir.attrs_dict(op)
    operation = attrs["operation"]
    builders = {
        "addf": state.builder.fadd,
        "subf": state.builder.fsub,
        "mulf": state.builder.fmul,
    }
    emit = builders.get(operation)
    if emit is None:
        fail(
            "TLXW_EMIT_UNSUPPORTED_FLOAT_BINARY",
            STAGE,
            f"unsupported float binary operation {operation}",
            target_op_id=op.target_op_id,
        )
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
                lambda lhs_component=lhs_component, rhs_component=rhs_component: emit(
                    lhs_component,
                    rhs_component,
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
    if len(op.fact_target_ids) != len(op.fact_ids):
        fail(
            "TLXW_EMIT_FACT_TARGET_COUNT",
            STAGE,
            "fact materialization requires one target value per fact",
            target_op_id=op.target_op_id,
        )
    if state.fact_program is None:
        fail(
            "TLXW_EMIT_MISSING_FACT_PROGRAM",
            STAGE,
            "fact materialization requires verified fact records",
            target_op_id=op.target_op_id,
        )
    facts = {fact.fact_id: fact for fact in state.fact_program.facts}
    for fact_id, target_value_id in zip(op.fact_ids, op.fact_target_ids):
        fact = facts.get(fact_id)
        if fact is None:
            fail(
                "TLXW_EMIT_UNKNOWN_FACT",
                STAGE,
                f"assume references missing fact {fact_id}",
                target_op_id=op.target_op_id,
                fact_id=fact_id,
            )
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
    workitem = state.builder.workitem_id(0, element_type, width)
    start = int(attrs["start"])
    components = []
    if attrs.get("coordinate_mode") == "affine_workitem":
        component_bases = tuple(int(value) for value in attrs["component_bases"])
        stride = int(attrs["workitem_stride"])
        if len(component_bases) != _component_count(state, result_id):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "make_range component bases do not match result component count",
                target_op_id=op.target_op_id,
            )
        for component_base in component_bases:
            value = workitem
            if stride != 1:
                value = _simd_binary_const(state, "muli", value, stride, width)
            value = _add_simd_const(
                state,
                value,
                start + int(component_base),
                element_type,
                width,
            )
            components.append(value)
        state.values[result_id] = _pack_components(tuple(components))
        return
    if attrs.get("coordinate_mode") == "layout_coordinates":
        shape = tuple(int(value) for value in attrs["coordinate_shape"])
        component_bases = tuple(
            tuple(int(value) for value in bases)
            for bases in attrs["component_coordinate_bases"]
        )
        workitem_coefficients = tuple(
            tuple(int(value) for value in coefficients)
            for coefficients in attrs["workitem_coordinate_coefficients"]
        )
        if len(component_bases) != _component_count(state, result_id):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "make_range coordinate bases do not match result component count",
                target_op_id=op.target_op_id,
            )
        if any(len(bases) != len(shape) for bases in component_bases):
            fail(
                "TLXW_EMIT_BAD_COORDINATES",
                STAGE,
                "make_range component coordinate rank does not match shape",
                target_op_id=op.target_op_id,
            )
        if any(len(coefficients) != len(shape) for coefficients in workitem_coefficients):
            fail(
                "TLXW_EMIT_BAD_COORDINATES",
                STAGE,
                "make_range workitem coordinate rank does not match shape",
                target_op_id=op.target_op_id,
            )
        for component_base in component_bases:
            coords = tuple(
                _bit_linear_thread_coordinate(
                    state,
                    workitem,
                    int(base),
                    tuple(coefficients[dim] for coefficients in workitem_coefficients),
                    width,
                )
                for dim, base in enumerate(component_base)
            )
            value = _linearize_coordinates(state, coords, shape, width)
            value = _add_simd_const(state, value, start, element_type, width)
            components.append(value)
        state.values[result_id] = _pack_components(tuple(components))
        return
    if attrs.get("coordinate_mode") not in (None, "flat"):
        fail(
            "TLXW_EMIT_UNSUPPORTED_MAKE_RANGE",
            STAGE,
            f"unsupported make_range coordinate mode {attrs['coordinate_mode']}",
            target_op_id=op.target_op_id,
        )
    for component in range(_component_count(state, result_id)):
        component_start = start + component * width
        value = _add_simd_const(
            state,
            workitem,
            component_start,
            element_type,
            width,
        )
        components.append(value)
    state.values[result_id] = _pack_components(tuple(components))


def _linearize_coordinates(state, coords, shape, lane_width):
    if len(coords) != len(shape):
        fail(
            "TLXW_EMIT_BAD_COORDINATES",
            STAGE,
            "coordinate count does not match shape rank",
        )
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), 0),
        state.dsl.i32(),
        int(lane_width),
    )
    for dim, coord in enumerate(coords):
        stride = _product(shape[dim + 1 :])
        term = coord
        if int(stride) != 1:
            term = _simd_binary_const(state, "muli", term, int(stride), lane_width)
        result = state.builder.binary(state.dsl.BinaryKind.AddI, result, term)
    return result


def _bit_linear_thread_coordinate(state, workitem, base, coefficients, lane_width):
    lane_width = int(lane_width)
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), int(base)),
        state.dsl.i32(),
        lane_width,
    )
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = _simd_binary_const(state, "divui", workitem, 1 << bit, lane_width)
        bit_value = _simd_binary_const(state, "remui", bit_value, 2, lane_width)
        if coefficient != 1:
            bit_value = _simd_binary_const(
                state,
                "muli",
                bit_value,
                coefficient,
                lane_width,
            )
        result = state.builder.binary(state.dsl.BinaryKind.XOrI, result, bit_value)
    return result


def _add_simd_const(state, value, constant, element_type, width):
    if not int(constant):
        return value
    start_value = state.builder.splat(
        state.builder.constant(element_type, int(constant)),
        element_type,
        int(width),
    )
    return state.builder.binary(state.dsl.BinaryKind.AddI, value, start_value)


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


def _emit_for_loop(state, op):
    attrs = target_ir.attrs_dict(op)
    if len(op.region_ids) != 1:
        fail(
            "TLXW_EMIT_FOR_REGION_COUNT",
            STAGE,
            "for_loop target op requires exactly one region",
            target_op_id=op.target_op_id,
        )
    init_arg_count = int(attrs["init_arg_count"])
    if len(op.operands) != 3 + init_arg_count:
        fail(
            "TLXW_EMIT_FOR_OPERAND_COUNT",
            STAGE,
            "for_loop operand count must be lower, upper, step, and init args",
            target_op_id=op.target_op_id,
        )
    lower, upper, step = tuple(
        _require_value(state, target_value_id, op)
        for target_value_id in op.operands[:3]
    )
    init_target_ids = op.operands[3:]
    init_values = tuple(_require_value(state, target_value_id, op) for target_value_id in init_target_ids)
    flat_init_values, init_component_counts = _flatten_packed_values(init_values)
    region = state.target_program.regions[op.region_ids[0]]
    if len(region.block_arg_ids) != 1 + init_arg_count:
        fail(
            "TLXW_EMIT_FOR_BLOCK_ARGS",
            STAGE,
            "for_loop region block args must match induction plus init args",
            target_op_id=op.target_op_id,
        )
    if not flat_init_values and op.results:
        fail(
            "TLXW_EMIT_FOR_RESULT_COUNT",
            STAGE,
            "result-bearing for_loop requires init args",
            target_op_id=op.target_op_id,
        )

    outer_values = dict(state.values)
    outer_scratch_token = state.scratch_token
    with state.builder.for_loop(
        lower,
        upper,
        step,
        init_args=flat_init_values,
        nonzero_trip=bool(attrs.get("nonzero_trip", False)),
    ) as loop:
        if flat_init_values:
            induction_value = loop.induction_variable
            flat_iter_values = tuple(loop.inner_iter_args)
        else:
            induction_value = loop
            flat_iter_values = ()
        _bind_loop_region_args(
            state,
            region.block_arg_ids,
            induction_value,
            flat_iter_values,
            init_component_counts,
            op,
        )
        yielded_values = _emit_region(state, op.region_ids[0])
        flat_yield_values, yield_component_counts = _flatten_packed_values(yielded_values)
        if tuple(yield_component_counts) != tuple(init_component_counts):
            fail(
                "TLXW_EMIT_FOR_YIELD_COMPONENTS",
                STAGE,
                "for_loop yielded component shape must match init args",
                target_op_id=op.target_op_id,
            )
        if flat_init_values:
            state.builder.yield_(flat_yield_values)
        elif flat_yield_values:
            fail(
                "TLXW_EMIT_FOR_UNEXPECTED_YIELD",
                STAGE,
                "for_loop without init args must not yield values",
                target_op_id=op.target_op_id,
            )
    state.values = outer_values
    state.scratch_token = outer_scratch_token

    if len(op.results) != init_arg_count:
        fail(
            "TLXW_EMIT_FOR_RESULT_COUNT",
            STAGE,
            "for_loop result count must match init args",
            target_op_id=op.target_op_id,
        )
    if op.results:
        flat_results = tuple(loop.results)
        if len(flat_results) != len(flat_init_values):
            fail(
                "TLXW_EMIT_FOR_RESULT_COMPONENTS",
                STAGE,
                "for_loop result component count must match init args",
                target_op_id=op.target_op_id,
            )
        cursor = 0
        for result_id, component_count in zip(op.results, init_component_counts):
            state.values[result_id] = _pack_components(
                flat_results[cursor : cursor + component_count]
            )
            cursor += component_count


def _flatten_packed_values(values):
    flat_values = []
    component_counts = []
    for value in values:
        components = _as_components(value)
        component_counts.append(len(components))
        flat_values.extend(components)
    return tuple(flat_values), tuple(component_counts)


def _bind_loop_region_args(
    state,
    block_arg_ids,
    induction_value,
    flat_iter_values,
    init_component_counts,
    op,
):
    state.values[block_arg_ids[0]] = induction_value
    cursor = 0
    for block_arg_id, component_count in zip(block_arg_ids[1:], init_component_counts):
        state.values[block_arg_id] = _pack_components(
            flat_iter_values[cursor : cursor + component_count]
        )
        cursor += component_count
    if cursor != len(flat_iter_values):
        fail(
            "TLXW_EMIT_FOR_BLOCK_COMPONENTS",
            STAGE,
            "for_loop iter block arg component count does not match init args",
            target_op_id=op.target_op_id,
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
    issue_dependency_count = int(attrs.get("issue_dependency_count", 0))
    if attrs["mode"] == "dma_packet_lds":
        scalar_count = int(attrs["source_scalar_count"])
        operands = _operand_values(state, op, 2 + scalar_count + issue_dependency_count)
        dest_base, source_base = operands[:2]
        scalar_values = operands[2 : 2 + scalar_count]
        issue_dependencies = operands[2 + scalar_count :]
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
            issue_dependencies,
            element_type,
            lane_width,
        )
        return
    has_mask = bool(attrs.get("has_mask", False))
    operands = _operand_values(
        state,
        op,
        3 + int(has_mask) + issue_dependency_count,
    )
    dest_base, source_base, offsets = operands[:3]
    operand_index = 3
    masks = None
    if has_mask:
        masks = operands[operand_index]
        operand_index += 1
    issue_dependencies = operands[operand_index:]
    element_type = _scalar_type(state.dsl, attrs["element_type"])
    lane_width = int(attrs["lane_width"])
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
    destination_offset_mode = attrs.get("destination_offset_mode", "affine")
    if destination_offset_mode == "affine":
        destination_offsets = tuple(
            int(value) for value in attrs["destination_component_offsets"]
        )
        if len(destination_offsets) != expected_components:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "amdg.buffer_load_to_local destination component offsets do "
                "not match target op attrs",
                target_op_id=op.target_op_id,
            )
    elif destination_offset_mode == "layout_coordinates":
        destination_shape = tuple(int(value) for value in attrs["destination_coordinate_shape"])
        destination_component_bases = tuple(
            tuple(int(value) for value in bases)
            for bases in attrs["destination_component_coordinate_bases"]
        )
        destination_workitem_coefficients = tuple(
            tuple(int(value) for value in coefficients)
            for coefficients in attrs["destination_workitem_coordinate_coefficients"]
        )
        if (
            len(destination_component_bases) != expected_components
            or any(len(bases) != len(destination_shape) for bases in destination_component_bases)
            or any(
                len(coefficients) != len(destination_shape)
                for coefficients in destination_workitem_coefficients
            )
        ):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "amdg.buffer_load_to_local coordinate destination offsets "
                "do not match target op attrs",
                target_op_id=op.target_op_id,
            )
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "unsupported amdg.buffer_load_to_local destination offset mode "
            f"{destination_offset_mode}",
            target_op_id=op.target_op_id,
        )
    mask_components = None
    if masks is not None:
        mask_components = _broadcast_component(masks, expected_components, op)
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
            issue_dependencies,
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
    dependency = _memory_dependency_token(state, issue_dependencies)
    component_tokens = []
    workitem = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
    lane_offset = None
    if destination_offset_mode == "affine":
        lane_offset = _local_destination_lane_offset(
            state,
            workitem,
            lane_width,
            int(attrs.get("destination_lane_stride_elements", 1)),
            int(attrs.get("destination_wave_stride_elements", 0)),
        )
    value_type = state.dsl.simd_type(element_type, lane_width)
    mask_mode = attrs.get("mask_mode", "exec_where" if has_mask else "none")
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
    def component_destination_offset(component_index):
        if destination_offset_mode == "affine":
            dest_offset = lane_offset
            destination_base_offset = destination_offsets[component_index]
            if destination_base_offset:
                base_offset = state.builder.splat(
                    state.builder.constant(
                        state.dsl.i32(),
                        destination_base_offset,
                    ),
                    state.dsl.i32(),
                    lane_width,
                )
                dest_offset = state.builder.binary(
                    state.dsl.BinaryKind.AddI,
                    dest_offset,
                    base_offset,
                )
            return dest_offset
        coords = tuple(
            _bit_linear_thread_coordinate(
                state,
                workitem,
                int(base),
                tuple(
                    coefficients[dim]
                    for coefficients in destination_workitem_coefficients
                ),
                lane_width,
            )
            for dim, base in enumerate(destination_component_bases[component_index])
        )
        return _shared_destination_element_offset(
            state,
            attrs,
            coords,
            destination_shape,
            lane_width,
        )

    def emit_component_load_store(component_index, offset_component):
        offset_component = _assume_value_range(
            state,
            offset_component,
            attrs.get("offset_range"),
            op,
        )
        source_ptr = state.builder.ptr_add(
            buffer_base,
            offset_component,
            result_type=source_ptr_type,
        )
        loaded, load_token = state.builder.load(
            source_ptr,
            value_type,
            after=dependency,
        )
        dest_offset = component_destination_offset(component_index)
        dest_ptr = state.builder.ptr_add(
            dest_base,
            dest_offset,
            result_type=dest_ptr_type,
        )
        return state.builder.store(loaded, dest_ptr, after=load_token)

    for index, offset_component in enumerate(offset_components):
        if mask_components is None:
            component_tokens.append(
                emit_component_load_store(index, offset_component)
            )
            continue
        if mask_mode != "exec_where":
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC_MASK",
                STAGE,
                f"unsupported buffer_load_to_local mask mode {mask_mode}",
                target_op_id=op.target_op_id,
            )
        with state.builder.where(
            mask_components[index],
            [state.dsl.mem_token_type()],
        ) as where:
            store_token = emit_component_load_store(
                index,
                offset_component,
            )
            state.builder.yield_([store_token])
        component_tokens.append(where.results[0])
    state.values[_single_result(op)] = _join_memory_tokens(state, component_tokens)


def _local_destination_lane_offset(
    state,
    workitem,
    lane_width,
    lane_stride,
    wave_stride,
):
    lane_width = int(lane_width)
    lane_stride = int(lane_stride)
    wave_stride = int(wave_stride)
    if wave_stride == 0:
        if lane_stride == 1:
            return workitem
        return _simd_binary_const(state, "muli", workitem, lane_stride, lane_width)
    lane = _simd_binary_const(state, "remui", workitem, lane_width, lane_width)
    if lane_stride != 1:
        lane = _simd_binary_const(state, "muli", lane, lane_stride, lane_width)
    wave_first = state.builder.read_first(workitem)
    wave_id = _scalar_binary_const_i32(state, "divui", wave_first, lane_width)
    wave_offset = _scalar_binary_const_i32(state, "muli", wave_id, wave_stride)
    wave_offset = state.builder.splat(wave_offset, state.dsl.i32(), lane_width)
    return state.builder.binary(state.dsl.BinaryKind.AddI, lane, wave_offset)


def _shared_destination_element_offset(state, attrs, coords, shape, lane_width):
    layout = attrs.get("destination_shared_layout", "dense")
    if layout == "dense":
        return _linearize_coordinates(state, coords, shape, lane_width)
    if layout == "padded":
        logical = _linearize_coordinates(state, coords, shape, lane_width)
        encoded = logical
        intervals = tuple(int(value) for value in attrs["destination_padded_intervals"])
        paddings = tuple(int(value) for value in attrs["destination_padded_paddings"])
        for interval, padding in zip(intervals, paddings):
            term = _simd_binary_const(state, "divui", logical, interval, lane_width)
            if padding != 1:
                term = _simd_binary_const(state, "muli", term, padding, lane_width)
            encoded = state.builder.binary(state.dsl.BinaryKind.AddI, encoded, term)
        return encoded
    if layout == "swizzled":
        order = tuple(int(value) for value in attrs["destination_swizzled_order"])
        minor_dim = int(order[0])
        major_dim = int(order[1])
        minor_extent = int(shape[minor_dim])
        vec = int(attrs["destination_swizzled_vec"])
        per_phase = int(attrs["destination_swizzled_per_phase"])
        max_phase = int(attrs["destination_swizzled_max_phase"])
        major = coords[major_dim]
        minor = coords[minor_dim]
        phase = _simd_binary_const(state, "divui", major, per_phase, lane_width)
        phase = _simd_binary_const(state, "remui", phase, max_phase, lane_width)
        minor_group = _simd_binary_const(state, "divui", minor, vec, lane_width)
        minor_inner = _simd_binary_const(state, "remui", minor, vec, lane_width)
        swizzled_minor = state.builder.binary(
            state.dsl.BinaryKind.XOrI,
            minor_group,
            phase,
        )
        if vec != 1:
            swizzled_minor = _simd_binary_const(
                state,
                "muli",
                swizzled_minor,
                vec,
                lane_width,
            )
        swizzled_minor = state.builder.binary(
            state.dsl.BinaryKind.AddI,
            swizzled_minor,
            minor_inner,
        )
        major_offset = major
        if minor_extent != 1:
            major_offset = _simd_binary_const(
                state,
                "muli",
                major_offset,
                minor_extent,
                lane_width,
            )
        return state.builder.binary(
            state.dsl.BinaryKind.AddI,
            major_offset,
            swizzled_minor,
        )
    fail(
        "TLXW_EMIT_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"unsupported scalarized shared destination layout {layout}",
    )


def _emit_buffer_load_to_local_packet_dma(
    state,
    op,
    attrs,
    dest_base,
    buffer_base,
    scalar_values,
    issue_dependencies,
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
    dependency = _memory_dependency_token(state, issue_dependencies)
    component_tokens = []
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
        component_tokens.append(
            state.builder.dma_load_lds(
                source_ptr,
                dest_ptr,
                after=dependency,
                bytes=packet_bytes,
            )
        )
    return _join_memory_tokens(state, component_tokens)


def _emit_buffer_load_to_local_dma(
    state,
    op,
    attrs,
    dest_base,
    buffer_base,
    offset_components,
    destination_offsets,
    issue_dependencies,
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
    dependency = _memory_dependency_token(state, issue_dependencies)
    component_tokens = []
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
        component_tokens.append(
            state.builder.dma_load_lds(
                source_ptr,
                dest_ptr,
                after=dependency,
                bytes=packet_bytes,
            )
        )
    return _join_memory_tokens(state, component_tokens)


def _emit_async_commit_group(state, op):
    tokens = tuple(_require_value(state, target_value_id, op) for target_value_id in op.operands)
    if tokens:
        token = state.builder.join(*tokens)
    else:
        token = state.builder.token()
    if op.results:
        state.values[_single_result(op)] = token


def _emit_token(state, op):
    state.values[_single_result(op)] = state.builder.token()


def _join_memory_tokens(state, tokens):
    tokens = tuple(tokens)
    if not tokens:
        return state.builder.token()
    if len(tokens) == 1:
        return tokens[0]
    return state.builder.join(*tokens)


def _memory_dependency_token(state, tokens):
    tokens = tuple(tokens)
    if not tokens:
        return state.builder.token()
    return _join_memory_tokens(state, tokens)


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
    ptr_type = state.dsl.simd_ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
        lane_width,
    )
    fragments = []
    for component_offset in component_offsets:
        offset = _linear_local_fragment_index_offset(
            state,
            wi,
            lane_width,
            elements_per_lane=registers,
            wave_tile_axis=wave_tile_axis,
            warps_per_cta=warps_per_cta,
            wave_tile_stride=wave_tile_stride_dwords,
            extra_elements=component_offset,
            op=op,
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
    chunk_element_deltas = attrs.get("chunk_element_deltas")
    fragments = []
    for component_index, tile_offsets in enumerate(attrs["component_tile_offsets"]):
        components = []
        component_deltas = None
        if chunk_element_deltas is not None:
            component_deltas = tuple(
                int(value) for value in chunk_element_deltas[component_index]
            )
        for chunk in range(int(attrs["chunks_per_component"])):
            logical_extra_elements = (
                0
                if component_deltas is not None
                else int(attrs["chunk_elements"]) * chunk
            )
            physical_extra_elements = (
                int(component_deltas[chunk])
                if component_deltas is not None
                else 0
            )
            offset = _local_fragment_element_offset(
                state,
                attrs,
                wi,
                tuple(int(value) for value in tile_offsets),
                logical_extra_elements,
                lane_width,
                elements_per_lane=int(attrs["elements_per_lane"]),
                wave_tile_axis=attrs.get("wave_tile_axis", "none"),
                warps_per_cta=tuple(
                    int(value) for value in attrs.get("warps_per_cta", (1, 1))
                ),
                wave_tile_stride=int(attrs.get("wave_tile_stride_elements", 0)),
                op=op,
                physical_extra_elements=physical_extra_elements,
            )
            ptr = state.builder.ptr_add(base, offset, result_type=ptr_type)
            loaded, _token = state.builder.transpose_load(ptr, load_type)
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
):
    lane_width = int(attrs["lane_width"])
    i32_shared = state.dsl.ptr_type(state.dsl.i32(), state.dsl.shared_address_space())
    base_i32 = _ptr_cast(state, base, i32_shared)
    ptr_type = state.dsl.simd_ptr_type(
        state.dsl.i32(),
        state.dsl.shared_address_space(),
        lane_width,
    )
    fragments = []
    for tile_offsets in attrs["component_tile_offsets"]:
        offset = _local_fragment_element_offset(
            state,
            attrs,
            wi,
            tuple(int(value) for value in tile_offsets),
            0,
            lane_width,
            elements_per_lane=int(attrs["elements_per_lane"]),
            wave_tile_axis=attrs.get("wave_tile_axis", "none"),
            warps_per_cta=tuple(
                int(value) for value in attrs.get("warps_per_cta", (1, 1))
            ),
            wave_tile_stride=int(attrs.get("wave_tile_stride_elements", 0)),
            op=op,
            elements_per_offset_unit=2,
        )
        ptr = state.builder.ptr_add(base_i32, offset, result_type=ptr_type)
        fragment, _token = state.builder.fragment_load(ptr, fragment_type)
        fragments.append(fragment)
    return _pack_components(tuple(fragments))


def _local_fragment_element_offset(
    state,
    attrs,
    wi,
    tile_offsets,
    extra_elements,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    op,
    elements_per_offset_unit=1,
    physical_extra_elements=0,
):
    tile_base = _dense_tile_base_elements(
        attrs.get("memdesc_shape", attrs["source_shape"]),
        tile_offsets,
    )
    logical = _linear_local_fragment_offset_expr(
        state,
        lane_width,
        elements_per_lane=elements_per_lane,
        wave_tile_axis=wave_tile_axis,
        warps_per_cta=warps_per_cta,
        wave_tile_stride=wave_tile_stride,
        extra_elements=int(tile_base) + int(extra_elements),
        op=op,
    )
    layout_kind = attrs.get("shared_layout_kind", "dense")
    if layout_kind == "dense":
        encoded = logical
    elif layout_kind == "swizzled_shared":
        encoded = _swizzled_element_offset_expr(state, attrs, logical)
    elif layout_kind == "padded_shared":
        encoded = _padded_element_offset_expr(state, attrs, logical)
    else:
        fail(
            "TLXW_EMIT_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported local_load shared layout {layout_kind}",
        )
    if int(elements_per_offset_unit) != 1:
        encoded = state.dsl.floor(encoded / int(elements_per_offset_unit))
    if int(physical_extra_elements):
        encoded += int(physical_extra_elements)
    wi_sym = state.dsl.sym("wi")
    return state.builder.index_expr(encoded, bindings={wi_sym: wi})


def _linear_local_fragment_index_offset(
    state,
    wi,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    extra_elements,
    op,
):
    expr = _linear_local_fragment_offset_expr(
        state,
        lane_width,
        elements_per_lane=elements_per_lane,
        wave_tile_axis=wave_tile_axis,
        warps_per_cta=warps_per_cta,
        wave_tile_stride=wave_tile_stride,
        extra_elements=extra_elements,
        op=op,
    )
    wi_sym = state.dsl.sym("wi")
    return state.builder.index_expr(expr, bindings={wi_sym: wi})


def _linear_local_fragment_offset_expr(
    state,
    lane_width,
    *,
    elements_per_lane,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
    extra_elements,
    op,
):
    wi = state.dsl.sym("wi")
    lane = state.dsl.mod(wi, int(lane_width))
    expr = lane * int(elements_per_lane)
    wave_tile = _wave_tile_offset_expr(
        state,
        wi,
        lane_width,
        wave_tile_axis,
        warps_per_cta,
        wave_tile_stride,
        op,
    )
    if wave_tile is not None:
        expr += wave_tile
    if int(extra_elements):
        expr += int(extra_elements)
    return expr


def _wave_tile_offset_expr(
    state,
    wi,
    lane_width,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride,
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
    wave_id = state.dsl.floor(wi / int(lane_width))
    if wave_tile_axis == "m":
        wave_coord = state.dsl.floor(wave_id / int(warps_per_cta[1]))
    elif wave_tile_axis == "n":
        wave_coord = state.dsl.mod(wave_id, int(warps_per_cta[1]))
    else:
        fail(
            "TLXW_EMIT_LOCAL_LOAD_TILE_MAP",
            STAGE,
            f"unsupported local_load_fragment wave axis {wave_tile_axis}",
            target_op_id=op.target_op_id,
        )
    return wave_coord * int(wave_tile_stride)


def _swizzled_element_offset_expr(state, attrs, logical):
    cols = int(attrs.get("memdesc_shape", attrs["source_shape"])[-1])
    vec = int(attrs["swizzled_vec"])
    row = state.dsl.floor(logical / cols)
    col = state.dsl.mod(logical, cols)
    row_phase = state.dsl.floor(row / int(attrs["swizzled_per_phase"]))
    phase = state.dsl.mod(row_phase, int(attrs["swizzled_max_phase"]))
    col_group = state.dsl.floor(col / vec)
    swizzled_group = state.dsl.xor(col_group, phase)
    swizzled_col = swizzled_group * vec + state.dsl.mod(col, vec)
    return row * cols + swizzled_col


def _padded_element_offset_expr(state, attrs, logical):
    encoded = logical
    for interval, padding in zip(
        attrs.get("padded_intervals", ()),
        attrs.get("padded_paddings", ()),
    ):
        encoded += state.dsl.floor(logical / int(interval)) * int(padding)
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
    if mode in {"same_lane_register_remap", "cross_lane_register_remap"}:
        result_count = int(attrs["result_component_count"])
        source_indices = tuple(int(index) for index in attrs["source_indices"])
        source_element_indices = tuple(
            int(index) for index in attrs["source_element_indices"]
        )
        registers_per_component = int(attrs["source_registers_per_component"])
        if len(source_indices) != result_count or len(source_element_indices) != result_count:
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "layout_convert remap attrs do not match result component count",
                target_op_id=op.target_op_id,
            )
        if len(components) != int(attrs["source_component_count"]):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "layout_convert remap source component count does not match attrs",
                target_op_id=op.target_op_id,
            )
        result_id = _single_result(op)
        result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
        extracted = {}

        def scalar_component(component_index, element_index):
            key = (int(component_index), int(element_index))
            if key in extracted:
                return extracted[key]
            component = components[int(component_index)]
            if registers_per_component == 1:
                if int(element_index) != 0:
                    fail(
                        "TLXW_EMIT_LAYOUT_REMAP",
                        STAGE,
                        "scalar layout remap requested a non-zero element index",
                        target_op_id=op.target_op_id,
                    )
                extracted[key] = component
                return component
            if str(component.type).startswith("!waveamd.fragment"):
                fail(
                    "TLXW_EMIT_LAYOUT_REMAP",
                    STAGE,
                    "layout_convert register remap cannot extract directly "
                    "from a WaveAMD fragment; fragment unpack is required",
                    target_op_id=op.target_op_id,
                )
            extracted[key] = state.dsl.wave.ExtractOp(
                result_type,
                component,
                int(element_index),
            ).result
            return extracted[key]

        source_lane = None

        def remapped_component(component_index, element_index):
            component = scalar_component(component_index, element_index)
            if mode == "same_lane_register_remap":
                return component
            nonlocal source_lane
            if source_lane is None:
                source_lane = _layout_convert_source_lane(
                    state,
                    attrs,
                    component,
                    op,
                )
            return _shuffle_component(state, component, source_lane, op)

        state.values[result_id] = _pack_components(
            tuple(
                remapped_component(component_index, element_index)
                for component_index, element_index in zip(
                    source_indices,
                    source_element_indices,
                )
            )
        )
        return
    if mode == "cta_exchange_register_remap":
        result_count = int(attrs["result_component_count"])
        registers_per_component = int(attrs["source_registers_per_component"])
        if len(components) != int(attrs["source_component_count"]):
            fail(
                "TLXW_EMIT_COMPONENT_COUNT",
                STAGE,
                "layout_convert CTA exchange source component count does not "
                "match attrs",
                target_op_id=op.target_op_id,
            )
        result_id = _single_result(op)
        target_type = state.target_program.values[result_id].type
        result_type = _wave_type(state.dsl, target_type)
        lane_width = int(target_type.lane_width or 64)
        element_type = _scalar_type(state.dsl, target_type.element_type)
        cta_thread_count = int(attrs["cta_thread_count"])
        if cta_thread_count % lane_width:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "CTA exchange thread count must be a multiple of lane width",
                target_op_id=op.target_op_id,
            )
        exchange_groups = tuple(attrs["exchange_groups"])
        scratch_base = state.builder.lds_base(
            element_type,
            offset=int(attrs["scratch_byte_offset"]),
        )
        ptr_type = state.dsl.simd_ptr_type(
            element_type,
            state.dsl.shared_address_space(),
            lane_width,
        )
        workitem = state.builder.workitem_id(0, state.dsl.i32(), lane_width)
        result_components = [None] * result_count
        extracted = {}

        def scalar_component(component_index, element_index):
            key = (int(component_index), int(element_index))
            if key in extracted:
                return extracted[key]
            component = components[int(component_index)]
            if registers_per_component == 1:
                if int(element_index) != 0:
                    fail(
                        "TLXW_EMIT_LAYOUT_REMAP",
                        STAGE,
                        "scalar CTA exchange remap requested a non-zero "
                        "element index",
                        target_op_id=op.target_op_id,
                    )
                extracted[key] = component
                return component
            if str(component.type).startswith("!waveamd.fragment"):
                fail(
                    "TLXW_EMIT_LAYOUT_REMAP",
                    STAGE,
                    "layout_convert CTA exchange cannot extract directly "
                    "from a WaveAMD fragment; fragment unpack is required",
                    target_op_id=op.target_op_id,
                )
            extracted[key] = state.dsl.wave.ExtractOp(
                result_type,
                component,
                int(element_index),
            ).result
            return extracted[key]

        group_dependency = state.scratch_token
        for group in exchange_groups:
            source_slots, result_indices, load_bases, load_coefficients = group
            store_tokens = []
            for slot_index, source_slot in enumerate(source_slots):
                source_slot = int(source_slot)
                store_offset = workitem
                base_offset = int(slot_index) * cta_thread_count
                if base_offset:
                    store_offset = _simd_binary_const(
                        state,
                        "addi",
                        store_offset,
                        base_offset,
                        lane_width,
                    )
                ptr = state.builder.ptr_add(
                    scratch_base,
                    store_offset,
                    result_type=ptr_type,
                )
                value = scalar_component(
                    source_slot // registers_per_component,
                    source_slot % registers_per_component,
                )
                store_tokens.append(
                    state.builder.store(value, ptr, after=group_dependency)
                )
            barrier_token = state.builder.barrier(*store_tokens)
            load_tokens = []
            for result_index, load_base, coefficients in zip(
                result_indices,
                load_bases,
                load_coefficients,
            ):
                load_offset = _bit_affine_thread_offset(
                    state,
                    workitem,
                    load_base,
                    coefficients,
                    lane_width,
                )
                ptr = state.builder.ptr_add(
                    scratch_base,
                    load_offset,
                    result_type=ptr_type,
                )
                loaded, load_token = state.builder.load(
                    ptr,
                    result_type,
                    after=barrier_token,
                )
                result_components[int(result_index)] = loaded
                load_tokens.append(load_token)
            group_dependency = state.builder.barrier(*load_tokens)
        state.scratch_token = group_dependency
        missing = [
            index for index, component in enumerate(result_components) if component is None
        ]
        if missing:
            fail(
                "TLXW_EMIT_LAYOUT_REMAP",
                STAGE,
                "CTA exchange remap did not populate every result component",
                target_op_id=op.target_op_id,
            )
        state.values[result_id] = _pack_components(tuple(result_components))
        return
    fail(
        "TLXW_EMIT_UNSUPPORTED_LAYOUT_CONVERT",
        STAGE,
        f"unsupported layout_convert mode {mode}",
        target_op_id=op.target_op_id,
    )


def _bit_affine_thread_offset(state, workitem, base, coefficients, lane_width):
    lane_width = int(lane_width)
    result = state.builder.splat(
        state.builder.constant(state.dsl.i32(), int(base)),
        state.dsl.i32(),
        lane_width,
    )
    for bit, coefficient in enumerate(coefficients):
        coefficient = int(coefficient)
        if coefficient == 0:
            continue
        bit_value = _simd_binary_const(state, "divui", workitem, 1 << bit, lane_width)
        bit_value = _simd_binary_const(state, "remui", bit_value, 2, lane_width)
        if coefficient != 1:
            bit_value = _simd_binary_const(
                state,
                "muli",
                bit_value,
                coefficient,
                lane_width,
            )
        result = state.builder.binary(state.dsl.BinaryKind.AddI, result, bit_value)
    return result


def _layout_convert_source_lane(state, attrs, component, op):
    simd = _require_shuffle_simd(state, component, op)
    lane_width = int(simd.width)
    lane = state.builder.lane_id(state.dsl.i32(), lane_width)
    kind = attrs.get("source_lane_map_kind")
    if kind == "affine":
        stride = int(attrs["source_lane_affine_stride"])
        base = int(attrs["source_lane_affine_base"])
        if stride == 0:
            lane = state.builder.splat(
                state.builder.constant(state.dsl.i32(), base),
                state.dsl.i32(),
                lane_width,
            )
        else:
            if stride != 1:
                lane = _simd_binary_const(state, "muli", lane, stride, lane_width)
            if base:
                lane = _simd_binary_const(state, "addi", lane, base, lane_width)
        return lane
    if kind == "transpose":
        inner = int(attrs["source_lane_transpose_inner"])
        outer = int(attrs["source_lane_transpose_outer"])
        minor = _simd_binary_const(state, "remui", lane, inner, lane_width)
        major = _simd_binary_const(state, "divui", lane, inner, lane_width)
        minor = _simd_binary_const(state, "muli", minor, outer, lane_width)
        return state.builder.binary(_binary_kind(state.dsl, "addi"), minor, major)
    fail(
        "TLXW_EMIT_LAYOUT_REMAP",
        STAGE,
        f"unsupported layout_convert source lane map {kind!r}",
        target_op_id=op.target_op_id,
    )


def _shuffle_component(state, component, source_lane, op):
    _require_shuffle_simd(state, component, op)
    return state.ir.Operation.create(
        "wave.shuffle",
        results=[component.type],
        operands=[component, source_lane],
    ).results[0]


def _require_shuffle_simd(state, component, op):
    try:
        simd = state.dsl.SimdType(component.type)
    except Exception:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "layout_convert shuffle requires a Wave SIMD component",
            target_op_id=op.target_op_id,
        )
    element_type = str(simd.element_type)
    if element_type.startswith("vector<") or element_type not in {
        "i8",
        "i16",
        "i32",
        "index",
        "f16",
        "bf16",
        "f32",
    }:
        fail(
            "TLXW_EMIT_LAYOUT_REMAP",
            STAGE,
            "layout_convert shuffle supports only b32-compatible scalar "
            f"SIMD payloads, got {component.type}",
            target_op_id=op.target_op_id,
        )
    return simd


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


def _emit_buffer_load(state, op):
    attrs = target_ir.attrs_dict(op)
    operand_count = 2 + int(bool(attrs["has_mask"])) + int(bool(attrs["has_other"]))
    operands = _operand_values(state, op, operand_count)
    source_base, offsets = operands[:2]
    operand_index = 2
    masks = None
    if attrs["has_mask"]:
        masks = operands[operand_index]
        operand_index += 1
    other = operands[operand_index] if attrs["has_other"] else None
    offset_components = _as_components(offsets)
    mask_components = None if masks is None else _as_components(masks)
    other_components = None if other is None else _as_components(other)
    component_count = int(attrs["component_count"])
    if len(offset_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load offset component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if mask_components is not None and len(mask_components) != component_count:
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load mask component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    if other_components is not None and len(other_components) not in (1, component_count):
        fail(
            "TLXW_EMIT_COMPONENT_COUNT",
            STAGE,
            "buffer_load other component count does not match attrs",
            target_op_id=op.target_op_id,
        )
    result_id = _single_result(op)
    result_type = _wave_type(state.dsl, state.target_program.values[result_id].type)
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
    loaded_components = []
    for index, offset_component in enumerate(offset_components):
        if mask_components is None:
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
            loaded, _token = state.builder.load(ptr, result_type)
        else:
            if mask_mode != "exec_where":
                fail(
                    "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_MASK",
                    STAGE,
                    f"unsupported buffer_load mask mode {mask_mode}",
                    target_op_id=op.target_op_id,
                )
            with state.builder.where(mask_components[index], [result_type]) as where:
                active_offset = _assume_value_range(
                    state,
                    offset_component,
                    attrs.get("offset_range"),
                    op,
                )
                ptr = state.builder.ptr_add(
                    buffer_base,
                    active_offset,
                    result_type=ptr_type,
                )
                loaded, _token = state.builder.load(ptr, result_type)
                state.builder.yield_([loaded])
            loaded = where.results[0]
            if other_components is not None:
                other_component = (
                    other_components[0]
                    if len(other_components) == 1
                    else other_components[index]
                )
                loaded = state.builder.select(
                    mask_components[index],
                    loaded,
                    other_component,
                )
        if other_components is not None and mask_components is None:
            fail(
                "TLXW_EMIT_UNSUPPORTED_BUFFER_LOAD_OTHER",
                STAGE,
                "buffer_load other requires a mask",
                target_op_id=op.target_op_id,
            )
        loaded_components.append(loaded)
    state.values[result_id] = _pack_components(tuple(loaded_components))


_TARGET_EMITTERS = {
    "constant": _emit_constant,
    "binary": _emit_binary,
    "float_binary": _emit_float_binary,
    "cmpi": _emit_cmpi,
    "minsi": _emit_minsi,
    "assume": _emit_assume,
    "make_range": _emit_make_range,
    "splat": _emit_splat,
    "broadcast": _emit_broadcast,
    "addptr": _emit_addptr,
    "expand_dims": _emit_expand_dims,
    "program_id": _emit_program_id,
    "for_loop": _emit_for_loop,
    "select": _emit_select,
    "local_alloc": _emit_local_alloc,
    "memdesc_index": _emit_memdesc_index,
    "buffer_load_to_local": _emit_buffer_load_to_local,
    "local_load_fragment": _emit_local_load_fragment,
    "fragment_fill": _emit_fragment_fill,
    "mma": _emit_mma,
    "fragment_truncf": _emit_fragment_truncf,
    "layout_convert": _emit_layout_convert,
    "buffer_store": _emit_buffer_store,
    "buffer_load": _emit_buffer_load,
    "token": _emit_token,
    "async_commit_group": _emit_async_commit_group,
    "async_wait": _emit_async_wait,
    "return": _emit_return,
}

_UNOWNED_TARGET_OPS = frozenset(_TARGET_EMITTERS) - domains.all_target_ops()
if _UNOWNED_TARGET_OPS:
    raise RuntimeError(f"unsupported target op domains: {sorted(_UNOWNED_TARGET_OPS)}")


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
        attrs = target_ir.attrs_dict(op)
        if op.kind == "local_alloc":
            end = int(attrs.get("byte_offset", 0)) + int(attrs.get("allocation_bytes", 0))
        elif op.kind == "layout_convert" and "scratch_allocation_bytes" in attrs:
            end = int(attrs.get("scratch_byte_offset", 0)) + int(
                attrs.get("scratch_allocation_bytes", 0)
            )
        else:
            continue
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
        op_kind = "<region-yield>" if op is None else op.kind
        op_id = None if op is None else op.target_op_id
        fail(
            "TLXW_EMIT_UNBOUND_VALUE",
            STAGE,
            f"target value {target_value_id} is not bound before {op_kind}",
            target_op_id=op_id,
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


def _kernel_num_warps(kernel):
    return int(kernel.num_warps or 1)


def _kernel_threads_per_warp(kernel):
    return int(kernel.threads_per_warp or 64)


def _kernel_workgroup_size(kernel):
    # Triton supports ND launch grids, but its per-CTA block shape is flat X:
    # AMD/NVIDIA launchers pass (warp_size * num_warps, 1, 1). num_ctas is a
    # cluster/CTA count and is not part of the per-workgroup thread shape.
    return [_kernel_num_warps(kernel) * _kernel_threads_per_warp(kernel), 1, 1]


def _function_attrs(dsl, ir, kernel):
    num_warps = _kernel_num_warps(kernel)
    return {
        "tlx_wave.converter.stage": ir.StringAttr.get("structural-emission"),
        "tlx_wave.num_warps": ir.IntegerAttr.get(dsl.i32(), num_warps),
        "tlx_wave.wave_size": ir.IntegerAttr.get(
            dsl.i32(),
            _kernel_threads_per_warp(kernel),
        ),
        "tlx_wave.ttgir.noinline": ir.Attribute.parse(
            "true" if kernel.noinline else "false"
        ),
        "wave.waves_per_workgroup": dsl.i64_attr(num_warps),
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
