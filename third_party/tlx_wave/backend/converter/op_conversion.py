"""Stateless source-op to target-program conversion."""

from dataclasses import dataclass, replace
import re

from .diagnostics import fail
from . import domains
from . import layouts
from . import coordinates
from . import layout_remap
from . import target_ir


STAGE = "op_conversion"

_BINARY_OPS = {
    "arith.addi": "addi",
    "arith.subi": "subi",
    "arith.muli": "muli",
    "arith.andi": "andi",
    "arith.ori": "ori",
    "arith.xori": "xori",
    "arith.divsi": "divsi",
    "arith.divui": "divui",
    "arith.remsi": "remsi",
    "arith.remui": "remui",
}

_FLOAT_BINARY_OPS = {
    "arith.addf": "addf",
    "arith.subf": "subf",
    "arith.mulf": "mulf",
}


@dataclass(frozen=True)
class OpConversionView:
    op_index: int
    op_name: str
    attrs: dict
    operand_target_ids: tuple[int, ...]
    result_target_ids: tuple[int, ...]
    result_layout_map_ids: tuple[int, ...]
    fact_ids: tuple[int, ...]
    fact_target_ids: tuple[int, ...]
    operand_fact_ids: tuple[int, ...]
    operand_fact_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class MemdescInfo:
    value_id: int
    element_type: str | None
    element_byte_width: int | None
    shape: tuple[int, ...]
    alloc_shape: tuple[int, ...]
    allocation_bytes: int


@dataclass(frozen=True)
class ConversionInput:
    kernel: target_ir.TargetKernel
    kernel_arg_ids: tuple[int, ...]
    top_region_id: int
    ops: tuple
    regions: tuple
    num_warps: int
    threads_per_warp: int
    value_element_byte_widths: dict[int, int | None]
    memdescs: dict[int, MemdescInfo]
    memdesc_physical_allocation_bytes: dict[int, int]
    local_alloc_allocation_bytes: dict[int, int]
    constant_ints: dict[int, int]
    fact_ids_by_op: dict[int, tuple[int, ...]]
    token_nodes_by_op: dict[int, object]
    token_groups_by_commit: dict[int, object]
    token_groups_by_id: dict[int, object]
    async_issue_dependency_target_ids_by_op: dict[int, tuple[int, ...]]
    local_alloc_byte_offsets: dict[int, int]
    lds_size: int
    static_memdesc_byte_offsets: dict[int, int]


def convert_ops(source_program, type_layout_program, fact_program, token_program):
    conversion_input = _build_conversion_input(
        source_program,
        type_layout_program,
        fact_program,
        token_program,
    )
    builder = target_ir.TargetBuilder(conversion_input.kernel)
    _seed_kernel_arguments(builder, conversion_input, type_layout_program)

    _convert_region(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        conversion_input.top_region_id,
        allow_yield=False,
    )

    return builder.build()


def _build_conversion_input(source_program, type_layout_program, fact_program, token_program):
    memdescs = _memdesc_infos(source_program)
    constant_ints = _constant_ints(source_program)
    memdesc_physical_allocation_bytes = _compute_memdesc_physical_allocation_bytes(
        source_program.values,
        source_program.ops,
        type_layout_program,
        memdescs,
    )
    local_alloc_allocation_bytes = _compute_local_alloc_allocation_bytes(
        source_program.ops,
        memdescs,
        memdesc_physical_allocation_bytes,
    )
    local_alloc_byte_offsets, lds_size = _compute_local_alloc_layout(
        source_program.ops,
        memdescs,
        local_alloc_allocation_bytes,
    )
    static_memdesc_byte_offsets = _compute_static_memdesc_byte_offsets(
        source_program.ops,
        memdescs,
        memdesc_physical_allocation_bytes,
        constant_ints,
        local_alloc_byte_offsets,
    )
    kernel = target_ir.TargetKernel(
        source_program.kernel.name,
        source_program.kernel.target,
        source_program.kernel.num_ctas,
        source_program.kernel.num_warps,
        source_program.kernel.threads_per_warp,
        source_program.kernel.noinline,
    )
    return ConversionInput(
        kernel,
        tuple(source_program.kernel.arg_ids),
        int(source_program.top_region_id),
        tuple(source_program.ops),
        tuple(source_program.regions),
        int(source_program.kernel.num_warps or 1),
        int(source_program.kernel.threads_per_warp or 64),
        {
            value_id: value.type.element_byte_width
            for value_id, value in source_program.values.items()
        },
        memdescs,
        memdesc_physical_allocation_bytes,
        local_alloc_allocation_bytes,
        constant_ints,
        _fact_ids_by_source_op(fact_program),
        {node.op_index: node for node in token_program.nodes},
        {group.commit_op_index: group for group in token_program.groups},
        {group.group_id: group for group in token_program.groups},
        {},
        local_alloc_byte_offsets,
        int(lds_size),
        static_memdesc_byte_offsets,
    )


def _convert_region(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    region_id,
    *,
    allow_yield,
):
    for op_index in conversion_input.regions[region_id].op_indices:
        op = conversion_input.ops[op_index]
        if op.name == "scf.yield":
            if not allow_yield:
                fail(
                    "TLXW_OP_UNEXPECTED_YIELD",
                    STAGE,
                    "scf.yield is only valid inside a converted region",
                    source_op_index=op.index,
                )
            return op.operands
        _convert_source_op(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
    if allow_yield:
        fail(
            "TLXW_OP_MISSING_YIELD",
            STAGE,
            f"region {region_id} has no scf.yield",
        )
    return ()


def _convert_source_op(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    if op.name == "scf.for":
        _convert_for(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "scf.if":
        _convert_if(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "arith.constant" and _has_fragment_result(type_layout_program, op):
        _convert_fragment_constant(builder, type_layout_program, op)
        return
    if op.name == "arith.truncf" and _has_fragment_result(type_layout_program, op):
        _convert_fragment_truncf(builder, type_layout_program, op)
        return
    if op.name == "ttg.local_alloc":
        _convert_local_alloc(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "ttg.memdesc_index":
        _convert_memdesc_index(
            builder,
            conversion_input,
            type_layout_program,
            op,
        )
        return
    if op.name == "ttg.local_load":
        _convert_local_load(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "ttg.convert_layout":
        _convert_layout(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "tt.dot":
        _convert_dot(builder, type_layout_program, op)
        return
    if op.name == "amdg.buffer_load_to_local":
        _convert_buffer_load_to_local(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "amdg.buffer_load":
        _convert_buffer_load(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "amdg.buffer_store":
        _convert_buffer_store(
            builder,
            conversion_input,
            type_layout_program,
            fact_program,
            op,
        )
        return
    if op.name == "tt.load":
        _convert_load(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "tt.store":
        _convert_store(builder, conversion_input, type_layout_program, op)
        return
    if op.name == "ttg.async_commit_group":
        _convert_async_commit_group(
            builder,
            type_layout_program,
            conversion_input.token_groups_by_commit,
            op,
        )
        return
    if op.name == "ttg.async_wait":
        _convert_async_wait(
            builder,
            type_layout_program,
            conversion_input.token_nodes_by_op,
            conversion_input.token_groups_by_id,
            op,
        )
        return
    if op.name == "rocdl.sched.barrier":
        _convert_sched_barrier(op)
        return
    if op.name == "tt.make_range":
        _convert_make_range(builder, type_layout_program, op)
        return
    if op.name == "tt.broadcast":
        _convert_broadcast(builder, type_layout_program, op)
        return
    converter = _converter_for_op(op.name)
    if converter is None:
        fail(
            "TLXW_OP_UNSUPPORTED",
            STAGE,
            f"no op conversion for {op.name}",
            source_op_index=op.index,
        )
    operand_target_ids = _operand_target_ids(builder, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    fact_ids = conversion_input.fact_ids_by_op.get(op.index, ())
    operand_fact_ids = _operand_assume_fact_ids(fact_program, op)
    view = OpConversionView(
        op.index,
        op.name,
        dict(op.attrs),
        operand_target_ids,
        result_target_ids,
        result_layout_map_ids,
        fact_ids,
        _fact_target_ids(builder, fact_program, fact_ids, op),
        operand_fact_ids,
        _fact_target_ids(builder, fact_program, operand_fact_ids, op),
    )
    converter(builder, view)


def _seed_kernel_arguments(builder, conversion_input, type_layout_program):
    arg_target_ids = []
    for source_value_id in conversion_input.kernel_arg_ids:
        converted = type_layout_program.values[source_value_id]
        arg_target_ids.append(
            builder.add_value(
                target_ir.target_type_from_converted(converted.type),
                source_value_id=source_value_id,
                debug_name=f"arg{source_value_id}",
            )
        )
    builder.set_kernel_arg_targets(tuple(arg_target_ids))


def _declare_results(builder, op, type_layout_program):
    result_target_ids = []
    result_layout_map_ids = []
    for source_value_id in op.results:
        converted = type_layout_program.values[source_value_id]
        result_target_ids.append(
            builder.add_value(
                target_ir.target_type_from_converted(converted.type),
                source_value_id=source_value_id,
                debug_name=f"v{source_value_id}",
            )
        )
        if converted.layout_map_id is not None:
            result_layout_map_ids.append(converted.layout_map_id)
    return tuple(result_target_ids), tuple(result_layout_map_ids)


def _operand_target_ids(builder, op):
    operand_target_ids = []
    for source_value_id in op.operands:
        targets = builder.source_value_targets.get(source_value_id)
        if not targets:
            fail(
                "TLXW_OP_UNCONVERTED_OPERAND",
                STAGE,
                f"operand {source_value_id} has no converted target value",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
        if len(targets) != 1:
            fail(
                "TLXW_OP_MULTI_VALUE_OPERAND",
                STAGE,
                f"operand {source_value_id} maps to multiple target values {targets}",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
        operand_target_ids.append(targets[0])
    return tuple(operand_target_ids)


def _converter_for_op(op_name):
    return _SIMPLE_OP_CONVERTERS.get(op_name)


def _convert_constant(builder, view):
    builder.add_op(
        "constant",
        results=view.result_target_ids,
        attrs={
            "value": _constant_literal(
                view.attrs.get("value"),
                source_op_index=view.op_index,
            )
        },
        source_op_index=view.op_index,
    )


def _convert_binary(builder, view):
    attrs = {
        "operation": _BINARY_OPS[view.op_name],
        "source_width": _target_int_width(builder, view.result_target_ids),
    }
    nsw, nuw = _arith_overflow_flags(view)
    if nsw:
        attrs["nsw"] = True
    if nuw:
        attrs["nuw"] = True
    builder.add_op(
        "binary",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs=attrs,
        fact_ids=view.operand_fact_ids,
        fact_target_ids=view.operand_fact_target_ids,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_float_binary(builder, view):
    result_type = builder.values[view.result_target_ids[0]].type
    for target_value_id in (*view.operand_target_ids, *view.result_target_ids):
        target_type = builder.values[target_value_id].type
        if not _supports_float_binary_type(view.op_name, target_type, result_type):
            fail(
                "TLXW_OP_UNSUPPORTED_FLOAT_BINARY",
                STAGE,
                f"{view.op_name} requires supported Wave SIMD float operands",
                source_op_index=view.op_index,
                target_value_id=target_value_id,
            )
    builder.add_op(
        "float_binary",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"operation": _FLOAT_BINARY_OPS[view.op_name]},
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _supports_float_binary_type(op_name, target_type, result_type):
    if target_type.representation not in {"simd", "simd_tuple"}:
        return False
    if target_type.element_type != result_type.element_type:
        return False
    if op_name == "arith.subf":
        return target_type.element_type == "f32"
    return target_type.element_type in {"f16", "f32"}


def _convert_cmpi(builder, view):
    builder.add_op(
        "cmpi",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={
            "predicate": _cmpi_predicate(view.attrs.get("predicate")),
            "source_width": _target_int_width(builder, view.operand_target_ids),
        },
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_minsi(builder, view):
    builder.add_op(
        "minsi",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"source_width": _target_int_width(builder, view.result_target_ids)},
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_assume(builder, view):
    if not view.fact_ids:
        return
    builder.add_op(
        "assume",
        operands=view.operand_target_ids,
        fact_ids=view.fact_ids,
        fact_target_ids=view.fact_target_ids,
        source_op_index=view.op_index,
    )


def _arith_overflow_flags(view):
    if view.op_name not in {"arith.addi", "arith.muli", "arith.subi"}:
        return False, False
    attr = view.attrs.get("overflowFlags")
    if attr is None:
        return False, False
    text = str(attr)
    return "nsw" in text, "nuw" in text


def _convert_make_range(builder, type_layout_program, op):
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    attrs = {
        "start": _int_attr(op.attrs, "start"),
        "end": _int_attr(op.attrs, "end"),
    }
    attrs.update(_make_range_coordinate_attrs(type_layout_program, op))
    builder.add_op(
        "make_range",
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _make_range_coordinate_attrs(type_layout_program, op):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_MAKE_RANGE",
            STAGE,
            "tt.make_range requires one result",
            source_op_index=op.index,
        )
    result = type_layout_program.values[op.results[0]]
    if result.layout_map_id is None:
        return {}
    layout = type_layout_program.layouts[int(result.layout_map_id)]
    if layout.kind not in {"blocked", "linear", "slice"}:
        return {}
    lane_width = int(result.type.lane_width or layout.lane_width)
    warp_count = _layout_warp_count(layout)
    plan = coordinates.layout_coordinate_plan(
        layout,
        int(result.type.component_count),
        lane_width,
        warp_count,
        op,
        result.value_id,
    )
    if coordinates.is_default_flat_make_range(plan, lane_width):
        return {}
    affine = coordinates.is_flat_affine_make_range(plan, lane_width, warp_count)
    if affine is not None:
        bases, stride = affine
        return {
            "coordinate_mode": "affine_workitem",
            "component_bases": tuple(int(base) for base in bases),
            "workitem_stride": int(stride),
        }
    bit_affine = coordinates.is_flat_bit_affine_make_range(plan)
    if bit_affine is not None:
        bases, coefficients = bit_affine
        return {
            "coordinate_mode": "bit_affine_workitem",
            "component_bases": tuple(int(base) for base in bases),
            "workitem_coefficients": tuple(
                int(coefficient) for coefficient in coefficients
            ),
        }
    return {
        "coordinate_mode": "layout_coordinates",
        "coordinate_shape": tuple(int(dim) for dim in plan.shape),
        "component_coordinate_bases": tuple(
            tuple(int(value) for value in bases)
            for bases in plan.component_bases
        ),
        "workitem_coordinate_coefficients": tuple(
            tuple(int(value) for value in coefficients)
            for coefficients in plan.workitem_coefficients
        ),
    }


def _layout_warp_count(layout):
    return layouts.layout_warp_count(layout)


def _convert_splat(builder, view):
    builder.add_op(
        "splat",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_addptr(builder, view):
    builder.add_op(
        "addptr",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_expand_dims(builder, view):
    builder.add_op(
        "expand_dims",
        operands=view.operand_target_ids,
        results=view.result_target_ids,
        attrs={"axis": _int_attr(view.attrs, "axis")},
        layout_map_ids=view.result_layout_map_ids,
        source_op_index=view.op_index,
    )


def _convert_broadcast(builder, type_layout_program, op):
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    attrs = {}
    component_sources = _broadcast_component_sources(type_layout_program, op)
    if component_sources is not None:
        attrs["component_sources"] = component_sources
    builder.add_op(
        "broadcast",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _broadcast_component_sources(type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.layout_map_id is None or result.layout_map_id is None:
        return None
    operand_layout = type_layout_program.layouts[int(operand.layout_map_id)]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if len(operand_layout.shape) != len(result_layout.shape):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast requires rank-matched source and result layouts",
            source_op_index=op.index,
        )
    if operand_layout.kind not in {"blocked", "linear", "slice"}:
        return None
    if result_layout.kind not in {"blocked", "linear", "slice"}:
        return None
    if int(operand.type.component_count) == int(result.type.component_count):
        return tuple(range(int(result.type.component_count)))
    for source_extent, result_extent in zip(operand_layout.shape, result_layout.shape):
        if int(source_extent) not in {1, int(result_extent)}:
            fail(
                "TLXW_OP_BROADCAST",
                STAGE,
                "tt.broadcast source dimensions must either match the result "
                "or have extent one",
                source_op_index=op.index,
            )

    lane_width = int(
        result.type.lane_width
        or operand.type.lane_width
        or result_layout.lane_width
        or operand_layout.lane_width
        or 64
    )
    warp_count = max(
        layouts.layout_warp_count(operand_layout),
        layouts.layout_warp_count(result_layout),
    )
    source_linear = layouts.distributed_linear_layout(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_linear = layouts.distributed_linear_layout(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_register_count = layouts.linear_layout_in_dim_size(source_linear, "register")
    if int(source_register_count) != int(operand.type.component_count):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast source component model does not match its layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    result_register_count = layouts.linear_layout_in_dim_size(result_linear, "register")
    if int(result_register_count) != int(result.type.component_count):
        fail(
            "TLXW_OP_BROADCAST",
            STAGE,
            "tt.broadcast result component model does not match its layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    source_by_thread_coord = {}
    for warp in range(int(warp_count)):
        for source_register in range(int(source_register_count)):
            for lane in range(int(lane_width)):
                coords = layouts.linear_layout_coords(
                    source_linear,
                    source_register,
                    lane,
                    warp=warp,
                )
                key = (int(warp), int(lane), tuple(int(coord) for coord in coords))
                existing = source_by_thread_coord.get(key)
                if existing is not None and int(existing) != int(source_register):
                    fail(
                        "TLXW_OP_BROADCAST",
                        STAGE,
                        "tt.broadcast source layout maps multiple components to "
                        "one thread coordinate",
                        source_op_index=op.index,
                        source_value_id=operand.value_id,
                    )
                source_by_thread_coord[key] = int(source_register)

    component_sources = []
    for result_register in range(int(result_register_count)):
        source_registers = set()
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                result_coords = layouts.linear_layout_coords(
                    result_linear,
                    result_register,
                    lane,
                    warp=warp,
                )
                source_coords = tuple(
                    0 if int(source_extent) == 1 else int(coord)
                    for source_extent, coord in zip(operand_layout.shape, result_coords)
                )
                source_register = source_by_thread_coord.get(
                    (int(warp), int(lane), source_coords)
                )
                if source_register is None:
                    fail(
                        "TLXW_OP_BROADCAST",
                        STAGE,
                        "tt.broadcast result coordinate is not covered by the "
                        "source layout",
                        source_op_index=op.index,
                        source_value_id=result.value_id,
                    )
                source_registers.add(int(source_register))
        if len(source_registers) != 1:
            fail(
                "TLXW_OP_BROADCAST",
                STAGE,
                "tt.broadcast requires a component-invariant source mapping",
                source_op_index=op.index,
                source_value_id=result.value_id,
            )
        component_sources.append(next(iter(source_registers)))
    return tuple(int(source) for source in component_sources)


def _convert_program_id(builder, view):
    builder.add_op(
        "program_id",
        results=view.result_target_ids,
        attrs={"axis": _int_attr(view.attrs, "axis")},
        source_op_index=view.op_index,
    )


def _convert_if(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    if len(op.operands) != 1 or len(op.region_ids) != 2:
        fail(
            "TLXW_OP_UNSUPPORTED_IF",
            STAGE,
            "scf.if conversion requires one condition and then/else regions",
            source_op_index=op.index,
        )
    condition_targets = _operand_target_ids(builder, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    then_yields = _convert_region(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        op.region_ids[0],
        allow_yield=True,
    )
    else_yields = _convert_region(
        builder,
        conversion_input,
        type_layout_program,
        fact_program,
        op.region_ids[1],
        allow_yield=True,
    )
    if len(then_yields) != len(result_target_ids) or len(else_yields) != len(result_target_ids):
        fail(
            "TLXW_OP_IF_YIELD_MISMATCH",
            STAGE,
            "scf.if yield counts must match result count",
            source_op_index=op.index,
        )
    for index, result_target_id in enumerate(result_target_ids):
        true_target_id = _single_source_target(builder, then_yields[index], op)
        false_target_id = _single_source_target(builder, else_yields[index], op)
        builder.add_op(
            "select",
            operands=(condition_targets[0], true_target_id, false_target_id),
            results=(result_target_id,),
            layout_map_ids=result_layout_map_ids,
            source_op_index=op.index,
        )


def _convert_for(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    if len(op.region_ids) != 1 or len(op.operands) < 3:
        fail(
            "TLXW_OP_UNSUPPORTED_FOR",
            STAGE,
            "scf.for conversion requires lower, upper, step, and one body region",
            source_op_index=op.index,
        )
    data_init_arg_count = len(op.operands) - 3
    if len(op.results) != data_init_arg_count:
        fail(
            "TLXW_OP_FOR_RESULT_MISMATCH",
            STAGE,
            "scf.for result count must match iter_args count",
            source_op_index=op.index,
        )
    source_region = conversion_input.regions[op.region_ids[0]]
    if len(source_region.block_arg_ids) != 1 + data_init_arg_count:
        fail(
            "TLXW_OP_FOR_REGION_ARGS",
            STAGE,
            "scf.for body must have induction variable plus iter_arg block args",
            source_op_index=op.index,
        )

    token_carries = _loop_token_carries(conversion_input, op)
    source_loop_operands = _operand_target_ids(builder, op)
    token_init_target_ids = tuple(
        _loop_token_init_target_id(
            builder,
            type_layout_program,
            op,
            carry,
        )
        for carry in token_carries
    )
    loop_operands = (*source_loop_operands, *token_init_target_ids)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    token_result_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(
                type_layout_program.values[carry["yield_source_value_id"]].type
            ),
            debug_name=f"loop_token_result_{op.index}_{index}",
        )
        for index, carry in enumerate(token_carries)
    )
    result_target_ids = (*result_target_ids, *token_result_target_ids)
    block_arg_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(
                type_layout_program.values[source_value_id].type
            ),
            source_value_id=source_value_id,
            debug_name=f"r{op.region_ids[0]}_arg{index}",
        )
        for index, source_value_id in enumerate(source_region.block_arg_ids)
    )
    token_block_arg_target_ids = tuple(
        builder.add_value(
            target_ir.target_type_from_converted(
                type_layout_program.values[carry["yield_source_value_id"]].type
            ),
            debug_name=f"loop_token_arg_{op.index}_{index}",
        )
        for index, carry in enumerate(token_carries)
    )
    block_arg_target_ids = (*block_arg_target_ids, *token_block_arg_target_ids)
    target_region_id = builder.add_region(block_arg_ids=block_arg_target_ids)
    token_issue_dependency_pairs = _loop_token_carry_issue_dependencies(
        token_carries,
        token_block_arg_target_ids,
    )
    issue_dependencies = _loop_async_issue_dependencies(
        conversion_input,
        tuple(carry for carry, _token_block_arg_target_id in token_issue_dependency_pairs),
        tuple(token_block_arg_target_id for _carry, token_block_arg_target_id in token_issue_dependency_pairs),
    )
    body_conversion_input = replace(
        conversion_input,
        async_issue_dependency_target_ids_by_op={
            **conversion_input.async_issue_dependency_target_ids_by_op,
            **issue_dependencies,
        },
    )
    saved_token_targets = _replace_source_targets(
        builder,
        tuple(
            (carry["init_source_value_id"], token_block_arg_target_id)
            for carry, token_block_arg_target_id in zip(
                token_carries,
                token_block_arg_target_ids,
            )
            if carry.get("init_source_value_id") is not None
        ),
    )
    with builder.insertion_region(target_region_id):
        try:
            yielded_source_values = _convert_region(
                builder,
                body_conversion_input,
                type_layout_program,
                fact_program,
                op.region_ids[0],
                allow_yield=True,
            )
        finally:
            _restore_source_targets(builder, saved_token_targets)
    if len(yielded_source_values) != data_init_arg_count:
        fail(
            "TLXW_OP_FOR_YIELD_MISMATCH",
            STAGE,
            "scf.for yield count must match iter_args count",
            source_op_index=op.index,
        )
    yielded_target_ids = tuple(
        _single_source_target(builder, source_value_id, op)
        for source_value_id in yielded_source_values
    )
    yielded_token_target_ids = tuple(
        _single_source_target(builder, carry["yield_source_value_id"], op)
        for carry in token_carries
    )
    builder.set_region_yields(
        target_region_id,
        (*yielded_target_ids, *yielded_token_target_ids),
    )
    builder.add_op(
        "for_loop",
        operands=loop_operands,
        results=result_target_ids,
        attrs={
            "init_arg_count": data_init_arg_count + len(token_carries),
            "source_result_count": data_init_arg_count,
        },
        layout_map_ids=result_layout_map_ids,
        region_ids=(target_region_id,),
        source_op_index=op.index,
    )
    _replace_source_targets(
        builder,
        tuple(
            (carry["yield_source_value_id"], token_result_target_id)
            for carry, token_result_target_id in zip(
                token_carries,
                token_result_target_ids,
            )
        ),
    )


def _loop_token_carries(conversion_input, op):
    body_op_indices = _region_op_indices_recursive(conversion_input, op.region_ids[0])
    waited_external_tokens = []
    for node in conversion_input.token_nodes_by_op.values():
        if node.op_index not in body_op_indices or node.op_name != "ttg.async_wait":
            continue
        for group_id in node.waited_group_ids:
            group = conversion_input.token_groups_by_id[group_id]
            if group.commit_op_index in body_op_indices or group.token_value_id is None:
                continue
            waited_external_tokens.append(group.token_value_id)
    committed_body_tokens = tuple(
        group.token_value_id
        for group in sorted(
            conversion_input.token_groups_by_commit.values(),
            key=lambda group: group.commit_op_index,
        )
        if group.commit_op_index in body_op_indices and group.token_value_id is not None
    )
    waited_external_tokens = _dedupe_preserving_order(waited_external_tokens)
    externally_waited_body_tokens = _externally_waited_body_tokens(
        conversion_input,
        body_op_indices,
    )
    if waited_external_tokens:
        if len(waited_external_tokens) != 1 or len(committed_body_tokens) != 1:
            fail(
                "TLXW_OP_UNSUPPORTED_FOR_TOKENS",
                STAGE,
                "scf.for async token carry supports one external waited group "
                "and one body commit group",
                source_op_index=op.index,
            )
        return (
            {
                "init_source_value_id": waited_external_tokens[0],
                "yield_source_value_id": committed_body_tokens[0],
                "add_issue_dependency": True,
            },
        )
    return tuple(
        {
            "init_source_value_id": None,
            "yield_source_value_id": body_token,
            "add_issue_dependency": False,
        }
        for body_token in externally_waited_body_tokens
    )


def _externally_waited_body_tokens(conversion_input, body_op_indices):
    body_tokens = []
    for node in sorted(
        conversion_input.token_nodes_by_op.values(),
        key=lambda node: node.op_index,
    ):
        if node.op_index in body_op_indices or node.op_name != "ttg.async_wait":
            continue
        for group_id in node.waited_group_ids:
            group = conversion_input.token_groups_by_id[group_id]
            if group.commit_op_index not in body_op_indices or group.token_value_id is None:
                continue
            body_tokens.append(group.token_value_id)
    return _dedupe_preserving_order(body_tokens)


def _loop_token_init_target_id(builder, type_layout_program, op, carry):
    init_source_value_id = carry.get("init_source_value_id")
    if init_source_value_id is not None:
        return _single_source_target(builder, init_source_value_id, op)
    yield_source_value_id = carry["yield_source_value_id"]
    token_target_id = builder.add_value(
        target_ir.target_type_from_converted(
            type_layout_program.values[yield_source_value_id].type
        ),
        debug_name=f"loop_token_init_{op.index}",
    )
    builder.add_op(
        "token",
        results=(token_target_id,),
        source_op_index=op.index,
    )
    return token_target_id


def _loop_token_carry_issue_dependencies(token_carries, token_block_arg_target_ids):
    return tuple(
        (carry, token_block_arg_target_id)
        for carry, token_block_arg_target_id in zip(
            token_carries,
            token_block_arg_target_ids,
        )
        if carry.get("add_issue_dependency", True)
    )


def _loop_async_issue_dependencies(
    conversion_input,
    token_carries,
    token_block_arg_target_ids,
):
    dependencies_by_op = {}
    for carry, token_block_arg_target_id in zip(token_carries, token_block_arg_target_ids):
        body_group = _token_group_by_value_id(
            conversion_input,
            carry["yield_source_value_id"],
        )
        if body_group is None:
            continue
        for member_token_id in body_group.member_token_ids:
            member_node = _token_node_by_value_id(conversion_input, member_token_id)
            if member_node is None:
                continue
            existing = dependencies_by_op.setdefault(member_node.op_index, tuple())
            dependencies_by_op[member_node.op_index] = (
                *existing,
                int(token_block_arg_target_id),
            )
    return dependencies_by_op


def _token_group_by_value_id(conversion_input, value_id):
    for group in conversion_input.token_groups_by_commit.values():
        if group.token_value_id == value_id:
            return group
    return None


def _token_node_by_value_id(conversion_input, value_id):
    for node in conversion_input.token_nodes_by_op.values():
        if node.value_id == value_id:
            return node
    return None


def _region_op_indices_recursive(conversion_input, region_id):
    result = []
    for op_index in conversion_input.regions[region_id].op_indices:
        result.append(op_index)
        for child_region_id in conversion_input.ops[op_index].region_ids:
            result.extend(_region_op_indices_recursive(conversion_input, child_region_id))
    return frozenset(result)


def _replace_source_targets(builder, replacements):
    saved = {}
    for source_value_id, target_value_id in replacements:
        source_value_id = int(source_value_id)
        saved[source_value_id] = builder.source_value_targets.get(source_value_id)
        builder.source_value_targets[source_value_id] = (int(target_value_id),)
    return saved


def _restore_source_targets(builder, saved):
    for source_value_id, targets in saved.items():
        if targets is None:
            builder.source_value_targets.pop(source_value_id, None)
        else:
            builder.source_value_targets[source_value_id] = targets


def _dedupe_preserving_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _convert_local_alloc(
    builder,
    conversion_input,
    type_layout_program,
    op,
):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_LOCAL_ALLOC_RESULT",
            STAGE,
            "ttg.local_alloc must produce one memdesc result",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    memdesc = _memdesc_info(conversion_input, op.results[0], op)
    shape = tuple(memdesc.shape or memdesc.alloc_shape)
    builder.add_op(
        "local_alloc",
        results=result_target_ids,
        attrs={
            "allocation_bytes": int(
                conversion_input.local_alloc_allocation_bytes.get(
                    op.results[0],
                    memdesc.allocation_bytes,
                )
            ),
            "byte_offset": int(conversion_input.local_alloc_byte_offsets[op.results[0]]),
            "element_type": memdesc.element_type,
            "shape": tuple(int(dim) for dim in shape),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_memdesc_index(
    builder,
    conversion_input,
    type_layout_program,
    op,
):
    if len(op.operands) != 2 or len(op.results) != 1:
        fail(
            "TLXW_OP_MEMDESC_INDEX",
            STAGE,
            "ttg.memdesc_index requires memdesc/index operands and one result",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    memdesc = _memdesc_info(conversion_input, op.results[0], op)
    slot_size_bytes = int(
        conversion_input.memdesc_physical_allocation_bytes.get(
            op.results[0],
            memdesc.allocation_bytes,
        )
    )
    element_byte_width = memdesc.element_byte_width
    if element_byte_width is None or slot_size_bytes % int(element_byte_width):
        fail(
            "TLXW_OP_MEMDESC_INDEX",
            STAGE,
            "ttg.memdesc_index slot size is not element aligned",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    element_count = slot_size_bytes // int(element_byte_width)
    static_lds_byte_offset = conversion_input.static_memdesc_byte_offsets.get(
        op.results[0]
    )
    builder.add_op(
        "memdesc_index",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "element_byte_width": memdesc.element_byte_width,
            "elements_per_slot": element_count,
            "static_lds_byte_offset": static_lds_byte_offset,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_buffer_load_to_local(
    builder,
    conversion_input,
    type_layout_program,
    fact_program,
    op,
):
    fields = _buffer_load_to_local_fields(op)
    if fields["other_value_id"] is not None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local other fallback is not converted yet",
            source_op_index=op.index,
        )
    if fields["stride_value_id"] is not None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local stride operand is not converted yet",
            source_op_index=op.index,
        )
    _require_default_cache(fields["cache"], op)
    token_node = conversion_input.token_nodes_by_op.get(op.index)
    if token_node is None or token_node.value_id not in op.results:
        fail(
            "TLXW_OP_BUFFER_ASYNC_TOKEN",
            STAGE,
            "amdg.buffer_load_to_local requires a token graph node",
            source_op_index=op.index,
        )
    range_fact = _pointer_byte_range_fact(
        fact_program,
        fields["base_value_id"],
        op,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    base_target_id = _single_source_target(builder, fields["base_value_id"], op)
    issue_dependency_target_ids = (
        conversion_input.async_issue_dependency_target_ids_by_op.get(op.index, ())
    )
    operands = [
        _single_source_target(builder, fields["memdesc_value_id"], op),
        base_target_id,
        _single_source_target(builder, fields["offset_value_id"], op),
    ]
    memdesc = _memdesc_info(conversion_input, fields["memdesc_value_id"], op)
    if memdesc.element_byte_width is None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local DMA requires known element byte width",
            source_op_index=op.index,
            source_value_id=fields["memdesc_value_id"],
        )
    offset_type = type_layout_program.values[fields["offset_value_id"]].type
    if offset_type.representation not in {"simd", "simd_tuple"}:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local requires SIMD offset components",
            source_op_index=op.index,
            source_value_id=fields["offset_value_id"],
        )
    has_mask = fields["mask_value_id"] is not None
    if has_mask:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) not in (1, int(offset_type.component_count)):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "amdg.buffer_load_to_local mask must be scalar or match "
                "offset components",
                source_op_index=op.index,
                source_value_id=fields["mask_value_id"],
            )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    operands.extend(issue_dependency_target_ids)
    packet_plan = None
    if not has_mask:
        packet_plan = _buffer_load_to_local_packet_plan(
            conversion_input,
            type_layout_program,
            fact_program,
            fields["memdesc_value_id"],
            fields["offset_value_id"],
            memdesc,
            int(offset_type.lane_width or conversion_input.threads_per_warp),
            op,
        )
    if packet_plan is not None:
        scalar_target_ids = tuple(
            _single_source_target(builder, source_value_id, op)
            for source_value_id in packet_plan["scalar_value_ids"]
        )
        builder.add_op(
            "buffer_load_to_local",
            operands=(operands[0], operands[1], *scalar_target_ids, *issue_dependency_target_ids),
            results=result_target_ids,
            attrs={
                "cache_modifier": int(fields["cache"] or 1),
                "component_count": int(packet_plan["component_count"]),
                "component_thread_count": int(packet_plan["component_thread_count"]),
                "destination_component_offsets": tuple(
                    packet_plan["destination_component_offsets"]
                ),
                "destination_wave_count": int(packet_plan["destination_wave_count"]),
                "destination_wave_stride_dwords": int(
                    packet_plan["destination_wave_stride_dwords"]
                ),
                "element_byte_width": int(memdesc.element_byte_width),
                "element_type": memdesc.element_type,
                "lane_width": int(
                    offset_type.lane_width or conversion_input.threads_per_warp
                ),
                "mode": "dma_packet_lds",
                "packet_bytes": int(packet_plan["packet_bytes"]),
                "packet_elements": int(packet_plan["packet_elements"]),
                "range_bytes": int(range_fact.upper),
                "source_offset_range": (
                    0,
                    _buffer_source_offset_upper(
                        range_fact.upper,
                        packet_plan["packet_bytes"],
                        memdesc.element_byte_width,
                        op,
                    ),
                ),
                "source_offset_terms": tuple(packet_plan["source_offset_terms"]),
                "source_rank": len(memdesc.shape),
                "source_scalar_count": len(scalar_target_ids),
                "source_shape": tuple(int(dim) for dim in memdesc.shape),
                "issue_dependency_count": len(issue_dependency_target_ids),
            },
            fact_ids=(range_fact.fact_id,),
            fact_target_ids=(base_target_id,),
            layout_map_ids=result_layout_map_ids,
            source_op_index=op.index,
        )
        return
    destination_plan = _local_component_store_plan(
        conversion_input,
        type_layout_program,
        fields["memdesc_value_id"],
        fields["offset_value_id"],
        int(offset_type.component_count),
        int(offset_type.lane_width or conversion_input.threads_per_warp),
        op,
    )
    scalar_offset_upper = _buffer_source_offset_upper(
        range_fact.upper,
        memdesc.element_byte_width,
        memdesc.element_byte_width,
        op,
    )
    builder.add_op(
        "buffer_load_to_local",
        operands=tuple(operands),
        results=result_target_ids,
        attrs={
            "cache_modifier": int(fields["cache"] or 1),
            "component_count": int(offset_type.component_count),
            **_local_component_store_plan_attrs(destination_plan),
            "element_byte_width": int(memdesc.element_byte_width),
            "element_type": memdesc.element_type,
            "has_mask": has_mask,
            "lane_width": int(
                offset_type.lane_width or conversion_input.threads_per_warp
            ),
            "mask_mode": "exec_where" if has_mask else "none",
            "offset_range": (0, int(scalar_offset_upper)),
            "mode": "scalarized_load_store",
            "range_bytes": int(range_fact.upper),
            "issue_dependency_count": len(issue_dependency_target_ids),
        },
        fact_ids=(range_fact.fact_id,),
        fact_target_ids=(base_target_id,),
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _local_component_store_plan_attrs(destination_plan):
    offset_mode = destination_plan["offset_mode"]
    if offset_mode == "affine":
        return {
            "destination_offset_mode": "affine",
            "destination_component_offsets": tuple(
                destination_plan["component_offsets"]
            ),
            "destination_lane_stride_elements": int(
                destination_plan["lane_stride_elements"]
            ),
            "destination_wave_stride_elements": int(
                destination_plan["wave_stride_elements"]
            ),
        }
    if offset_mode == "layout_coordinates":
        return {
            "destination_offset_mode": "layout_coordinates",
            "destination_coordinate_shape": tuple(
                int(dim) for dim in destination_plan["coordinate_shape"]
            ),
            "destination_component_coordinate_bases": tuple(
                tuple(int(value) for value in bases)
                for bases in destination_plan["component_coordinate_bases"]
            ),
            "destination_workitem_coordinate_coefficients": tuple(
                tuple(int(value) for value in coefficients)
                for coefficients in destination_plan[
                    "workitem_coordinate_coefficients"
                ]
            ),
            **destination_plan["shared_layout_attrs"],
        }
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"unsupported scalarized destination offset mode {offset_mode}",
    )


def _convert_buffer_load(builder, conversion_input, type_layout_program, fact_program, op):
    fields = _buffer_load_fields(op)
    _require_default_cache(fields["cache"], op)
    if fields["stride_value_id"] is not None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load stride operand is not converted yet",
            source_op_index=op.index,
        )
    if fields["other_value_id"] is not None and fields["mask_value_id"] is None:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load other operand requires a mask operand",
            source_op_index=op.index,
        )
    range_fact = _pointer_byte_range_fact(fact_program, fields["base_value_id"], op)
    loaded = type_layout_program.values[op.results[0]]
    offsets = type_layout_program.values[fields["offset_value_id"]]
    if int(loaded.type.component_count) != int(offsets.type.component_count):
        fail(
            "TLXW_OP_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load result and offset components must match",
            source_op_index=op.index,
        )
    base_target_id = _single_source_target(builder, fields["base_value_id"], op)
    operands = [
        base_target_id,
        _single_source_target(builder, fields["offset_value_id"], op),
    ]
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) != int(loaded.type.component_count):
            fail(
                "TLXW_OP_BUFFER_LOAD",
                STAGE,
                "amdg.buffer_load mask and result components must match",
                source_op_index=op.index,
            )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    if fields["other_value_id"] is not None:
        other = type_layout_program.values[fields["other_value_id"]]
        if int(other.type.component_count) not in (
            1,
            int(loaded.type.component_count),
        ):
            fail(
                "TLXW_OP_BUFFER_LOAD",
                STAGE,
                "amdg.buffer_load other must be scalar or match result components",
                source_op_index=op.index,
            )
        operands.append(_single_source_target(builder, fields["other_value_id"], op))
    element_byte_width = conversion_input.value_element_byte_widths.get(op.results[0])
    if element_byte_width is None:
        fail(
            "TLXW_OP_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load requires known result element byte width",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    access_element_count = int(fields["contiguity"] or 1)
    if access_element_count <= 0:
        fail(
            "TLXW_OP_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load contiguity must be positive",
            source_op_index=op.index,
        )
    access_bytes = int(element_byte_width) * access_element_count
    has_mask = fields["mask_value_id"] is not None
    has_other = fields["other_value_id"] is not None
    offset_upper = _buffer_source_offset_upper(
        range_fact.upper,
        access_bytes,
        element_byte_width,
        op,
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "buffer_load",
        operands=tuple(operands),
        results=result_target_ids,
        attrs={
            "access_element_count": access_element_count,
            "cache_modifier": int(fields["cache"] or 1),
            "component_count": int(loaded.type.component_count),
            "element_byte_width": int(element_byte_width),
            "element_type": loaded.type.element_type,
            "has_mask": has_mask,
            "has_other": has_other,
            "inactive_byte_offset": _buffer_inactive_byte_offset(),
            "inactive_offset": _buffer_inactive_element_offset(element_byte_width, op),
            "lane_width": int(loaded.type.lane_width or offsets.type.lane_width or 64),
            "mask_mode": "exec_where" if has_mask else "none",
            "offset_range": (0, int(offset_upper)),
            "range_bytes": int(range_fact.upper),
        },
        fact_ids=(range_fact.fact_id,),
        fact_target_ids=(base_target_id,),
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_buffer_store(builder, conversion_input, type_layout_program, fact_program, op):
    fields = _buffer_store_fields(op)
    _require_default_cache(fields["cache"], op)
    range_fact = _pointer_byte_range_fact(fact_program, fields["base_value_id"], op)
    value = type_layout_program.values[fields["value_value_id"]]
    offsets = type_layout_program.values[fields["offset_value_id"]]
    if int(value.type.component_count) != int(offsets.type.component_count):
        fail(
            "TLXW_OP_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store value and offset components must match",
            source_op_index=op.index,
        )
    base_target_id = _single_source_target(builder, fields["base_value_id"], op)
    operands = [
        _single_source_target(builder, fields["value_value_id"], op),
        base_target_id,
        _single_source_target(builder, fields["offset_value_id"], op),
    ]
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) != int(value.type.component_count):
            fail(
                "TLXW_OP_BUFFER_STORE",
                STAGE,
                "amdg.buffer_store mask and value components must match",
                source_op_index=op.index,
            )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    element_byte_width = conversion_input.value_element_byte_widths.get(
        fields["value_value_id"]
    )
    if element_byte_width is None:
        fail(
            "TLXW_OP_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store requires known value element byte width",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    access_element_count = int(fields["contiguity"] or 1)
    if access_element_count <= 0:
        fail(
            "TLXW_OP_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store contiguity must be positive",
            source_op_index=op.index,
        )
    access_bytes = int(element_byte_width) * access_element_count
    has_mask = fields["mask_value_id"] is not None
    offset_upper = _buffer_source_offset_upper(
        range_fact.upper,
        access_bytes,
        element_byte_width,
        op,
    )
    builder.add_op(
        "buffer_store",
        operands=tuple(operands),
        attrs={
            "access_element_count": access_element_count,
            "cache_modifier": int(fields["cache"] or 1),
            "component_count": int(value.type.component_count),
            "element_byte_width": int(element_byte_width),
            "element_type": value.type.element_type,
            "has_mask": has_mask,
            "inactive_byte_offset": _buffer_inactive_byte_offset(),
            "inactive_offset": _buffer_inactive_element_offset(element_byte_width, op),
            "lane_width": int(value.type.lane_width or offsets.type.lane_width or 64),
            "mask_mode": "select_oob_offset" if has_mask else "none",
            "offset_range": (0, int(offset_upper)),
            "range_bytes": int(range_fact.upper),
        },
        fact_ids=(range_fact.fact_id,),
        fact_target_ids=(base_target_id,),
        source_op_index=op.index,
    )


def _convert_load(builder, conversion_input, type_layout_program, op):
    del conversion_input
    fields = _load_fields(op)
    _require_default_tt_memory_attrs(op)
    pointer = type_layout_program.values[fields["pointer_value_id"]]
    loaded = type_layout_program.values[op.results[0]]
    if pointer.type.representation not in {"per_lane_pointer", "pointer_tuple"}:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load requires a tensor pointer operand",
            source_op_index=op.index,
            source_value_id=fields["pointer_value_id"],
        )
    if loaded.type.representation not in {"simd", "simd_tuple"}:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load requires a tensor result",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    if pointer.type.element_type != loaded.type.element_type:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load pointer/result element types must match",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    component_count = int(loaded.type.component_count)
    if int(pointer.type.component_count) != component_count:
        fail(
            "TLXW_OP_LOAD",
            STAGE,
            "tt.load pointer/result component counts must match",
            source_op_index=op.index,
        )
    operands = [_single_source_target(builder, fields["pointer_value_id"], op)]
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) not in (1, component_count):
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load mask must be scalar or match result components",
                source_op_index=op.index,
                source_value_id=fields["mask_value_id"],
            )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    if fields["other_value_id"] is not None:
        if fields["mask_value_id"] is None:
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other operand requires a mask operand",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        other = type_layout_program.values[fields["other_value_id"]]
        if other.type.representation not in {"scalar", "simd", "simd_tuple"}:
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other requires a scalar or tensor value operand",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        if other.type.element_type != loaded.type.element_type:
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other/result element types must match",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        if int(other.type.component_count) not in (1, component_count):
            fail(
                "TLXW_OP_LOAD",
                STAGE,
                "tt.load other must be scalar or match result components",
                source_op_index=op.index,
                source_value_id=fields["other_value_id"],
            )
        operands.append(_single_source_target(builder, fields["other_value_id"], op))
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "load",
        operands=tuple(operands),
        results=result_target_ids,
        attrs={
            "component_count": component_count,
            "element_type": loaded.type.element_type,
            "has_mask": fields["mask_value_id"] is not None,
            "has_other": fields["other_value_id"] is not None,
            "lane_width": int(loaded.type.lane_width or pointer.type.lane_width or 64),
            "mask_mode": "exec_where" if fields["mask_value_id"] is not None else "none",
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_store(builder, conversion_input, type_layout_program, op):
    del conversion_input
    fields = _store_fields(op)
    _require_default_tt_memory_attrs(op)
    pointer = type_layout_program.values[fields["pointer_value_id"]]
    value = type_layout_program.values[fields["value_value_id"]]
    if pointer.type.representation not in {"per_lane_pointer", "pointer_tuple"}:
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store requires a tensor pointer operand",
            source_op_index=op.index,
            source_value_id=fields["pointer_value_id"],
        )
    if value.type.representation not in {"scalar", "simd", "simd_tuple"}:
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store requires a scalar or tensor value operand",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    if pointer.type.element_type != value.type.element_type:
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store pointer/value element types must match",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    component_count = int(pointer.type.component_count)
    if int(value.type.component_count) not in (1, component_count):
        fail(
            "TLXW_OP_STORE",
            STAGE,
            "tt.store value must be scalar or match pointer components",
            source_op_index=op.index,
            source_value_id=fields["value_value_id"],
        )
    operands = [
        _single_source_target(builder, fields["pointer_value_id"], op),
        _single_source_target(builder, fields["value_value_id"], op),
    ]
    if fields["mask_value_id"] is not None:
        mask = type_layout_program.values[fields["mask_value_id"]]
        if int(mask.type.component_count) not in (1, component_count):
            fail(
                "TLXW_OP_STORE",
                STAGE,
                "tt.store mask must be scalar or match pointer components",
                source_op_index=op.index,
                source_value_id=fields["mask_value_id"],
            )
        operands.append(_single_source_target(builder, fields["mask_value_id"], op))
    builder.add_op(
        "store",
        operands=tuple(operands),
        attrs={
            "component_count": component_count,
            "element_type": pointer.type.element_type,
            "has_mask": fields["mask_value_id"] is not None,
            "lane_width": int(pointer.type.lane_width or value.type.lane_width or 64),
            "mask_mode": "exec_where" if fields["mask_value_id"] is not None else "none",
        },
        source_op_index=op.index,
    )


def _convert_local_load(builder, conversion_input, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_LOCAL_LOAD",
            STAGE,
            "ttg.local_load requires one memdesc operand and one result",
            source_op_index=op.index,
        )
    result_value_id = op.results[0]
    result = type_layout_program.values[result_value_id]
    result_layout = (
        None
        if result.layout_map_id is None
        else type_layout_program.layouts[int(result.layout_map_id)]
    )
    if result_layout is None or result_layout.kind != "dot_operand":
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "only dot-operand ttg.local_load is converted yet",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    memdesc_value_id = op.operands[0]
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    registers = _fragment_registers(memdesc.element_type, result_layout, op)
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    fragment_rows, fragment_columns = _operand_fragment_shape(instr_shape, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    load_plan = _fragment_local_load_plan(
        conversion_input,
        type_layout_program,
        memdesc_value_id,
        result_layout,
        int(result.type.component_count),
        registers,
        op,
    )
    builder.add_op(
        "local_load_fragment",
        operands=(_single_source_target(builder, memdesc_value_id, op),),
        results=result_target_ids,
        attrs={
            "columns": int(fragment_columns),
            "component_count": int(result.type.component_count),
            "element_type": memdesc.element_type,
            "lane_width": int(result.type.lane_width or 64),
            "registers": int(registers),
            "role": int(result_layout.properties["op_idx"]),
            "rows": int(fragment_rows),
            **load_plan,
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_fragment_constant(builder, type_layout_program, op):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_FRAGMENT_CONSTANT",
            STAGE,
            "fragment constants must have one result",
            source_op_index=op.index,
        )
    result = type_layout_program.values[op.results[0]]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if result_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_FRAGMENT_CONSTANT",
            STAGE,
            "only amd_mfma accumulator constants are converted as fragments",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    value = _constant_literal(op.attrs.get("value"), source_op_index=op.index)
    if not _is_zero_literal(value):
        fail(
            "TLXW_OP_FRAGMENT_CONSTANT",
            STAGE,
            "only zero fragment constants are converted yet",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    registers = _acc_fragment_registers(result_layout, op)
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    fragment_rows, fragment_columns = _acc_fragment_shape(instr_shape, op)
    builder.add_op(
        "fragment_fill",
        results=result_target_ids,
        attrs={
            "columns": int(fragment_columns),
            "component_count": int(result.type.component_count),
            "element_type": result_layout.element_type,
            "fill_value": 0,
            "lane_width": int(result.type.lane_width or 64),
            "registers": int(registers),
            "role": 2,
            "rows": int(fragment_rows),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_dot(builder, type_layout_program, op):
    if len(op.operands) != 3 or len(op.results) != 1:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot requires lhs, rhs, accumulator, and one result",
            source_op_index=op.index,
        )
    lhs = type_layout_program.values[op.operands[0]]
    rhs = type_layout_program.values[op.operands[1]]
    acc = type_layout_program.values[op.operands[2]]
    result = type_layout_program.values[op.results[0]]
    result_layout = type_layout_program.layouts[int(result.layout_map_id)]
    if result_layout.kind != "amd_mfma":
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot result must use an amd_mfma layout",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    if instr_shape not in {(16, 16, 32), (32, 32, 16)}:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            f"unsupported MFMA instruction shape {instr_shape}",
            source_op_index=op.index,
            source_value_id=op.results[0],
        )
    if lhs.type.element_type != rhs.type.element_type:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot lhs/rhs element types must match",
            source_op_index=op.index,
        )
    kind = _mma_kind(lhs.type.element_type, instr_shape, op)
    warps_per_cta = tuple(result_layout.properties.get("warps_per_cta", ()))
    m_tiles, n_tiles = _mfma_per_wave_tiles(result_layout, instr_shape, warps_per_cta, op)
    acc_layout = _require_layout(type_layout_program, acc.layout_map_id, op)
    if not _same_layout_alias(acc, result, acc_layout, result_layout):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot accumulator layout must match the result layout",
            source_op_index=op.index,
            source_value_id=op.operands[2],
        )
    lhs_layout = _require_layout(type_layout_program, lhs.layout_map_id, op)
    rhs_layout = _require_layout(type_layout_program, rhs.layout_map_id, op)
    _require_dot_operand_layout(lhs_layout, 0, op)
    _require_dot_operand_layout(rhs_layout, 1, op)
    lhs_k_tiles = _dot_operand_k_tiles(lhs_layout, instr_shape, op)
    rhs_k_tiles = _dot_operand_k_tiles(rhs_layout, instr_shape, op)
    if lhs_k_tiles != rhs_k_tiles:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot lhs/rhs K tile counts do not match",
            source_op_index=op.index,
        )
    k_tiles = lhs_k_tiles
    if (
        int(lhs.type.component_count) != m_tiles * k_tiles
        or int(rhs.type.component_count) != n_tiles * k_tiles
    ):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot operand fragment component counts do not match "
            "the result MFMA tile grid",
            source_op_index=op.index,
        )
    if int(acc.type.component_count) != m_tiles * n_tiles:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot accumulator component count does not match "
            "the result MFMA tile grid",
            source_op_index=op.index,
        )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "mma",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "kind": kind,
            "k_tiles": int(k_tiles),
            "m_tiles": int(m_tiles),
            "n_tiles": int(n_tiles),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_fragment_truncf(builder, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_FRAGMENT_TRUNCF",
            STAGE,
            "fragment arith.truncf requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    if operand.type.element_type != "f32" or result.type.element_type != "f16":
        fail(
            "TLXW_OP_FRAGMENT_TRUNCF",
            STAGE,
            "only f32 to f16 fragment truncf is converted yet",
            source_op_index=op.index,
        )
    if int(operand.type.component_count) != int(result.type.component_count):
        fail(
            "TLXW_OP_FRAGMENT_TRUNCF",
            STAGE,
            "fragment truncf component counts must match",
            source_op_index=op.index,
        )
    operand_layout = type_layout_program.layouts[int(operand.layout_map_id)]
    registers = _acc_fragment_registers(operand_layout, op)
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    builder.add_op(
        "fragment_truncf",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs={
            "component_count": int(result.type.component_count),
            "lane_width": int(result.type.lane_width or 64),
            "registers": int(registers),
        },
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _convert_layout(builder, conversion_input, type_layout_program, op):
    if len(op.operands) != 1 or len(op.results) != 1:
        fail(
            "TLXW_OP_CONVERT_LAYOUT",
            STAGE,
            "ttg.convert_layout requires one operand and one result",
            source_op_index=op.index,
        )
    operand = type_layout_program.values[op.operands[0]]
    result = type_layout_program.values[op.results[0]]
    operand_layout = (
        None
        if operand.layout_map_id is None
        else type_layout_program.layouts[int(operand.layout_map_id)]
    )
    result_layout = (
        None
        if result.layout_map_id is None
        else type_layout_program.layouts[int(result.layout_map_id)]
    )
    result_target_ids, result_layout_map_ids = _declare_results(
        builder,
        op,
        type_layout_program,
    )
    same_layout = _same_layout_alias(operand, result, operand_layout, result_layout)
    register_remap = None
    distributed_remap = None
    dot_operand_remap = None
    mfma_base_remap = None
    if not same_layout:
        register_remap = layout_remap.register_remap(
            operand,
            result,
            operand_layout,
            result_layout,
            op,
        )
        if register_remap is None:
            distributed_remap = layout_remap.distributed_remap(
                operand,
                result,
                operand_layout,
                result_layout,
                op,
            )
        if (
            register_remap is None
            and distributed_remap is None
            and dot_operand_remap is None
            and operand.type.element_type in {"bf16", "f16", "f32"}
            and int(result.type.component_count) == 1
        ):
            candidate = layout_remap.distributed_to_mfma_base_remap(
                operand,
                result,
                operand_layout,
                result_layout,
                op,
            )
            if candidate is not None and candidate.get("mode") != "cta_exchange_register_remap":
                mfma_base_remap = candidate
        if register_remap is None and distributed_remap is None:
            dot_operand_remap = layout_remap.dot_operand_fragment_pack(
                operand,
                result,
                operand_layout,
                result_layout,
                op,
            )
    if same_layout:
        mode = "alias"
        attrs = {
            "fact_policy": "preserve_equivalent",
            "group_size": 1,
            "mode": mode,
            "result_component_count": int(result.type.component_count),
        }
    elif (
        register_remap is not None
        or distributed_remap is not None
        or dot_operand_remap is not None
        or mfma_base_remap is not None
    ):
        if (
            operand.type.representation in {"fragment", "fragment_tuple"}
            and operand.type.element_type == "f32"
        ):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "fragment-backed f32 MFMA convert_layout requires an "
                "explicit fragment unpack before register remap",
                source_op_index=op.index,
                source_value_id=operand.value_id,
            )
        remap = (
            register_remap
            if register_remap is not None
            else distributed_remap
            if distributed_remap is not None
            else dot_operand_remap
            if dot_operand_remap is not None
            else mfma_base_remap
        )
        attrs = {
            "fact_policy": "invalidate_layout_sensitive",
            "result_component_count": int(result.type.component_count),
            **remap,
        }
        attrs = _add_layout_remap_scratch_attrs(
            attrs,
            conversion_input,
            result,
            op,
        )
    elif (
        operand_layout is not None
        and result_layout is not None
        and operand_layout.kind == "blocked"
        and result_layout.kind == "amd_mfma"
        and int(operand.type.component_count) % int(result.type.component_count) == 0
    ):
        mode = "component_group_first"
        group_size = int(operand.type.component_count) // int(result.type.component_count)
        attrs = {
            "fact_policy": "invalidate_layout_sensitive",
            "group_size": int(group_size),
            "mode": mode,
            "result_component_count": int(result.type.component_count),
        }
    else:
        layout_remap.reject_unsupported_pair(operand_layout, result_layout, op)
    builder.add_op(
        "layout_convert",
        operands=_operand_target_ids(builder, op),
        results=result_target_ids,
        attrs=attrs,
        layout_map_ids=result_layout_map_ids,
        source_op_index=op.index,
    )


def _add_layout_remap_scratch_attrs(attrs, conversion_input, result, op):
    if "scratch_element_count" not in attrs:
        return attrs
    element_byte_width = conversion_input.value_element_byte_widths.get(
        result.value_id
    )
    if element_byte_width is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "CTA exchange layout remap requires a known element byte width",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    scratch_elements = int(attrs["scratch_element_count"])
    if scratch_elements <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "CTA exchange layout remap produced an empty scratch allocation",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    scratch_byte_offset = _align_to(conversion_input.lds_size, 16)
    scratch_bytes = _align_to(scratch_elements * int(element_byte_width), 16)
    return {
        **attrs,
        "scratch_allocation_bytes": int(scratch_bytes),
        "scratch_byte_offset": int(scratch_byte_offset),
    }


def _same_layout_alias(operand, result, operand_layout, result_layout):
    if int(operand.type.component_count) != int(result.type.component_count):
        return False
    if operand.type.element_type != result.type.element_type:
        return False
    if operand_layout is None or result_layout is None:
        return operand_layout is result_layout
    return (
        operand_layout.kind == result_layout.kind
        and tuple(operand_layout.shape) == tuple(result_layout.shape)
        and operand_layout.element_type == result_layout.element_type
        and operand_layout.properties == result_layout.properties
    )


def _convert_async_commit_group(builder, type_layout_program, token_groups_by_commit, op):
    group = token_groups_by_commit.get(op.index)
    if group is None:
        fail(
            "TLXW_OP_ASYNC_COMMIT_TOKEN",
            STAGE,
            "ttg.async_commit_group requires a token group",
            source_op_index=op.index,
        )
    result_target_ids, _ = _declare_results(builder, op, type_layout_program)
    operands = tuple(
        _single_source_target(builder, token_value_id, op)
        for token_value_id in group.member_token_ids
    )
    builder.add_op(
        "async_commit_group",
        operands=operands,
        results=result_target_ids,
        attrs={
            "group_id": int(group.group_id),
            "member_count": len(group.member_token_ids),
        },
        source_op_index=op.index,
    )


def _convert_async_wait(
    builder,
    type_layout_program,
    token_nodes_by_op,
    token_groups_by_id,
    op,
):
    node = token_nodes_by_op.get(op.index)
    if node is None:
        fail(
            "TLXW_OP_ASYNC_WAIT_TOKEN",
            STAGE,
            "ttg.async_wait requires a token graph node",
            source_op_index=op.index,
        )
    result_target_ids, _ = _declare_results(builder, op, type_layout_program)
    if node.input_token_ids:
        wait_token_ids = node.input_token_ids
    else:
        wait_token_ids = tuple(
            token_groups_by_id[group_id].token_value_id
            for group_id in node.waited_group_ids
            if token_groups_by_id[group_id].token_value_id is not None
        )
    operands = tuple(
        _single_source_target(builder, token_value_id, op)
        for token_value_id in wait_token_ids
    )
    builder.add_op(
        "async_wait",
        operands=operands,
        results=result_target_ids,
        attrs={
            "wait_group": -1 if node.wait_group is None else int(node.wait_group),
            "waited_group_ids": tuple(int(group_id) for group_id in node.waited_group_ids),
        },
        source_op_index=op.index,
    )


def _convert_return(builder, view):
    builder.add_op(
        "return",
        operands=view.operand_target_ids,
        source_op_index=view.op_index,
    )


_SIMPLE_OP_CONVERTERS = {
    "arith.constant": _convert_constant,
    **{op_name: _convert_binary for op_name in _BINARY_OPS},
    **{op_name: _convert_float_binary for op_name in _FLOAT_BINARY_OPS},
    "arith.cmpi": _convert_cmpi,
    "arith.minsi": _convert_minsi,
    "llvm.intr.assume": _convert_assume,
    "tt.splat": _convert_splat,
    "tt.addptr": _convert_addptr,
    "tt.expand_dims": _convert_expand_dims,
    "tt.get_program_id": _convert_program_id,
    "tt.return": _convert_return,
}

_SPECIALIZED_SOURCE_OPS = frozenset(
    {
        "arith.truncf",
        "rocdl.sched.barrier",
        "scf.for",
        "scf.if",
        "tt.broadcast",
        "tt.make_range",
        "ttg.local_alloc",
        "ttg.memdesc_index",
        "amdg.buffer_load_to_local",
        "amdg.buffer_load",
        "amdg.buffer_store",
        "tt.load",
        "tt.store",
        "ttg.local_load",
        "ttg.convert_layout",
        "tt.dot",
        "ttg.async_commit_group",
        "ttg.async_wait",
    }
)

_SUPPORTED_SOURCE_OPS = frozenset(_SIMPLE_OP_CONVERTERS) | _SPECIALIZED_SOURCE_OPS
_UNOWNED_SOURCE_OPS = _SUPPORTED_SOURCE_OPS - domains.all_source_ops()
if _UNOWNED_SOURCE_OPS:
    raise RuntimeError(f"unsupported source op domains: {sorted(_UNOWNED_SOURCE_OPS)}")


def _convert_sched_barrier(op):
    if op.results:
        fail(
            "TLXW_OP_UNEXPECTED_RESULT",
            STAGE,
            "rocdl.sched.barrier must not produce values",
            source_op_index=op.index,
        )


def _fact_ids_by_source_op(fact_program):
    result = {}
    for fact in fact_program.facts:
        if fact.source_op_index is None:
            continue
        result.setdefault(fact.source_op_index, tuple())
        result[fact.source_op_index] = (*result[fact.source_op_index], fact.fact_id)
    return result


def _fact_target_ids(builder, fact_program, fact_ids, op):
    target_ids = []
    for fact_id in fact_ids:
        try:
            fact = fact_program.facts[fact_id]
        except IndexError:
            fail(
                "TLXW_OP_UNKNOWN_FACT",
                STAGE,
                f"op references missing fact {fact_id}",
                source_op_index=op.index,
                fact_id=fact_id,
            )
        target_ids.append(_fact_target_id(builder, fact, op))
    return tuple(target_ids)


def _fact_target_id(builder, fact, op):
    targets = builder.source_value_targets.get(fact.subject_value_id)
    if not targets:
        fail(
            "TLXW_OP_FACT_TARGET",
            STAGE,
            f"fact {fact.fact_id} subject has no converted target value",
            source_op_index=op.index,
            source_value_id=fact.subject_value_id,
            fact_id=fact.fact_id,
        )
    if len(targets) != 1:
        fail(
            "TLXW_OP_FACT_TARGET",
            STAGE,
            f"fact {fact.fact_id} subject maps to multiple target values {targets}",
            source_op_index=op.index,
            source_value_id=fact.subject_value_id,
            fact_id=fact.fact_id,
        )
    return targets[0]


def _operand_assume_fact_ids(fact_program, op):
    fact_ids = []
    seen = set()
    for source_value_id in op.operands:
        for fact_id in fact_program.by_value.get(source_value_id, ()):
            fact = fact_program.facts[fact_id]
            if fact.kind != "range" or fact.provenance != "llvm.intr.assume":
                continue
            if fact_id in seen:
                continue
            seen.add(fact_id)
            fact_ids.append(fact_id)
    return tuple(fact_ids)


def _pointer_byte_range_fact(fact_program, value_id, op):
    for fact_id in fact_program.by_value.get(value_id, ()):
        fact = fact_program.facts[fact_id]
        if fact.kind == "pointer_byte_range" and fact.upper is not None:
            return fact
    fail(
        "TLXW_OP_MISSING_POINTER_RANGE_FACT",
        STAGE,
        "amdg.buffer_load_to_local requires a pointer byte-range fact",
        source_op_index=op.index,
        source_value_id=value_id,
    )


def _memdesc_infos(source_program):
    result = {}
    for value_id, value in source_program.values.items():
        if value.type.kind != "memdesc":
            continue
        result[value_id] = MemdescInfo(
            value_id,
            value.type.element_type,
            value.type.element_byte_width,
            tuple(value.type.shape),
            tuple(value.type.alloc_shape),
            _memdesc_size_bytes(value.type, value.owner_op_index, value_id),
        )
    return result


def _constant_ints(source_program):
    result = {}
    for op in source_program.ops:
        if op.name != "arith.constant" or len(op.results) != 1:
            continue
        literal = _constant_literal(
            op.attrs.get("value"),
            source_op_index=op.index,
        )
        if type(literal) is int:
            result[op.results[0]] = literal
    return result


def _compute_memdesc_physical_allocation_bytes(
    source_values,
    ops,
    type_layout_program,
    memdescs,
):
    ops_by_index = {op.index: op for op in ops}
    result = {}
    for value_id, memdesc in memdescs.items():
        value = source_values.get(value_id)
        op = (
            None
            if value is None or value.owner_op_index is None
            else ops_by_index.get(int(value.owner_op_index))
        )
        result[value_id] = _memdesc_physical_allocation_bytes(
            memdesc,
            _layout_for_value(type_layout_program, value_id),
            op,
        )
    return result


def _layout_for_value(type_layout_program, value_id):
    converted = type_layout_program.values.get(value_id)
    if converted is None or converted.layout_map_id is None:
        return None
    return type_layout_program.layouts[int(converted.layout_map_id)]


def _memdesc_physical_allocation_bytes(memdesc, layout, op):
    dense_size = int(memdesc.allocation_bytes)
    if layout is None or layout.kind in {"none", "linear", "swizzled_shared"}:
        return dense_size
    if layout.kind != "padded_shared":
        return dense_size
    element_byte_width = memdesc.element_byte_width
    shape = tuple(int(dim) for dim in (memdesc.alloc_shape or memdesc.shape or ()))
    if element_byte_width is None or not shape:
        return dense_size
    element_count = _product(shape)
    if element_count <= 0:
        return dense_size
    last_offset = _static_shared_byte_offset_from_linear(
        layout,
        shape,
        element_count - 1,
        int(element_byte_width),
        op,
    )
    if last_offset is None:
        return dense_size
    return _align_to(max(dense_size, int(last_offset) + int(element_byte_width)), 16)


def _compute_local_alloc_allocation_bytes(
    ops,
    memdescs,
    memdesc_physical_allocation_bytes,
):
    indexed_children_by_parent = {}
    for op in ops:
        if op.name != "ttg.memdesc_index" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        indexed_children_by_parent.setdefault(op.operands[0], []).append(
            (op, op.results[0])
        )

    result = {}
    for op in ops:
        if op.name != "ttg.local_alloc" or not op.results:
            continue
        value_id = op.results[0]
        memdesc = _memdesc_info_from_table(memdescs, value_id, op)
        children = indexed_children_by_parent.get(value_id)
        if not children:
            result[value_id] = int(
                memdesc_physical_allocation_bytes.get(
                    value_id,
                    memdesc.allocation_bytes,
                )
            )
            continue

        parent_elements = _product(memdesc.alloc_shape or memdesc.shape or (1,))
        child_slot_elements = None
        child_slot_bytes = None
        for child_op, child_value_id in children:
            child_memdesc = _memdesc_info_from_table(memdescs, child_value_id, child_op)
            child_elements = _product(
                child_memdesc.alloc_shape or child_memdesc.shape or (1,)
            )
            if child_elements <= 0 or parent_elements % child_elements:
                fail(
                    "TLXW_OP_MEMDESC_INDEX",
                    STAGE,
                    "ttg.memdesc_index child shape does not evenly tile the "
                    "local allocation",
                    source_op_index=child_op.index,
                    source_value_id=child_value_id,
                )
            if child_slot_elements is None:
                child_slot_elements = int(child_elements)
            elif child_slot_elements != int(child_elements):
                fail(
                    "TLXW_OP_MEMDESC_INDEX",
                    STAGE,
                    "ttg.memdesc_index children for a local allocation must "
                    "have matching slot sizes",
                    source_op_index=child_op.index,
                    source_value_id=child_value_id,
                )
            child_size = int(
                memdesc_physical_allocation_bytes.get(
                    child_value_id,
                    child_memdesc.allocation_bytes,
                )
            )
            child_slot_bytes = (
                child_size
                if child_slot_bytes is None
                else max(int(child_slot_bytes), child_size)
            )

        slot_count = parent_elements // int(child_slot_elements)
        result[value_id] = _align_to(slot_count * int(child_slot_bytes), 16)
    return result


def _compute_local_alloc_layout(ops, memdescs, local_alloc_allocation_bytes):
    offsets = {}
    cursor = 0
    for op in ops:
        if op.name != "ttg.local_alloc" or not op.results:
            continue
        value_id = op.results[0]
        memdesc = _memdesc_info_from_table(memdescs, value_id, op)
        size = local_alloc_allocation_bytes.get(value_id, memdesc.allocation_bytes)
        offsets[value_id] = cursor
        cursor = _align_to(cursor + size, 16)
    return offsets, cursor


def _compute_static_memdesc_byte_offsets(
    ops,
    memdescs,
    memdesc_physical_allocation_bytes,
    constant_ints,
    local_alloc_byte_offsets,
):
    offsets = dict(local_alloc_byte_offsets)
    for op in ops:
        if op.name != "ttg.memdesc_index" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        base_offset = offsets.get(op.operands[0])
        static_index = constant_ints.get(op.operands[1])
        if base_offset is None or static_index is None:
            continue
        slot_size = _memdesc_info_from_table(
            memdescs,
            op.results[0],
            op,
        ).allocation_bytes
        slot_size = memdesc_physical_allocation_bytes.get(op.results[0], slot_size)
        offsets[op.results[0]] = int(base_offset) + int(static_index) * int(slot_size)
    return offsets


def _memdesc_size_bytes(source_type, source_op_index, source_value_id):
    element_byte_width = source_type.element_byte_width
    if element_byte_width is None:
        fail(
            "TLXW_OP_MEMDESC_ELEMENT_SIZE",
            STAGE,
            f"cannot size LDS allocation {source_type.raw}: unknown element byte width",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return _product(source_type.alloc_shape or source_type.shape or (1,)) * int(
        element_byte_width
    )


def _memdesc_info(conversion_input, value_id, op):
    return _memdesc_info_from_table(conversion_input.memdescs, value_id, op)


def _memdesc_info_from_table(memdescs, value_id, op):
    memdesc = memdescs.get(value_id)
    if memdesc is not None:
        return memdesc
    fail(
        "TLXW_OP_MEMDESC_INFO",
        STAGE,
        f"expected memdesc metadata for value {value_id}",
        source_op_index=op.index if op is not None else None,
        source_value_id=value_id,
    )


def _align_to(value, alignment):
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


def _single_source_target(builder, source_value_id, op):
    targets = builder.source_value_targets.get(source_value_id)
    if not targets:
        fail(
            "TLXW_OP_UNCONVERTED_OPERAND",
            STAGE,
            f"yielded value {source_value_id} has no converted target value",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    if len(targets) != 1:
        fail(
            "TLXW_OP_MULTI_VALUE_OPERAND",
            STAGE,
            f"yielded value {source_value_id} maps to multiple target values {targets}",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return targets[0]


def _buffer_load_to_local_fields(op):
    segments = _operand_segments(op, 6, (1, 1, 1, 0, 0, 0))
    if int(segments[0]) != 1 or int(segments[1]) != 1 or int(segments[2]) != 1:
        fail(
            "TLXW_OP_MALFORMED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local requires destination, base, and offsets",
            source_op_index=op.index,
        )
    if any(int(segment) > 1 for segment in segments[3:]):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_ASYNC",
            STAGE,
            "amdg.buffer_load_to_local optional segments must be scalar",
            source_op_index=op.index,
        )
    _require_operand_count(op, segments)
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    stride_index = other_index + int(segments[4])
    return {
        "memdesc_value_id": op.operands[0],
        "base_value_id": op.operands[base_index],
        "offset_value_id": op.operands[offset_index],
        "mask_value_id": op.operands[mask_index] if int(segments[3]) else None,
        "other_value_id": op.operands[other_index] if int(segments[4]) else None,
        "stride_value_id": op.operands[stride_index] if int(segments[5]) else None,
        "cache": _int_attr_or_default(op.attrs, "cache", 1),
    }


def _load_fields(op):
    if len(op.results) != 1:
        fail(
            "TLXW_OP_MALFORMED_LOAD",
            STAGE,
            "tt.load requires one result",
            source_op_index=op.index,
        )
    if len(op.operands) not in (1, 2, 3):
        fail(
            "TLXW_OP_MALFORMED_LOAD",
            STAGE,
            "tt.load requires pointer plus optional mask/other operands",
            source_op_index=op.index,
        )
    return {
        "pointer_value_id": op.operands[0],
        "mask_value_id": op.operands[1] if len(op.operands) >= 2 else None,
        "other_value_id": op.operands[2] if len(op.operands) >= 3 else None,
    }


def _store_fields(op):
    if op.results:
        fail(
            "TLXW_OP_MALFORMED_STORE",
            STAGE,
            "tt.store must not produce results",
            source_op_index=op.index,
        )
    if len(op.operands) not in (2, 3):
        fail(
            "TLXW_OP_MALFORMED_STORE",
            STAGE,
            "tt.store requires pointer, value, and optional mask operands",
            source_op_index=op.index,
        )
    return {
        "pointer_value_id": op.operands[0],
        "value_value_id": op.operands[1],
        "mask_value_id": op.operands[2] if len(op.operands) == 3 else None,
    }


def _buffer_load_fields(op):
    segments = _operand_segments(op, 5, None)
    if int(segments[0]) != 1 or int(segments[1]) != 1:
        fail(
            "TLXW_OP_MALFORMED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load requires base pointer and offsets operands",
            source_op_index=op.index,
        )
    if int(segments[2]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load supports at most one stride operand",
            source_op_index=op.index,
        )
    if int(segments[3]) not in (0, 1) or int(segments[4]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_LOAD",
            STAGE,
            "amdg.buffer_load supports at most one mask and one other operand",
            source_op_index=op.index,
        )
    _require_operand_count(op, segments)
    offset_index = int(segments[0])
    stride_index = offset_index + int(segments[1])
    mask_index = stride_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    return {
        "base_value_id": op.operands[0],
        "offset_value_id": op.operands[offset_index],
        "stride_value_id": op.operands[stride_index] if int(segments[2]) else None,
        "mask_value_id": op.operands[mask_index] if int(segments[3]) else None,
        "other_value_id": op.operands[other_index] if int(segments[4]) else None,
        "cache": _int_attr_or_default(op.attrs, "cache", 1),
        "contiguity": _int_attr_or_default(op.attrs, "contiguity", 1),
    }


def _buffer_store_fields(op):
    segments = _operand_segments(op, 5, None)
    if int(segments[0]) != 1 or int(segments[1]) != 1 or int(segments[2]) != 1:
        fail(
            "TLXW_OP_MALFORMED_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store requires value, base pointer, and offsets",
            source_op_index=op.index,
        )
    if int(segments[3]) != 0:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store boundary-check operands are not converted yet",
            source_op_index=op.index,
        )
    if int(segments[4]) not in (0, 1):
        fail(
            "TLXW_OP_MALFORMED_BUFFER_STORE",
            STAGE,
            "amdg.buffer_store supports at most one mask operand",
            source_op_index=op.index,
        )
    _require_operand_count(op, segments)
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2]) + int(segments[3])
    return {
        "value_value_id": op.operands[0],
        "base_value_id": op.operands[base_index],
        "offset_value_id": op.operands[offset_index],
        "mask_value_id": op.operands[mask_index] if int(segments[4]) else None,
        "cache": _int_attr_or_default(op.attrs, "cache", 1),
        "contiguity": _int_attr_or_default(op.attrs, "contiguity", 1),
    }


def _operand_segments(op, expected_len, default):
    segments = op.attrs.get("operandSegmentSizes")
    if segments is None:
        segments = default
    segments = tuple(int(segment) for segment in segments)
    if len(segments) != int(expected_len):
        fail(
            "TLXW_OP_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} expected {expected_len} operand segments, got {segments}",
            source_op_index=op.index,
        )
    if any(segment < 0 for segment in segments):
        fail(
            "TLXW_OP_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments must be nonnegative, got {segments}",
            source_op_index=op.index,
        )
    return segments


def _require_operand_count(op, segments):
    if sum(int(segment) for segment in segments) != len(op.operands):
        fail(
            "TLXW_OP_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments {segments} do not match "
            f"{len(op.operands)} operands",
            source_op_index=op.index,
        )


def _require_default_cache(cache, op):
    if cache in (None, 1):
        return
    fail(
        "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
        STAGE,
        f"Wave lowering does not support {op.name} cacheModifier={cache}",
        source_op_index=op.index,
    )


def _require_default_tt_memory_attrs(op):
    cache = op.attrs.get("cache")
    if cache is not None and int(cache) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
            STAGE,
            f"Wave lowering does not support {op.name} cache={cache}",
            source_op_index=op.index,
        )
    cache_modifier = _attr_text(op.attrs.get("cacheModifier"))
    if cache_modifier not in {"", "none", "#tt.cache_modifier<none>"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER",
            STAGE,
            f"Wave lowering does not support {op.name} cacheModifier={cache_modifier}",
            source_op_index=op.index,
        )
    evict = op.attrs.get("evict")
    if evict is not None and int(evict) != 1:
        fail(
            "TLXW_OP_UNSUPPORTED_EVICTION_POLICY",
            STAGE,
            f"Wave lowering does not support {op.name} evict={evict}",
            source_op_index=op.index,
        )
    eviction_policy = _attr_text(op.attrs.get("evictionPolicy"))
    if eviction_policy not in {"", "none", "evict_normal", "#tt.eviction_policy<normal>"}:
        fail(
            "TLXW_OP_UNSUPPORTED_EVICTION_POLICY",
            STAGE,
            f"Wave lowering does not support {op.name} evictionPolicy={eviction_policy}",
            source_op_index=op.index,
        )
    if _attr_bool(op.attrs.get("isVolatile")):
        fail(
            "TLXW_OP_UNSUPPORTED_VOLATILE",
            STAGE,
            f"Wave lowering does not support volatile {op.name}",
            source_op_index=op.index,
        )


def _attr_text(value):
    if value is None:
        return ""
    return str(value).strip().strip('"')


def _attr_bool(value):
    text = _attr_text(value).lower()
    return text in {"true", "1"}


def _local_component_store_plan(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    offset_value_id,
    component_count,
    lane_width,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    shape = tuple(int(dim) for dim in (memdesc.shape or memdesc.alloc_shape))
    total_elements = _product(shape)
    wave_count = max(1, int(conversion_input.num_warps))
    if int(component_count) * int(lane_width) * wave_count != total_elements:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized amdg.buffer_load_to_local currently requires "
            "all-active full per-wave components",
            source_op_index=op.index,
            source_value_id=memdesc_value_id,
        )
    memdesc_layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    memdesc_layout = (
        None
        if memdesc_layout_id is None
        else type_layout_program.layouts[int(memdesc_layout_id)]
    )
    offset_layout_id = type_layout_program.values[offset_value_id].layout_map_id
    offset_layout = (
        None
        if offset_layout_id is None
        else type_layout_program.layouts[int(offset_layout_id)]
    )
    if (
        offset_layout is None
        or offset_layout.kind not in {"blocked", "linear"}
        or len(offset_layout.shape) != len(shape)
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized amdg.buffer_load_to_local requires a structural "
            "distributed offset layout for local destination mapping",
            source_op_index=op.index,
            source_value_id=offset_value_id,
        )
    linear = layouts.distributed_linear_layout(
        offset_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    if not linear.is_injective():
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "scalarized amdg.buffer_load_to_local requires an injective "
            "offset layout for local destination mapping",
            source_op_index=op.index,
            source_value_id=offset_value_id,
        )
    component_wave_offsets = []
    for component in range(int(component_count)):
        component_offsets = []
        for wave in range(wave_count):
            component_offsets.append(
                tuple(
                    _local_physical_offset_for_distributed_slot(
                        memdesc_layout,
                        shape,
                        memdesc.element_byte_width,
                        linear,
                        component,
                        lane,
                        wave,
                        op,
                        offset_value_id,
                    )
                    for lane in range(int(lane_width))
                )
            )
        component_wave_offsets.append(tuple(component_offsets))
    affine_plan = _try_affine_local_component_store_plan(
        component_wave_offsets,
        int(lane_width),
        wave_count,
    )
    if affine_plan is not None:
        return affine_plan
    return _coordinate_local_component_store_plan(
        offset_layout,
        memdesc_layout,
        shape,
        int(component_count),
        int(lane_width),
        wave_count,
        op,
        offset_value_id,
    )


def _try_affine_local_component_store_plan(
    component_wave_offsets,
    lane_width,
    wave_count,
):
    component_offsets = []
    lane_stride = None
    wave_stride = None
    for wave_offsets_by_lane in component_wave_offsets:
        wave_offsets = []
        for lane_offsets in wave_offsets_by_lane:
            base = int(lane_offsets[0])
            current_lane_stride = (
                0 if int(lane_width) == 1 else int(lane_offsets[1]) - base
            )
            if any(
                int(offset) != base + lane * current_lane_stride
                for lane, offset in enumerate(lane_offsets)
            ):
                return None
            if lane_stride is None:
                lane_stride = current_lane_stride
            elif lane_stride != current_lane_stride:
                return None
            wave_offsets.append(base)
        component_offsets.append(wave_offsets[0])
        current_wave_stride = (
            0 if wave_count == 1 else int(wave_offsets[1]) - int(wave_offsets[0])
        )
        if any(
            int(offset) != int(wave_offsets[0]) + wave * current_wave_stride
            for wave, offset in enumerate(wave_offsets)
        ):
            return None
        if wave_stride is None:
            wave_stride = current_wave_stride
        elif wave_stride != current_wave_stride:
            return None
    return {
        "offset_mode": "affine",
        "component_offsets": tuple(int(offset) for offset in component_offsets),
        "lane_stride_elements": int(lane_stride if lane_stride is not None else 1),
        "wave_stride_elements": int(wave_stride or 0),
    }


def _coordinate_local_component_store_plan(
    offset_layout,
    memdesc_layout,
    shape,
    component_count,
    lane_width,
    wave_count,
    op,
    offset_value_id,
):
    plan = coordinates.layout_coordinate_plan(
        offset_layout,
        int(component_count),
        int(lane_width),
        int(wave_count),
        op,
        offset_value_id,
    )
    return {
        "offset_mode": "layout_coordinates",
        "coordinate_shape": tuple(int(dim) for dim in plan.shape),
        "component_coordinate_bases": tuple(
            tuple(int(value) for value in bases) for bases in plan.component_bases
        ),
        "workitem_coordinate_coefficients": tuple(
            tuple(int(value) for value in coefficients)
            for coefficients in plan.workitem_coefficients
        ),
        "shared_layout_attrs": _scalarized_shared_layout_attrs(
            memdesc_layout,
            shape,
            op,
        ),
    }


def _scalarized_shared_layout_attrs(layout, shape, op):
    if layout is None or layout.kind in {"none", "linear"}:
        return {"destination_shared_layout": "dense"}
    if layout.kind == "swizzled_shared":
        order, vec, per_phase, max_phase = _swizzled_shared_parameters(
            layout,
            shape,
            op,
        )
        return {
            "destination_shared_layout": "swizzled",
            "destination_swizzled_order": tuple(int(dim) for dim in order),
            "destination_swizzled_vec": int(vec),
            "destination_swizzled_per_phase": int(per_phase),
            "destination_swizzled_max_phase": int(max_phase),
        }
    if layout.kind == "padded_shared":
        intervals, paddings = _padded_shared_parameters(layout, op)
        return {
            "destination_shared_layout": "padded",
            "destination_padded_intervals": tuple(int(value) for value in intervals),
            "destination_padded_paddings": tuple(int(value) for value in paddings),
        }
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"amdg.buffer_load_to_local destination layout {layout.kind} "
        "is not converted yet",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _local_physical_offset_for_distributed_slot(
    memdesc_layout,
    shape,
    element_byte_width,
    distributed_layout,
    component,
    lane,
    wave,
    op,
    source_value_id,
):
    coords = layouts.linear_layout_coords(
        distributed_layout,
        int(component),
        int(lane),
        warp=int(wave),
    )
    if len(coords) != len(shape):
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "distributed offset layout rank does not match local memdesc rank",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    for coord, extent in zip(coords, shape):
        if int(coord) < 0 or int(coord) >= int(extent):
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "distributed offset layout maps a component outside the "
                "local memdesc shape",
                source_op_index=op.index,
                source_value_id=source_value_id,
            )
    byte_offset = _static_shared_byte_offset(
        memdesc_layout,
        shape,
        coords,
        int(element_byte_width),
        op,
    )
    if int(byte_offset) % int(element_byte_width):
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            "local destination physical byte offset is not element aligned",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return int(byte_offset) // int(element_byte_width)


def _buffer_load_to_local_packet_plan(
    conversion_input,
    type_layout_program,
    fact_program,
    memdesc_value_id,
    offset_value_id,
    memdesc,
    lane_width,
    op,
):
    affine = fact_program.tensor_affine.get(offset_value_id)
    if affine is None:
        return None
    shape = tuple(int(dim) for dim in memdesc.shape)
    if tuple(affine.shape) != shape or not shape:
        return None
    packet_bytes = _dma_packet_bytes_for_element(memdesc.element_byte_width)
    if packet_bytes is None:
        return None
    packet_elements = packet_bytes // int(memdesc.element_byte_width)
    total_elements = _product(shape)
    wave_count = int(conversion_input.num_warps)
    if wave_count <= 0:
        return None
    elements_per_wave_packet = int(lane_width) * int(packet_elements)
    elements_per_cta_packet = int(wave_count) * elements_per_wave_packet
    if total_elements % elements_per_cta_packet:
        return None
    component_count = total_elements // elements_per_cta_packet
    layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    layout = (
        None
        if layout_id is None
        else type_layout_program.layouts[int(layout_id)]
    )
    destination_offsets, destination_wave_stride_dwords = (
        _packet_destination_offsets(
            layout,
            component_count,
            elements_per_cta_packet,
            elements_per_wave_packet,
            memdesc.element_byte_width,
            wave_count,
            op,
        )
    )
    scalar_value_ids, terms = _packet_affine_terms(affine)
    return {
        "component_thread_count": int(wave_count) * int(lane_width),
        "component_count": int(component_count),
        "destination_component_offsets": destination_offsets,
        "destination_wave_count": int(wave_count),
        "destination_wave_stride_dwords": int(destination_wave_stride_dwords),
        "packet_bytes": int(packet_bytes),
        "packet_elements": int(packet_elements),
        "scalar_value_ids": tuple(scalar_value_ids),
        "source_offset_terms": tuple(terms),
    }


def _packet_destination_offsets(
    layout,
    component_count,
    elements_per_cta_packet,
    elements_per_wave_packet,
    element_byte_width,
    wave_count,
    op,
):
    destination_offsets = []
    wave_stride_dwords = None
    for component in range(int(component_count)):
        component_linear = int(component) * int(elements_per_cta_packet)
        wave_offsets = tuple(
            _physical_component_offset(
                layout,
                component_linear + wave * int(elements_per_wave_packet),
                elements_per_wave_packet,
                op,
            )
            for wave in range(int(wave_count))
        )
        destination_offsets.append(wave_offsets[0])
        component_deltas = []
        for wave, wave_offset in enumerate(wave_offsets):
            byte_delta = (int(wave_offset) - int(wave_offsets[0])) * int(element_byte_width)
            if byte_delta % 4:
                fail(
                    "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                    STAGE,
                    "packet DMA destination wave offset must be dword aligned",
                    source_op_index=op.index,
                    source_value_id=layout.value_id if layout is not None else None,
                )
            component_deltas.append(byte_delta // 4)
            if wave == 0:
                continue
            stride = component_deltas[1]
            if byte_delta // 4 != stride * wave:
                fail(
                    "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                    STAGE,
                    "packet DMA destination wave offsets must form a uniform stride",
                    source_op_index=op.index,
                    source_value_id=layout.value_id if layout is not None else None,
                )
        component_stride = 0 if int(wave_count) == 1 else int(component_deltas[1])
        if wave_stride_dwords is None:
            wave_stride_dwords = component_stride
        elif wave_stride_dwords != component_stride:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "packet DMA destination wave stride must be identical for all components",
                source_op_index=op.index,
                source_value_id=layout.value_id if layout is not None else None,
            )
    return tuple(destination_offsets), int(wave_stride_dwords or 0)


def _dma_packet_bytes_for_element(element_byte_width):
    if element_byte_width is None:
        return None
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        return None
    if 16 % element_byte_width == 0:
        return 16
    if 4 % element_byte_width == 0:
        return 4
    return None


def _buffer_source_offset_upper(range_upper_bytes, packet_bytes, element_byte_width, op):
    range_upper_bytes = int(range_upper_bytes)
    packet_bytes = int(packet_bytes)
    element_byte_width = int(element_byte_width)
    if packet_bytes <= 0 or element_byte_width <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} source offset range requires positive access and "
            "element byte widths",
            source_op_index=op.index,
        )
    if range_upper_bytes < packet_bytes - 1:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} access exceeds source pointer range",
            source_op_index=op.index,
        )
    return (range_upper_bytes - packet_bytes + 1) // element_byte_width


def _buffer_inactive_byte_offset():
    return 1 << 31


def _buffer_inactive_element_offset(element_byte_width, op):
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} inactive source offset requires positive element byte width",
            source_op_index=op.index,
        )
    inactive_byte_offset = _buffer_inactive_byte_offset()
    if inactive_byte_offset % element_byte_width:
        fail(
            "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
            STAGE,
            f"{op.name} inactive byte offset is not aligned to element size",
            source_op_index=op.index,
        )
    return inactive_byte_offset // element_byte_width


def _packet_affine_terms(affine):
    scalar_value_ids = []
    scalar_slots = {}
    terms = []
    for term in affine.terms:
        slots = tuple(_scalar_slot(value_id, scalar_value_ids, scalar_slots) for value_id in term.scalar_value_ids)
        terms.append(
            (
                term.kind,
                int(term.coefficient),
                -1 if term.dim is None else int(term.dim),
                slots,
            )
        )
    return tuple(scalar_value_ids), tuple(terms)


def _scalar_slot(value_id, scalar_value_ids, scalar_slots):
    if value_id not in scalar_slots:
        scalar_slots[value_id] = len(scalar_value_ids)
        scalar_value_ids.append(value_id)
    return scalar_slots[value_id]


def _fragment_registers(element_type, result_layout, op):
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    if (
        element_type in {"f16", "bf16"}
        and instr_shape in {(16, 16, 32), (32, 32, 16)}
    ):
        return 4
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        "ttg.local_load fragment registers are not known for "
        f"element_type={element_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
        source_value_id=result_layout.value_id,
    )


def _acc_fragment_registers(result_layout, op):
    instr_shape = tuple(result_layout.properties.get("instr_shape", ()))
    if result_layout.element_type == "f32" and instr_shape == (16, 16, 32):
        return 4
    if result_layout.element_type == "f32" and instr_shape == (32, 32, 16):
        return 16
    fail(
        "TLXW_OP_FRAGMENT_CONSTANT",
        STAGE,
        "accumulator fragment registers are not known for "
        f"element_type={result_layout.element_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
        source_value_id=result_layout.value_id,
    )


def _mma_kind(element_type, instr_shape, op):
    if instr_shape == (16, 16, 32) and element_type == "f16":
        return "mfma.f32.16x16x32.f16"
    if instr_shape == (16, 16, 32) and element_type == "bf16":
        return "mfma.f32.16x16x32.bf16"
    if instr_shape == (32, 32, 16) and element_type == "f16":
        return "mfma.f32.32x32x16.f16"
    if instr_shape == (32, 32, 16) and element_type == "bf16":
        return "mfma.f32.32x32x16.bf16"
    fail(
        "TLXW_OP_DOT",
        STAGE,
        f"unsupported MFMA element type {element_type} for {instr_shape}",
        source_op_index=op.index,
    )


def _operand_fragment_shape(instr_shape, op):
    if instr_shape == (16, 16, 32):
        return 16, 16
    if instr_shape == (32, 32, 16):
        return 32, 32
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported local_load MFMA instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _acc_fragment_shape(instr_shape, op):
    if instr_shape == (16, 16, 32):
        return 16, 16
    if instr_shape == (32, 32, 16):
        return 32, 32
    fail(
        "TLXW_OP_FRAGMENT_CONSTANT",
        STAGE,
        f"unsupported accumulator MFMA instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _mfma_output_tile_shape(instr_shape, op):
    if instr_shape in {(16, 16, 32), (32, 32, 16)}:
        return 32, 32
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported MFMA instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _mfma_k_dim(instr_shape, op):
    if instr_shape == (16, 16, 32):
        return 32
    if instr_shape == (32, 32, 16):
        return 16
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported MFMA K dimension for instruction shape {instr_shape}",
        source_op_index=op.index,
    )


def _has_fragment_result(type_layout_program, op):
    for value_id in op.results:
        converted = type_layout_program.values[value_id]
        if converted.type.representation in {"fragment", "fragment_tuple"}:
            return True
    return False


def _require_layout(type_layout_program, layout_map_id, op):
    if layout_map_id is None:
        fail(
            "TLXW_OP_MISSING_LAYOUT",
            STAGE,
            "operation requires a converted layout map",
            source_op_index=op.index,
        )
    return type_layout_program.layouts[int(layout_map_id)]


def _require_dot_operand_layout(layout, op_idx, op):
    if layout.kind != "dot_operand" or int(layout.properties.get("op_idx", -1)) != int(op_idx):
        fail(
            "TLXW_OP_DOT",
            STAGE,
            f"tt.dot operand {op_idx} must use matching dot_operand layout",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )


def _mfma_per_wave_tiles(result_layout, instr_shape, warps_per_cta, op):
    if len(instr_shape) < 3 or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            "tt.dot requires MFMA instrShape and warpsPerCTA metadata",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    warps_m = int(warps_per_cta[0])
    warps_n = int(warps_per_cta[1])
    if warps_m <= 0 or warps_n <= 0:
        fail(
            "TLXW_OP_DOT",
            STAGE,
            f"invalid MFMA warpsPerCTA {warps_per_cta}",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    total_m_tiles = _ceil_div(int(result_layout.shape[0]), int(instr_shape[0]))
    total_n_tiles = _ceil_div(int(result_layout.shape[1]), int(instr_shape[1]))
    return _ceil_div(total_m_tiles, warps_m), _ceil_div(total_n_tiles, warps_n)


def _dot_operand_k_tiles(layout, instr_shape, op):
    op_idx = int(layout.properties["op_idx"])
    if op_idx == 0:
        if len(layout.shape) < 2:
            fail(
                "TLXW_OP_DOT",
                STAGE,
                "A dot operand requires rank-2 shape",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        return _ceil_div(int(layout.shape[1]), int(instr_shape[2]))
    if op_idx == 1:
        if len(layout.shape) < 2:
            fail(
                "TLXW_OP_DOT",
                STAGE,
                "B dot operand requires rank-2 shape",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        return _ceil_div(int(layout.shape[0]), int(instr_shape[2]))
    fail(
        "TLXW_OP_DOT",
        STAGE,
        f"unsupported dot operand index {op_idx}",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _is_zero_literal(value):
    if value in (0, 0.0, "0", "0.0"):
        return True
    text = str(value).strip().lower()
    match = re.match(r"dense<([^>]+)>", text)
    if match is not None:
        try:
            return float(match.group(1)) == 0.0
        except ValueError:
            return False
    return False


def _fragment_local_load_plan(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    result_layout,
    component_count,
    registers,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    layout = (
        None
        if layout_id is None
        else type_layout_program.layouts[int(layout_id)]
    )
    transpose_plan = _b16_transpose_fragment_load_plan(
        memdesc,
        layout,
        result_layout,
        component_count,
        registers,
        op,
    )
    if transpose_plan is not None:
        return transpose_plan
    indexed_plan = _indexed_fragment_load_plan(
        memdesc,
        layout,
        result_layout,
        component_count,
        registers,
        op,
    )
    if indexed_plan is not None:
        return indexed_plan
    offset_plan = _fragment_component_dword_offsets(
        conversion_input,
        type_layout_program,
        memdesc_value_id,
        result_layout,
        component_count,
        registers,
        op,
    )
    return {
        "component_dword_offsets": tuple(offset_plan["component_dword_offsets"]),
        "load_mode": "fragment_load",
        "warps_per_cta": tuple(offset_plan["warps_per_cta"]),
        "wave_tile_axis": offset_plan["wave_tile_axis"],
        "wave_tile_stride_dwords": int(offset_plan["wave_tile_stride_dwords"]),
    }


def _b16_transpose_fragment_load_plan(
    memdesc,
    layout,
    result_layout,
    component_count,
    registers,
    op,
):
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    op_idx = int(result_layout.properties.get("op_idx", -1))
    if not (
        op_idx == 1
        and instr_shape in {(16, 16, 32), (32, 32, 16)}
        and int(result_layout.lane_width) == 64
        and int(registers) == 4
        and result_layout.element_type in {"f16", "bf16"}
        and memdesc.element_type == result_layout.element_type
        and int(memdesc.element_byte_width or 0) == 2
    ):
        return None
    if not _is_supported_b16_transpose_layout(layout):
        return None
    tile_plan = _fragment_component_tile_offsets(
        memdesc,
        result_layout,
        component_count,
        op,
    )
    source_shape = _dot_operand_source_shape(result_layout, instr_shape, op)
    elements_per_lane = int(registers) * (4 // int(memdesc.element_byte_width))
    if elements_per_lane != 8:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"b16 transpose local_load expects 8 elements per lane, got {elements_per_lane}",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    for tile_offsets in tile_plan["component_tile_offsets"]:
        _validate_b16_transpose_packets(
            layout,
            tuple(int(dim) for dim in memdesc.shape),
            source_shape,
            tile_offsets,
            int(memdesc.element_byte_width),
            int(result_layout.lane_width),
            elements_per_lane,
            (0,),
            _fragment_lane_layout(
                result_layout,
                instr_shape,
                int(elements_per_lane),
                transpose_load=True,
            ),
            op,
        )
    lane_layout = _fragment_lane_layout(
        result_layout,
        instr_shape,
        int(elements_per_lane),
        transpose_load=True,
    )
    chunk_element_deltas = _b16_transpose_chunk_element_deltas(
        layout,
        tuple(int(dim) for dim in memdesc.shape),
        source_shape,
        tuple(tile_plan["component_tile_offsets"]),
        int(memdesc.element_byte_width),
        int(result_layout.lane_width),
        elements_per_lane,
        2,
        4,
        tile_plan["wave_tile_axis"],
        tuple(tile_plan["warps_per_cta"]),
        int(tile_plan["wave_tile_stride_elements"]),
        lane_layout,
        op,
    )
    attrs = _encoded_shared_layout_attrs(layout)
    result = {
        **attrs,
        "chunk_elements": 4,
        "chunks_per_component": 2,
        "component_tile_offsets": tuple(tile_plan["component_tile_offsets"]),
        "elements_per_lane": int(elements_per_lane),
        "fragment_lane_layout": lane_layout,
        "load_mode": "b16_transpose",
        "memdesc_shape": tuple(int(dim) for dim in memdesc.shape),
        "source_shape": tuple(source_shape),
        "warps_per_cta": tuple(tile_plan["warps_per_cta"]),
        "wave_tile_axis": tile_plan["wave_tile_axis"],
        "wave_tile_stride_elements": int(tile_plan["wave_tile_stride_elements"]),
    }
    if chunk_element_deltas is not None:
        result["chunk_element_deltas"] = tuple(chunk_element_deltas)
    return result


def _indexed_fragment_load_plan(
    memdesc,
    layout,
    result_layout,
    component_count,
    registers,
    op,
):
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    if result_layout.element_type not in {"f16", "bf16"}:
        return None
    if memdesc.element_type != result_layout.element_type:
        return None
    if int(memdesc.element_byte_width or 0) != 2:
        return None
    requires_indexed = False
    layout_attrs = {"shared_layout_kind": "dense"}
    if layout is None or layout.kind in {"none", "linear"}:
        pass
    elif layout.kind == "swizzled_shared":
        if _is_identity_swizzled_layout(layout):
            pass
        elif _is_supported_swizzled_layout(layout):
            requires_indexed = True
            layout_attrs = _encoded_shared_layout_attrs(layout)
        else:
            return None
    elif layout.kind == "padded_shared":
        _padded_shared_parameters(layout, op)
        requires_indexed = True
        layout_attrs = _encoded_shared_layout_attrs(layout)
    else:
        return None
    tile_plan = _fragment_component_tile_offsets(
        memdesc,
        result_layout,
        component_count,
        op,
    )
    source_shape = _dot_operand_source_shape(result_layout, instr_shape, op)
    if len(memdesc.shape) != len(source_shape):
        return None
    if tuple(int(dim) for dim in memdesc.shape) != tuple(int(dim) for dim in source_shape):
        requires_indexed = True
    if not requires_indexed:
        return None
    elements_per_lane = int(registers) * (4 // int(memdesc.element_byte_width))
    lane_layout = _fragment_lane_layout(
        result_layout,
        instr_shape,
        int(elements_per_lane),
    )
    load_mode = (
        "swizzled_fragment_load"
        if layout is not None
        and layout.kind == "swizzled_shared"
        and not _is_identity_swizzled_layout(layout)
        else "indexed_fragment_load"
    )
    wave_offsets = (
        (0,)
        if load_mode == "swizzled_fragment_load"
        else _possible_wave_tile_element_offsets(
            tile_plan["wave_tile_axis"],
            tuple(tile_plan["warps_per_cta"]),
            int(tile_plan["wave_tile_stride_elements"]),
            op,
        )
    )
    for tile_offsets in tile_plan["component_tile_offsets"]:
        _validate_fragment_load_packets(
            layout,
            tuple(int(dim) for dim in memdesc.shape),
            source_shape,
            tile_offsets,
            int(memdesc.element_byte_width),
            int(result_layout.lane_width),
            elements_per_lane,
            wave_offsets,
            op,
            lane_layout=lane_layout,
        )
    return {
        **layout_attrs,
        "component_tile_offsets": tuple(tile_plan["component_tile_offsets"]),
        "elements_per_lane": int(elements_per_lane),
        "fragment_lane_layout": lane_layout,
        "load_mode": load_mode,
        "memdesc_shape": tuple(int(dim) for dim in memdesc.shape),
        "source_shape": tuple(source_shape),
        "warps_per_cta": tuple(tile_plan["warps_per_cta"]),
        "wave_tile_axis": tile_plan["wave_tile_axis"],
        "wave_tile_stride_elements": int(tile_plan["wave_tile_stride_elements"]),
    }


def _fragment_component_tile_offsets(memdesc, result_layout, component_count, op):
    shape = tuple(int(dim) for dim in memdesc.shape)
    op_idx = int(result_layout.properties["op_idx"])
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    warps_per_cta = tuple(parent.get("warps_per_cta", ()))
    if len(instr_shape) < 3 or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "dot-operand local_load requires MFMA instrShape and warpsPerCTA",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    tile_offsets = []
    wave_tile_axis = "none"
    wave_tile_stride_elements = 0
    for component in range(int(component_count)):
        if op_idx == 0:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            k_tiles = _ceil_div(shape[1], instr_shape[2])
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            m_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_m = max(1, int(warps_per_cta[0]))
            tile_offsets.append(
                (
                    m_tile * warps_m * instr_shape[0],
                    k_tile * instr_shape[2],
                )
            )
            wave_tile_axis = "m"
            wave_tile_stride_elements = instr_shape[0] * shape[1]
        elif op_idx == 1:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            k_tiles = _ceil_div(shape[0], instr_shape[2])
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            n_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_n = max(1, int(warps_per_cta[1]))
            tile_offsets.append(
                (
                    k_tile * instr_shape[2],
                    n_tile * warps_n * instr_shape[1],
                )
            )
            wave_tile_axis = "n"
            wave_tile_stride_elements = instr_shape[1]
        else:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                f"unsupported dot operand index {op_idx}",
                source_op_index=op.index,
                source_value_id=result_layout.value_id,
            )
    return {
        "component_tile_offsets": tuple(tuple(int(coord) for coord in offsets) for offsets in tile_offsets),
        "warps_per_cta": tuple(int(value) for value in warps_per_cta),
        "wave_tile_axis": wave_tile_axis,
        "wave_tile_stride_elements": int(wave_tile_stride_elements),
    }


def _dot_operand_source_shape(result_layout, instr_shape, op):
    k_dim = _mfma_k_dim(instr_shape, op)
    op_idx = int(result_layout.properties.get("op_idx", -1))
    if op_idx == 0:
        return (int(instr_shape[0]), k_dim)
    if op_idx == 1:
        return (k_dim, int(instr_shape[1]))
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported dot operand index {op_idx}",
        source_op_index=op.index,
        source_value_id=result_layout.value_id,
    )


def _fragment_lane_layout(
    result_layout,
    instr_shape,
    elements_per_lane,
    *,
    transpose_load=False,
):
    op_idx = int(result_layout.properties.get("op_idx", -1))
    if (
        op_idx == 0
        and tuple(int(value) for value in instr_shape)
        in {(16, 16, 32), (32, 32, 16)}
        and int(result_layout.lane_width) == 64
        and int(elements_per_lane) == 8
    ):
        return "gfx950_mfma_a"
    if (
        op_idx == 1
        and bool(transpose_load)
        and tuple(int(value) for value in instr_shape)
        in {(16, 16, 32), (32, 32, 16)}
        and int(result_layout.lane_width) == 64
        and int(elements_per_lane) == 8
    ):
        return "gfx950_mfma_b_transpose"
    return "row_major_linear"


def _is_supported_b16_transpose_layout(layout):
    if layout is None:
        return False
    if layout.kind == "padded_shared":
        return True
    return _is_supported_swizzled_layout(layout)


def _is_supported_swizzled_layout(layout):
    return (
        layout is not None
        and layout.kind == "swizzled_shared"
        and int(layout.properties.get("vec", 0)) == 8
        and int(layout.properties.get("per_phase", 0)) == 4
        and int(layout.properties.get("max_phase", 0)) == 4
        and tuple(layout.properties.get("order", ())) == (1, 0)
    )


def _is_identity_swizzled_layout(layout):
    order = tuple(layout.properties.get("order", ())) if layout is not None else ()
    return (
        layout is not None
        and layout.kind == "swizzled_shared"
        and int(layout.properties.get("vec", 0)) == 1
        and int(layout.properties.get("per_phase", 0)) == 1
        and int(layout.properties.get("max_phase", 0)) == 1
        and order in {(1, 0), (0,), ()}
    )


def _encoded_shared_layout_attrs(layout):
    if layout.kind == "swizzled_shared":
        return {
            "shared_layout_kind": "swizzled_shared",
            "swizzled_max_phase": int(layout.properties["max_phase"]),
            "swizzled_order": tuple(layout.properties["order"]),
            "swizzled_per_phase": int(layout.properties["per_phase"]),
            "swizzled_vec": int(layout.properties["vec"]),
        }
    if layout.kind == "padded_shared":
        return {
            "padded_intervals": tuple(layout.properties["intervals"]),
            "padded_order": tuple(layout.properties["order"]),
            "padded_paddings": tuple(layout.properties["paddings"]),
            "shared_layout_kind": "padded_shared",
        }
    return {"shared_layout_kind": "dense"}


def _validate_b16_transpose_packets(
    layout,
    memdesc_shape,
    source_shape,
    tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    lane_layout,
    op,
):
    for chunk in range(2):
        _validate_fragment_load_packets(
            layout,
            memdesc_shape,
            source_shape,
            tile_offsets,
            element_byte_width,
            lane_width,
            elements_per_lane,
            wave_offsets,
            op,
            local_extra_elements=4 * chunk,
            packet_elements=4,
            required_alignment=8,
            packet_description="transpose load",
            lane_layout=lane_layout,
        )


def _b16_transpose_chunk_element_deltas(
    layout,
    memdesc_shape,
    source_shape,
    component_tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    chunks_per_component,
    chunk_elements,
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride_elements,
    lane_layout,
    op,
):
    wave_offsets = _possible_wave_tile_element_offsets(
        wave_tile_axis,
        warps_per_cta,
        wave_tile_stride_elements,
        op,
    )
    result = []
    for tile_offsets in component_tile_offsets:
        component_deltas = []
        for chunk in range(int(chunks_per_component)):
            extra_elements = int(chunk_elements) * chunk
            byte_deltas = set()
            for wave_offset in wave_offsets:
                for lane in range(int(lane_width)):
                    base_linear = _static_local_fragment_lane_offset(
                        memdesc_shape,
                        source_shape,
                        tile_offsets,
                        int(lane),
                        int(elements_per_lane),
                        0,
                        int(wave_offset),
                        lane_layout,
                        op,
                        fail_on_oob=False,
                    )
                    if base_linear is None:
                        return None
                    chunk_linear = _static_local_fragment_lane_offset(
                        memdesc_shape,
                        source_shape,
                        tile_offsets,
                        int(lane),
                        int(elements_per_lane),
                        int(extra_elements),
                        int(wave_offset),
                        lane_layout,
                        op,
                        fail_on_oob=False,
                    )
                    if chunk_linear is None:
                        return None
                    base_byte = _static_shared_byte_offset_from_linear(
                        layout,
                        memdesc_shape,
                        base_linear,
                        int(element_byte_width),
                        op,
                    )
                    chunk_byte = _static_shared_byte_offset_from_linear(
                        layout,
                        memdesc_shape,
                        chunk_linear,
                        int(element_byte_width),
                        op,
                    )
                    if base_byte is None or chunk_byte is None:
                        return None
                    byte_deltas.add(chunk_byte - base_byte)
            if len(byte_deltas) != 1:
                return None
            byte_delta = next(iter(byte_deltas))
            if byte_delta % int(element_byte_width):
                return None
            component_deltas.append(byte_delta // int(element_byte_width))
        result.append(tuple(component_deltas))
    return tuple(result)


def _possible_wave_tile_element_offsets(
    wave_tile_axis,
    warps_per_cta,
    wave_tile_stride_elements,
    op,
):
    stride = int(wave_tile_stride_elements)
    if wave_tile_axis == "none" or stride == 0:
        return (0,)
    if len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose load requires warpsPerCTA for wave-tile delta proof",
            source_op_index=op.index,
        )
    if wave_tile_axis == "m":
        count = int(warps_per_cta[0])
    elif wave_tile_axis == "n":
        count = int(warps_per_cta[1])
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported transpose load wave axis {wave_tile_axis}",
            source_op_index=op.index,
        )
    if count <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "transpose load wave-tile delta proof requires positive warp count",
            source_op_index=op.index,
        )
    return tuple(index * stride for index in range(count))


def _validate_fragment_load_packets(
    layout,
    memdesc_shape,
    source_shape,
    tile_offsets,
    element_byte_width,
    lane_width,
    elements_per_lane,
    wave_offsets,
    op,
    *,
    local_extra_elements=0,
    packet_elements=None,
    required_alignment=4,
    packet_description="fragment load",
    lane_layout="row_major_linear",
):
    packet_elements = (
        int(elements_per_lane) if packet_elements is None else int(packet_elements)
    )
    for wave_offset in wave_offsets:
        for lane in range(int(lane_width)):
            first = None
            for element in range(int(packet_elements)):
                linear = _static_local_fragment_lane_offset(
                    memdesc_shape,
                    source_shape,
                    tile_offsets,
                    int(lane),
                    int(elements_per_lane),
                    int(local_extra_elements) + int(element),
                    int(wave_offset),
                    lane_layout,
                    op,
                )
                byte_offset = _static_shared_byte_offset_from_linear(
                    layout,
                    memdesc_shape,
                    linear,
                    int(element_byte_width),
                    op,
                )
                if byte_offset is None:
                    fail(
                        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                        STAGE,
                        "fragment load coordinate exceeds memdesc shape "
                        f"{memdesc_shape}",
                        source_op_index=op.index,
                    )
                if first is None:
                    first = byte_offset
                    if first % int(required_alignment):
                        fail(
                            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                            STAGE,
                            f"{packet_description} packet physical byte offset "
                            f"{first} is not {required_alignment}-byte aligned",
                            source_op_index=op.index,
                        )
                    continue
                expected = first + element * int(element_byte_width)
                if byte_offset != expected:
                    fail(
                        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                        STAGE,
                        f"{packet_description} packet is not physically "
                        f"contiguous at linear offset {linear}",
                        source_op_index=op.index,
                    )


def _static_local_fragment_lane_offset(
    memdesc_shape,
    source_shape,
    tile_offsets,
    lane,
    elements_per_lane,
    extra_elements,
    wave_offset,
    lane_layout,
    op,
    *,
    fail_on_oob=True,
):
    if lane_layout == "row_major_linear":
        return _static_local_fragment_linear_offset(
            memdesc_shape,
            source_shape,
            tile_offsets,
            int(lane) * int(elements_per_lane) + int(extra_elements),
            int(wave_offset),
            op,
            fail_on_oob=fail_on_oob,
        )
    if lane_layout == "gfx950_mfma_a":
        if len(memdesc_shape) != 2 or len(source_shape) != 2 or len(tile_offsets) != 2:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA A fragment load requires rank-2 source and memdesc shapes",
                source_op_index=op.index,
            )
        row = int(lane) % int(source_shape[0])
        col = (
            (int(lane) // int(source_shape[0])) * int(elements_per_lane)
            + int(extra_elements)
        )
        coords = (int(tile_offsets[0]) + row, int(tile_offsets[1]) + col)
        for dim, coord in enumerate(coords):
            if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
                if not fail_on_oob:
                    return None
                _check_coords_in_bounds(coords, memdesc_shape, op)
        linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
        if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
            if not fail_on_oob:
                return None
        return linear
    if lane_layout == "gfx950_mfma_b_transpose":
        if len(memdesc_shape) != 2 or len(source_shape) != 2 or len(tile_offsets) != 2:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B transpose load requires rank-2 source and memdesc shapes",
                source_op_index=op.index,
            )
        non_k_dim = int(source_shape[1])
        if non_k_dim % 16:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "gfx950 MFMA B transpose load requires the non-K dimension "
                "to be a multiple of the ds_read_tr group width",
                source_op_index=op.index,
            )
        lane_in_group = int(lane) % 16
        non_k_group = (int(lane) % non_k_dim) // 16
        k_group = int(lane) // non_k_dim
        chunk_k = (int(extra_elements) // 4) * 4
        packet_col = int(extra_elements) % 4
        row = (
            k_group * int(elements_per_lane)
            + chunk_k
            + lane_in_group // 4
        )
        col = non_k_group * 16 + 4 * (lane_in_group % 4) + packet_col
        coords = (int(tile_offsets[0]) + row, int(tile_offsets[1]) + col)
        for dim, coord in enumerate(coords):
            if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
                if not fail_on_oob:
                    return None
                _check_coords_in_bounds(coords, memdesc_shape, op)
        linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
        if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
            if not fail_on_oob:
                return None
        return linear
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported fragment lane layout {lane_layout}",
        source_op_index=op.index,
    )


def _static_local_fragment_linear_offset(
    memdesc_shape,
    source_shape,
    tile_offsets,
    local_linear,
    wave_offset,
    op,
    *,
    fail_on_oob=True,
):
    if len(memdesc_shape) != len(source_shape) or len(tile_offsets) != len(source_shape):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "fragment load coordinate remap requires matching ranks",
            source_op_index=op.index,
        )
    local_coords = _static_delinearize_row_major(
        int(local_linear),
        source_shape,
        op,
    )
    coords = tuple(
        int(tile_offsets[dim]) + int(local_coords[dim])
        for dim in range(len(source_shape))
    )
    for dim, coord in enumerate(coords):
        if int(coord) < 0 or int(coord) >= int(memdesc_shape[dim]):
            if not fail_on_oob:
                return None
            _check_coords_in_bounds(coords, memdesc_shape, op)
    linear = _static_linear_offset(memdesc_shape, coords) + int(wave_offset)
    if int(linear) < 0 or int(linear) >= _product(memdesc_shape):
        if not fail_on_oob:
            return None
    return linear


def _check_coords_in_bounds(coords, shape, op):
    for dim, coord in enumerate(coords):
        if int(coord) < 0 or int(coord) >= int(shape[dim]):
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                f"fragment load coordinate {tuple(coords)} exceeds memdesc shape {shape}",
                source_op_index=op.index,
            )


def _static_shared_byte_offset(layout, shape, coords, element_byte_width, op):
    if layout is None or layout.kind in {"none", "linear"}:
        return _static_linear_offset(shape, coords) * int(element_byte_width)
    if layout.kind == "swizzled_shared":
        if _is_identity_swizzled_layout(layout):
            return _static_linear_offset(shape, coords) * int(element_byte_width)
        return _static_swizzled_byte_offset(layout, shape, coords, element_byte_width, op)
    if layout.kind == "padded_shared":
        return _static_padded_byte_offset(layout, shape, coords, element_byte_width, op)
    fail(
        "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
        STAGE,
        f"unsupported shared layout {layout.kind} for static packet validation",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _static_shared_byte_offset_from_linear(
    layout,
    shape,
    linear,
    element_byte_width,
    op,
):
    if int(linear) < 0 or int(linear) >= _product(shape):
        return None
    coords = _static_delinearize_row_major(linear, shape, op)
    return _static_shared_byte_offset(
        layout,
        shape,
        coords,
        int(element_byte_width),
        op,
    )


def _static_linear_offset(shape, coords):
    offset = 0
    for dim, coord in enumerate(coords):
        stride = _product(shape[dim + 1 :])
        offset += int(coord) * stride
    return int(offset)


def _static_swizzled_byte_offset(layout, shape, coords, element_byte_width, op):
    order, vec, per_phase, max_phase = _swizzled_shared_parameters(
        layout,
        shape,
        op,
    )
    minor_dim = int(order[0])
    major_dim = int(order[1])
    minor_extent = int(shape[minor_dim])
    major = int(coords[major_dim])
    minor = int(coords[minor_dim])
    phase = (major // per_phase) % max_phase
    swizzled_minor = ((minor // vec) ^ phase) * vec + (minor % vec)
    if swizzled_minor >= minor_extent:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"swizzled shared minor coordinate {swizzled_minor} exceeds "
            f"extent {minor_extent}; {_swizzled_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return (major * minor_extent + swizzled_minor) * int(element_byte_width)


def _static_padded_byte_offset(layout, shape, coords, element_byte_width, op):
    intervals, paddings = _padded_shared_parameters(layout, op)
    linear = _static_linear_offset(shape, coords)
    encoded = linear
    for interval, padding in zip(intervals, paddings):
        encoded += (linear // int(interval)) * int(padding)
    return encoded * int(element_byte_width)


def _swizzled_shared_parameters(layout, shape, op):
    order = tuple(layout.properties.get("order", ()))
    if len(shape) != 2 or order not in {(1, 0), (0, 1)}:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "static swizzled shared LDS offsets support only rank-2 "
            f"order=[1,0] or order=[0,1]; got {_swizzled_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    vec = int(layout.properties["vec"])
    per_phase = int(layout.properties["per_phase"])
    max_phase = int(layout.properties["max_phase"])
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "swizzled shared layout requires positive "
            f"vec/per_phase/max_phase; got {_swizzled_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    minor_extent = int(shape[int(order[0])])
    if minor_extent % vec:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"swizzled shared minor extent {minor_extent} is not divisible "
            f"by vec={vec}; {_swizzled_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return order, vec, per_phase, max_phase


def _padded_shared_parameters(layout, op):
    if tuple(layout.properties.get("order", ())) not in {(0, 1), (1, 0), (0,), ()}:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"unsupported padded shared order; {_padded_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    intervals = tuple(int(value) for value in layout.properties.get("intervals", ()))
    paddings = tuple(int(value) for value in layout.properties.get("paddings", ()))
    if len(intervals) != len(paddings):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "padded shared layout requires matching interval/padding "
            f"counts; {_padded_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    if any(interval <= 0 for interval in intervals):
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "padded shared intervals must be positive; "
            f"{_padded_shared_description(layout)}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return intervals, paddings


def _swizzled_shared_description(layout):
    props = layout.properties
    return (
        f"order={tuple(props.get('order', ()))}, "
        f"vec={int(props.get('vec', 0))}, "
        f"per_phase={int(props.get('per_phase', 0))}, "
        f"max_phase={int(props.get('max_phase', 0))}"
    )


def _padded_shared_description(layout):
    props = layout.properties
    paddings = tuple(int(value) for value in props.get("paddings", ()))
    return (
        f"order={tuple(props.get('order', ()))}, "
        f"intervals={tuple(int(value) for value in props.get('intervals', ()))}, "
        f"paddings={paddings}"
    )


def _static_delinearize_row_major(linear, shape, op):
    coords = [0] * len(shape)
    remainder = int(linear)
    for dim in reversed(range(len(shape))):
        extent = int(shape[dim])
        coords[dim] = remainder % extent
        remainder //= extent
    if remainder:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"fragment packet linear index {linear} exceeds shape {shape}",
            source_op_index=op.index,
        )
    return tuple(coords)


def _fragment_component_dword_offsets(
    conversion_input,
    type_layout_program,
    memdesc_value_id,
    result_layout,
    component_count,
    registers,
    op,
):
    memdesc = _memdesc_info(conversion_input, memdesc_value_id, op)
    if memdesc.element_byte_width is None:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "ttg.local_load requires a known memdesc element width",
            source_op_index=op.index,
            source_value_id=memdesc_value_id,
        )
    layout_id = type_layout_program.values[memdesc_value_id].layout_map_id
    layout = (
        None
        if layout_id is None
        else type_layout_program.layouts[int(layout_id)]
    )
    shape = tuple(int(dim) for dim in memdesc.shape)
    op_idx = int(result_layout.properties["op_idx"])
    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(parent.get("instr_shape", ()))
    warps_per_cta = tuple(parent.get("warps_per_cta", ()))
    if len(instr_shape) < 3 or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "dot-operand local_load requires MFMA instrShape and warpsPerCTA",
            source_op_index=op.index,
            source_value_id=result_layout.value_id,
        )
    offsets = []
    linear_offsets = []
    wave_tile_axis = "none"
    wave_tile_stride_elements = 0
    for component in range(int(component_count)):
        if op_idx == 0:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=memdesc_value_id,
                )
            k_tiles = _ceil_div(shape[1], instr_shape[2])
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "A-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            m_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_m = max(1, int(warps_per_cta[0]))
            linear = (
                m_tile * warps_m * instr_shape[0] * shape[1]
                + k_tile * instr_shape[2]
            )
            wave_tile_axis = "m"
            wave_tile_stride_elements = instr_shape[0] * shape[1]
        elif op_idx == 1:
            if len(shape) < 2:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment local_load requires rank-2 memdesc shape",
                    source_op_index=op.index,
                    source_value_id=memdesc_value_id,
                )
            k_tiles = _ceil_div(shape[0], instr_shape[2])
            if component_count % k_tiles:
                fail(
                    "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                    STAGE,
                    "B-fragment component count is not divisible by K tiles",
                    source_op_index=op.index,
                    source_value_id=result_layout.value_id,
                )
            n_tile = component // k_tiles
            k_tile = component % k_tiles
            warps_n = max(1, int(warps_per_cta[1]))
            linear = (
                k_tile * instr_shape[2] * shape[1]
                + n_tile * warps_n * instr_shape[1]
            )
            wave_tile_axis = "n"
            wave_tile_stride_elements = instr_shape[1]
        else:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                f"unsupported dot operand index {op_idx}",
                source_op_index=op.index,
                source_value_id=result_layout.value_id,
            )
        linear_offsets.append(int(linear))
        physical = _physical_element_offset(layout, linear, op)
        byte_offset = physical * int(memdesc.element_byte_width)
        if byte_offset % 4:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "fragment local_load base must be dword aligned",
                source_op_index=op.index,
                source_value_id=memdesc_value_id,
            )
        offsets.append(byte_offset // 4)
    wave_tile_stride_dwords = _element_stride_to_dwords(
        layout,
        wave_tile_stride_elements,
        int(memdesc.element_byte_width),
        tuple(linear_offsets),
        op,
    )
    return {
        "component_dword_offsets": tuple(offsets),
        "warps_per_cta": tuple(int(value) for value in warps_per_cta),
        "wave_tile_axis": wave_tile_axis,
        "wave_tile_stride_dwords": int(wave_tile_stride_dwords),
    }


def _physical_element_offset(layout, linear, op):
    return _physical_component_offset(layout, int(linear), 1, op)


def _element_stride_to_dwords(layout, element_stride, element_byte_width, linear_offsets, op):
    if int(element_stride) == 0:
        return 0
    if layout is None or layout.kind in {"none", "swizzled_shared"}:
        if layout is not None and layout.kind == "swizzled_shared":
            _require_identity_swizzled(layout, op)
        byte_stride = int(element_stride) * int(element_byte_width)
    elif layout.kind == "padded_shared":
        physical_strides = {
            _physical_component_offset(layout, int(linear) + int(element_stride), 1, op)
            - _physical_component_offset(layout, int(linear), 1, op)
            for linear in linear_offsets
        }
        if len(physical_strides) != 1:
            fail(
                "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
                STAGE,
                "padded shared MFMA local_load needs per-component wave "
                "stride remaps",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        byte_stride = next(iter(physical_strides)) * int(element_byte_width)
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            f"MFMA local_load destination layout {layout.kind} is not converted yet",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    if byte_stride % 4:
        fail(
            "TLXW_OP_UNSUPPORTED_LOCAL_LOAD",
            STAGE,
            "fragment local_load wave tile stride must be dword aligned",
            source_op_index=op.index,
            source_value_id=layout.value_id if layout is not None else None,
        )
    return byte_stride // 4


def _physical_component_offset(layout, linear_start, lane_width, op):
    linear_end = int(linear_start) + int(lane_width) - 1
    if layout is None or layout.kind in {"none", "swizzled_shared"}:
        if layout is not None and layout.kind == "swizzled_shared":
            _require_identity_swizzled(layout, op)
        return int(linear_start)
    if layout.kind == "padded_shared":
        intervals = layout.properties.get("intervals", ())
        paddings = layout.properties.get("paddings", ())
        if len(intervals) != 1 or len(paddings) != 1:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "scalarized amdg.buffer_load_to_local supports one "
                "padded interval",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        interval = int(intervals[0])
        padding = int(paddings[0])
        if interval <= 0:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "padded shared interval must be positive",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        if linear_start // interval != linear_end // interval:
            fail(
                "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
                STAGE,
                "scalarized amdg.buffer_load_to_local component crosses "
                "a padded LDS interval",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        return int(linear_start) + (int(linear_start) // interval) * padding
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        f"amdg.buffer_load_to_local destination layout {layout.kind} "
        "is not converted yet",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _require_identity_swizzled(layout, op):
    props = layout.properties
    if (
        int(props.get("vec", 0)) == 1
        and int(props.get("per_phase", 0)) == 1
        and int(props.get("max_phase", 0)) == 1
    ):
        return
    fail(
        "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC",
        STAGE,
        "amdg.buffer_load_to_local swizzled destination requires an "
        "explicit remap target op",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _constant_literal(value, *, source_op_index=None):
    if value is None:
        return None
    text = str(value).strip()
    dense = re.fullmatch(r"dense<([^>]*)>\s*(?::.*)?", text, re.DOTALL)
    if dense is not None:
        payload = dense.group(1).strip()
        if payload.startswith("[") or "," in payload:
            fail(
                "TLXW_OP_UNSUPPORTED_CONSTANT",
                STAGE,
                "tensor constants are converted only for dense splats",
                source_op_index=source_op_index,
            )
        text = payload
    else:
        text = text.split(":", 1)[0].strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    match = re.fullmatch(r"([+-]?(?:0[xX][0-9a-fA-F]+|\d+))", text)
    if match is not None:
        return int(match.group(1), 0)
    match = re.fullmatch(
        r"([+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?)",
        text,
    )
    if match is not None:
        return float(match.group(1))
    return text


def _cmpi_predicate(value):
    predicates = {
        0: "eq",
        1: "ne",
        2: "slt",
        3: "sle",
        4: "sgt",
        5: "sge",
        6: "ult",
        7: "ule",
        8: "ugt",
        9: "uge",
    }
    if value is None:
        return "unknown"
    if isinstance(value, str) and value in predicates.values():
        return value
    return predicates.get(int(value), str(value))


def _target_int_width(builder, target_value_ids):
    for target_value_id in target_value_ids:
        target_type = builder.values[target_value_id].type
        width = _int_width(target_type.element_type)
        if width is not None:
            return width
    return None


def _int_width(raw_type):
    match = re.fullmatch(r"i([0-9]+)", str(raw_type))
    return None if match is None else int(match.group(1))


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _int_attr(attrs, name):
    value = attrs.get(name)
    if value is None:
        fail(
            "TLXW_OP_MISSING_ATTR",
            STAGE,
            f"required attr {name} is missing",
        )
    return int(value)


def _int_attr_or_default(attrs, name, default):
    value = attrs.get(name)
    return int(default) if value is None else int(value)
