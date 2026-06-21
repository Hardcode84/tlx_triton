"""Structural layout-remap helpers for TLX Wave conversion."""

from .diagnostics import fail
from . import layouts


STAGE = "op_conversion"

_DISTRIBUTED_REMAP_KINDS = frozenset({"blocked", "linear"})
_DISTRIBUTED_REMAP_REPRESENTATIONS = frozenset(
    {
        "mask",
        "mask_tuple",
        "per_lane_pointer",
        "pointer_tuple",
        "simd",
        "simd_tuple",
    }
)


def register_remap(operand, result, operand_layout, result_layout, op):
    if operand_layout is None or result_layout is None:
        return None
    if not (operand_layout.kind == "amd_mfma" and result_layout.kind == "blocked"):
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if result.type.representation not in {"simd", "simd_tuple"}:
        return None

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    source_registers_per_component = layouts.mfma_registers_per_component(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_scalar_count = (
        int(operand.type.component_count) * source_registers_per_component
    )
    if source_scalar_count != source_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA convert_layout source component model does not match "
            "the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if int(result.type.component_count) != result_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout requires a warp-aware result "
            f"component model: result has {int(result.type.component_count)} "
            f"components but the per-wave register layout has "
            f"{result_register_count}",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
    )

    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
        )
        for result_register in range(result_register_count)
    )

    simple_remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result.value_id,
    )
    if simple_remap is not None:
        return {
            "source_component_count": int(operand.type.component_count),
            "source_registers_per_component": int(source_registers_per_component),
            **simple_remap,
        }

    exchange_remap = _cta_exchange_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        op,
        result.value_id,
    )
    return {
        "mode": "cta_exchange_register_remap",
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": int(source_registers_per_component),
        **exchange_remap,
    }


def distributed_remap(operand, result, operand_layout, result_layout, op):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if operand.type.kind != result.type.kind:
        return None
    if operand.type.representation not in _DISTRIBUTED_REMAP_REPRESENTATIONS:
        return None
    if result.type.representation not in _DISTRIBUTED_REMAP_REPRESENTATIONS:
        return None

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    description = (
        f"{operand_layout.kind} to {result_layout.kind} convert_layout"
    )
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        f"{description} source layout",
    )
    _require_injective_layout(
        result_layout_ll,
        op,
        result.value_id,
        f"{description} result layout",
    )
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    if int(operand.type.component_count) != source_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} source component model does not match the "
            "distributed register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if int(result.type.component_count) != result_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} result component model does not match the "
            "distributed register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description=description,
    )
    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description=description,
        )
        for result_register in range(result_register_count)
    )
    remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        1,
        op,
        result.value_id,
        description=description,
    )
    if remap is None:
        _reject_distributed_movement(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description,
        )
    if remap["mode"] == "cross_lane_register_remap" and result.type.representation not in {
        "simd",
        "simd_tuple",
    }:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires cross-lane movement for "
            f"{result.type.representation}; only Wave SIMD payloads can be "
            "shuffled",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    return {
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": 1,
        **remap,
    }


def reject_unsupported_pair(operand_layout, result_layout, op):
    operand_kind = "none" if operand_layout is None else operand_layout.kind
    result_kind = "none" if result_layout is None else result_layout.kind
    if operand_kind in {"slice", "dot_operand"} or result_kind in {"slice", "dot_operand"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{operand_kind} to {result_kind} convert_layout requires parent "
            "layout movement support",
            source_op_index=op.index,
        )
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{operand_kind} to {result_kind} convert_layout has unknown "
        "movement class",
        source_op_index=op.index,
    )


def _source_slots_by_coord(
    source_layout,
    source_register_count,
    lane_width,
    cta_warp_count,
    op,
    source_value_id,
    *,
    description="MFMA convert_layout",
):
    source_by_coord = {}
    for source_warp in range(int(cta_warp_count)):
        for source_register in range(int(source_register_count)):
            for lane in range(int(lane_width)):
                coords = layouts.linear_layout_coords(
                    source_layout,
                    source_register,
                    lane,
                    warp=source_warp,
                )
                if coords in source_by_coord:
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} source layout is not injective "
                        "within the CTA distributed map",
                        source_op_index=op.index,
                        source_value_id=source_value_id,
                    )
                source_by_coord[coords] = (source_warp, lane, source_register)
    return source_by_coord


def _sources_for_result_slot(
    result_layout,
    result_register,
    source_by_coord,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    sources = []
    for result_warp in range(int(cta_warp_count)):
        for lane in range(int(lane_width)):
            coords = layouts.linear_layout_coords(
                result_layout,
                result_register,
                lane,
                warp=result_warp,
            )
            source = source_by_coord.get(coords)
            if source is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    f"{description} result coordinate is not "
                    "covered by the source distributed layout",
                    source_op_index=op.index,
                    source_value_id=result_value_id,
                )
            sources.append(tuple(int(value) for value in source))
    return tuple(sources)


def _simple_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    source_registers_per_component,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    source_indices = []
    source_element_indices = []
    source_lane_maps = []
    for sources in result_sources:
        if len(sources) != int(cta_warp_count) * int(lane_width):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a malformed "
                "source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        lane_maps = []
        registers = []
        for result_warp in range(int(cta_warp_count)):
            wave_lane_map = []
            for lane in range(int(lane_width)):
                source_warp, source_lane, source_register = sources[
                    result_warp * int(lane_width) + lane
                ]
                if source_warp != result_warp:
                    return None
                registers.append(source_register)
                wave_lane_map.append(source_lane)
            lane_maps.append(tuple(int(lane) for lane in wave_lane_map))

        first_source = registers[0]
        if not all(source == first_source for source in registers):
            return None
        first_lane_map = lane_maps[0]
        if not all(lane_map == first_lane_map for lane_map in lane_maps):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires a wave-varying source "
                "lane map; explicit CTA-wave remap support is required",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        if all(source_lane == lane for lane, source_lane in enumerate(first_lane_map)):
            source_lane_map = None
        else:
            source_lane_map = first_lane_map
        source_indices.append(first_source // int(source_registers_per_component))
        source_element_indices.append(first_source % int(source_registers_per_component))
        source_lane_maps.append(source_lane_map)

    lane_map_attrs = _classify_source_lane_maps(
        source_lane_maps,
        lane_width,
        op,
        result_value_id,
        description=description,
    )
    return {
        "mode": "cross_lane_register_remap"
        if lane_map_attrs is not None
        else "same_lane_register_remap",
        "source_element_indices": tuple(source_element_indices),
        "source_indices": tuple(source_indices),
        **(lane_map_attrs or {}),
    }


def _cta_exchange_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
):
    cta_thread_count = int(lane_width) * int(cta_warp_count)
    groups = {}
    max_group_slots = 0
    for result_index, sources in enumerate(result_sources):
        if len(sources) != cta_thread_count:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "MFMA to blocked convert_layout produced a malformed "
                "CTA source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        source_slots = tuple(sorted({int(source[2]) for source in sources}))
        if not source_slots:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "MFMA to blocked convert_layout produced an empty "
                "CTA source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        max_group_slots = max(max_group_slots, len(source_slots))
        source_slot_indices = {
            source_slot: index for index, source_slot in enumerate(source_slots)
        }
        load_offsets = []
        for source_warp, source_lane, source_register in sources:
            load_offsets.append(
                source_slot_indices[int(source_register)] * cta_thread_count
                + int(source_warp) * int(lane_width)
                + int(source_lane)
            )
        base, coefficients = _fit_bit_affine_offsets(
            load_offsets,
            cta_thread_count,
            op,
            result_value_id,
        )
        groups.setdefault(source_slots, []).append(
            (int(result_index), int(base), tuple(int(value) for value in coefficients))
        )

    exchange_groups = []
    for source_slots, result_entries in groups.items():
        exchange_groups.append(
            (
                tuple(int(slot) for slot in source_slots),
                tuple(int(entry[0]) for entry in result_entries),
                tuple(int(entry[1]) for entry in result_entries),
                tuple(
                    tuple(int(value) for value in entry[2])
                    for entry in result_entries
                ),
            )
        )
    return {
        "cta_thread_count": int(cta_thread_count),
        "exchange_groups": tuple(exchange_groups),
        "scratch_element_count": int(max_group_slots) * int(cta_thread_count),
    }


def _reject_distributed_movement(
    result_sources,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    description,
):
    for sources in result_sources:
        for result_warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                source_warp, _source_lane, _source_register = sources[
                    result_warp * int(lane_width) + lane
                ]
                if int(source_warp) != int(result_warp):
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} requires cross-warp movement",
                        source_op_index=op.index,
                        source_value_id=result_value_id,
                    )
        source_registers = {int(source[2]) for source in sources}
        if len(source_registers) > 1:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires per-lane source component selection",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{description} has unknown movement class",
        source_op_index=op.index,
        source_value_id=result_value_id,
    )


def _require_injective_layout(linear, op, source_value_id, description):
    if linear.is_injective():
        return
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{description} is non-injective",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _fit_bit_affine_offsets(load_offsets, cta_thread_count, op, result_value_id):
    load_offsets = tuple(int(offset) for offset in load_offsets)
    if len(load_offsets) != int(cta_thread_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout produced a malformed CTA "
            "exchange load map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if int(cta_thread_count) <= 0 or int(cta_thread_count) & (
        int(cta_thread_count) - 1
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout CTA exchange requires a "
            "power-of-two CTA thread count",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    base = load_offsets[0]
    coefficients = []
    for bit in range(int(cta_thread_count).bit_length() - 1):
        coefficients.append(load_offsets[1 << bit] - base)
    for thread in range(int(cta_thread_count)):
        expected = base
        for bit, coefficient in enumerate(coefficients):
            if thread & (1 << bit):
                expected += int(coefficient)
        if load_offsets[thread] != expected:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "MFMA to blocked convert_layout requires a non-bit-affine "
                "CTA exchange load map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    return int(base), tuple(int(value) for value in coefficients)


def _classify_source_lane_maps(
    source_lane_maps,
    lane_width,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    concrete_maps = [lane_map for lane_map in source_lane_maps if lane_map is not None]
    if not concrete_maps:
        return None
    first = concrete_maps[0]
    if not all(lane_map == first for lane_map in concrete_maps):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires per-component source "
            "lane maps; explicit lane-map remap support is required",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if len(first) != int(lane_width):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if any(int(lane) < 0 or int(lane) >= int(lane_width) for lane in first):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced an out-of-range source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    kind, attrs = _classify_lane_map(first, lane_width)
    if kind is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires a non-affine source "
            "lane map; Wave shuffle emission only supports affine and "
            "2-D transpose lane maps",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    return {
        "source_lane_map": tuple(int(lane) for lane in first),
        "source_lane_map_kind": kind,
        **attrs,
    }


def _classify_lane_map(lane_map, lane_width):
    lane_width = int(lane_width)
    identity = tuple(range(lane_width))
    if tuple(lane_map) == identity:
        return None, {}
    if lane_width > 1:
        stride = int(lane_map[1]) - int(lane_map[0])
        base = int(lane_map[0])
        if all(int(value) == base + lane * stride for lane, value in enumerate(lane_map)):
            return "affine", {
                "source_lane_affine_base": int(base),
                "source_lane_affine_stride": int(stride),
            }
    for inner in _lane_width_factors(lane_width):
        outer = lane_width // inner
        transposed = tuple((lane % inner) * outer + lane // inner for lane in range(lane_width))
        if tuple(lane_map) == transposed:
            return "transpose", {
                "source_lane_transpose_inner": int(inner),
                "source_lane_transpose_outer": int(outer),
            }
    return None, {}


def _lane_width_factors(lane_width):
    lane_width = int(lane_width)
    for factor in range(2, lane_width):
        if lane_width % factor == 0:
            yield factor


def _distributed_linear_layout(layout, op):
    if layout.kind not in {"blocked", "linear", "amd_mfma"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"layout {layout.kind} is not converted through linear-layout remap",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return layouts.distributed_linear_layout(
        layout,
        stage=STAGE,
        source_op_index=op.index,
    )


def _layout_warp_count(layout):
    return layouts.layout_warp_count(layout)
