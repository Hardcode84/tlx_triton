"""Structural layout-remap helpers for TLX Wave conversion."""

from .diagnostics import fail
from . import layouts


STAGE = "op_conversion"


def same_lane_register_remap(operand, result, operand_layout, result_layout, op):
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

    source_indices = []
    source_element_indices = []
    for result_register in range(result_register_count):
        slot = _same_lane_source_register_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
        )
        source_indices.append(slot // source_registers_per_component)
        source_element_indices.append(slot % source_registers_per_component)

    return {
        "source_component_count": int(operand.type.component_count),
        "source_element_indices": tuple(source_element_indices),
        "source_indices": tuple(source_indices),
        "source_registers_per_component": int(source_registers_per_component),
    }


def _source_slots_by_coord(
    source_layout,
    source_register_count,
    lane_width,
    cta_warp_count,
    op,
    source_value_id,
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
                        "MFMA convert_layout source layout is not injective "
                        "within the CTA distributed map",
                        source_op_index=op.index,
                        source_value_id=source_value_id,
                    )
                source_by_coord[coords] = (source_warp, lane, source_register)
    return source_by_coord


def _same_lane_source_register_for_result_slot(
    result_layout,
    result_register,
    source_by_coord,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
):
    sources = []
    needs_cross_lane = False
    needs_cross_warp = False
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
                    "MFMA to blocked convert_layout result coordinate is not "
                    "covered by the source distributed layout",
                    source_op_index=op.index,
                    source_value_id=result_value_id,
                )
            source_warp, source_lane, source_register = source
            if source_warp != result_warp:
                needs_cross_warp = True
            elif source_lane != lane:
                needs_cross_lane = True
            sources.append(source_register)

    first_source = sources[0]
    if all(source == first_source for source in sources):
        if needs_cross_warp:
            _reject_cross_warp(op, result_value_id)
        if needs_cross_lane:
            _reject_cross_lane(op, result_value_id)
        return first_source
    if needs_cross_warp:
        _reject_cross_warp(op, result_value_id)
    if needs_cross_lane:
        _reject_cross_lane(op, result_value_id)
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        "MFMA to blocked convert_layout maps one result component to "
        "different source registers across lanes or waves",
        source_op_index=op.index,
        source_value_id=result_value_id,
    )


def _reject_cross_lane(op, source_value_id):
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        "MFMA to blocked convert_layout requires cross-lane movement; "
        "Wave cross-lane remap support is required",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _reject_cross_warp(op, source_value_id):
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        "MFMA to blocked convert_layout requires cross-warp movement; "
        "Wave does not expose a CTA-local layout remap operation yet",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


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
    warps_per_cta = tuple(
        int(value) for value in layout.properties.get("warps_per_cta", ())
    )
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result
