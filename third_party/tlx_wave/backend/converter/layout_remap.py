"""Structural layout-remap helpers for TLX Wave conversion."""

from .diagnostics import fail
from . import layouts


STAGE = "op_conversion"


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

    source_indices = []
    source_element_indices = []
    source_lane_maps = []
    for result_register in range(result_register_count):
        slot, source_lane_map = _source_register_for_result_slot(
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
        source_lane_maps.append(source_lane_map)

    lane_map_attrs = _classify_source_lane_maps(
        source_lane_maps,
        lane_width,
        op,
        result.value_id,
    )

    return {
        "mode": "cross_lane_register_remap"
        if lane_map_attrs is not None
        else "same_lane_register_remap",
        "source_component_count": int(operand.type.component_count),
        "source_element_indices": tuple(source_element_indices),
        "source_indices": tuple(source_indices),
        "source_registers_per_component": int(source_registers_per_component),
        **(lane_map_attrs or {}),
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


def _source_register_for_result_slot(
    result_layout,
    result_register,
    source_by_coord,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
):
    registers = []
    lane_maps = []
    needs_cross_warp = False
    for result_warp in range(int(cta_warp_count)):
        wave_registers = []
        wave_lane_map = []
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
            wave_registers.append(source_register)
            wave_lane_map.append(source_lane)
        registers.extend(wave_registers)
        lane_maps.append(tuple(int(lane) for lane in wave_lane_map))

    if needs_cross_warp:
        _reject_cross_warp(op, result_value_id)
    first_source = registers[0]
    if not all(source == first_source for source in registers):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout maps one result component to "
            "different source registers across lanes or waves",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    first_lane_map = lane_maps[0]
    if not all(lane_map == first_lane_map for lane_map in lane_maps):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout requires a wave-varying source "
            "lane map; explicit CTA-wave remap support is required",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if all(source_lane == lane for lane, source_lane in enumerate(first_lane_map)):
        return first_source, None
    return first_source, first_lane_map


def _classify_source_lane_maps(source_lane_maps, lane_width, op, result_value_id):
    concrete_maps = [lane_map for lane_map in source_lane_maps if lane_map is not None]
    if not concrete_maps:
        return None
    first = concrete_maps[0]
    if not all(lane_map == first for lane_map in concrete_maps):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout requires per-component source "
            "lane maps; explicit lane-map remap support is required",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if len(first) != int(lane_width):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout produced a malformed source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if any(int(lane) < 0 or int(lane) >= int(lane_width) for lane in first):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout produced an out-of-range source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    kind, attrs = _classify_lane_map(first, lane_width)
    if kind is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout requires a non-affine source "
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
