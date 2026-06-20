"""Structural layout maps for the TLX Wave converter."""

from dataclasses import dataclass

from .diagnostics import fail


STAGE = "type_layout"


@dataclass(frozen=True)
class LayoutMap:
    layout_map_id: int
    value_id: int
    kind: str
    shape: tuple[int, ...]
    element_type: str | None
    component_count: int
    lane_width: int
    properties: dict


def build_layout_map(layout_map_id, value_id, source_type, lane_width):
    if source_type.kind not in {"tensor", "memdesc"}:
        return None
    attr = source_type.encoding_attr
    kind, properties = _layout_kind_and_properties(attr, value_id)
    component_count = _layout_component_count(source_type, kind, properties, lane_width)
    return LayoutMap(
        layout_map_id,
        value_id,
        kind,
        tuple(source_type.shape),
        source_type.element_type,
        int(component_count),
        int(lane_width),
        properties,
    )


def _layout_kind_and_properties(attr, value_id):
    if attr is None:
        return "none", {}
    if _attr_bool(attr, "is_blocked_encoding"):
        return "blocked", {
            "size_per_thread": _int_tuple(_attr_value(attr, "get_blocked_size_per_thread")),
            "threads_per_warp": _int_tuple(_attr_value(attr, "get_blocked_threads_per_warp")),
            "warps_per_cta": _int_tuple(_attr_value(attr, "get_blocked_warps_per_cta")),
            "order": _int_tuple(_attr_value(attr, "get_blocked_order")),
        }
    if _attr_bool(attr, "is_linear_encoding"):
        return "linear", {
            "register_bases": _basis_tuple(_attr_value(attr, "get_linear_register_bases")),
            "lane_bases": _basis_tuple(_attr_value(attr, "get_linear_lane_bases")),
            "warp_bases": _basis_tuple(_attr_value(attr, "get_linear_warp_bases")),
            "block_bases": _basis_tuple(_attr_value(attr, "get_linear_block_bases")),
        }
    if _attr_bool(attr, "is_slice_encoding"):
        parent = _attr_value(attr, "get_slice_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(parent, value_id)
        return "slice", {
            "dim": int(_attr_value(attr, "get_slice_dim")),
            "parent_kind": parent_kind,
            "parent_properties": parent_properties,
        }
    if _attr_bool(attr, "is_dot_operand_encoding"):
        parent = _attr_value(attr, "get_dot_operand_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(parent, value_id)
        return "dot_operand", {
            "op_idx": int(_attr_value(attr, "get_dot_operand_op_idx")),
            "k_width": int(_attr_value(attr, "get_dot_operand_k_width")),
            "parent_kind": parent_kind,
            "parent_properties": parent_properties,
        }
    if _attr_bool(attr, "is_amd_mfma_encoding"):
        return "amd_mfma", {
            "version": int(_attr_value(attr, "get_amd_mfma_version")),
            "warps_per_cta": _int_tuple(_attr_value(attr, "get_amd_mfma_warps_per_cta")),
            "instr_shape": _int_tuple(_attr_value(attr, "get_amd_mfma_instr_shape")),
            "is_transposed": bool(_attr_value(attr, "get_amd_mfma_is_transposed")),
            "tiles_per_warp": _int_tuple(_attr_value(attr, "get_amd_mfma_tiles_per_warp")),
            "element_bit_width": int(_attr_value(attr, "get_amd_mfma_element_bit_width")),
        }
    if _attr_bool(attr, "is_swizzled_shared_encoding"):
        return "swizzled_shared", {
            "vec": int(_attr_value(attr, "get_swizzled_shared_vec")),
            "per_phase": int(_attr_value(attr, "get_swizzled_shared_per_phase")),
            "max_phase": int(_attr_value(attr, "get_swizzled_shared_max_phase")),
            "order": _int_tuple(_attr_value(attr, "get_swizzled_shared_order")),
        }
    if _attr_bool(attr, "is_padded_shared_encoding"):
        return "padded_shared", {
            "intervals": _int_tuple(_attr_value(attr, "get_padded_shared_intervals")),
            "paddings": _int_tuple(_attr_value(attr, "get_padded_shared_paddings")),
            "order": _int_tuple(_attr_value(attr, "get_padded_shared_order")),
        }
    fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        STAGE,
        f"unsupported layout encoding {attr}",
        source_value_id=value_id,
    )


def _layout_component_count(source_type, kind, properties, lane_width):
    if kind == "blocked" and len(source_type.shape) == 1:
        size_per_thread = properties.get("size_per_thread", ())
        if len(size_per_thread) == 1 and int(size_per_thread[0]) > 1:
            element_count = _product(source_type.shape)
            elements_per_thread = int(size_per_thread[0])
            if element_count <= int(lane_width) * elements_per_thread:
                return elements_per_thread
    if kind == "dot_operand":
        parent_properties = properties.get("parent_properties", {})
        instr_shape = parent_properties.get("instr_shape", ())
        warps_per_cta = parent_properties.get("warps_per_cta", ())
        if len(instr_shape) >= 3 and len(warps_per_cta) >= 2 and len(source_type.shape) >= 2:
            op_idx = int(properties.get("op_idx", -1))
            if op_idx == 0:
                m_tiles = _per_wave_tile_count(
                    int(source_type.shape[0]),
                    int(instr_shape[0]),
                    int(warps_per_cta[0]),
                )
                k_tiles = _ceil_div(int(source_type.shape[1]), int(instr_shape[2]))
                return m_tiles * k_tiles
            if op_idx == 1:
                n_tiles = _per_wave_tile_count(
                    int(source_type.shape[1]),
                    int(instr_shape[1]),
                    int(warps_per_cta[1]),
                )
                k_tiles = _ceil_div(int(source_type.shape[0]), int(instr_shape[2]))
                return n_tiles * k_tiles
        return 1
    if kind == "amd_mfma":
        instr_shape = properties.get("instr_shape", ())
        warps_per_cta = properties.get("warps_per_cta", ())
        if len(instr_shape) >= 2 and len(warps_per_cta) >= 2 and len(source_type.shape) >= 2:
            m_tiles, n_tiles = _per_wave_mfma_tiles(
                source_type.shape,
                instr_shape,
                warps_per_cta,
            )
            return m_tiles * n_tiles
        return 1
    element_count = _product(source_type.shape)
    return max(1, _ceil_div(element_count, int(lane_width)))


def _attr_bool(attr, method):
    fn = getattr(attr, method, None)
    return bool(fn()) if fn is not None else False


def _attr_value(attr, method):
    fn = getattr(attr, method, None)
    if fn is None:
        fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            STAGE,
            f"layout encoding is missing {method}",
        )
    return fn()


def _int_tuple(values):
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _basis_tuple(values):
    if values is None:
        return ()
    return tuple(tuple(int(dim) for dim in basis) for basis in values)


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _per_wave_mfma_tiles(shape, instr_shape, warps_per_cta):
    total_m_tiles = _ceil_div(int(shape[0]), int(instr_shape[0]))
    total_n_tiles = _ceil_div(int(shape[1]), int(instr_shape[1]))
    warps_m = max(1, int(warps_per_cta[0]))
    warps_n = max(1, int(warps_per_cta[1]))
    return _ceil_div(total_m_tiles, warps_m), _ceil_div(total_n_tiles, warps_n)


def _per_wave_tile_count(extent, instr_extent, warp_extent):
    return _ceil_div(_ceil_div(int(extent), int(instr_extent)), max(1, int(warp_extent)))


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)
