"""Structural layout maps for the TLX Wave converter."""

from dataclasses import dataclass

from triton._C.libtriton.linear_layout import LinearLayout

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
    component_count = _layout_component_count(
        source_type,
        kind,
        properties,
        lane_width,
        value_id,
    )
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


def _layout_component_count(source_type, kind, properties, lane_width, value_id):
    if kind in {"blocked", "linear"}:
        linear = distributed_linear_layout_from_parts(
            kind,
            source_type.shape,
            properties,
            lane_width,
            source_value_id=value_id,
        )
        if linear.is_injective():
            return linear_layout_in_dim_size(linear, "register")
        element_count = _product(source_type.shape)
        return max(1, _ceil_div(element_count, int(lane_width)))
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


def distributed_register_count(
    kind,
    shape,
    properties,
    lane_width,
    *,
    stage=STAGE,
    source_op_index=None,
    source_value_id=None,
):
    linear = distributed_linear_layout_from_parts(
        kind,
        shape,
        properties,
        lane_width,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    return linear_layout_in_dim_size(linear, "register")


def distributed_linear_layout(
    layout,
    *,
    stage=STAGE,
    source_op_index=None,
):
    return distributed_linear_layout_from_parts(
        layout.kind,
        layout.shape,
        layout.properties,
        layout.lane_width,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=layout.value_id,
    )


def distributed_linear_layout_from_parts(
    kind,
    shape,
    properties,
    lane_width,
    *,
    stage=STAGE,
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    if kind == "blocked":
        return _blocked_linear_layout(
            shape,
            properties,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if kind == "linear":
        return _linear_encoding_layout(
            shape,
            properties,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if kind == "amd_mfma":
        return _mfma_linear_layout(
            shape,
            properties,
            lane_width,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        stage,
        f"layout {kind} does not have a distributed register map",
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def linear_layout_in_dim_size(linear, dim):
    for in_dim, bases in linear.bases:
        if in_dim == dim:
            return 1 << len(bases)
    return 1


def linear_layout_out_dim_size(linear, dim, *, stage=STAGE):
    for out_dim, size in linear.out_dims:
        if out_dim == dim:
            return int(size)
    _layout_fail(
        "TLXW_TYPE_MALFORMED_LAYOUT",
        stage,
        f"linear layout is missing output dimension {dim}",
    )


def linear_layout_coords(linear, register, lane, *, warp):
    available = {
        "block": 0,
        "register": int(register),
        "lane": int(lane),
        "warp": int(warp),
    }
    coords = linear.apply(
        {name: available[name] for name in linear.get_in_dim_names()}
    )
    return tuple(int(coords[f"dim{dim}"]) for dim in range(len(coords)))


def mfma_registers_per_component(
    layout,
    *,
    stage=STAGE,
    source_op_index=None,
):
    instr_shape = tuple(int(value) for value in layout.properties.get("instr_shape", ()))
    if instr_shape == (16, 16, 32):
        return 4
    if instr_shape == (32, 32, 16):
        return 16
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        stage,
        f"unsupported MFMA register count for instrShape={instr_shape}",
        source_op_index=source_op_index,
        source_value_id=layout.value_id,
    )


def _blocked_linear_layout(
    shape,
    properties,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    rank = len(shape)
    size_per_thread = tuple(int(value) for value in properties["size_per_thread"])
    threads_per_warp = tuple(int(value) for value in properties["threads_per_warp"])
    warps_per_cta = tuple(int(value) for value in properties["warps_per_cta"])
    order = tuple(int(value) for value in properties["order"])
    if not (
        len(size_per_thread)
        == len(threads_per_warp)
        == len(warps_per_cta)
        == len(order)
        == rank
    ):
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "blocked layout requires rank-matched metadata",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    linear = (
        _identity_standard_nd("register", size_per_thread, order)
        * _identity_standard_nd("lane", threads_per_warp, order)
        * _identity_standard_nd("warp", warps_per_cta, order)
    )
    return _ensure_layout_matches_shape(
        linear,
        shape,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _linear_encoding_layout(
    shape,
    properties,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    rank = len(shape)
    out_dims = [f"dim{dim}" for dim in range(rank)]
    bases = [
        ("register", [list(basis) for basis in properties.get("register_bases", ())]),
        ("lane", [list(basis) for basis in properties.get("lane_bases", ())]),
        ("warp", [list(basis) for basis in properties.get("warp_bases", ())]),
        ("block", [list(basis) for basis in properties.get("block_bases", ())]),
    ]
    for in_dim, in_bases in bases:
        for basis in in_bases:
            if len(basis) != rank:
                _layout_fail(
                    "TLXW_TYPE_MALFORMED_LAYOUT",
                    stage,
                    f"linear layout {in_dim} basis rank does not match tensor rank",
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
    return LinearLayout.from_bases(bases, out_dims, list(shape), False)


def _mfma_linear_layout(
    shape,
    properties,
    lane_width,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    if len(shape) != 2:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            "MFMA distributed layout currently requires rank-2 tensors",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    instr_shape = tuple(int(value) for value in properties.get("instr_shape", ()))
    if instr_shape not in {(16, 16, 32), (32, 32, 16)}:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            f"unsupported MFMA instruction shape {instr_shape}",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if not bool(properties.get("is_transposed", False)):
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            "non-transposed MFMA distributed layout is not implemented yet",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    element_bit_width = int(properties.get("element_bit_width", 32))
    height = 1 if element_bit_width == 64 else 4
    m_dim, n_dim = int(instr_shape[0]), int(instr_shape[1])
    warp_size = int(lane_width)
    tiles = (m_dim * n_dim) // (warp_size * height)
    if tiles <= 0:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            "MFMA distributed layout requires at least one register tile",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    dim_m = "dim0"
    dim_n = "dim1"
    linear = LinearLayout.identity_1d(height, "register", dim_n)
    linear *= (
        LinearLayout.identity_1d(m_dim, "lane", dim_m)
        * LinearLayout.identity_1d(warp_size // m_dim, "lane", dim_n)
    )
    linear *= LinearLayout.identity_1d(tiles, "register", dim_n)
    tiles_per_warp = tuple(int(value) for value in properties.get("tiles_per_warp", ()))
    if len(tiles_per_warp) < 2:
        tiles_per_warp = (1, 1)
    warps_per_cta = tuple(int(value) for value in properties.get("warps_per_cta", ()))
    if len(warps_per_cta) != 2:
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "MFMA distributed layout requires rank-2 warpsPerCTA metadata",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    tiles_per_warp_m = max(1, int(tiles_per_warp[0]))
    tiles_per_warp_n = max(1, int(tiles_per_warp[1]))
    warps_per_cta_m = max(1, int(warps_per_cta[0]))
    warps_per_cta_n = max(1, int(warps_per_cta[1]))
    linear *= LinearLayout.identity_1d(tiles_per_warp_n, "register", dim_n)
    linear *= LinearLayout.identity_1d(warps_per_cta_n, "warp", dim_n)
    n_remainder = shape[1] // (n_dim * warps_per_cta_n * tiles_per_warp_n)
    linear *= LinearLayout.identity_1d(max(1, n_remainder), "register", dim_n)
    linear *= LinearLayout.identity_1d(tiles_per_warp_m, "register", dim_m)
    linear *= LinearLayout.identity_1d(warps_per_cta_m, "warp", dim_m)
    return _ensure_layout_matches_shape(
        linear,
        shape,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _identity_standard_nd(in_dim, shape, order):
    linear = LinearLayout()
    for dim in order:
        linear *= LinearLayout.identity_1d(int(shape[dim]), in_dim, f"dim{dim}")
    return linear


def _ensure_layout_matches_shape(
    linear,
    shape,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    shape_by_dim = {f"dim{dim}": int(size) for dim, size in enumerate(shape)}
    linear = _ensure_layout_not_smaller_than(
        linear,
        shape_by_dim,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    return _ensure_layout_not_larger_than(linear, shape_by_dim)


def _ensure_layout_not_smaller_than(
    linear,
    shape_by_dim,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    for dim, desired_size in shape_by_dim.items():
        actual_size = linear_layout_out_dim_size(linear, dim, stage=stage)
        if desired_size > actual_size:
            if desired_size % actual_size:
                _layout_fail(
                    "TLXW_TYPE_UNSUPPORTED_LAYOUT",
                    stage,
                    "layout remap requires non-integral shape extension",
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
            linear *= LinearLayout.identity_1d(
                desired_size // actual_size,
                "register",
                dim,
            )
    return linear


def _ensure_layout_not_larger_than(linear, shape_by_dim):
    out_dims = []
    out_sizes = []
    for dim, size in linear.out_dims:
        resized = min(int(size), int(shape_by_dim[dim]))
        out_dims.append(dim)
        out_sizes.append(resized)
    bases = []
    for in_dim, in_bases in linear.bases:
        rewritten = []
        for basis in in_bases:
            basis = [int(value) for value in basis]
            was_zero = all(value == 0 for value in basis)
            for index, value in enumerate(tuple(basis)):
                if value >= out_sizes[index]:
                    basis[index] = 0
            is_zero = all(value == 0 for value in basis)
            if in_dim == "register":
                if was_zero or not is_zero:
                    rewritten.append(basis)
            else:
                rewritten.append(basis)
        bases.append((in_dim, rewritten))
    return LinearLayout.from_bases(
        bases,
        out_dims,
        out_sizes,
        False,
    )


def _layout_fail(
    code,
    stage,
    message,
    *,
    source_op_index=None,
    source_value_id=None,
):
    fail(
        code,
        stage,
        message,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


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
