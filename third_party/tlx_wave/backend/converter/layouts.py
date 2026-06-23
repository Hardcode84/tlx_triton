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


@dataclass(frozen=True)
class PhysicalOffsetRecord:
    element_offset: int
    byte_offset: int
    dword_offset: int | None
    element_byte_width: int
    layout_kind: str
    order: tuple[int, ...]
    logical_coords: tuple[int, ...]
    logical_linear_offset: int
    bindings: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    proof_status: str = "static"
    provenance: str = "shared_physical_offset"


@dataclass(frozen=True)
class PhysicalOffsetExpressionPlan:
    expression_kind: str
    offset_unit: str
    element_byte_width: int
    layout_kind: str
    order: tuple[int, ...]
    bindings: tuple[str, ...] = ("logical_coords",)
    assumptions: tuple[str, ...] = ()
    proof_status: str = "symbolic_verified"
    provenance: str = "shared_physical_offset"
    intervals: tuple[int, ...] = ()
    paddings: tuple[int, ...] = ()
    swizzled_vec: int | None = None
    swizzled_per_phase: int | None = None
    swizzled_max_phase: int | None = None


def build_layout_map(layout_map_id, value_id, source_type, lane_width):
    if source_type.kind not in {"tensor", "memdesc"}:
        return None
    attr = source_type.encoding_attr
    kind, properties = _layout_kind_and_properties(
        attr,
        value_id,
        encoding=str(source_type.encoding or ""),
    )
    if kind in {"blocked", "linear", "generic_linear"}:
        coordinate_domain = _layout_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            lane_width,
            value_id,
        )
        _require_supported_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            coordinate_domain,
            value_id,
        )
        properties = {**properties, "coordinate_domain": coordinate_domain}
        component_count = int(coordinate_domain["component_count"])
    else:
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


def _layout_kind_and_properties(attr, value_id, *, encoding=None):
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
        kind = (
            "generic_linear"
            if str(encoding or "").startswith("#ttg.generic_linear")
            else "linear"
        )
        return kind, {
            "register_bases": _basis_tuple(_attr_value(attr, "get_linear_register_bases")),
            "lane_bases": _basis_tuple(_attr_value(attr, "get_linear_lane_bases")),
            "warp_bases": _basis_tuple(_attr_value(attr, "get_linear_warp_bases")),
            "block_bases": _basis_tuple(_attr_value(attr, "get_linear_block_bases")),
            "linear_encoding_kind": kind,
        }
    if _attr_bool(attr, "is_slice_encoding"):
        parent = _attr_value(attr, "get_slice_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(
            parent,
            value_id,
            encoding=str(parent or ""),
        )
        return "slice", {
            "dim": int(_attr_value(attr, "get_slice_dim")),
            "parent_kind": parent_kind,
            "parent_properties": parent_properties,
        }
    if _attr_bool(attr, "is_dot_operand_encoding"):
        parent = _attr_value(attr, "get_dot_operand_parent")
        parent_kind, parent_properties = _layout_kind_and_properties(
            parent,
            value_id,
            encoding=str(parent or ""),
        )
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
    if kind in {"blocked", "linear", "generic_linear"}:
        coordinate_domain = _layout_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            lane_width,
            source_value_id=value_id,
        )
        _require_supported_coordinate_domain(
            kind,
            source_type.shape,
            properties,
            coordinate_domain,
            value_id,
        )
        return int(coordinate_domain["component_count"])
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
    if kind == "slice":
        parent_kind = properties.get("parent_kind")
        parent_properties = properties.get("parent_properties", {})
        if parent_kind in {"blocked", "linear", "generic_linear"}:
            dim = int(properties.get("dim", 0))
            parent_shape = list(int(value) for value in source_type.shape)
            if dim < 0 or dim > len(parent_shape):
                _layout_fail(
                    "TLXW_TYPE_MALFORMED_LAYOUT",
                    STAGE,
                    "slice layout dimension is outside the parent rank",
                    source_value_id=value_id,
                )
            parent_shape.insert(dim, 1)
            linear = distributed_linear_layout_from_parts(
                parent_kind,
                tuple(parent_shape),
                parent_properties,
                lane_width,
                source_value_id=value_id,
            )
            return linear_layout_in_dim_size(linear, "register")
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
    if kind in {"linear", "generic_linear"}:
        return _linear_encoding_layout(
            shape,
            properties,
            stage=stage,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if kind == "slice":
        return _slice_linear_layout(
            shape,
            properties,
            lane_width,
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


def linear_layout_bases(linear, in_dim):
    for name, bases in linear.bases:
        if name == in_dim:
            return tuple(tuple(int(value) for value in basis) for basis in bases)
    return ()


def layout_warp_count(layout):
    if layout.kind in {"linear", "generic_linear"}:
        return 1 << len(tuple(layout.properties.get("warp_bases", ())))
    if layout.kind == "slice":
        return _layout_warp_count_from_parts(
            layout.properties.get("parent_kind"),
            layout.properties.get("parent_properties", {}),
        )
    warps_per_cta = tuple(
        int(value) for value in layout.properties.get("warps_per_cta", ())
    )
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result


def shared_physical_offset(
    layout,
    shape,
    coords,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    coords = tuple(int(coord) for coord in coords)
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared physical offset requires a positive element byte width",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if len(coords) != len(shape):
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared physical offset coordinate rank does not match shape rank",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    for coord, extent in zip(coords, shape):
        if int(coord) < 0 or int(coord) >= int(extent):
            _shared_layout_fail(
                diagnostic,
                stage,
                f"shared physical offset coordinate {coords} exceeds shape {shape}",
                layout=layout,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )

    kind = shared_layout_kind(layout)
    assumptions = ()
    if kind == "dense":
        order = default_physical_order(shape)
        element_offset = static_linear_offset(shape, coords)
        logical_linear_offset = element_offset
        provenance = "dense_row_major"
    elif kind == "swizzled_shared":
        if is_identity_swizzled_shared(layout):
            order = default_physical_order(shape)
            element_offset = static_linear_offset(shape, coords)
            logical_linear_offset = element_offset
            provenance = "identity_swizzled_row_major"
        else:
            order, vec, per_phase, max_phase = swizzled_shared_parameters(
                layout,
                shape,
                stage=stage,
                diagnostic=diagnostic,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
            minor_dim = int(order[0])
            major_dim = int(order[1])
            minor_extent = int(shape[minor_dim])
            major = int(coords[major_dim])
            minor = int(coords[minor_dim])
            phase = (major // int(per_phase)) % int(max_phase)
            swizzled_minor = ((minor // int(vec)) ^ phase) * int(vec) + (
                minor % int(vec)
            )
            if swizzled_minor >= minor_extent:
                _shared_layout_fail(
                    diagnostic,
                    stage,
                    "swizzled shared physical offset produces an "
                    f"out-of-bounds minor coordinate {swizzled_minor}; "
                    f"{swizzled_shared_description(layout)}",
                    layout=layout,
                    source_op_index=source_op_index,
                    source_value_id=source_value_id,
                )
            element_offset = major * minor_extent + swizzled_minor
            logical_linear_offset = ordered_linear_offset(shape, coords, order)
            provenance = "swizzled_shared"
            assumptions = ("minor_extent_divisible_by_vec",)
    elif kind == "padded_shared":
        intervals, paddings = padded_shared_parameters(
            layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        _require_identity_padded_linear_component(
            layout,
            shape,
            order,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        logical_linear_offset = ordered_linear_offset(shape, coords, order)
        element_offset = int(logical_linear_offset)
        for interval, padding in zip(intervals, paddings):
            element_offset += (logical_linear_offset // int(interval)) * int(padding)
        provenance = "padded_shared"
        assumptions = ("valid_padded_intervals",)
    else:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"shared physical offset does not support layout {kind}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )

    byte_offset = int(element_offset) * int(element_byte_width)
    dword_offset = byte_offset // 4 if byte_offset % 4 == 0 else None
    return PhysicalOffsetRecord(
        element_offset=int(element_offset),
        byte_offset=int(byte_offset),
        dword_offset=dword_offset,
        element_byte_width=int(element_byte_width),
        layout_kind=kind,
        order=tuple(int(dim) for dim in order),
        logical_coords=coords,
        logical_linear_offset=int(logical_linear_offset),
        assumptions=tuple(assumptions),
        provenance=provenance,
    )


def shared_physical_offset_from_linear(
    layout,
    shape,
    linear,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    linear = int(linear)
    if linear < 0 or linear >= _product(shape):
        return None
    coords = static_delinearize_row_major(
        linear,
        shape,
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
        layout=layout,
    )
    return shared_physical_offset(
        layout,
        shape,
        coords,
        int(element_byte_width),
        stage=stage,
        diagnostic=diagnostic,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def shared_physical_offset_expression_plan(
    layout,
    shape,
    element_byte_width,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    element_byte_width = int(element_byte_width)
    if element_byte_width <= 0:
        _shared_layout_fail(
            diagnostic,
            stage,
            "shared physical offset expression requires a positive element byte width",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    kind = shared_layout_kind(layout)
    if kind == "dense":
        return PhysicalOffsetExpressionPlan(
            expression_kind="dense_row_major",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=default_physical_order(shape),
            provenance="dense_row_major",
        )
    if kind == "swizzled_shared":
        if is_identity_swizzled_shared(layout):
            return PhysicalOffsetExpressionPlan(
                expression_kind="dense_row_major",
                offset_unit="element",
                element_byte_width=element_byte_width,
                layout_kind=kind,
                order=default_physical_order(shape),
                assumptions=("identity_swizzled_shared",),
                provenance="identity_swizzled_row_major",
            )
        order, vec, per_phase, max_phase = swizzled_shared_parameters(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        return PhysicalOffsetExpressionPlan(
            expression_kind="swizzled_xor",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=tuple(int(dim) for dim in order),
            assumptions=("minor_extent_divisible_by_vec",),
            provenance="swizzled_shared",
            swizzled_vec=int(vec),
            swizzled_per_phase=int(per_phase),
            swizzled_max_phase=int(max_phase),
        )
    if kind == "padded_shared":
        intervals, paddings = padded_shared_parameters(
            layout,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        order = shared_layout_physical_order(
            layout,
            shape,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        _require_identity_padded_linear_component(
            layout,
            shape,
            order,
            stage=stage,
            diagnostic=diagnostic,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
        return PhysicalOffsetExpressionPlan(
            expression_kind="padded_linear",
            offset_unit="element",
            element_byte_width=element_byte_width,
            layout_kind=kind,
            order=tuple(int(dim) for dim in order),
            assumptions=("valid_padded_intervals",),
            provenance="padded_shared",
            intervals=tuple(int(value) for value in intervals),
            paddings=tuple(int(value) for value in paddings),
        )
    _shared_layout_fail(
        diagnostic,
        stage,
        f"shared physical offset expression does not support layout {kind}",
        layout=layout,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def physical_offset_expression_plan_attrs(plan, prefix):
    prefix = str(prefix)
    attrs = {
        f"{prefix}_physical_offset_plan": plan.expression_kind,
        f"{prefix}_physical_offset_unit": plan.offset_unit,
        f"{prefix}_physical_element_byte_width": int(plan.element_byte_width),
        f"{prefix}_physical_layout_kind": plan.layout_kind,
        f"{prefix}_physical_order": tuple(int(dim) for dim in plan.order),
        f"{prefix}_physical_bindings": tuple(str(name) for name in plan.bindings),
        f"{prefix}_physical_assumptions": tuple(
            str(assumption) for assumption in plan.assumptions
        ),
        f"{prefix}_physical_proof_status": plan.proof_status,
        f"{prefix}_physical_provenance": plan.provenance,
    }
    if plan.expression_kind == "padded_linear" or plan.intervals:
        attrs[f"{prefix}_physical_intervals"] = tuple(
            int(value) for value in plan.intervals
        )
    if plan.expression_kind == "padded_linear" or plan.paddings:
        attrs[f"{prefix}_physical_paddings"] = tuple(
            int(value) for value in plan.paddings
        )
    if plan.swizzled_vec is not None:
        attrs[f"{prefix}_physical_swizzled_vec"] = int(plan.swizzled_vec)
    if plan.swizzled_per_phase is not None:
        attrs[f"{prefix}_physical_swizzled_per_phase"] = int(
            plan.swizzled_per_phase
        )
    if plan.swizzled_max_phase is not None:
        attrs[f"{prefix}_physical_swizzled_max_phase"] = int(
            plan.swizzled_max_phase
        )
    return attrs


def shared_layout_kind(layout):
    if layout is None or layout.kind == "none":
        return "dense"
    if layout.kind in {"linear", "generic_linear"}:
        return "linear_shared"
    return str(layout.kind)


def default_physical_order(shape):
    return tuple(reversed(range(len(tuple(shape)))))


def expand_physical_order(
    order,
    rank,
    *,
    layout=None,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    order = tuple(int(dim) for dim in order)
    rank = int(rank)
    if len(order) > rank or sorted(order) != list(range(len(order))):
        _shared_layout_fail(
            diagnostic,
            stage,
            f"shared layout order {order} cannot be applied to rank-{rank} shape",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    prefix_rank = rank - len(order)
    mapped = tuple(prefix_rank + int(dim) for dim in order)
    return mapped + tuple(reversed(range(prefix_rank)))


def shared_layout_physical_order(
    layout,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    shape = tuple(int(dim) for dim in shape)
    if not shape:
        return ()
    if layout is not None and layout.kind == "padded_shared":
        order = tuple(int(dim) for dim in layout.properties.get("order", ()))
        if order:
            return expand_physical_order(
                order,
                len(shape),
                layout=layout,
                stage=stage,
                diagnostic=diagnostic,
                source_op_index=source_op_index,
                source_value_id=source_value_id,
            )
    return default_physical_order(shape)


def _require_identity_padded_linear_component(
    layout,
    shape,
    order,
    *,
    stage,
    diagnostic,
    source_op_index,
    source_value_id,
):
    if tuple(order) == default_physical_order(shape):
        return
    _shared_layout_fail(
        diagnostic,
        stage,
        "non-identity padded shared physical offsets require the full "
        f"linearComponent; {padded_shared_description(layout)}",
        layout=layout,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def static_linear_offset(shape, coords):
    offset = 0
    shape = tuple(int(dim) for dim in shape)
    for dim, coord in enumerate(coords):
        stride = _product(shape[dim + 1 :])
        offset += int(coord) * stride
    return int(offset)


def ordered_linear_offset(shape, coords, order):
    offset = 0
    stride = 1
    shape = tuple(int(dim) for dim in shape)
    for dim in order:
        offset += int(coords[int(dim)]) * stride
        stride *= int(shape[int(dim)])
    return int(offset)


def ordered_coords_from_linear(linear, shape, order):
    coords = [0] * len(shape)
    remainder = int(linear)
    for dim in order:
        extent = int(shape[int(dim)])
        coords[int(dim)] = remainder % extent
        remainder //= extent
    return tuple(coords)


def static_delinearize_row_major(
    linear,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
    layout=None,
):
    shape = tuple(int(dim) for dim in shape)
    coords = [0] * len(shape)
    remainder = int(linear)
    for dim in reversed(range(len(shape))):
        extent = int(shape[dim])
        coords[dim] = remainder % extent
        remainder //= extent
    if remainder:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"linear index {linear} exceeds shape {shape}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return tuple(coords)


def swizzled_shared_parameters(
    layout,
    shape,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    order = tuple(int(dim) for dim in layout.properties.get("order", ()))
    shape = tuple(int(dim) for dim in shape)
    if len(shape) != 2 or order not in {(1, 0), (0, 1)}:
        _shared_layout_fail(
            diagnostic,
            stage,
            "swizzled shared physical offsets support only rank-2 "
            f"order=[1,0] or order=[0,1]; got {swizzled_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    vec = int(layout.properties["vec"])
    per_phase = int(layout.properties["per_phase"])
    max_phase = int(layout.properties["max_phase"])
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        _shared_layout_fail(
            diagnostic,
            stage,
            "swizzled shared layout requires positive vec/perPhase/maxPhase; "
            f"got {swizzled_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    minor_extent = int(shape[int(order[0])])
    if minor_extent % vec:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"swizzled shared minor extent {minor_extent} is not divisible "
            f"by vec={vec}; {swizzled_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return order, vec, per_phase, max_phase


def padded_shared_parameters(
    layout,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    if tuple(layout.properties.get("order", ())) not in {(0, 1), (1, 0), (0,), ()}:
        _shared_layout_fail(
            diagnostic,
            stage,
            f"unsupported padded shared order; {padded_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    intervals = tuple(int(value) for value in layout.properties.get("intervals", ()))
    paddings = tuple(int(value) for value in layout.properties.get("paddings", ()))
    if len(intervals) != len(paddings):
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared layout requires matching interval/padding counts; "
            f"{padded_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    if any(interval <= 0 for interval in intervals):
        _shared_layout_fail(
            diagnostic,
            stage,
            "padded shared intervals must be positive; "
            f"{padded_shared_description(layout)}",
            layout=layout,
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    return intervals, paddings


def is_identity_swizzled_shared(layout):
    props = layout.properties
    order = tuple(props.get("order", ()))
    return (
        int(props.get("vec", 0)) == 1
        and int(props.get("per_phase", 0)) == 1
        and int(props.get("max_phase", 0)) == 1
        and order in {(1, 0), (0,), ()}
    )


def require_identity_swizzled_shared(
    layout,
    *,
    stage=STAGE,
    diagnostic="TLXW_TYPE_UNSUPPORTED_LAYOUT",
    source_op_index=None,
    source_value_id=None,
):
    props = layout.properties
    if (
        int(props.get("vec", 0)) == 1
        and int(props.get("per_phase", 0)) == 1
        and int(props.get("max_phase", 0)) == 1
    ):
        return
    _shared_layout_fail(
        diagnostic,
        stage,
        "swizzled shared layout requires an explicit remap target op",
        layout=layout,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def swizzled_shared_description(layout):
    props = layout.properties
    return (
        f"order={tuple(props.get('order', ()))}, "
        f"vec={int(props.get('vec', 0))}, "
        f"per_phase={int(props.get('per_phase', 0))}, "
        f"max_phase={int(props.get('max_phase', 0))}"
    )


def padded_shared_description(layout):
    props = layout.properties
    return (
        f"order={tuple(props.get('order', ()))}, "
        f"intervals={tuple(int(value) for value in props.get('intervals', ()))}, "
        f"paddings={tuple(int(value) for value in props.get('paddings', ()))}"
    )


def _shared_layout_fail(
    code,
    stage,
    message,
    *,
    layout=None,
    source_op_index=None,
    source_value_id=None,
):
    if source_value_id is None and layout is not None:
        source_value_id = layout.value_id
    fail(
        code,
        stage,
        message,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )


def _layout_coordinate_domain(kind, shape, properties, lane_width, source_value_id):
    linear = distributed_linear_layout_from_parts(
        kind,
        shape,
        properties,
        lane_width,
        source_value_id=source_value_id,
    )
    component_count = linear_layout_in_dim_size(linear, "register")
    warp_count = _layout_warp_count_from_parts(kind, properties)
    block_count = linear_layout_in_dim_size(linear, "block")
    shape = tuple(int(dim) for dim in shape)
    total_elements = _product(shape)
    seen = set()
    duplicate_slots = 0
    out_of_bounds_slots = 0
    for component in range(int(component_count)):
        for warp in range(int(warp_count)):
            for lane in range(int(lane_width)):
                coords = linear_layout_coords(linear, component, lane, warp=warp)
                if len(coords) != len(shape) or any(
                    int(coord) < 0 or int(coord) >= int(extent)
                    for coord, extent in zip(coords, shape)
                ):
                    out_of_bounds_slots += 1
                    continue
                if coords in seen:
                    duplicate_slots += 1
                seen.add(coords)
    physical_slots = int(component_count) * int(lane_width) * int(warp_count)
    if int(block_count) <= 0 or total_elements % int(block_count):
        coverage = "block_mismatch"
        local_elements = total_elements
    else:
        local_elements = total_elements // int(block_count)
        if out_of_bounds_slots:
            coverage = "out_of_bounds"
        elif len(seen) == local_elements and duplicate_slots == 0:
            coverage = "exact"
        elif len(seen) == local_elements:
            coverage = "replicated"
        elif duplicate_slots:
            coverage = "duplicate_partial"
        else:
            coverage = "partial"
    return {
        "coverage": coverage,
        "component_count": int(component_count),
        "covered_elements": int(len(seen)),
        "duplicate_slots": int(duplicate_slots),
        "local_elements": int(local_elements),
        "physical_slots": int(physical_slots),
        "out_of_bounds_slots": int(out_of_bounds_slots),
        "block_count": int(block_count),
    }


def _require_supported_coordinate_domain(
    kind,
    shape,
    properties,
    coordinate_domain,
    source_value_id,
):
    if coordinate_domain["coverage"] in {"exact", "replicated"}:
        return
    _layout_fail(
        "TLXW_TYPE_UNSUPPORTED_LAYOUT",
        STAGE,
        "unsupported distributed layout coordinate domain "
        f"{coordinate_domain['coverage']}; kind={kind} shape={tuple(shape)} "
        f"domain={coordinate_domain} bases={_basis_pattern(kind, properties)}",
        source_value_id=source_value_id,
    )


def _layout_warp_count_from_parts(kind, properties):
    if kind in {"linear", "generic_linear"}:
        return 1 << len(tuple(properties.get("warp_bases", ())))
    if kind == "slice":
        return _layout_warp_count_from_parts(
            properties.get("parent_kind"),
            properties.get("parent_properties", {}),
        )
    warps_per_cta = tuple(int(value) for value in properties.get("warps_per_cta", ()))
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result


def _basis_pattern(kind, properties):
    if kind in {"linear", "generic_linear"}:
        return {
            "register": tuple(properties.get("register_bases", ())),
            "lane": tuple(properties.get("lane_bases", ())),
            "warp": tuple(properties.get("warp_bases", ())),
            "block": tuple(properties.get("block_bases", ())),
        }
    if kind == "blocked":
        return {
            "size_per_thread": tuple(properties.get("size_per_thread", ())),
            "threads_per_warp": tuple(properties.get("threads_per_warp", ())),
            "warps_per_cta": tuple(properties.get("warps_per_cta", ())),
            "order": tuple(properties.get("order", ())),
        }
    return dict(properties)


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


def _slice_linear_layout(
    shape,
    properties,
    lane_width,
    *,
    stage,
    source_op_index,
    source_value_id,
):
    parent_kind = properties.get("parent_kind")
    parent_properties = properties.get("parent_properties", {})
    if parent_kind not in {"blocked", "linear", "generic_linear"}:
        _layout_fail(
            "TLXW_TYPE_UNSUPPORTED_LAYOUT",
            stage,
            f"slice parent layout {parent_kind} does not have a distributed register map",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    dim = int(properties.get("dim", 0))
    shape = tuple(int(value) for value in shape)
    parent_shape = list(shape)
    if dim < 0 or dim > len(parent_shape):
        _layout_fail(
            "TLXW_TYPE_MALFORMED_LAYOUT",
            stage,
            "slice layout dimension is outside the parent rank",
            source_op_index=source_op_index,
            source_value_id=source_value_id,
        )
    parent_shape.insert(dim, 1)
    parent = distributed_linear_layout_from_parts(
        parent_kind,
        tuple(parent_shape),
        parent_properties,
        lane_width,
        stage=stage,
        source_op_index=source_op_index,
        source_value_id=source_value_id,
    )
    parent_dim_to_basis_index = {
        str(name): index for index, (name, _size) in enumerate(parent.out_dims)
    }
    kept_parent_dims = [
        f"dim{index}" for index in range(len(parent_shape)) if index != dim
    ]
    out_dims = [f"dim{index}" for index in range(len(shape))]
    bases = []
    for in_dim, in_bases in parent.bases:
        projected = []
        for basis in in_bases:
            basis = tuple(int(value) for value in basis)
            projected.append(
                [
                    basis[parent_dim_to_basis_index[parent_dim]]
                    for parent_dim in kept_parent_dims
                ]
            )
        bases.append((in_dim, projected))
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
    if bool(properties.get("is_transposed", False)):
        linear = LinearLayout.identity_1d(height, "register", dim_n)
        linear *= (
            LinearLayout.identity_1d(m_dim, "lane", dim_m)
            * LinearLayout.identity_1d(warp_size // m_dim, "lane", dim_n)
        )
        linear *= LinearLayout.identity_1d(tiles, "register", dim_n)
    else:
        linear = LinearLayout.identity_1d(height, "register", dim_m)
        linear *= (
            LinearLayout.identity_1d(n_dim, "lane", dim_n)
            * LinearLayout.identity_1d(warp_size // n_dim, "lane", dim_m)
        )
        linear *= LinearLayout.identity_1d(tiles, "register", dim_m)
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
