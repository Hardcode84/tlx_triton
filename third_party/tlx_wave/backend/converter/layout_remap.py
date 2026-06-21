"""Structural layout-remap helpers for TLX Wave conversion."""

from triton._C.libtriton.linear_layout import LinearLayout

from .diagnostics import fail


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
    source_register_count = _linear_layout_in_dim_size(source_layout, "register")
    result_register_count = _linear_layout_in_dim_size(result_layout_ll, "register")
    source_registers_per_component = _mfma_registers_per_component(
        operand_layout,
        op,
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
    source_by_lane_coord = {}
    for source_register in range(source_register_count):
        for lane in range(lane_width):
            coords = _linear_layout_coords(
                source_layout,
                source_register,
                lane,
                warp=0,
            )
            key = (lane, coords)
            if key in source_by_lane_coord:
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    "MFMA convert_layout source layout is not injective "
                    "within a wave",
                    source_op_index=op.index,
                    source_value_id=operand.value_id,
                )
            source_by_lane_coord[key] = source_register

    source_indices = []
    source_element_indices = []
    for result_register in range(result_register_count):
        lane_sources = []
        for lane in range(lane_width):
            coords = _linear_layout_coords(
                result_layout_ll,
                result_register,
                lane,
                warp=0,
            )
            source_register = source_by_lane_coord.get((lane, coords))
            if source_register is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    "MFMA to blocked convert_layout requires cross-lane or "
                    "cross-warp movement; the TLX Wave converter only lowers "
                    "same-lane register remaps",
                    source_op_index=op.index,
                    source_value_id=result.value_id,
                )
            lane_sources.append(source_register)
        first_source = lane_sources[0]
        if any(source != first_source for source in lane_sources):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "MFMA to blocked convert_layout maps one result component "
                "to different source registers in different lanes; Wave "
                "cross-lane remap support is required",
                source_op_index=op.index,
                source_value_id=result.value_id,
            )
        source_indices.append(first_source // source_registers_per_component)
        source_element_indices.append(first_source % source_registers_per_component)

    return {
        "source_component_count": int(operand.type.component_count),
        "source_element_indices": tuple(source_element_indices),
        "source_indices": tuple(source_indices),
        "source_registers_per_component": int(source_registers_per_component),
    }


def _distributed_linear_layout(layout, op):
    if layout.kind == "blocked":
        return _blocked_linear_layout(layout, op)
    if layout.kind == "amd_mfma":
        return _mfma_linear_layout(layout, op)
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"layout {layout.kind} is not converted through linear-layout remap",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )


def _blocked_linear_layout(layout, op):
    shape = tuple(int(dim) for dim in layout.shape)
    rank = len(shape)
    size_per_thread = tuple(int(value) for value in layout.properties["size_per_thread"])
    threads_per_warp = tuple(
        int(value) for value in layout.properties["threads_per_warp"]
    )
    warps_per_cta = tuple(int(value) for value in layout.properties["warps_per_cta"])
    order = tuple(int(value) for value in layout.properties["order"])
    if not (
        len(size_per_thread)
        == len(threads_per_warp)
        == len(warps_per_cta)
        == len(order)
        == rank
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "blocked convert_layout requires rank-matched layout metadata",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    linear = (
        _identity_standard_nd("register", size_per_thread, order)
        * _identity_standard_nd("lane", threads_per_warp, order)
        * _identity_standard_nd("warp", warps_per_cta, order)
    )
    return _ensure_layout_matches_shape(linear, shape)


def _mfma_linear_layout(layout, op):
    shape = tuple(int(dim) for dim in layout.shape)
    if len(shape) != 2:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA convert_layout remap currently requires rank-2 tensors",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    instr_shape = tuple(int(value) for value in layout.properties.get("instr_shape", ()))
    if instr_shape not in {(16, 16, 32), (32, 32, 16)}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"unsupported MFMA convert_layout instruction shape {instr_shape}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    if not bool(layout.properties.get("is_transposed", False)):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "non-transposed MFMA convert_layout remap is not implemented yet",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    element_bit_width = int(layout.properties.get("element_bit_width", 32))
    height = 1 if element_bit_width == 64 else 4
    m_dim, n_dim = int(instr_shape[0]), int(instr_shape[1])
    warp_size = int(layout.lane_width)
    tiles = (m_dim * n_dim) // (warp_size * height)
    if tiles <= 0:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA convert_layout requires at least one register tile",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    dim_m = "dim0"
    dim_n = "dim1"
    linear = LinearLayout.identity_1d(height, "register", dim_n)
    linear *= (
        LinearLayout.identity_1d(m_dim, "lane", dim_m)
        * LinearLayout.identity_1d(warp_size // m_dim, "lane", dim_n)
    )
    linear *= LinearLayout.identity_1d(tiles, "register", dim_n)
    tiles_per_warp = tuple(
        int(value) for value in layout.properties.get("tiles_per_warp", ())
    )
    if len(tiles_per_warp) < 2:
        tiles_per_warp = (1, 1)
    warps_per_cta = tuple(
        int(value) for value in layout.properties.get("warps_per_cta", ())
    )
    if len(warps_per_cta) != 2:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA convert_layout requires rank-2 warpsPerCTA metadata",
            source_op_index=op.index,
            source_value_id=layout.value_id,
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
    return _ensure_layout_matches_shape(linear, shape)


def _identity_standard_nd(in_dim, shape, order):
    linear = LinearLayout()
    for dim in order:
        linear *= LinearLayout.identity_1d(int(shape[dim]), in_dim, f"dim{dim}")
    return linear


def _ensure_layout_matches_shape(linear, shape):
    shape_by_dim = {f"dim{dim}": int(size) for dim, size in enumerate(shape)}
    linear = _ensure_layout_not_smaller_than(linear, shape_by_dim)
    return _ensure_layout_not_larger_than(linear, shape_by_dim)


def _ensure_layout_not_smaller_than(linear, shape_by_dim):
    for dim, desired_size in shape_by_dim.items():
        actual_size = _linear_layout_out_dim_size(linear, dim)
        if desired_size > actual_size:
            if desired_size % actual_size:
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    "layout remap requires non-integral shape extension",
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


def _linear_layout_in_dim_size(linear, dim):
    for in_dim, bases in linear.bases:
        if in_dim == dim:
            return 1 << len(bases)
    return 1


def _linear_layout_out_dim_size(linear, dim):
    for out_dim, size in linear.out_dims:
        if out_dim == dim:
            return int(size)
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"linear layout is missing output dimension {dim}",
    )


def _linear_layout_coords(linear, register, lane, *, warp):
    coords = linear.apply(
        {
            "register": int(register),
            "lane": int(lane),
            "warp": int(warp),
        }
    )
    return tuple(int(coords[f"dim{dim}"]) for dim in range(len(coords)))


def _mfma_registers_per_component(layout, op):
    instr_shape = tuple(int(value) for value in layout.properties.get("instr_shape", ()))
    if instr_shape == (16, 16, 32):
        return 4
    if instr_shape == (32, 32, 16):
        return 16
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"unsupported MFMA register count for instrShape={instr_shape}",
        source_op_index=op.index,
        source_value_id=layout.value_id,
    )
