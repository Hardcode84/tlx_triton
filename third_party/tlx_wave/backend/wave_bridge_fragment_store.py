"""Fragment store lowering helpers for the TLX Wave bridge.

This module is intentionally narrow: it owns stores of MFMA fragments to global
and local memory.  The main bridge configures it with the shared layout, mask,
and pointer helpers after those helpers are defined, keeping this extraction
mechanical while reducing the size of ``wave_bridge_emit.py``.
"""

from dataclasses import replace


_DEPENDENCY_NAMES = (
    "_GFX950_DOT_PARENT_LAYOUT",
    "_GFX950_MMA32_INFO",
    "_GFX950_MMA_SHAPE",
    "_MaskAnd",
    "_MaskCompare",
    "_MaskConst",
    "_amd_mfma_encoding_info",
    "_assume_nonnegative",
    "_blocked_encoding_info",
    "_blocked_layout_static_coord",
    "_delinearize_expr",
    "_dim_symbol",
    "_emit_memdesc_ptr",
    "_fragment_tile_shape_for_rank2_shape",
    "_fragment_tuple_index",
    "_is_supported_fragment_store_layout",
    "_ixsimpl_assume_fact_exprs",
    "_ixsimpl_expr_equal",
    "_ixsimpl_index_expr",
    "_ixsimpl_mask_compare_expr",
    "_ixsimpl_mask_expr",
    "_ixsimpl_mod_zero",
    "_ixsimpl_pointer_offset_expr",
    "_ixsimpl_predicate_equivalent",
    "_materialize_bounded_pointer_value",
    "_materialize_mask_value",
    "_maybe_splat",
    "_mfma32_accumulator_dim_exprs",
    "_mma_shape_for_result_value",
    "_product",
    "_require_lowered_value",
    "_same_blocked_encoding",
    "_store_dim_bindings",
    "_wave_cmpi",
    "_wave_element_type",
    "_wave_mask_and",
)


def configure_fragment_store_dependencies(namespace):
    missing = [name for name in _DEPENDENCY_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "tlx_wave fragment-store helper is missing bridge dependencies: "
            + ", ".join(missing)
        )
    for name in _DEPENDENCY_NAMES:
        globals()[name] = namespace[name]


def _validate_fragment_store_value(value_plan, physical_plan):
    if (
        value_plan.element_type not in {"f32", "f16"}
        or value_plan.shape != _GFX950_MMA_SHAPE
    ):
        raise ValueError(
            "tlx_wave bridge supports fragment tt.store only for f32/f16 32x32 "
            f"values, got type={value_plan.type}, encoding={value_plan.encoding}"
        )
    if physical_plan.element_type != "f32" or physical_plan.shape != value_plan.shape:
        raise ValueError(
            "tlx_wave bridge cannot store fragment through a layout conversion "
            "that does not preserve an f32 physical accumulator with the same "
            "shape: "
            f"physical type={physical_plan.type}, store type={value_plan.type}"
        )
    physical_layout = None
    try:
        physical_layout = _blocked_encoding_info(
            physical_plan.encoding_attr,
            physical_plan.encoding,
            "fragment store physical value",
        )
    except ValueError:
        pass
    if physical_layout is not None:
        if not _is_supported_fragment_store_layout(physical_layout):
            raise ValueError(
                "tlx_wave bridge cannot lower fragment store: unsupported "
                "blocked fragment layout; expected the gfx950 dot-parent "
                "layout or a known propagated store layout, got "
                f"sizePerThread={physical_layout.size_per_thread}, "
                f"threadsPerWarp={physical_layout.threads_per_warp}, "
                f"warpsPerCTA={physical_layout.warps_per_cta}, "
                f"order={physical_layout.order}"
            )
        return replace(
            physical_plan,
            type=value_plan.type,
            element_type=value_plan.element_type,
            element_byte_width=value_plan.element_byte_width,
        )
    try:
        mfma = _amd_mfma_encoding_info(
            physical_plan.encoding,
            "fragment store physical value",
        )
    except ValueError as exc:
        raise ValueError(f"tlx_wave bridge cannot lower fragment store: {exc}") from exc
    if mfma is None:
        raise ValueError(
            "tlx_wave bridge expected blocked or #ttg.amd_mfma encoding for "
            f"fragment store physical value, got {physical_plan.encoding}"
        )
    value_layout = _blocked_encoding_info(
        value_plan.encoding_attr,
        value_plan.encoding,
        "fragment store value",
    )
    if not _is_supported_fragment_store_layout(value_layout):
        raise ValueError(
            "tlx_wave bridge cannot lower fragment store through #ttg.amd_mfma "
            "using unsupported blocked store layout; expected the gfx950 "
            "dot-parent layout or a known propagated store layout, got "
            f"sizePerThread={value_layout.size_per_thread}, "
            f"threadsPerWarp={value_layout.threads_per_warp}, "
            f"warpsPerCTA={value_layout.warps_per_cta}, "
            f"order={value_layout.order}"
        )
    return value_plan


def _fragment_unpack_as(builder, fragment, element_type, w):
    if element_type == "i32":
        return builder.fragment_unpack(fragment)
    waveamd = getattr(w, "waveamd", None)
    if waveamd is None:
        raise RuntimeError(
            "tlx_wave bridge requires mlir.dialects.wave_dsl to expose the "
            "generated waveamd dialect module"
        )
    frag = w.FragmentType(fragment.type)
    result_type = w.simd_type(
        w.vector_type(
            frag.registers,
            _wave_element_type(element_type, w, "fragment unpack"),
        ),
        width=frag.wave_size,
    )
    return waveamd.FragmentUnpackOp(result_type, fragment).result


def _fragment_store_unpack_element_type(value_plan):
    if value_plan.element_type == "f16":
        return "f32"
    return "i32"


def _convert_fragment_store_value(builder, value, value_plan, width, w):
    if value_plan.element_type == "f16":
        return builder.fpconvert(value, w.simd_type(w.f16(), width))
    return value


def _extract_converted_fragment_store_value(
    builder,
    regs,
    component,
    count,
    width,
    value_plan,
    w,
    unpack_element_type,
):
    if value_plan.element_type == "f16" and count == 2:
        value = _extract_fragment_store_value(
            regs, component, count, width, w, unpack_element_type
        )
        value_type = w.simd_type(w.vector_type(count, w.f16()), width)
        return builder.fpconvert(value, value_type)
    if value_plan.element_type == "f16" and count > 1:
        wave = getattr(w, "wave", None)
        if wave is None:
            raise RuntimeError(
                "tlx_wave bridge requires mlir.dialects.wave_dsl to expose the "
                "generated wave dialect module"
            )
        values = []
        for index in range(count):
            scalar = _extract_fragment_component(
                regs, component + index, width, w, unpack_element_type
            )
            values.append(
                _convert_fragment_store_value(builder, scalar, value_plan, width, w)
            )
        value_type = w.simd_type(w.vector_type(count, w.f16()), width)
        return wave.PackOp(value_type, values).result
    value = _extract_fragment_store_value(
        regs, component, count, width, w, unpack_element_type
    )
    return _convert_fragment_store_value(builder, value, value_plan, width, w)


def _extract_fragment_component(regs, component, width, w, element_type="i32"):
    wave = getattr(w, "wave", None)
    if wave is None:
        raise RuntimeError(
            "tlx_wave bridge requires mlir.dialects.wave_dsl to expose the "
            "generated wave dialect module"
        )
    return wave.ExtractOp(
        w.simd_type(
            _wave_element_type(element_type, w, "fragment component extract"),
            width,
        ),
        regs,
        component,
    ).result


def _mfma_fragment_store_vector_width(value_plan, frag):
    if (
        value_plan.element_type != "f32"
        or len(value_plan.shape) != 2
        or frag.registers != _GFX950_MMA32_INFO.acc_registers
    ):
        return 1
    if value_plan.element_byte_width is None:
        raise ValueError(
            "tlx_wave bridge cannot vectorize MFMA fragment store: unknown "
            f"element byte width for {value_plan.type}"
        )
    element_byte_width = int(value_plan.element_byte_width)
    if element_byte_width <= 0 or 16 % element_byte_width:
        raise ValueError(
            "tlx_wave bridge cannot vectorize MFMA fragment store: unsupported "
            f"element byte width {element_byte_width} for {value_plan.type}"
        )
    vector_width = 16 // element_byte_width
    if frag.registers % vector_width:
        return 1
    return vector_width


def _mfma32_component_dim_exprs(value_plan, component, w):
    thread_sym = w.sym(f"tlx_store_{value_plan.value_id}_thread")
    coords = _mfma32_accumulator_dim_exprs(thread_sym, component, w)
    return {
        _dim_symbol(w, dim): w.mod(coords[dim], value_plan.shape[dim])
        for dim in range(len(value_plan.shape))
    }


def _store_component_pointer_offset_expr(pointer_offset, value_plan, component, w):
    expr = pointer_offset
    for symbol, replacement in _mfma32_component_dim_exprs(
        value_plan, component, w
    ).items():
        expr = expr.subs(symbol, replacement)
    return expr


def _store_component_expr(expr, value_plan, component, w):
    for symbol, replacement in _mfma32_component_dim_exprs(
        value_plan, component, w
    ).items():
        expr = expr.subs(symbol, replacement)
    return expr


def _mfma32_store_pointer_components_contiguous(
    pointer_source, value_plan, component, count, w
):
    if count <= 1:
        return True
    unknowns = {}
    pointer_offset = _ixsimpl_pointer_offset_expr(pointer_source, w, unknowns)
    if pointer_offset is None:
        return False
    assumptions = ()
    first = _store_component_pointer_offset_expr(
        pointer_offset, value_plan, component, w
    )
    for index in range(1, count):
        candidate = _store_component_pointer_offset_expr(
            pointer_offset, value_plan, component + index, w
        )
        if not _ixsimpl_expr_equal(candidate, first + index, assumptions, w):
            return False
    return True


def _store_component_index_expr(source, value_plan, component, w, unknowns):
    expr = _ixsimpl_index_expr(source, w, unknowns)
    if expr is None:
        return None
    return _store_component_expr(expr, value_plan, component, w)


def _mfma32_store_component_slt_uniform(
    first_lhs,
    first_rhs,
    lhs,
    rhs,
    index,
    count,
    assumptions,
    w,
):
    if not _ixsimpl_expr_equal(lhs, first_lhs + int(index), assumptions, w):
        return False
    if not _ixsimpl_expr_equal(rhs, first_rhs, assumptions, w):
        return False
    return _ixsimpl_mod_zero(first_lhs, count, assumptions, w) and _ixsimpl_mod_zero(
        first_rhs, count, assumptions, w
    )


def _mfma32_store_mask_compare_components_uniform(
    source,
    value_plan,
    component,
    count,
    assumptions,
    w,
    unknowns,
):
    first_lhs = _store_component_index_expr(
        source.lhs, value_plan, component, w, unknowns
    )
    first_rhs = _store_component_index_expr(
        source.rhs, value_plan, component, w, unknowns
    )
    if first_lhs is None or first_rhs is None:
        return False
    first = _ixsimpl_mask_compare_expr(source.predicate, first_lhs, first_rhs, w)
    if first is None:
        return False
    for index in range(1, count):
        lhs = _store_component_index_expr(
            source.lhs, value_plan, component + index, w, unknowns
        )
        rhs = _store_component_index_expr(
            source.rhs, value_plan, component + index, w, unknowns
        )
        if lhs is None or rhs is None:
            return False
        candidate = _ixsimpl_mask_compare_expr(source.predicate, lhs, rhs, w)
        if candidate is None:
            return False
        if _ixsimpl_predicate_equivalent(first, candidate, assumptions, w):
            continue
        if source.predicate == "slt" and _mfma32_store_component_slt_uniform(
            first_lhs, first_rhs, lhs, rhs, index, count, assumptions, w
        ):
            continue
        if source.predicate == "sgt" and _mfma32_store_component_slt_uniform(
            first_rhs, first_lhs, rhs, lhs, index, count, assumptions, w
        ):
            continue
        return False
    return True


def _mfma32_store_mask_components_uniform(
    state, mask_source, value_plan, component, count, w
):
    if mask_source is None or count <= 1:
        return True
    if isinstance(mask_source, _MaskConst):
        return True
    unknowns = {}
    assumptions = _ixsimpl_assume_fact_exprs(state, unknowns, w)
    if isinstance(mask_source, _MaskAnd):
        return _mfma32_store_mask_components_uniform(
            state, mask_source.lhs, value_plan, component, count, w
        ) and _mfma32_store_mask_components_uniform(
            state, mask_source.rhs, value_plan, component, count, w
        )
    if isinstance(mask_source, _MaskCompare):
        return _mfma32_store_mask_compare_components_uniform(
            mask_source, value_plan, component, count, assumptions, w, unknowns
        )
    mask = _ixsimpl_mask_expr(mask_source, w, unknowns)
    if mask is None:
        return False
    first = _store_component_expr(mask, value_plan, component, w)
    for index in range(1, count):
        candidate = _store_component_expr(mask, value_plan, component + index, w)
        if not _ixsimpl_predicate_equivalent(first, candidate, assumptions, w):
            return False
    return True


def _mfma_fragment_store_vector_width_for_store(
    state, pointer_source, mask_source, value_plan, frag, component, w
):
    vector_width = _mfma_fragment_store_vector_width(value_plan, frag)
    if component % vector_width or component + vector_width > frag.registers:
        return 1
    if not _mfma32_store_pointer_components_contiguous(
        pointer_source, value_plan, component, vector_width, w
    ):
        return 1
    if not _mfma32_store_mask_components_uniform(
        state, mask_source, value_plan, component, vector_width, w
    ):
        return 1
    return vector_width


def _fragment_store_vector_width_candidates(value_plan, frag, component):
    if (
        value_plan.element_type not in {"f32", "f16"}
        or len(value_plan.shape) != 2
        or value_plan.element_byte_width is None
    ):
        return ()
    element_byte_width = int(value_plan.element_byte_width)
    if element_byte_width <= 0:
        return ()
    max_count = min(int(frag.registers) - int(component), 16 // element_byte_width)
    candidates = []
    count = 1
    while count * 2 <= max_count:
        count *= 2
    while count > 1:
        if component % count == 0 and (count * element_byte_width) % 4 == 0:
            candidates.append(count)
        count //= 2
    return tuple(candidates)


def _fragment_store_tile_component_dim_exprs(
    value_plan,
    fragment,
    tile_offsets,
    component,
    w,
):
    frag = w.FragmentType(fragment.type)
    suffix = "_".join(str(int(offset)) for offset in tile_offsets)
    thread_sym = w.sym(f"tlx_store_{value_plan.value_id}_{suffix}_thread")
    if frag.registers == _GFX950_MMA32_INFO.acc_registers:
        coords = _mfma32_accumulator_dim_exprs(thread_sym, component, w)
        return {
            _dim_symbol(w, dim): int(tile_offsets[dim]) + coords[dim]
            for dim in range(len(value_plan.shape))
        }

    layout = _GFX950_DOT_PARENT_LAYOUT
    threads_per_warp = _product(layout.threads_per_warp)
    register_coords = _delinearize_expr(
        w, w.sym_ctx.int_(component), layout.size_per_thread, layout.order
    )
    lane_coords = _delinearize_expr(
        w,
        w.mod(thread_sym, threads_per_warp),
        layout.threads_per_warp,
        layout.order,
    )
    warp_coords = _delinearize_expr(
        w,
        w.floor(thread_sym / threads_per_warp),
        layout.warps_per_cta,
        layout.order,
    )
    return {
        _dim_symbol(w, dim): int(tile_offsets[dim])
        + register_coords[dim]
        + layout.size_per_thread[dim]
        * (lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim])
        for dim in range(len(value_plan.shape))
    }


def _fragment_store_tile_component_expr(
    expr,
    value_plan,
    fragment,
    tile_offsets,
    component,
    w,
):
    for symbol, replacement in _fragment_store_tile_component_dim_exprs(
        value_plan, fragment, tile_offsets, component, w
    ).items():
        expr = expr.subs(symbol, replacement)
    return expr


def _fragment_store_tile_pointer_components_contiguous(
    pointer_source,
    value_plan,
    fragment,
    tile_offsets,
    component,
    count,
    w,
):
    if count <= 1:
        return True
    unknowns = {}
    pointer_offset = _ixsimpl_pointer_offset_expr(pointer_source, w, unknowns)
    if pointer_offset is None:
        return False
    assumptions = ()
    first = _fragment_store_tile_component_expr(
        pointer_offset, value_plan, fragment, tile_offsets, component, w
    )
    for index in range(1, count):
        candidate = _fragment_store_tile_component_expr(
            pointer_offset,
            value_plan,
            fragment,
            tile_offsets,
            component + index,
            w,
        )
        if not _ixsimpl_expr_equal(candidate, first + index, assumptions, w):
            return False
    return True


def _fragment_store_tile_component_index_expr(
    source,
    value_plan,
    fragment,
    tile_offsets,
    component,
    w,
    unknowns,
):
    expr = _ixsimpl_index_expr(source, w, unknowns)
    if expr is None:
        return None
    return _fragment_store_tile_component_expr(
        expr, value_plan, fragment, tile_offsets, component, w
    )


def _fragment_store_tile_mask_compare_components_uniform(
    source,
    value_plan,
    fragment,
    tile_offsets,
    component,
    count,
    assumptions,
    w,
    unknowns,
):
    first_lhs = _fragment_store_tile_component_index_expr(
        source.lhs, value_plan, fragment, tile_offsets, component, w, unknowns
    )
    first_rhs = _fragment_store_tile_component_index_expr(
        source.rhs, value_plan, fragment, tile_offsets, component, w, unknowns
    )
    if first_lhs is None or first_rhs is None:
        return False
    first = _ixsimpl_mask_compare_expr(source.predicate, first_lhs, first_rhs, w)
    if first is None:
        return False
    for index in range(1, count):
        lhs = _fragment_store_tile_component_index_expr(
            source.lhs,
            value_plan,
            fragment,
            tile_offsets,
            component + index,
            w,
            unknowns,
        )
        rhs = _fragment_store_tile_component_index_expr(
            source.rhs,
            value_plan,
            fragment,
            tile_offsets,
            component + index,
            w,
            unknowns,
        )
        if lhs is None or rhs is None:
            return False
        candidate = _ixsimpl_mask_compare_expr(source.predicate, lhs, rhs, w)
        if candidate is None:
            return False
        if _ixsimpl_predicate_equivalent(first, candidate, assumptions, w):
            continue
        if source.predicate == "slt" and _mfma32_store_component_slt_uniform(
            first_lhs, first_rhs, lhs, rhs, index, count, assumptions, w
        ):
            continue
        if source.predicate == "sgt" and _mfma32_store_component_slt_uniform(
            first_rhs, first_lhs, rhs, lhs, index, count, assumptions, w
        ):
            continue
        return False
    return True


def _fragment_store_tile_mask_components_uniform(
    state,
    mask_source,
    value_plan,
    fragment,
    tile_offsets,
    component,
    count,
    w,
):
    if mask_source is None or count <= 1:
        return True
    if isinstance(mask_source, _MaskConst):
        return True
    unknowns = {}
    assumptions = _ixsimpl_assume_fact_exprs(state, unknowns, w)
    if isinstance(mask_source, _MaskAnd):
        return _fragment_store_tile_mask_components_uniform(
            state,
            mask_source.lhs,
            value_plan,
            fragment,
            tile_offsets,
            component,
            count,
            w,
        ) and _fragment_store_tile_mask_components_uniform(
            state,
            mask_source.rhs,
            value_plan,
            fragment,
            tile_offsets,
            component,
            count,
            w,
        )
    if isinstance(mask_source, _MaskCompare):
        return _fragment_store_tile_mask_compare_components_uniform(
            mask_source,
            value_plan,
            fragment,
            tile_offsets,
            component,
            count,
            assumptions,
            w,
            unknowns,
        )
    mask = _ixsimpl_mask_expr(mask_source, w, unknowns)
    if mask is None:
        return False
    first = _fragment_store_tile_component_expr(
        mask, value_plan, fragment, tile_offsets, component, w
    )
    for index in range(1, count):
        candidate = _fragment_store_tile_component_expr(
            mask, value_plan, fragment, tile_offsets, component + index, w
        )
        if not _ixsimpl_predicate_equivalent(first, candidate, assumptions, w):
            return False
    return True


def _fragment_store_tile_components_all_lanes_active(
    value_plan,
    fragment,
    tile_offsets,
    component,
    count,
    w,
):
    return all(
        _fragment_store_tile_component_all_lanes_active(
            value_plan, fragment, tile_offsets, component + index, w
        )
        for index in range(count)
    )


def _fragment_store_tile_vector_width_for_store(
    state,
    pointer_source,
    mask_source,
    value_plan,
    fragment,
    tile_offsets,
    component,
    w,
):
    frag = w.FragmentType(fragment.type)
    for count in _fragment_store_vector_width_candidates(value_plan, frag, component):
        if not _fragment_store_tile_components_all_lanes_active(
            value_plan, fragment, tile_offsets, component, count, w
        ):
            continue
        if not _fragment_store_tile_pointer_components_contiguous(
            pointer_source, value_plan, fragment, tile_offsets, component, count, w
        ):
            continue
        if not _fragment_store_tile_mask_components_uniform(
            state, mask_source, value_plan, fragment, tile_offsets, component, count, w
        ):
            continue
        return count
    return 1


def _extract_fragment_store_value(regs, component, count, width, w, element_type="i32"):
    if count == 1:
        return _extract_fragment_component(regs, component, width, w, element_type)
    wave = getattr(w, "wave", None)
    if wave is None:
        raise RuntimeError(
            "tlx_wave bridge requires mlir.dialects.wave_dsl to expose the "
            "generated wave dialect module"
        )
    value_type = w.simd_type(
        w.vector_type(
            count,
            _wave_element_type(element_type, w, "fragment packed store value"),
        ),
        width,
    )
    values = [
        _extract_fragment_component(regs, component + index, width, w, element_type)
        for index in range(count)
    ]
    return wave.PackOp(value_type, values).result


def _emit_component_store(builder, value, ptr, mask, after_token, w):
    if mask is None:
        return builder.store(value, ptr, after=after_token)
    if after_token is None:
        after_token = builder.token()
    with builder.where(mask, [w.mem_token_type()]) as where_op:
        token = builder.store(value, ptr, after=after_token)
        builder.yield_([token])
    return where_op.results[0]


def _emit_masked_bounded_component_store(
    builder,
    state,
    value,
    pointer_source,
    dim_bindings,
    value_plan,
    access_byte_width,
    mask,
    after_token,
    w,
):
    if after_token is None:
        after_token = builder.token()
    # Keep pointer-range assumptions under the mask so inactive lanes do not
    # need to satisfy bounds that only hold for the actual store.
    with builder.where(mask, [w.mem_token_type()]) as where_op:
        ptr = _materialize_bounded_pointer_value(
            builder,
            state,
            pointer_source,
            dim_bindings,
            value_plan.element_byte_width,
            access_byte_width,
            value_plan.shape,
            w,
            assume_pointer_range=True,
        )
        token = builder.store(value, ptr, after=after_token)
        builder.yield_([token])
    return where_op.results[0]


def _emit_fragment_store(
    builder,
    state,
    lowered,
    store_plan,
    ptr_id,
    mask_id,
    after_token,
    w,
):
    fragment = lowered.value
    frag = w.FragmentType(fragment.type)
    unpack_element_type = _fragment_store_unpack_element_type(store_plan)
    regs = _fragment_unpack_as(builder, fragment, unpack_element_type, w)
    token = after_token
    pointer_source = _require_lowered_value(
        state["wave_values"],
        ptr_id,
        "pointer_expr",
        "tt.store pointer",
    )
    mask_source = (
        _require_lowered_value(
            state["wave_values"],
            mask_id,
            "mask_expr",
            "tt.store mask",
        )
        if mask_id is not None
        else None
    )
    component = 0
    while component < frag.registers:
        vector_width = _mfma_fragment_store_vector_width_for_store(
            state, pointer_source, mask_source, store_plan, frag, component, w
        )
        dim_bindings, width = _store_dim_bindings(
            builder, store_plan, lowered, w, component=component
        )
        access_byte_width = (
            int(store_plan.element_byte_width) * vector_width
            if store_plan.element_byte_width is not None
            else None
        )
        mask = (
            _materialize_mask_value(
                builder,
                mask_source,
                dim_bindings,
                w,
                width,
            )
            if mask_id is not None
            else None
        )
        value = _extract_converted_fragment_store_value(
            builder,
            regs,
            component,
            vector_width,
            width,
            store_plan,
            w,
            unpack_element_type,
        )
        if mask is not None and vector_width > 1:
            token = _emit_masked_bounded_component_store(
                builder,
                state,
                value,
                pointer_source,
                dim_bindings,
                store_plan,
                access_byte_width,
                mask,
                token,
                w,
            )
        else:
            ptr = _materialize_bounded_pointer_value(
                builder,
                state,
                pointer_source,
                dim_bindings,
                store_plan.element_byte_width,
                access_byte_width,
                store_plan.shape,
                w,
                assume_pointer_range=mask_id is None,
            )
            token = _emit_component_store(builder, value, ptr, mask, token, w)
        component += vector_width
    return token


def _emit_fragment_local_store(
    builder,
    state,
    lowered,
    store_plan,
    memdesc,
    memdescs,
    lds_layout,
    after_token,
    w,
):
    fragment = lowered.value
    frag = w.FragmentType(fragment.type)
    unpack_element_type = _fragment_store_unpack_element_type(store_plan)
    regs = _fragment_unpack_as(builder, fragment, unpack_element_type, w)
    token = after_token
    for component in range(frag.registers):
        dim_bindings, width = _store_dim_bindings(
            builder, store_plan, lowered, w, component=component
        )
        ptr = _emit_memdesc_ptr(
            builder,
            memdesc,
            memdescs,
            lds_layout,
            state,
            dim_bindings,
            width,
            w,
            "ttg.local_store",
        )
        value = _extract_converted_fragment_store_value(
            builder,
            regs,
            component,
            1,
            width,
            store_plan,
            w,
            unpack_element_type,
        )
        token = _emit_component_store(builder, value, ptr, None, token, w)
    return token


def _fragment_store_tile_dim_bindings(
    builder,
    value_plan,
    fragment,
    tile_offsets,
    component,
    w,
):
    frag = w.FragmentType(fragment.type)
    def nonnegative_index_expr(expr, bindings):
        return _assume_nonnegative(builder, builder.index_expr(expr, bindings), w)

    if frag.registers == _GFX950_MMA32_INFO.acc_registers:
        if component < 0 or component >= frag.registers:
            raise ValueError(
                f"tlx_wave bridge fragment component {component} is out of range "
                f"for {frag.registers} registers"
            )
        thread = builder.workitem_id(axis=0, width=frag.wave_size)
        thread = _assume_nonnegative(builder, thread, w)
        suffix = "_".join(str(int(offset)) for offset in tile_offsets)
        thread_sym = w.sym(f"tlx_store_{value_plan.value_id}_{suffix}_thread")
        coords = _mfma32_accumulator_dim_exprs(thread_sym, component, w)
        dim_bindings = {}
        active = None
        for dim in range(2):
            coord = nonnegative_index_expr(
                int(tile_offsets[dim]) + coords[dim],
                {thread_sym: thread},
            )
            dim_bindings[_dim_symbol(w, dim)] = coord
            extent = builder.splat(
                builder.constant(w.index_type(), value_plan.shape[dim]),
                width=frag.wave_size,
            )
            in_bounds = _wave_cmpi(
                builder,
                "ult",
                _maybe_splat(builder, coord, frag.wave_size, w),
                extent,
                w,
            )
            active = _wave_mask_and(builder, active, in_bounds, w, frag.wave_size)
        return dim_bindings, frag.wave_size, active

    layout = _GFX950_DOT_PARENT_LAYOUT
    if _product(layout.size_per_thread) != frag.registers:
        raise ValueError(
            "tlx_wave bridge cannot store MFMA fragment tile with incompatible "
            f"register count: registers={frag.registers}, "
            f"sizePerThread={layout.size_per_thread}"
        )
    if _product(layout.threads_per_warp) != frag.wave_size:
        raise ValueError(
            "tlx_wave bridge cannot store MFMA fragment tile with incompatible "
            f"wave size: fragment wave_size={frag.wave_size}, "
            f"threadsPerWarp={layout.threads_per_warp}"
        )
    if component < 0 or component >= frag.registers:
        raise ValueError(
            f"tlx_wave bridge fragment component {component} is out of range "
            f"for {frag.registers} registers"
        )

    thread = builder.workitem_id(axis=0, width=frag.wave_size)
    thread = _assume_nonnegative(builder, thread, w)
    suffix = "_".join(str(int(offset)) for offset in tile_offsets)
    thread_sym = w.sym(f"tlx_store_{value_plan.value_id}_{suffix}_thread")
    thread_expr = thread_sym
    threads_per_warp = _product(layout.threads_per_warp)
    register_coords = _delinearize_expr(
        w, w.sym_ctx.int_(component), layout.size_per_thread, layout.order
    )
    lane_coords = _delinearize_expr(
        w,
        w.mod(thread_expr, threads_per_warp),
        layout.threads_per_warp,
        layout.order,
    )
    warp_coords = _delinearize_expr(
        w,
        w.floor(thread_expr / threads_per_warp),
        layout.warps_per_cta,
        layout.order,
    )
    dim_bindings = {}
    active = None
    for dim in range(2):
        expr = (
            int(tile_offsets[dim])
            + register_coords[dim]
            + layout.size_per_thread[dim]
            * (lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim])
        )
        coord = nonnegative_index_expr(expr, {thread_sym: thread})
        dim_bindings[_dim_symbol(w, dim)] = coord
        extent = builder.splat(
            builder.constant(w.index_type(), value_plan.shape[dim]),
            width=frag.wave_size,
        )
        in_bounds = _wave_cmpi(
            builder, "ult", _maybe_splat(builder, coord, frag.wave_size, w), extent, w
        )
        active = _wave_mask_and(builder, active, in_bounds, w, frag.wave_size)
    return dim_bindings, frag.wave_size, active


def _fragment_store_tile_component_all_lanes_active(
    value_plan,
    fragment,
    tile_offsets,
    component,
    w,
):
    frag = w.FragmentType(fragment.type)
    if frag.registers == _GFX950_MMA32_INFO.acc_registers:
        for thread in range(frag.wave_size):
            lane = thread % _GFX950_MMA32_INFO.wave_size
            coords = (
                int(tile_offsets[0]) + lane % _GFX950_MMA32_INFO.output_tile_shape[0],
                int(tile_offsets[1])
                + 4 * (lane // _GFX950_MMA32_INFO.output_tile_shape[0])
                + int(component % 4)
                + int(8 * (component // 4)),
            )
            if any(
                coord >= int(extent) for coord, extent in zip(coords, value_plan.shape)
            ):
                return False
        return True

    layout = _GFX950_DOT_PARENT_LAYOUT
    for thread in range(frag.wave_size):
        local_coords = _blocked_layout_static_coord(
            layout,
            _GFX950_MMA_SHAPE,
            thread,
            component,
        )
        coords = tuple(
            int(tile_offsets[dim]) + local_coords[dim]
            for dim in range(len(local_coords))
        )
        if any(coord >= int(extent) for coord, extent in zip(coords, value_plan.shape)):
            return False
    return True


def _emit_mfma_fragment_tile_store(
    builder,
    state,
    fragments,
    tile_shape,
    value_plan,
    ptr_id,
    mask_id,
    after_token,
    w,
):
    if value_plan.element_type not in {"f32", "f16"}:
        raise ValueError(
            "tlx_wave bridge supports MFMA fragment stores only for f32/f16 "
            f"values, got type={value_plan.type}"
        )
    mma = _mma_shape_for_result_value(value_plan, "tt.store MFMA value")
    expected_tile_shape = _fragment_tile_shape_for_rank2_shape(
        value_plan.shape,
        "tt.store MFMA value",
        mma.output_tile_shape,
    )
    if tuple(tile_shape) != expected_tile_shape:
        raise ValueError(
            "tlx_wave bridge cannot lower MFMA fragment store: fragment tile "
            f"shape {tile_shape} does not match value tile shape "
            f"{expected_tile_shape}"
        )
    if len(fragments) != tile_shape[0] * tile_shape[1]:
        raise ValueError(
            "tlx_wave bridge cannot lower MFMA fragment store: got "
            f"{len(fragments)} fragment(s), expected "
            f"{tile_shape[0] * tile_shape[1]}"
        )

    token = after_token
    unpack_element_type = _fragment_store_unpack_element_type(value_plan)
    pointer_source = _require_lowered_value(
        state["wave_values"],
        ptr_id,
        "pointer_expr",
        "tt.store pointer",
    )
    mask_source = (
        _require_lowered_value(
            state["wave_values"],
            mask_id,
            "mask_expr",
            "tt.store mask",
        )
        if mask_id is not None
        else None
    )
    for row in range(tile_shape[0]):
        for col in range(tile_shape[1]):
            fragment = fragments[_fragment_tuple_index(tile_shape, row, col)]
            regs = _fragment_unpack_as(builder, fragment, unpack_element_type, w)
            frag = w.FragmentType(fragment.type)
            tile_offsets = (
                row * mma.output_tile_shape[0],
                col * mma.output_tile_shape[1],
            )
            component = 0
            while component < frag.registers:
                vector_width = _fragment_store_tile_vector_width_for_store(
                    state,
                    pointer_source,
                    mask_source,
                    value_plan,
                    fragment,
                    tile_offsets,
                    component,
                    w,
                )
                dim_bindings, width, active = _fragment_store_tile_dim_bindings(
                    builder,
                    value_plan,
                    fragment,
                    tile_offsets,
                    component,
                    w,
                )
                components_active = _fragment_store_tile_components_all_lanes_active(
                    value_plan,
                    fragment,
                    tile_offsets,
                    component,
                    vector_width,
                    w,
                )
                access_byte_width = (
                    int(value_plan.element_byte_width) * vector_width
                    if value_plan.element_byte_width is not None
                    else None
                )
                mask = None if components_active else active
                if mask_id is not None:
                    user_mask = _materialize_mask_value(
                        builder,
                        mask_source,
                        dim_bindings,
                        w,
                        width,
                    )
                    mask = _wave_mask_and(builder, mask, user_mask, w, width)
                value = _extract_converted_fragment_store_value(
                    builder,
                    regs,
                    component,
                    vector_width,
                    width,
                    value_plan,
                    w,
                    unpack_element_type,
                )
                if mask is not None and vector_width > 1:
                    token = _emit_masked_bounded_component_store(
                        builder,
                        state,
                        value,
                        pointer_source,
                        dim_bindings,
                        value_plan,
                        access_byte_width,
                        mask,
                        token,
                        w,
                    )
                else:
                    ptr = _materialize_bounded_pointer_value(
                        builder,
                        state,
                        pointer_source,
                        dim_bindings,
                        value_plan.element_byte_width,
                        access_byte_width,
                        value_plan.shape,
                        w,
                        assume_pointer_range=mask_id is None and components_active,
                    )
                    token = _emit_component_store(
                        builder,
                        value,
                        ptr,
                        mask,
                        token,
                        w,
                    )
                component += vector_width
    return token


def _emit_mfma_fragment_tile_local_store(
    builder,
    state,
    fragments,
    tile_shape,
    value_plan,
    memdesc,
    memdescs,
    lds_layout,
    after_token,
    w,
):
    if value_plan.element_type not in {"f32", "f16"}:
        raise ValueError(
            "tlx_wave bridge supports MFMA fragment local stores only for f32/f16 "
            f"values, got type={value_plan.type}"
        )
    mma = _mma_shape_for_result_value(value_plan, "ttg.local_store MFMA value")
    expected_tile_shape = _fragment_tile_shape_for_rank2_shape(
        value_plan.shape,
        "ttg.local_store MFMA value",
        mma.output_tile_shape,
    )
    if tuple(tile_shape) != expected_tile_shape:
        raise ValueError(
            "tlx_wave bridge cannot lower MFMA fragment local store: fragment "
            f"tile shape {tile_shape} does not match value tile shape "
            f"{expected_tile_shape}"
        )
    if len(fragments) != tile_shape[0] * tile_shape[1]:
        raise ValueError(
            "tlx_wave bridge cannot lower MFMA fragment local store: got "
            f"{len(fragments)} fragment(s), expected "
            f"{tile_shape[0] * tile_shape[1]}"
        )

    token = after_token
    unpack_element_type = _fragment_store_unpack_element_type(value_plan)
    for row in range(tile_shape[0]):
        for col in range(tile_shape[1]):
            fragment = fragments[_fragment_tuple_index(tile_shape, row, col)]
            regs = _fragment_unpack_as(builder, fragment, unpack_element_type, w)
            frag = w.FragmentType(fragment.type)
            tile_offsets = (
                row * mma.output_tile_shape[0],
                col * mma.output_tile_shape[1],
            )
            for component in range(frag.registers):
                dim_bindings, width, active = _fragment_store_tile_dim_bindings(
                    builder,
                    value_plan,
                    fragment,
                    tile_offsets,
                    component,
                    w,
                )
                ptr = _emit_memdesc_ptr(
                    builder,
                    memdesc,
                    memdescs,
                    lds_layout,
                    state,
                    dim_bindings,
                    width,
                    w,
                    "ttg.local_store",
                )
                value = _extract_converted_fragment_store_value(
                    builder,
                    regs,
                    component,
                    1,
                    width,
                    value_plan,
                    w,
                    unpack_element_type,
                )
                token = _emit_component_store(
                    builder,
                    value,
                    ptr,
                    active,
                    token,
                    w,
                )
    return token


def _reject_unlowered_fragment_local_store_conversion(value_plan, physical_plan):
    if value_plan.value_id == physical_plan.value_id:
        return
    try:
        physical_layout = _blocked_encoding_info(
            physical_plan.encoding_attr,
            physical_plan.encoding,
            "fragment local-store physical value",
        )
    except ValueError:
        return
    try:
        value_layout = _blocked_encoding_info(
            value_plan.encoding_attr,
            value_plan.encoding,
            "fragment local-store value",
        )
    except ValueError:
        value_layout = None
    if value_layout is not None and _same_blocked_encoding(
        value_layout, physical_layout
    ):
        return
    raise ValueError(
        "tlx_wave bridge cannot lower fragment local store after unlowered "
        "layout conversion from blocked physical layout; physical "
        f"encoding={physical_plan.encoding}; requested encoding={value_plan.encoding}"
    )


def _physical_plan_is_mfma(physical_plan, context):
    try:
        return _amd_mfma_encoding_info(physical_plan.encoding, context) is not None
    except ValueError as exc:
        raise ValueError(f"tlx_wave bridge cannot lower {context}: {exc}") from exc
