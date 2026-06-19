"""Async-copy and DMA packet lowering helpers for the TLX Wave bridge.

The functions here lower ``ttg.async_copy_global_to_local`` and the symbolic
packet-contiguity proofs used by that path.  ``wave_bridge_emit.py`` configures
this module with the shared bridge helper namespace after those helpers are
available, and re-exports these names for existing tests.
"""

from dataclasses import replace


_DEPENDENCY_NAMES = (
    "_AssumedIndexBinding",
    "_DimBinding",
    "_IndexBinary",
    "_IndexExpr",
    "_IndexSelectCompare",
    "_LinearEncodingInfo",
    "_MaskAnd",
    "_MaskCompare",
    "_MaskConst",
    "_PointerAdd",
    "_PointerBase",
    "_WaveValue",
    "_assume_nonnegative",
    "_async_copy_source_component_all_lanes_active",
    "_async_copy_source_dim_bindings",
    "_async_copy_source_layout_component_count",
    "_ceil_div",
    "_dim_symbol",
    "_dma_packet_physical_coords_expr",
    "_dma_packet_start_coords_static",
    "_dma_packet_uniform_destination_dword_offset_expr",
    "_dma_source_tensor_layout_info",
    "_emit_component_store",
    "_emit_masked_load",
    "_emit_memdesc_base_ptr",
    "_emit_memdesc_ptr",
    "_inline_index_expr_bindings",
    "_is_identity_shared_layout",
    "_is_supported_swizzled_shared_layout",
    "_is_wave_simd_index_type",
    "_is_wave_simd_pointer_type",
    "_load_other_value",
    "_materialize_bounded_pointer_value",
    "_materialize_component_mask_value",
    "_materialize_dma_source_pointer_value",
    "_materialize_index_value",
    "_memdesc_base_is_aligned",
    "_memdesc_dma_can_use_base_pointer",
    "_memdesc_dma_dword_offset_expr",
    "_product",
    "_require_dma_destination_physical_packets",
    "_require_lowered_value",
    "_require_typed_wave_value",
    "_splat_constant_value",
    "_swizzled_shared_encoding_info",
    "_validate_padded_shared_layout",
    "_wave_cmpi",
    "_wave_element_type",
    "_wave_mask_and",
)


def configure_async_copy_dependencies(namespace):
    missing = [name for name in _DEPENDENCY_NAMES if name not in namespace]
    if missing:
        raise RuntimeError(
            "tlx_wave async-copy helper is missing bridge dependencies: "
            + ", ".join(missing)
        )
    for name in _DEPENDENCY_NAMES:
        globals()[name] = namespace[name]


def _dma_packet_layout_supported(address, memdesc):
    padded = _validate_padded_shared_layout(
        memdesc,
        "ttg.async_copy_global_to_local destination",
    )
    if padded is not None:
        return (
            memdesc.element_type in {"f16", "bf16"}
            and address.shape == memdesc.shape
        )
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr,
            memdesc.encoding,
            "ttg.async_copy_global_to_local destination",
        )
    except ValueError:
        return False
    if _is_identity_shared_layout(memdesc, shared):
        return True
    return _is_supported_swizzled_shared_layout(memdesc, shared) and (
        memdesc.element_type in {"f16", "bf16"}
        and address.shape == memdesc.shape
    )


def _dma_packet_byte_candidates(address, memdesc, memdescs, lds_layout):
    if address.element_byte_width is None:
        return ()
    if address.element_byte_width == 2 and address.element_type in {"f16", "bf16"}:
        raw_candidates = (16, 4)
    elif address.element_byte_width in (1, 2, 4):
        raw_candidates = (4,)
    elif address.element_byte_width == 16:
        raw_candidates = (16,)
    else:
        return ()
    if not _dma_packet_layout_supported(address, memdesc):
        return ()
    if not _memdesc_dma_can_use_base_pointer(
        memdesc,
        memdescs,
        "ttg.async_copy_global_to_local destination",
    ):
        return ()
    candidates = []
    for packet_bytes in raw_candidates:
        if _memdesc_base_is_aligned(
            memdesc,
            memdescs,
            lds_layout,
            packet_bytes,
            "ttg.async_copy_global_to_local destination",
        ):
            candidates.append(packet_bytes)
    return tuple(candidates)


def _dma_packet_bytes(address, memdesc, memdescs, lds_layout):
    candidates = _dma_packet_byte_candidates(address, memdesc, memdescs, lds_layout)
    return candidates[0] if candidates else None


def _dma_packet_elements(address, packet_bytes):
    if address.element_byte_width is None:
        return None
    if packet_bytes % address.element_byte_width:
        return None
    return packet_bytes // address.element_byte_width


def _dma_layout_width(layout):
    return _product(layout.threads_per_warp)


def _dma_layout_cta_threads(layout):
    return _dma_layout_width(layout) * _product(layout.warps_per_cta)


def _log2_int(value):
    if value <= 0:
        return None
    bits = int(value).bit_length() - 1
    return bits if (1 << bits) == int(value) else None


def _linear_layout_static_coords(layout, rank, register, lane, warp):
    coords = [0 for _ in range(rank)]
    for value, bases in (
        (register, layout.register_bases),
        (lane, layout.lane_bases),
        (warp, layout.warp_bases),
    ):
        for bit, basis in enumerate(bases):
            if int(value) & (1 << bit):
                for dim in range(rank):
                    coords[dim] += int(basis[dim])
    return tuple(coords)


def _validate_linear_dma_packet_layout(layout, address_plan, memdesc, packet_elements):
    packet_bits = _log2_int(packet_elements)
    if packet_bits is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: #ttg.linear DMA packet element count "
            f"{packet_elements} is not a power of two"
        )
    if len(layout.register_bases) < packet_bits:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: #ttg.linear source layout has only "
            f"{len(layout.register_bases)} register basis bit(s), fewer than "
            f"the {packet_bits} needed for {packet_elements} element packets"
        )
    rank = len(address_plan.shape)
    inner_dim = rank - 1
    for bit in range(packet_bits):
        expected = [0 for _ in range(rank)]
        expected[inner_dim] = 1 << bit
        if layout.register_bases[bit] != tuple(expected):
            raise ValueError(
                "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                "without faithful DMA: #ttg.linear source layout does not map "
                f"packet register bit {bit} to contiguous innermost elements; "
                f"basis={layout.register_bases[bit]}, expected={tuple(expected)}"
            )
    width = _dma_layout_width(layout)
    cta_threads = _dma_layout_cta_threads(layout)
    total_packets = _product(address_plan.shape) // int(packet_elements)
    for packet_index in range(total_packets):
        component = packet_index // cta_threads
        thread = packet_index % cta_threads
        lane = thread % width
        warp = thread // width
        register = component << packet_bits
        coords = _linear_layout_static_coords(layout, rank, register, lane, warp)
        expected = _dma_packet_start_coords_static(
            memdesc,
            packet_elements,
            packet_index,
            "ttg.async_copy_global_to_local destination",
        )
        if coords != expected:
            raise ValueError(
                "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                "without faithful DMA: #ttg.linear source layout does not "
                "match destination physical packet order; packet "
                f"{packet_index} maps to {coords}, expected {expected}"
            )


def _validate_dma_packet_layout(address_plan, address, memdesc, packet_bytes):
    context = (
        "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
        "without faithful DMA"
    )
    packet_elements = _dma_packet_elements(address, packet_bytes)
    if packet_elements is None or packet_elements < 1:
        raise ValueError(
            f"{context}: source element width {address.element_byte_width} "
            f"does not divide DMA packet size {packet_bytes}"
        )
    if address_plan.shape != memdesc.shape:
        raise ValueError(
            f"{context}: source tensor shape {address_plan.shape} does not "
            f"match destination memdesc shape {memdesc.shape}"
        )
    layout = _dma_source_tensor_layout_info(
        address_plan,
        "ttg.async_copy_global_to_local source",
        "async copy DMA packet lowering",
    )
    rank = len(address_plan.shape)
    inner_dim = rank - 1
    if int(address_plan.shape[inner_dim]) % packet_elements:
        raise ValueError(
            f"{context}: innermost tensor extent {address_plan.shape[inner_dim]} "
            f"is not divisible by {packet_elements} element DMA packets"
        )
    if isinstance(layout, _LinearEncodingInfo):
        _validate_linear_dma_packet_layout(
            layout, address_plan, memdesc, packet_elements
        )
    _require_dma_destination_physical_packets(
        memdesc,
        packet_elements,
        packet_bytes,
        _dma_layout_width(layout),
        context,
    )
    return layout


def _require_dma_packet_byte_candidates(address, memdesc, memdescs, lds_layout):
    context = (
        "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
        "without faithful DMA"
    )
    candidates = _dma_packet_byte_candidates(address, memdesc, memdescs, lds_layout)
    if not candidates:
        raise ValueError(
            f"{context}: unsupported source element width, destination shared "
            "layout, memdesc view, or LDS alignment for waveamd.dma_load_lds"
        )
    return candidates


def _dma_packet_component_count(value_plan, layout, packet_elements):
    total_elements = _product(value_plan.shape)
    if total_elements % int(packet_elements):
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: tensor element count is not divisible by "
            f"{packet_elements} element DMA packets"
        )
    cta_threads = _dma_layout_cta_threads(layout)
    if cta_threads <= 0:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            f"without faithful DMA: non-positive CTA thread count {cta_threads}"
        )
    return _ceil_div(total_elements // int(packet_elements), cta_threads)


def _dma_packet_dim_bindings(
    builder,
    value_plan,
    memdesc,
    layout,
    packet_elements,
    component,
    w,
):
    component_count = _dma_packet_component_count(value_plan, layout, packet_elements)
    if component < 0 or component >= component_count:
        raise ValueError(
            "tlx_wave bridge async copy DMA packet component "
            f"{component} is out of range for {component_count} packet "
            "component(s)"
        )
    width = _dma_layout_width(layout)
    cta_threads = _dma_layout_cta_threads(layout)
    total_packets = _product(value_plan.shape) // int(packet_elements)
    thread = builder.workitem_id(axis=0, width=width)
    thread = _assume_nonnegative(builder, thread, w)
    thread_sym = w.sym(f"tlx_dma_{value_plan.value_id}_thread")
    packet_index_expr = w.sym_ctx.int_(component * cta_threads) + thread_sym
    element_linear = packet_index_expr * int(packet_elements)
    coords = _dma_packet_physical_coords_expr(
        w,
        element_linear,
        memdesc,
        "ttg.async_copy_global_to_local destination",
    )

    dim_bindings = {}
    component_packet_start = component * cta_threads
    active = None
    if component_packet_start + cta_threads > total_packets:
        packet_index = thread
        if component_packet_start:
            packet_index = builder.binary(
                w.BinaryKind.AddI,
                packet_index,
                builder.splat(
                    builder.constant(w.i32(), component_packet_start),
                    width=width,
                ),
            )
        active = _wave_cmpi(
            builder,
            "ult",
            packet_index,
            builder.splat(builder.constant(w.i32(), total_packets), width=width),
            w,
        )
    for dim in range(len(value_plan.shape)):
        coord = builder.index_expr(coords[dim], {thread_sym: thread})
        dim_bindings[_dim_symbol(w, dim)] = coord
    return dim_bindings, width, active, thread


def _dma_packet_uniform_dim_bindings(
    builder,
    value_plan,
    memdesc,
    layout,
    packet_elements,
    component,
    thread,
    w,
):
    width = _dma_layout_width(layout)
    cta_threads = _dma_layout_cta_threads(layout)
    thread_first = builder.read_first(thread)
    thread_first = _assume_nonnegative(builder, thread_first, w)
    thread_sym = w.sym(f"tlx_dma_{value_plan.value_id}_thread_first")
    packet_index_expr = w.sym_ctx.int_(component * cta_threads) + thread_sym
    element_linear = packet_index_expr * int(packet_elements)
    coords = _dma_packet_physical_coords_expr(
        w,
        element_linear,
        memdesc,
        "ttg.async_copy_global_to_local destination",
    )

    dim_bindings = {}
    for dim in range(len(value_plan.shape)):
        dim_bindings[_dim_symbol(w, dim)] = _IndexExpr(
            coords[dim],
            {thread_sym: thread_first},
        )
    return dim_bindings


def _offset_packet_dim_bindings(builder, dim_bindings, inner_dim, offset, w):
    if offset == 0:
        return dim_bindings
    adjusted = dict(dim_bindings)
    symbol = _dim_symbol(w, inner_dim)
    source = adjusted[symbol]
    offset_sym = w.sym(f"tlx_dma_packet_elem_{inner_dim}_{offset}")
    adjusted[symbol] = builder.index_expr(
        offset_sym + int(offset),
        {offset_sym: source},
    )
    return adjusted


def _ixsimpl_module():
    import ixsimpl

    return ixsimpl


def _ixsimpl_simplify(expr, assumptions, w):
    values = [expr]
    w.sym_ctx.simplify_batch(values, assumptions=list(assumptions))
    return values[0]


def _ixsimpl_is_true(expr):
    ixs = _ixsimpl_module()
    return ixs.same_node(expr, expr._ctx.true_())


def _ixsimpl_is_false(expr):
    ixs = _ixsimpl_module()
    return ixs.same_node(expr, expr._ctx.false_())


def _ixsimpl_proves(expr, assumptions, w):
    simplified = _ixsimpl_simplify(expr, assumptions, w)
    if _ixsimpl_is_true(simplified):
        return True
    if _ixsimpl_is_false(simplified):
        return False
    return w.sym_ctx.check(simplified, assumptions=list(assumptions)) is True


def _ixsimpl_expr_equal(lhs, rhs, assumptions, w):
    ixs = _ixsimpl_module()
    lhs = _ixsimpl_simplify(lhs, assumptions, w)
    rhs = _ixsimpl_simplify(rhs, assumptions, w)
    if ixs.same_node(lhs, rhs):
        return True
    return _ixsimpl_proves(w.sym_ctx.eq(lhs, rhs), assumptions, w)


def _ixsimpl_predicate_equivalent(lhs, rhs, assumptions, w):
    ixs = _ixsimpl_module()
    lhs = _ixsimpl_simplify(lhs, assumptions, w)
    rhs = _ixsimpl_simplify(rhs, assumptions, w)
    if ixs.same_node(lhs, rhs):
        return True
    if _ixsimpl_is_true(lhs):
        return _ixsimpl_proves(rhs, assumptions, w)
    if _ixsimpl_is_false(lhs):
        return _ixsimpl_proves(ixs.not_(rhs), assumptions, w)
    if _ixsimpl_is_true(rhs):
        return _ixsimpl_proves(lhs, assumptions, w)
    if _ixsimpl_is_false(rhs):
        return _ixsimpl_proves(ixs.not_(lhs), assumptions, w)
    if _ixsimpl_proves(lhs, assumptions, w) and _ixsimpl_proves(rhs, assumptions, w):
        return True
    if _ixsimpl_proves(ixs.not_(lhs), assumptions, w) and _ixsimpl_proves(
        ixs.not_(rhs), assumptions, w
    ):
        return True
    return False


def _ixsimpl_unknown_expr(source, unknowns, w):
    key = id(source)
    expr = unknowns.get(key)
    if expr is None:
        expr = w.sym(f"tlx_dma_unknown_{len(unknowns)}")
        unknowns[key] = expr
    return expr


def _index_source_depends_on_dim(source, dim):
    if isinstance(source, _AssumedIndexBinding):
        return _index_source_depends_on_dim(source.value, dim)
    if isinstance(source, _DimBinding):
        return source.dim == dim
    if isinstance(source, _IndexExpr):
        return any(
            _index_source_depends_on_dim(binding, dim)
            for binding in source.bindings.values()
        )
    if isinstance(source, _IndexBinary):
        return _index_source_depends_on_dim(
            source.lhs, dim
        ) or _index_source_depends_on_dim(source.rhs, dim)
    if isinstance(source, _IndexSelectCompare):
        return (
            _index_source_depends_on_dim(source.lhs, dim)
            or _index_source_depends_on_dim(source.rhs, dim)
            or _index_source_depends_on_dim(source.true_value, dim)
            or _index_source_depends_on_dim(source.false_value, dim)
        )
    return False


def _mask_source_depends_on_dim(source, dim):
    if isinstance(source, _MaskAnd):
        return _mask_source_depends_on_dim(
            source.lhs, dim
        ) or _mask_source_depends_on_dim(source.rhs, dim)
    if isinstance(source, _MaskCompare):
        return _index_source_depends_on_dim(
            source.lhs, dim
        ) or _index_source_depends_on_dim(source.rhs, dim)
    return False


def _ixsimpl_apply_index_binary(kind, lhs, rhs):
    kind = str(kind).lower()
    if kind.endswith("addi"):
        return lhs + rhs
    if kind.endswith("muli"):
        return lhs * rhs
    if kind.endswith("subi"):
        return lhs - rhs
    return None


def _ixsimpl_index_expr(source, w, unknowns, opaque_dim=None):
    if isinstance(source, _AssumedIndexBinding):
        return _ixsimpl_index_expr(source.value, w, unknowns, opaque_dim)
    if isinstance(source, bool):
        return None
    if isinstance(source, int):
        return w.sym_ctx.int_(source)
    if isinstance(source, _DimBinding):
        return _dim_symbol(w, source.dim)
    if isinstance(source, _IndexExpr):
        expr = source.expr
        for symbol, binding in source.bindings.items():
            replacement = _ixsimpl_index_expr(binding, w, unknowns, opaque_dim)
            if replacement is None:
                return None
            expr = expr.subs(symbol, replacement)
        return expr
    if isinstance(source, _IndexBinary):
        if opaque_dim is not None and not _index_source_depends_on_dim(
            source, opaque_dim
        ):
            return _ixsimpl_unknown_expr(source, unknowns, w)
        lhs = _ixsimpl_index_expr(source.lhs, w, unknowns, opaque_dim)
        rhs = _ixsimpl_index_expr(source.rhs, w, unknowns, opaque_dim)
        if lhs is not None and rhs is not None:
            return _ixsimpl_apply_index_binary(source.kind, lhs, rhs)
        return None
    if isinstance(source, _IndexSelectCompare):
        if opaque_dim is not None and not _index_source_depends_on_dim(
            source, opaque_dim
        ):
            return _ixsimpl_unknown_expr(source, unknowns, w)
        return None
    typ = getattr(source, "type", None)
    if typ is None:
        return None
    if _is_wave_simd_index_type(typ, w):
        return None
    return _ixsimpl_unknown_expr(source, unknowns, w)


def _ixsimpl_index_packet_expr(
    source, w, unknowns, inner_dim, packet_elements, assumptions
):
    if isinstance(source, _AssumedIndexBinding):
        return _ixsimpl_index_packet_expr(
            source.value, w, unknowns, inner_dim, packet_elements, assumptions
        )
    if isinstance(source, bool):
        return None
    if isinstance(source, int):
        return w.sym_ctx.int_(source)
    if isinstance(source, _DimBinding):
        return _dim_symbol(w, source.dim)
    if isinstance(source, _IndexExpr):
        expr = source.expr
        for symbol, binding in source.bindings.items():
            replacement = _ixsimpl_index_packet_expr(
                binding, w, unknowns, inner_dim, packet_elements, assumptions
            )
            if replacement is None:
                return None
            expr = expr.subs(symbol, replacement)
        return expr
    if isinstance(source, _IndexBinary):
        if not _index_source_depends_on_dim(source, inner_dim):
            return _ixsimpl_unknown_expr(source, unknowns, w)
        lhs = _ixsimpl_index_packet_expr(
            source.lhs, w, unknowns, inner_dim, packet_elements, assumptions
        )
        rhs = _ixsimpl_index_packet_expr(
            source.rhs, w, unknowns, inner_dim, packet_elements, assumptions
        )
        if lhs is not None and rhs is not None:
            expr = _ixsimpl_apply_index_binary(source.kind, lhs, rhs)
            if expr is not None:
                return expr
        if str(source.kind) not in {"remsi", "remui"}:
            return None
        if _index_source_depends_on_dim(source.rhs, inner_dim):
            return None
        if lhs is None or rhs is None:
            return None
        if not _ixsimpl_expr_equal(
            _ixsimpl_shift_dim(lhs, inner_dim, 1, w),
            _ixsimpl_add_const(lhs, 1, w),
            assumptions,
            w,
        ):
            return None
        if not _ixsimpl_mod_zero(lhs, packet_elements, assumptions, w):
            return None
        if not _ixsimpl_mod_zero(rhs, packet_elements, assumptions, w):
            return None
        return _ixsimpl_unknown_expr(source, unknowns, w) + _dim_symbol(w, inner_dim)
    if isinstance(source, _IndexSelectCompare):
        if not _index_source_depends_on_dim(source, inner_dim):
            return _ixsimpl_unknown_expr(source, unknowns, w)
        return None
    typ = getattr(source, "type", None)
    if typ is None:
        return None
    if _is_wave_simd_index_type(typ, w):
        return None
    return _ixsimpl_unknown_expr(source, unknowns, w)


def _ixsimpl_pointer_offset_expr(source, w, unknowns, opaque_dim=None):
    if isinstance(source, _PointerBase):
        return w.sym_ctx.int_(0)
    if isinstance(source, _PointerAdd):
        base = _ixsimpl_pointer_offset_expr(source.base, w, unknowns, opaque_dim)
        offset = _ixsimpl_index_expr(source.offset, w, unknowns, opaque_dim)
        if base is None or offset is None:
            return None
        return base + offset
    typ = getattr(source, "type", None)
    if typ is not None and _is_wave_simd_pointer_type(typ, w):
        return None
    return w.sym_ctx.int_(0)


def _ixsimpl_pointer_packet_offset_expr(
    source, w, unknowns, inner_dim, packet_elements, assumptions
):
    if isinstance(source, _PointerBase):
        return w.sym_ctx.int_(0)
    if isinstance(source, _PointerAdd):
        base = _ixsimpl_pointer_packet_offset_expr(
            source.base, w, unknowns, inner_dim, packet_elements, assumptions
        )
        offset = _ixsimpl_index_packet_expr(
            source.offset, w, unknowns, inner_dim, packet_elements, assumptions
        )
        if base is None or offset is None:
            return None
        return base + offset
    typ = getattr(source, "type", None)
    if typ is not None and _is_wave_simd_pointer_type(typ, w):
        return None
    return w.sym_ctx.int_(0)


def _ixsimpl_pointer_byte_offset_expr(
    source, source_element_byte_width, w, unknowns, opaque_dim=None
):
    if source_element_byte_width is None or source_element_byte_width <= 0:
        return None
    offset = _ixsimpl_pointer_offset_expr(source, w, unknowns, opaque_dim)
    if offset is None:
        return None
    return offset * int(source_element_byte_width)


def _ixsimpl_pointer_packet_byte_offset_expr(
    source,
    source_element_byte_width,
    w,
    unknowns,
    inner_dim,
    packet_elements,
    assumptions,
):
    if source_element_byte_width is None or source_element_byte_width <= 0:
        return None
    offset = _ixsimpl_pointer_packet_offset_expr(
        source, w, unknowns, inner_dim, packet_elements, assumptions
    )
    if offset is None:
        return None
    return offset * int(source_element_byte_width)


def _ixsimpl_shift_dim(expr, dim, offset, w):
    if offset == 0:
        return expr
    symbol = _dim_symbol(w, dim)
    return expr.subs(symbol, symbol + int(offset))


def _ixsimpl_mask_compare_expr(predicate, lhs, rhs, w):
    if predicate == "eq":
        return w.sym_ctx.eq(lhs, rhs)
    if predicate == "ne":
        return w.sym_ctx.ne(lhs, rhs)
    if predicate == "slt":
        return lhs < rhs
    if predicate == "sle":
        return lhs <= rhs
    if predicate == "sgt":
        return lhs > rhs
    if predicate == "sge":
        return lhs >= rhs
    return None


def _ixsimpl_mask_expr(source, w, unknowns, opaque_dim=None):
    ixs = _ixsimpl_module()
    if isinstance(source, _MaskConst):
        return w.sym_ctx.true_() if source.value else w.sym_ctx.false_()
    if isinstance(source, _MaskAnd):
        lhs = _ixsimpl_mask_expr(source.lhs, w, unknowns, opaque_dim)
        rhs = _ixsimpl_mask_expr(source.rhs, w, unknowns, opaque_dim)
        if lhs is None or rhs is None:
            return None
        return ixs.and_(lhs, rhs)
    if isinstance(source, _MaskCompare):
        lhs = _ixsimpl_index_expr(source.lhs, w, unknowns, opaque_dim)
        rhs = _ixsimpl_index_expr(source.rhs, w, unknowns, opaque_dim)
        if lhs is None or rhs is None:
            return None
        return _ixsimpl_mask_compare_expr(source.predicate, lhs, rhs, w)
    return None


def _dma_packet_dim_assumptions(shape, inner_dim, packet_elements, w):
    assumptions = []
    for dim, extent in enumerate(shape):
        symbol = _dim_symbol(w, dim)
        assumptions.append(symbol >= 0)
        upper = int(extent) - 1
        if dim == inner_dim:
            upper = int(extent) - int(packet_elements)
        assumptions.append(symbol <= upper)
    if packet_elements > 1:
        symbol = _dim_symbol(w, inner_dim)
        assumptions.append(
            w.sym_ctx.eq(
                w.mod(symbol, int(packet_elements)),
                w.sym_ctx.int_(0),
            )
        )
    return tuple(assumptions)


def _ixsimpl_assume_fact_exprs(state, unknowns, w, *, include_planned=False):
    assumptions = []
    facts = tuple(state.get("assume_facts", ()))
    if include_planned:
        facts = facts + tuple(state.get("planned_assume_facts", ()))
    for fact in facts:
        lowered = state["wave_values"].get(fact.value_id)
        if not isinstance(lowered, _WaveValue) or lowered.kind != "index_expr":
            continue
        value = _ixsimpl_index_expr(lowered.value, w, unknowns)
        if value is None:
            continue
        if fact.kind == "range":
            if fact.lower is not None:
                assumptions.append(value >= fact.lower)
            if fact.upper is not None:
                assumptions.append(value <= fact.upper)
        elif fact.kind == "divisible" and fact.divisor is not None:
            divisor = int(fact.divisor)
            divisors = {divisor}
            candidate = 2
            while candidate * candidate <= divisor:
                if divisor % candidate == 0:
                    divisors.add(candidate)
                    divisors.add(divisor // candidate)
                candidate += 1
            for implied_divisor in sorted(divisors):
                assumptions.append(
                    w.sym_ctx.eq(
                        w.mod(value, implied_divisor),
                        w.sym_ctx.int_(0),
                    )
                )
    return tuple(assumptions)


def _ixsimpl_index_source_proves_lower_bound(
    source,
    lower_bound,
    assumptions,
    w,
    unknowns,
):
    if (
        isinstance(source, _AssumedIndexBinding)
        and source.lower is not None
        and int(source.lower) >= int(lower_bound)
    ):
        return True
    expr = _ixsimpl_index_expr(source, w, unknowns)
    if expr is not None and _ixsimpl_proves(expr >= int(lower_bound), assumptions, w):
        return True
    if isinstance(source, _IndexExpr):
        if (
            source.materialized is not None
            and _ixsimpl_index_source_proves_lower_bound(
                source.materialized,
                lower_bound,
                assumptions,
                w,
                unknowns,
            )
        ):
            return True
        binding_assumptions = []
        for symbol, binding in source.bindings.items():
            for candidate_lower in (1, 0):
                if _ixsimpl_index_source_proves_lower_bound(
                    binding,
                    candidate_lower,
                    assumptions,
                    w,
                    unknowns,
                ):
                    binding_assumptions.append(symbol >= candidate_lower)
                    break
        if binding_assumptions and _ixsimpl_proves(
            source.expr >= int(lower_bound),
            assumptions + tuple(binding_assumptions),
            w,
        ):
            return True
        return False
    if isinstance(source, _IndexSelectCompare):
        return _ixsimpl_index_source_proves_lower_bound(
            source.true_value,
            lower_bound,
            assumptions,
            w,
            unknowns,
        ) and _ixsimpl_index_source_proves_lower_bound(
            source.false_value,
            lower_bound,
            assumptions,
            w,
            unknowns,
        )
    if not isinstance(source, _IndexBinary):
        return False
    kind = str(source.kind).lower()
    if kind.endswith("addi"):
        if lower_bound != 0:
            return False
        return _ixsimpl_index_source_proves_lower_bound(
            source.lhs,
            0,
            assumptions,
            w,
            unknowns,
        ) and _ixsimpl_index_source_proves_lower_bound(
            source.rhs,
            0,
            assumptions,
            w,
            unknowns,
        )
    if kind.endswith("muli"):
        if lower_bound != 0:
            return False
        return _ixsimpl_index_source_proves_lower_bound(
            source.lhs,
            0,
            assumptions,
            w,
            unknowns,
        ) and _ixsimpl_index_source_proves_lower_bound(
            source.rhs,
            0,
            assumptions,
            w,
            unknowns,
        )
    if kind.endswith("divsi") or kind.endswith("divui"):
        if lower_bound != 0:
            return False
        return _ixsimpl_index_source_proves_lower_bound(
            source.lhs,
            0,
            assumptions,
            w,
            unknowns,
        ) and _ixsimpl_index_source_proves_lower_bound(
            source.rhs,
            1,
            assumptions,
            w,
            unknowns,
        )
    if kind.endswith("remsi") or kind.endswith("remui"):
        if lower_bound != 0:
            return False
        return _ixsimpl_index_source_proves_lower_bound(
            source.lhs,
            0,
            assumptions,
            w,
            unknowns,
        ) and _ixsimpl_index_source_proves_lower_bound(
            source.rhs,
            1,
            assumptions,
            w,
            unknowns,
        )
    return False


def _ixsimpl_pointer_source_offset_proves_nonnegative(
    source,
    assumptions,
    w,
    unknowns,
):
    if isinstance(source, _PointerBase):
        return True
    if isinstance(source, _PointerAdd):
        return _ixsimpl_pointer_source_offset_proves_nonnegative(
            source.base,
            assumptions,
            w,
            unknowns,
        ) and _ixsimpl_index_source_proves_lower_bound(
            source.offset,
            0,
            assumptions,
            w,
            unknowns,
        )
    offset = _ixsimpl_pointer_offset_expr(source, w, unknowns)
    if offset is None:
        return False
    return _ixsimpl_proves(offset >= 0, assumptions, w)


def _dma_source_element_offset_proves_nonnegative(
    state,
    pointer_source,
    shape,
    inner_dim,
    packet_elements,
    w,
):
    unknowns = {}
    assumptions = _dma_packet_dim_assumptions(
        shape,
        inner_dim,
        packet_elements,
        w,
    ) + _ixsimpl_assume_fact_exprs(state, unknowns, w)
    return _ixsimpl_pointer_source_offset_proves_nonnegative(
        pointer_source,
        assumptions,
        w,
        unknowns,
    )


def _require_dma_packet_source_contiguous_bytes(
    state,
    pointer_source,
    source_element_byte_width,
    packet_bytes,
    shape,
    inner_dim,
    w,
):
    if (
        source_element_byte_width is None
        or source_element_byte_width <= 0
        or packet_bytes % source_element_byte_width
    ):
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: source element byte width does not divide "
            f"{packet_bytes}-byte DMA packet"
        )
    packet_elements = int(packet_bytes) // int(source_element_byte_width)
    if packet_elements <= 1:
        return
    unknowns = {}
    assumptions = _dma_packet_dim_assumptions(
        shape, inner_dim, packet_elements, w
    ) + _ixsimpl_assume_fact_exprs(state, unknowns, w)
    offset = _ixsimpl_pointer_byte_offset_expr(
        pointer_source, source_element_byte_width, w, unknowns, inner_dim
    )
    if offset is None:
        offset = _ixsimpl_pointer_packet_byte_offset_expr(
            pointer_source,
            source_element_byte_width,
            w,
            unknowns,
            inner_dim,
            packet_elements,
            assumptions,
        )
    if offset is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: DMA packet source pointer logical bytes are "
            f"not provably contiguous across {packet_bytes}-byte packets"
        )
    for packet_element in range(1, packet_elements):
        shifted = _ixsimpl_shift_dim(offset, inner_dim, packet_element, w)
        expected = offset + int(packet_element * source_element_byte_width)
        if not _ixsimpl_expr_equal(shifted, expected, assumptions, w):
            raise ValueError(
                "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                "without faithful DMA: DMA packet source pointer logical bytes "
                f"are not provably contiguous across {packet_bytes}-byte packets"
            )


def _require_dma_packet_mask_uniform(
    state,
    mask_source,
    shape,
    packet_elements,
    inner_dim,
    w,
):
    if mask_source is None or packet_elements <= 1:
        return
    unknowns = {}
    mask = _ixsimpl_mask_expr(mask_source, w, unknowns, inner_dim)
    if mask is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: DMA packet mask is not provably uniform "
            "across packet elements"
        )
    assumptions = _dma_packet_dim_assumptions(
        shape, inner_dim, packet_elements, w
    ) + _ixsimpl_assume_fact_exprs(state, unknowns, w)
    for packet_element in range(1, packet_elements):
        shifted = _ixsimpl_shift_dim(mask, inner_dim, packet_element, w)
        if not _ixsimpl_predicate_equivalent(mask, shifted, assumptions, w):
            if _ixsimpl_packet_mask_uniform(
                mask_source, inner_dim, packet_elements, assumptions, w, unknowns
            ):
                continue
            raise ValueError(
                "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                "without faithful DMA: DMA packet mask is not provably uniform "
                "across packet elements"
            )


def _ixsimpl_mod_zero(expr, divisor, assumptions, w):
    return _ixsimpl_proves(
        w.sym_ctx.eq(w.mod(expr, int(divisor)), w.sym_ctx.int_(0)),
        assumptions,
        w,
    )


def _ixsimpl_add_const(expr, value, w):
    return expr + w.sym_ctx.int_(int(value))


def _ixsimpl_packet_slt_uniform(lhs, rhs, inner_dim, packet_elements, assumptions, w):
    shifted_lhs = _ixsimpl_shift_dim(lhs, inner_dim, 1, w)
    shifted_rhs = _ixsimpl_shift_dim(rhs, inner_dim, 1, w)
    if not _ixsimpl_expr_equal(
        shifted_lhs, _ixsimpl_add_const(lhs, 1, w), assumptions, w
    ):
        return False
    if not _ixsimpl_expr_equal(shifted_rhs, rhs, assumptions, w):
        return False
    return _ixsimpl_mod_zero(
        lhs, packet_elements, assumptions, w
    ) and _ixsimpl_mod_zero(rhs, packet_elements, assumptions, w)


def _ixsimpl_packet_mask_uniform(
    source, inner_dim, packet_elements, assumptions, w, unknowns
):
    if source is None or packet_elements <= 1:
        return True
    if not _mask_source_depends_on_dim(source, inner_dim):
        return True
    if isinstance(source, _MaskConst):
        return True
    if isinstance(source, _MaskAnd):
        return _ixsimpl_packet_mask_uniform(
            source.lhs, inner_dim, packet_elements, assumptions, w, unknowns
        ) and _ixsimpl_packet_mask_uniform(
            source.rhs, inner_dim, packet_elements, assumptions, w, unknowns
        )
    if not isinstance(source, _MaskCompare):
        return False
    lhs = _ixsimpl_index_expr(source.lhs, w, unknowns, inner_dim)
    rhs = _ixsimpl_index_expr(source.rhs, w, unknowns, inner_dim)
    if lhs is None or rhs is None:
        return False
    if source.predicate in {"slt", "ult"}:
        return _ixsimpl_packet_slt_uniform(
            lhs, rhs, inner_dim, packet_elements, assumptions, w
        )
    if source.predicate in {"sgt", "ugt"}:
        return _ixsimpl_packet_slt_uniform(
            rhs, lhs, inner_dim, packet_elements, assumptions, w
        )
    return False


def _emit_dma_packet_ptrs(
    builder,
    state,
    address,
    address_plan,
    memdesc,
    memdescs,
    lds_layout,
    packet_bytes,
    layout,
    component,
    w,
):
    packet_elements = _dma_packet_elements(address, packet_bytes)
    if packet_elements is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: unknown DMA packet element count"
        )
    pointer_source = _async_copy_pointer_source(
        state,
        address,
        "ttg.async_copy_global_to_local source",
    )
    _require_dma_packet_source_contiguous_bytes(
        state,
        pointer_source,
        address.element_byte_width,
        packet_bytes,
        address_plan.shape,
        len(address_plan.shape) - 1,
        w,
    )
    dim_bindings, width, active, thread = _dma_packet_dim_bindings(
        builder,
        address_plan,
        memdesc,
        layout,
        packet_elements,
        component,
        w,
    )
    source = _materialize_dma_source_pointer_value(
        builder,
        state,
        pointer_source,
        dim_bindings,
        address.element_byte_width,
        address_plan.shape,
        len(address_plan.shape) - 1,
        packet_elements,
        w,
    )
    destination_base = _emit_memdesc_base_ptr(
        builder,
        memdesc,
        memdescs,
        lds_layout,
        state,
        w.i32(),
        4,
        w,
        "ttg.async_copy_global_to_local destination",
    )
    destination_offset = _dma_packet_uniform_destination_dword_offset_expr(
        builder,
        address_plan,
        memdesc,
        layout,
        packet_elements,
        packet_bytes,
        component,
        thread,
        w,
    )
    if destination_offset is None:
        uniform_bindings = _dma_packet_uniform_dim_bindings(
            builder,
            address_plan,
            memdesc,
            layout,
            packet_elements,
            component,
            thread,
            w,
        )
        destination_offset = _memdesc_dma_dword_offset_expr(
            memdesc,
            memdesc.shape,
            uniform_bindings,
            packet_elements,
            packet_bytes,
            w,
        )
    destination_offset = _inline_index_expr_bindings(destination_offset, w)
    destination = builder.ptr_add(
        destination_base,
        _materialize_index_value(builder, destination_offset, {}, w),
    )
    return source, destination, dim_bindings, width, active


def _select_dma_packet_lowering(
    state,
    address,
    address_plan,
    memdesc,
    memdescs,
    lds_layout,
    mask_value,
    w,
):
    errors = []
    pointer_source = _async_copy_pointer_source(
        state,
        address,
        "ttg.async_copy_global_to_local source",
    )
    for dma_bytes in _require_dma_packet_byte_candidates(
        address, memdesc, memdescs, lds_layout
    ):
        try:
            dma_packet_layout = _validate_dma_packet_layout(
                address_plan, address, memdesc, dma_bytes
            )
            packet_elements = _dma_packet_elements(address, dma_bytes)
            if packet_elements is None:
                raise ValueError(
                    "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                    "without faithful DMA: unknown DMA packet element count"
                )
            _require_dma_packet_mask_uniform(
                state,
                mask_value,
                address_plan.shape,
                packet_elements,
                len(address_plan.shape) - 1,
                w,
            )
            _require_dma_packet_source_contiguous_bytes(
                state,
                pointer_source,
                address.element_byte_width,
                dma_bytes,
                address_plan.shape,
                len(address_plan.shape) - 1,
                w,
            )
            copy_component_count = _dma_packet_component_count(
                address_plan, dma_packet_layout, packet_elements
            )
        except ValueError as exc:
            errors.append(exc)
            continue
        return dma_bytes, dma_packet_layout, copy_component_count
    if errors:
        raise errors[-1]
    raise ValueError(
        "tlx_wave bridge cannot lower ttg.async_copy_global_to_local without "
        "faithful DMA: no legal DMA packet size was found"
    )


def _zero_simd_for_element(builder, element_type, width, w, context):
    return _splat_constant_value(
        builder,
        element_type,
        0.0 if element_type in {"f16", "bf16", "f32"} else 0,
        width,
        w,
        context,
    )


def _async_copy_other_value_plan(address_plan, address):
    return replace(
        address_plan,
        element_type=address.element_type,
        element_byte_width=address.element_byte_width,
        pointee_type=None,
    )


def _async_copy_source_plan(address, values):
    if address.op == "amdg.buffer_load_to_local":
        if address.offset_value_id is None:
            raise ValueError(
                "tlx_wave bridge cannot lower amdg.buffer_load_to_local "
                "without offsets"
            )
        return _async_copy_other_value_plan(values[address.offset_value_id], address)
    return values[address.address_value_id]


def _async_copy_pointer_source(state, address, context):
    base = _require_lowered_value(
        state["wave_values"],
        address.address_value_id,
        "pointer_expr",
        context,
    )
    if address.op != "amdg.buffer_load_to_local":
        return base
    if address.offset_value_id is None:
        raise ValueError(
            "tlx_wave bridge cannot lower amdg.buffer_load_to_local without offsets"
        )
    offset = _require_lowered_value(
        state["wave_values"],
        address.offset_value_id,
        "index_expr",
        "amdg.buffer_load_to_local offsets",
    )
    return _PointerAdd(base, offset)


def _emit_async_copy_via_load_store(
    builder,
    state,
    address,
    address_plan,
    memdesc,
    memdescs,
    lds_layout,
    mask_value,
    after_token,
    w,
    stats,
):
    if address_plan.shape != memdesc.shape:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            "without faithful DMA: source tensor shape "
            f"{address_plan.shape} does not match destination memdesc shape "
            f"{memdesc.shape}"
        )
    pointer_source = _async_copy_pointer_source(
        state,
        address,
        "ttg.async_copy_global_to_local source",
    )
    source_layout = _dma_source_tensor_layout_info(
        address_plan,
        "ttg.async_copy_global_to_local source",
        "generic async copy lowering",
    )
    component_count = _async_copy_source_layout_component_count(
        address_plan,
        source_layout,
        "ttg.async_copy_global_to_local source",
        "generic async copy lowering",
    )
    token = after_token
    for component in range(component_count):
        dim_bindings, width, active = _async_copy_source_dim_bindings(
            builder,
            address_plan,
            source_layout,
            w,
            "ttg.async_copy_global_to_local source",
            component=component,
        )
        source = _materialize_bounded_pointer_value(
            builder,
            state,
            pointer_source,
            dim_bindings,
            address.element_byte_width,
            address.element_byte_width,
            address_plan.shape,
            w,
            assume_pointer_range=mask_value is None
            and _async_copy_source_component_all_lanes_active(
                address_plan,
                source_layout,
                "ttg.async_copy_global_to_local source",
                "generic tensor lowering",
                component=component,
            ),
        )
        mask = active
        if mask_value is not None:
            user_mask = _materialize_component_mask_value(
                builder,
                mask_value,
                dim_bindings,
                w,
                width,
                "ttg.async_copy_global_to_local mask",
                component=component,
            )
            mask = _wave_mask_and(builder, mask, user_mask, w, width)

        result_type = w.simd_type(
            _wave_element_type(
                address.element_type,
                w,
                "ttg.async_copy_global_to_local source",
            ),
            width,
        )
        fallback = (
            _load_other_value(
                builder,
                state,
                address.other_value_id,
                _async_copy_other_value_plan(address_plan, address),
                width,
                w,
                "ttg.async_copy_global_to_local other",
                component=component,
            )
            if address.other_value_id is not None
            else _zero_simd_for_element(
                builder,
                address.element_type,
                width,
                w,
                "ttg.async_copy_global_to_local inactive lanes",
            )
        )
        loaded, token = _emit_masked_load(
            builder,
            source,
            result_type,
            mask,
            fallback,
            token,
            w,
        )
        destination = _emit_memdesc_ptr(
            builder,
            memdesc,
            memdescs,
            lds_layout,
            state,
            dim_bindings,
            width,
            w,
            "ttg.async_copy_global_to_local destination",
        )
        store_mask = active if address.other_value_id is not None else mask
        token = _emit_component_store(
            builder,
            loaded,
            destination,
            store_mask,
            token,
            w,
        )
    stats.async_copies += 1
    return token


def _emit_async_copy(
    builder, state, kernel, address, lds_layout, after_token, w, stats
):
    if address.memdesc_value_id is None:
        raise ValueError("tlx_wave bridge cannot lower async copy without LDS memdesc")
    memdescs = state["memdescs"]
    if address.memdesc_value_id not in memdescs:
        raise ValueError(
            "tlx_wave bridge cannot lower async copy: unknown LDS memdesc "
            f"{address.memdesc_value_id}"
        )

    memdesc = memdescs[address.memdesc_value_id]
    if address.address_value_id is None:
        raise ValueError(
            "tlx_wave bridge cannot lower async copy without a source address value"
        )
    address_plan = _async_copy_source_plan(address, state["values"])
    if address.element_type != memdesc.element_type:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local: "
            f"source element type {address.element_type} does not match "
            f"destination memdesc element type {memdesc.element_type}"
        )
    mask_value = None
    if address.mask_value_id is not None:
        mask_value = _require_typed_wave_value(
            state["wave_values"],
            address.mask_value_id,
            "ttg.async_copy_global_to_local mask",
        )
        if mask_value.kind not in {"mask_expr", "mask_tuple"}:
            raise ValueError(
                "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                f"mask lowered as {mask_value.kind}"
            )
        if mask_value.kind == "mask_expr":
            mask_value = mask_value.value
    if address.other_value_id is not None:
        return _emit_async_copy_via_load_store(
            builder,
            state,
            address,
            address_plan,
            memdesc,
            memdescs,
            lds_layout,
            mask_value,
            after_token,
            w,
            stats,
        )
    try:
        dma_bytes, dma_packet_layout, copy_component_count = (
            _select_dma_packet_lowering(
                state,
                address,
                address_plan,
                memdesc,
                memdescs,
                lds_layout,
                mask_value,
                w,
            )
        )
    except ValueError:
        return _emit_async_copy_via_load_store(
            builder,
            state,
            address,
            address_plan,
            memdesc,
            memdescs,
            lds_layout,
            mask_value,
            after_token,
            w,
            stats,
        )
    stats.async_copies += 1

    token = after_token
    for component in range(copy_component_count):
        source, destination, dim_bindings, width, active = _emit_dma_packet_ptrs(
            builder,
            state,
            address,
            address_plan,
            memdesc,
            memdescs,
            lds_layout,
            dma_bytes,
            dma_packet_layout,
            component,
            w,
        )
        mask = active
        if mask_value is not None:
            user_mask = _materialize_component_mask_value(
                builder,
                mask_value,
                dim_bindings,
                w,
                width,
                "ttg.async_copy_global_to_local mask",
                component=component,
            )
            mask = _wave_mask_and(builder, mask, user_mask, w, width)

        def emit_copy(copy_after):
            stats.dma_load_lds += 1
            return builder.dma_load_lds(
                source, destination, after=copy_after, bytes=dma_bytes
            )

        if mask is None:
            token = emit_copy(token)
            continue
        if token is None:
            token = builder.token()
        with builder.where(mask, [w.mem_token_type()]) as where_op:
            component_token = emit_copy(token)
            builder.yield_([component_token])
        token = where_op.results[0]
    return token


