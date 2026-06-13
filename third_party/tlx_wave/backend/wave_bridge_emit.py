import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .wave_bridge_plan import (
    _GFX950_DOT_PARENT_LAYOUT,
    _GFX950_F16_MMA_KIND,
    _GFX950_MMA_M,
    _GFX950_MMA_N,
    _GFX950_MMA_REGS,
    _GFX950_MMA_SHAPE,
    _GFX950_MMA_WAVE,
    _GFX950_SHARED_LAYOUT,
    _BlockedEncodingInfo,
    _DotOperandEncodingInfo,
    _SwizzledSharedEncodingInfo,
    _WaveAsyncStats,
    _async_address_by_token,
    _bridge_stage,
    _compute_lds_layout,
    _local_load_address_by_result,
    _memdescs_by_id,
    _target_triple,
    _values_by_id,
)


_WAVE_TOOL_NAMES = (
    "wave-calibrate-report",
    "wave-opt",
    "wave-sim-report",
    "wave-symbols-test",
    "wave-translate",
    "wavec",
)


@dataclass(frozen=True)
class _WaveValue:
    kind: str
    value: object
    physical_value_id: int | None = None


@dataclass(frozen=True)
class _IndexExpr:
    expr: object
    bindings: dict


@dataclass(frozen=True)
class _DimBinding:
    dim: int


@dataclass(frozen=True)
class _MaskConst:
    value: bool


@dataclass(frozen=True)
class _MaskAnd:
    lhs: object
    rhs: object


@dataclass(frozen=True)
class _MaskCompare:
    predicate: str
    lhs: object
    rhs: object


@dataclass(frozen=True)
class _PointerBase:
    value: object


@dataclass(frozen=True)
class _PointerAdd:
    base: object
    offset: object


def _is_deferred_index(source):
    if isinstance(source, _DimBinding):
        return True
    if isinstance(source, _IndexExpr):
        return any(_is_deferred_index(value) for value in source.bindings.values())
    return False


def _is_deferred_mask(source):
    if isinstance(source, _MaskAnd):
        return _is_deferred_mask(source.lhs) or _is_deferred_mask(source.rhs)
    if isinstance(source, _MaskCompare):
        return _is_deferred_index(source.lhs) or _is_deferred_index(source.rhs)
    return False


def _is_deferred_pointer(source):
    if isinstance(source, _PointerAdd):
        return _is_deferred_pointer(source.base) or _is_deferred_index(source.offset)
    return False


_CMPI_PREDICATES = {
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

def _attr_method(attr, method):
    if attr is None:
        return None
    fn = getattr(attr, method, None)
    if fn is None:
        raise RuntimeError(
            "tlx_wave bridge requires typed TTGIR encoding Attribute helpers; "
            f"missing {method}. Rebuild Triton after updating python/src/ir.cc."
        )
    return fn


def _attr_bool(attr, method):
    fn = _attr_method(attr, method)
    return bool(fn()) if fn is not None else False


def _attr_value(attr, method):
    fn = _attr_method(attr, method)
    return fn() if fn is not None else None


def _int_tuple(values):
    if values is None:
        raise ValueError("missing typed TTGIR encoding field")
    return tuple(int(value) for value in values)


def _blocked_encoding_info(attr, raw_encoding, context):
    if attr is None or not _attr_bool(attr, "is_blocked_encoding"):
        raise ValueError(
            f"tlx_wave bridge expected blocked TTGIR encoding for {context}, "
            f"got {raw_encoding}"
        )
    return _BlockedEncodingInfo(
        _int_tuple(_attr_value(attr, "get_blocked_size_per_thread")),
        _int_tuple(_attr_value(attr, "get_blocked_threads_per_warp")),
        _int_tuple(_attr_value(attr, "get_blocked_warps_per_cta")),
        _int_tuple(_attr_value(attr, "get_blocked_order")),
    )


def _swizzled_shared_encoding_info(attr, raw_encoding, context):
    if attr is None or not _attr_bool(attr, "is_swizzled_shared_encoding"):
        raise ValueError(
            "tlx_wave bridge expected #ttg.swizzled_shared encoding for "
            f"{context}, got {raw_encoding}"
        )
    return _SwizzledSharedEncodingInfo(
        int(_attr_value(attr, "get_swizzled_shared_vec")),
        int(_attr_value(attr, "get_swizzled_shared_per_phase")),
        int(_attr_value(attr, "get_swizzled_shared_max_phase")),
        _int_tuple(_attr_value(attr, "get_swizzled_shared_order")),
    )


def _same_blocked_encoding(lhs, rhs):
    return (
        lhs.size_per_thread == rhs.size_per_thread
        and lhs.threads_per_warp == rhs.threads_per_warp
        and lhs.warps_per_cta == rhs.warps_per_cta
        and lhs.order == rhs.order
    )


def _same_layout_encoding(lhs, rhs):
    if lhs.encoding_attr is None or rhs.encoding_attr is None:
        return lhs.encoding_attr is None and rhs.encoding_attr is None
    if _attr_bool(lhs.encoding_attr, "is_blocked_encoding") and _attr_bool(
        rhs.encoding_attr, "is_blocked_encoding"
    ):
        lhs_info = _blocked_encoding_info(
            lhs.encoding_attr, lhs.encoding, "layout equality source"
        )
        rhs_info = _blocked_encoding_info(
            rhs.encoding_attr, rhs.encoding, "layout equality result"
        )
        return _same_blocked_encoding(lhs_info, rhs_info)
    return False


def _set_wave_value(wave_values, value_id, kind, value, physical_value_id=None):
    if physical_value_id is None and kind == "fragment":
        physical_value_id = value_id
    wave_values[value_id] = _WaveValue(kind, value, physical_value_id)


def _raw_wave_value(value):
    return value.value if isinstance(value, _WaveValue) else value


def _physical_value_plan(values, lowered, value_id):
    if isinstance(lowered, _WaveValue) and lowered.physical_value_id is not None:
        return values[lowered.physical_value_id]
    return values[value_id]


def _require_wave_value(wave_values, value_id, kinds, context):
    value = wave_values.get(value_id)
    if value is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: TTGIR value {value_id} "
            "has not been lowered"
        )
    if not isinstance(value, _WaveValue):
        return value
    if value.kind not in kinds:
        expected = ", ".join(kinds)
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: TTGIR value {value_id} "
            f"lowered as {value.kind}, expected {expected}"
        )
    return value.value


def _op_by_result_id(plan):
    return {
        result_id: op
        for op in plan.ops
        for result_id in op.results
    }


def _dim_symbol(w, dim):
    return w.sym(f"tlx_dim{dim}")


def _is_bool_value(value):
    return value.type == "i1" or value.element_type == "i1"


def _is_integer_or_index_value(value):
    types = {value.type, value.element_type}
    return "index" in types or any(f"i{bits}" in types for bits in (1, 8, 16, 32, 64))


def _tensor_lane_width(value_plan, context):
    if getattr(value_plan, "type_kind", None) != "tensor":
        return None
    layout = _blocked_encoding_info(
        value_plan.encoding_attr,
        value_plan.encoding,
        context,
    )
    rank = len(value_plan.shape)
    if (
        len(layout.size_per_thread) != rank
        or len(layout.threads_per_warp) != rank
        or len(layout.warps_per_cta) != rank
    ):
        raise ValueError(
            f"tlx_wave bridge blocked layout rank does not match {context}: "
            f"shape={value_plan.shape}, encoding={value_plan.encoding}"
        )
    return _product(layout.threads_per_warp)


def _maybe_splat(builder, value, force_width, w):
    if force_width is None:
        return value
    if not w.SimdType.isinstance(value.type):
        return builder.splat(value, width=force_width)
    width = w.SimdType(value.type).width
    if width != force_width:
        raise ValueError(
            "tlx_wave bridge cannot use SIMD value with mismatched lane width: "
            f"got {width}, expected {force_width}"
        )
    return value


def _is_wave_index_type(typ, w):
    if typ is None:
        return False
    if str(typ) == "index":
        return True
    try:
        return typ == w.index_type()
    except AttributeError:
        return False


def _is_wave_simd_index_type(typ, w):
    if typ is None:
        return False
    simd_type = getattr(w, "SimdType", None)
    if simd_type is not None:
        try:
            if simd_type.isinstance(typ):
                simd = simd_type(typ)
                for attr in ("element_type", "elementType", "element"):
                    element_type = getattr(simd, attr, None)
                    if element_type is not None:
                        return _is_wave_index_type(element_type, w)
        except (AttributeError, TypeError):
            pass
    compact = str(typ).replace(" ", "")
    return "simd<index" in compact or "xindex>" in compact


def _wave_cmpi_operand(builder, value, w):
    typ = getattr(value, "type", None)
    if _is_wave_index_type(typ, w) or _is_wave_simd_index_type(typ, w):
        return builder.index_cast(value, w.i64())
    return value


def _wave_cmpi(builder, predicate, lhs, rhs, w):
    return builder.cmpi(
        predicate,
        _wave_cmpi_operand(builder, lhs, w),
        _wave_cmpi_operand(builder, rhs, w),
    )


def _dim_binding_value(dim_bindings, binding, w):
    symbol = _dim_symbol(w, binding.dim)
    value = dim_bindings.get(symbol)
    if value is None:
        raise ValueError(
            "tlx_wave bridge cannot materialize layout-dependent index dim "
            f"{binding.dim}; available dims={list(dim_bindings)}"
        )
    return value


def _materialize_index_value(builder, source, dim_bindings, w, force_width=None):
    if isinstance(source, _IndexExpr):
        bindings = {
            symbol: _materialize_index_value(builder, value, dim_bindings, w)
            for symbol, value in source.bindings.items()
        }
        value = builder.index_expr(source.expr, bindings)
        return _maybe_splat(builder, value, force_width, w)
    if isinstance(source, _DimBinding):
        return _maybe_splat(
            builder, _dim_binding_value(dim_bindings, source, w), force_width, w
        )
    return _maybe_splat(builder, source, force_width, w)


def _false_mask(builder, w, width):
    lane = builder.lane_id(width=width)
    return _wave_cmpi(builder, "ne", lane, lane, w)


def _true_mask(builder, w, width):
    lane = builder.lane_id(width=width)
    return _wave_cmpi(builder, "eq", lane, lane, w)


def _materialize_mask_value(builder, source, dim_bindings, w, width):
    if isinstance(source, _MaskConst):
        return _true_mask(builder, w, width) if source.value else _false_mask(builder, w, width)
    if isinstance(source, _MaskAnd):
        lhs = _materialize_mask_value(builder, source.lhs, dim_bindings, w, width)
        rhs = _materialize_mask_value(builder, source.rhs, dim_bindings, w, width)
        return builder.select(lhs, rhs, _false_mask(builder, w, width))
    if isinstance(source, _MaskCompare):
        lhs = _materialize_index_value(
            builder, source.lhs, dim_bindings, w, force_width=width
        )
        rhs = _materialize_index_value(
            builder, source.rhs, dim_bindings, w, force_width=width
        )
        return _wave_cmpi(builder, source.predicate, lhs, rhs, w)
    return source


def _materialize_pointer_value(builder, source, dim_bindings, w):
    if isinstance(source, _PointerBase):
        return source.value
    if isinstance(source, _PointerAdd):
        base = _materialize_pointer_value(builder, source.base, dim_bindings, w)
        offset = _materialize_index_value(builder, source.offset, dim_bindings, w)
        return builder.ptr_add(base, offset)
    return source


def _wave_element_type(element_type, w, context):
    if element_type == "i1":
        return w.i1()
    if element_type == "i8":
        return w.i8()
    if element_type == "i16":
        return w.IntegerType.get_signless(16)
    if element_type == "i32":
        return w.i32()
    if element_type == "i64":
        return w.i64()
    if element_type == "f16":
        return w.f16()
    if element_type == "bf16":
        return w.bf16()
    if element_type == "f32":
        return w.f32()
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported element type "
        f"{element_type}"
    )


def _simd_type_for_value(value_plan, width, w, context):
    if value_plan.type_kind != "tensor":
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected tensor value, "
            f"got {value_plan.type}"
        )
    return w.simd_type(
        _wave_element_type(value_plan.element_type, w, context),
        width,
    )


def _constant_value(builder, element_type, value, w, context):
    wave_type = _wave_element_type(element_type, w, context)
    if element_type in {"f16", "bf16", "f32"}:
        return builder.constant(wave_type, float(value))
    if element_type == "i1":
        return builder.constant(wave_type, int(bool(value)))
    return builder.constant(wave_type, int(value))


def _splat_constant_value(builder, element_type, value, width, w, context):
    scalar = _constant_value(builder, element_type, value, w, context)
    return builder.splat(scalar, element_type=scalar.type, width=width)


def _zero_simd_value(builder, value_plan, width, w, context):
    return _splat_constant_value(
        builder,
        value_plan.element_type,
        0.0 if value_plan.element_type in {"f16", "bf16", "f32"} else 0,
        width,
        w,
        context,
    )


def _wave_mask_and(builder, lhs, rhs, w, width):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    return builder.select(lhs, rhs, _false_mask(builder, w, width))


def _blocked_tensor_layout_info(value_plan, context, lowering_name):
    if value_plan.type_kind != "tensor" or not value_plan.shape:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected ranked tensor, "
            f"got {value_plan.type}"
        )
    if len(value_plan.shape) > 2:
        raise ValueError(
            f"tlx_wave bridge {lowering_name} supports rank <= 2 for "
            f"{context}, got shape={value_plan.shape}"
        )
    layout = _blocked_encoding_info(
        value_plan.encoding_attr,
        value_plan.encoding,
        context,
    )
    rank = len(value_plan.shape)
    if (
        len(layout.size_per_thread) != rank
        or len(layout.threads_per_warp) != rank
        or len(layout.warps_per_cta) != rank
    ):
        raise ValueError(
            f"tlx_wave bridge blocked layout rank does not match {context}: "
            f"shape={value_plan.shape}, encoding={value_plan.encoding}"
        )
    return layout


def _blocked_layout_component_count(value_plan, context, lowering_name):
    layout = _blocked_tensor_layout_info(value_plan, context, lowering_name)
    return _product(layout.size_per_thread)


def _blocked_layout_dim_bindings(
    builder, value_plan, w, context, symbol_prefix, lowering_name, component=0
):
    layout = _blocked_tensor_layout_info(value_plan, context, lowering_name)
    component_count = _product(layout.size_per_thread)
    if component < 0 or component >= component_count:
        raise ValueError(
            f"tlx_wave bridge {lowering_name} component {component} is out of "
            f"range for {context}: sizePerThread={layout.size_per_thread}, "
            f"encoding={value_plan.encoding}"
        )
    rank = len(value_plan.shape)
    for dim, extent in enumerate(value_plan.shape):
        covered = (
            layout.size_per_thread[dim]
            * layout.threads_per_warp[dim]
            * layout.warps_per_cta[dim]
        )
        if covered < extent:
            raise ValueError(
                f"tlx_wave bridge {lowering_name} cannot cover the "
                f"full tensor extent for {context}: dim {dim} has extent "
                f"{extent}, but sizePerThread * threadsPerWarp * warpsPerCTA "
                f"covers only {covered}; encoding={value_plan.encoding}"
            )

    width = _product(layout.threads_per_warp)
    thread = builder.workitem_id(axis=0, width=width)
    thread_sym = w.sym(f"{symbol_prefix}_{value_plan.value_id}_thread")
    lane_coords = _delinearize_expr(
        w,
        w.mod(thread_sym, width),
        layout.threads_per_warp,
        layout.order,
    )
    warp_coords = _delinearize_expr(
        w,
        w.floor(thread_sym / width),
        layout.warps_per_cta,
        layout.order,
    )
    component_coords = (
        _delinearize_expr(
            w, w.sym_ctx.int_(component), layout.size_per_thread, layout.order
        )
        if component_count != 1
        else None
    )

    dim_bindings = {}
    active = None
    for dim in range(rank):
        tile_coord = lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim]
        coord_expr = (
            component_coords[dim] + layout.size_per_thread[dim] * tile_coord
            if component_coords is not None
            else tile_coord
        )
        coord = builder.index_expr(coord_expr, {thread_sym: thread})
        dim_bindings[_dim_symbol(w, dim)] = coord
        extent = builder.splat(
            builder.constant(w.index_type(), value_plan.shape[dim]),
            width=width,
        )
        in_bounds = _wave_cmpi(builder, "ult", coord, extent, w)
        active = _wave_mask_and(builder, active, in_bounds, w, width)
    return dim_bindings, width, active


def _blocked_tensor_dim_bindings(builder, value_plan, w, context, component=0):
    return _blocked_layout_dim_bindings(
        builder,
        value_plan,
        w,
        context,
        "tlx_tensor",
        "generic tensor lowering",
        component=component,
    )


def _linearized_tensor_offset(builder, shape, dim_bindings, w):
    offset = w.sym_ctx.int_(0)
    stride = 1
    for dim in reversed(range(len(shape))):
        offset = offset + _dim_symbol(w, dim) * stride
        stride *= int(shape[dim])
    return builder.index_expr(offset, dim_bindings)


def _memdesc_logical_size_bytes(memdesc):
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot size LDS memdesc {memdesc.name or memdesc.source} "
            f"with element type {memdesc.element_type}"
        )
    return _product(memdesc.shape) * memdesc.element_byte_width


def _unsupported_memdesc_view(context, memdesc):
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported memdesc view "
        f"{memdesc.view_op}; view value={memdesc.value_id}, "
        f"encoding={memdesc.encoding}"
    )


def _is_identity_shared_layout(memdesc, shared):
    return (
        len(memdesc.shape) == 1
        and shared.vec == 1
        and shared.per_phase == 1
        and shared.max_phase == 1
        and shared.order == (0,)
    )


def _validate_generic_shared_layout(memdesc, context):
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, context
        )
    except ValueError as exc:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: {exc}"
        ) from exc
    if _is_identity_shared_layout(memdesc, shared):
        return
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported shared-memory "
        "encoding for generic LDS addressing; expected one-dimensional "
        "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, "
        "order = [0]}> until shared-layout transforms are implemented; "
        f"got shape={memdesc.shape}, vec={shared.vec}, "
        f"perPhase={shared.per_phase}, maxPhase={shared.max_phase}, "
        f"order={shared.order}, encoding={memdesc.encoding}"
    )


def _emit_memdesc_index_offset(
    builder,
    memdesc,
    state,
    pointer_element_bytes,
    w,
    context,
):
    if pointer_element_bytes is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_index view has "
            f"unknown element byte width for {memdesc.element_type}"
        )
    view_bytes = _memdesc_logical_size_bytes(memdesc)
    if memdesc.static_index is not None:
        byte_offset = memdesc.static_index * view_bytes
        if byte_offset % pointer_element_bytes:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: memdesc_index byte "
                f"offset {byte_offset} is not aligned to pointer element size "
                f"{pointer_element_bytes}"
            )
        return builder.index_expr(w.sym_ctx.int_(byte_offset // pointer_element_bytes))
    if view_bytes % pointer_element_bytes:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_index view size "
            f"{view_bytes} is not aligned to pointer element size "
            f"{pointer_element_bytes}"
        )
    stride = view_bytes // pointer_element_bytes
    if len(memdesc.view_operands) < 2:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_index view "
            "does not record an index operand"
        )
    slot = _materialize_index_value(
        builder,
        _require_lowered_value(
            state["wave_values"],
            memdesc.view_operands[1],
            "index_expr",
            f"{context} memdesc_index slot",
        ),
        {},
        w,
    )
    slot_sym = w.sym(f"tlx_memdesc_{memdesc.value_id}_slot")
    return builder.index_expr(slot_sym * stride, {slot_sym: slot})


def _emit_memdesc_base_ptr(
    builder,
    memdesc,
    memdescs,
    lds_layout,
    state,
    pointer_element_type,
    pointer_element_bytes,
    w,
    context,
):
    if memdesc.kind == "allocation":
        if memdesc.value_id not in lds_layout.offsets:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: no LDS placement for "
                f"memdesc {memdesc.value_id}"
            )
        return builder.lds_base(
            pointer_element_type,
            offset=lds_layout.offsets[memdesc.value_id],
        )

    if memdesc.base_value_id is None or memdesc.base_value_id not in memdescs:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc view "
            f"{memdesc.value_id} has no known base"
        )
    if memdesc.view_op != "ttg.memdesc_index":
        _unsupported_memdesc_view(context, memdesc)

    base = _emit_memdesc_base_ptr(
        builder,
        memdescs[memdesc.base_value_id],
        memdescs,
        lds_layout,
        state,
        pointer_element_type,
        pointer_element_bytes,
        w,
        context,
    )
    return builder.ptr_add(
        base,
        _emit_memdesc_index_offset(
            builder,
            memdesc,
            state,
            pointer_element_bytes,
            w,
            context,
        ),
    )


def _memdesc_base_is_aligned(
    memdesc,
    memdescs,
    lds_layout,
    pointer_element_bytes,
    context,
):
    if pointer_element_bytes is None:
        return False
    if memdesc.kind == "allocation":
        if memdesc.value_id not in lds_layout.offsets:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: no LDS placement for "
                f"memdesc {memdesc.value_id}"
            )
        return lds_layout.offsets[memdesc.value_id] % pointer_element_bytes == 0

    if memdesc.base_value_id is None or memdesc.base_value_id not in memdescs:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc view "
            f"{memdesc.value_id} has no known base"
        )
    if memdesc.view_op != "ttg.memdesc_index":
        _unsupported_memdesc_view(context, memdesc)
    if not _memdesc_base_is_aligned(
        memdescs[memdesc.base_value_id],
        memdescs,
        lds_layout,
        pointer_element_bytes,
        context,
    ):
        return False

    view_bytes = _memdesc_logical_size_bytes(memdesc)
    if memdesc.static_index is not None:
        return (memdesc.static_index * view_bytes) % pointer_element_bytes == 0
    return view_bytes % pointer_element_bytes == 0


def _emit_memdesc_ptr(
    builder,
    memdesc,
    memdescs,
    lds_layout,
    state,
    dim_bindings,
    width,
    w,
    context,
):
    if memdesc.kind == "view" and memdesc.view_op != "ttg.memdesc_index":
        _unsupported_memdesc_view(context, memdesc)
    _validate_generic_shared_layout(memdesc, context)
    if memdesc.value_id not in lds_layout.offsets and memdesc.kind == "allocation":
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: no LDS placement for "
            f"memdesc {memdesc.value_id}"
        )
    element_type = _wave_element_type(memdesc.element_type, w, context)
    base = _emit_memdesc_base_ptr(
        builder,
        memdesc,
        memdescs,
        lds_layout,
        state,
        element_type,
        memdesc.element_byte_width,
        w,
        context,
    )
    offset = _linearized_tensor_offset(builder, memdesc.shape, dim_bindings, w)
    return builder.ptr_add(
        base,
        offset,
        w.simd_ptr_type(element_type, w.shared_address_space(), width),
    )


def _validate_generic_local_tensor(value_plan, memdesc, context):
    if value_plan.type_kind != "tensor":
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected tensor value, "
            f"got {value_plan.type}"
        )
    if value_plan.shape != memdesc.shape:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: tensor shape "
            f"{value_plan.shape} does not match memdesc shape {memdesc.shape}"
        )
    if value_plan.element_type != memdesc.element_type:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: tensor element type "
            f"{value_plan.element_type} does not match memdesc element type "
            f"{memdesc.element_type}"
        )


def _materialize_tensor_data(
    builder, lowered, value_plan, width, w, context, component=0
):
    if not isinstance(lowered, _WaveValue):
        raise ValueError(f"tlx_wave bridge internal error: untyped {context} value")
    if lowered.kind == "simd":
        if component != 0:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: requested component "
                f"{component} from single-component SIMD value"
            )
        if not w.SimdType.isinstance(lowered.value.type):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: value is not SIMD"
            )
        got_width = w.SimdType(lowered.value.type).width
        if got_width != width:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: SIMD width {got_width} "
                f"does not match layout width {width}"
            )
        return lowered.value
    if lowered.kind == "simd_tuple":
        if component < 0 or component >= len(lowered.value):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: component {component} "
                f"is out of range for {len(lowered.value)} SIMD components"
            )
        value = lowered.value[component]
        if not w.SimdType.isinstance(value.type):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: component "
                f"{component} is not SIMD"
            )
        got_width = w.SimdType(value.type).width
        if got_width != width:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: component {component} "
                f"SIMD width {got_width} does not match layout width {width}"
            )
        return value
    if lowered.kind == "index_expr":
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: integer index_expr data "
            "needs explicit value-type materialization"
        )
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: value lowered as {lowered.kind}; "
        f"TTGIR type={value_plan.type}"
    )


def _load_other_value(
    builder, state, other_id, result_plan, width, w, context, component=0
):
    other_plan = state["values"][other_id]
    lowered = state["wave_values"].get(other_id)
    if isinstance(lowered, _WaveValue) and lowered.kind in {"simd", "simd_tuple"}:
        return _materialize_tensor_data(
            builder, lowered, result_plan, width, w, context, component=component
        )
    if (
        other_plan.producer == "arith.constant"
        and other_plan.const_value is not None
    ):
        return _splat_constant_value(
            builder,
            result_plan.element_type,
            other_plan.const_value,
            width,
            w,
            context,
        )
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported `other` value "
        f"producer={other_plan.producer}, type={other_plan.type}"
    )


def _emit_masked_load(builder, ptr, result_type, mask, fallback, after_token, w):
    if mask is None:
        return builder.load(ptr, result_type, after=after_token)
    if after_token is None:
        after_token = builder.token()
    with builder.where(mask, [result_type, w.mem_token_type()]) as where_op:
        value, token = builder.load(ptr, result_type, after=after_token)
        builder.yield_([value, token])
        with where_op.otherwise():
            builder.yield_([fallback, after_token])
    return where_op.results[0], where_op.results[1]


def _shift_index_dims(source, axis):
    if isinstance(source, _DimBinding):
        dim = source.dim + 1 if source.dim >= axis else source.dim
        return _DimBinding(dim)
    if isinstance(source, _IndexExpr):
        return _IndexExpr(
            source.expr,
            {
                symbol: _shift_index_dims(binding, axis)
                for symbol, binding in source.bindings.items()
            },
        )
    return source


def _shift_mask_dims(source, axis):
    if isinstance(source, _MaskAnd):
        return _MaskAnd(
            _shift_mask_dims(source.lhs, axis),
            _shift_mask_dims(source.rhs, axis),
        )
    if isinstance(source, _MaskCompare):
        return _MaskCompare(
            source.predicate,
            _shift_index_dims(source.lhs, axis),
            _shift_index_dims(source.rhs, axis),
        )
    return source


def _shift_pointer_dims(source, axis):
    if isinstance(source, _PointerAdd):
        return _PointerAdd(
            _shift_pointer_dims(source.base, axis),
            _shift_index_dims(source.offset, axis),
        )
    return source


def _require_lowered_value(wave_values, value_id, kind, context):
    lowered = wave_values.get(value_id)
    if lowered is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: TTGIR value {value_id} "
            "has not been lowered by a preceding ordered TTGIR op"
        )
    if not isinstance(lowered, _WaveValue) or lowered.kind != kind:
        got = lowered.kind if isinstance(lowered, _WaveValue) else type(lowered).__name__
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: TTGIR value {value_id} "
            f"lowered as {got}, expected {kind}"
        )
    return lowered.value


def _init_argument_wave_values(builder, values, wave_values, w):
    for value in values.values():
        if value.kind != "argument" or value.base_arg_index is None:
            continue
        arg = builder.args[value.base_arg_index]
        if value.type_kind == "pointer":
            _set_wave_value(
                wave_values,
                value.value_id,
                "pointer_expr",
                arg,
            )
        elif value.type_kind == "scalar" and _is_integer_or_index_value(value):
            bound = arg if value.type == "index" else builder.index_cast(arg, w.index_type())
            _set_wave_value(
                wave_values,
                value.value_id,
                "index_expr",
                bound,
            )


def _emit_constant_op(builder, op, values, wave_values, w):
    if len(op.results) != 1:
        return
    value = values[op.results[0]]
    const = value.const_value
    if _is_bool_value(value) and isinstance(const, (bool, int)):
        _set_wave_value(
            wave_values,
            value.value_id,
            "mask_expr",
            _MaskConst(bool(const)),
        )
    elif (
        _is_integer_or_index_value(value)
        and isinstance(const, int)
        and not isinstance(const, bool)
    ):
        _set_wave_value(
            wave_values,
            value.value_id,
            "index_expr",
            builder.index_expr(w.sym_ctx.int_(const)),
        )


def _emit_program_id_op(builder, state, op, values, wave_values, w):
    if len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected tt.get_program_id with one result")
    value = values[op.results[0]]
    axis = int(op.attrs.get("axis", 0) or 0)
    if axis not in (0, 1, 2):
        raise ValueError(f"tlx_wave bridge cannot lower tt.get_program_id axis {axis}")
    bindings = state["program_id_bindings"]
    binding = bindings.get(axis)
    if binding is None:
        binding = builder.index_cast(builder.workgroup_id(axis), w.index_type())
        bindings[axis] = binding
    symbol = w.sym(f"tlx_program_id_{axis}")
    _set_wave_value(
        wave_values,
        value.value_id,
        "index_expr",
        builder.index_expr(symbol, {symbol: binding}),
    )


def _emit_make_range_op(op, values, wave_values, w):
    if len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected tt.make_range with one result")
    value = values[op.results[0]]
    if len(value.varying_dims) != 1:
        raise ValueError(
            "tlx_wave bridge expected tt.make_range to vary one dimension, "
            f"got dims={value.varying_dims}, type={value.type}"
        )
    dim = value.varying_dims[0]
    symbol = _dim_symbol(w, dim)
    start = int(op.attrs.get("start", 0) or 0)
    _set_wave_value(
        wave_values,
        value.value_id,
        "index_expr",
        _IndexExpr(symbol + start, {symbol: _DimBinding(dim)}),
    )


def _emit_index_binary_op(builder, op, values, wave_values, w):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError(f"tlx_wave bridge expected {op.name} with two operands")
    lhs = _require_lowered_value(
        wave_values, op.operands[0], "index_expr", op.name
    )
    rhs = _require_lowered_value(
        wave_values, op.operands[1], "index_expr", op.name
    )
    result = values[op.results[0]]
    lhs_symbol = w.sym(f"tlx_v{result.value_id}_lhs")
    rhs_symbol = w.sym(f"tlx_v{result.value_id}_rhs")
    expr = lhs_symbol + rhs_symbol if op.name == "arith.addi" else lhs_symbol * rhs_symbol
    if not _is_deferred_index(lhs) and not _is_deferred_index(rhs):
        lowered_lhs = _materialize_index_value(builder, lhs, {}, w)
        lowered_rhs = _materialize_index_value(builder, rhs, {}, w)
        _set_wave_value(
            wave_values,
            result.value_id,
            "index_expr",
            builder.index_expr(
                expr, {lhs_symbol: lowered_lhs, rhs_symbol: lowered_rhs}
            ),
        )
        return
    _set_wave_value(
        wave_values,
        result.value_id,
        "index_expr",
        _IndexExpr(expr, {lhs_symbol: lhs, rhs_symbol: rhs}),
    )


def _emit_cmp_op(builder, op, values, wave_values, w):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected arith.cmpi with two operands")
    predicate = _CMPI_PREDICATES.get(int(op.attrs.get("predicate")))
    if predicate is None:
        raise ValueError(
            "tlx_wave bridge cannot lower arith.cmpi predicate "
            f"{op.attrs.get('predicate')}"
        )
    lhs = _require_lowered_value(
        wave_values, op.operands[0], "index_expr", "arith.cmpi"
    )
    rhs = _require_lowered_value(
        wave_values, op.operands[1], "index_expr", "arith.cmpi"
    )
    result = values[op.results[0]]
    if not _is_deferred_index(lhs) and not _is_deferred_index(rhs):
        width = _tensor_lane_width(result, "arith.cmpi result")
        lhs_value = _materialize_index_value(builder, lhs, {}, w, force_width=width)
        rhs_value = _materialize_index_value(builder, rhs, {}, w, force_width=width)
        _set_wave_value(
            wave_values,
            result.value_id,
            "mask_expr",
            _wave_cmpi(builder, predicate, lhs_value, rhs_value, w),
        )
        return
    _set_wave_value(
        wave_values,
        result.value_id,
        "mask_expr",
        _MaskCompare(predicate, lhs, rhs),
    )


def _emit_mask_and_op(builder, op, values, wave_values, w):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected arith.andi with two operands")
    lhs = _require_lowered_value(
        wave_values, op.operands[0], "mask_expr", "arith.andi"
    )
    rhs = _require_lowered_value(
        wave_values, op.operands[1], "mask_expr", "arith.andi"
    )
    result = values[op.results[0]]
    if not _is_deferred_mask(lhs) and not _is_deferred_mask(rhs):
        width = None
        for source in (lhs, rhs):
            if not isinstance(source, _MaskConst):
                width = w.MaskType(source.type).width
                break
        if width is None:
            layout = _blocked_encoding_info(
                result.encoding_attr,
                result.encoding,
                "arith.andi result",
            )
            width = _product(layout.threads_per_warp)
        if isinstance(lhs, _MaskConst):
            lhs = _materialize_mask_value(builder, lhs, {}, w, width)
        if isinstance(rhs, _MaskConst):
            rhs = _materialize_mask_value(builder, rhs, {}, w, width)
        _set_wave_value(
            wave_values,
            result.value_id,
            "mask_expr",
            builder.select(lhs, rhs, _false_mask(builder, w, width)),
        )
        return
    _set_wave_value(
        wave_values,
        result.value_id,
        "mask_expr",
        _MaskAnd(lhs, rhs),
    )


def _emit_addptr_op(builder, op, values, wave_values, w):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected tt.addptr with two operands")
    base = _require_lowered_value(
        wave_values, op.operands[0], "pointer_expr", "tt.addptr"
    )
    offset = _require_lowered_value(
        wave_values, op.operands[1], "index_expr", "tt.addptr"
    )
    result = values[op.results[0]]
    if not _is_deferred_pointer(base) and not _is_deferred_index(offset):
        _set_wave_value(
            wave_values,
            result.value_id,
            "pointer_expr",
            builder.ptr_add(
                _materialize_pointer_value(builder, base, {}, w),
                _materialize_index_value(builder, offset, {}, w),
            ),
        )
        return
    _set_wave_value(
        wave_values,
        result.value_id,
        "pointer_expr",
        _PointerAdd(base, offset),
    )


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _delinearize_expr(w, linear, shape, order):
    if len(shape) != len(order):
        raise ValueError(
            "tlx_wave bridge blocked layout has mismatched shape/order lengths: "
            f"shape={shape}, order={order}"
        )
    coords = [w.sym_ctx.int_(0) for _ in shape]
    remainder = linear
    for dim in order:
        extent = int(shape[dim])
        coords[dim] = w.mod(remainder, extent)
        remainder = w.floor(remainder / extent)
    return tuple(coords)


def _delinearize_index(linear, shape, order):
    if len(shape) != len(order):
        raise ValueError(
            "tlx_wave bridge blocked layout has mismatched shape/order lengths: "
            f"shape={shape}, order={order}"
        )
    coords = [0 for _ in shape]
    remainder = int(linear)
    for dim in order:
        extent = int(shape[dim])
        coords[dim] = remainder % extent
        remainder //= extent
    return tuple(coords)


def _blocked_layout_static_coord(layout, thread, component):
    width = _product(layout.threads_per_warp)
    lane_coords = _delinearize_index(
        thread % width,
        layout.threads_per_warp,
        layout.order,
    )
    warp_coords = _delinearize_index(
        thread // width,
        layout.warps_per_cta,
        layout.order,
    )
    component_coords = _delinearize_index(
        component,
        layout.size_per_thread,
        layout.order,
    )
    return tuple(
        component_coords[dim]
        + layout.size_per_thread[dim]
        * (lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim])
        for dim in range(len(layout.size_per_thread))
    )


def _layout_thread_count(layout):
    return _product(layout.threads_per_warp) * _product(layout.warps_per_cta)


def _simd_convert_error(reason, source_plan, result_plan):
    raise ValueError(
        "tlx_wave bridge cannot lower ttg.convert_layout for SIMD tensor data: "
        f"{reason}; source encoding: {source_plan.encoding}; "
        f"result encoding: {result_plan.encoding}"
    )


def _simd_components_for_layout(source, source_plan, component_count, context):
    if source.kind == "simd":
        if component_count != 1:
            raise ValueError(
                f"tlx_wave bridge internal error while lowering {context}: "
                f"single SIMD value for {component_count} layout components; "
                f"encoding={source_plan.encoding}"
            )
        return (source.value,)
    if source.kind == "simd_tuple":
        if len(source.value) != component_count:
            raise ValueError(
                f"tlx_wave bridge internal error while lowering {context}: "
                f"SIMD tuple has {len(source.value)} components, expected "
                f"{component_count}; encoding={source_plan.encoding}"
            )
        return tuple(source.value)
    raise ValueError(
        f"tlx_wave bridge internal error while lowering {context}: "
        f"expected SIMD value, got {source.kind}"
    )


def _blocked_layout_component_permutation(source_plan, result_plan):
    source_layout = _blocked_tensor_layout_info(
        source_plan,
        "ttg.convert_layout source",
        "SIMD layout conversion",
    )
    result_layout = _blocked_tensor_layout_info(
        result_plan,
        "ttg.convert_layout result",
        "SIMD layout conversion",
    )
    source_width = _product(source_layout.threads_per_warp)
    result_width = _product(result_layout.threads_per_warp)
    if source_width != result_width:
        _simd_convert_error(
            f"SIMD widths differ ({source_width} -> {result_width})",
            source_plan,
            result_plan,
        )
    source_threads = _layout_thread_count(source_layout)
    result_threads = _layout_thread_count(result_layout)
    if source_threads != result_threads:
        _simd_convert_error(
            f"CTA thread counts differ ({source_threads} -> {result_threads})",
            source_plan,
            result_plan,
        )

    source_components = _product(source_layout.size_per_thread)
    result_components = _product(result_layout.size_per_thread)
    permutation = []
    for result_component in range(result_components):
        source_component = None
        for candidate in range(source_components):
            matches = True
            for thread in range(result_threads):
                result_coord = _blocked_layout_static_coord(
                    result_layout, thread, result_component
                )
                if any(
                    result_coord[dim] >= result_plan.shape[dim]
                    for dim in range(len(result_plan.shape))
                ):
                    continue
                source_coord = _blocked_layout_static_coord(
                    source_layout, thread, candidate
                )
                if source_coord != result_coord:
                    matches = False
                    break
            if matches:
                source_component = candidate
                break
        if source_component is None:
            _simd_convert_error(
                "conversion requires a lane-dependent remap, which is not "
                "representable by the current Wave SIMD tuple value model",
                source_plan,
                result_plan,
            )
        permutation.append(source_component)
    return tuple(permutation)


def _store_dim_bindings(builder, value_plan, wave_value, w, component=0):
    raw_value = _raw_wave_value(wave_value)
    if not value_plan.shape:
        raise ValueError(
            "tlx_wave bridge cannot lower tensor store without a ranked shape: "
            f"type={value_plan.type}, producer={value_plan.producer}"
        )
    if len(value_plan.shape) > 2:
        raise ValueError(
            "tlx_wave bridge store lowering currently supports rank <= 2, "
            f"got shape={value_plan.shape}"
        )
    frag = w.FragmentType(raw_value.type)
    layout = _blocked_encoding_info(
        value_plan.encoding_attr,
        value_plan.encoding,
        "fragment store physical value",
    )
    rank = len(value_plan.shape)
    if (
        len(layout.size_per_thread) != rank
        or len(layout.threads_per_warp) != rank
        or len(layout.warps_per_cta) != rank
    ):
        raise ValueError(
            "tlx_wave bridge blocked layout rank does not match stored tensor: "
            f"shape={value_plan.shape}, encoding={value_plan.encoding}"
        )
    if _product(layout.size_per_thread) != frag.registers:
        raise ValueError(
            "tlx_wave bridge cannot map fragment registers through blocked "
            "layout with incompatible per-thread element count: "
            f"registers={frag.registers}, sizePerThread={layout.size_per_thread}, "
            f"encoding={value_plan.encoding}"
        )
    threads_per_warp = _product(layout.threads_per_warp)
    if threads_per_warp != frag.wave_size:
        raise ValueError(
            "tlx_wave bridge cannot map fragment registers through blocked "
            "layout with incompatible threadsPerWarp: "
            f"fragment wave_size={frag.wave_size}, threadsPerWarp={layout.threads_per_warp}, "
            f"encoding={value_plan.encoding}"
        )
    if component < 0 or component >= frag.registers:
        raise ValueError(
            f"tlx_wave bridge fragment component {component} is out of range "
            f"for {frag.registers} registers"
        )

    thread = builder.workitem_id(axis=0, width=frag.wave_size)
    thread_sym = w.sym(f"tlx_store_{value_plan.value_id}_thread")
    thread_expr = thread_sym
    register_coords = _delinearize_expr(
        w, w.sym_ctx.int_(component), layout.size_per_thread, layout.order
    )
    lane_coords = _delinearize_expr(
        w, w.mod(thread_expr, threads_per_warp), layout.threads_per_warp, layout.order
    )
    warp_coords = _delinearize_expr(
        w,
        w.floor(thread_expr / threads_per_warp),
        layout.warps_per_cta,
        layout.order,
    )
    dim_bindings = {}
    for dim in range(rank):
        extent = value_plan.shape[dim]
        expr = register_coords[dim] + layout.size_per_thread[dim] * (
            lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim]
        )
        dim_bindings[_dim_symbol(w, dim)] = builder.index_expr(
            w.mod(expr, extent),
            {thread_sym: thread},
        )
    return dim_bindings, frag.wave_size


def _async_dim_bindings(builder, value_plan, w, component=0):
    return _blocked_layout_dim_bindings(
        builder,
        value_plan,
        w,
        "ttg.async_copy_global_to_local source",
        "tlx_async",
        "async copy address lowering",
        component=component,
    )


def _dot_operand_encoding_info(value, context):
    attr = value.encoding_attr
    if attr is None or not _attr_bool(attr, "is_dot_operand_encoding"):
        raise ValueError(
            f"tlx_wave bridge expected #ttg.dot_op encoding for {context}, "
            f"got {value.encoding}"
        )
    parent_attr = _attr_value(attr, "get_dot_operand_parent")
    return _DotOperandEncodingInfo(
        int(_attr_value(attr, "get_dot_operand_op_idx")),
        int(_attr_value(attr, "get_dot_operand_k_width")),
        _blocked_encoding_info(parent_attr, value.encoding, f"{context} parent"),
    )


def _same_dot_operand_encoding(lhs, rhs):
    return (
        lhs.op_idx == rhs.op_idx
        and lhs.k_width == rhs.k_width
        and _same_blocked_encoding(lhs.parent, rhs.parent)
    )


def _require_physical_dot_operand_fragment(values, wave_values, value, context):
    lowered = wave_values.get(value.value_id)
    if lowered is None:
        raise ValueError(
            "tlx_wave bridge cannot lower tt.dot: operand is not lowered from "
            f"ttg.local_load; {context} encoding: {value.encoding}"
        )
    if isinstance(lowered, _WaveValue) and lowered.kind != "fragment":
        raise ValueError(
            "tlx_wave bridge cannot lower tt.dot: operand lowered as "
            f"{lowered.kind}, expected fragment; {context} encoding: {value.encoding}"
        )
    physical_plan = _physical_value_plan(values, lowered, value.value_id)
    if physical_plan.element_type != value.element_type or physical_plan.shape != value.shape:
        raise ValueError(
            "tlx_wave bridge cannot use a fragment with an unlowered dot operand "
            "layout conversion that changes element type or shape; "
            f"physical type={physical_plan.type}, operand type={value.type}"
        )
    physical_info = _dot_operand_encoding_info(
        physical_plan, f"{context} physical fragment"
    )
    value_info = _dot_operand_encoding_info(value, context)
    if not _same_dot_operand_encoding(physical_info, value_info):
        raise ValueError(
            "tlx_wave bridge cannot use a fragment through an unlowered dot "
            f"operand layout conversion for {context}; "
            f"physical encoding: {physical_plan.encoding}; "
            f"operand encoding: {value.encoding}"
        )
    return _require_wave_value(
        wave_values,
        value.value_id,
        ("fragment",),
        context,
    )



def _source_pointee_type(kernel, address):
    if address.base_arg_index is None:
        raise ValueError(
            "tlx_wave bridge cannot lower async copy without a source pointer base"
        )
    if address.base_arg_index >= len(kernel.args):
        raise ValueError(
            f"tlx_wave bridge async copy references missing kernel arg {address.base_arg_index}"
        )
    pointee_type = kernel.args[address.base_arg_index].ttgir_type_obj.get_pointee_type()
    if pointee_type is None:
        raise ValueError(
            f"tlx_wave bridge async copy source %{address.base_arg_name} is not a pointer"
        )
    return pointee_type


def _dma_packet_bytes(address, memdesc, memdescs, lds_layout):
    if address.element_byte_width is None:
        return None
    if address.element_byte_width in (2, 4):
        packet_bytes = 4
    elif address.element_byte_width == 16:
        packet_bytes = 16
    else:
        return None
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr,
            memdesc.encoding,
            "ttg.async_copy_global_to_local destination",
        )
    except ValueError:
        return None
    if not (
        _is_identity_shared_layout(memdesc, shared)
        or (
            memdesc.element_type == "f16"
            and memdesc.shape == _GFX950_MMA_SHAPE
            and address.shape == memdesc.shape
            and shared == _GFX950_SHARED_LAYOUT
        )
    ):
        return None
    if not _memdesc_base_is_aligned(
        memdesc,
        memdescs,
        lds_layout,
        4,
        "ttg.async_copy_global_to_local destination",
    ):
        return None
    return packet_bytes


def _emit_async_source_ptr(builder, state, kernel, address, w, component=0):
    pointee_type = _source_pointee_type(kernel, address)
    element_type = _binding_type(pointee_type, w)
    if address.address_value_id is None:
        raise ValueError(
            "tlx_wave bridge cannot lower async copy without a source address value"
        )
    address_plan = state["values"][address.address_value_id]
    dim_bindings, width, active = _async_dim_bindings(
        builder, address_plan, w, component=component
    )
    # Async copy inputs are regular SSA values. They must have been produced by
    # earlier ordered op lowering; this path must not recursively lower a use-def
    # slice around the async op.
    source = _materialize_pointer_value(
        builder,
        _require_lowered_value(
            state["wave_values"],
            address.address_value_id,
            "pointer_expr",
            "ttg.async_copy_global_to_local source",
        ),
        dim_bindings,
        w,
    )
    return source, element_type, dim_bindings, width, active


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
    if address.other_value_id is not None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local with "
            "`other` values yet; emitting a masked copy would leave inactive "
            "LDS lanes undefined"
        )

    memdesc = memdescs[address.memdesc_value_id]
    if address.address_value_id is None:
        raise ValueError(
            "tlx_wave bridge cannot lower async copy without a source address value"
        )
    address_plan = state["values"][address.address_value_id]
    component_count = _blocked_layout_component_count(
        address_plan,
        "ttg.async_copy_global_to_local source",
        "async copy address lowering",
    )
    if address.element_type != memdesc.element_type:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local: "
            f"source element type {address.element_type} does not match "
            f"destination memdesc element type {memdesc.element_type}"
        )
    dma_bytes = (
        _dma_packet_bytes(address, memdesc, memdescs, lds_layout)
        if component_count == 1
        else None
    )
    stats.async_copies += 1
    if dma_bytes is not None:
        stats.dma_load_lds += 1
    else:
        stats.load_store_fallbacks += 1

    token = after_token
    for component in range(component_count):
        source, element_type, dim_bindings, width, active = _emit_async_source_ptr(
            builder, state, kernel, address, w, component=component
        )
        mask = active
        if address.mask_value_id is not None:
            user_mask = _materialize_mask_value(
                builder,
                _require_lowered_value(
                    state["wave_values"],
                    address.mask_value_id,
                    "mask_expr",
                    "ttg.async_copy_global_to_local mask",
                ),
                dim_bindings,
                w,
                width,
            )
            mask = _wave_mask_and(builder, mask, user_mask, w, width)

        def emit_copy(copy_after):
            if dma_bytes is not None:
                destination = _emit_memdesc_base_ptr(
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
                return builder.dma_load_lds(
                    source, destination, after=copy_after, bytes=dma_bytes
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
            values, load_token = builder.load(
                source, w.simd_type(element_type, width), after=copy_after
            )
            return builder.store(values, destination, after=load_token)

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


def _join_tokens(builder, tokens, stats):
    if not tokens:
        return builder.token()
    stats.joins += 1
    return builder.join(*tokens)


def _initial_lowering_state(builder, plan, w):
    values = _values_by_id(plan)
    state = {
        "values": values,
        "memdescs": _memdescs_by_id(plan),
        "op_by_result": _op_by_result_id(plan),
        "wave_values": {},
        "program_id_bindings": {},
        "pending_copy_tokens": [],
        "committed_groups": [],
        "last_order_token": None,
    }
    _init_argument_wave_values(builder, values, state["wave_values"], w)
    return state


def _emit_generic_value_op(builder, state, op, w):
    values = state["values"]
    wave_values = state["wave_values"]
    if op.name == "arith.constant":
        _emit_constant_op(builder, op, values, wave_values, w)
    elif op.name == "tt.get_program_id":
        _emit_program_id_op(builder, state, op, values, wave_values, w)
    elif op.name == "tt.make_range":
        _emit_make_range_op(op, values, wave_values, w)
    elif op.name in {"tt.broadcast", "tt.splat", "tt.expand_dims"}:
        _forward_lowered_value(op, values, wave_values)
    elif op.name in {"arith.addi", "arith.muli"}:
        _emit_index_binary_op(builder, op, values, wave_values, w)
    elif op.name == "arith.cmpi":
        _emit_cmp_op(builder, op, values, wave_values, w)
    elif op.name == "arith.andi":
        _emit_mask_and_op(builder, op, values, wave_values, w)
    elif op.name == "tt.addptr":
        _emit_addptr_op(builder, op, values, wave_values, w)
    elif op.name == "ttg.convert_layout":
        _forward_lowered_value(op, values, wave_values)
    else:
        return False
    return True


_PLANNING_ONLY_OPS = {
    "tt.return",
    "ttg.local_alloc",
    "ttg.memdesc_index",
    "ttg.memdesc_subslice",
    "ttg.memdesc_reinterpret",
    "ttg.memdesc_reshape",
    "ttg.memdesc_trans",
}


def _fragment_type_for_dot_operand(info, w):
    if info.op_idx not in (0, 1):
        raise ValueError(
            f"tlx_wave bridge supports #ttg.dot_op opIdx 0/1, got {info.op_idx}"
        )
    return w.fragment_type(
        info.op_idx,
        w.f16(),
        rows=_GFX950_MMA_M,
        columns=_GFX950_MMA_N,
        wave_size=_GFX950_MMA_WAVE,
        registers=_GFX950_MMA_REGS,
    )


def _acc_fragment_type(w):
    return w.fragment_type(
        2,
        w.f32(),
        rows=_GFX950_MMA_M,
        columns=_GFX950_MMA_N,
        wave_size=_GFX950_MMA_WAVE,
        registers=_GFX950_MMA_REGS,
    )


def _unsupported_local_load(value, reason, memdesc=None):
    message = (
        "tlx_wave bridge cannot lower ttg.local_load for WaveAMD MMA: "
        f"{reason}. original TTGIR encoding: {value.encoding}"
    )
    if memdesc is not None:
        message += f"; memdesc encoding: {memdesc.encoding}"
    raise ValueError(message)


def _validate_supported_dot_local_load_layout(value, memdesc, info):
    if not _same_blocked_encoding(info.parent, _GFX950_DOT_PARENT_LAYOUT):
        _unsupported_local_load(
            value,
            "current flat gfx950 fragment loader supports only dot operand "
            "parent layout sizePerThread=(2, 2), threadsPerWarp=(4, 16), "
            "warpsPerCTA=(4, 1), order=(1, 0); "
            f"got sizePerThread={info.parent.size_per_thread}, "
            f"threadsPerWarp={info.parent.threads_per_warp}, "
            f"warpsPerCTA={info.parent.warps_per_cta}, order={info.parent.order}",
            memdesc,
        )
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, "ttg.local_load memdesc"
        )
    except ValueError as exc:
        _unsupported_local_load(value, str(exc), memdesc)
    if shared != _GFX950_SHARED_LAYOUT:
        _unsupported_local_load(
            value,
            "current flat gfx950 fragment loader supports only "
            "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, "
            "order = [1, 0]}> for f16 32x32 dot operands; "
            f"got vec={shared.vec}, perPhase={shared.per_phase}, "
            f"maxPhase={shared.max_phase}, order={shared.order}",
            memdesc,
        )


def _validate_dot_local_load(value, memdesc, info):
    if value.element_type != "f16" or value.element_byte_width != 2:
        _unsupported_local_load(
            value, f"expected f16 dot operand, got {value.element_type}"
        )
    if value.shape != _GFX950_MMA_SHAPE:
        _unsupported_local_load(
            value, f"expected static 32x32 operand shape, got {value.shape}"
        )
    if info.op_idx not in (0, 1):
        _unsupported_local_load(value, f"expected opIdx 0/1, got {info.op_idx}")
    if info.k_width not in (0, 4, 32):
        _unsupported_local_load(
            value, f"expected kWidth 0, 4, or 32 for gfx950 f16 MFMA, got {info.k_width}"
        )
    if memdesc.element_type != "f16" or memdesc.element_byte_width != 2:
        _unsupported_local_load(
            value,
            f"expected f16 shared memdesc source, got {memdesc.element_type}",
            memdesc,
        )
    if memdesc.shape != _GFX950_MMA_SHAPE:
        _unsupported_local_load(
            value,
            f"expected 32x32 shared memdesc source, got {memdesc.shape}",
            memdesc,
        )
    _validate_supported_dot_local_load_layout(value, memdesc, info)


def _emit_memdesc_i32_ptr(builder, value, memdesc, memdescs, lds_layout, state, w):
    try:
        return _emit_memdesc_base_ptr(
            builder,
            memdesc,
            memdescs,
            lds_layout,
            state,
            w.i32(),
            4,
            w,
            "ttg.local_load",
        )
    except ValueError as exc:
        _unsupported_local_load(value, str(exc), memdesc)


def _emit_local_load_fragment(
    builder,
    address,
    value,
    memdesc,
    memdescs,
    info,
    lds_layout,
    state,
    after_token,
    w,
    stats,
):
    _validate_dot_local_load(value, memdesc, info)

    lane = builder.lane_id(width=_GFX950_MMA_WAVE)
    lane_sym = w.sym(f"tlx_local_load_{value.value_id}_lane")
    lane_offset = builder.index_expr(
        lane_sym * _GFX950_MMA_REGS,
        {lane_sym: lane},
    )
    base = _emit_memdesc_i32_ptr(
        builder, value, memdesc, memdescs, lds_layout, state, w
    )
    ptr = builder.ptr_add(
        base,
        lane_offset,
        w.simd_ptr_type(w.i32(), w.shared_address_space(), _GFX950_MMA_WAVE),
    )
    fragment, token = builder.fragment_load(
        ptr, _fragment_type_for_dot_operand(info, w), after=after_token
    )
    stats.local_loads += 1
    stats.fragment_packs += 1
    return fragment, token


def _emit_accumulator_fragment(builder, value, wave_values, w, stats, values=None):
    if value.value_id in wave_values:
        lowered = wave_values[value.value_id]
        if values is not None and isinstance(lowered, _WaveValue):
            physical_plan = _physical_value_plan(values, lowered, value.value_id)
            if not _same_layout_encoding(physical_plan, value):
                raise ValueError(
                    "tlx_wave bridge cannot use a fragment with an unlowered "
                    "layout conversion as a tt.dot accumulator; "
                    f"physical encoding: {physical_plan.encoding}; "
                    f"accumulator encoding: {value.encoding}"
                )
        return _require_wave_value(
            wave_values,
            value.value_id,
            ("fragment",),
            "tt.dot accumulator",
        )
    if (
        value.producer == "arith.constant"
        and value.type_kind == "tensor"
        and value.element_type == "f32"
        and value.shape == _GFX950_MMA_SHAPE
        and value.const_value in (0, 0.0)
    ):
        zero = builder.constant(w.i32(), 0)
        fragment = builder.fragment_fill(zero, _acc_fragment_type(w))
        _set_wave_value(wave_values, value.value_id, "fragment", fragment)
        stats.fragment_fills += 1
        return fragment
    raise ValueError(
        "tlx_wave bridge supports tt.dot accumulators only from prior tt.dot "
        "results or zero f32 tensor constants; "
        f"got producer={value.producer}, type={value.type}, encoding={value.encoding}"
    )


def _validate_dot_op(operands, acc, result):
    role_values = {}
    role_infos = {}
    for value in operands:
        info = _dot_operand_encoding_info(value, "tt.dot operand")
        if info.op_idx not in (0, 1):
            raise ValueError(
                "tlx_wave bridge expected tt.dot operands with #ttg.dot_op "
                f"opIdx 0/1, got {info.op_idx}; operand encoding: {value.encoding}"
            )
        if info.op_idx in role_values:
            raise ValueError(
                "tlx_wave bridge found duplicate #ttg.dot_op opIdx for tt.dot; "
                f"operand encoding: {value.encoding}"
            )
        role_values[info.op_idx] = value
        role_infos[info.op_idx] = info
    if set(role_values) != {0, 1}:
        encodings = ", ".join(value.encoding or "<none>" for value in operands)
        raise ValueError(
            "tlx_wave bridge expected tt.dot operands with #ttg.dot_op opIdx "
            f"0 and 1; operand encodings: {encodings}"
        )

    if any(value.element_type != "f16" for value in role_values.values()):
        encodings = ", ".join(value.encoding or "<none>" for value in operands)
        raise ValueError(
            "tlx_wave bridge supports only f16 x f16 tt.dot operands for "
            f"{_GFX950_F16_MMA_KIND}; operand encodings: {encodings}"
        )
    if any(value.shape != _GFX950_MMA_SHAPE for value in role_values.values()):
        shapes = ", ".join(str(value.shape) for value in operands)
        encodings = ", ".join(value.encoding or "<none>" for value in operands)
        raise ValueError(
            "tlx_wave bridge supports only static 32x32 tt.dot operands; "
            f"shapes={shapes}; operand encodings: {encodings}"
        )
    if not _same_blocked_encoding(role_infos[0].parent, role_infos[1].parent):
        raise ValueError(
            "tlx_wave bridge expected matching tt.dot operand parent layouts; "
            f"role0 encoding: {role_values[0].encoding}; "
            f"role1 encoding: {role_values[1].encoding}"
        )

    result_layout = _blocked_encoding_info(
        result.encoding_attr, result.encoding, "tt.dot result"
    )
    if not _same_blocked_encoding(role_infos[0].parent, result_layout):
        raise ValueError(
            "tlx_wave bridge expected tt.dot result layout to match dot operand parent; "
            f"result encoding: {result.encoding}; "
            f"operand encoding: {role_values[0].encoding}"
        )
    if result.element_type != "f32" or result.shape != _GFX950_MMA_SHAPE:
        raise ValueError(
            "tlx_wave bridge supports only f32 32x32 tt.dot results; "
            f"got type={result.type}, encoding={result.encoding}"
        )
    if acc.type_kind == "tensor":
        acc_layout = _blocked_encoding_info(
            acc.encoding_attr, acc.encoding, "tt.dot accumulator"
        )
        if not _same_blocked_encoding(result_layout, acc_layout):
            raise ValueError(
                "tlx_wave bridge expected tt.dot accumulator layout to match result; "
                f"accumulator encoding: {acc.encoding}; result encoding: {result.encoding}"
            )
    return role_values


def _emit_dot_op(builder, op, values, wave_values, w, stats):
    if len(op.operands) < 3 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected tt.dot with 3 operands and 1 result")
    operands = tuple(values[operand] for operand in op.operands[:2])
    acc = values[op.operands[2]]
    result = values[op.results[0]]
    role_values = _validate_dot_op(operands, acc, result)
    if (
        role_values[0].value_id not in wave_values
        or role_values[1].value_id not in wave_values
    ):
        raise ValueError(
            "tlx_wave bridge supports tt.dot only when both operands are "
            "lowered from ttg.local_load; "
            f"role0 encoding: {role_values[0].encoding}; "
            f"role1 encoding: {role_values[1].encoding}"
        )
    acc_fragment = _emit_accumulator_fragment(
        builder, acc, wave_values, w, stats, values=values
    )
    lhs_fragment = _require_physical_dot_operand_fragment(
        values, wave_values, role_values[0], "tt.dot operand role0"
    )
    rhs_fragment = _require_physical_dot_operand_fragment(
        values, wave_values, role_values[1], "tt.dot operand role1"
    )
    dot = builder.mma(
        _GFX950_F16_MMA_KIND,
        lhs_fragment,
        rhs_fragment,
        acc_fragment,
    )
    _set_wave_value(wave_values, result.value_id, "fragment", dot)
    stats.mmas += 1


def _forward_nonfragment_value(source):
    if source.kind == "fragment":
        raise ValueError("internal error: fragment value reached non-fragment forward")
    return source


def _shift_forwarded_value(source, axis):
    if source.kind == "index_expr":
        return _WaveValue(source.kind, _shift_index_dims(source.value, axis))
    if source.kind == "mask_expr":
        return _WaveValue(source.kind, _shift_mask_dims(source.value, axis))
    if source.kind == "pointer_expr":
        return _WaveValue(source.kind, _shift_pointer_dims(source.value, axis))
    raise ValueError(
        "tlx_wave bridge cannot lower tt.expand_dims for Wave value kind "
        f"{source.kind}"
    )


def _convert_simd_layout(source, source_plan, result_plan):
    source_count = _blocked_layout_component_count(
        source_plan,
        "ttg.convert_layout source",
        "SIMD layout conversion",
    )
    result_count = _blocked_layout_component_count(
        result_plan,
        "ttg.convert_layout result",
        "SIMD layout conversion",
    )
    source_components = _simd_components_for_layout(
        source, source_plan, source_count, "ttg.convert_layout"
    )
    permutation = _blocked_layout_component_permutation(source_plan, result_plan)
    if len(permutation) != result_count:
        raise ValueError(
            "tlx_wave bridge internal error: convert_layout permutation length "
            f"{len(permutation)} does not match result components {result_count}"
        )
    result_components = tuple(source_components[index] for index in permutation)
    if result_count == 1:
        return _WaveValue("simd", result_components[0])
    return _WaveValue("simd_tuple", result_components)


def _forward_lowered_value(op, values, wave_values):
    if len(op.operands) != 1 or len(op.results) != 1:
        raise ValueError(f"tlx_wave bridge expected {op.name} with one operand/result")
    source_id = op.operands[0]
    result_id = op.results[0]
    source = wave_values.get(source_id)
    if source is None:
        return
    source_plan = values[source_id]
    result_plan = values[result_id]
    if op.name in {"tt.broadcast", "tt.splat"}:
        if isinstance(source, _WaveValue) and source.kind != "fragment":
            wave_values[result_id] = _forward_nonfragment_value(source)
            return
        raise ValueError(
            f"tlx_wave bridge cannot forward {op.name} for lowered "
            f"{source.kind if isinstance(source, _WaveValue) else type(source).__name__} "
            "value"
        )

    if op.name == "tt.expand_dims":
        if not isinstance(source, _WaveValue):
            raise ValueError("tlx_wave bridge internal error: untyped Wave value")
        axis = int(op.attrs.get("axis", 0) or 0)
        wave_values[result_id] = _shift_forwarded_value(source, axis)
        return

    if source_plan.element_type != result_plan.element_type:
        raise ValueError(
            "tlx_wave bridge cannot forward layout conversion with element type "
            f"change: {source_plan.type} -> {result_plan.type}"
        )
    if source_plan.shape != result_plan.shape:
        raise ValueError(
            "tlx_wave bridge cannot forward layout conversion with shape change: "
            f"{source_plan.shape} -> {result_plan.shape}"
        )
    if not isinstance(source, _WaveValue):
        wave_values[result_id] = source
        return
    if source.kind != "fragment":
        if source.kind in {"simd", "simd_tuple"} and not _same_layout_encoding(
            source_plan, result_plan
        ):
            wave_values[result_id] = _convert_simd_layout(
                source, source_plan, result_plan
            )
            return
        wave_values[result_id] = _forward_nonfragment_value(source)
        return
    if _same_layout_encoding(source_plan, result_plan):
        wave_values[result_id] = source
        return
    wave_values[result_id] = _WaveValue(
        source.kind,
        source.value,
        source.physical_value_id if source.physical_value_id is not None else source_id,
    )


def _validate_fragment_store_value(value_plan, physical_plan):
    if value_plan.element_type != "f32" or value_plan.shape != _GFX950_MMA_SHAPE:
        raise ValueError(
            "tlx_wave bridge supports fragment tt.store only for f32 32x32 "
            f"values, got type={value_plan.type}, encoding={value_plan.encoding}"
        )
    if (
        physical_plan.element_type != value_plan.element_type
        or physical_plan.shape != value_plan.shape
    ):
        raise ValueError(
            "tlx_wave bridge cannot store fragment through a layout conversion "
            "that changes element type or shape: "
            f"physical type={physical_plan.type}, store type={value_plan.type}"
        )
    _blocked_encoding_info(
        physical_plan.encoding_attr,
        physical_plan.encoding,
        "fragment store physical value",
    )


def _extract_fragment_component(regs, component, width, w):
    wave = getattr(w, "wave", None)
    if wave is None:
        raise RuntimeError(
            "tlx_wave bridge requires mlir.dialects.wave_dsl to expose the "
            "generated wave dialect module"
        )
    return wave.ExtractOp(w.simd_type(w.i32(), width), regs, component).result


def _emit_component_store(builder, value, ptr, mask, after_token, w):
    if mask is None:
        return builder.store(value, ptr, after=after_token)
    if after_token is None:
        after_token = builder.token()
    with builder.where(mask, [w.mem_token_type()]) as where_op:
        token = builder.store(value, ptr, after=after_token)
        builder.yield_([token])
    return where_op.results[0]


def _emit_fragment_store(
    builder,
    state,
    lowered,
    physical_plan,
    ptr_id,
    mask_id,
    after_token,
    w,
):
    fragment = lowered.value
    regs = builder.fragment_unpack(fragment)
    frag = w.FragmentType(fragment.type)
    token = after_token
    for component in range(frag.registers):
        dim_bindings, width = _store_dim_bindings(
            builder, physical_plan, lowered, w, component=component
        )
        ptr = _materialize_pointer_value(
            builder,
            _require_lowered_value(
                state["wave_values"],
                ptr_id,
                "pointer_expr",
                "tt.store pointer",
            ),
            dim_bindings,
            w,
        )
        mask = (
            _materialize_mask_value(
                builder,
                _require_lowered_value(
                    state["wave_values"],
                    mask_id,
                    "mask_expr",
                    "tt.store mask",
                ),
                dim_bindings,
                w,
                width,
            )
            if mask_id is not None
            else None
        )
        value = _extract_fragment_component(regs, component, width, w)
        token = _emit_component_store(builder, value, ptr, mask, token, w)
    return token


def _emit_global_load_op(builder, op, state, w):
    values = state["values"]
    wave_values = state["wave_values"]
    if len(op.operands) < 1 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected tt.load with pointer and result")
    if len(op.operands) > 3:
        raise ValueError(
            "tlx_wave bridge cannot lower tt.load with more than pointer, "
            "mask, and `other` operands yet"
        )

    result_id = op.results[0]
    result_plan = values[result_id]
    component_count = _blocked_layout_component_count(
        result_plan, "tt.load result", "generic tensor lowering"
    )
    components = []
    token = state["last_order_token"]
    for component in range(component_count):
        dim_bindings, width, active = _blocked_tensor_dim_bindings(
            builder, result_plan, w, "tt.load result", component=component
        )
        ptr = _materialize_pointer_value(
            builder,
            _require_lowered_value(
                wave_values,
                op.operands[0],
                "pointer_expr",
                "tt.load pointer",
            ),
            dim_bindings,
            w,
        )
        mask = active
        if len(op.operands) > 1:
            user_mask = _materialize_mask_value(
                builder,
                _require_lowered_value(
                    wave_values,
                    op.operands[1],
                    "mask_expr",
                    "tt.load mask",
                ),
                dim_bindings,
                w,
                width,
            )
            mask = _wave_mask_and(builder, mask, user_mask, w, width)

        result_type = _simd_type_for_value(result_plan, width, w, "tt.load result")
        fallback = (
            _load_other_value(
                builder,
                state,
                op.operands[2],
                result_plan,
                width,
                w,
                "tt.load other",
                component=component,
            )
            if len(op.operands) > 2
            else _zero_simd_value(
                builder, result_plan, width, w, "tt.load inactive lanes"
            )
        )
        loaded, token = _emit_masked_load(
            builder,
            ptr,
            result_type,
            mask,
            fallback,
            token,
            w,
        )
        components.append(loaded)
    state["last_order_token"] = token
    if component_count == 1:
        _set_wave_value(wave_values, result_id, "simd", components[0])
    else:
        _set_wave_value(wave_values, result_id, "simd_tuple", tuple(components))


def _emit_generic_local_store_op(builder, op, state, memdescs, lds_layout, w, stats):
    values = state["values"]
    wave_values = state["wave_values"]
    if len(op.operands) < 2:
        raise ValueError(
            "tlx_wave bridge expected ttg.local_store with value and memdesc"
        )

    value_id = op.operands[0]
    memdesc_id = op.operands[1]
    if memdesc_id not in memdescs:
        raise ValueError(
            f"tlx_wave bridge cannot lower ttg.local_store: unknown memdesc "
            f"{memdesc_id}"
        )
    value_plan = values[value_id]
    memdesc = memdescs[memdesc_id]
    _validate_generic_local_tensor(value_plan, memdesc, "ttg.local_store")
    lowered = wave_values.get(value_id)
    if lowered is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.local_store: stored value "
            f"{value_id} from {value_plan.producer} has not been lowered"
        )
    component_count = _blocked_layout_component_count(
        value_plan, "ttg.local_store value", "generic tensor lowering"
    )
    token = state["last_order_token"]
    for component in range(component_count):
        dim_bindings, width, active = _blocked_tensor_dim_bindings(
            builder, value_plan, w, "ttg.local_store value", component=component
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
        data = _materialize_tensor_data(
            builder,
            lowered,
            value_plan,
            width,
            w,
            "ttg.local_store",
            component=component,
        )
        token = _emit_component_store(
            builder,
            data,
            ptr,
            active,
            token,
            w,
        )
    state["last_order_token"] = builder.barrier(token)
    stats.barriers += 1


def _emit_generic_local_load(
    builder,
    address,
    value,
    memdesc,
    memdescs,
    lds_layout,
    state,
    after_token,
    w,
    stats,
):
    _validate_generic_local_tensor(value, memdesc, "ttg.local_load")
    component_count = _blocked_layout_component_count(
        value, "ttg.local_load result", "generic tensor lowering"
    )
    components = []
    token = after_token
    for component in range(component_count):
        dim_bindings, width, active = _blocked_tensor_dim_bindings(
            builder, value, w, "ttg.local_load result", component=component
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
            "ttg.local_load",
        )
        result_type = _simd_type_for_value(value, width, w, "ttg.local_load result")
        fallback = _zero_simd_value(
            builder, value, width, w, "ttg.local_load inactive"
        )
        loaded, token = _emit_masked_load(
            builder,
            ptr,
            result_type,
            active,
            fallback,
            token,
            w,
        )
        components.append(loaded)
    stats.local_loads += 1
    if component_count == 1:
        return components[0], token
    return tuple(components), token


def _emit_store_op(builder, op, state, w):
    values = state["values"]
    wave_values = state["wave_values"]
    if len(op.operands) < 2:
        raise ValueError("tlx_wave bridge expected tt.store with pointer and value")
    ptr_id = op.operands[0]
    value_id = op.operands[1]
    mask_id = op.operands[2] if len(op.operands) > 2 else None
    value_plan = values[value_id]
    lowered = wave_values.get(value_id)
    if lowered is None:
        raise ValueError(
            "tlx_wave bridge cannot lower tt.store: stored value "
            f"{value_id} from {value_plan.producer} is not lowered"
        )
    if not isinstance(lowered, _WaveValue):
        raise ValueError("tlx_wave bridge internal error: untyped Wave value")

    if lowered.kind == "fragment":
        physical_plan = _physical_value_plan(values, lowered, value_id)
        _validate_fragment_store_value(value_plan, physical_plan)
        state["last_order_token"] = _emit_fragment_store(
            builder,
            state,
            lowered,
            physical_plan,
            ptr_id,
            mask_id,
            state["last_order_token"],
            w,
        )
        return

    if lowered.kind in {"simd", "simd_tuple"}:
        component_count = _blocked_layout_component_count(
            value_plan, "tt.store value", "generic tensor lowering"
        )
        token = state["last_order_token"]
        for component in range(component_count):
            dim_bindings, width, active = _blocked_tensor_dim_bindings(
                builder, value_plan, w, "tt.store value", component=component
            )
            ptr = _materialize_pointer_value(
                builder,
                _require_lowered_value(
                    wave_values,
                    ptr_id,
                    "pointer_expr",
                    "tt.store pointer",
                ),
                dim_bindings,
                w,
            )
            mask = active
            if mask_id is not None:
                user_mask = _materialize_mask_value(
                    builder,
                    _require_lowered_value(
                        wave_values,
                        mask_id,
                        "mask_expr",
                        "tt.store mask",
                    ),
                    dim_bindings,
                    w,
                    width,
                )
                mask = _wave_mask_and(builder, mask, user_mask, w, width)
            value = _materialize_tensor_data(
                builder,
                lowered,
                value_plan,
                width,
                w,
                "tt.store",
                component=component,
            )
            token = _emit_component_store(
                builder,
                value,
                ptr,
                mask,
                token,
                w,
            )
        state["last_order_token"] = token
        return

    raise ValueError(
        "tlx_wave bridge cannot lower tt.store for Wave value kind "
        f"{lowered.kind}; stored TTGIR type={value_plan.type}"
    )


def _emit_ordered_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats):
    state = _initial_lowering_state(builder, plan, w)
    values = state["values"]
    memdescs = _memdescs_by_id(plan)
    address_by_token = _async_address_by_token(plan)
    local_loads = _local_load_address_by_result(plan)
    wave_values = state["wave_values"]

    for op in plan.ops:
        if _emit_generic_value_op(builder, state, op, w):
            continue
        if op.name == "tt.load":
            _emit_global_load_op(builder, op, state, w)
        elif op.name == "ttg.async_copy_global_to_local":
            token_id = op.results[0] if op.results else None
            address = address_by_token.get(token_id)
            if address is None:
                raise ValueError("tlx_wave bridge could not match async copy token")
            if state["last_order_token"] is None:
                state["last_order_token"] = builder.token()
            state["last_order_token"] = _emit_async_copy(
                builder,
                state,
                kernel,
                address,
                lds_layout,
                state["last_order_token"],
                w,
                stats,
            )
            if token_id is not None:
                _set_wave_value(
                    wave_values, token_id, "token", state["last_order_token"]
                )
            state["pending_copy_tokens"].append(state["last_order_token"])
        elif op.name == "ttg.async_commit_group":
            group = _join_tokens(
                builder, tuple(state["pending_copy_tokens"]), stats
            )
            state["pending_copy_tokens"].clear()
            state["committed_groups"].append(group)
            state["last_order_token"] = group
            for result_id in op.results:
                _set_wave_value(wave_values, result_id, "token", group)
            stats.commit_groups += 1
        elif op.name == "ttg.async_wait":
            keep_groups = int(op.attrs.get("num", 0) or 0)
            wait_count = max(0, len(state["committed_groups"]) - keep_groups)
            if wait_count:
                waited_groups = tuple(state["committed_groups"][:wait_count])
                wait_token = _join_tokens(builder, waited_groups, stats)
                builder.wait(wait_token)
                stats.waits += 1
                state["committed_groups"] = state["committed_groups"][wait_count:]
                state["last_order_token"] = builder.barrier(wait_token)
                stats.barriers += 1
            ready_token = (
                state["last_order_token"]
                if state["last_order_token"] is not None
                else builder.token()
            )
            for result_id in op.results:
                _set_wave_value(wave_values, result_id, "token", ready_token)
        elif op.name == "ttg.local_store":
            _emit_generic_local_store_op(
                builder, op, state, memdescs, lds_layout, w, stats
            )
        elif op.name == "ttg.local_load":
            result_id = op.results[0] if op.results else None
            if result_id is None:
                raise ValueError("tlx_wave bridge expected ttg.local_load result")
            value = values[result_id]
            address = local_loads.get(result_id)
            if address is None or address.memdesc_value_id is None:
                _unsupported_local_load(value, "missing shared memdesc source")
            memdesc = memdescs[address.memdesc_value_id]
            after = (
                _require_wave_value(
                    wave_values,
                    address.token_value_id,
                    ("token",),
                    "ttg.local_load token",
                )
                if address.token_value_id is not None
                else state["last_order_token"]
            )
            if (
                value.encoding_attr is not None
                and _attr_bool(value.encoding_attr, "is_dot_operand_encoding")
            ):
                info = _dot_operand_encoding_info(value, "ttg.local_load result")
                fragment, state["last_order_token"] = _emit_local_load_fragment(
                    builder,
                    address,
                    value,
                    memdesc,
                    memdescs,
                    info,
                    lds_layout,
                    state,
                    after,
                    w,
                    stats,
                )
                _set_wave_value(wave_values, result_id, "fragment", fragment)
            else:
                loaded, state["last_order_token"] = _emit_generic_local_load(
                    builder,
                    address,
                    value,
                    memdesc,
                    memdescs,
                    lds_layout,
                    state,
                    after,
                    w,
                    stats,
                )
                if isinstance(loaded, tuple):
                    _set_wave_value(wave_values, result_id, "simd_tuple", loaded)
                else:
                    _set_wave_value(wave_values, result_id, "simd", loaded)
        elif op.name == "tt.dot":
            _emit_dot_op(builder, op, values, wave_values, w, stats)
        elif op.name == "tt.store":
            _emit_store_op(builder, op, state, w)
        elif op.name in _PLANNING_ONLY_OPS:
            continue
        else:
            region_note = (
                " with nested regions"
                if getattr(op, "get_num_regions", lambda: 0)()
                else ""
            )
            raise ValueError(
                "tlx_wave bridge cannot lower unsupported TTGIR op in unified "
                f"body lowering: {op.name}{region_note}"
            )


def _emit_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats):
    _emit_ordered_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats)



def _repo_root():
    return Path(__file__).resolve().parents[3]


def _cmake_build_dirs():
    try:
        from build_helpers import get_cmake_dir

        yield Path(get_cmake_dir())
    except Exception:
        pass

    build_root = _repo_root() / "build"
    if build_root.is_dir():
        yield from sorted(path for path in build_root.glob("cmake.*") if path.is_dir())


def _wave_build_dirs():
    for build_dir in _cmake_build_dirs():
        yield build_dir
        yield build_dir / "third_party" / "tlx_wave" / "wave"
        yield build_dir / "third_party" / "wave"
    yield _repo_root() / "third_party" / "wave" / "build"


def _candidate_wave_python_paths():
    override = os.environ.get("TRITON_WAVE_PYTHONPATH")
    if override:
        for entry in override.split(os.pathsep):
            if entry:
                yield Path(entry)

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "python_packages" / "wave_mlir"


def _candidate_wave_tool_paths(tool_name, override=None):
    if override:
        yield Path(override)

    tools_dir = os.environ.get("TRITON_WAVE_TOOLS_DIR")
    if tools_dir:
        yield Path(tools_dir) / tool_name

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "bin" / tool_name


def _candidate_wave_opt_paths():
    yield from _candidate_wave_tool_paths(
        "wave-opt",
        override=os.environ.get("TRITON_WAVE_OPT"),
    )


def _existing_paths(candidates):
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            yield path


def _load_wave_dsl():
    for path in _existing_paths(_candidate_wave_python_paths()):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        from mlir.dialects import wave_dsl as w
    except Exception as exc:
        candidates = "\n  ".join(str(path) for path in _candidate_wave_python_paths())
        raise RuntimeError(
            "tlx_wave requires Wave MLIR Python bindings from the third_party/wave submodule build. "
            "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave and WAVE_ENABLE_PYTHON_BINDINGS=ON; "
            "the underlying MLIR install must have MLIR_ENABLE_BINDINGS_PYTHON=ON. "
            f"Unable to import mlir.dialects.wave_dsl: {type(exc).__name__}: {exc}. "
            f"Checked Wave Python package candidates:\n  {candidates}"
        ) from exc
    return w


def _wave_tool(tool_name, override_env=None):
    override = os.environ.get(override_env) if override_env else None
    for path in _existing_paths(_candidate_wave_tool_paths(tool_name, override)):
        if os.access(path, os.X_OK):
            return str(path)
    candidates = "\n  ".join(
        str(path) for path in _candidate_wave_tool_paths(tool_name, override)
    )
    raise RuntimeError(
        f"tlx_wave requires {tool_name} from the third_party/wave submodule build. "
        "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave so the Wave tools are built. "
        f"Checked {tool_name} candidates:\n  {candidates}"
    )


def _wave_opt():
    return _wave_tool("wave-opt", override_env="TRITON_WAVE_OPT")


def _binding_type(ttgir_type, w):
    pointee_type = ttgir_type.get_pointee_type()
    if pointee_type is not None:
        return w.ptr_type(_binding_type(pointee_type, w))
    if ttgir_type.is_integer(1):
        return w.i1()
    if ttgir_type.is_integer(8):
        return w.i8()
    if ttgir_type.is_integer(16):
        return w.IntegerType.get_signless(16)
    if ttgir_type.is_integer(32):
        return w.i32()
    if ttgir_type.is_integer(64):
        return w.i64()
    if ttgir_type.is_index():
        return w.index_type()
    if ttgir_type.is_fp16():
        return w.f16()
    if ttgir_type.is_bf16():
        return w.bf16()
    if ttgir_type.is_fp32():
        return w.f32()
    raise ValueError(
        f"tlx_wave bridge does not yet support Wave binding type {ttgir_type}"
    )


def _binding_i32_attr(w, value):
    return w.IntegerAttr.get(w.i32(), int(value))


def _binding_bool_attr(w, value):
    return w.IntegerAttr.get(w.i1(), int(bool(value)))


def _binding_attrs(w, attrs):
    return {
        "tlx_wave.bridge.stage": w.StringAttr.get(attrs["bridge_stage"]),
        "tlx_wave.source_op": w.StringAttr.get("tt.func"),
        "tlx_wave.num_pointer_args": _binding_i32_attr(w, attrs["pointer_count"]),
        "tlx_wave.num_scalar_args": _binding_i32_attr(w, attrs["scalar_count"]),
        "tlx_wave.wave_size": _binding_i32_attr(w, attrs["wave_size"]),
        "tlx_wave.num_warps": _binding_i32_attr(w, attrs["num_warps"]),
        "tlx_wave.async.num_copies": _binding_i32_attr(w, attrs["async_copy_count"]),
        "tlx_wave.async.num_waits": _binding_i32_attr(w, attrs["async_wait_count"]),
    }


def _emit_wave_skeleton_with_bindings(kernel, attrs, plan):
    w = _load_wave_dsl()
    target_triple = _target_triple(attrs)
    pointer_count = sum(arg.kind == "pointer" for arg in kernel.args)
    scalar_count = sum(arg.kind == "scalar" for arg in kernel.args)
    lds_layout = _compute_lds_layout(plan)
    stats = _WaveAsyncStats(lds_size_bytes=lds_layout.size_bytes)
    bridge_stage = _bridge_stage(plan)
    with w.module() as module_builder:
        func_attrs = _binding_attrs(
            w,
            {
                "bridge_stage": bridge_stage,
                "pointer_count": pointer_count,
                "scalar_count": scalar_count,
                "wave_size": attrs.threads_per_warp,
                "num_warps": attrs.num_warps,
                "async_copy_count": plan.op_counts.get(
                    "ttg.async_copy_global_to_local", 0
                ),
                "async_wait_count": plan.op_counts.get("ttg.async_wait", 0),
            },
        )
        if kernel.noinline is not None:
            func_attrs["tlx_wave.ttgir.noinline"] = _binding_bool_attr(
                w, kernel.noinline
            )

        arg_types = [_binding_type(arg.ttgir_type_obj, w) for arg in kernel.args]
        module_builder.module.operation.attributes["waveamdmachine.target"] = (
            w.StringAttr.get(target_triple)
        )
        module_builder.module.operation.attributes["tlx_wave.source_target"] = (
            w.StringAttr.get(attrs.target)
        )
        module_builder.module.operation.attributes["tlx_wave.num_ctas"] = (
            _binding_i32_attr(w, attrs.num_ctas)
        )
        module_builder.module.operation.attributes["tlx_wave.num_warps"] = (
            _binding_i32_attr(w, attrs.num_warps)
        )
        module_builder.module.operation.attributes["tlx_wave.threads_per_warp"] = (
            _binding_i32_attr(w, attrs.threads_per_warp)
        )
        module_builder.module.operation.attributes[
            "tlx_wave.has_explicit_local_mem_access"
        ] = _binding_bool_attr(w, attrs.has_explicit_local_mem_access)
        module_builder.module.operation.attributes["tlx_wave.plan.kind"] = (
            w.StringAttr.get(plan.kind)
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_ops"] = (
            _binding_i32_attr(w, len(plan.ops))
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_values"] = (
            _binding_i32_attr(w, len(plan.values))
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_addresses"] = (
            _binding_i32_attr(w, len(plan.addresses))
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_memdescs"] = (
            _binding_i32_attr(w, len(plan.memdescs))
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_layouts"] = (
            _binding_i32_attr(w, len(plan.layouts))
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_tokens"] = (
            _binding_i32_attr(w, len(plan.tokens))
        )
        with module_builder.function(
            kernel.name,
            arg_types,
            kernel=True,
            lds_size=lds_layout.size_bytes if lds_layout.size_bytes else None,
            attrs=func_attrs,
        ) as builder:
            _emit_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats)

        module_builder.module.operation.attributes["tlx_wave.lds_size_bytes"] = (
            _binding_i32_attr(w, lds_layout.size_bytes)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.async_copies"] = (
            _binding_i32_attr(w, stats.async_copies)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.dma_load_lds"] = (
            _binding_i32_attr(w, stats.dma_load_lds)
        )
        module_builder.module.operation.attributes[
            "tlx_wave.emitted.load_store_fallbacks"
        ] = _binding_i32_attr(w, stats.load_store_fallbacks)
        module_builder.module.operation.attributes["tlx_wave.emitted.joins"] = (
            _binding_i32_attr(w, stats.joins)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.waits"] = (
            _binding_i32_attr(w, stats.waits)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.barriers"] = (
            _binding_i32_attr(w, stats.barriers)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.local_loads"] = (
            _binding_i32_attr(w, stats.local_loads)
        )
        module_builder.module.operation.attributes[
            "tlx_wave.emitted.fragment_packs"
        ] = _binding_i32_attr(w, stats.fragment_packs)
        module_builder.module.operation.attributes[
            "tlx_wave.emitted.fragment_fills"
        ] = _binding_i32_attr(w, stats.fragment_fills)
        module_builder.module.operation.attributes["tlx_wave.emitted.mmas"] = (
            _binding_i32_attr(w, stats.mmas)
        )
        return str(module_builder.module), stats


def _emit_wave_skeleton(kernel, attrs, plan):
    wave_text, stats = _emit_wave_skeleton_with_bindings(kernel, attrs, plan)
    return wave_text, "wave-dsl", stats


def _verify_wave_module(wave_text, wave_opt):
    result = subprocess.run(
        [wave_opt, "-", "--verify-diagnostics"],
        input=wave_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"tlx_wave generated Wave module failed wave-opt verification: {detail}"
        )
