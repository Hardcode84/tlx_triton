import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .wave_bridge_plan import (
    _ASSUME_TREE_OPS,
    _AMDMfmaEncodingInfo,
    _GFX950_DOT_PARENT_LAYOUT,
    _GFX950_F16_MMA_KIND,
    _GFX950_MMA_M,
    _GFX950_MMA_N,
    _GFX950_MMA_REGS,
    _GFX950_MMA_SHAPE,
    _GFX950_MMA_WAVE,
    _BlockedEncodingInfo,
    _DotOperandEncodingInfo,
    _SwizzledSharedEncodingInfo,
    _WaveAsyncStats,
    _amd_mfma_encoding_info,
    _async_address_by_token,
    _bridge_stage,
    _compute_lds_layout,
    _local_load_address_by_result,
    _memdescs_by_id,
    _memdesc_size_bytes,
    _padded_layout_bit_mapping,
    _padded_shared_encoding_info,
    _padded_shared_tile_storage_bytes,
    _padded_static_byte_offset,
    _target_triple,
    _value_id,
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


_GFX950_PROPAGATED_STORE_LAYOUT = _BlockedEncodingInfo(
    size_per_thread=(1, 1),
    threads_per_warp=(2, 32),
    warps_per_cta=(4, 1),
    order=(1, 0),
)
_GFX950_PIPELINED_STORE_LAYOUT = _BlockedEncodingInfo(
    size_per_thread=(1, 4),
    threads_per_warp=(8, 8),
    warps_per_cta=(4, 1),
    order=(1, 0),
)


class _DmaDestinationNotWholeWaveContiguous(ValueError):
    pass


@dataclass(frozen=True)
class _WaveValue:
    kind: str
    value: object
    physical_value_id: int | None = None
    aux: object | None = None


@dataclass(frozen=True)
class _DotOperandFragmentLoad:
    info: _DotOperandEncodingInfo


@dataclass(frozen=True)
class _IndexExpr:
    expr: object
    bindings: dict


@dataclass(frozen=True)
class _IndexBinary:
    kind: object
    lhs: object
    rhs: object


@dataclass(frozen=True)
class _IndexSelectCompare:
    predicate: str
    lhs: object
    rhs: object
    true_value: object
    false_value: object


@dataclass(frozen=True)
class _DimBinding:
    dim: int


@dataclass(frozen=True)
class _MaskConst:
    value: bool


@dataclass(frozen=True)
class _ScalarBool:
    value: object
    const_value: bool | None = None


@dataclass(frozen=True)
class _MemState:
    root_token: object | None = None
    open_async_tokens: tuple[object, ...] = ()
    committed_groups: tuple[object, ...] = ()
    committed_group_capacity: int | None = None


@dataclass(frozen=True)
class _LoopMemShape:
    group_count: int


@dataclass(frozen=True)
class _AssumeFact:
    value_id: int
    kind: str
    lower: int | None = None
    upper: int | None = None
    divisor: int | None = None


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
    if isinstance(source, _IndexBinary):
        return _is_deferred_index(source.lhs) or _is_deferred_index(source.rhs)
    if isinstance(source, _IndexSelectCompare):
        return (
            _is_deferred_index(source.lhs)
            or _is_deferred_index(source.rhs)
            or _is_deferred_index(source.true_value)
            or _is_deferred_index(source.false_value)
        )
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


def _dot_parent_encoding_info(attr, raw_encoding, context):
    if attr is not None and _attr_bool(attr, "is_blocked_encoding"):
        return _blocked_encoding_info(attr, raw_encoding, context)
    parent_raw = None if attr is None else str(attr)
    try:
        mfma = _amd_mfma_encoding_info(parent_raw, context)
    except ValueError as exc:
        raise ValueError(f"tlx_wave bridge cannot lower {context}: {exc}") from exc
    if mfma is not None:
        return mfma
    raise ValueError(
        f"tlx_wave bridge expected blocked or #ttg.amd_mfma TTGIR encoding "
        f"for {context}, got {parent_raw or raw_encoding}"
    )


def _dot_result_layout_info(attr, raw_encoding, context):
    if attr is not None and _attr_bool(attr, "is_blocked_encoding"):
        return _blocked_encoding_info(attr, raw_encoding, context)
    try:
        mfma = _amd_mfma_encoding_info(raw_encoding, context)
    except ValueError as exc:
        raise ValueError(f"tlx_wave bridge cannot lower {context}: {exc}") from exc
    if mfma is not None:
        return mfma
    raise ValueError(
        f"tlx_wave bridge expected blocked or #ttg.amd_mfma TTGIR encoding "
        f"for {context}, got {raw_encoding}"
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


def _same_amd_mfma_encoding(lhs, rhs):
    return (
        lhs.version == rhs.version
        and lhs.warps_per_cta == rhs.warps_per_cta
        and lhs.instr_shape == rhs.instr_shape
        and lhs.is_transposed == rhs.is_transposed
        and lhs.tiles_per_warp == rhs.tiles_per_warp
        and lhs.element_bit_width == rhs.element_bit_width
        and lhs.cga_layout == rhs.cga_layout
    )


def _same_dot_parent_encoding(lhs, rhs):
    if isinstance(lhs, _BlockedEncodingInfo) and isinstance(rhs, _BlockedEncodingInfo):
        return _same_blocked_encoding(lhs, rhs)
    if isinstance(lhs, _AMDMfmaEncodingInfo) and isinstance(rhs, _AMDMfmaEncodingInfo):
        return _same_amd_mfma_encoding(lhs, rhs)
    return False


def _same_layout_encoding(lhs, rhs):
    if lhs.encoding is not None and lhs.encoding == rhs.encoding:
        return True
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


def _scalar_mlir_value(value):
    return value.value if isinstance(value, _ScalarBool) else value


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


def _users_by_value_id(plan):
    users = {}
    for op in plan.ops:
        for operand_id in op.operands:
            users.setdefault(operand_id, []).append(op)
    return users


def _assume_tree_info(plan):
    op_by_result = _op_by_result_id(plan)
    users_by_value = _users_by_value_id(plan)
    value_ids = set()
    expanded = set()

    def walk(value_id):
        op = op_by_result.get(value_id)
        if op is None or op.name not in _ASSUME_TREE_OPS:
            return
        value_ids.add(value_id)
        if value_id in expanded:
            return
        expanded.add(value_id)
        for operand_id in op.operands:
            walk(operand_id)

    for op in plan.ops:
        if op.name == "llvm.intr.assume":
            for operand_id in op.operands:
                walk(operand_id)

    assume_tree_values = frozenset(value_ids)
    assume_only_values = set()
    visiting = set()

    def is_assume_only(value_id):
        if value_id in assume_only_values:
            return True
        if value_id not in assume_tree_values or value_id in visiting:
            return False
        visiting.add(value_id)
        try:
            users = users_by_value.get(value_id, ())
            if not users:
                return False
            for user in users:
                if user.name == "llvm.intr.assume":
                    continue
                if user.name not in _ASSUME_TREE_OPS or not user.results:
                    return False
                if not all(is_assume_only(result_id) for result_id in user.results):
                    return False
            assume_only_values.add(value_id)
            return True
        finally:
            visiting.remove(value_id)

    for value_id in assume_tree_values:
        is_assume_only(value_id)
    return assume_tree_values, frozenset(assume_only_values)


def _raw_op_plan(raw_op):
    attrs = dict(raw_op.get_attrs())
    if raw_op.get_name() in {
        "tt.get_program_id",
        "tt.get_num_programs",
        "tt.expand_dims",
    }:
        axis = raw_op.get_int_attr("axis")
        if axis is not None:
            attrs["axis"] = axis
    return SimpleNamespace(
        index=-1,
        name=raw_op.get_name(),
        operands=tuple(
            _value_id(raw_op.get_operand(index))
            for index in range(raw_op.get_num_operands())
        ),
        results=tuple(
            _value_id(raw_op.get_result(index))
            for index in range(raw_op.get_num_results())
        ),
        attrs=attrs,
        raw_op=raw_op,
    )


def _raw_block_ops(block):
    return tuple(
        block.get_operation(index) for index in range(block.get_num_operations())
    )


def _raw_block_args(block):
    return tuple(
        block.get_argument(index) for index in range(block.get_num_arguments())
    )


def _single_region_block(raw_op, region_index, context):
    if region_index >= raw_op.get_num_regions():
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: missing region "
            f"{region_index}"
        )
    region = raw_op.get_region(region_index)
    if region.empty():
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: region {region_index} "
            "is empty"
        )
    if region.size() != 1:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected one block in "
            f"region {region_index}, got {region.size()}"
        )
    return region.get_block(0)


def _dim_symbol(w, dim):
    return w.sym(f"tlx_dim{dim}")


def _is_bool_value(value):
    return value.type == "i1" or value.element_type == "i1"


def _is_integer_or_index_value(value):
    types = {value.type, value.element_type}
    return "index" in types or any(f"i{bits}" in types for bits in (1, 8, 16, 32, 64))


def _scalar_data_type(value):
    if value.type_kind != "scalar":
        return None
    if value.type in {"f16", "bf16", "f32"}:
        return value.type
    return None


def _is_data_tensor(value):
    return (
        value.type_kind == "tensor"
        and value.pointee_type is None
        and value.element_type in {"i8", "i16", "i32", "i64", "f16", "bf16", "f32"}
    )


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


def _is_wave_simd_pointer_type(typ, w):
    if typ is None:
        return False
    simd_type = getattr(w, "SimdType", None)
    if simd_type is not None:
        try:
            return simd_type.isinstance(typ) and "ptr" in str(typ)
        except (AttributeError, TypeError):
            pass
    return "simd<" in str(typ) and "ptr" in str(typ)


def _wave_cmpi_operand(builder, value, w):
    typ = getattr(value, "type", None)
    if _is_wave_simd_index_type(typ, w):
        return value
    if _is_wave_index_type(typ, w):
        return builder.index_cast(value, w.i64())
    return value


def _wave_cmpi(builder, predicate, lhs, rhs, w):
    return builder.cmpi(
        predicate,
        _wave_cmpi_operand(builder, lhs, w),
        _wave_cmpi_operand(builder, rhs, w),
    )


def _arith_cmpi(builder, predicate, lhs, rhs, w):
    return w.arith.CmpIOp(w.CmpIPredicate[predicate], lhs, rhs).result


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
    if isinstance(source, _IndexBinary):
        lhs = _materialize_index_value(builder, source.lhs, dim_bindings, w)
        rhs = _materialize_index_value(builder, source.rhs, dim_bindings, w)
        return _maybe_splat(builder, builder.binary(source.kind, lhs, rhs), force_width, w)
    if isinstance(source, _IndexSelectCompare):
        lhs = _materialize_index_value(builder, source.lhs, dim_bindings, w)
        rhs = _materialize_index_value(builder, source.rhs, dim_bindings, w)
        true_value = _materialize_index_value(
            builder, source.true_value, dim_bindings, w
        )
        false_value = _materialize_index_value(
            builder, source.false_value, dim_bindings, w
        )
        return _maybe_splat(
            builder,
            _index_select_compare_value(
                builder,
                source.predicate,
                lhs,
                rhs,
                true_value,
                false_value,
                w,
            ),
            force_width,
            w,
        )
    if isinstance(source, _DimBinding):
        return _maybe_splat(
            builder, _dim_binding_value(dim_bindings, source, w), force_width, w
        )
    return _maybe_splat(builder, source, force_width, w)


def _index_select_compare_value(
    builder,
    predicate,
    lhs,
    rhs,
    true_value,
    false_value,
    w,
):
    width = None
    for value in (lhs, rhs):
        if w.SimdType.isinstance(value.type):
            width = w.SimdType(value.type).width
            break
    if width is None:
        condition = _arith_cmpi(builder, predicate, lhs, rhs, w)
    else:
        lhs = _maybe_splat(builder, lhs, width, w)
        rhs = _maybe_splat(builder, rhs, width, w)
        true_value = _maybe_splat(builder, true_value, width, w)
        false_value = _maybe_splat(builder, false_value, width, w)
        condition = _wave_cmpi(builder, predicate, lhs, rhs, w)
    return builder.select(condition, true_value, false_value)


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
    return _product(
        _blocked_layout_component_shape(
            value_plan.shape,
            layout,
            context,
            lowering_name,
        )
    )


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _blocked_layout_dim_coverage(layout, dim, context, lowering_name):
    covered = (
        int(layout.size_per_thread[dim])
        * int(layout.threads_per_warp[dim])
        * int(layout.warps_per_cta[dim])
    )
    if covered <= 0:
        raise ValueError(
            f"tlx_wave bridge {lowering_name} cannot lower {context}: "
            f"dim {dim} has non-positive blocked layout coverage {covered}; "
            f"sizePerThread={layout.size_per_thread}, "
            f"threadsPerWarp={layout.threads_per_warp}, "
            f"warpsPerCTA={layout.warps_per_cta}"
        )
    return covered


def _blocked_layout_component_shape(shape, layout, context, lowering_name):
    component_shape = []
    for dim, extent in enumerate(shape):
        covered = _blocked_layout_dim_coverage(layout, dim, context, lowering_name)
        repeats = max(1, _ceil_div(extent, covered))
        component_shape.append(int(layout.size_per_thread[dim]) * repeats)
    return tuple(component_shape)


def _blocked_layout_dim_bindings(
    builder, value_plan, w, context, symbol_prefix, lowering_name, component=0
):
    layout = _blocked_tensor_layout_info(value_plan, context, lowering_name)
    component_shape = _blocked_layout_component_shape(
        value_plan.shape,
        layout,
        context,
        lowering_name,
    )
    component_count = _product(component_shape)
    if component < 0 or component >= component_count:
        raise ValueError(
            f"tlx_wave bridge {lowering_name} component {component} is out of "
            f"range for {context}: componentShape={component_shape}, "
            f"encoding={value_plan.encoding}"
        )
    rank = len(value_plan.shape)
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
        _delinearize_expr(w, w.sym_ctx.int_(component), component_shape, layout.order)
        if component_count != 1
        else None
    )

    dim_bindings = {}
    active = None
    for dim in range(rank):
        tile_coord = lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim]
        expanded_component = (
            component_coords[dim]
            if component_coords is not None
            else w.sym_ctx.int_(0)
        )
        size_per_thread = int(layout.size_per_thread[dim])
        covered = _blocked_layout_dim_coverage(layout, dim, context, lowering_name)
        local_component = w.mod(expanded_component, size_per_thread)
        repeat_component = w.floor(expanded_component / size_per_thread)
        coord_expr = (
            local_component
            + size_per_thread * tile_coord
            + covered * repeat_component
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
    return _materialize_index_value(
        builder,
        _linearized_tensor_offset_expr(shape, dim_bindings, w),
        {},
        w,
    )


def _linearized_tensor_offset_expr(shape, dim_bindings, w):
    offset = w.sym_ctx.int_(0)
    stride = 1
    for dim in reversed(range(len(shape))):
        offset = offset + _dim_symbol(w, dim) * stride
        stride *= int(shape[dim])
    return _IndexExpr(offset, dict(dim_bindings))


def _linearized_tensor_packet_offset_expr(
    shape, dim_bindings, packet_elements, w
):
    offset = _linearized_tensor_offset_expr(shape, dim_bindings, w)
    if packet_elements == 1:
        return offset
    return _IndexExpr(
        w.floor(offset.expr / int(packet_elements)),
        dict(offset.bindings),
    )


def _linearized_tensor_dma_dword_offset_expr(
    shape, dim_bindings, packet_elements, packet_bytes, w
):
    if packet_bytes % 4:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            f"without faithful DMA: {packet_bytes}-byte DMA packet is not "
            "addressable as i32 LDS words"
        )
    packet_offset = _linearized_tensor_packet_offset_expr(
        shape, dim_bindings, packet_elements, w
    )
    dwords_per_packet = packet_bytes // 4
    if dwords_per_packet == 1:
        return packet_offset
    return _IndexExpr(
        packet_offset.expr * int(dwords_per_packet),
        dict(packet_offset.bindings),
    )


def _apply_symbolic_padding(byte_offset, element_byte_width, info, w, context):
    padded = byte_offset
    for interval, padding in zip(info.intervals, info.paddings):
        interval_bytes = int(interval) * int(element_byte_width)
        padding_bytes = int(padding) * int(element_byte_width)
        if interval_bytes <= 0 or padding_bytes <= 0:
            raise ValueError(f"tlx_wave bridge cannot lower {context}: invalid padded_shared interval")
        padded = padded + w.floor(byte_offset / interval_bytes) * padding_bytes
    return padded


def _padded_shared_byte_offset_expr(
    shape,
    dim_bindings,
    element_byte_width,
    raw_encoding,
    w,
    context,
):
    info = _padded_shared_encoding_info(raw_encoding, context)
    if info is None:
        return None
    if element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown element byte width "
            "for padded shared-memory address"
        )
    rank = len(info.offset_vectors[0])
    if len(shape) < rank:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: padded_shared rank {rank} "
            f"exceeds shape {shape}"
        )
    prefix_rank = len(shape) - rank
    bindings = dict(dim_bindings)
    byte_offset = w.sym_ctx.int_(0)
    if prefix_rank:
        prefix_shape = shape[:prefix_rank]
        prefix = _linearized_tensor_offset_expr(prefix_shape, dim_bindings, w)
        tile_bytes = _padded_shared_tile_storage_bytes(
            shape[prefix_rank:],
            element_byte_width,
            info,
            context,
        )
        byte_offset = byte_offset + prefix.expr * int(tile_bytes)
        bindings.update(prefix.bindings)

    mapping = _padded_layout_bit_mapping(info, context)
    tile_element_offset = w.sym_ctx.int_(0)
    for (layout_dim, logical_bit), physical_bit in mapping.items():
        dim = prefix_rank + int(layout_dim)
        symbol = _dim_symbol(w, dim)
        bit = w.mod(w.floor(symbol / (1 << int(logical_bit))), 2)
        tile_element_offset = tile_element_offset + bit * (1 << int(physical_bit))
    tile_byte_offset = tile_element_offset * int(element_byte_width)
    byte_offset = byte_offset + _apply_symbolic_padding(
        tile_byte_offset,
        element_byte_width,
        info,
        w,
        context,
    )
    return _IndexExpr(byte_offset, bindings)


def _swizzled_shared_byte_offset_expr(
    memdesc,
    shape,
    dim_bindings,
    w,
    context,
):
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, context
        )
    except ValueError:
        return None
    if _is_identity_shared_layout(memdesc, shared):
        return None
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown element byte width "
            "for swizzled shared-memory address"
        )
    if len(shape) < 2 or shared.order != (1, 0):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unsupported swizzled "
            f"shared layout shape={shape}, order={shared.order}; only rank-2 "
            "order=[1, 0] swizzles are supported"
        )
    vec = int(shared.vec)
    per_phase = int(shared.per_phase)
    max_phase = int(shared.max_phase)
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: invalid swizzled shared "
            f"parameters vec={vec}, perPhase={per_phase}, maxPhase={max_phase}"
        )
    prefix_rank = len(shape) - 2
    rows = int(shape[prefix_rank])
    cols = int(shape[prefix_rank + 1])
    if cols % vec:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: swizzled shared columns "
            f"{cols} are not divisible by vec={vec}"
        )
    bindings = dict(dim_bindings)
    byte_offset = w.sym_ctx.int_(0)
    if prefix_rank:
        prefix = _linearized_tensor_offset_expr(shape[:prefix_rank], dim_bindings, w)
        byte_offset = (
            byte_offset
            + prefix.expr * rows * cols * int(memdesc.element_byte_width)
        )
        bindings.update(prefix.bindings)

    row = _dim_symbol(w, prefix_rank)
    col = _dim_symbol(w, prefix_rank + 1)
    phase = w.mod(w.floor(row / per_phase), max_phase)
    col_group = w.floor(col / vec)
    swizzled_col = w.xor(col_group, phase) * vec + w.mod(col, vec)
    element_offset = row * cols + swizzled_col
    byte_offset = byte_offset + element_offset * int(memdesc.element_byte_width)
    return _IndexExpr(byte_offset, bindings)


def _memdesc_pointer_offset_expr(
    memdesc,
    shape,
    dim_bindings,
    pointer_element_bytes,
    w,
    context,
):
    if pointer_element_bytes is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown pointer element "
            "byte width for LDS address"
        )
    offset = _padded_shared_byte_offset_expr(
        shape,
        dim_bindings,
        memdesc.element_byte_width,
        memdesc.encoding,
        w,
        context,
    )
    if offset is None:
        offset = _swizzled_shared_byte_offset_expr(
            memdesc,
            shape,
            dim_bindings,
            w,
            context,
        )
    if offset is None:
        offset = _linearized_tensor_scaled_offset_expr(
            shape,
            dim_bindings,
            memdesc.element_byte_width,
            pointer_element_bytes,
            w,
            context,
        )
        return offset
    if pointer_element_bytes == 1:
        return offset
    return _IndexExpr(
        w.floor(offset.expr / int(pointer_element_bytes)),
        dict(offset.bindings),
    )


def _memdesc_dma_dword_offset_expr(
    memdesc,
    shape,
    dim_bindings,
    packet_elements,
    packet_bytes,
    w,
):
    if packet_bytes % 4:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
            f"without faithful DMA: {packet_bytes}-byte DMA packet is not "
            "addressable as i32 LDS words"
        )
    if _padded_shared_encoding_info(
        memdesc.encoding,
        "ttg.async_copy_global_to_local destination",
    ) is None:
        try:
            shared = _swizzled_shared_encoding_info(
                memdesc.encoding_attr,
                memdesc.encoding,
                "ttg.async_copy_global_to_local destination",
            )
        except ValueError:
            shared = None
        if shared is None or _is_identity_shared_layout(memdesc, shared):
            return _linearized_tensor_dma_dword_offset_expr(
                shape,
                dim_bindings,
                packet_elements,
                packet_bytes,
                w,
            )
    return _memdesc_pointer_offset_expr(
        memdesc,
        shape,
        dim_bindings,
        4,
        w,
        "ttg.async_copy_global_to_local destination",
    )


def _zero_index_expr(w):
    return _IndexExpr(w.sym_ctx.int_(0), {})


def _is_zero_index_source(source):
    if isinstance(source, int):
        return source == 0
    if isinstance(source, _IndexExpr) and not source.bindings:
        try:
            return int(source.expr) == 0
        except (TypeError, ValueError):
            return str(source.expr) == "0"
    return False


def _stage_index_identity(w, memdesc, label, dim, source):
    symbol = w.sym(f"tlx_memdesc_{memdesc.value_id}_{label}_{dim}")
    return _IndexExpr(symbol, {symbol: source})


def _stage_index_add_const(w, memdesc, label, dim, source, offset):
    symbol = w.sym(f"tlx_memdesc_{memdesc.value_id}_{label}_{dim}")
    return _IndexExpr(symbol + int(offset), {symbol: source})


def _linearized_tensor_scaled_offset_expr(
    shape,
    dim_bindings,
    scale,
    divisor,
    w,
    context,
):
    if scale is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown memdesc element "
            "byte width for LDS address"
        )
    if divisor is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown pointer element "
            "byte width for LDS address"
        )
    offset = w.sym_ctx.int_(0)
    stride = 1
    for dim in reversed(range(len(shape))):
        coefficient = stride * int(scale)
        dim_symbol = _dim_symbol(w, dim)
        dim_source = dim_bindings.get(dim_symbol)
        if coefficient % int(divisor):
            if _is_zero_index_source(dim_source):
                stride *= int(shape[dim])
                continue
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: byte offset for dim "
                f"{dim} has stride {coefficient}, which is not aligned to "
                f"pointer element size {divisor}"
            )
        offset = offset + dim_symbol * (coefficient // int(divisor))
        stride *= int(shape[dim])
    return _IndexExpr(offset, dict(dim_bindings))


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


_TRANSPARENT_MEMDESC_VIEW_OPS = {"tlx.local_alias", "tlx.require_layout"}


def _is_identity_shared_layout(memdesc, shared):
    rank = len(memdesc.shape)
    contiguous_order = tuple(reversed(range(rank)))
    return (
        rank >= 1
        and shared.vec == 1
        and shared.per_phase == 1
        and shared.max_phase == 1
        and shared.order == contiguous_order
    )


def _is_supported_swizzled_shared_layout(memdesc, shared):
    if _is_identity_shared_layout(memdesc, shared):
        return True
    return (
        len(memdesc.shape) >= 2
        and shared.order == (1, 0)
        and int(shared.vec) > 0
        and int(shared.per_phase) > 0
        and int(shared.max_phase) > 0
        and int(memdesc.shape[-1]) % int(shared.vec) == 0
    )


def _static_delinearize_row_major(linear, shape):
    coords = [0] * len(shape)
    remainder = int(linear)
    for dim in reversed(range(len(shape))):
        extent = int(shape[dim])
        coords[dim] = remainder % extent
        remainder //= extent
    if remainder:
        raise ValueError(
            f"tlx_wave bridge internal error: linear index {linear} exceeds "
            f"shape {shape}"
        )
    return tuple(coords)


def _swizzled_static_byte_offset(memdesc, shape, coords, shared, context):
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown element byte width "
            "for swizzled shared-memory address"
        )
    if len(shape) < 2 or shared.order != (1, 0):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unsupported swizzled "
            f"shared layout shape={shape}, order={shared.order}; only rank-2 "
            "order=[1, 0] swizzles are supported"
        )
    vec = int(shared.vec)
    per_phase = int(shared.per_phase)
    max_phase = int(shared.max_phase)
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: invalid swizzled shared "
            f"parameters vec={vec}, perPhase={per_phase}, maxPhase={max_phase}"
        )
    prefix_rank = len(shape) - 2
    rows = int(shape[prefix_rank])
    cols = int(shape[prefix_rank + 1])
    if cols % vec:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: swizzled shared columns "
            f"{cols} are not divisible by vec={vec}"
        )
    prefix_shape = shape[:prefix_rank]
    prefix_coords = coords[:prefix_rank]
    prefix_index = _static_linear_offset(prefix_shape, prefix_coords) if prefix_shape else 0
    row = int(coords[prefix_rank])
    col = int(coords[prefix_rank + 1])
    phase = (row // per_phase) % max_phase
    swizzled_col = ((col // vec) ^ phase) * vec + (col % vec)
    if swizzled_col >= cols:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: swizzled column "
            f"{swizzled_col} exceeds extent {cols}"
        )
    return (
        (prefix_index * rows * cols + row * cols + swizzled_col)
        * int(memdesc.element_byte_width)
    )


def _memdesc_static_byte_offset(memdesc, shape, coords, context):
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown element byte width "
            "for shared-memory address"
        )
    padded_offset = _padded_static_byte_offset(
        shape,
        coords,
        memdesc.element_byte_width,
        memdesc.encoding,
        context,
    )
    if padded_offset is not None:
        return padded_offset
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, context
        )
    except ValueError:
        return _static_linear_offset(shape, coords) * int(memdesc.element_byte_width)
    if _is_identity_shared_layout(memdesc, shared):
        return _static_linear_offset(shape, coords) * int(memdesc.element_byte_width)
    return _swizzled_static_byte_offset(memdesc, shape, coords, shared, context)


def _require_contiguous_physical_window(
    memdesc,
    shape,
    start_linear,
    element_count,
    alignment_bytes,
    context,
):
    total_elements = _product(shape)
    if start_linear < 0 or start_linear + element_count > total_elements:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: physical window "
            f"{start_linear}:{start_linear + element_count} exceeds shape {shape}"
        )
    first = None
    for element in range(int(element_count)):
        coords = _static_delinearize_row_major(start_linear + element, shape)
        byte_offset = _memdesc_static_byte_offset(memdesc, shape, coords, context)
        if first is None:
            first = byte_offset
            if first % int(alignment_bytes):
                raise ValueError(
                    f"tlx_wave bridge cannot lower {context}: physical byte "
                    f"offset {first} is not {alignment_bytes}-byte aligned"
                )
            continue
        expected = first + element * int(memdesc.element_byte_width)
        if byte_offset != expected:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: logical elements "
                f"{start_linear}:{start_linear + element_count} are not "
                "physically contiguous in shared memory"
            )


def _require_dma_destination_physical_contiguous(
    memdesc,
    shape,
    packet_elements,
    packet_bytes,
    context,
):
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"{context}: unknown destination element byte width for DMA packet"
        )
    if packet_elements * int(memdesc.element_byte_width) != packet_bytes:
        raise ValueError(
            f"{context}: DMA packet has {packet_elements} elements but "
            f"{packet_bytes} bytes for {memdesc.element_type}"
        )
    for start in range(0, _product(shape), int(packet_elements)):
        _require_contiguous_physical_window(
            memdesc,
            shape,
            start,
            packet_elements,
            4,
            "ttg.async_copy_global_to_local destination",
        )


def _require_dma_destination_whole_wave_contiguous(
    value_plan,
    layout,
    memdesc,
    packet_elements,
    packet_bytes,
    component,
    context,
):
    width = _product(layout.threads_per_warp)
    cta_threads = width * _product(layout.warps_per_cta)
    total_packets = _product(value_plan.shape) // int(packet_elements)
    component_packet_start = int(component) * cta_threads
    active_lanes = max(0, min(width, total_packets - component_packet_start))
    if active_lanes <= 1:
        return

    first_offset = None
    for lane in range(active_lanes):
        packet_index = component_packet_start + lane
        packet_start = packet_index * int(packet_elements)
        coords = _static_delinearize_row_major(packet_start, memdesc.shape)
        byte_offset = _memdesc_static_byte_offset(
            memdesc,
            memdesc.shape,
            coords,
            "ttg.async_copy_global_to_local destination",
        )
        if first_offset is None:
            first_offset = byte_offset
            if first_offset % 4:
                raise _DmaDestinationNotWholeWaveContiguous(
                    f"{context}: DMA destination byte offset {first_offset} "
                    "is not 4-byte aligned"
                )
            continue
        expected = first_offset + lane * int(packet_bytes)
        if byte_offset != expected:
            raise _DmaDestinationNotWholeWaveContiguous(
                f"{context}: DMA destination packet starts are not "
                "whole-wave contiguous; lane "
                f"{lane} starts at byte {byte_offset}, expected {expected}"
            )


def _shared_layout_kind(memdesc, context):
    padded = _padded_shared_encoding_info(memdesc.encoding, context)
    if padded is not None:
        return "padded", padded
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr,
            memdesc.encoding,
            context,
        )
    except ValueError:
        return "linear", None
    if _is_identity_shared_layout(memdesc, shared):
        return "linear", shared
    return "swizzled", shared


def _padded_physical_coords_expr(w, data_element_linear, shape, info, context):
    rank = len(info.offset_vectors[0])
    if len(shape) < rank:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: padded_shared rank {rank} "
            f"exceeds memdesc shape {shape}"
        )
    prefix_rank = len(shape) - rank
    prefix_shape = tuple(int(dim) for dim in shape[:prefix_rank])
    tile_shape = tuple(int(dim) for dim in shape[prefix_rank:])
    tile_elements = _product(tile_shape)
    prefix_linear = w.floor(data_element_linear / tile_elements)
    tile_linear = w.mod(data_element_linear, tile_elements)
    row_major_prefix = tuple(reversed(range(prefix_rank)))
    prefix_coords = (
        _delinearize_expr(w, prefix_linear, prefix_shape, row_major_prefix)
        if prefix_rank
        else ()
    )
    tile_coords = [w.sym_ctx.int_(0) for _ in tile_shape]
    for (layout_dim, logical_bit), physical_bit in _padded_layout_bit_mapping(
        info, context
    ).items():
        bit = w.mod(w.floor(tile_linear / (1 << int(physical_bit))), 2)
        tile_coords[int(layout_dim)] = (
            tile_coords[int(layout_dim)] + bit * (1 << int(logical_bit))
        )
    return tuple(prefix_coords) + tuple(tile_coords)


def _padded_physical_coords_static(data_element_linear, shape, info, context):
    rank = len(info.offset_vectors[0])
    if len(shape) < rank:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: padded_shared rank {rank} "
            f"exceeds memdesc shape {shape}"
        )
    prefix_rank = len(shape) - rank
    prefix_shape = tuple(int(dim) for dim in shape[:prefix_rank])
    tile_shape = tuple(int(dim) for dim in shape[prefix_rank:])
    tile_elements = _product(tile_shape)
    prefix_linear = int(data_element_linear) // tile_elements
    tile_linear = int(data_element_linear) % tile_elements
    prefix_coords = (
        _static_delinearize_row_major(prefix_linear, prefix_shape)
        if prefix_rank
        else ()
    )
    tile_coords = [0 for _ in tile_shape]
    for (layout_dim, logical_bit), physical_bit in _padded_layout_bit_mapping(
        info, context
    ).items():
        bit = (tile_linear // (1 << int(physical_bit))) % 2
        tile_coords[int(layout_dim)] += bit * (1 << int(logical_bit))
    coords = tuple(prefix_coords) + tuple(tile_coords)
    for dim, coord in enumerate(coords):
        if coord < 0 or coord >= int(shape[dim]):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: physical padded "
                f"element {data_element_linear} maps to out-of-bounds coords "
                f"{coords} for shape {shape}"
            )
    return coords


def _swizzled_physical_coords_expr(w, physical_element_linear, shape, shared, context):
    if len(shape) < 2 or shared.order != (1, 0):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unsupported swizzled "
            f"shared layout shape={shape}, order={shared.order}; only rank-2 "
            "order=[1, 0] swizzles are supported"
        )
    vec = int(shared.vec)
    per_phase = int(shared.per_phase)
    max_phase = int(shared.max_phase)
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: invalid swizzled shared "
            f"parameters vec={vec}, perPhase={per_phase}, maxPhase={max_phase}"
        )
    coords = list(
        _delinearize_expr(
            w,
            physical_element_linear,
            shape,
            tuple(reversed(range(len(shape)))),
        )
    )
    prefix_rank = len(shape) - 2
    row = coords[prefix_rank]
    swizzled_col = coords[prefix_rank + 1]
    phase = w.mod(w.floor(row / per_phase), max_phase)
    col_group = w.xor(w.floor(swizzled_col / vec), phase)
    coords[prefix_rank + 1] = col_group * vec + w.mod(swizzled_col, vec)
    return tuple(coords)


def _swizzled_physical_coords_static(physical_element_linear, shape, shared, context):
    if len(shape) < 2 or shared.order != (1, 0):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unsupported swizzled "
            f"shared layout shape={shape}, order={shared.order}; only rank-2 "
            "order=[1, 0] swizzles are supported"
        )
    vec = int(shared.vec)
    per_phase = int(shared.per_phase)
    max_phase = int(shared.max_phase)
    if vec <= 0 or per_phase <= 0 or max_phase <= 0:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: invalid swizzled shared "
            f"parameters vec={vec}, perPhase={per_phase}, maxPhase={max_phase}"
        )
    coords = list(_static_delinearize_row_major(physical_element_linear, shape))
    prefix_rank = len(shape) - 2
    row = int(coords[prefix_rank])
    swizzled_col = int(coords[prefix_rank + 1])
    phase = (row // per_phase) % max_phase
    col_group = (swizzled_col // vec) ^ phase
    coords[prefix_rank + 1] = col_group * vec + (swizzled_col % vec)
    coords = tuple(coords)
    for dim, coord in enumerate(coords):
        if coord < 0 or coord >= int(shape[dim]):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: physical swizzled "
                f"element {physical_element_linear} maps to out-of-bounds coords "
                f"{coords} for shape {shape}"
            )
    return coords


def _dma_packet_start_coords_static(memdesc, packet_elements, packet_index, context):
    physical_element_linear = int(packet_index) * int(packet_elements)
    kind, info = _shared_layout_kind(memdesc, context)
    if kind == "padded":
        return _padded_physical_coords_static(
            physical_element_linear,
            memdesc.shape,
            info,
            context,
        )
    if kind == "swizzled":
        return _swizzled_physical_coords_static(
            physical_element_linear,
            memdesc.shape,
            info,
            context,
        )
    return _static_delinearize_row_major(physical_element_linear, memdesc.shape)


def _dma_packet_physical_coords_expr(
    w,
    physical_element_linear,
    memdesc,
    context,
):
    kind, info = _shared_layout_kind(memdesc, context)
    if kind == "padded":
        return _padded_physical_coords_expr(
            w,
            physical_element_linear,
            memdesc.shape,
            info,
            context,
        )
    if kind == "swizzled":
        return _swizzled_physical_coords_expr(
            w,
            physical_element_linear,
            memdesc.shape,
            info,
            context,
        )
    return _delinearize_expr(
        w,
        physical_element_linear,
        memdesc.shape,
        tuple(reversed(range(len(memdesc.shape)))),
    )


def _require_physical_packet_contiguous_from_coords(
    memdesc,
    shape,
    coords,
    packet_elements,
    packet_bytes,
    context,
):
    first = None
    inner_dim = len(shape) - 1
    for element in range(int(packet_elements)):
        packet_coords = list(coords)
        packet_coords[inner_dim] += element
        if packet_coords[inner_dim] >= int(shape[inner_dim]):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: DMA packet starting "
                f"at {coords} crosses innermost dimension extent {shape[inner_dim]}"
            )
        byte_offset = _memdesc_static_byte_offset(
            memdesc,
            shape,
            tuple(packet_coords),
            context,
        )
        if first is None:
            first = byte_offset
            if first % 4:
                raise ValueError(
                    f"tlx_wave bridge cannot lower {context}: physical byte "
                    f"offset {first} is not 4-byte aligned"
                )
            continue
        expected = first + element * int(memdesc.element_byte_width)
        if byte_offset != expected:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: DMA packet starting "
                f"at {coords} is not physically contiguous in shared memory"
            )
    if int(packet_elements) * int(memdesc.element_byte_width) != int(packet_bytes):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: DMA packet has "
            f"{packet_elements} elements but {packet_bytes} bytes for "
            f"{memdesc.element_type}"
        )
    return first


def _require_dma_destination_physical_packets(
    memdesc,
    packet_elements,
    packet_bytes,
    width,
    context,
):
    total_elements = _product(memdesc.shape)
    if total_elements % int(packet_elements):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: tensor element count "
            f"{total_elements} is not divisible by {packet_elements} element "
            "DMA packets"
        )
    total_packets = total_elements // int(packet_elements)
    packet_offsets = []
    for packet_index in range(total_packets):
        coords = _dma_packet_start_coords_static(
            memdesc,
            packet_elements,
            packet_index,
            context,
        )
        packet_offsets.append(
            _require_physical_packet_contiguous_from_coords(
                memdesc,
                memdesc.shape,
                coords,
                packet_elements,
                packet_bytes,
                context,
            )
        )

    for chunk_start in range(0, total_packets, int(width)):
        chunk_offsets = packet_offsets[chunk_start : chunk_start + int(width)]
        if not chunk_offsets:
            continue
        first_offset = chunk_offsets[0]
        if first_offset % 4:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: physical byte offset "
                f"{first_offset} is not 4-byte aligned"
            )
        for lane, byte_offset in enumerate(chunk_offsets[1:], start=1):
            expected = first_offset + lane * int(packet_bytes)
            if byte_offset != expected:
                raise _DmaDestinationNotWholeWaveContiguous(
                    f"tlx_wave bridge cannot lower {context}: DMA destination "
                    "packet starts are not whole-wave contiguous; lane "
                    f"{lane} starts at byte {byte_offset}, expected {expected}"
                )


def _require_fragment_load_physical_contiguous(value, memdesc):
    element_count = _GFX950_MMA_REGS * (4 // int(memdesc.element_byte_width))
    for lane in range(_GFX950_MMA_WAVE):
        _require_contiguous_physical_window(
            memdesc,
            memdesc.shape,
            lane * element_count,
            element_count,
            4,
            "ttg.local_load fragment source",
        )


def _padded_shared_layout_info(memdesc, context):
    try:
        return _padded_shared_encoding_info(memdesc.encoding, context)
    except ValueError as exc:
        raise ValueError(f"tlx_wave bridge cannot lower {context}: {exc}") from exc


def _validate_padded_shared_layout(memdesc, context):
    info = _padded_shared_layout_info(memdesc, context)
    if info is None:
        return None
    rank = len(info.offset_vectors[0])
    if len(memdesc.shape) < rank:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: padded_shared rank {rank} "
            f"exceeds memdesc shape {memdesc.shape}"
        )
    _padded_shared_tile_storage_bytes(
        memdesc.shape[len(memdesc.shape) - rank :],
        memdesc.element_byte_width,
        info,
        context,
    )
    return info


def _validate_generic_shared_layout(memdesc, context):
    padded = _validate_padded_shared_layout(memdesc, context)
    if padded is not None:
        return
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, context
        )
    except ValueError as exc:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: {exc}"
        ) from exc
    if _is_supported_swizzled_shared_layout(memdesc, shared):
        return
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported shared-memory "
        "encoding for generic LDS addressing; expected contiguous unswizzled "
        "#ttg.swizzled_shared or rank-2 order=[1, 0] swizzled_shared; "
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
    view_bytes = _memdesc_size_bytes(memdesc)
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


def _base_memdesc(memdesc, memdescs, context):
    if memdesc.base_value_id is None or memdesc.base_value_id not in memdescs:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc view "
            f"{memdesc.value_id} has no known base"
        )
    return memdescs[memdesc.base_value_id]


def _memdesc_index_slot_source(memdesc, state, w, context):
    if memdesc.static_index is not None:
        return _IndexExpr(w.sym_ctx.int_(int(memdesc.static_index)), {})
    if len(memdesc.view_operands) < 2:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_index view "
            "does not record an index operand"
        )
    return _require_lowered_value(
        state["wave_values"],
        memdesc.view_operands[1],
        "index_expr",
        f"{context} memdesc_index slot",
    )


def _memdesc_subslice_offsets(memdesc, context):
    offsets = tuple(int(offset) for offset in getattr(memdesc, "view_offsets", ()))
    if len(offsets) != len(memdesc.shape):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_subslice "
            f"records {len(offsets)} offset(s) for rank {len(memdesc.shape)}"
        )
    return offsets


def _memdesc_trans_order(memdesc, context):
    order = tuple(int(dim) for dim in getattr(memdesc, "view_order", ()))
    rank = len(memdesc.shape)
    if len(order) != rank or sorted(order) != list(range(rank)):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_trans order "
            f"{order} is not a permutation of rank {rank}"
        )
    return order


def _memdesc_parent_dim_bindings(
    memdesc,
    parent,
    state,
    dim_bindings,
    w,
    context,
):
    if memdesc.view_op in _TRANSPARENT_MEMDESC_VIEW_OPS:
        return dim_bindings

    if memdesc.view_op == "ttg.memdesc_index":
        if len(parent.shape) != len(memdesc.shape) + 1:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: memdesc_index "
                f"parent rank {len(parent.shape)} does not drop to child rank "
                f"{len(memdesc.shape)}"
            )
        parent_bindings = {
            _dim_symbol(w, 0): _stage_index_identity(
                w,
                memdesc,
                "index",
                0,
                _memdesc_index_slot_source(memdesc, state, w, context),
            )
        }
        for dim in range(len(memdesc.shape)):
            parent_bindings[_dim_symbol(w, dim + 1)] = _stage_index_identity(
                w,
                memdesc,
                "index",
                dim + 1,
                dim_bindings[_dim_symbol(w, dim)],
            )
        return parent_bindings

    if memdesc.view_op == "ttg.memdesc_subslice":
        if len(parent.shape) != len(memdesc.shape):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: memdesc_subslice "
                f"parent rank {len(parent.shape)} does not match child rank "
                f"{len(memdesc.shape)}"
            )
        offsets = _memdesc_subslice_offsets(memdesc, context)
        return {
            _dim_symbol(w, dim): _stage_index_add_const(
                w,
                memdesc,
                "subslice",
                dim,
                dim_bindings[_dim_symbol(w, dim)],
                offsets[dim],
            )
            for dim in range(len(memdesc.shape))
        }

    if memdesc.view_op == "ttg.memdesc_trans":
        if len(parent.shape) != len(memdesc.shape):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: memdesc_trans parent "
                f"rank {len(parent.shape)} does not match child rank "
                f"{len(memdesc.shape)}"
            )
        order = _memdesc_trans_order(memdesc, context)
        parent_bindings = {}
        for child_dim, parent_dim in enumerate(order):
            parent_bindings[_dim_symbol(w, parent_dim)] = _stage_index_identity(
                w,
                memdesc,
                "trans",
                parent_dim,
                dim_bindings[_dim_symbol(w, child_dim)],
            )
        return parent_bindings

    if memdesc.view_op == "ttg.memdesc_reshape":
        if _product(parent.shape) != _product(memdesc.shape):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: memdesc_reshape "
                f"changes logical element count from {_product(parent.shape)} "
                f"to {_product(memdesc.shape)}"
            )
        linear = _linearized_tensor_offset_expr(memdesc.shape, dim_bindings, w)
        linear_symbol = w.sym(f"tlx_memdesc_{memdesc.value_id}_reshape_linear")
        coords = _delinearize_expr(
            w,
            linear_symbol,
            parent.shape,
            tuple(reversed(range(len(parent.shape)))),
        )
        return {
            _dim_symbol(w, dim): _IndexExpr(coords[dim], {linear_symbol: linear})
            for dim in range(len(parent.shape))
        }

    _unsupported_memdesc_view(context, memdesc)


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

    parent = _base_memdesc(memdesc, memdescs, context)
    if memdesc.view_op in _TRANSPARENT_MEMDESC_VIEW_OPS or memdesc.view_op in {
        "ttg.memdesc_trans",
        "ttg.memdesc_reshape",
        "ttg.memdesc_reinterpret",
    }:
        return _emit_memdesc_base_ptr(
            builder,
            parent,
            memdescs,
            lds_layout,
            state,
            pointer_element_type,
            pointer_element_bytes,
            w,
            context,
        )

    if memdesc.view_op not in {"ttg.memdesc_index", "ttg.memdesc_subslice"}:
        _unsupported_memdesc_view(context, memdesc)

    base = _emit_memdesc_base_ptr(
        builder,
        parent,
        memdescs,
        lds_layout,
        state,
        pointer_element_type,
        pointer_element_bytes,
        w,
        context,
    )
    if memdesc.view_op == "ttg.memdesc_index":
        offset = _emit_memdesc_index_offset(
            builder,
            memdesc,
            state,
            pointer_element_bytes,
            w,
            context,
        )
    else:
        byte_offset = _memdesc_static_subslice_offset_bytes(
            memdesc, parent, context
        )
        if byte_offset % pointer_element_bytes:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: memdesc_subslice "
                f"byte offset {byte_offset} is not aligned to pointer element "
                f"size {pointer_element_bytes}"
            )
        offset = builder.index_expr(
            w.sym_ctx.int_(byte_offset // pointer_element_bytes)
        )
    return builder.ptr_add(base, offset)


def _static_linear_offset(shape, coords):
    if len(shape) != len(coords):
        raise ValueError(
            "tlx_wave bridge internal error: static offset rank mismatch "
            f"for shape={shape}, coords={coords}"
        )
    offset = 0
    stride = 1
    for dim in reversed(range(len(shape))):
        offset += int(coords[dim]) * stride
        stride *= int(shape[dim])
    return offset


def _memdesc_static_subslice_offset_bytes(memdesc, parent, context):
    if parent.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_subslice parent "
            f"has unknown element byte width for {parent.element_type}"
        )
    shape = parent.alloc_shape or parent.shape
    padded_offset = _padded_static_byte_offset(
        shape,
        _memdesc_subslice_offsets(memdesc, context),
        parent.element_byte_width,
        parent.encoding,
        context,
    )
    if padded_offset is not None:
        return padded_offset
    return (
        _static_linear_offset(shape, _memdesc_subslice_offsets(memdesc, context))
        * parent.element_byte_width
    )


def _memdesc_subslice_is_contiguous(memdesc, context):
    shape = tuple(int(dim) for dim in memdesc.shape)
    alloc_shape = tuple(int(dim) for dim in (memdesc.alloc_shape or memdesc.shape))
    offsets = _memdesc_subslice_offsets(memdesc, context)
    if len(shape) != len(alloc_shape):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: memdesc_subslice shape "
            f"rank {len(shape)} does not match alloc_shape rank {len(alloc_shape)}"
        )
    first_varying_dim = next(
        (dim for dim, extent in enumerate(shape) if extent > 1),
        None,
    )
    if first_varying_dim is None:
        return True
    for dim in range(first_varying_dim + 1, len(shape)):
        if offsets[dim] != 0 or shape[dim] != alloc_shape[dim]:
            return False
    return True


def _memdesc_dma_can_use_base_pointer(memdesc, memdescs, context):
    if memdesc.kind == "allocation":
        return True
    parent = _base_memdesc(memdesc, memdescs, context)
    if memdesc.view_op in _TRANSPARENT_MEMDESC_VIEW_OPS:
        return _memdesc_dma_can_use_base_pointer(parent, memdescs, context)
    if memdesc.view_op == "ttg.memdesc_index":
        return _memdesc_dma_can_use_base_pointer(parent, memdescs, context)
    if memdesc.view_op == "ttg.memdesc_subslice":
        return (
            _memdesc_dma_can_use_base_pointer(parent, memdescs, context)
            and _memdesc_subslice_is_contiguous(memdesc, context)
        )
    if memdesc.view_op in {"ttg.memdesc_reshape", "ttg.memdesc_reinterpret"}:
        return _memdesc_dma_can_use_base_pointer(parent, memdescs, context)
    return False


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

    parent = _base_memdesc(memdesc, memdescs, context)
    if memdesc.view_op in _TRANSPARENT_MEMDESC_VIEW_OPS:
        return _memdesc_base_is_aligned(
            parent,
            memdescs,
            lds_layout,
            pointer_element_bytes,
            context,
        )
    if memdesc.view_op in {
        "ttg.memdesc_trans",
        "ttg.memdesc_reshape",
        "ttg.memdesc_reinterpret",
    }:
        return _memdesc_base_is_aligned(
            parent,
            memdescs,
            lds_layout,
            pointer_element_bytes,
            context,
        )
    if memdesc.view_op not in {"ttg.memdesc_index", "ttg.memdesc_subslice"}:
        _unsupported_memdesc_view(context, memdesc)
    if not _memdesc_base_is_aligned(
        parent,
        memdescs,
        lds_layout,
        pointer_element_bytes,
        context,
    ):
        return False

    if memdesc.view_op == "ttg.memdesc_index":
        view_bytes = _memdesc_size_bytes(memdesc)
        if memdesc.static_index is not None:
            return (memdesc.static_index * view_bytes) % pointer_element_bytes == 0
        return view_bytes % pointer_element_bytes == 0

    return (
        _memdesc_static_subslice_offset_bytes(memdesc, parent, context)
        % pointer_element_bytes
        == 0
    )


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
    element_type = _wave_element_type(memdesc.element_type, w, context)
    return _emit_memdesc_ptr_for_type(
        builder,
        memdesc,
        memdescs,
        lds_layout,
        state,
        element_type,
        memdesc.element_byte_width,
        dim_bindings,
        width,
        w,
        context,
    )


def _emit_memdesc_ptr_for_type(
    builder,
    memdesc,
    memdescs,
    lds_layout,
    state,
    pointer_element_type,
    pointer_element_bytes,
    dim_bindings,
    width,
    w,
    context,
):
    if pointer_element_bytes is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: unknown pointer element "
            "byte width for LDS address"
        )
    if memdesc.kind == "view" and memdesc.view_op in _TRANSPARENT_MEMDESC_VIEW_OPS:
        _validate_generic_shared_layout(memdesc, context)
        return _emit_memdesc_ptr_for_type(
            builder,
            _base_memdesc(memdesc, memdescs, context),
            memdescs,
            lds_layout,
            state,
            pointer_element_type,
            pointer_element_bytes,
            dim_bindings,
            width,
            w,
            context,
        )

    if memdesc.kind == "view" and memdesc.view_op == "ttg.memdesc_reinterpret":
        _validate_generic_shared_layout(memdesc, context)
        base = _emit_memdesc_base_ptr(
            builder,
            _base_memdesc(memdesc, memdescs, context),
            memdescs,
            lds_layout,
            state,
            pointer_element_type,
            pointer_element_bytes,
            w,
            context,
        )
        offset = _memdesc_pointer_offset_expr(
            memdesc,
            memdesc.shape,
            dim_bindings,
            pointer_element_bytes,
            w,
            context,
        )
        return builder.ptr_add(
            base,
            _materialize_index_value(builder, offset, {}, w),
            w.simd_ptr_type(pointer_element_type, w.shared_address_space(), width),
        )

    if memdesc.kind == "view" and memdesc.view_op in {
        "ttg.memdesc_index",
        "ttg.memdesc_subslice",
    }:
        parent = _base_memdesc(memdesc, memdescs, context)
        if parent.kind != "allocation":
            return _emit_memdesc_ptr_for_type(
                builder,
                parent,
                memdescs,
                lds_layout,
                state,
                pointer_element_type,
                pointer_element_bytes,
                _memdesc_parent_dim_bindings(
                    memdesc,
                    parent,
                    state,
                    dim_bindings,
                    w,
                    context,
                ),
                width,
                w,
                context,
            )
        _validate_generic_shared_layout(memdesc, context)
        base = _emit_memdesc_base_ptr(
            builder,
            memdesc,
            memdescs,
            lds_layout,
            state,
            pointer_element_type,
            pointer_element_bytes,
            w,
            context,
        )
        offset_shape = (
            memdesc.alloc_shape
            if memdesc.view_op == "ttg.memdesc_subslice"
            else memdesc.shape
        )
        offset = _memdesc_pointer_offset_expr(
            memdesc,
            offset_shape,
            dim_bindings,
            pointer_element_bytes,
            w,
            context,
        )
        return builder.ptr_add(
            base,
            _materialize_index_value(builder, offset, {}, w),
            w.simd_ptr_type(pointer_element_type, w.shared_address_space(), width),
        )

    if memdesc.kind == "view" and memdesc.view_op in {
        "ttg.memdesc_trans",
        "ttg.memdesc_reshape",
    }:
        parent = _base_memdesc(memdesc, memdescs, context)
        return _emit_memdesc_ptr_for_type(
            builder,
            parent,
            memdescs,
            lds_layout,
            state,
            pointer_element_type,
            pointer_element_bytes,
            _memdesc_parent_dim_bindings(
                memdesc,
                parent,
                state,
                dim_bindings,
                w,
                context,
            ),
            width,
            w,
            context,
        )

    if memdesc.kind != "allocation":
        _unsupported_memdesc_view(context, memdesc)

    _validate_generic_shared_layout(memdesc, context)
    if memdesc.value_id not in lds_layout.offsets and memdesc.kind == "allocation":
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: no LDS placement for "
            f"memdesc {memdesc.value_id}"
        )
    base = builder.lds_base(
        pointer_element_type,
        offset=lds_layout.offsets[memdesc.value_id],
    )
    offset = _memdesc_pointer_offset_expr(
        memdesc,
        memdesc.shape,
        dim_bindings,
        pointer_element_bytes,
        w,
        context,
    )
    return builder.ptr_add(
        base,
        _materialize_index_value(builder, offset, {}, w),
        w.simd_ptr_type(pointer_element_type, w.shared_address_space(), width),
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
        return _simd_splat_index_operand(
            builder,
            lowered.value,
            value_plan,
            width,
            w,
            context,
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
    if isinstance(source, _IndexBinary):
        return _IndexBinary(
            source.kind,
            _shift_index_dims(source.lhs, axis),
            _shift_index_dims(source.rhs, axis),
        )
    if isinstance(source, _IndexSelectCompare):
        return _IndexSelectCompare(
            source.predicate,
            _shift_index_dims(source.lhs, axis),
            _shift_index_dims(source.rhs, axis),
            _shift_index_dims(source.true_value, axis),
            _shift_index_dims(source.false_value, axis),
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


def _require_typed_wave_value(wave_values, value_id, context):
    lowered = wave_values.get(value_id)
    if lowered is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: TTGIR value {value_id} "
            "has not been lowered by a preceding ordered TTGIR op"
        )
    if not isinstance(lowered, _WaveValue):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: TTGIR value {value_id} "
            f"lowered as {type(lowered).__name__}, expected typed Wave value"
        )
    return lowered


def _fragment_regs_aux(lowered):
    if not isinstance(lowered, _WaveValue) or not isinstance(lowered.aux, dict):
        return None
    return lowered.aux.get("fragment_regs")


def _fragment_regs_source_aux(lowered):
    if not isinstance(lowered, _WaveValue) or not isinstance(lowered.aux, dict):
        return None
    return lowered.aux.get("fragment_regs_source")


def _has_fragment_regs_aux(lowered):
    return (
        _fragment_regs_aux(lowered) is not None
        or _fragment_regs_source_aux(lowered) is not None
    )


def _materialize_fragment_regs_aux(lowered):
    fragment_regs = _fragment_regs_aux(lowered)
    if fragment_regs is not None:
        return fragment_regs
    source = _fragment_regs_source_aux(lowered)
    if source is None:
        return None
    return source()


def _with_fragment_regs_aux(lowered, regs):
    if not isinstance(lowered, _WaveValue):
        raise ValueError("tlx_wave bridge internal error: untyped auxiliary value")
    aux = dict(lowered.aux) if isinstance(lowered.aux, dict) else {}
    aux["fragment_regs"] = regs
    return _WaveValue(lowered.kind, lowered.value, lowered.physical_value_id, aux)


def _with_fragment_regs_source_aux(lowered, source):
    if not isinstance(lowered, _WaveValue):
        raise ValueError("tlx_wave bridge internal error: untyped auxiliary value")
    aux = dict(lowered.aux) if isinstance(lowered.aux, dict) else {}
    aux["fragment_regs_source"] = source
    return _WaveValue(lowered.kind, lowered.value, lowered.physical_value_id, aux)


def _arith_mixed_error(op_name, lhs, rhs):
    raise ValueError(
        f"tlx_wave bridge cannot lower {op_name}: unsupported mixed "
        f"formula/data operands lowered as {lhs.kind} and {rhs.kind}"
    )


def _init_argument_wave_values(builder, kernel, values, state, w):
    wave_values = state["wave_values"]
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
        elif value.type_kind == "scalar" and _is_bool_value(value):
            _set_wave_value(
                wave_values,
                value.value_id,
                "scalar",
                _ScalarBool(arg),
            )
        elif value.type_kind == "scalar" and _is_integer_or_index_value(value):
            symbol = w.sym(f"tlx_arg_{value.value_id}")
            _set_wave_value(
                wave_values,
                value.value_id,
                "index_expr",
                _IndexExpr(symbol, {symbol: arg}),
            )
            arg_info = kernel.args[value.base_arg_index]
            if arg_info.divisibility is not None and arg_info.divisibility > 1:
                state["assume_facts"].append(
                    _AssumeFact(
                        value.value_id,
                        "divisible",
                        divisor=arg_info.divisibility,
                    )
                )
        elif value.type_kind == "scalar" and _scalar_data_type(value) is not None:
            _set_wave_value(
                wave_values,
                value.value_id,
                "scalar",
                arg,
            )


def _emit_constant_op(builder, op, values, wave_values, w, stats=None):
    if len(op.results) != 1:
        return
    value = values[op.results[0]]
    const = value.const_value
    if value.type_kind == "scalar" and _is_bool_value(value) and isinstance(const, (bool, int)):
        _set_wave_value(
            wave_values,
            value.value_id,
            "scalar",
            _ScalarBool(
                _constant_value(builder, "i1", const, w, "arith.constant scalar i1"),
                bool(const),
            ),
        )
    elif _is_bool_value(value) and isinstance(const, (bool, int)):
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
            _IndexExpr(w.sym_ctx.int_(const), {}),
        )
    elif (
        value.type_kind == "tensor"
        and value.element_type == "f32"
        and value.shape == _GFX950_MMA_SHAPE
        and const in (0, 0.0)
    ):
        zero = builder.constant(w.i32(), 0)
        _set_wave_value(
            wave_values,
            value.value_id,
            "fragment",
            builder.fragment_fill(zero, _acc_fragment_type(w)),
        )
        if stats is not None:
            stats.fragment_fills += 1
    elif _is_data_tensor(value) and isinstance(const, (int, float)):
        component_count = _blocked_layout_component_count(
            value, "arith.constant result", "SIMD tensor constant"
        )
        width = _tensor_lane_width(value, "arith.constant result")
        component = _splat_constant_value(
            builder,
            value.element_type,
            const,
            width,
            w,
            "arith.constant tensor data",
        )
        if component_count == 1:
            _set_wave_value(wave_values, value.value_id, "simd", component)
        else:
            _set_wave_value(
                wave_values,
                value.value_id,
                "simd_tuple",
                tuple(component for _ in range(component_count)),
            )
    elif _scalar_data_type(value) is not None and isinstance(const, (int, float)):
        _set_wave_value(
            wave_values,
            value.value_id,
            "scalar",
            _constant_value(
                builder,
                value.type,
                const,
                w,
                "arith.constant scalar data",
            ),
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
        binding = builder.workgroup_id(axis)
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
    expr = None
    if op.name == "arith.addi":
        expr = lhs_symbol + rhs_symbol
    elif op.name == "arith.muli":
        expr = lhs_symbol * rhs_symbol
    elif op.name == "arith.subi":
        expr = lhs_symbol - rhs_symbol
    if expr is not None:
        _set_wave_value(
            wave_values,
            result.value_id,
            "index_expr",
            _IndexExpr(expr, {lhs_symbol: lhs, rhs_symbol: rhs}),
        )
        return

    binary_kind = _wave_binary_kind_for_op(op.name, w)
    if binary_kind is not None:
        _set_wave_value(
            wave_values,
            result.value_id,
            "index_expr",
            _IndexBinary(binary_kind, lhs, rhs),
        )
        return

    minmax_predicate = _minmax_select_predicate(op.name)
    if minmax_predicate is None:
        raise ValueError(f"tlx_wave bridge cannot lower scalar/index {op.name}")
    _set_wave_value(
        wave_values,
        result.value_id,
        "index_expr",
        _IndexSelectCompare(minmax_predicate, lhs, rhs, lhs, rhs),
    )


def _wave_binary_kind_for_op(op_name, w):
    return {
        "arith.divsi": w.BinaryKind.DivSI,
        "arith.divui": w.BinaryKind.DivUI,
        "arith.remsi": w.BinaryKind.RemSI,
        "arith.remui": w.BinaryKind.RemUI,
    }.get(op_name)


def _simd_binary_kind_for_op(op_name, w):
    return {
        "arith.addi": w.BinaryKind.AddI,
        "arith.muli": w.BinaryKind.MulI,
        "arith.subi": w.BinaryKind.SubI,
        "arith.divsi": w.BinaryKind.DivSI,
        "arith.divui": w.BinaryKind.DivUI,
        "arith.remsi": w.BinaryKind.RemSI,
        "arith.remui": w.BinaryKind.RemUI,
        "arith.andi": w.BinaryKind.AndI,
        "arith.ori": w.BinaryKind.OrI,
        "arith.xori": w.BinaryKind.XOrI,
    }.get(op_name)


def _minmax_select_predicate(op_name):
    return {
        "arith.minsi": "sle",
        "arith.minui": "ule",
        "arith.maxsi": "sge",
        "arith.maxui": "uge",
    }.get(op_name)


def _validate_simd_operand_layout(operand_plan, result_plan, op_name):
    if operand_plan.type_kind != "tensor" or result_plan.type_kind != "tensor":
        raise ValueError(
            f"tlx_wave bridge cannot lower {op_name} as SIMD data: expected "
            f"tensor operands/results, got {operand_plan.type} -> {result_plan.type}"
        )
    if operand_plan.shape != result_plan.shape:
        raise ValueError(
            f"tlx_wave bridge cannot lower {op_name} as SIMD data: operand "
            f"shape {operand_plan.shape} does not match result shape "
            f"{result_plan.shape}"
        )
    if not _same_layout_encoding(operand_plan, result_plan):
        raise ValueError(
            f"tlx_wave bridge cannot lower {op_name} as SIMD data through "
            "an implicit layout conversion; operand encoding: "
            f"{operand_plan.encoding}; result encoding: {result_plan.encoding}"
        )


def _simd_splat_index_operand(builder, source, result_plan, width, w, context):
    if _is_deferred_index(source):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: deferred address/index "
            "formula cannot be mixed with SIMD tensor data"
        )
    scalar = _materialize_index_value(builder, source, {}, w)
    element_type = _wave_element_type(result_plan.element_type, w, context)
    if _is_wave_index_type(getattr(scalar, "type", None), w):
        scalar = builder.index_cast(scalar, element_type)
    return builder.splat(scalar, width=width)


def _simd_arith_operand_components(
    builder,
    values,
    lowered,
    operand_id,
    result_plan,
    component_count,
    width,
    w,
    context,
    *,
    require_same_element=True,
):
    operand_plan = values[operand_id]
    if lowered.kind in {"simd", "simd_tuple"}:
        _validate_simd_operand_layout(operand_plan, result_plan, context)
        if require_same_element and operand_plan.element_type != result_plan.element_type:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context} as SIMD data: "
                f"operand element type {operand_plan.element_type} does not "
                f"match result element type {result_plan.element_type}"
            )
        return _simd_components_for_layout(
            lowered, operand_plan, component_count, context
        )
    if lowered.kind == "index_expr":
        component = _simd_splat_index_operand(
            builder, lowered.value, result_plan, width, w, context
        )
        return tuple(component for _ in range(component_count))
    raise ValueError(
        f"tlx_wave bridge cannot lower {context} as SIMD data: operand "
        f"{operand_id} lowered as {lowered.kind}"
    )


def _result_wave_value_from_components(components):
    if len(components) == 1:
        return _WaveValue("simd", components[0])
    return _WaveValue("simd_tuple", tuple(components))


def _emit_simd_binary_op(builder, op, values, wave_values, w, kind):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError(f"tlx_wave bridge expected {op.name} with two operands")
    result = values[op.results[0]]
    if result.type_kind != "tensor" or not _is_integer_or_index_value(result):
        raise ValueError(
            f"tlx_wave bridge cannot lower {op.name} as SIMD data: expected "
            f"integer tensor result, got {result.type}"
        )
    lhs = _require_typed_wave_value(wave_values, op.operands[0], op.name)
    rhs = _require_typed_wave_value(wave_values, op.operands[1], op.name)
    component_count = _blocked_layout_component_count(
        result, f"{op.name} result", "SIMD data arithmetic"
    )
    width = _tensor_lane_width(result, f"{op.name} result")
    lhs_components = _simd_arith_operand_components(
        builder, values, lhs, op.operands[0], result, component_count, width, w, op.name
    )
    rhs_components = _simd_arith_operand_components(
        builder, values, rhs, op.operands[1], result, component_count, width, w, op.name
    )
    components = tuple(
        builder.binary(kind, lhs_component, rhs_component)
        for lhs_component, rhs_component in zip(lhs_components, rhs_components)
    )
    wave_values[result.value_id] = _result_wave_value_from_components(components)


def _emit_simd_minmax_op(builder, op, values, wave_values, w, predicate):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError(f"tlx_wave bridge expected {op.name} with two operands")
    result = values[op.results[0]]
    if result.type_kind != "tensor" or not _is_integer_or_index_value(result):
        raise ValueError(
            f"tlx_wave bridge cannot lower {op.name} as SIMD data: expected "
            f"integer tensor result, got {result.type}"
        )
    lhs = _require_typed_wave_value(wave_values, op.operands[0], op.name)
    rhs = _require_typed_wave_value(wave_values, op.operands[1], op.name)
    component_count = _blocked_layout_component_count(
        result, f"{op.name} result", "SIMD data min/max"
    )
    width = _tensor_lane_width(result, f"{op.name} result")
    lhs_components = _simd_arith_operand_components(
        builder, values, lhs, op.operands[0], result, component_count, width, w, op.name
    )
    rhs_components = _simd_arith_operand_components(
        builder, values, rhs, op.operands[1], result, component_count, width, w, op.name
    )
    components = tuple(
        builder.select(
            _wave_cmpi(builder, predicate, lhs_component, rhs_component, w),
            lhs_component,
            rhs_component,
        )
        for lhs_component, rhs_component in zip(lhs_components, rhs_components)
    )
    wave_values[result.value_id] = _result_wave_value_from_components(components)


def _emit_arith_binary_op(builder, op, values, wave_values, w):
    lhs = _require_typed_wave_value(wave_values, op.operands[0], op.name)
    rhs = _require_typed_wave_value(wave_values, op.operands[1], op.name)
    if lhs.kind == "index_expr" and rhs.kind == "index_expr":
        _emit_index_binary_op(builder, op, values, wave_values, w)
        return
    data_kinds = {"simd", "simd_tuple", "index_expr"}
    if lhs.kind in data_kinds and rhs.kind in data_kinds:
        binary_kind = _simd_binary_kind_for_op(op.name, w)
        if binary_kind is not None:
            _emit_simd_binary_op(builder, op, values, wave_values, w, binary_kind)
            return
        minmax_predicate = _minmax_select_predicate(op.name)
        if minmax_predicate is not None:
            _emit_simd_minmax_op(
                builder, op, values, wave_values, w, minmax_predicate
            )
            return
    _arith_mixed_error(op.name, lhs, rhs)


def _float_cast_result_type(result_plan, width, w, context):
    result_element = _scalar_data_type(result_plan) or result_plan.element_type
    if result_element not in {"f16", "bf16", "f32"}:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: result is not a "
            f"supported floating-point type, got {result_plan.type}"
        )
    result_type = _wave_element_type(result_element, w, context)
    return w.simd_type(result_type, width) if width is not None else result_type


def _validate_float_cast_plans(source_plan, result_plan, context):
    if source_plan.type_kind != result_plan.type_kind:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: source/result type kinds "
            f"differ ({source_plan.type_kind} -> {result_plan.type_kind})"
        )
    if source_plan.type_kind == "scalar":
        if (
            _scalar_data_type(source_plan) is None
            or _scalar_data_type(result_plan) is None
        ):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: expected scalar "
                f"floating-point cast, got {source_plan.type} -> {result_plan.type}"
            )
        return
    if source_plan.type_kind != "tensor" or not _is_data_tensor(result_plan):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected tensor "
            f"floating-point cast, got {source_plan.type} -> {result_plan.type}"
        )
    if source_plan.element_type not in {"f16", "bf16", "f32"}:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: source element type "
            f"{source_plan.element_type} is not a supported floating-point type"
        )
    if result_plan.element_type not in {"f16", "bf16", "f32"}:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: result element type "
            f"{result_plan.element_type} is not a supported floating-point type"
        )
    if source_plan.shape != result_plan.shape:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: cast changes tensor "
            f"shape {source_plan.shape} -> {result_plan.shape}"
        )
    if not _same_layout_encoding(source_plan, result_plan):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: cast changes tensor "
            f"layout {source_plan.encoding} -> {result_plan.encoding}"
        )


def _emit_float_cast_op(builder, op, values, wave_values, w):
    if len(op.operands) != 1 or len(op.results) != 1:
        raise ValueError(f"tlx_wave bridge expected {op.name} with one operand")
    source_id = op.operands[0]
    result_id = op.results[0]
    source_plan = values[source_id]
    result_plan = values[result_id]
    _validate_float_cast_plans(source_plan, result_plan, op.name)
    source = _require_typed_wave_value(wave_values, source_id, op.name)

    if source.kind == "scalar":
        casted = builder.fpconvert(
            _scalar_mlir_value(source.value),
            _float_cast_result_type(result_plan, None, w, op.name),
        )
        _set_wave_value(wave_values, result_id, "scalar", casted)
        return

    if source.kind in {"simd", "simd_tuple"}:
        component_count = _blocked_layout_component_count(
            result_plan, f"{op.name} result", "SIMD tensor fpconvert"
        )
        source_components = _simd_components_for_layout(
            source, source_plan, component_count, op.name
        )
        result_width = _tensor_lane_width(result_plan, f"{op.name} result")
        components = tuple(
            builder.fpconvert(
                component,
                _float_cast_result_type(result_plan, result_width, w, op.name),
            )
            for component in source_components
        )
        wave_values[result_id] = _result_wave_value_from_components(components)
        return

    raise ValueError(
        f"tlx_wave bridge cannot lower {op.name}: source value lowered as "
        f"{source.kind}, expected scalar or SIMD tensor data"
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
    if result.type_kind == "scalar":
        if not _is_bool_value(result):
            raise ValueError(
                "tlx_wave bridge cannot lower scalar arith.cmpi with "
                f"non-i1 result {result.type}"
            )
        if _is_deferred_index(lhs) or _is_deferred_index(rhs):
            raise ValueError(
                "tlx_wave bridge cannot lower scalar arith.cmpi from "
                "layout-dependent operands"
            )
        lhs_value = _materialize_index_value(builder, lhs, {}, w)
        rhs_value = _materialize_index_value(builder, rhs, {}, w)
        _set_wave_value(
            wave_values,
            result.value_id,
            "scalar",
            _arith_cmpi(builder, predicate, lhs_value, rhs_value, w),
        )
        return
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


def _emit_simd_cmp_op(builder, op, values, wave_values, w, predicate):
    result = values[op.results[0]]
    if result.type_kind != "tensor" or not _is_bool_value(result):
        raise ValueError(
            f"tlx_wave bridge cannot lower arith.cmpi as SIMD data: expected "
            f"i1 tensor result, got {result.type}"
        )
    component_count = _blocked_layout_component_count(
        result, "arith.cmpi result", "SIMD data compare"
    )
    if component_count != 1:
        raise ValueError(
            "tlx_wave bridge cannot lower arith.cmpi as SIMD data for "
            f"multi-component tensor layouts yet; encoding={result.encoding}"
        )
    width = _tensor_lane_width(result, "arith.cmpi result")
    lhs = _require_typed_wave_value(wave_values, op.operands[0], "arith.cmpi")
    rhs = _require_typed_wave_value(wave_values, op.operands[1], "arith.cmpi")
    lhs_components = _simd_arith_operand_components(
        builder,
        values,
        lhs,
        op.operands[0],
        result,
        component_count,
        width,
        w,
        "arith.cmpi",
        require_same_element=False,
    )
    rhs_components = _simd_arith_operand_components(
        builder,
        values,
        rhs,
        op.operands[1],
        result,
        component_count,
        width,
        w,
        "arith.cmpi",
        require_same_element=False,
    )
    _set_wave_value(
        wave_values,
        result.value_id,
        "mask_expr",
        _wave_cmpi(builder, predicate, lhs_components[0], rhs_components[0], w),
    )


def _emit_typed_cmp_op(builder, op, values, wave_values, w):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected arith.cmpi with two operands")
    predicate = _CMPI_PREDICATES.get(int(op.attrs.get("predicate")))
    if predicate is None:
        raise ValueError(
            "tlx_wave bridge cannot lower arith.cmpi predicate "
            f"{op.attrs.get('predicate')}"
        )
    lhs = _require_typed_wave_value(wave_values, op.operands[0], "arith.cmpi")
    rhs = _require_typed_wave_value(wave_values, op.operands[1], "arith.cmpi")
    if lhs.kind == "index_expr" and rhs.kind == "index_expr":
        _emit_cmp_op(builder, op, values, wave_values, w)
        return
    data_kinds = {"simd", "simd_tuple", "index_expr"}
    if lhs.kind in data_kinds and rhs.kind in data_kinds:
        _emit_simd_cmp_op(builder, op, values, wave_values, w, predicate)
        return
    _arith_mixed_error("arith.cmpi", lhs, rhs)


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


def _emit_typed_and_op(builder, op, values, wave_values, w):
    if len(op.operands) != 2 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected arith.andi with two operands")
    lhs = _require_typed_wave_value(wave_values, op.operands[0], "arith.andi")
    rhs = _require_typed_wave_value(wave_values, op.operands[1], "arith.andi")
    if lhs.kind == "mask_expr" and rhs.kind == "mask_expr":
        _emit_mask_and_op(builder, op, values, wave_values, w)
        return
    data_kinds = {"simd", "simd_tuple", "index_expr"}
    if lhs.kind in data_kinds and rhs.kind in data_kinds:
        _emit_simd_binary_op(builder, op, values, wave_values, w, w.BinaryKind.AndI)
        return
    _arith_mixed_error("arith.andi", lhs, rhs)


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


def _const_int_value(values, value_id):
    value = values.get(value_id)
    if value is None or _is_bool_value(value):
        return None
    const = value.const_value
    return const if type(const) is int else None


def _is_assumable_index_value(state, value_id):
    value = state["values"].get(value_id)
    if (
        value is None
        or value.type_kind != "scalar"
        or not _is_integer_or_index_value(value)
    ):
        return False
    lowered = state["wave_values"].get(value_id)
    return (
        isinstance(lowered, _WaveValue)
        and lowered.kind == "index_expr"
        and not _is_deferred_index(lowered.value)
    )


def _range_fact_from_value_const(state, value_id, predicate, const_value):
    if not _is_assumable_index_value(state, value_id):
        return None
    if predicate == "sgt":
        return _AssumeFact(value_id, "range", lower=const_value + 1)
    if predicate == "sge":
        return _AssumeFact(value_id, "range", lower=const_value)
    if predicate == "slt":
        return _AssumeFact(value_id, "range", upper=const_value - 1)
    if predicate == "sle":
        return _AssumeFact(value_id, "range", upper=const_value)
    if predicate == "eq":
        return _AssumeFact(value_id, "range", lower=const_value, upper=const_value)
    return None


def _invert_cmpi_predicate(predicate):
    return {
        "eq": "eq",
        "ne": "ne",
        "slt": "sgt",
        "sle": "sge",
        "sgt": "slt",
        "sge": "sle",
        "ult": "ugt",
        "ule": "uge",
        "ugt": "ult",
        "uge": "ule",
    }.get(predicate)


def _divisibility_fact_from_remainder(state, rem_value_id):
    rem_op = state["op_by_result"].get(rem_value_id)
    if rem_op is None or rem_op.name != "arith.remsi":
        return None
    if len(rem_op.operands) != 2:
        return None
    value_id, divisor_id = rem_op.operands
    divisor = _const_int_value(state["values"], divisor_id)
    if divisor is None or divisor <= 0:
        return None
    if not _is_assumable_index_value(state, value_id):
        return None
    return _AssumeFact(value_id, "divisible", divisor=divisor)


def _assume_fact_from_cmp(op, state):
    if len(op.operands) != 2:
        return None
    predicate = _CMPI_PREDICATES.get(int(op.attrs.get("predicate")))
    if predicate is None:
        return None

    lhs_id, rhs_id = op.operands
    lhs_const = _const_int_value(state["values"], lhs_id)
    rhs_const = _const_int_value(state["values"], rhs_id)
    if predicate == "eq":
        if rhs_const == 0:
            fact = _divisibility_fact_from_remainder(state, lhs_id)
            if fact is not None:
                return fact
        if lhs_const == 0:
            fact = _divisibility_fact_from_remainder(state, rhs_id)
            if fact is not None:
                return fact
    if rhs_const is not None:
        return _range_fact_from_value_const(state, lhs_id, predicate, rhs_const)
    if lhs_const is not None:
        inverted = _invert_cmpi_predicate(predicate)
        if inverted is not None:
            return _range_fact_from_value_const(state, rhs_id, inverted, lhs_const)
    return None


def _assume_facts_for_value(state, value_id):
    op = state["op_by_result"].get(value_id)
    if op is None:
        return ()
    if op.name == "arith.andi":
        facts = []
        for operand_id in op.operands:
            facts.extend(_assume_facts_for_value(state, operand_id))
        return tuple(facts)
    if op.name == "arith.cmpi":
        fact = _assume_fact_from_cmp(op, state)
        return (fact,) if fact is not None else ()
    return ()


def _emit_assume_fact(builder, state, fact, w):
    lowered = state["wave_values"].get(fact.value_id)
    if (
        not isinstance(lowered, _WaveValue)
        or lowered.kind != "index_expr"
        or _is_deferred_index(lowered.value)
    ):
        return
    value = _materialize_index_value(builder, lowered.value, {}, w)
    if fact.kind == "range":
        x = w.sym_ctx.sym("x")
        assumptions = []
        if fact.lower is not None:
            assumptions.append(x >= fact.lower)
        if fact.upper is not None:
            assumptions.append(x <= fact.upper)
        if not assumptions:
            return
        value = builder.assume(value, assumptions, name="x")
    elif fact.kind == "divisible" and fact.divisor is not None:
        value = builder.assume_divisible(value, fact.divisor)
    else:
        return
    _set_wave_value(state["wave_values"], fact.value_id, "index_expr", value)


def _emit_assume_op(builder, op, state, w):
    for operand_id in op.operands:
        for fact in _assume_facts_for_value(state, operand_id):
            state["assume_facts"].append(fact)
            _emit_assume_fact(builder, state, fact, w)


def _is_assume_tree_helper_op(state, op):
    if not op.results:
        return False
    return op.name in _ASSUME_TREE_OPS and all(
        result_id in state["assume_only_values"] for result_id in op.results
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


def _blocked_layout_static_coord(layout, shape, thread, component):
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
    component_shape = _blocked_layout_component_shape(
        shape,
        layout,
        "ttg.convert_layout tensor shape",
        "SIMD layout conversion",
    )
    component_coords = _delinearize_index(
        component,
        component_shape,
        layout.order,
    )
    coords = []
    for dim in range(len(layout.size_per_thread)):
        size_per_thread = int(layout.size_per_thread[dim])
        covered = _blocked_layout_dim_coverage(
            layout,
            dim,
            "ttg.convert_layout tensor shape",
            "SIMD layout conversion",
        )
        local_component = component_coords[dim] % size_per_thread
        repeat_component = component_coords[dim] // size_per_thread
        coords.append(
            local_component
            + size_per_thread
            * (lane_coords[dim] + layout.threads_per_warp[dim] * warp_coords[dim])
            + covered * repeat_component
        )
    return tuple(coords)


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

    source_components = _blocked_layout_component_count(
        source_plan,
        "ttg.convert_layout source",
        "SIMD layout conversion",
    )
    result_components = _blocked_layout_component_count(
        result_plan,
        "ttg.convert_layout result",
        "SIMD layout conversion",
    )
    permutation = []
    for result_component in range(result_components):
        source_component = None
        for candidate in range(source_components):
            matches = True
            for thread in range(result_threads):
                result_coord = _blocked_layout_static_coord(
                    result_layout, result_plan.shape, thread, result_component
                )
                if any(
                    result_coord[dim] >= result_plan.shape[dim]
                    for dim in range(len(result_plan.shape))
                ):
                    continue
                source_coord = _blocked_layout_static_coord(
                    source_layout, source_plan.shape, thread, candidate
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
    if (
        _product(layout.size_per_thread) != frag.registers
        and value_plan.element_type == "f32"
        and value_plan.shape == _GFX950_MMA_SHAPE
        and frag.registers == _product(_GFX950_DOT_PARENT_LAYOUT.size_per_thread)
        and _same_blocked_encoding(layout, _GFX950_PROPAGATED_STORE_LAYOUT)
    ):
        layout = _GFX950_DOT_PARENT_LAYOUT
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


def _is_supported_fragment_store_layout(layout):
    return any(
        _same_blocked_encoding(layout, supported)
        for supported in (
            _GFX950_DOT_PARENT_LAYOUT,
            _GFX950_PROPAGATED_STORE_LAYOUT,
            _GFX950_PIPELINED_STORE_LAYOUT,
        )
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
        _dot_parent_encoding_info(parent_attr, value.encoding, f"{context} parent"),
    )


def _same_dot_operand_encoding(lhs, rhs):
    return (
        lhs.op_idx == rhs.op_idx
        and lhs.k_width == rhs.k_width
        and _same_dot_parent_encoding(lhs.parent, rhs.parent)
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

def _dma_packet_layout_supported(address, memdesc):
    padded = _validate_padded_shared_layout(
        memdesc,
        "ttg.async_copy_global_to_local destination",
    )
    if padded is not None:
        return (
            memdesc.element_type == "f16"
            and memdesc.shape == _GFX950_MMA_SHAPE
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
        memdesc.element_type == "f16"
        and memdesc.shape == _GFX950_MMA_SHAPE
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
    layout = _blocked_tensor_layout_info(
        address_plan,
        "ttg.async_copy_global_to_local source",
        "async copy DMA packet lowering",
    )
    width = _product(layout.threads_per_warp)
    rank = len(address_plan.shape)
    inner_dim = rank - 1
    if int(address_plan.shape[inner_dim]) % packet_elements:
        raise ValueError(
            f"{context}: innermost tensor extent {address_plan.shape[inner_dim]} "
            f"is not divisible by {packet_elements} element DMA packets"
        )
    _require_dma_destination_physical_packets(
        memdesc,
        packet_elements,
        packet_bytes,
        width,
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
    width = _product(layout.threads_per_warp)
    cta_threads = width * _product(layout.warps_per_cta)
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
    component_count = _dma_packet_component_count(
        value_plan, layout, packet_elements
    )
    if component < 0 or component >= component_count:
        raise ValueError(
            "tlx_wave bridge async copy DMA packet component "
            f"{component} is out of range for {component_count} packet "
            "component(s)"
        )
    width = _product(layout.threads_per_warp)
    cta_threads = width * _product(layout.warps_per_cta)
    total_packets = _product(value_plan.shape) // int(packet_elements)
    thread = builder.workitem_id(axis=0, width=width)
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
    width = _product(layout.threads_per_warp)
    cta_threads = width * _product(layout.warps_per_cta)
    thread_first = builder.read_first(thread)
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
        dim_bindings[_dim_symbol(w, dim)] = builder.index_expr(
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
    return str(expr) == "True"


def _ixsimpl_is_false(expr):
    return str(expr) == "False"


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


def _ixsimpl_index_expr(source, w, unknowns, opaque_dim=None):
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
    if isinstance(source, (_IndexBinary, _IndexSelectCompare)):
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
        if str(source.kind) not in {"remsi", "remui"}:
            return None
        if _index_source_depends_on_dim(source.rhs, inner_dim):
            return None
        lhs = _ixsimpl_index_packet_expr(
            source.lhs, w, unknowns, inner_dim, packet_elements, assumptions
        )
        rhs = _ixsimpl_index_packet_expr(
            source.rhs, w, unknowns, inner_dim, packet_elements, assumptions
        )
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


def _ixsimpl_assume_fact_exprs(state, unknowns, w):
    assumptions = []
    for fact in state.get("assume_facts", ()):
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
    assumptions = (
        _dma_packet_dim_assumptions(shape, inner_dim, packet_elements, w)
        + _ixsimpl_assume_fact_exprs(state, unknowns, w)
    )
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
    assumptions = (
        _dma_packet_dim_assumptions(shape, inner_dim, packet_elements, w)
        + _ixsimpl_assume_fact_exprs(state, unknowns, w)
    )
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


def _ixsimpl_packet_slt_uniform(
    lhs, rhs, inner_dim, packet_elements, assumptions, w
):
    shifted_lhs = _ixsimpl_shift_dim(lhs, inner_dim, 1, w)
    shifted_rhs = _ixsimpl_shift_dim(rhs, inner_dim, 1, w)
    if not _ixsimpl_expr_equal(
        shifted_lhs, _ixsimpl_add_const(lhs, 1, w), assumptions, w
    ):
        return False
    if not _ixsimpl_expr_equal(shifted_rhs, rhs, assumptions, w):
        return False
    return _ixsimpl_mod_zero(lhs, packet_elements, assumptions, w) and _ixsimpl_mod_zero(
        rhs, packet_elements, assumptions, w
    )


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
    pointer_source = _require_lowered_value(
        state["wave_values"],
        address.address_value_id,
        "pointer_expr",
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
    source = _materialize_pointer_value(
        builder,
        pointer_source,
        dim_bindings,
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
    pointer_source = _require_lowered_value(
        state["wave_values"],
        address.address_value_id,
        "pointer_expr",
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
    pointer_source = _require_lowered_value(
        state["wave_values"],
        address.address_value_id,
        "pointer_expr",
        "ttg.async_copy_global_to_local source",
    )
    component_count = _blocked_layout_component_count(
        address_plan,
        "ttg.async_copy_global_to_local source",
        "generic async copy lowering",
    )
    token = after_token
    for component in range(component_count):
        dim_bindings, width, active = _blocked_tensor_dim_bindings(
            builder,
            address_plan,
            w,
            "ttg.async_copy_global_to_local source",
            component=component,
        )
        source = _materialize_pointer_value(
            builder,
            pointer_source,
            dim_bindings,
            w,
        )
        mask = active
        if mask_value is not None:
            user_mask = _materialize_mask_value(
                builder,
                mask_value,
                dim_bindings,
                w,
                width,
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
        fallback = _zero_simd_for_element(
            builder,
            address.element_type,
            width,
            w,
            "ttg.async_copy_global_to_local inactive lanes",
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
        token = _emit_component_store(
            builder,
            loaded,
            destination,
            mask,
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
    if address.element_type != memdesc.element_type:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.async_copy_global_to_local: "
            f"source element type {address.element_type} does not match "
            f"destination memdesc element type {memdesc.element_type}"
        )
    mask_value = None
    if address.mask_value_id is not None:
        mask_value = _require_lowered_value(
            state["wave_values"],
            address.mask_value_id,
            "mask_expr",
            "ttg.async_copy_global_to_local mask",
        )
    try:
        dma_bytes, dma_packet_layout, copy_component_count = _select_dma_packet_lowering(
            state,
            address,
            address_plan,
            memdesc,
            memdescs,
            lds_layout,
            mask_value,
            w,
        )
    except _DmaDestinationNotWholeWaveContiguous:
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
            user_mask = _materialize_mask_value(
                builder,
                mask_value,
                dim_bindings,
                w,
                width,
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


def _join_tokens(builder, tokens, stats):
    if not tokens:
        return builder.token()
    stats.joins += 1
    return builder.join(*tokens)


def _mem_state(state):
    return state["mem_state"]


def _set_mem_state(state, mem_state):
    state["mem_state"] = mem_state


def _copy_mem_state(mem_state):
    return _MemState(
        mem_state.root_token,
        tuple(mem_state.open_async_tokens),
        tuple(mem_state.committed_groups),
        mem_state.committed_group_capacity,
    )


def _mem_root(state):
    return _mem_state(state).root_token


def _ensure_mem_root(builder, state):
    mem_state = _mem_state(state)
    if mem_state.root_token is not None:
        return mem_state.root_token
    token = builder.token()
    _set_mem_state(
        state,
        _MemState(
            token,
            mem_state.open_async_tokens,
            mem_state.committed_groups,
            mem_state.committed_group_capacity,
        ),
    )
    return token


def _set_mem_root(state, token):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            token,
            mem_state.open_async_tokens,
            mem_state.committed_groups,
            mem_state.committed_group_capacity,
        ),
    )


def _append_open_async_token(state, token):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            mem_state.open_async_tokens + (token,),
            mem_state.committed_groups,
            mem_state.committed_group_capacity,
        ),
    )


def _clear_open_async_tokens(state):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            (),
            mem_state.committed_groups,
            mem_state.committed_group_capacity,
        ),
    )


def _remove_token_values(values, removed):
    removed_ids = {id(value) for value in removed}
    if not removed_ids:
        return tuple(values)
    return tuple(value for value in values if id(value) not in removed_ids)


def _remove_open_async_tokens(state, tokens):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            _remove_token_values(mem_state.open_async_tokens, tokens),
            mem_state.committed_groups,
            mem_state.committed_group_capacity,
        ),
    )


def _append_committed_group(state, token):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            mem_state.open_async_tokens,
            mem_state.committed_groups + (token,),
            mem_state.committed_group_capacity,
        ),
    )


def _set_committed_groups(state, groups):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            mem_state.open_async_tokens,
            tuple(groups),
            mem_state.committed_group_capacity,
        ),
    )


def _remove_committed_groups(state, groups):
    mem_state = _mem_state(state)
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            mem_state.open_async_tokens,
            _remove_token_values(mem_state.committed_groups, groups),
            mem_state.committed_group_capacity,
        ),
    )


def _cap_committed_groups(state):
    mem_state = _mem_state(state)
    capacity = mem_state.committed_group_capacity
    if capacity is None or len(mem_state.committed_groups) <= capacity:
        return
    _set_mem_state(
        state,
        _MemState(
            mem_state.root_token,
            mem_state.open_async_tokens,
            mem_state.committed_groups[-capacity:] if capacity else (),
            capacity,
        ),
    )


def _open_async_tokens(state):
    return _mem_state(state).open_async_tokens


def _committed_groups(state):
    return _mem_state(state).committed_groups


def _initial_lowering_state(builder, kernel, plan, w):
    values = _values_by_id(plan)
    _, assume_only_values = _assume_tree_info(plan)
    state = {
        "values": values,
        "memdescs": _memdescs_by_id(plan),
        "op_by_result": _op_by_result_id(plan),
        "assume_only_values": assume_only_values,
        "wave_values": {},
        "program_id_bindings": {},
        "mem_state": _MemState(),
        "assume_facts": [],
    }
    _init_argument_wave_values(builder, kernel, values, state, w)
    return state


def _emit_generic_value_op(builder, state, op, w):
    values = state["values"]
    wave_values = state["wave_values"]
    if _is_assume_tree_helper_op(state, op):
        return True
    if op.name == "arith.constant":
        _emit_constant_op(builder, op, values, wave_values, w, state.get("stats"))
    elif op.name == "tt.get_program_id":
        _emit_program_id_op(builder, state, op, values, wave_values, w)
    elif op.name == "tt.make_range":
        _emit_make_range_op(op, values, wave_values, w)
    elif op.name in {"tt.broadcast", "tt.splat"}:
        _emit_splat_or_broadcast_op(builder, op, values, wave_values, w)
    elif op.name == "tt.expand_dims":
        _forward_lowered_value(op, values, wave_values)
    elif op.name in {
        "arith.addi",
        "arith.divsi",
        "arith.divui",
        "arith.maxsi",
        "arith.maxui",
        "arith.minsi",
        "arith.minui",
        "arith.muli",
        "arith.remsi",
        "arith.remui",
        "arith.subi",
    }:
        _emit_arith_binary_op(builder, op, values, wave_values, w)
    elif op.name in {"arith.extf", "arith.truncf"}:
        _emit_float_cast_op(builder, op, values, wave_values, w)
    elif op.name == "arith.cmpi":
        _emit_typed_cmp_op(builder, op, values, wave_values, w)
    elif op.name == "arith.andi":
        _emit_typed_and_op(builder, op, values, wave_values, w)
    elif op.name == "tt.addptr":
        _emit_addptr_op(builder, op, values, wave_values, w)
    elif op.name in {
        "tlx.local_alias",
        "tlx.release_layout",
        "tlx.require_layout",
        "ttg.convert_layout",
    }:
        _forward_lowered_value(op, values, wave_values)
    else:
        return False
    return True


_PLANNING_ONLY_OPS = {
    "rocdl.sched.barrier",
    "rocdl.sched.group.barrier",
    "rocdl.setprio",
    "tt.return",
    "tlx.reuse_group",
    "tlx.set_buffer_overlap",
    "tlx.storage_alias_local_alloc",
    "tlx.storage_alias_spec",
    "ttg.local_alloc",
    "ttg.memdesc_index",
    "ttg.memdesc_subslice",
    "ttg.memdesc_reinterpret",
    "ttg.memdesc_reshape",
    "ttg.memdesc_trans",
}


_CONTROL_REGION_EFFECT_OPS = {
    "tt.dot",
    "tt.load",
    "tt.store",
    "ttg.async_commit_group",
    "ttg.async_copy_global_to_local",
    "ttg.async_wait",
    "ttg.local_load",
    "ttg.local_store",
}


def _save_control_scope(state):
    return (
        state["wave_values"],
        state["program_id_bindings"],
        state["mem_state"],
        state["assume_facts"],
    )


def _enter_control_scope(state):
    saved = _save_control_scope(state)
    state["wave_values"] = dict(state["wave_values"])
    state["program_id_bindings"] = dict(state["program_id_bindings"])
    state["assume_facts"] = list(state["assume_facts"])
    return saved


def _restore_control_scope(state, saved):
    (
        state["wave_values"],
        state["program_id_bindings"],
        state["mem_state"],
        state["assume_facts"],
    ) = saved


def _control_kind_for_value_plan(value_plan, context):
    if value_plan.type_kind == "token":
        return "token"
    if value_plan.type_kind == "scalar" and _is_bool_value(value_plan):
        return "scalar"
    if value_plan.type_kind == "scalar" and _is_integer_or_index_value(value_plan):
        return "index_expr"
    if value_plan.type_kind == "scalar" and _scalar_data_type(value_plan) is not None:
        return "scalar"
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported control-flow "
        f"value type {value_plan.type}"
    )


def _control_kind_for_lowered_value(lowered, value_plan, context):
    if (
        value_plan.type_kind == "tensor"
        and isinstance(lowered, _WaveValue)
        and lowered.kind in {"fragment", "fragment_regs", "simd", "simd_tuple"}
    ):
        return lowered.kind
    return _control_kind_for_value_plan(value_plan, context)


def _control_result_type(value_plan, w, context):
    kind = _control_kind_for_value_plan(value_plan, context)
    if kind == "token":
        return w.mem_token_type()
    if kind == "index_expr":
        return w.index_type()
    return _wave_element_type(value_plan.type, w, context)


def _validate_control_value_kind(lowered, expected_kind, context):
    if not isinstance(lowered, _WaveValue):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected typed Wave "
            f"value, got {type(lowered).__name__}"
        )
    if lowered.kind != expected_kind:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: yielded value lowered as "
            f"{lowered.kind}, expected {expected_kind}"
        )


def _control_value_to_mlir(builder, lowered, w, context):
    if not isinstance(lowered, _WaveValue):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected typed Wave "
            f"value, got {type(lowered).__name__}"
        )
    if lowered.kind == "index_expr":
        if _is_deferred_index(lowered.value):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: layout-dependent "
                "index expression cannot be used as a control-flow value"
            )
        return _materialize_index_value(builder, lowered.value, {}, w)
    if lowered.kind in {"scalar", "token"}:
        return _scalar_mlir_value(lowered.value)
    if lowered.kind in {"fragment", "fragment_regs", "simd"}:
        return lowered.value
    raise ValueError(
        f"tlx_wave bridge cannot lower {context}: unsupported control-flow "
        f"value lowered as {lowered.kind}"
    )


def _control_value_to_mlir_values(builder, lowered, w, context):
    if isinstance(lowered, _WaveValue) and lowered.kind == "simd_tuple":
        values = tuple(lowered.value)
    else:
        values = (_control_value_to_mlir(builder, lowered, w, context),)
    fragment_regs = _materialize_fragment_regs_aux(lowered)
    if fragment_regs is not None:
        values = values + (fragment_regs,)
    return values


def _control_primary_count(kind, value):
    if kind == "simd_tuple":
        if not isinstance(value, tuple):
            raise ValueError("tlx_wave bridge internal error: simd_tuple needs tuple")
        return len(value)
    return 1


def _set_control_result(wave_values, value_id, kind, value, aux=None):
    if kind not in {
        "fragment",
        "fragment_regs",
        "index_expr",
        "scalar",
        "simd",
        "simd_tuple",
        "token",
    }:
        raise ValueError(
            f"tlx_wave bridge internal error: unsupported SCF result kind {kind}"
        )
    if kind == "simd_tuple":
        if not isinstance(value, tuple):
            raise ValueError("tlx_wave bridge internal error: simd_tuple needs tuple")
        wave_values[value_id] = _WaveValue(kind, value, None, aux)
        return
    wave_values[value_id] = _WaveValue(kind, value, None, aux)


def _control_info_for_lowered(kind, lowered):
    primary_count = _control_primary_count(kind, lowered.value)
    return {
        "kind": kind,
        "count": primary_count + (1 if _has_fragment_regs_aux(lowered) else 0),
        "primary_count": primary_count,
        "has_fragment_regs": _has_fragment_regs_aux(lowered),
    }


def _control_result_from_mlir_values(wave_values, value_id, info, values):
    kind = info["kind"] if isinstance(info, dict) else info
    primary_count = (
        info.get("primary_count", len(values)) if isinstance(info, dict) else len(values)
    )
    has_fragment_regs = (
        info.get("has_fragment_regs", False) if isinstance(info, dict) else False
    )
    primary_values = tuple(values[:primary_count])
    aux_values = tuple(values[primary_count:])
    aux = None
    if has_fragment_regs:
        if len(aux_values) != 1:
            raise ValueError(
                "tlx_wave bridge internal error: expected one fragment register "
                f"auxiliary value, got {len(aux_values)}"
            )
        aux = {"fragment_regs": aux_values[0]}
    if kind == "simd_tuple":
        _set_control_result(wave_values, value_id, kind, primary_values, aux)
        return
    if len(primary_values) != 1:
        raise ValueError(
            f"tlx_wave bridge internal error: SCF kind {kind} expected one "
            f"value, got {len(primary_values)}"
        )
    _set_control_result(wave_values, value_id, kind, primary_values[0], aux)


def _control_value_from_mlir_values(value_id, info, values):
    temp = {}
    _control_result_from_mlir_values(temp, value_id, info, values)
    lowered = temp[value_id]
    if info["kind"] == "fragment":
        return _WaveValue(
            lowered.kind,
            lowered.value,
            value_id,
            lowered.aux,
        )
    return lowered


def _control_condition_value(builder, state, op, w):
    if len(op.operands) != 1:
        raise ValueError("tlx_wave bridge expected scf.if with one condition")
    lowered = _require_typed_wave_value(
        state["wave_values"], op.operands[0], "scf.if condition"
    )
    if lowered.kind == "scalar":
        value_plan = state["values"][op.operands[0]]
        if not _is_bool_value(value_plan):
            raise ValueError(
                "tlx_wave bridge cannot lower scf.if condition: expected "
                f"scalar i1, got {value_plan.type}"
            )
        return _scalar_mlir_value(lowered.value)
    if lowered.kind == "mask_expr" and isinstance(lowered.value, _MaskConst):
        return _constant_value(
            builder,
            "i1",
            lowered.value.value,
            w,
            "scf.if constant condition",
        )
    raise ValueError(
        "tlx_wave bridge cannot lower scf.if condition: expected scalar i1, "
        f"got lowered {lowered.kind}"
    )


def _control_yields_for_result_ids(
    builder, state, yielded, result_ids, w, context, expected_infos=None
):
    if yielded is None:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: missing scf.yield"
        )
    if len(yielded) != len(result_ids):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: scf.yield has "
            f"{len(yielded)} value(s), expected {len(result_ids)}"
        )
    materialized = []
    infos = []
    for index, (lowered, result_id) in enumerate(zip(yielded, result_ids)):
        value_plan = state["values"][result_id]
        expected_info = expected_infos[index] if expected_infos is not None else None
        expected_kind = (
            expected_info["kind"]
            if expected_info is not None
            else _control_kind_for_value_plan(value_plan, f"{context} result {index}")
        )
        _validate_control_value_kind(
            lowered, expected_kind, f"{context} result {index}"
        )
        values = _control_value_to_mlir_values(
            builder,
            lowered,
            w,
            f"{context} result {index}",
        )
        if expected_info is not None and len(values) != expected_info["count"]:
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: result {index} "
                f"materialized to {len(values)} value(s), expected "
                f"{expected_info['count']}"
            )
        materialized.extend(values)
        infos.append(_control_info_for_lowered(expected_kind, lowered))
    return tuple(materialized), tuple(infos)


def _emit_scoped_control_block(
    builder,
    kernel,
    raw_ops,
    state,
    lds_layout,
    w,
    stats,
    control_context,
    bindings=(),
    mem_state=None,
    capture_mem_state=False,
):
    saved = _enter_control_scope(state)
    try:
        if mem_state is not None:
            state["mem_state"] = mem_state
        for value_id, lowered in bindings:
            state["wave_values"][value_id] = lowered
        yielded = _emit_raw_block(
            builder,
            kernel,
            raw_ops,
            state,
            lds_layout,
            w,
            stats,
            control_context=control_context,
        )
        if capture_mem_state:
            return yielded, _copy_mem_state(state["mem_state"])
        return yielded
    finally:
        _restore_control_scope(state, saved)


def _emit_scf_if_op(builder, kernel, raw_op, op, state, lds_layout, w, stats):
    condition = _control_condition_value(builder, state, op, w)
    result_types = tuple(
        _control_result_type(
            state["values"][result_id],
            w,
            f"scf.if result {index}",
        )
        for index, result_id in enumerate(op.results)
    )
    has_else = raw_op.get_num_regions() > 1 and not raw_op.get_region(1).empty()
    if result_types and not has_else:
        raise ValueError(
            "tlx_wave bridge cannot lower result-bearing scf.if without an "
            "else region"
        )
    then_block = _single_region_block(raw_op, 0, "scf.if then")
    else_block = (
        _single_region_block(raw_op, 1, "scf.if else") if has_else else None
    )
    result_infos = ()
    if_builder_ref = None
    with builder.if_(condition, result_types, otherwise=has_else) as if_builder:
        if_builder_ref = if_builder
        then_yielded = _emit_scoped_control_block(
            builder,
            kernel,
            _raw_block_ops(then_block),
            state,
            lds_layout,
            w,
            stats,
            "scf.if",
        )
        if result_types:
            then_values, result_infos = _control_yields_for_result_ids(
                builder, state, then_yielded, op.results, w, "scf.if then"
            )
            builder.yield_(then_values)
        if has_else:
            with if_builder.otherwise():
                else_yielded = _emit_scoped_control_block(
                    builder,
                    kernel,
                    _raw_block_ops(else_block),
                    state,
                    lds_layout,
                    w,
                    stats,
                    "scf.if",
                )
                if result_types:
                    else_values, else_infos = _control_yields_for_result_ids(
                        builder,
                        state,
                        else_yielded,
                        op.results,
                        w,
                        "scf.if else",
                    )
                    if else_infos != result_infos:
                        raise ValueError(
                            "tlx_wave bridge cannot lower scf.if: then/else "
                            f"yield kinds differ ({result_infos} vs {else_infos})"
                        )
                    builder.yield_(else_values)
    if result_types:
        for result_id, result_info, result in zip(
            op.results, result_infos, if_builder_ref.op.results
        ):
            _control_result_from_mlir_values(
                state["wave_values"], result_id, result_info, (result,)
            )


def _materialize_for_bound(builder, state, value_id, w, context):
    lowered = _require_typed_wave_value(state["wave_values"], value_id, context)
    if lowered.kind != "index_expr":
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: expected index_expr, "
            f"got {lowered.kind}"
        )
    return _control_value_to_mlir(builder, lowered, w, context)


def _walk_raw_region_ops(raw_ops):
    for raw_op in raw_ops:
        yield raw_op
        for region_index in range(raw_op.get_num_regions()):
            region = raw_op.get_region(region_index)
            for block_index in range(region.size()):
                yield from _walk_raw_region_ops(
                    _raw_block_ops(region.get_block(block_index))
                )


def _raw_ops_have_memory_effects(raw_ops):
    return any(
        raw_op.get_name() in _CONTROL_REGION_EFFECT_OPS
        for raw_op in _walk_raw_region_ops(raw_ops)
    )


def _raw_op_has_memory_effects(raw_op):
    return _raw_ops_have_memory_effects((raw_op,))


def _max_async_wait_keep(raw_ops):
    keep = 0
    for raw_op in _walk_raw_region_ops(raw_ops):
        if raw_op.get_name() == "ttg.async_wait":
            keep = max(keep, int(raw_op.get_int_attr("num") or 0))
    return keep


def _async_commit_group_count(raw_ops):
    return sum(
        1
        for raw_op in _walk_raw_region_ops(raw_ops)
        if raw_op.get_name() == "ttg.async_commit_group"
    )


def _loop_hidden_mem_init(builder, state, raw_ops):
    if not _raw_ops_have_memory_effects(raw_ops):
        return (), None
    group_count = max(
        len(_committed_groups(state)),
        _max_async_wait_keep(raw_ops) + _async_commit_group_count(raw_ops),
    )
    groups = tuple(_committed_groups(state))
    if len(groups) < group_count:
        groups = (
            tuple(builder.token() for _ in range(group_count - len(groups)))
            + groups
        )
    elif len(groups) > group_count:
        groups = groups[-group_count:]
    return (_ensure_mem_root(builder, state),) + groups, _LoopMemShape(group_count)


def _loop_mem_state_from_iter_args(iter_args, shape):
    if shape is None:
        return None
    if len(iter_args) != 1 + shape.group_count:
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: hidden memory state has "
            f"{len(iter_args)} iter arg(s), expected {1 + shape.group_count}"
        )
    return _MemState(iter_args[0], (), tuple(iter_args[1:]), shape.group_count)


def _loop_hidden_mem_yields(body_mem_state, shape):
    if shape is None:
        return ()
    if body_mem_state is None:
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: missing loop memory state"
        )
    if body_mem_state.open_async_tokens:
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for with uncommitted async copies "
            "live across the loop backedge"
        )
    if len(body_mem_state.committed_groups) != shape.group_count:
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: async committed-group queue "
            "changes shape across the loop backedge; add an async_wait that "
            "keeps a static number of groups"
        )
    if body_mem_state.root_token is None:
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: missing hidden root memory token"
        )
    return (body_mem_state.root_token,) + tuple(body_mem_state.committed_groups)


def _apply_loop_hidden_mem_results(state, results, shape):
    if shape is None:
        return
    if len(results) != 1 + shape.group_count:
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: hidden memory result count "
            f"{len(results)} does not match shape {1 + shape.group_count}"
        )
    _set_mem_state(state, _MemState(results[0], (), tuple(results[1:])))


def _emit_scf_for_op(builder, kernel, raw_op, op, state, lds_layout, w, stats):
    if len(op.operands) < 3:
        raise ValueError("tlx_wave bridge expected scf.for with at least 3 operands")
    init_ids = op.operands[3:]
    if len(init_ids) != len(op.results):
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: iter_args/result count "
            f"mismatch ({len(init_ids)} init arg(s), {len(op.results)} result(s))"
        )
    lower = _materialize_for_bound(builder, state, op.operands[0], w, "scf.for lower")
    upper = _materialize_for_bound(builder, state, op.operands[1], w, "scf.for upper")
    step = _materialize_for_bound(builder, state, op.operands[2], w, "scf.for step")
    user_init_values = []
    carry_infos = []
    for index, (init_id, result_id) in enumerate(zip(init_ids, op.results)):
        value_plan = state["values"][result_id]
        lowered = _require_typed_wave_value(
            state["wave_values"], init_id, f"scf.for iter_arg {index}"
        )
        expected_kind = _control_kind_for_lowered_value(
            lowered, value_plan, f"unsupported loop-carried scf.for value {index}"
        )
        _validate_control_value_kind(
            lowered,
            expected_kind,
            f"unsupported loop-carried scf.for value {index}",
        )
        init_values = _control_value_to_mlir_values(
            builder,
            lowered,
            w,
            f"scf.for iter_arg {index}",
        )
        user_init_values.extend(init_values)
        carry_infos.append(_control_info_for_lowered(expected_kind, lowered))

    body_block = _single_region_block(raw_op, 0, "scf.for body")
    body_ops = _raw_block_ops(body_block)
    body_args = _raw_block_args(body_block)
    if len(body_args) != 1 + len(init_ids):
        raise ValueError(
            "tlx_wave bridge cannot lower scf.for: body argument count "
            f"{len(body_args)} does not match induction plus {len(init_ids)} "
            "iter_arg(s)"
        )

    hidden_init_values, mem_shape = _loop_hidden_mem_init(builder, state, body_ops)
    all_init_values = tuple(user_init_values) + tuple(hidden_init_values)

    if all_init_values:
        with builder.for_loop(lower, upper, step, all_init_values) as for_op:
            iter_args = tuple(for_op.inner_iter_args)
            if len(iter_args) != len(all_init_values):
                raise ValueError(
                    "tlx_wave bridge cannot lower scf.for: Wave loop exposes "
                    f"{len(iter_args)} iter arg(s), expected {len(all_init_values)}"
                )
            user_iter_args = iter_args[: len(user_init_values)]
            hidden_iter_args = iter_args[len(user_init_values) :]
            body_mem_state = _loop_mem_state_from_iter_args(
                hidden_iter_args, mem_shape
            )
            bindings = [
                (
                    _value_id(body_args[0]),
                    _WaveValue("index_expr", for_op.induction_variable),
                )
            ]
            iter_offset = 0
            for block_arg, info in zip(body_args[1:], carry_infos):
                iter_values = user_iter_args[iter_offset : iter_offset + info["count"]]
                bindings.append(
                    (
                        _value_id(block_arg),
                        _control_value_from_mlir_values(
                            _value_id(block_arg),
                            info,
                            tuple(iter_values),
                        ),
                    )
                )
                iter_offset += info["count"]
            scoped_result = _emit_scoped_control_block(
                builder,
                kernel,
                body_ops,
                state,
                lds_layout,
                w,
                stats,
                "scf.for",
                bindings=tuple(bindings),
                mem_state=body_mem_state,
                capture_mem_state=mem_shape is not None,
            )
            if mem_shape is not None:
                yielded, yielded_mem_state = scoped_result
            else:
                yielded = scoped_result
                yielded_mem_state = None
            if yielded is None and not op.results:
                yielded = ()
            yield_values, yield_infos = _control_yields_for_result_ids(
                builder,
                state,
                yielded,
                op.results,
                w,
                "scf.for",
                expected_infos=carry_infos,
            )
            if yield_infos != tuple(carry_infos):
                raise ValueError(
                    "tlx_wave bridge cannot lower scf.for: iter_arg/yield "
                    f"kinds differ ({tuple(carry_infos)} vs {yield_infos})"
                )
            hidden_yields = _loop_hidden_mem_yields(yielded_mem_state, mem_shape)
            builder.yield_(yield_values + hidden_yields)
        result_offset = 0
        for result_id, info in zip(op.results, carry_infos):
            result_values = tuple(
                for_op.results[result_offset : result_offset + info["count"]]
            )
            _control_result_from_mlir_values(
                state["wave_values"], result_id, info, result_values
            )
            result_offset += info["count"]
        _apply_loop_hidden_mem_results(
            state, for_op.results[len(user_init_values) :], mem_shape
        )
        return

    with builder.for_loop(lower, upper, step) as induction:
        bindings = (
            (
                _value_id(body_args[0]),
                _WaveValue("index_expr", induction),
            ),
        )
        yielded = _emit_scoped_control_block(
            builder,
            kernel,
            body_ops,
            state,
            lds_layout,
            w,
            stats,
            "scf.for",
            bindings=bindings,
        )
        if yielded not in (None, ()):
            raise ValueError(
                "tlx_wave bridge cannot lower scf.for without iter_args: "
                f"body yielded {len(yielded)} value(s)"
            )


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


def _unsupported_fragment_local_load(value, reason, memdesc=None):
    message = (
        "tlx_wave bridge cannot pack ttg.local_load as a WaveAMD fragment: "
        f"{reason}. original TTGIR encoding: {value.encoding}"
    )
    if memdesc is not None:
        message += f"; memdesc encoding: {memdesc.encoding}"
    raise ValueError(message)


def _validate_supported_dot_local_load_layout(value, memdesc, info):
    if isinstance(info.parent, _BlockedEncodingInfo):
        if not _same_blocked_encoding(info.parent, _GFX950_DOT_PARENT_LAYOUT):
            _unsupported_fragment_local_load(
                value,
                "current flat gfx950 fragment loader supports only dot operand "
                "parent layout sizePerThread=(2, 2), threadsPerWarp=(4, 16), "
                "warpsPerCTA=(4, 1), order=(1, 0); "
                f"got sizePerThread={info.parent.size_per_thread}, "
                f"threadsPerWarp={info.parent.threads_per_warp}, "
                f"warpsPerCTA={info.parent.warps_per_cta}, order={info.parent.order}",
                memdesc,
            )
    elif isinstance(info.parent, _AMDMfmaEncodingInfo):
        if not (
            info.parent.version == 4
            and info.parent.warps_per_cta == (2, 2)
            and info.parent.instr_shape == (16, 16, 32)
            and info.parent.is_transposed
        ):
            _unsupported_fragment_local_load(
                value,
                "current flat gfx950 fragment loader supports only "
                "#ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], "
                "instrShape = [16, 16, 32], isTransposed = true}>; "
                f"got version={info.parent.version}, "
                f"warpsPerCTA={info.parent.warps_per_cta}, "
                f"instrShape={info.parent.instr_shape}, "
                f"isTransposed={info.parent.is_transposed}",
                memdesc,
            )
    else:
        _unsupported_fragment_local_load(
            value,
            f"unsupported dot operand parent layout {info.parent}",
            memdesc,
        )
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, "ttg.local_load memdesc"
        )
    except ValueError as exc:
        padded = _validate_padded_shared_layout(memdesc, "ttg.local_load memdesc")
        if padded is not None:
            return
        _unsupported_fragment_local_load(value, str(exc), memdesc)
    if not _is_supported_swizzled_shared_layout(memdesc, shared):
        _unsupported_fragment_local_load(
            value,
            "current flat gfx950 fragment loader supports only "
            "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, "
            "order = [1, 0]}> or a supported #ttg.padded_shared layout "
            "for f16 32x32 dot operands; "
            f"got vec={shared.vec}, perPhase={shared.per_phase}, "
            f"maxPhase={shared.max_phase}, order={shared.order}",
            memdesc,
        )


def _validate_dot_operand_fragment_load(value, memdesc, info):
    if value.element_type != "f16" or value.element_byte_width != 2:
        _unsupported_fragment_local_load(
            value, f"expected f16 dot operand, got {value.element_type}", memdesc
        )
    if value.shape != _GFX950_MMA_SHAPE:
        _unsupported_fragment_local_load(
            value, f"expected static 32x32 operand shape, got {value.shape}", memdesc
        )
    if info.op_idx not in (0, 1):
        _unsupported_fragment_local_load(
            value, f"expected opIdx 0/1, got {info.op_idx}", memdesc
        )
    if info.k_width not in (0, 4, 8, 32):
        _unsupported_fragment_local_load(
            value,
            f"expected kWidth 0, 4, 8, or 32 for gfx950 f16 MFMA, got {info.k_width}",
            memdesc,
        )
    if memdesc.element_type != "f16" or memdesc.element_byte_width != 2:
        _unsupported_fragment_local_load(
            value,
            f"expected f16 shared memdesc source, got {memdesc.element_type}",
            memdesc,
        )
    if memdesc.shape != _GFX950_MMA_SHAPE:
        _unsupported_fragment_local_load(
            value,
            f"expected 32x32 shared memdesc source, got {memdesc.shape}",
            memdesc,
        )
    _validate_supported_dot_local_load_layout(value, memdesc, info)
    _require_fragment_load_physical_contiguous(value, memdesc)


def _physical_local_load_fragment_capability(value, memdesc):
    if value.encoding_attr is None or not _attr_bool(
        value.encoding_attr, "is_dot_operand_encoding"
    ):
        return None
    info = _dot_operand_encoding_info(value, "ttg.local_load result")
    _validate_dot_operand_fragment_load(value, memdesc, info)
    return _DotOperandFragmentLoad(info)


def _physical_local_load_fragment_regs_capability(value, memdesc):
    if value.encoding_attr is None or not _attr_bool(
        value.encoding_attr, "is_blocked_encoding"
    ):
        return None
    if value.element_type != "f16" or value.element_byte_width != 2:
        return None
    if value.shape != _GFX950_MMA_SHAPE:
        return None
    if memdesc.element_type != "f16" or memdesc.element_byte_width != 2:
        _unsupported_fragment_local_load(
            value,
            f"expected f16 shared memdesc source, got {memdesc.element_type}",
            memdesc,
        )
    if memdesc.shape != _GFX950_MMA_SHAPE:
        _unsupported_fragment_local_load(
            value,
            f"expected 32x32 shared memdesc source, got {memdesc.shape}",
            memdesc,
        )
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, "ttg.local_load memdesc"
        )
    except ValueError as exc:
        padded = _validate_padded_shared_layout(memdesc, "ttg.local_load memdesc")
        if padded is not None:
            _require_fragment_load_physical_contiguous(value, memdesc)
            return True
        _unsupported_fragment_local_load(value, str(exc), memdesc)
    if not _is_supported_swizzled_shared_layout(memdesc, shared):
        _unsupported_fragment_local_load(
            value,
            "current flat gfx950 fragment register loader supports only "
            "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, "
            "order = [1, 0]}> or a supported #ttg.padded_shared layout "
            "for f16 32x32 dot operands; "
            f"got vec={shared.vec}, perPhase={shared.per_phase}, "
            f"maxPhase={shared.max_phase}, order={shared.order}",
            memdesc,
        )
    _require_fragment_load_physical_contiguous(value, memdesc)
    return True


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
        _unsupported_fragment_local_load(value, str(exc), memdesc)


def _fragment_lane_dim_bindings(builder, value, memdesc, w):
    lane = builder.lane_id(width=_GFX950_MMA_WAVE)
    lane_sym = w.sym(f"tlx_local_load_{value.value_id}_lane")
    elements_per_lane = _GFX950_MMA_REGS * (4 // int(memdesc.element_byte_width))
    element_linear = lane_sym * int(elements_per_lane)
    row_major_order = tuple(reversed(range(len(memdesc.shape))))
    coords = _delinearize_expr(w, element_linear, memdesc.shape, row_major_order)
    return {
        _dim_symbol(w, dim): builder.index_expr(coords[dim], {lane_sym: lane})
        for dim in range(len(memdesc.shape))
    }


def _memdesc_needs_encoded_fragment_offset(memdesc):
    if _padded_shared_encoding_info(
        memdesc.encoding,
        "ttg.local_load fragment source",
    ) is not None:
        return True
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr,
            memdesc.encoding,
            "ttg.local_load fragment source",
        )
    except ValueError:
        return False
    return not _is_identity_shared_layout(memdesc, shared)


def _fragment_dense_i32_offset(builder, value, w):
    lane = builder.lane_id(width=_GFX950_MMA_WAVE)
    lane_sym = w.sym(f"tlx_local_load_{value.value_id}_lane_dense")
    return builder.index_expr(
        lane_sym * _GFX950_MMA_REGS,
        {lane_sym: lane},
    )


def _emit_fragment_i32_ptr(builder, value, memdesc, memdescs, lds_layout, state, w):
    base = _emit_memdesc_i32_ptr(
        builder, value, memdesc, memdescs, lds_layout, state, w
    )
    if _memdesc_needs_encoded_fragment_offset(memdesc):
        dim_bindings = _fragment_lane_dim_bindings(builder, value, memdesc, w)
        offset = _memdesc_pointer_offset_expr(
            memdesc,
            memdesc.shape,
            dim_bindings,
            4,
            w,
            "ttg.local_load fragment source",
        )
        offset = _materialize_index_value(builder, offset, {}, w)
    else:
        offset = _fragment_dense_i32_offset(builder, value, w)
    return builder.ptr_add(
        base,
        offset,
        w.simd_ptr_type(w.i32(), w.shared_address_space(), _GFX950_MMA_WAVE),
    )


def _emit_dot_operand_fragment_load(
    builder,
    address,
    value,
    memdesc,
    memdescs,
    capability,
    lds_layout,
    state,
    after_token,
    w,
    stats,
):
    ptr = _emit_fragment_i32_ptr(
        builder,
        value,
        memdesc,
        memdescs,
        lds_layout,
        state,
        w,
    )
    fragment, token = builder.fragment_load(
        ptr, _fragment_type_for_dot_operand(capability.info, w), after=after_token
    )
    stats.local_loads += 1
    stats.fragment_packs += 1
    return fragment, token


def _emit_dot_operand_register_load(
    builder,
    value,
    memdesc,
    memdescs,
    lds_layout,
    state,
    after_token,
    w,
    stats,
):
    ptr = _emit_fragment_i32_ptr(
        builder,
        value,
        memdesc,
        memdescs,
        lds_layout,
        state,
        w,
    )
    load_type = w.simd_type(
        w.vector_type(_GFX950_MMA_REGS, w.i32()),
        _GFX950_MMA_WAVE,
    )
    regs, token = builder.load(ptr, load_type, after=after_token)
    stats.local_loads += 1
    return regs, token


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
    if not _same_dot_parent_encoding(role_infos[0].parent, role_infos[1].parent):
        raise ValueError(
            "tlx_wave bridge expected matching tt.dot operand parent layouts; "
            f"role0 encoding: {role_values[0].encoding}; "
            f"role1 encoding: {role_values[1].encoding}"
        )

    result_layout = _dot_result_layout_info(
        result.encoding_attr, result.encoding, "tt.dot result"
    )
    if not _same_dot_parent_encoding(role_infos[0].parent, result_layout):
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
        acc_layout = _dot_result_layout_info(
            acc.encoding_attr, acc.encoding, "tt.dot accumulator"
        )
        if not _same_dot_parent_encoding(result_layout, acc_layout):
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


def _simd_splat_from_scalar(builder, scalar, result_plan, w, context):
    if not _is_data_tensor(result_plan):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context} as tensor data: "
            f"unsupported result type {result_plan.type}"
        )
    component_count = _blocked_layout_component_count(
        result_plan,
        f"{context} result",
        "SIMD tensor splat",
    )
    width = _tensor_lane_width(result_plan, f"{context} result")
    component = builder.splat(_scalar_mlir_value(scalar), width=width)
    if component_count == 1:
        return _WaveValue("simd", component)
    return _WaveValue("simd_tuple", tuple(component for _ in range(component_count)))


def _mask_splat_from_scalar(source, source_plan, result_plan, context):
    if result_plan.type_kind == "tensor" and _is_bool_value(result_plan):
        if source_plan.type_kind != "scalar" or not _is_bool_value(source_plan):
            raise ValueError(
                f"tlx_wave bridge cannot lower {context}: non-i1 scalar splat "
                f"to tensor mask from {source_plan.type}"
            )
        if isinstance(source.value, _ScalarBool) and source.value.const_value is not None:
            return _WaveValue("mask_expr", _MaskConst(source.value.const_value))
        raise ValueError(
            f"tlx_wave bridge cannot lower {context}: dynamic scalar i1 splat "
            "to tensor mask is not supported yet"
        )
    return None


def _broadcast_uniform_simd_data(source, source_plan, result_plan, context):
    if source_plan.element_type != result_plan.element_type:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context} for SIMD tensor data with "
            f"element type change: {source_plan.type} -> {result_plan.type}"
        )
    if not _is_data_tensor(result_plan):
        raise ValueError(
            f"tlx_wave bridge cannot lower {context} as SIMD tensor data: "
            f"unsupported result type {result_plan.type}"
        )
    source_count = _blocked_layout_component_count(
        source_plan,
        f"{context} source",
        "SIMD tensor broadcast",
    )
    result_count = _blocked_layout_component_count(
        result_plan,
        f"{context} result",
        "SIMD tensor broadcast",
    )
    source_width = _tensor_lane_width(source_plan, f"{context} source")
    result_width = _tensor_lane_width(result_plan, f"{context} result")
    if source_width != result_width:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context} for SIMD tensor data: "
            f"SIMD widths differ ({source_width} -> {result_width}); "
            f"source encoding: {source_plan.encoding}; "
            f"result encoding: {result_plan.encoding}"
        )
    if source_plan.shape == result_plan.shape:
        if _same_layout_encoding(source_plan, result_plan):
            return _forward_nonfragment_value(source)
        return _convert_simd_layout(source, source_plan, result_plan)
    if source_plan.varying_dims:
        raise ValueError(
            f"tlx_wave bridge cannot lower {context} for non-uniform SIMD "
            f"tensor data with shape change: source shape {source_plan.shape}, "
            f"result shape {result_plan.shape}, varying dims="
            f"{source_plan.varying_dims}"
        )
    source_components = _simd_components_for_layout(
        source, source_plan, source_count, context
    )
    component = source_components[0]
    if result_count == 1:
        return _WaveValue("simd", component)
    return _WaveValue("simd_tuple", tuple(component for _ in range(result_count)))


def _emit_splat_or_broadcast_op(builder, op, values, wave_values, w):
    if len(op.operands) != 1 or len(op.results) != 1:
        raise ValueError(f"tlx_wave bridge expected {op.name} with one operand/result")
    source_id = op.operands[0]
    result_id = op.results[0]
    source = wave_values.get(source_id)
    if source is None:
        return
    if not isinstance(source, _WaveValue):
        raise ValueError("tlx_wave bridge internal error: untyped Wave value")
    source_plan = values[source_id]
    result_plan = values[result_id]
    if source.kind in {"pointer_expr", "index_expr", "mask_expr"}:
        wave_values[result_id] = _forward_nonfragment_value(source)
        return
    if source.kind == "scalar":
        mask_splat = _mask_splat_from_scalar(
            source, source_plan, result_plan, op.name
        )
        if mask_splat is not None:
            wave_values[result_id] = mask_splat
            return
        wave_values[result_id] = _simd_splat_from_scalar(
            builder, source.value, result_plan, w, op.name
        )
        return
    if op.name == "tt.broadcast" and source.kind in {"simd", "simd_tuple"}:
        wave_values[result_id] = _broadcast_uniform_simd_data(
            source, source_plan, result_plan, op.name
        )
        return
    raise ValueError(
        f"tlx_wave bridge cannot lower {op.name} for lowered {source.kind} value"
    )


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


def _emit_dot_operand_convert_layout(
    builder, op, state, lds_layout, w, stats
):
    if len(op.operands) != 1 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected ttg.convert_layout with one value")
    source_id = op.operands[0]
    result_id = op.results[0]
    values = state["values"]
    result = values[result_id]
    if result.encoding_attr is None or not _attr_bool(
        result.encoding_attr, "is_dot_operand_encoding"
    ):
        return False
    source = state["wave_values"].get(source_id)
    if isinstance(source, _WaveValue) and source.kind == "fragment_regs":
        info = _dot_operand_encoding_info(result, "ttg.convert_layout result")
        fragment = builder.fragment_pack(
            source.value,
            _fragment_type_for_dot_operand(info, w),
        )
        stats.fragment_packs += 1
        _set_wave_value(state["wave_values"], result_id, "fragment", fragment)
        return True
    fragment_regs = _materialize_fragment_regs_aux(source)
    if fragment_regs is not None:
        info = _dot_operand_encoding_info(result, "ttg.convert_layout result")
        fragment = builder.fragment_pack(
            fragment_regs,
            _fragment_type_for_dot_operand(info, w),
        )
        stats.fragment_packs += 1
        _set_wave_value(state["wave_values"], result_id, "fragment", fragment)
        return True
    address = state["local_loads"].get(source_id)
    if address is None or address.memdesc_value_id is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.convert_layout to a dot operand "
            "fragment without a ttg.local_load source"
        )
    memdescs = state["memdescs"]
    memdesc = memdescs[address.memdesc_value_id]
    capability = _physical_local_load_fragment_capability(result, memdesc)
    fragment, token = _emit_dot_operand_fragment_load(
        builder,
        address,
        result,
        memdesc,
        memdescs,
        capability,
        lds_layout,
        state,
        _mem_root(state),
        w,
        stats,
    )
    _set_mem_root(state, token)
    _set_wave_value(state["wave_values"], result_id, "fragment", fragment)
    return True


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
        return physical_plan
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
    store_plan,
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
            builder, store_plan, lowered, w, component=component
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
    token = _mem_root(state)
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
    _set_mem_root(state, token)
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
    token = _mem_root(state)
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
    _set_mem_root(state, builder.barrier(token))
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


def _emit_local_load_op(
    builder,
    op,
    state,
    local_loads,
    lds_layout,
    w,
    stats,
    *,
    control_context=None,
):
    values = state["values"]
    wave_values = state["wave_values"]
    memdescs = state["memdescs"]
    result_id = op.results[0] if op.results else None
    if result_id is None:
        raise ValueError("tlx_wave bridge expected ttg.local_load result")
    value = values[result_id]
    address = local_loads.get(result_id)
    if address is None or address.memdesc_value_id is None:
        raise ValueError(
            "tlx_wave bridge cannot lower ttg.local_load: missing shared "
            f"memdesc source. original TTGIR encoding: {value.encoding}"
        )
    memdesc = memdescs[address.memdesc_value_id]
    after = (
        _require_wave_value(
            wave_values,
            address.token_value_id,
            ("token",),
            "ttg.local_load token",
        )
        if address.token_value_id is not None
        else _mem_root(state)
    )

    fragment_capability = _physical_local_load_fragment_capability(value, memdesc)
    if fragment_capability is not None:
        fragment, token = _emit_dot_operand_fragment_load(
            builder,
            address,
            value,
            memdesc,
            memdescs,
            fragment_capability,
            lds_layout,
            state,
            after,
            w,
            stats,
        )
        _set_mem_root(state, token)
        _set_wave_value(wave_values, result_id, "fragment", fragment)
        return

    loaded, token = _emit_generic_local_load(
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
    fragment_regs = None
    fragment_regs_source = None
    try:
        fragment_regs_capability = _physical_local_load_fragment_regs_capability(
            value, memdesc
        )
    except ValueError:
        fragment_regs_capability = None
    if fragment_regs_capability is not None and control_context == "scf.for":
        fragment_regs, token = _emit_dot_operand_register_load(
            builder,
            value,
            memdesc,
            memdescs,
            lds_layout,
            state,
            token,
            w,
            stats,
        )
    elif fragment_regs_capability is not None:
        def fragment_regs_source():
            regs, regs_token = _emit_dot_operand_register_load(
                builder,
                value,
                memdesc,
                memdescs,
                lds_layout,
                state,
                _mem_root(state),
                w,
                stats,
            )
            _set_mem_root(state, regs_token)
            return regs

    _set_mem_root(state, token)
    if isinstance(loaded, tuple):
        _set_wave_value(wave_values, result_id, "simd_tuple", loaded)
    else:
        _set_wave_value(wave_values, result_id, "simd", loaded)
    if fragment_regs is not None:
        wave_values[result_id] = _with_fragment_regs_aux(
            wave_values[result_id], fragment_regs
        )
    if fragment_regs_source is not None:
        wave_values[result_id] = _with_fragment_regs_source_aux(
            wave_values[result_id], fragment_regs_source
        )


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
        store_plan = _validate_fragment_store_value(value_plan, physical_plan)
        token = _emit_fragment_store(
            builder,
            state,
            lowered,
            store_plan,
            ptr_id,
            mask_id,
            _mem_root(state),
            w,
        )
        _set_mem_root(state, token)
        return

    if lowered.kind in {"simd", "simd_tuple", "index_expr"}:
        component_count = _blocked_layout_component_count(
            value_plan, "tt.store value", "generic tensor lowering"
        )
        token = _mem_root(state)
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
        _set_mem_root(state, token)
        return

    raise ValueError(
        "tlx_wave bridge cannot lower tt.store for Wave value kind "
        f"{lowered.kind}; stored TTGIR type={value_plan.type}"
    )


def _emit_ordered_raw_op(
    builder,
    kernel,
    raw_op,
    state,
    lds_layout,
    w,
    stats,
    *,
    control_context=None,
):
    op = _raw_op_plan(raw_op)
    values = state["values"]
    wave_values = state["wave_values"]

    if control_context == "scf.if" and _raw_op_has_memory_effects(raw_op):
        raise ValueError(
            "tlx_wave bridge cannot lower side effects inside "
            f"{control_context} yet: {op.name}"
        )
    if op.name == "ttg.convert_layout" and _emit_dot_operand_convert_layout(
        builder, op, state, lds_layout, w, stats
    ):
        return
    if _emit_generic_value_op(builder, state, op, w):
        return
    if op.name == "llvm.intr.assume":
        _emit_assume_op(builder, op, state, w)
    elif op.name == "scf.if":
        _emit_scf_if_op(builder, kernel, raw_op, op, state, lds_layout, w, stats)
    elif op.name == "scf.for":
        _emit_scf_for_op(builder, kernel, raw_op, op, state, lds_layout, w, stats)
    elif op.name == "tt.load":
        _emit_global_load_op(builder, op, state, w)
    elif op.name == "ttg.async_copy_global_to_local":
        token_id = op.results[0] if op.results else None
        address = state["address_by_token"].get(token_id)
        if address is None:
            raise ValueError("tlx_wave bridge could not match async copy token")
        token = _emit_async_copy(
            builder,
            state,
            kernel,
            address,
            lds_layout,
            _ensure_mem_root(builder, state),
            w,
            stats,
        )
        if token_id is not None:
            _set_wave_value(wave_values, token_id, "token", token)
        _append_open_async_token(state, token)
    elif op.name == "ttg.async_commit_group":
        input_tokens = tuple(
            _require_wave_value(
                wave_values,
                token_id,
                ("token",),
                "ttg.async_commit_group token",
            )
            for token_id in op.operands
        )
        group = _join_tokens(
            builder,
            input_tokens if input_tokens else _open_async_tokens(state),
            stats,
        )
        if input_tokens:
            _remove_open_async_tokens(state, input_tokens)
        else:
            _clear_open_async_tokens(state)
        _append_committed_group(state, group)
        _set_mem_root(state, group)
        for result_id in op.results:
            _set_wave_value(wave_values, result_id, "token", group)
        stats.commit_groups += 1
    elif op.name == "ttg.async_wait":
        keep_groups = int(op.attrs.get("num", 0) or 0)
        input_groups = tuple(
            _require_wave_value(
                wave_values,
                token_id,
                ("token",),
                "ttg.async_wait token",
            )
            for token_id in op.operands
        )
        groups = input_groups if input_groups else _committed_groups(state)
        wait_count = max(0, len(groups) - keep_groups)
        if wait_count:
            waited_groups = tuple(groups[:wait_count])
            wait_token = _join_tokens(builder, waited_groups, stats)
            builder.wait(wait_token)
            stats.waits += 1
            if input_groups:
                _remove_committed_groups(state, waited_groups)
            else:
                _set_committed_groups(state, groups[wait_count:])
            _set_mem_root(state, builder.barrier(wait_token))
            stats.barriers += 1
        _cap_committed_groups(state)
        ready_token = (
            _mem_root(state)
            if _mem_root(state) is not None
            else builder.token()
        )
        for result_id in op.results:
            _set_wave_value(wave_values, result_id, "token", ready_token)
    elif op.name == "ttg.local_store":
        _emit_generic_local_store_op(
            builder, op, state, state["memdescs"], lds_layout, w, stats
        )
    elif op.name == "ttg.local_load":
        _emit_local_load_op(
            builder,
            op,
            state,
            state["local_loads"],
            lds_layout,
            w,
            stats,
            control_context=control_context,
        )
    elif op.name == "tt.dot":
        _emit_dot_op(builder, op, values, wave_values, w, stats)
    elif op.name == "tt.store":
        _emit_store_op(builder, op, state, w)
    elif op.name in _PLANNING_ONLY_OPS:
        return
    else:
        region_note = (
            " with nested regions"
            if getattr(raw_op, "get_num_regions", lambda: 0)()
            else ""
        )
        raise ValueError(
            "tlx_wave bridge cannot lower unsupported TTGIR op in unified "
            f"body lowering: {op.name}{region_note}"
        )


def _emit_raw_block(
    builder,
    kernel,
    raw_ops,
    state,
    lds_layout,
    w,
    stats,
    *,
    control_context=None,
):
    for raw_op in raw_ops:
        if raw_op.get_name() == "scf.yield":
            op = _raw_op_plan(raw_op)
            return tuple(
                _require_typed_wave_value(
                    state["wave_values"],
                    operand_id,
                    f"{control_context or 'top-level'} scf.yield",
                )
                for operand_id in op.operands
            )
        _emit_ordered_raw_op(
            builder,
            kernel,
            raw_op,
            state,
            lds_layout,
            w,
            stats,
            control_context=control_context,
        )
    return None


def _emit_ordered_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats):
    state = _initial_lowering_state(builder, kernel, plan, w)
    state["stats"] = stats
    state["address_by_token"] = _async_address_by_token(plan)
    state["local_loads"] = _local_load_address_by_result(plan)
    _emit_raw_block(
        builder,
        kernel,
        plan.body_ops,
        state,
        lds_layout,
        w,
        stats,
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
    yield _repo_root() / "third_party" / "wave" / "build" / "wave-build"
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
            "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave, or run "
            "`python third_party/wave/build_tools/build_llvm.py --python-bindings` followed by "
            "`cmake -S third_party/wave -B third_party/wave/build/wave-build "
            "-G Ninja -DWAVE_ENABLE_PYTHON_BINDINGS=ON` and "
            "`cmake --build third_party/wave/build/wave-build`. "
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
        "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave, or run the standalone "
        "third_party/wave build documented in third_party/wave/README.md. "
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
        module_builder.module.operation.attributes[
            "tlx_wave.plan.num_layout_constraints"
        ] = _binding_i32_attr(w, len(plan.layout_constraints))
        module_builder.module.operation.attributes[
            "tlx_wave.plan.num_storage_aliases"
        ] = _binding_i32_attr(w, len(plan.storage_aliases))
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
