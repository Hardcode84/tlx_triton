import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _KernelArg:
    index: int
    name: str
    ttgir_type_obj: object
    ttgir_type: str
    type_kind: str
    wave_type: str
    kind: str
    divisibility: int | None
    pointer_range: int | None


@dataclass(frozen=True)
class _Kernel:
    name: str
    args: tuple[_KernelArg, ...]
    noinline: bool | None


@dataclass(frozen=True)
class _ModuleAttrs:
    target: str
    num_ctas: int
    num_warps: int
    threads_per_warp: int
    has_explicit_local_mem_access: bool


@dataclass(frozen=True)
class _TypePlan:
    raw: str
    kind: str
    shape: tuple[int, ...]
    element_type: str | None
    element_byte_width: int | None
    pointee_type: str | None
    encoding: str | None
    encoding_attr: object | None
    memory_space: str | None
    mutable: bool | None
    alloc_shape: tuple[int, ...]
    address_space: int | None


@dataclass(frozen=True)
class _ValuePlan:
    value_id: int
    kind: str
    producer: str
    result_index: int | None
    type: str
    type_kind: str
    shape: tuple[int, ...]
    element_type: str | None
    element_byte_width: int | None
    pointee_type: str | None
    encoding: str | None
    encoding_attr: object | None
    memory_space: str | None
    const_value: int | float | bool | None
    base_arg_index: int | None
    base_arg_name: str | None
    varying_dims: tuple[int, ...]
    variability: str


@dataclass(frozen=True)
class _OpPlan:
    index: int
    name: str
    operands: tuple[int, ...]
    results: tuple[int, ...]
    attrs: dict


@dataclass(frozen=True)
class _LayoutPlan:
    value_id: int
    source: str
    shape: tuple[int, ...]
    element_type: str | None
    element_byte_width: int | None
    encoding: str | None
    encoding_attr: object | None
    memory_space: str | None


@dataclass(frozen=True)
class _LayoutConstraintPlan:
    op: str
    source_value_id: int
    result_value_id: int
    value_kind: str
    source_type: str
    result_type: str
    source_encoding: str | None
    result_encoding: str | None
    source_memory_space: str | None
    result_memory_space: str | None


@dataclass(frozen=True)
class _MemDescPlan:
    value_id: int
    kind: str
    source: str
    name: str | None
    shape: tuple[int, ...]
    alloc_shape: tuple[int, ...]
    element_type: str | None
    element_byte_width: int | None
    encoding: str | None
    encoding_attr: object | None
    memory_space: str | None
    mutable: bool | None
    base_value_id: int | None
    view_op: str | None
    view_operands: tuple[int, ...]
    static_index: int | None
    view_offsets: tuple[int, ...]
    view_order: tuple[int, ...]
    alias_spec_value_id: int | None = None


@dataclass(frozen=True)
class _StorageAliasPlan:
    op: str
    value_id: int | None
    spec_value_id: int | None
    alloc_value_id: int | None
    storage: str | None
    buffer_size_bytes: int | None
    group_kind: str | None
    group_size: int | None
    elements: tuple[int, ...]


@dataclass(frozen=True)
class _AddressExprPlan:
    op: str
    address_value_id: int | None
    memdesc_value_id: int | None
    value_value_id: int | None
    result_value_id: int | None
    element_type: str | None
    element_byte_width: int | None
    shape: tuple[int, ...]
    base_arg_index: int | None
    base_arg_name: str | None
    offset_value_id: int | None
    mask_value_id: int | None
    other_value_id: int | None
    token_value_id: int | None
    varying_dims: tuple[int, ...]
    variability: str


@dataclass(frozen=True)
class _TokenPlan:
    value_id: int | None
    op: str
    source_address_value_id: int | None
    memdesc_value_id: int | None
    mask_value_id: int | None
    input_token_ids: tuple[int, ...]
    wait_group: int | None


@dataclass(frozen=True)
class _BridgePlan:
    kind: str
    op_counts: dict[str, int]
    ops: tuple[_OpPlan, ...]
    body_ops: tuple[object, ...]
    values: tuple[_ValuePlan, ...]
    addresses: tuple[_AddressExprPlan, ...]
    layouts: tuple[_LayoutPlan, ...]
    layout_constraints: tuple[_LayoutConstraintPlan, ...]
    memdescs: tuple[_MemDescPlan, ...]
    storage_aliases: tuple[_StorageAliasPlan, ...]
    tokens: tuple[_TokenPlan, ...]


@dataclass(frozen=True)
class _LdsLayout:
    size_bytes: int
    offsets: dict[int, int]


@dataclass
class _WaveAsyncStats:
    lds_size_bytes: int = 0
    async_copies: int = 0
    dma_load_lds: int = 0
    commit_groups: int = 0
    waits: int = 0
    joins: int = 0
    barriers: int = 0
    local_loads: int = 0
    fragment_packs: int = 0
    fragment_fills: int = 0
    mmas: int = 0


_ORDERED_BODY_VALUE_OPS = {
    "arith.addi",
    "arith.andi",
    "arith.cmpi",
    "arith.constant",
    "arith.divsi",
    "arith.divui",
    "arith.extf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minsi",
    "arith.minui",
    "arith.muli",
    "arith.remsi",
    "arith.remui",
    "arith.subi",
    "arith.truncf",
    "tt.addptr",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.get_program_id",
    "tt.make_range",
    "tt.splat",
    "tlx.local_alias",
    "tlx.release_layout",
    "tlx.require_layout",
    "ttg.convert_layout",
}

_ORDERED_BODY_EFFECT_OPS = {
    "amdg.buffer_load_to_local",
    "tt.dot",
    "tt.load",
    "tt.store",
    "ttg.async_commit_group",
    "ttg.async_copy_global_to_local",
    "ttg.async_wait",
    "ttg.local_load",
    "ttg.local_store",
}

_ORDERED_BODY_PLANNED_OPS = {
    "llvm.intr.assume",
    "rocdl.sched.barrier",
    "rocdl.sched.group.barrier",
    "rocdl.setprio",
    "scf.for",
    "scf.if",
    "scf.yield",
    "tt.return",
    "tlx.reuse_group",
    "tlx.set_buffer_overlap",
    "tlx.storage_alias_local_alloc",
    "tlx.storage_alias_spec",
    "ttg.local_alloc",
    "ttg.memdesc_index",
    "ttg.memdesc_reinterpret",
    "ttg.memdesc_reshape",
    "ttg.memdesc_subslice",
    "ttg.memdesc_trans",
}

_ORDERED_BODY_SUPPORTED_OPS = (
    _ORDERED_BODY_VALUE_OPS
    | _ORDERED_BODY_EFFECT_OPS
    | _ORDERED_BODY_PLANNED_OPS
)

_ASSUME_TREE_OPS = {
    "arith.addi",
    "arith.andi",
    "arith.cmpi",
    "arith.divsi",
    "arith.divui",
    "arith.extsi",
    "arith.extui",
    "arith.index_cast",
    "arith.maxsi",
    "arith.maxui",
    "arith.minsi",
    "arith.minui",
    "arith.muli",
    "arith.ori",
    "arith.remsi",
    "arith.remui",
    "arith.subi",
    "arith.trunci",
    "arith.xori",
}


@dataclass(frozen=True)
class _BlockedEncodingInfo:
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]


@dataclass(frozen=True)
class _SwizzledSharedEncodingInfo:
    vec: int
    per_phase: int
    max_phase: int
    order: tuple[int, ...]


@dataclass(frozen=True)
class _PaddedSharedEncodingInfo:
    intervals: tuple[int, ...]
    paddings: tuple[int, ...]
    offset_vectors: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _AMDMfmaEncodingInfo:
    version: int
    warps_per_cta: tuple[int, ...]
    instr_shape: tuple[int, ...]
    is_transposed: bool
    tiles_per_warp: tuple[int, ...] | None = None
    element_bit_width: int | None = None
    cga_layout: tuple[int, ...] | None = None


@dataclass(frozen=True)
class _DotOperandEncodingInfo:
    op_idx: int
    k_width: int
    parent: _BlockedEncodingInfo | _AMDMfmaEncodingInfo


_GFX950_DOT_PARENT_LAYOUT = _BlockedEncodingInfo(
    size_per_thread=(2, 2),
    threads_per_warp=(4, 16),
    warps_per_cta=(4, 1),
    order=(1, 0),
)
_GFX950_SHARED_LAYOUT = _SwizzledSharedEncodingInfo(
    vec=1,
    per_phase=1,
    max_phase=1,
    order=(1, 0),
)
_GFX950_F16_MMA_KIND = "mfma.f32.16x16x32.f16"
_GFX950_BF16_MMA_KIND = "mfma.f32.16x16x32.bf16"
_GFX950_F16_MMA32_KIND = "mfma.f32.32x32x16.f16"
_GFX950_BF16_MMA32_KIND = "mfma.f32.32x32x16.bf16"
_GFX942_F16_MMA_KIND = "mfma.f32.16x16x16.f16"
_GFX942_BF16_MMA_KIND = "mfma.f32.16x16x16.bf16"
_GFX950_MMA_M = 16
_GFX950_MMA_N = 16
_GFX950_MMA_WAVE = 64
_GFX950_MMA_REGS = 4
_GFX950_MMA_SHAPE = (32, 32)


def _value_id(value):
    return int(value.id())


def _value_type(value):
    return value.get_type()


def _tuple_or_empty(values):
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _align_to(value, alignment):
    if value == 0:
        return 0
    return ((value + alignment - 1) // alignment) * alignment


def _attr_str(attr):
    return None if attr is None else str(attr)


def _is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _log2_power_of_two(value, context):
    if not _is_power_of_two(value):
        raise ValueError(f"{context}: expected a positive power of two, got {value}")
    return value.bit_length() - 1


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


def _extract_bracket_literal(text, key, context):
    key_pos = text.find(key)
    if key_pos < 0:
        return None
    start = text.find("[", key_pos)
    if start < 0:
        raise ValueError(f"{context}: malformed {key} list in {text}")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError(f"{context}: unterminated {key} list in {text}")


def _int_list_literal(text, key, context):
    literal = _extract_bracket_literal(text, key, context)
    if literal is None:
        return None
    try:
        values = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{context}: cannot parse {key} list {literal}") from exc
    return tuple(int(value) for value in values)


def _offset_vectors_from_order_shape(order, shape, context):
    vectors = []
    rank = len(shape)
    seen = sorted(int(dim) for dim in order)
    if seen != list(range(rank)):
        raise ValueError(
            f"{context}: padded_shared order {order} is not a permutation of rank {rank}"
        )
    for dim in order:
        extent = int(shape[dim])
        bits = _log2_power_of_two(extent, context)
        for bit in range(bits):
            vector = [0] * rank
            vector[int(dim)] = 1 << bit
            vectors.append(tuple(vector))
    return tuple(vectors)


_PADDED_SHARED_RE = re.compile(
    r"^#ttg\.padded_shared<\[(?P<pairs>[^\]]*)\]\s*\{(?P<body>.*)\}>$"
)
_PADDED_PAIR_RE = re.compile(r"(\d+)\s*:\+\s*(\d+)")
_AMD_MFMA_RE = re.compile(r"^#ttg\.amd_mfma<\{(?P<body>.*)\}>$")


def _padded_shared_encoding_info(raw_encoding, context="padded shared encoding"):
    if raw_encoding is None:
        return None
    match = _PADDED_SHARED_RE.match(raw_encoding)
    if match is None:
        return None
    pairs = tuple(
        (int(interval), int(padding))
        for interval, padding in _PADDED_PAIR_RE.findall(match.group("pairs"))
    )
    if not pairs:
        raise ValueError(f"{context}: padded_shared has no interval/padding pairs")
    for interval, padding in pairs:
        _log2_power_of_two(interval, context)
        _log2_power_of_two(padding, context)

    body = match.group("body")
    block = _int_list_literal(body, "block", context)
    if block not in (None, ()):
        raise ValueError(
            f"{context}: padded_shared block layout {block} is not supported"
        )
    offset_literal = _extract_bracket_literal(body, "offset", context)
    if offset_literal is not None:
        try:
            offset_vectors = tuple(
                tuple(int(component) for component in vector)
                for vector in ast.literal_eval(offset_literal)
            )
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(
                f"{context}: cannot parse padded_shared offset list {offset_literal}"
            ) from exc
    else:
        order = _int_list_literal(body, "order", context)
        shape = _int_list_literal(body, "shape", context)
        if order is None or shape is None:
            raise ValueError(
                f"{context}: padded_shared requires either offset vectors or order/shape"
            )
        offset_vectors = _offset_vectors_from_order_shape(order, shape, context)

    if not offset_vectors:
        raise ValueError(f"{context}: padded_shared has no offset vectors")
    rank = len(offset_vectors[0])
    if rank == 0 or any(len(vector) != rank for vector in offset_vectors):
        raise ValueError(f"{context}: padded_shared offset vectors have inconsistent rank")
    return _PaddedSharedEncodingInfo(
        tuple(interval for interval, _ in pairs),
        tuple(padding for _, padding in pairs),
        offset_vectors,
    )


def _amd_mfma_encoding_info(raw_encoding, context="AMD MFMA encoding"):
    if raw_encoding is None:
        return None
    match = _AMD_MFMA_RE.match(raw_encoding)
    if match is None:
        return None
    body = match.group("body")
    version_match = re.search(r"\bversion\s*=\s*(\d+)", body)
    transposed_match = re.search(r"\bisTransposed\s*=\s*(true|false)", body)
    warps_per_cta = _int_list_literal(body, "warpsPerCTA", context)
    instr_shape = _int_list_literal(body, "instrShape", context)
    tiles_per_warp = _int_list_literal(body, "tilesPerWarp", context)
    cga_layout = _int_list_literal(body, "CGALayout", context)
    element_bit_width_match = re.search(r"\belementBitWidth\s*=\s*(\d+)", body)
    if (
        version_match is None
        or transposed_match is None
        or warps_per_cta is None
        or instr_shape is None
    ):
        raise ValueError(f"{context}: malformed #ttg.amd_mfma encoding {raw_encoding}")
    return _AMDMfmaEncodingInfo(
        int(version_match.group(1)),
        warps_per_cta,
        instr_shape,
        transposed_match.group(1) == "true",
        tiles_per_warp,
        int(element_bit_width_match.group(1)) if element_bit_width_match else None,
        cga_layout,
    )


def _padded_layout_bit_mapping(info, context):
    mapping = {}
    for physical_bit, vector in enumerate(info.offset_vectors):
        nonzero = [
            (dim, value)
            for dim, value in enumerate(vector)
            if int(value) != 0
        ]
        if len(nonzero) != 1:
            raise ValueError(
                f"{context}: only one-hot padded_shared offset vectors are supported"
            )
        dim, value = nonzero[0]
        logical_bit = _log2_power_of_two(int(value), context)
        key = (dim, logical_bit)
        if key in mapping:
            raise ValueError(
                f"{context}: duplicate padded_shared basis for dim {dim} bit {logical_bit}"
            )
        mapping[key] = physical_bit
    return mapping


def _padded_logical_offset_elements(coords, info, context):
    mapping = _padded_layout_bit_mapping(info, context)
    offset = 0
    for dim, coord in enumerate(coords):
        coord = int(coord)
        bit = 0
        while (1 << bit) <= coord:
            if coord & (1 << bit):
                key = (dim, bit)
                if key not in mapping:
                    raise ValueError(
                        f"{context}: padded_shared has no basis for dim {dim} bit {bit}"
                    )
                offset += 1 << mapping[key]
            bit += 1
    return offset


def _apply_static_padding(byte_offset, element_byte_width, info, context):
    result = int(byte_offset)
    for interval, padding in zip(info.intervals, info.paddings):
        interval_bytes = int(interval) * int(element_byte_width)
        padding_bytes = int(padding) * int(element_byte_width)
        _log2_power_of_two(interval_bytes, context)
        _log2_power_of_two(padding_bytes, context)
        result += (int(byte_offset) // interval_bytes) * padding_bytes
    return result


def _padded_shared_tile_storage_bytes(tile_shape, element_byte_width, info, context):
    if element_byte_width is None:
        raise ValueError(f"{context}: unknown element byte width for padded_shared")
    if len(tile_shape) != len(info.offset_vectors[0]):
        raise ValueError(
            f"{context}: padded_shared rank {len(info.offset_vectors[0])} "
            f"does not match tile shape {tile_shape}"
        )
    if any(int(extent) <= 0 for extent in tile_shape):
        raise ValueError(f"{context}: padded_shared tile shape must be positive")
    max_coords = tuple(int(extent) - 1 for extent in tile_shape)
    max_offset_elements = _padded_logical_offset_elements(max_coords, info, context)
    max_offset_bytes = max_offset_elements * int(element_byte_width)
    return (
        _apply_static_padding(max_offset_bytes, element_byte_width, info, context)
        + int(element_byte_width)
    )


def _padded_static_byte_offset(shape, coords, element_byte_width, raw_encoding, context):
    info = _padded_shared_encoding_info(raw_encoding, context)
    if info is None:
        return None
    rank = len(info.offset_vectors[0])
    if len(shape) < rank or len(coords) != len(shape):
        raise ValueError(
            f"{context}: padded_shared rank {rank} is incompatible with "
            f"shape={shape}, coords={coords}"
        )
    prefix_rank = len(shape) - rank
    prefix_shape = tuple(int(dim) for dim in shape[:prefix_rank])
    tile_shape = tuple(int(dim) for dim in shape[prefix_rank:])
    prefix_coords = tuple(int(coord) for coord in coords[:prefix_rank])
    tile_coords = tuple(int(coord) for coord in coords[prefix_rank:])
    tile_bytes = _padded_shared_tile_storage_bytes(
        tile_shape, element_byte_width, info, context
    )
    prefix_index = _static_linear_offset(prefix_shape, prefix_coords) if prefix_shape else 0
    tile_offset = _padded_logical_offset_elements(tile_coords, info, context)
    tile_offset_bytes = _apply_static_padding(
        tile_offset * int(element_byte_width), element_byte_width, info, context
    )
    return prefix_index * tile_bytes + tile_offset_bytes


def _type_str(type_obj):
    return str(type_obj)


def _type_method(type_obj, method, default=None):
    fn = getattr(type_obj, method, None)
    if fn is None:
        return default
    return fn()


def _type_predicate(type_obj, method):
    return bool(_type_method(type_obj, method, False))


def _type_is_integer_width(type_obj, width):
    fn = getattr(type_obj, "is_integer", None)
    if fn is None:
        return False
    return bool(fn(width))


def _scalar_byte_width(type_obj):
    if type_obj is None:
        return None
    if _type_is_integer_width(type_obj, 1) or _type_is_integer_width(type_obj, 8):
        return 1
    if (
        _type_is_integer_width(type_obj, 16)
        or _type_predicate(type_obj, "is_fp16")
        or _type_predicate(type_obj, "is_bf16")
    ):
        return 2
    if (
        _type_is_integer_width(type_obj, 32)
        or _type_predicate(type_obj, "is_fp32")
    ):
        return 4
    if (
        _type_is_integer_width(type_obj, 64)
        or _type_predicate(type_obj, "is_fp64")
        or _type_predicate(type_obj, "is_index")
    ):
        return 8
    return None


def _type_kind(type_obj):
    if _type_predicate(type_obj, "is_memdesc"):
        return "memdesc"
    if _type_predicate(type_obj, "is_ranked_tensor"):
        return "tensor"
    if _type_predicate(type_obj, "is_ptr"):
        return "pointer"
    if _type_predicate(type_obj, "is_async_token"):
        return "token"
    if _is_scalar_type(type_obj):
        return "scalar"
    return "other"


def _type_plan(type_obj):
    element_type = _type_method(type_obj, "get_element_type")
    pointee_type = _type_method(type_obj, "get_pointee_type")
    encoding_attr = _type_method(type_obj, "get_encoding")
    if pointee_type is None and element_type is not None:
        pointee_type = _type_method(element_type, "get_pointee_type")
    element_byte_width = _scalar_byte_width(element_type)
    if element_byte_width is None:
        element_byte_width = _scalar_byte_width(pointee_type)
    if element_byte_width is None and _is_scalar_type(type_obj):
        element_byte_width = _scalar_byte_width(type_obj)
    return _TypePlan(
        _type_str(type_obj),
        _type_kind(type_obj),
        _tuple_or_empty(_type_method(type_obj, "get_shape")),
        _type_str(element_type) if element_type is not None else None,
        element_byte_width,
        _type_str(pointee_type) if pointee_type is not None else None,
        _attr_str(encoding_attr),
        encoding_attr,
        _attr_str(_type_method(type_obj, "get_memory_space")),
        _type_method(type_obj, "get_mutable_memory"),
        _tuple_or_empty(_type_method(type_obj, "get_alloc_shape")),
        _type_method(type_obj, "get_address_space"),
    )


def _address_element_type(type_obj):
    pointee_type = type_obj.get_pointee_type()
    if pointee_type is not None:
        return _type_str(pointee_type)
    element_type = type_obj.get_element_type()
    if element_type is None:
        return _type_str(type_obj)
    pointee_type = element_type.get_pointee_type()
    return _type_str(pointee_type if pointee_type is not None else element_type)


def _address_element_byte_width(type_obj):
    pointee_type = type_obj.get_pointee_type()
    if pointee_type is not None:
        return _scalar_byte_width(pointee_type)
    element_type = type_obj.get_element_type()
    if element_type is None:
        return _scalar_byte_width(type_obj)
    pointee_type = element_type.get_pointee_type()
    return _scalar_byte_width(
        pointee_type if pointee_type is not None else element_type
    )


def _is_scalar_type(type_obj):
    return (
        _type_predicate(type_obj, "is_index")
        or _type_predicate(type_obj, "is_fp16")
        or _type_predicate(type_obj, "is_bf16")
        or _type_predicate(type_obj, "is_fp32")
        or _type_predicate(type_obj, "is_fp64")
        or any(_type_is_integer_width(type_obj, width) for width in (1, 8, 16, 32, 64))
    )


def _wave_scalar_name(name, type_obj):
    if type_obj.is_integer(1):
        return "i1"
    if type_obj.is_integer(8):
        return "i8"
    if type_obj.is_integer(16):
        return "i16"
    if type_obj.is_integer(32):
        return "i32"
    if type_obj.is_integer(64):
        return "i64"
    if type_obj.is_index():
        return "index"
    if type_obj.is_fp16():
        return "f16"
    if type_obj.is_bf16():
        return "bf16"
    if type_obj.is_fp32():
        return "f32"
    raise ValueError(
        f"tlx_wave bridge does not yet support kernel argument %{name} with type {type_obj}"
    )


def _wave_arg_type(name, ttgir_type):
    pointee_type = ttgir_type.get_pointee_type()
    if pointee_type is not None:
        return (
            f"!wave.ptr<#wave.global, {_wave_scalar_name(name, pointee_type)}>",
            "pointer",
        )
    return _wave_scalar_name(name, ttgir_type), "scalar"


def _jsonify(value):
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _public_dicts(records):
    result = []
    for record in records:
        row = {}
        for key, value in record.__dict__.items():
            if (
                key == "value_id"
                or key.endswith("_value_id")
                or key.endswith("_attr")
                or key in {"operands", "results", "input_token_ids"}
            ):
                continue
            row[key] = _jsonify(value)
        result.append(row)
    return result


def _walk_block_ops(block):
    ops = []
    for op_index in range(block.get_num_operations()):
        op = block.get_operation(op_index)
        ops.append(op)
        for region_index in range(op.get_num_regions()):
            ops.extend(_walk_region_ops(op.get_region(region_index)))
    return tuple(ops)


def _walk_region_ops(region):
    ops = []
    for block_index in range(region.size()):
        ops.extend(_walk_block_ops(region.get_block(block_index)))
    return tuple(ops)


def _walk_ops(mod, kernel):
    fn = mod.get_function(kernel.name)
    return _walk_region_ops(fn.get_region(0))


_SUPPORTED_REGION_OPS = {"scf.for", "scf.if"}


def _validate_straight_line_kernel_ops(ops):
    for op in ops:
        if op.get_num_regions() and op.get_name() not in _SUPPORTED_REGION_OPS:
            raise ValueError(
                "tlx_wave bridge supports recursive TTGIR lowering only for "
                "scf.if/scf.for regions; unsupported control-flow or "
                "nested-region op "
                f"{op.get_name()} has {op.get_num_regions()} nested region(s)"
            )


def _raw_op_result_ids(op):
    return tuple(
        _value_id(op.get_result(index)) for index in range(op.get_num_results())
    )


def _raw_op_operand_ids(op):
    return tuple(
        _value_id(op.get_operand(index)) for index in range(op.get_num_operands())
    )


def _ordered_body_users_by_value_id(ops):
    users = {}
    for op in ops:
        for operand_id in _raw_op_operand_ids(op):
            users.setdefault(operand_id, []).append(op)
    return users


def _ordered_body_op_by_result_id(ops):
    return {
        result_id: op
        for op in ops
        for result_id in _raw_op_result_ids(op)
    }


def _ordered_body_assume_tree_info(ops):
    op_by_result = _ordered_body_op_by_result_id(ops)
    users_by_value = _ordered_body_users_by_value_id(ops)
    value_ids = set()
    expanded = set()

    def walk(value_id):
        op = op_by_result.get(value_id)
        if op is None or op.get_name() not in _ASSUME_TREE_OPS:
            return
        value_ids.add(value_id)
        if value_id in expanded:
            return
        expanded.add(value_id)
        for operand_id in _raw_op_operand_ids(op):
            walk(operand_id)

    for op in ops:
        if op.get_name() == "llvm.intr.assume":
            for operand_id in _raw_op_operand_ids(op):
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
                if user.get_name() == "llvm.intr.assume":
                    continue
                if user.get_name() not in _ASSUME_TREE_OPS:
                    return False
                result_ids = _raw_op_result_ids(user)
                if not result_ids:
                    return False
                if not all(is_assume_only(result_id) for result_id in result_ids):
                    return False
            assume_only_values.add(value_id)
            return True
        finally:
            visiting.remove(value_id)

    for value_id in assume_tree_values:
        is_assume_only(value_id)
    return assume_tree_values, frozenset(assume_only_values)


def _is_assume_only_helper_op(
    op,
    assume_only_values,
):
    if op.get_name() not in _ASSUME_TREE_OPS:
        return False
    result_ids = _raw_op_result_ids(op)
    if not result_ids:
        return False
    return all(result_id in assume_only_values for result_id in result_ids)


def _validate_ordered_body_supported_ops(ops):
    _, assume_only_values = _ordered_body_assume_tree_info(ops)
    for op in ops:
        name = op.get_name()
        if name in _ORDERED_BODY_SUPPORTED_OPS:
            continue
        if _is_assume_only_helper_op(
            op,
            assume_only_values,
        ):
            continue
        raise ValueError(
            "tlx_wave bridge cannot lower unsupported TTGIR op in ordered "
            f"body lowering: {name}"
        )


def _op_results(op):
    return tuple(op.get_result(index) for index in range(op.get_num_results()))


def _op_operands(op):
    return tuple(op.get_operand(index) for index in range(op.get_num_operands()))


def _block_arguments(block):
    return tuple(block.get_argument(index) for index in range(block.get_num_arguments()))


def _walk_blocks_in_region(region):
    blocks = []
    for block_index in range(region.size()):
        block = region.get_block(block_index)
        blocks.append(block)
        for op_index in range(block.get_num_operations()):
            op = block.get_operation(op_index)
            for region_index in range(op.get_num_regions()):
                blocks.extend(_walk_blocks_in_region(op.get_region(region_index)))
    return tuple(blocks)


def _kernel_body_ops(mod, kernel):
    fn = mod.get_function(kernel.name)
    block = fn.get_region(0).get_block(0)
    return tuple(
        block.get_operation(index) for index in range(block.get_num_operations())
    )


def _op_int_array_attr(op, name):
    value = op.get_int_array_attr(name)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def _async_copy_operands(op):
    operands = _op_operands(op)
    segments = _op_int_array_attr(op, "operandSegmentSizes")
    if segments is None:
        raise ValueError(
            "tlx_wave bridge expected ttg.async_copy_global_to_local "
            "operandSegmentSizes attribute"
        )
    if len(segments) != 4:
        raise ValueError(
            "tlx_wave bridge expected ttg.async_copy_global_to_local "
            f"operandSegmentSizes with four entries, got {segments}"
        )
    if sum(segments) != len(operands):
        raise ValueError(
            "tlx_wave bridge found inconsistent ttg.async_copy_global_to_local "
            f"operandSegmentSizes={segments} for {len(operands)} operands"
        )
    if segments[0] != 1 or segments[1] != 1:
        raise ValueError(
            "tlx_wave bridge expected ttg.async_copy_global_to_local source "
            f"and destination operands, got operandSegmentSizes={segments}"
        )
    if segments[2] not in (0, 1) or segments[3] not in (0, 1):
        raise ValueError(
            "tlx_wave bridge expected optional single mask/other operands for "
            f"ttg.async_copy_global_to_local, got operandSegmentSizes={segments}"
        )

    index = 0
    address_value = operands[index]
    index += segments[0]
    memdesc_value = operands[index]
    index += segments[1]
    mask_value = operands[index] if segments[2] else None
    index += segments[2]
    other_value = operands[index] if segments[3] else None
    return address_value, memdesc_value, mask_value, other_value


def _require_default_cache_modifier(op, context):
    cache = dict(op.get_attrs()).get("cache")
    if cache is None or cache == 1 or str(cache) == "1":
        return
    raise ValueError(
        f"tlx_wave bridge cannot lower {context} with cacheModifier={cache}; "
        "Wave lowering does not support cache modifiers yet"
    )


def _buffer_load_to_local_operands(op):
    _require_default_cache_modifier(op, "amdg.buffer_load_to_local")
    operands = _op_operands(op)
    segments = _op_int_array_attr(op, "operandSegmentSizes")
    if segments is None:
        raise ValueError(
            "tlx_wave bridge expected amdg.buffer_load_to_local "
            "operandSegmentSizes attribute"
        )
    if len(segments) != 6:
        raise ValueError(
            "tlx_wave bridge expected amdg.buffer_load_to_local "
            f"operandSegmentSizes with six entries, got {segments}"
        )
    if sum(segments) != len(operands):
        raise ValueError(
            "tlx_wave bridge found inconsistent amdg.buffer_load_to_local "
            f"operandSegmentSizes={segments} for {len(operands)} operands"
        )
    if segments[0] != 1 or segments[1] != 1 or segments[2] != 1:
        raise ValueError(
            "tlx_wave bridge expected amdg.buffer_load_to_local destination, "
            f"base pointer, and offsets operands, got operandSegmentSizes={segments}"
        )
    if segments[3] not in (0, 1) or segments[4] not in (0, 1):
        raise ValueError(
            "tlx_wave bridge expected optional single mask/other operands for "
            f"amdg.buffer_load_to_local, got operandSegmentSizes={segments}"
        )
    if segments[5] not in (0, 1):
        raise ValueError(
            "tlx_wave bridge expected optional single stride operand for "
            f"amdg.buffer_load_to_local, got operandSegmentSizes={segments}"
        )

    index = 0
    memdesc_value = operands[index]
    index += segments[0]
    ptr_value = operands[index]
    index += segments[1]
    offsets_value = operands[index]
    index += segments[2]
    mask_value = operands[index] if segments[3] else None
    index += segments[3]
    other_value = operands[index] if segments[4] else None
    index += segments[4]
    stride_value = operands[index] if segments[5] else None
    if stride_value is not None:
        raise ValueError(
            "tlx_wave bridge cannot lower amdg.buffer_load_to_local with "
            "a stride operand yet"
        )
    return ptr_value, offsets_value, memdesc_value, mask_value, other_value


def _result_owner_map(ops):
    owners = {}
    for op in ops:
        for result in _op_results(op):
            owners[_value_id(result)] = op
    return owners


def _variability(varying_dims):
    if not varying_dims:
        return "uniform"
    if len(varying_dims) == 1:
        return "lane-varying"
    return "tile-varying"


def _merge_varying_dims(*plans):
    dims = set()
    for plan in plans:
        if plan is not None:
            dims.update(plan.varying_dims)
    return tuple(sorted(dims))


def _base_from_plans(plans):
    bases = {
        (plan.base_arg_index, plan.base_arg_name)
        for plan in plans
        if plan and plan.base_arg_index is not None
    }
    if len(bases) != 1:
        return None, None
    return next(iter(bases))


def _const_int(plan):
    if (
        plan is None
        or isinstance(plan.const_value, bool)
        or not isinstance(plan.const_value, int)
    ):
        return None
    return plan.const_value


def _shift_dims_for_expand(plan, axis):
    if plan is None:
        return ()
    return tuple(dim + 1 if dim >= axis else dim for dim in plan.varying_dims)


def _argument_value_plan(value, index):
    type_plan = _type_plan(_value_type(value))
    return _ValuePlan(
        _value_id(value),
        "argument",
        "argument",
        None,
        type_plan.raw,
        type_plan.kind,
        type_plan.shape,
        type_plan.element_type,
        type_plan.element_byte_width,
        type_plan.pointee_type,
        type_plan.encoding,
        type_plan.encoding_attr,
        type_plan.memory_space,
        None,
        index,
        f"arg{index}",
        (),
        "uniform",
    )


def _block_argument_value_plan(value):
    type_plan = _type_plan(_value_type(value))
    return _ValuePlan(
        _value_id(value),
        "block_argument",
        "block_argument",
        None,
        type_plan.raw,
        type_plan.kind,
        type_plan.shape,
        type_plan.element_type,
        type_plan.element_byte_width,
        type_plan.pointee_type,
        type_plan.encoding,
        type_plan.encoding_attr,
        type_plan.memory_space,
        None,
        None,
        None,
        (),
        "uniform",
    )


def _value_plan_from_result(op, result_index, result, operand_plans, arg_info):
    value_id = _value_id(result)
    if value_id in arg_info:
        index, _name = arg_info[value_id]
        return _argument_value_plan(result, index)

    op_name = op.get_name()
    type_plan = _type_plan(_value_type(result))
    const_value = op.get_constant_value() if op_name == "arith.constant" else None
    base_arg_index, base_arg_name = _base_from_plans(operand_plans)

    if op_name == "arith.constant":
        varying_dims = ()
        kind = "constant"
    elif op_name == "tt.get_program_id":
        varying_dims = ()
        kind = "program_id"
    elif op_name == "tt.make_range":
        varying_dims = tuple(range(len(type_plan.shape)))
        kind = "make_range"
    elif op_name == "tt.expand_dims":
        varying_dims = _shift_dims_for_expand(
            operand_plans[0] if operand_plans else None, op.get_int_attr("axis") or 0
        )
        kind = "expand_dims"
    elif op_name in {
        "arith.extf",
        "arith.truncf",
        "tt.broadcast",
        "tt.splat",
        "ttg.convert_layout",
    }:
        varying_dims = (
            operand_plans[0].varying_dims if operand_plans and operand_plans[0] else ()
        )
        kind = op_name.split(".")[-1]
    elif op_name in {
        "arith.addi",
        "arith.andi",
        "arith.cmpi",
        "arith.muli",
        "arith.subi",
    }:
        varying_dims = _merge_varying_dims(*operand_plans)
        kind = op_name.split(".")[-1]
    elif op_name == "tt.addptr":
        varying_dims = _merge_varying_dims(*operand_plans)
        ptr_base = operand_plans[0] if operand_plans else None
        if ptr_base is not None:
            base_arg_index = ptr_base.base_arg_index
            base_arg_name = ptr_base.base_arg_name
        kind = "addptr"
    elif type_plan.kind == "token":
        varying_dims = ()
        kind = "token"
    elif type_plan.kind == "memdesc":
        varying_dims = ()
        kind = "memdesc"
    elif op_name in {"ttg.local_load", "tt.dot"}:
        varying_dims = tuple(range(len(type_plan.shape)))
        kind = op_name.split(".")[-1]
    else:
        varying_dims = _merge_varying_dims(*operand_plans)
        kind = op_name

    return _ValuePlan(
        value_id,
        kind,
        op_name,
        result_index,
        type_plan.raw,
        type_plan.kind,
        type_plan.shape,
        type_plan.element_type,
        type_plan.element_byte_width,
        type_plan.pointee_type,
        type_plan.encoding,
        type_plan.encoding_attr,
        type_plan.memory_space,
        const_value,
        base_arg_index,
        base_arg_name,
        varying_dims,
        _variability(varying_dims),
    )


def _build_value_plans(mod, kernel, ops):
    fn = mod.get_function(kernel.name)
    values = {}
    arg_info = {}
    for index in range(fn.get_num_args()):
        arg_value = fn.args(index)
        arg_info[_value_id(arg_value)] = (index, f"arg{index}")
        arg_plan = _argument_value_plan(arg_value, index)
        values[arg_plan.value_id] = arg_plan
    for block in _walk_blocks_in_region(fn.get_region(0)):
        for block_arg in _block_arguments(block):
            value_id = _value_id(block_arg)
            if value_id in values:
                continue
            plan = _block_argument_value_plan(block_arg)
            values[plan.value_id] = plan

    for op in ops:
        operands = _op_operands(op)
        operand_plans = tuple(values.get(_value_id(operand)) for operand in operands)
        for result_index, result in enumerate(_op_results(op)):
            plan = _value_plan_from_result(
                op, result_index, result, operand_plans, arg_info
            )
            values[plan.value_id] = plan
    return values


def _build_op_plans(ops):
    plans = []
    for index, op in enumerate(ops):
        attrs = dict(op.get_attrs())
        if op.get_name() in {
            "tt.get_program_id",
            "tt.get_num_programs",
            "tt.expand_dims",
        }:
            axis = op.get_int_attr("axis")
            if axis is not None:
                attrs["axis"] = axis
        plans.append(
            _OpPlan(
                index,
                op.get_name(),
                tuple(_value_id(operand) for operand in _op_operands(op)),
                tuple(_value_id(result) for result in _op_results(op)),
                attrs,
            )
        )
    return tuple(plans)


def _layout_plan(value, source):
    type_plan = _type_plan(_value_type(value))
    if type_plan.encoding is None and type_plan.memory_space is None:
        return None
    return _LayoutPlan(
        _value_id(value),
        source,
        type_plan.shape,
        type_plan.element_type,
        type_plan.element_byte_width,
        type_plan.encoding,
        type_plan.encoding_attr,
        type_plan.memory_space,
    )


def _build_layout_plans(ops):
    layouts = {}
    for op in ops:
        for result in _op_results(op):
            plan = _layout_plan(result, op.get_name())
            if plan is not None:
                layouts[plan.value_id] = plan
    return tuple(layouts.values())


_LAYOUT_CONSTRAINT_OPS = {"tlx.require_layout", "tlx.release_layout"}


def _layout_constraint_plan(op):
    name = op.get_name()
    if name not in _LAYOUT_CONSTRAINT_OPS:
        return None
    operands = _op_operands(op)
    results = _op_results(op)
    if len(operands) != 1 or len(results) != 1:
        raise ValueError(
            f"tlx_wave bridge expected {name} with one operand and one result"
        )
    source = _type_plan(_value_type(operands[0]))
    result = _type_plan(_value_type(results[0]))
    if source.kind != result.kind:
        raise ValueError(
            f"tlx_wave bridge cannot lower {name}: source/result type kinds "
            f"differ ({source.kind} -> {result.kind})"
        )
    if source.kind not in {"memdesc", "tensor"}:
        raise ValueError(
            f"tlx_wave bridge cannot lower {name}: unsupported layout "
            f"constraint type {source.raw}"
        )
    if name == "tlx.release_layout" and source.kind != "tensor":
        raise ValueError(
            "tlx_wave bridge cannot lower tlx.release_layout: expected tensor "
            f"constraint, got {source.raw}"
        )
    if source.shape != result.shape or source.element_type != result.element_type:
        raise ValueError(
            f"tlx_wave bridge cannot lower {name}: layout constraint changes "
            f"shape or element type ({source.raw} -> {result.raw})"
        )
    if source.memory_space != result.memory_space:
        raise ValueError(
            f"tlx_wave bridge cannot lower {name}: layout constraint changes "
            f"memory space ({source.memory_space} -> {result.memory_space})"
        )
    return _LayoutConstraintPlan(
        name,
        _value_id(operands[0]),
        _value_id(results[0]),
        source.kind,
        source.raw,
        result.raw,
        source.encoding,
        result.encoding,
        source.memory_space,
        result.memory_space,
    )


def _build_layout_constraint_plans(ops):
    constraints = []
    requirements_by_source = {}
    for op in ops:
        plan = _layout_constraint_plan(op)
        if plan is None:
            continue
        if plan.op == "tlx.require_layout":
            previous = requirements_by_source.get(plan.source_value_id)
            if previous is not None and (
                previous.result_encoding != plan.result_encoding
                or previous.result_memory_space != plan.result_memory_space
            ):
                raise ValueError(
                    "tlx_wave bridge cannot lower conflicting "
                    "tlx.require_layout constraints for TTGIR value "
                    f"{plan.source_value_id}: {previous.result_type} vs "
                    f"{plan.result_type}"
                )
            requirements_by_source[plan.source_value_id] = plan
        constraints.append(plan)
    return tuple(constraints)


def _op_attr_text(op, name):
    attr = dict(op.get_attrs()).get(name)
    return None if attr is None else str(attr)


def _storage_kind_from_text(*texts):
    for text in texts:
        if text is None:
            continue
        compact = text.replace(" ", "")
        if "smemCluster" in compact or "smem_cluster" in compact:
            return "smemCluster"
        if "smem" in compact:
            return "smem"
        if "tmem" in compact:
            return "tmem"
    return None


def _reuse_group_kind_from_text(*texts):
    for text in texts:
        if text is None:
            continue
        compact = text.replace(" ", "")
        if "distinct" in compact:
            return "distinct"
        if "shared" in compact:
            return "shared"
    return None


def _storage_alias_plan(op, values):
    name = op.get_name()
    operands = _op_operands(op)
    results = _op_results(op)
    if name == "tlx.storage_alias_spec":
        if len(results) != 1:
            raise ValueError(
                "tlx_wave bridge expected tlx.storage_alias_spec to produce one result"
            )
        result_id = _value_id(results[0])
        storage = _storage_kind_from_text(
            _op_attr_text(op, "storage"), values[result_id].type
        )
        if storage != "smem":
            raise ValueError(
                "tlx_wave bridge cannot lower tlx.storage_alias_spec: "
                f"unsupported storage {storage or 'unknown'}; only smem is supported"
            )
        return _StorageAliasPlan(
            name,
            result_id,
            result_id,
            None,
            storage,
            op.get_int_attr("buffer_size_bytes"),
            None,
            None,
            (),
        )
    if name == "tlx.storage_alias_local_alloc":
        if len(operands) != 1 or len(results) != 1:
            raise ValueError(
                "tlx_wave bridge expected tlx.storage_alias_local_alloc with "
                "one spec operand and one memdesc result"
            )
        spec_id = _value_id(operands[0])
        result_id = _value_id(results[0])
        spec_plan = values.get(spec_id)
        storage = _storage_kind_from_text(
            spec_plan.type if spec_plan is not None else None
        )
        if storage != "smem":
            raise ValueError(
                "tlx_wave bridge cannot lower tlx.storage_alias_local_alloc: "
                f"unsupported storage {storage or 'unknown'}; only smem is supported"
            )
        return _StorageAliasPlan(
            name,
            result_id,
            spec_id,
            result_id,
            storage,
            None,
            None,
            None,
            (),
        )
    if name == "tlx.reuse_group":
        if len(results) != 1:
            raise ValueError("tlx_wave bridge expected tlx.reuse_group result")
        result_id = _value_id(results[0])
        return _StorageAliasPlan(
            name,
            result_id,
            None,
            None,
            None,
            None,
            _reuse_group_kind_from_text(
                _op_attr_text(op, "group_kind"), values[result_id].type
            ),
            op.get_int_attr("group_size") or 1,
            tuple(_value_id(operand) for operand in operands),
        )
    if name == "tlx.set_buffer_overlap":
        if len(operands) != 2:
            raise ValueError(
                "tlx_wave bridge expected tlx.set_buffer_overlap with spec and group"
            )
        return _StorageAliasPlan(
            name,
            None,
            _value_id(operands[0]),
            None,
            None,
            None,
            None,
            None,
            (_value_id(operands[1]),),
        )
    return None


def _build_storage_alias_plans(ops, values):
    aliases = []
    specs = set()
    alloc_specs = {}
    groups = {}
    overlaps = {}
    for op in ops:
        plan = _storage_alias_plan(op, values)
        if plan is None:
            continue
        if plan.op == "tlx.storage_alias_spec":
            specs.add(plan.spec_value_id)
        elif plan.op == "tlx.storage_alias_local_alloc":
            if plan.spec_value_id not in specs:
                raise ValueError(
                    "tlx_wave bridge cannot lower "
                    "tlx.storage_alias_local_alloc: referenced "
                    f"tlx.storage_alias_spec {plan.spec_value_id} is not known"
                )
            alloc_specs[plan.alloc_value_id] = plan.spec_value_id
        elif plan.op == "tlx.reuse_group":
            groups[plan.value_id] = plan
        elif plan.op == "tlx.set_buffer_overlap":
            if plan.spec_value_id not in specs:
                raise ValueError(
                    "tlx_wave bridge cannot lower tlx.set_buffer_overlap: "
                    f"unknown tlx.storage_alias_spec {plan.spec_value_id}"
                )
            if plan.spec_value_id in overlaps:
                raise ValueError(
                    "tlx_wave bridge cannot lower tlx.set_buffer_overlap: "
                    f"duplicate overlap constraint for spec {plan.spec_value_id}"
                )
            group_id = plan.elements[0]
            group = groups.get(group_id)
            if group is None:
                raise ValueError(
                    "tlx_wave bridge cannot lower tlx.set_buffer_overlap: "
                    f"unknown tlx.reuse_group {group_id}"
                )
            if group.group_kind != "shared" or group.group_size != 1:
                raise ValueError(
                    "tlx_wave bridge cannot lower tlx.set_buffer_overlap: "
                    "only flat shared reuse_group constraints with group_size=1 "
                    f"are supported, got group_kind={group.group_kind}, "
                    f"group_size={group.group_size}"
                )
            for element in group.elements:
                if element not in alloc_specs or alloc_specs[element] != plan.spec_value_id:
                    raise ValueError(
                        "tlx_wave bridge cannot lower tlx.set_buffer_overlap: "
                        "reuse_group element does not reference the same "
                        f"tlx.storage_alias_spec {plan.spec_value_id}"
                    )
            overlaps[plan.spec_value_id] = group_id
        aliases.append(plan)
    return tuple(aliases)


_MEMDESC_VIEW_OPS = {
    "tlx.local_alias",
    "tlx.require_layout",
    "ttg.memdesc_index",
    "ttg.memdesc_subslice",
    "ttg.memdesc_reinterpret",
    "ttg.memdesc_reshape",
    "ttg.memdesc_trans",
}


def _memdesc_plan_from_value(
    value,
    source,
    kind,
    name=None,
    base_value_id=None,
    view_op=None,
    view_operands=(),
    static_index=None,
    view_offsets=(),
    view_order=(),
    alias_spec_value_id=None,
):
    type_plan = _type_plan(_value_type(value))
    if type_plan.kind != "memdesc":
        raise ValueError(
            f"tlx_wave bridge expected {source} to produce a memdesc, got {type_plan.raw}"
        )
    return _MemDescPlan(
        _value_id(value),
        kind,
        source,
        name,
        type_plan.shape,
        type_plan.alloc_shape,
        type_plan.element_type,
        type_plan.element_byte_width,
        type_plan.encoding,
        type_plan.encoding_attr,
        type_plan.memory_space,
        type_plan.mutable,
        base_value_id,
        view_op,
        view_operands,
        static_index,
        view_offsets,
        view_order,
        alias_spec_value_id,
    )


def _build_memdesc_plans(ops, values):
    memdescs = {}
    alloc_index = 0
    view_index = 0
    for op in ops:
        name = op.get_name()
        if name == "ttg.local_alloc":
            result = op.get_result(0)
            plan = _memdesc_plan_from_value(
                result, name, "allocation", name=f"alloc{alloc_index}"
            )
            alloc_index += 1
            memdescs[plan.value_id] = plan
        elif name == "tlx.storage_alias_local_alloc":
            result = op.get_result(0)
            operands = _op_operands(op)
            plan = _memdesc_plan_from_value(
                result,
                name,
                "allocation",
                name=f"alias_alloc{alloc_index}",
                alias_spec_value_id=_value_id(operands[0]) if operands else None,
            )
            alloc_index += 1
            memdescs[plan.value_id] = plan
        elif name in _MEMDESC_VIEW_OPS:
            result = op.get_result(0)
            type_plan = _type_plan(_value_type(result))
            if type_plan.kind != "memdesc":
                continue
            operands = _op_operands(op)
            static_index = (
                _const_int(values.get(_value_id(operands[1])))
                if name == "ttg.memdesc_index" and len(operands) > 1
                else None
            )
            view_offsets = (
                _op_int_array_attr(op, "offsets")
                if name == "ttg.memdesc_subslice"
                else None
            )
            view_order = (
                _op_int_array_attr(op, "order")
                if name == "ttg.memdesc_trans"
                else None
            )
            plan = _memdesc_plan_from_value(
                result,
                name,
                "view",
                name=f"view{view_index}",
                base_value_id=_value_id(operands[0]) if operands else None,
                view_op=name,
                view_operands=tuple(_value_id(operand) for operand in operands),
                static_index=static_index,
                view_offsets=view_offsets or (),
                view_order=view_order or (),
            )
            view_index += 1
            memdescs[plan.value_id] = plan
    return memdescs


def _token_id(value, values):
    if value is None:
        return None
    plan = values.get(_value_id(value))
    if plan is not None and plan.type_kind == "token":
        return plan.value_id
    return None


def _address_plan(op, values, owners):
    name = op.get_name()
    operands = _op_operands(op)
    results = _op_results(op)
    address_value = memdesc_value = value_value = mask_value = other_value = None
    offset_value = None
    result_value_id = token_value_id = None

    if name == "ttg.async_copy_global_to_local":
        address_value, memdesc_value, mask_value, other_value = _async_copy_operands(op)
        token_value_id = _value_id(results[0]) if results else None
        result_value_id = token_value_id
    elif name == "amdg.buffer_load_to_local":
        address_value, offset_value, memdesc_value, mask_value, other_value = (
            _buffer_load_to_local_operands(op)
        )
        token_value_id = _value_id(results[0]) if results else None
        result_value_id = token_value_id
    elif name == "tt.store":
        address_value = operands[0] if len(operands) > 0 else None
        value_value = operands[1] if len(operands) > 1 else None
        mask_value = operands[2] if len(operands) > 2 else None
    elif name == "tt.load":
        address_value = operands[0] if len(operands) > 0 else None
        mask_value = operands[1] if len(operands) > 1 else None
        result_value_id = _value_id(results[0]) if results else None
    elif name == "ttg.local_load":
        memdesc_value = operands[0] if len(operands) > 0 else None
        token_value_id = _token_id(operands[1], values) if len(operands) > 1 else None
        result_value_id = _value_id(results[0]) if results else None
    elif name == "ttg.local_store":
        value_value = operands[0] if len(operands) > 0 else None
        memdesc_value = operands[1] if len(operands) > 1 else None
    else:
        return None

    source_value = (
        offset_value
        if name == "amdg.buffer_load_to_local"
        else (address_value or memdesc_value)
    )
    source_plan = (
        values.get(_value_id(source_value)) if source_value is not None else None
    )
    base_plan = (
        values.get(_value_id(address_value))
        if address_value is not None
        else source_plan
    )
    offset_value_id = None
    if offset_value is not None:
        offset_value_id = _value_id(offset_value)
    elif address_value is not None:
        owner = owners.get(_value_id(address_value))
        if (
            owner is not None
            and owner.get_name() == "tt.addptr"
            and owner.get_num_operands() > 1
        ):
            offset_value_id = _value_id(owner.get_operand(1))

    return _AddressExprPlan(
        name,
        _value_id(address_value) if address_value is not None else None,
        _value_id(memdesc_value) if memdesc_value is not None else None,
        _value_id(value_value) if value_value is not None else None,
        result_value_id,
        (
            _address_element_type(_value_type(address_value))
            if address_value is not None
            else (source_plan.element_type if source_plan is not None else None)
        ),
        (
            _address_element_byte_width(_value_type(address_value))
            if address_value is not None
            else (source_plan.element_byte_width if source_plan is not None else None)
        ),
        source_plan.shape if source_plan is not None else (),
        (
            base_plan.base_arg_index
            if base_plan is not None and base_plan.base_arg_index is not None
            else (source_plan.base_arg_index if source_plan is not None else None)
        ),
        (
            base_plan.base_arg_name
            if base_plan is not None and base_plan.base_arg_name is not None
            else (source_plan.base_arg_name if source_plan is not None else None)
        ),
        offset_value_id,
        _value_id(mask_value) if mask_value is not None else None,
        _value_id(other_value) if other_value is not None else None,
        token_value_id,
        source_plan.varying_dims if source_plan is not None else (),
        source_plan.variability if source_plan is not None else "uniform",
    )


def _build_address_plans(ops, values, owners):
    addresses = []
    for op in ops:
        plan = _address_plan(op, values, owners)
        if plan is not None:
            addresses.append(plan)
    return tuple(addresses)


def _build_token_plans(ops, values):
    tokens = []
    for op in ops:
        name = op.get_name()
        operands = _op_operands(op)
        results = _op_results(op)
        async_source_value = async_memdesc_value = async_mask_value = None
        if name == "ttg.async_copy_global_to_local":
            async_source_value, async_memdesc_value, async_mask_value, _ = (
                _async_copy_operands(op)
            )
        elif name == "amdg.buffer_load_to_local":
            async_source_value, _, async_memdesc_value, async_mask_value, _ = (
                _buffer_load_to_local_operands(op)
            )
        result_token_id = (
            _value_id(results[0])
            if results and values.get(_value_id(results[0])).type_kind == "token"
            else None
        )
        input_token_ids = tuple(
            token_id
            for token_id in (_token_id(operand, values) for operand in operands)
            if token_id is not None
        )
        if (
            result_token_id is not None
            or input_token_ids
            or name in {"ttg.async_wait", "ttg.async_commit_group"}
        ):
            tokens.append(
                _TokenPlan(
                    result_token_id,
                    name,
                    (
                        _value_id(async_source_value)
                        if async_source_value is not None
                        else None
                    ),
                    (
                        _value_id(async_memdesc_value)
                        if async_memdesc_value is not None
                        else None
                    ),
                    (
                        _value_id(async_mask_value)
                        if async_mask_value is not None
                        else None
                    ),
                    input_token_ids,
                    op.get_int_attr("num") if name == "ttg.async_wait" else None,
                )
            )
    return tuple(tokens)


def _build_bridge_plan(mod, kernel):
    ops = _walk_ops(mod, kernel)
    _validate_straight_line_kernel_ops(ops)
    _validate_ordered_body_supported_ops(ops)
    owners = _result_owner_map(ops)
    values_by_id = _build_value_plans(mod, kernel, ops)
    op_plans = _build_op_plans(ops)
    op_counts = {}
    for op in ops:
        name = op.get_name()
        op_counts[name] = op_counts.get(name, 0) + 1
    layout_constraints = _build_layout_constraint_plans(ops)
    storage_aliases = _build_storage_alias_plans(ops, values_by_id)
    memdescs_by_id = _build_memdesc_plans(ops, values_by_id)
    address_plans = _build_address_plans(ops, values_by_id, owners)
    return _BridgePlan(
        "ttgir_graph",
        op_counts,
        op_plans,
        _kernel_body_ops(mod, kernel),
        tuple(values_by_id.values()),
        address_plans,
        _build_layout_plans(ops),
        layout_constraints,
        tuple(memdescs_by_id.values()),
        storage_aliases,
        _build_token_plans(ops, values_by_id),
    )


def _bridge_plan_metadata(plan):
    return {
        "kind": plan.kind,
        "op_counts": _jsonify(plan.op_counts),
        "num_ops": len(plan.ops),
        "num_values": len(plan.values),
        "num_addresses": len(plan.addresses),
        "num_layouts": len(plan.layouts),
        "num_layout_constraints": len(plan.layout_constraints),
        "num_memdescs": len(plan.memdescs),
        "num_storage_aliases": len(plan.storage_aliases),
        "num_tokens": len(plan.tokens),
        "ops": _public_dicts(plan.ops),
        "values": _public_dicts(plan.values),
        "addresses": _public_dicts(plan.addresses),
        "layouts": _public_dicts(plan.layouts),
        "layout_constraints": _public_dicts(plan.layout_constraints),
        "memdescs": _public_dicts(plan.memdescs),
        "storage_aliases": _public_dicts(plan.storage_aliases),
        "tokens": _public_dicts(plan.tokens),
    }


def _memdesc_size_bytes(memdesc):
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot size LDS memdesc {memdesc.name or memdesc.source} "
            f"with element type {memdesc.element_type}"
        )
    shape = memdesc.alloc_shape or memdesc.shape
    padded = _padded_shared_encoding_info(
        memdesc.encoding,
        f"tlx_wave bridge cannot size LDS memdesc {memdesc.name or memdesc.source}",
    )
    if padded is not None:
        rank = len(padded.offset_vectors[0])
        if len(shape) < rank:
            raise ValueError(
                f"tlx_wave bridge cannot size LDS memdesc {memdesc.name or memdesc.source}: "
                f"padded_shared rank {rank} exceeds alloc shape {shape}"
            )
        prefix_shape = shape[: len(shape) - rank]
        tile_shape = shape[len(shape) - rank :]
        return _product(prefix_shape) * _padded_shared_tile_storage_bytes(
            tile_shape,
            memdesc.element_byte_width,
            padded,
            f"tlx_wave bridge cannot size LDS memdesc {memdesc.name or memdesc.source}",
        )
    return _product(shape) * memdesc.element_byte_width


def _storage_alias_specs(plan):
    return {
        alias.spec_value_id: alias
        for alias in plan.storage_aliases
        if alias.op == "tlx.storage_alias_spec"
    }


def _storage_alias_allocs_by_spec(plan):
    allocs = {}
    for memdesc in plan.memdescs:
        if memdesc.alias_spec_value_id is None:
            continue
        allocs.setdefault(memdesc.alias_spec_value_id, []).append(memdesc)
    return allocs


def _validate_storage_alias_liveness(plan, allocs_by_spec):
    alias_alloc_ids = {
        memdesc.value_id
        for allocs in allocs_by_spec.values()
        for memdesc in allocs
    }
    for alias in plan.storage_aliases:
        if alias.op != "tlx.reuse_group":
            continue
        for element in alias.elements:
            if element not in alias_alloc_ids:
                raise ValueError(
                    "tlx_wave bridge cannot lower tlx.reuse_group: only flat "
                    "SMEM alias allocations are supported; unsupported nested "
                    f"or non-alias element {element}"
                )


def _compute_lds_layout(plan):
    offsets = {}
    cursor = 0
    specs = _storage_alias_specs(plan)
    alias_allocs_by_spec = _storage_alias_allocs_by_spec(plan)
    _validate_storage_alias_liveness(plan, alias_allocs_by_spec)

    for memdesc in plan.memdescs:
        if memdesc.kind != "allocation" or memdesc.alias_spec_value_id is not None:
            continue
        cursor = _align_to(cursor, 16)
        offsets[memdesc.value_id] = cursor
        cursor += _memdesc_size_bytes(memdesc)

    for spec_id, allocs in alias_allocs_by_spec.items():
        spec = specs.get(spec_id)
        if spec is None:
            raise ValueError(
                "tlx_wave bridge cannot lower tlx.storage_alias_local_alloc: "
                f"missing tlx.storage_alias_spec {spec_id}"
            )
        arena_size = max((_memdesc_size_bytes(memdesc) for memdesc in allocs), default=0)
        if spec.buffer_size_bytes is not None:
            if spec.buffer_size_bytes < arena_size:
                raise ValueError(
                    "tlx_wave bridge cannot lower tlx.storage_alias_spec: "
                    f"explicit size {spec.buffer_size_bytes} bytes is smaller "
                    f"than required SMEM arena size {arena_size} bytes"
                )
            arena_size = spec.buffer_size_bytes
        cursor = _align_to(cursor, 16)
        for memdesc in allocs:
            offsets[memdesc.value_id] = cursor
        cursor += arena_size

    return _LdsLayout(_align_to(cursor, 16), offsets)



def _async_address_by_token(plan):
    return {
        address.token_value_id: address
        for address in plan.addresses
        if address.op in {"ttg.async_copy_global_to_local", "amdg.buffer_load_to_local"}
        and address.token_value_id is not None
    }


def _values_by_id(plan):
    return {value.value_id: value for value in plan.values}


def _memdescs_by_id(plan):
    return {memdesc.value_id: memdesc for memdesc in plan.memdescs}


def _local_load_address_by_result(plan):
    return {
        address.result_value_id: address
        for address in plan.addresses
        if address.op == "ttg.local_load" and address.result_value_id is not None
    }


def _bridge_stage(plan):
    return "ttgir-op-lowering"


def _status_for_stage(stage):
    return f"emitted_wave_{stage.replace('-', '_')}"


def _validate_compute_support(attrs, plan):
    if plan.op_counts.get("tt.dot_scaled", 0):
        raise ValueError("tlx_wave bridge does not support tt.dot_scaled")
    if not plan.op_counts.get("tt.dot", 0):
        return
    if attrs.num_ctas != 1:
        raise ValueError(
            "tlx_wave bridge supports tt.dot only for one CTA; "
            f"got ttg.num-ctas={attrs.num_ctas}. split-K or multi-CTA "
            "accumulation are unsupported"
        )
    for op in getattr(plan, "ops", ()):
        if "atomic" in op.name:
            raise ValueError(
                "tlx_wave bridge does not support split-K or multi-CTA "
                f"accumulation for tt.dot; found {op.name}"
            )



def _entry_name(mod):
    try:
        return mod.get_entry_func_name()
    except Exception:
        return "tlx_wave_kernel"


def _required_int_attr(op, name):
    value = op.get_int_attr(name)
    if value is None:
        raise ValueError(f"tlx_wave bridge expected TTGIR module attribute {name}")
    return value


def _module_attrs(mod):
    op = mod.get_operation()
    target = op.get_str_attr("ttg.target")
    if target is None:
        raise ValueError("tlx_wave bridge expected TTGIR module attribute ttg.target")
    num_ctas = _required_int_attr(op, "ttg.num-ctas")
    num_warps = _required_int_attr(op, "ttg.num-warps")
    threads_per_warp = _required_int_attr(op, "ttg.threads-per-warp")
    has_explicit_local_mem_access = bool(
        op.get_bool_attr("tlx.has_explicit_local_mem_access")
    )
    return _ModuleAttrs(
        target, num_ctas, num_warps, threads_per_warp, has_explicit_local_mem_access
    )


def _public_tt_func_ops(mod):
    funcs = []

    def visit(op):
        if op.get_name() == "tt.func" and op.get_str_attr("sym_visibility") == "public":
            funcs.append(op)
        return True

    mod.walk(visit)
    return tuple(funcs)


def _tt_func_ops(mod):
    funcs = []

    def visit(op):
        if op.get_name() == "tt.func":
            funcs.append(op)
        return True

    mod.walk(visit)
    return tuple(funcs)


def _arg_int_attr(arg_attrs, name):
    if arg_attrs is None:
        return None
    prefix = f"{name} = "
    for part in str(arg_attrs).strip("{}").split(","):
        part = part.strip()
        if not part.startswith(prefix):
            continue
        raw_value = part[len(prefix) :].split(":", 1)[0].strip()
        try:
            return int(raw_value)
        except ValueError:
            return None
    return None


def _kernel_from_module(mod):
    funcs = _public_tt_func_ops(mod)
    if len(funcs) != 1:
        names = (
            ", ".join(func.get_str_attr("sym_name") or "<unnamed>" for func in funcs)
            or "none"
        )
        raise ValueError(
            f"tlx_wave bridge supports exactly one public tt.func kernel, found {len(funcs)} ({names})"
        )

    op = funcs[0]
    arg_attrs = dict(op.get_attrs()).get("arg_attrs") or ()
    name = op.get_str_attr("sym_name") or _entry_name(mod)
    helper_funcs = [
        func
        for func in _tt_func_ops(mod)
        if func.get_str_attr("sym_name") != name
    ]
    if helper_funcs:
        names = ", ".join(
            func.get_str_attr("sym_name") or "<unnamed>" for func in helper_funcs
        )
        raise ValueError(
            "tlx_wave bridge currently supports exactly one tt.func and cannot "
            f"preserve or inline private/helper tt.func definitions: {names}"
        )
    fn = mod.get_function(name)

    args = []
    for index in range(fn.get_num_args()):
        name_for_arg = f"arg{index}"
        ttgir_type_obj = fn.args(index).get_type()
        ttgir_type = _type_str(ttgir_type_obj)
        wave_type, kind = _wave_arg_type(name_for_arg, ttgir_type_obj)
        args.append(
            _KernelArg(
                index,
                name_for_arg,
                ttgir_type_obj,
                ttgir_type,
                _type_kind(ttgir_type_obj),
                wave_type,
                kind,
                _arg_int_attr(
                    arg_attrs[index] if index < len(arg_attrs) else None,
                    "tt.divisibility",
                ),
                _arg_int_attr(
                    arg_attrs[index] if index < len(arg_attrs) else None,
                    "tt.pointer_range",
                ),
            )
        )
    return _Kernel(name, tuple(args), op.get_bool_attr("noinline"))


def _validate_target(options, attrs):
    if options.arch not in {"gfx942", "gfx950"}:
        raise ValueError(
            f"tlx_wave bridge only supports gfx942/gfx950 Wave skeletons, got {options.arch}"
        )
    if options.warp_size != 64:
        raise ValueError(
            f"tlx_wave bridge only supports wave64 inputs, got warp_size={options.warp_size}"
        )
    expected_target = f"hip:{options.arch}"
    if attrs.target != expected_target:
        raise ValueError(
            f"tlx_wave bridge expected TTGIR target {expected_target}, got {attrs.target}"
        )
    if attrs.threads_per_warp != 64:
        raise ValueError(
            "tlx_wave bridge only supports wave64 TTGIR, "
            f'got "ttg.threads-per-warp" = {attrs.threads_per_warp}'
        )


def _target_triple(attrs):
    return attrs.target.replace("hip:", "amdgcn-amd-amdhsa--")
