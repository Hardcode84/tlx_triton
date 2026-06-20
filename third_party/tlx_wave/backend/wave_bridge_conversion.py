"""DialectConversion-style scaffolding for the next TLX Wave bridge.

This is intentionally separate from ``wave_bridge_emit.py``.  The types here are
small contracts for a new converter:

* type conversion is stateless;
* rewrite patterns are stateless and receive all facts through context;
* token ordering is precomputed as a token graph before rewrites run;
* rewriters consume only op operands/results/attrs and context maps, not def-use
  walks.
"""

from dataclasses import dataclass, field, replace
import re

from .wave_bridge_token_graph import (
    _TokenGraph,
    _build_token_graph,
    _collect_token_events,
)


@dataclass(frozen=True)
class _SourceType:
    raw: str
    kind: str
    shape: tuple[int, ...] = ()
    element_type: str | None = None
    element_byte_width: int | None = None
    pointee_type: str | None = None
    encoding: str | None = None
    encoding_attr: object | None = None
    memory_space: str | None = None
    mutable: bool | None = None
    alloc_shape: tuple[int, ...] = ()
    address_space: int | None = None
    pointer_range: int | None = None
    divisibility: int | None = None


@dataclass(frozen=True)
class _ConvertedType:
    kind: str
    wave_type: str
    value_representation: str
    lane_width: int | None = None
    component_count: int = 1


@dataclass(frozen=True)
class _AMDMfmaEncodingInfo:
    version: int
    warps_per_cta: tuple[int, ...]
    instr_shape: tuple[int, ...]
    is_transposed: bool
    tiles_per_warp: tuple[int, ...]
    element_bit_width: int


@dataclass(frozen=True)
class _BlockedEncodingInfo:
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]


@dataclass(frozen=True)
class _MmaShapeInfo:
    instr_shape: tuple[int, int, int]
    fragment_shape: tuple[int, int]
    output_tile_shape: tuple[int, int]
    k_dim: int
    wave_size: int
    operand_registers: int
    acc_registers: int
    kind_suffix: str


@dataclass(frozen=True)
class _DotOperandEncodingInfo:
    op_idx: int
    k_width: int
    parent: object


@dataclass(frozen=True)
class _SourceValue:
    value_id: int
    type: _SourceType
    producer: str | None = None


@dataclass(frozen=True)
class _SourceOp:
    index: int
    name: str
    operands: tuple[int, ...] = ()
    results: tuple[int, ...] = ()
    attrs: dict = field(default_factory=dict)
    regions: tuple[tuple[int, ...], ...] = ()
    region_args: tuple[tuple[int, ...], ...] = ()
    parent_index: int | None = None
    region_index: int | None = None


@dataclass(frozen=True)
class _RewriteResult:
    values: dict[int, object] = field(default_factory=dict)
    effects: tuple[object, ...] = ()


@dataclass(frozen=True)
class _RegionYield:
    operand_ids: tuple[int, ...]
    operands: tuple[object, ...]
    op_index: int = -1


@dataclass(frozen=True)
class _RegionRewriteResult:
    values: dict[int, object]
    effects: tuple[object, ...]
    yield_operand_ids: tuple[int, ...] = ()
    yield_operands: tuple[object, ...] = ()


@dataclass(frozen=True)
class _ConvertedInputValue:
    value_id: int
    source: str | None
    converted_type: _ConvertedType


@dataclass(frozen=True)
class _ConversionInput:
    ops: tuple[_SourceOp, ...]
    values: dict[int, _SourceValue]
    token_graph: _TokenGraph
    kernel_name: str | None = None
    kernel_args: tuple[int, ...] = ()
    target: str | None = None
    num_ctas: int | None = None
    num_warps: int | None = None
    threads_per_warp: int | None = None
    has_explicit_local_mem_access: bool | None = None
    noinline: bool | None = None
    top_level_ops: tuple[int, ...] = ()


@dataclass(frozen=True)
class _ConversionRun:
    program: _ConversionInput
    result: _RewriteResult


@dataclass(frozen=True)
class _ConversionContext:
    program: _ConversionInput
    type_converter: object
    registry: object | None = None

    def value(self, value_id):
        return self.program.values[value_id]

    def converted_type(self, value_id):
        return self.type_converter.convert_value(self.value(value_id))

    def token_node(self, value_id):
        return self.program.token_graph.node_for_value(value_id)

    def convert_region(self, op, region_index, values):
        if self.registry is None:
            raise ValueError("tlx_wave conversion has no region rewriter registry")
        if region_index >= len(op.regions):
            raise ValueError(
                f"tlx_wave conversion cannot convert region {region_index} of {op.name}"
            )
        return _run_rewriter_sequence(
            self,
            self.registry,
            values,
            op.regions[region_index],
        )


def _prepare_conversion_input(ops, values):
    values_by_id = dict(values)
    token_events = _collect_token_events(tuple(ops), values_by_id)
    return _ConversionInput(
        tuple(ops),
        values_by_id,
        _build_token_graph(token_events),
        top_level_ops=tuple(range(len(ops))),
    )


def _import_ttgir_conversion_input(mod, kernel_name=None):
    """Import TTGIR into the explicit source program used by new rewriters."""

    kernel_name = _resolve_kernel_name(mod, kernel_name)
    fn = mod.get_function(kernel_name)
    if fn is None:
        raise ValueError(f"tlx_wave conversion cannot find tt.func @{kernel_name}")
    func_op = _kernel_func_op(mod, kernel_name)
    raw_ops, regions_by_index, parents_by_index, top_level_ops = _walk_kernel_ops(
        mod, kernel_name
    )
    source_ops = tuple(
        _SourceOp(
            index,
            raw_op.get_name(),
            tuple(_value_id(raw_op.get_operand(i)) for i in range(raw_op.get_num_operands())),
            tuple(_value_id(raw_op.get_result(i)) for i in range(raw_op.get_num_results())),
            _source_attrs(raw_op),
            regions_by_index[index],
            _region_arg_ids(raw_op),
            parents_by_index[index][0],
            parents_by_index[index][1],
        )
        for index, raw_op in enumerate(raw_ops)
    )
    values = _source_values(mod, kernel_name, raw_ops)
    return _ConversionInput(
        source_ops,
        values,
        _build_token_graph(_collect_token_events(source_ops, values)),
        kernel_name,
        tuple(_value_id(fn.args(index)) for index in range(fn.get_num_args())),
        _module_str_attr(mod, "ttg.target"),
        _module_int_attr(mod, "ttg.num-ctas"),
        _module_int_attr(mod, "ttg.num-warps"),
        _module_int_attr(mod, "ttg.threads-per-warp"),
        bool(_module_bool_attr(mod, "tlx.has_explicit_local_mem_access")),
        func_op.get_bool_attr("noinline") if func_op is not None else None,
        top_level_ops,
    )


class _WaveTypeConverter:
    """Stateless source-to-Wave type converter.

    The converter returns a structural description instead of constructing MLIR
    types directly.  Emission-specific code can later lower ``wave_type`` into a
    concrete Wave Python binding type.
    """

    def convert_type(self, source_type, *, lane_width=64, component_count=1):
        if source_type.kind == "token":
            return _ConvertedType("token", "!wave.mem.token", "token")
        if source_type.kind == "pointer":
            address_space = _wave_address_space(source_type)
            pointee = source_type.pointee_type or "i8"
            return _ConvertedType(
                "pointer",
                f"!wave.ptr<#{address_space}, {pointee}>",
                "pointer",
            )
        if source_type.kind == "tensor":
            if source_type.element_type == "i1":
                if component_count == 1:
                    component_count = _mask_component_count(source_type, lane_width)
                return _ConvertedType(
                    "mask",
                    f"!wave.mask<{int(lane_width)}>",
                    "mask",
                    lane_width=int(lane_width),
                    component_count=int(component_count),
                )
            if component_count == 1:
                component_count = _dense_tensor_component_count(
                    source_type, lane_width
                )
            fragment = _fragment_converted_type(source_type)
            if fragment is not None:
                return fragment
            if source_type.pointee_type is not None:
                address_space = _wave_address_space(source_type)
                element = f"!wave.ptr<#{address_space}, {source_type.pointee_type}>"
            else:
                element = source_type.element_type or "unknown"
            representation = "simd" if component_count == 1 else "simd_tuple"
            return _ConvertedType(
                "tensor",
                f"!wave.simd<{element}, {int(lane_width)}>",
                representation,
                lane_width=int(lane_width),
                component_count=int(component_count),
            )
        if source_type.kind == "scalar":
            return _ConvertedType("scalar", _scalar_wave_type(source_type), "scalar")
        if source_type.kind == "memdesc":
            return _ConvertedType("memdesc", source_type.raw, "memdesc")
        raise ValueError(
            f"tlx_wave conversion cannot convert unsupported type kind "
            f"{source_type.kind}: {source_type.raw}"
        )

    def convert_value(self, source_value, *, lane_width=64, component_count=1):
        return self.convert_type(
            source_value.type,
            lane_width=lane_width,
            component_count=component_count,
        )


class _OpRewritePattern:
    op_name = None

    def match(self, op):
        return op.name == self.op_name

    def rewrite(self, context, op, values):
        raise NotImplementedError


@dataclass(frozen=True)
class _RewriterRegistry:
    patterns: dict[str, _OpRewritePattern]

    @classmethod
    def from_patterns(cls, patterns):
        by_name = {}
        for pattern in patterns:
            if not pattern.op_name:
                raise ValueError("tlx_wave conversion rewriter has no op_name")
            if pattern.op_name in by_name:
                raise ValueError(
                    f"tlx_wave conversion has duplicate rewriter for {pattern.op_name}"
                )
            by_name[pattern.op_name] = pattern
        return cls(by_name)

    def rewriter_for(self, op):
        pattern = self.patterns.get(op.name)
        if pattern is None:
            raise ValueError(f"tlx_wave conversion has no rewriter for {op.name}")
        return pattern

    def has_rewriter(self, op):
        return op.name in self.patterns


def _missing_rewriter_op_names(program, registry):
    missing = []
    seen = set()
    for op in program.ops:
        if registry.has_rewriter(op) or op.name in seen:
            continue
        seen.add(op.name)
        missing.append(op.name)
    return tuple(missing)


def _run_rewriters(program, registry, type_converter=None, initial_values=None):
    """Run stateless op rewriters in source order.

    ``values`` is the only mutable conversion product map.  Rewriters receive it
    as an explicit argument and must not recover values by walking IR def-use.
    """

    missing = _missing_rewriter_op_names(program, registry)
    if missing:
        raise ValueError(
            "tlx_wave conversion has no rewriters for: " + ", ".join(missing)
        )

    converter = _WaveTypeConverter() if type_converter is None else type_converter
    context = _ConversionContext(program, converter, registry)
    converted_values = _initial_converted_values(program, converter)
    if initial_values is not None:
        converted_values.update(initial_values)
    region_result = _run_rewriter_sequence(
        context,
        registry,
        converted_values,
        _top_level_op_indices(program),
    )
    if region_result.yield_operand_ids:
        raise ValueError("tlx_wave conversion encountered scf.yield at top level")
    return _RewriteResult(region_result.values, region_result.effects)


def _run_rewriter_sequence(context, registry, initial_values, op_indices):
    converted_values = dict(initial_values)
    effects = []
    for op_index in op_indices:
        op = context.program.ops[op_index]
        result = registry.rewriter_for(op).rewrite(context, op, converted_values)
        converted_values.update(result.values)
        for effect in result.effects:
            if isinstance(effect, _RegionYield):
                return _RegionRewriteResult(
                    converted_values,
                    tuple(effects),
                    effect.operand_ids,
                    effect.operands,
                )
            effects.append(effect)
    return _RegionRewriteResult(converted_values, tuple(effects))


def _top_level_op_indices(program):
    if program.top_level_ops:
        return program.top_level_ops
    return tuple(op.index for op in program.ops if op.parent_index is None)


def _convert_imported_ttgir(
    mod, registry, *, kernel_name=None, type_converter=None, initial_values=None
):
    program = _import_ttgir_conversion_input(mod, kernel_name)
    return _ConversionRun(
        program,
        _run_rewriters(
            program,
            registry,
            type_converter=type_converter,
            initial_values=initial_values,
        ),
    )


def _initial_converted_values(program, type_converter):
    values = {}
    for value_id, source_value in program.values.items():
        if not _source_value_is_input(source_value):
            continue
        values[value_id] = _ConvertedInputValue(
            value_id,
            source_value.producer,
            type_converter.convert_value(source_value),
        )
    return values


def _source_value_is_input(source_value):
    producer = source_value.producer or ""
    return producer == "block_argument" or producer.startswith("arg")


_GFX950_MMA16 = _MmaShapeInfo(
    (16, 16, 32),
    (16, 16),
    (32, 32),
    32,
    64,
    4,
    4,
    "16x16x32",
)
_GFX950_MMA32 = _MmaShapeInfo(
    (32, 32, 16),
    (32, 32),
    (32, 32),
    16,
    64,
    4,
    16,
    "32x32x16",
)
_GFX942_MMA16 = _MmaShapeInfo(
    (16, 16, 16),
    (16, 16),
    (32, 32),
    16,
    64,
    2,
    4,
    "16x16x16",
)


def _fragment_converted_type(source_type):
    dot = _dot_operand_encoding_info(source_type)
    if dot is not None:
        mma = _mma_shape_for_parent(dot.parent, "ttg.dot operand")
        return _ConvertedType(
            "fragment",
            _fragment_type_text(dot.op_idx, source_type.element_type, mma.operand_registers, mma),
            "fragment_tuple"
            if _dot_operand_tile_count(source_type, dot, mma) != 1
            else "fragment",
            lane_width=mma.wave_size,
            component_count=_dot_operand_tile_count(source_type, dot, mma),
        )
    parent = _amd_mfma_encoding_info(source_type.encoding_attr)
    if parent is None:
        blocked = _blocked_encoding_info(source_type.encoding_attr)
        if blocked is None or not _is_supported_gfx950_blocked_dot_parent(blocked):
            return None
        if len(source_type.shape) != 2 or source_type.element_type not in {"f32", "f16"}:
            return None
        mma = _GFX950_MMA16
        return _ConvertedType(
            "fragment",
            _fragment_type_text(2, source_type.element_type, mma.acc_registers, mma),
            "fragment_tuple" if _acc_tile_count(source_type, mma) != 1 else "fragment",
            lane_width=mma.wave_size,
            component_count=_acc_tile_count(source_type, mma),
        )
    mma = _mma_shape_for_parent(parent, "ttg.amd_mfma tensor")
    return _ConvertedType(
        "fragment",
        _fragment_type_text(2, source_type.element_type, mma.acc_registers, mma),
        "fragment_tuple" if _acc_tile_count(source_type, mma) != 1 else "fragment",
        lane_width=mma.wave_size,
        component_count=_acc_tile_count(source_type, mma),
    )


def _fragment_type_text(role, element_type, registers, mma):
    element = element_type or "unknown"
    return (
        f"!waveamd.fragment<{int(role)}, {element}, "
        f"{mma.fragment_shape[0]}, {mma.fragment_shape[1]}, "
        f"{mma.wave_size}, {int(registers)}>"
    )


def _dot_operand_tile_count(source_type, info, mma):
    if len(source_type.shape) != 2:
        return 1
    if info.op_idx == 0:
        return _tile_rep_count(source_type.shape[0], mma.output_tile_shape[0]) * _tile_rep_count(
            source_type.shape[1], mma.k_dim
        )
    if info.op_idx == 1:
        return _tile_rep_count(source_type.shape[0], mma.k_dim) * _tile_rep_count(
            source_type.shape[1], mma.output_tile_shape[1]
        )
    return 1


def _acc_tile_count(source_type, mma):
    if len(source_type.shape) != 2:
        return 1
    return _tile_rep_count(source_type.shape[0], mma.output_tile_shape[0]) * _tile_rep_count(
        source_type.shape[1], mma.output_tile_shape[1]
    )


def _tile_rep_count(dim, tile):
    dim = int(dim)
    tile = int(tile)
    if tile <= 0:
        raise ValueError("tlx_wave conversion expected positive tile extent")
    if dim <= tile:
        return 1
    if dim % tile:
        raise ValueError(
            f"tlx_wave conversion expected dimension {dim} to be a multiple of {tile}"
        )
    return dim // tile


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _dense_tensor_component_count(source_type, lane_width):
    element_count = 1
    for dim in source_type.shape:
        element_count *= int(dim)
    dense_count = max(1, _ceil_div(element_count, int(lane_width)))
    layout_count = _rank1_blocked_component_count(
        source_type,
        element_count,
        lane_width,
    )
    if layout_count is not None:
        return layout_count
    return dense_count


def _rank1_blocked_component_count(source_type, element_count, lane_width):
    if len(tuple(source_type.shape)) != 1:
        return None
    layout = _blocked_encoding_info(source_type.encoding_attr)
    if layout is None or len(layout.size_per_thread) != 1:
        return None
    elements_per_thread = int(layout.size_per_thread[0])
    if elements_per_thread <= 1:
        return None
    if int(element_count) <= int(lane_width) * elements_per_thread:
        return elements_per_thread
    return None


def _mask_component_count(source_type, lane_width):
    parent = _amd_mfma_encoding_info(source_type.encoding_attr)
    if parent is not None:
        return _acc_tile_count(source_type, _mma_shape_for_parent(parent, "i1 tensor"))
    return _dense_tensor_component_count(source_type, lane_width)


def _dot_operand_encoding_info(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_dot_operand_encoding"):
        return None
    parent_attr = _attr_value(attr, "get_dot_operand_parent")
    parent = _amd_mfma_encoding_info(parent_attr)
    if parent is None:
        parent = _blocked_encoding_info(parent_attr)
    if parent is None:
        return None
    return _DotOperandEncodingInfo(
        int(_attr_value(attr, "get_dot_operand_op_idx")),
        int(_attr_value(attr, "get_dot_operand_k_width")),
        parent,
    )


def _amd_mfma_encoding_info(attr):
    if attr is None or not _attr_bool(attr, "is_amd_mfma_encoding"):
        return None
    return _AMDMfmaEncodingInfo(
        int(_attr_value(attr, "get_amd_mfma_version")),
        tuple(int(value) for value in _attr_value(attr, "get_amd_mfma_warps_per_cta")),
        tuple(int(value) for value in _attr_value(attr, "get_amd_mfma_instr_shape")),
        bool(_attr_value(attr, "get_amd_mfma_is_transposed")),
        tuple(int(value) for value in _attr_value(attr, "get_amd_mfma_tiles_per_warp")),
        int(_attr_value(attr, "get_amd_mfma_element_bit_width")),
    )


def _blocked_encoding_info(attr):
    if attr is None or not _attr_bool(attr, "is_blocked_encoding"):
        return None
    return _BlockedEncodingInfo(
        tuple(int(value) for value in _attr_value(attr, "get_blocked_size_per_thread")),
        tuple(int(value) for value in _attr_value(attr, "get_blocked_threads_per_warp")),
        tuple(int(value) for value in _attr_value(attr, "get_blocked_warps_per_cta")),
        tuple(int(value) for value in _attr_value(attr, "get_blocked_order")),
    )


def _mma_shape_for_parent(parent, context):
    if isinstance(parent, _BlockedEncodingInfo):
        if _is_supported_gfx950_blocked_dot_parent(parent):
            return _GFX950_MMA16
        supported = ", ".join(
            _format_blocked_encoding(layout)
            for layout in _GFX950_BLOCKED_DOT_PARENT_MMA16_LAYOUTS
        )
        raise ValueError(
            "tlx_wave conversion supports blocked dot operand parents only for "
            "known gfx950 16x16x32 MFMA layouts; got "
            f"{_format_blocked_encoding(parent)}; supported layouts: {supported}"
        )
    if parent.version == 4 and parent.is_transposed and parent.instr_shape == (16, 16, 32):
        return _with_parent_cta_tile_shape(_GFX950_MMA16, parent, context)
    if parent.version == 4 and parent.is_transposed and parent.instr_shape == (32, 32, 16):
        return _with_parent_cta_tile_shape(_GFX950_MMA32, parent, context)
    if parent.version == 3 and parent.is_transposed and parent.instr_shape == (16, 16, 16):
        if parent.warps_per_cta != (2, 2) or not _all_ones(parent.tiles_per_warp):
            raise ValueError(
                "tlx_wave conversion currently supports gfx942 MFMA16 only "
                f"for warpsPerCTA=(2, 2), unit tilesPerWarp; got "
                f"warpsPerCTA={parent.warps_per_cta}, "
                f"tilesPerWarp={parent.tiles_per_warp} while lowering {context}"
            )
        return _with_parent_cta_tile_shape(_GFX942_MMA16, parent, context)
    raise ValueError(
        "tlx_wave conversion supports only gfx950 MFMA "
        "16x16x32/32x32x16 or gfx942 MFMA 16x16x16 transposed layouts; "
        f"got version={parent.version}, instrShape={parent.instr_shape}, "
        f"isTransposed={parent.is_transposed} while lowering {context}"
    )


_GFX950_BLOCKED_DOT_PARENT_MMA16_LAYOUTS = (
    _BlockedEncodingInfo((2, 2), (4, 16), (4, 1), (1, 0)),
    _BlockedEncodingInfo((1, 4), (4, 16), (4, 1), (1, 0)),
)


def _is_supported_gfx950_blocked_dot_parent(parent):
    return any(parent == layout for layout in _GFX950_BLOCKED_DOT_PARENT_MMA16_LAYOUTS)


def _format_blocked_encoding(layout):
    return (
        f"sizePerThread={layout.size_per_thread}, "
        f"threadsPerWarp={layout.threads_per_warp}, "
        f"warpsPerCTA={layout.warps_per_cta}, order={layout.order}"
    )


def _with_parent_cta_tile_shape(mma, parent, context):
    if len(parent.warps_per_cta) < 2 or len(parent.tiles_per_warp) < 2:
        raise ValueError(
            "tlx_wave conversion expected rank-2 MFMA warpsPerCTA/tilesPerWarp "
            f"while lowering {context}; got warpsPerCTA={parent.warps_per_cta}, "
            f"tilesPerWarp={parent.tiles_per_warp}"
        )
    output_tile_shape = (
        int(mma.instr_shape[0])
        * int(parent.tiles_per_warp[-2])
        * int(parent.warps_per_cta[-2]),
        int(mma.instr_shape[1])
        * int(parent.tiles_per_warp[-1])
        * int(parent.warps_per_cta[-1]),
    )
    return replace(mma, output_tile_shape=output_tile_shape)


def _all_ones(values):
    return bool(values) and all(int(value) == 1 for value in values)


def _attr_bool(attr, method):
    value = _attr_value(attr, method)
    return bool(value) if value is not None else False


def _attr_value(attr, method):
    if attr is None or not hasattr(attr, method):
        return None
    return getattr(attr, method)()


def _wave_address_space(source_type):
    if source_type.memory_space in {"shared", "workgroup"} or source_type.address_space == 3:
        return "wave.shared"
    return "wave.global"


def _scalar_wave_type(source_type):
    raw = source_type.raw
    if raw in {"i1", "i8", "i16", "i32", "i64", "index", "f16", "bf16", "f32", "f64"}:
        return raw
    if source_type.element_type is not None:
        return source_type.element_type
    return raw


def _resolve_kernel_name(mod, kernel_name):
    if kernel_name is not None:
        return kernel_name
    funcs = []

    def visit(op):
        if op.get_name() == "tt.func" and op.get_str_attr("sym_visibility") == "public":
            funcs.append(op)
        return True

    mod.walk(visit)
    if len(funcs) != 1:
        names = (
            ", ".join(func.get_str_attr("sym_name") or "<unnamed>" for func in funcs)
            or "none"
        )
        raise ValueError(
            "tlx_wave conversion expected exactly one public tt.func kernel, "
            f"found {len(funcs)} ({names})"
        )
    return funcs[0].get_str_attr("sym_name")


def _kernel_func_op(mod, kernel_name):
    found = []

    def visit(op):
        if op.get_name() == "tt.func" and op.get_str_attr("sym_name") == kernel_name:
            found.append(op)
        return True

    mod.walk(visit)
    return found[0] if found else None


def _walk_kernel_ops(mod, kernel_name):
    fn = mod.get_function(kernel_name)
    if fn is None:
        raise ValueError(f"tlx_wave conversion cannot find tt.func @{kernel_name}")
    raw_ops = []
    regions_by_index = {}
    parents_by_index = {}
    top_level_ops = _collect_region_ops(
        fn.get_region(0),
        raw_ops,
        regions_by_index,
        parents_by_index,
    )
    return tuple(raw_ops), regions_by_index, parents_by_index, top_level_ops


def _collect_region_ops(
    region,
    raw_ops,
    regions_by_index,
    parents_by_index,
    parent_index=None,
    region_index=None,
):
    indices = []
    for block_index in range(region.size()):
        block = region.get_block(block_index)
        for op_index in range(block.get_num_operations()):
            op = block.get_operation(op_index)
            index = len(raw_ops)
            raw_ops.append(op)
            parents_by_index[index] = (parent_index, region_index)
            indices.append(index)
            child_regions = []
            for region_index in range(op.get_num_regions()):
                child_regions.append(
                    _collect_region_ops(
                        op.get_region(region_index),
                        raw_ops,
                        regions_by_index,
                        parents_by_index,
                        index,
                        region_index,
                    )
                )
            regions_by_index[index] = tuple(child_regions)
    return tuple(indices)


def _module_str_attr(mod, name):
    return mod.get_operation().get_str_attr(name)


def _module_int_attr(mod, name):
    return mod.get_operation().get_int_attr(name)


def _module_bool_attr(mod, name):
    return mod.get_operation().get_bool_attr(name)


def _walk_region_blocks(region):
    blocks = []
    for block_index in range(region.size()):
        block = region.get_block(block_index)
        blocks.append(block)
        for op_index in range(block.get_num_operations()):
            op = block.get_operation(op_index)
            for region_index in range(op.get_num_regions()):
                blocks.extend(_walk_region_blocks(op.get_region(region_index)))
    return tuple(blocks)


def _source_values(mod, kernel_name, raw_ops):
    fn = mod.get_function(kernel_name)
    kernel_op = _kernel_func_op(mod, kernel_name)
    arg_attrs = dict(kernel_op.get_attrs()).get("arg_attrs") if kernel_op is not None else None
    arg_attrs = () if arg_attrs is None else tuple(arg_attrs)
    values = {}
    for index in range(fn.get_num_args()):
        value = fn.args(index)
        source_type = _source_type(value.get_type())
        source_type = replace(
            source_type,
            pointer_range=_arg_int_attr(
                arg_attrs[index] if index < len(arg_attrs) else None,
                "tt.pointer_range",
            ),
            divisibility=_arg_int_attr(
                arg_attrs[index] if index < len(arg_attrs) else None,
                "tt.divisibility",
            ),
        )
        source_value = _SourceValue(
            _value_id(value),
            source_type,
            producer=f"arg{index}",
        )
        values[source_value.value_id] = source_value
    for block in _walk_region_blocks(fn.get_region(0)):
        for index in range(block.get_num_arguments()):
            value = block.get_argument(index)
            value_id = _value_id(value)
            if value_id not in values:
                values[value_id] = _SourceValue(
                    value_id,
                    _source_type(value.get_type()),
                    producer="block_argument",
                )
    for op in raw_ops:
        for result_index in range(op.get_num_results()):
            value = op.get_result(result_index)
            source_value = _SourceValue(
                _value_id(value),
                _source_type(value.get_type()),
                producer=op.get_name(),
            )
            values[source_value.value_id] = source_value
    return values


def _source_attrs(op):
    attrs = dict(op.get_attrs())
    for name in ("axis", "end", "num", "predicate", "start"):
        value = op.get_int_attr(name)
        if value is not None:
            attrs[name] = int(value)
    segments = op.get_int_array_attr("operandSegmentSizes")
    if segments is not None:
        attrs["operandSegmentSizes"] = tuple(int(segment) for segment in segments)
    return attrs


def _region_arg_ids(op):
    regions = []
    for region_index in range(op.get_num_regions()):
        region = op.get_region(region_index)
        if region.size() == 0:
            regions.append(())
            continue
        block = region.get_block(0)
        regions.append(
            tuple(
                _value_id(block.get_argument(arg_index))
                for arg_index in range(block.get_num_arguments())
            )
        )
    return tuple(regions)


def _arg_int_attr(attrs, name):
    if attrs is None:
        return None
    if isinstance(attrs, str):
        match = re.search(rf"{re.escape(name)}\s*=\s*([+-]?\d+)\s*:", attrs)
        return None if match is None else int(match.group(1))
    attr = dict(attrs).get(name)
    if attr is None:
        return None
    try:
        return int(attr)
    except TypeError:
        return int(str(attr))


def _source_type(type_obj):
    element_type = _type_method(type_obj, "get_element_type")
    pointee_type = _type_method(type_obj, "get_pointee_type")
    encoding_attr = _type_method(type_obj, "get_encoding")
    if pointee_type is None and element_type is not None:
        pointee_type = _type_method(element_type, "get_pointee_type")
    address_space = _type_method(type_obj, "get_address_space")
    if address_space is None and element_type is not None:
        address_space = _type_method(element_type, "get_address_space")
    element_byte_width = _scalar_byte_width(element_type)
    if element_byte_width is None:
        element_byte_width = _scalar_byte_width(pointee_type)
    if element_byte_width is None and _is_scalar_type(type_obj):
        element_byte_width = _scalar_byte_width(type_obj)
    return _SourceType(
        _type_str(type_obj),
        _source_type_kind(type_obj),
        _tuple_or_empty(_type_method(type_obj, "get_shape")),
        _type_str(element_type) if element_type is not None else None,
        element_byte_width,
        _type_str(pointee_type) if pointee_type is not None else None,
        _attr_str(encoding_attr),
        encoding_attr,
        _attr_str(_type_method(type_obj, "get_memory_space")),
        _type_method(type_obj, "get_mutable_memory"),
        _tuple_or_empty(_type_method(type_obj, "get_alloc_shape")),
        address_space,
    )


def _source_type_kind(type_obj):
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


def _is_scalar_type(type_obj):
    return (
        _type_predicate(type_obj, "is_index")
        or _type_predicate(type_obj, "is_fp16")
        or _type_predicate(type_obj, "is_bf16")
        or _type_predicate(type_obj, "is_fp32")
        or _type_predicate(type_obj, "is_fp64")
        or any(_type_is_integer_width(type_obj, width) for width in (1, 8, 16, 32, 64))
    )


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


def _type_is_integer_width(type_obj, width):
    fn = getattr(type_obj, "is_integer", None)
    if fn is None:
        return False
    return bool(fn(width))


def _type_predicate(type_obj, method):
    return bool(_type_method(type_obj, method, False))


def _type_method(type_obj, method, default=None):
    fn = getattr(type_obj, method, None)
    if fn is None:
        return default
    return fn()


def _tuple_or_empty(values):
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _type_str(type_obj):
    return str(type_obj)


def _attr_str(attr):
    return None if attr is None else str(attr)


def _value_id(value):
    return int(value.id())
