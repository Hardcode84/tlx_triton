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
    values: tuple[_ValuePlan, ...]
    addresses: tuple[_AddressExprPlan, ...]
    layouts: tuple[_LayoutPlan, ...]
    memdescs: tuple[_MemDescPlan, ...]
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
    load_store_fallbacks: int = 0
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
    "arith.muli",
    "tt.addptr",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.get_program_id",
    "tt.make_range",
    "tt.splat",
    "ttg.convert_layout",
}

_ORDERED_BODY_EFFECT_OPS = {
    "tt.dot",
    "tt.store",
    "ttg.async_commit_group",
    "ttg.async_copy_global_to_local",
    "ttg.async_wait",
    "ttg.local_load",
}

_ORDERED_BODY_PLANNED_OPS = {
    "tt.return",
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
class _DotOperandEncodingInfo:
    op_idx: int
    k_width: int
    parent: _BlockedEncodingInfo


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


def _type_str(type_obj):
    return str(type_obj)


def _scalar_byte_width(type_obj):
    if type_obj is None:
        return None
    if type_obj.is_integer(1) or type_obj.is_integer(8):
        return 1
    if type_obj.is_integer(16) or type_obj.is_fp16() or type_obj.is_bf16():
        return 2
    if type_obj.is_integer(32) or type_obj.is_fp32():
        return 4
    if type_obj.is_integer(64) or type_obj.is_fp64() or type_obj.is_index():
        return 8
    return None


def _type_kind(type_obj):
    if type_obj.is_memdesc():
        return "memdesc"
    if type_obj.is_ranked_tensor():
        return "tensor"
    if type_obj.is_ptr():
        return "pointer"
    if type_obj.is_async_token():
        return "token"
    if _is_scalar_type(type_obj):
        return "scalar"
    return "other"


def _type_plan(type_obj):
    element_type = type_obj.get_element_type()
    pointee_type = type_obj.get_pointee_type()
    encoding_attr = type_obj.get_encoding()
    if pointee_type is None and element_type is not None:
        pointee_type = element_type.get_pointee_type()
    element_byte_width = _scalar_byte_width(element_type)
    if element_byte_width is None:
        element_byte_width = _scalar_byte_width(pointee_type)
    if element_byte_width is None and _is_scalar_type(type_obj):
        element_byte_width = _scalar_byte_width(type_obj)
    return _TypePlan(
        _type_str(type_obj),
        _type_kind(type_obj),
        _tuple_or_empty(type_obj.get_shape()),
        _type_str(element_type) if element_type is not None else None,
        element_byte_width,
        _type_str(pointee_type) if pointee_type is not None else None,
        _attr_str(encoding_attr),
        encoding_attr,
        _attr_str(type_obj.get_memory_space()),
        type_obj.get_mutable_memory(),
        _tuple_or_empty(type_obj.get_alloc_shape()),
        type_obj.get_address_space(),
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
        type_obj.is_index()
        or type_obj.is_fp16()
        or type_obj.is_bf16()
        or type_obj.is_fp32()
        or type_obj.is_fp64()
        or any(type_obj.is_integer(width) for width in (1, 8, 16, 32, 64))
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


def _validate_straight_line_kernel_ops(ops):
    for op in ops:
        if op.get_num_regions():
            raise ValueError(
                "tlx_wave bridge currently supports only straight-line TTGIR "
                "kernel bodies; unsupported control-flow or nested-region op "
                f"{op.get_name()} has {op.get_num_regions()} nested region(s)"
            )


def _validate_ordered_body_supported_ops(ops):
    for op in ops:
        name = op.get_name()
        if name not in _ORDERED_BODY_SUPPORTED_OPS:
            raise ValueError(
                "tlx_wave bridge cannot lower unsupported TTGIR op in ordered "
                f"body lowering: {name}"
            )


def _op_results(op):
    return tuple(op.get_result(index) for index in range(op.get_num_results()))


def _op_operands(op):
    return tuple(op.get_operand(index) for index in range(op.get_num_operands()))


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
    elif op_name in {"tt.broadcast", "tt.splat", "ttg.convert_layout"}:
        varying_dims = (
            operand_plans[0].varying_dims if operand_plans and operand_plans[0] else ()
        )
        kind = op_name.split(".")[-1]
    elif op_name in {"arith.addi", "arith.muli", "arith.andi", "arith.cmpi"}:
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


_MEMDESC_VIEW_OPS = {
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
        elif name in _MEMDESC_VIEW_OPS:
            result = op.get_result(0)
            operands = _op_operands(op)
            static_index = (
                _const_int(values.get(_value_id(operands[1])))
                if name == "ttg.memdesc_index" and len(operands) > 1
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
    result_value_id = token_value_id = None

    if name == "ttg.async_copy_global_to_local":
        address_value, memdesc_value, mask_value, other_value = _async_copy_operands(op)
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

    source_value = address_value or memdesc_value
    source_plan = (
        values.get(_value_id(source_value)) if source_value is not None else None
    )
    offset_value_id = None
    if address_value is not None:
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
        source_plan.base_arg_index if source_plan is not None else None,
        source_plan.base_arg_name if source_plan is not None else None,
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


def _validate_address_feature_support(addresses):
    for address in addresses:
        if (
            address.op == "ttg.async_copy_global_to_local"
            and address.other_value_id is not None
        ):
            raise ValueError(
                "tlx_wave bridge cannot lower ttg.async_copy_global_to_local "
                "with `other` fill values yet"
            )


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
    memdescs_by_id = _build_memdesc_plans(ops, values_by_id)
    address_plans = _build_address_plans(ops, values_by_id, owners)
    _validate_address_feature_support(address_plans)
    return _BridgePlan(
        "ttgir_graph",
        op_counts,
        op_plans,
        tuple(values_by_id.values()),
        address_plans,
        _build_layout_plans(ops),
        tuple(memdescs_by_id.values()),
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
        "num_memdescs": len(plan.memdescs),
        "num_tokens": len(plan.tokens),
        "ops": _public_dicts(plan.ops),
        "values": _public_dicts(plan.values),
        "addresses": _public_dicts(plan.addresses),
        "layouts": _public_dicts(plan.layouts),
        "memdescs": _public_dicts(plan.memdescs),
        "tokens": _public_dicts(plan.tokens),
    }


def _memdesc_size_bytes(memdesc):
    if memdesc.element_byte_width is None:
        raise ValueError(
            f"tlx_wave bridge cannot size LDS memdesc {memdesc.name or memdesc.source} "
            f"with element type {memdesc.element_type}"
        )
    shape = memdesc.alloc_shape or memdesc.shape
    return _product(shape) * memdesc.element_byte_width


def _compute_lds_layout(plan):
    offsets = {}
    memdescs = {memdesc.value_id: memdesc for memdesc in plan.memdescs}
    cursor = 0

    for memdesc in plan.memdescs:
        if memdesc.kind != "allocation":
            continue
        cursor = _align_to(cursor, 16)
        offsets[memdesc.value_id] = cursor
        cursor += _memdesc_size_bytes(memdesc)

    def assign_view(memdesc):
        if memdesc.value_id in offsets:
            return offsets[memdesc.value_id]
        if memdesc.base_value_id is None or memdesc.base_value_id not in memdescs:
            raise ValueError(
                f"tlx_wave bridge cannot place LDS view {memdesc.name or memdesc.source} "
                "without a known base memdesc"
            )
        base = memdescs[memdesc.base_value_id]
        offset = assign_view(base)
        if memdesc.view_op == "ttg.memdesc_index" and memdesc.static_index is not None:
            offset += memdesc.static_index * _memdesc_size_bytes(memdesc)
        offsets[memdesc.value_id] = offset
        return offset

    high_watermark = cursor
    for memdesc in plan.memdescs:
        if memdesc.kind == "view":
            offset = assign_view(memdesc)
            high_watermark = max(high_watermark, offset + _memdesc_size_bytes(memdesc))

    return _LdsLayout(_align_to(high_watermark, 16), offsets)



def _async_address_by_token(plan):
    return {
        address.token_value_id: address
        for address in plan.addresses
        if address.op == "ttg.async_copy_global_to_local"
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
            )
        )
    return _Kernel(name, tuple(args), op.get_bool_attr("noinline"))


def _validate_target(options, attrs):
    if options.arch != "gfx950":
        raise ValueError(
            f"tlx_wave bridge only supports gfx950 Wave skeletons, got {options.arch}"
        )
    if options.warp_size != 64:
        raise ValueError(
            f"tlx_wave bridge only supports wave64 inputs, got warp_size={options.warp_size}"
        )
    if attrs.target != "hip:gfx950":
        raise ValueError(
            f"tlx_wave bridge only supports TTGIR target hip:gfx950, got {attrs.target}"
        )
    if attrs.threads_per_warp != 64:
        raise ValueError(
            "tlx_wave bridge only supports wave64 TTGIR, "
            f'got "ttg.threads-per-warp" = {attrs.threads_per_warp}'
        )


def _target_triple(attrs):
    return attrs.target.replace("hip:", "amdgcn-amd-amdhsa--")
