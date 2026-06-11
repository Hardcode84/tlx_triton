import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


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
    pointee_type: str | None
    encoding: str | None
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
    pointee_type: str | None
    encoding: str | None
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
    encoding: str | None
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
    encoding: str | None
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
    shape: tuple[int, ...]
    base_arg_index: int | None
    base_arg_name: str | None
    offset_value_id: int | None
    mask_value_id: int | None
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


def _value_id(value):
    return int(value.id())


def _value_type(value):
    return value.get_type()


def _tuple_or_empty(values):
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _attr_str(attr):
    return None if attr is None else str(attr)


def _type_str(type_obj):
    return str(type_obj)


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
    if pointee_type is None and element_type is not None:
        pointee_type = element_type.get_pointee_type()
    return _TypePlan(
        _type_str(type_obj),
        _type_kind(type_obj),
        _tuple_or_empty(type_obj.get_shape()),
        _type_str(element_type) if element_type is not None else None,
        _type_str(pointee_type) if pointee_type is not None else None,
        _attr_str(type_obj.get_encoding()),
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
                or key in {"operands", "results", "input_token_ids"}
            ):
                continue
            row[key] = _jsonify(value)
        result.append(row)
    return result


def _walk_ops(mod):
    ops = []

    def visit(op):
        ops.append(op)
        return True

    mod.walk(visit)
    return tuple(ops)


def _op_results(op):
    return tuple(op.get_result(index) for index in range(op.get_num_results()))


def _op_operands(op):
    return tuple(op.get_operand(index) for index in range(op.get_num_operands()))


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
        type_plan.pointee_type,
        type_plan.encoding,
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
        type_plan.pointee_type,
        type_plan.encoding,
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
        plans.append(
            _OpPlan(
                index,
                op.get_name(),
                tuple(_value_id(operand) for operand in _op_operands(op)),
                tuple(_value_id(result) for result in _op_results(op)),
                dict(op.get_attrs()),
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
        type_plan.encoding,
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
        type_plan.encoding,
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
    address_value = memdesc_value = value_value = mask_value = None
    result_value_id = token_value_id = None

    if name == "ttg.async_copy_global_to_local":
        address_value = operands[0] if len(operands) > 0 else None
        memdesc_value = operands[1] if len(operands) > 1 else None
        mask_value = operands[2] if len(operands) > 2 else None
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
        source_plan.shape if source_plan is not None else (),
        source_plan.base_arg_index if source_plan is not None else None,
        source_plan.base_arg_name if source_plan is not None else None,
        offset_value_id,
        _value_id(mask_value) if mask_value is not None else None,
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
                        _value_id(operands[0])
                        if name == "ttg.async_copy_global_to_local"
                        and len(operands) > 0
                        else None
                    ),
                    (
                        _value_id(operands[1])
                        if name == "ttg.async_copy_global_to_local"
                        and len(operands) > 1
                        else None
                    ),
                    (
                        _value_id(operands[2])
                        if name == "ttg.async_copy_global_to_local"
                        and len(operands) > 2
                        else None
                    ),
                    input_token_ids,
                    op.get_int_attr("num") if name == "ttg.async_wait" else None,
                )
            )
    return tuple(tokens)


def _build_bridge_plan(mod, kernel):
    ops = _walk_ops(mod)
    owners = _result_owner_map(ops)
    values_by_id = _build_value_plans(mod, kernel, ops)
    op_plans = _build_op_plans(ops)
    op_counts = {}
    for op in ops:
        name = op.get_name()
        op_counts[name] = op_counts.get(name, 0) + 1
    memdescs_by_id = _build_memdesc_plans(ops, values_by_id)
    return _BridgePlan(
        "ttgir_graph",
        op_counts,
        op_plans,
        tuple(values_by_id.values()),
        _build_address_plans(ops, values_by_id, owners),
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


def _candidate_wave_opt_paths():
    override = os.environ.get("TRITON_WAVE_OPT")
    if override:
        yield Path(override)

    for wave_build_dir in _wave_build_dirs():
        yield wave_build_dir / "bin" / "wave-opt"


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


def _wave_opt():
    for path in _existing_paths(_candidate_wave_opt_paths()):
        if os.access(path, os.X_OK):
            return str(path)
    candidates = "\n  ".join(str(path) for path in _candidate_wave_opt_paths())
    raise RuntimeError(
        "tlx_wave requires wave-opt from the third_party/wave submodule build. "
        "Build Triton with TRITON_CODEGEN_BACKENDS including tlx_wave so the Wave tools are built. "
        f"Checked wave-opt candidates:\n  {candidates}"
    )


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
        "tlx_wave.bridge.stage": w.StringAttr.get("module-function-skeleton"),
        "tlx_wave.source_op": w.StringAttr.get("tt.func"),
        "tlx_wave.num_pointer_args": _binding_i32_attr(w, attrs["pointer_count"]),
        "tlx_wave.num_scalar_args": _binding_i32_attr(w, attrs["scalar_count"]),
        "tlx_wave.wave_size": _binding_i32_attr(w, attrs["wave_size"]),
        "tlx_wave.num_warps": _binding_i32_attr(w, attrs["num_warps"]),
    }


def _emit_wave_skeleton_with_bindings(kernel, attrs, plan):
    w = _load_wave_dsl()
    target_triple = _target_triple(attrs)
    pointer_count = sum(arg.kind == "pointer" for arg in kernel.args)
    scalar_count = sum(arg.kind == "scalar" for arg in kernel.args)
    with w.module() as module_builder:
        func_attrs = _binding_attrs(
            w,
            {
                "pointer_count": pointer_count,
                "scalar_count": scalar_count,
                "wave_size": attrs.threads_per_warp,
                "num_warps": attrs.num_warps,
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
        module_builder.module.operation.attributes["tlx_wave.plan.num_tokens"] = (
            _binding_i32_attr(w, len(plan.tokens))
        )
        with module_builder.function(
            kernel.name, arg_types, kernel=True, attrs=func_attrs
        ):
            pass
        return str(module_builder.module)


def _emit_wave_skeleton(kernel, attrs, plan):
    return _emit_wave_skeleton_with_bindings(kernel, attrs, plan), "wave-dsl"


def _verify_wave_skeleton(wave_text, wave_opt):
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
            f"tlx_wave generated Wave skeleton failed wave-opt verification: {detail}"
        )


def stop_before_wave_lowering(mod, metadata, options):
    """Emit the first Wave/WaveAMD module/function skeleton from cutoff TTGIR.

    This stage still does not lower TTGIR operations. It parses the handoff
    module enough to preserve the public kernel ABI and launch metadata, then
    emits a Wave textual MLIR shell for later bridge stages to fill.
    """
    attrs = _module_attrs(mod)
    _validate_target(options, attrs)
    kernel = _kernel_from_module(mod)
    plan = _build_bridge_plan(mod, kernel)

    metadata["name"] = kernel.name or _entry_name(mod)
    metadata["shared"] = 0
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = "emitted_wave_skeleton"
    metadata["tlx_wave_arch"] = options.arch
    metadata["tlx_wave_ttgir_target"] = attrs.target
    metadata["tlx_wave_num_warps"] = attrs.num_warps
    metadata["tlx_wave_threads_per_warp"] = attrs.threads_per_warp
    metadata["tlx_wave_num_ctas"] = attrs.num_ctas
    metadata["tlx_wave_num_kernel_args"] = len(kernel.args)
    metadata["tlx_wave_num_pointer_args"] = sum(
        arg.kind == "pointer" for arg in kernel.args
    )
    metadata["tlx_wave_num_scalar_args"] = sum(
        arg.kind == "scalar" for arg in kernel.args
    )
    metadata["tlx_wave_plan_kind"] = plan.kind
    metadata["tlx_wave_plan_num_ops"] = len(plan.ops)
    metadata["tlx_wave_plan_num_values"] = len(plan.values)
    metadata["tlx_wave_plan_num_addresses"] = len(plan.addresses)
    metadata["tlx_wave_plan_num_memdescs"] = len(plan.memdescs)
    metadata["tlx_wave_plan_num_tokens"] = len(plan.tokens)
    metadata["tlx_wave_plan_json"] = json.dumps(
        _bridge_plan_metadata(plan), sort_keys=True
    )
    wave_text, builder = _emit_wave_skeleton(kernel, attrs, plan)
    wave_opt = _wave_opt()
    _verify_wave_skeleton(wave_text, wave_opt)
    metadata["tlx_wave_wave_builder"] = builder
    metadata["tlx_wave_wave_opt"] = wave_opt
    return wave_text
