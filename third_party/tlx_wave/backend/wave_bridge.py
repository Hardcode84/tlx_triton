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
    signature_type: str
    ttgir_type: str
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
class _ShapedType:
    raw: str
    shape: tuple[int, ...]
    element_type: str
    layout: str | None
    storage: str | None
    mutable: bool


@dataclass(frozen=True)
class _ValuePlan:
    value_id: int
    kind: str
    type: str
    shape: tuple[int, ...]
    element_type: str | None
    varying_dims: tuple[int, ...]
    variability: str
    const_value: int | float | bool | None
    base_arg_index: int | None
    base_arg_name: str | None
    ops: tuple[str, ...]


@dataclass(frozen=True)
class _AddressExprPlan:
    role: str
    value_id: int
    element_type: str
    shape: tuple[int, ...]
    variability: str
    varying_dims: tuple[int, ...]
    base_arg_index: int | None
    base_arg_name: str | None
    offset_value_id: int | None
    offset_variability: str | None
    offset_varying_dims: tuple[int, ...]
    mask_value_id: int | None
    mask_variability: str | None
    mask_varying_dims: tuple[int, ...]
    memdesc_value_id: int | None
    ring_slot: int | None
    ops: tuple[str, ...]


@dataclass(frozen=True)
class _LayoutPlan:
    value_id: int
    kind: str
    shape: tuple[int, ...]
    element_type: str
    encoding: str | None
    storage: str | None


@dataclass(frozen=True)
class _MemDescPlan:
    value_id: int
    kind: str
    name: str | None
    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    element_type: str
    layout: str | None
    storage: str | None
    mutable: bool
    base_value_id: int | None
    ring_slots: int | None
    slot: int | None


@dataclass(frozen=True)
class _TokenPlan:
    value_id: int | None
    kind: str
    source_address_value_id: int | None
    memdesc_value_id: int | None
    mask_value_id: int | None
    input_token_ids: tuple[int, ...]
    wait_group: int | None


@dataclass(frozen=True)
class _BridgePlan:
    kind: str
    values: tuple[_ValuePlan, ...]
    addresses: tuple[_AddressExprPlan, ...]
    layouts: tuple[_LayoutPlan, ...]
    memdescs: tuple[_MemDescPlan, ...]
    tokens: tuple[_TokenPlan, ...]
    block_m: int | None
    block_n: int | None
    block_k: int | None
    ring_slots: int | None
    dot_count: int
    async_copy_count: int
    store_count: int


def _value_id(value):
    return int(value.id())


def _type_str(value_or_type):
    return str(value_or_type.get_type() if hasattr(value_or_type, "get_type") else value_or_type)


def _shape_tuple(value):
    shape = value.get_shape()
    if shape is None:
        return ()
    return tuple(int(dim) for dim in shape)


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


def _split_top_level(text):
    parts = []
    start = 0
    angle_depth = 0
    brace_depth = 0
    bracket_depth = 0
    for index, char in enumerate(text):
        if char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "," and angle_depth == 0 and brace_depth == 0 and bracket_depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return parts


def _parse_shape_and_element(text):
    pieces = text.split("x")
    if len(pieces) == 1:
        return (), pieces[0]

    dims = []
    for piece in pieces[:-1]:
        if piece == "?":
            dims.append(-1)
        elif piece.isdigit():
            dims.append(int(piece))
        else:
            return (), text
    return tuple(dims), pieces[-1]


def _parse_shaped_type(raw):
    raw = str(raw)
    if raw.startswith("tensor<") and raw.endswith(">"):
        body = raw[len("tensor<"):-1]
        parts = _split_top_level(body)
        shape, element = _parse_shape_and_element(parts[0])
        layout = parts[1] if len(parts) > 1 else None
        return _ShapedType(raw, shape, element, layout, None, False)
    if raw.startswith("!ttg.memdesc<") and raw.endswith(">"):
        body = raw[len("!ttg.memdesc<"):-1]
        parts = _split_top_level(body)
        shape, element = _parse_shape_and_element(parts[0])
        layout = parts[1] if len(parts) > 1 else None
        storage = parts[2] if len(parts) > 2 else None
        mutable = any(part == "mutable" for part in parts[3:])
        return _ShapedType(raw, shape, element, layout, storage, mutable)
    return None


def _element_type(raw_type):
    shaped = _parse_shaped_type(raw_type)
    if shaped is not None:
        return shaped.element_type
    raw_type = str(raw_type)
    if raw_type.startswith("!tt.ptr<") and raw_type.endswith(">"):
        return raw_type[len("!tt.ptr<"):-1]
    return raw_type


def _pointer_element_type(raw_type):
    element_type = _element_type(raw_type)
    if element_type.startswith("!tt.ptr<") and element_type.endswith(">"):
        return element_type[len("!tt.ptr<"):-1]
    return element_type


def _layout_plan(value, kind):
    shaped = _parse_shaped_type(_type_str(value))
    if shaped is None:
        return None
    return _LayoutPlan(
        _value_id(value),
        kind,
        shaped.shape,
        shaped.element_type,
        shaped.layout,
        shaped.storage,
    )


def _public_dicts(records):
    result = []
    for record in records:
        row = {}
        for key, value in record.__dict__.items():
            if key == "value_id" or key.endswith("_value_id") or key == "input_token_ids":
                continue
            if isinstance(value, tuple):
                row[key] = list(value)
            else:
                row[key] = value
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


def _unique_ops(ops):
    return tuple(dict.fromkeys(name for name in ops if name))


def _base_from_plans(plans):
    bases = {(plan.base_arg_index, plan.base_arg_name) for plan in plans if plan and plan.base_arg_index is not None}
    if len(bases) != 1:
        return None, None
    return next(iter(bases))


def _const_int(plan):
    if plan is None or isinstance(plan.const_value, bool) or not isinstance(plan.const_value, int):
        return None
    return plan.const_value


def _shift_dims_for_expand(plan, axis):
    if plan is None:
        return ()
    return tuple(dim + 1 if dim >= axis else dim for dim in plan.varying_dims)


def _argument_value_plan(value, index):
    raw_type = _type_str(value)
    shaped = _parse_shaped_type(raw_type)
    shape = _shape_tuple(value) or (shaped.shape if shaped else ())
    element_type = shaped.element_type if shaped else _pointer_element_type(raw_type)
    return _ValuePlan(
        _value_id(value),
        "argument",
        raw_type,
        shape,
        element_type,
        (),
        "uniform",
        None,
        index,
        f"arg{index}",
        ("argument",),
    )


def _value_plan_from_result(op, result, operand_plans, arg_info):
    op_name = op.get_name()
    value_id = _value_id(result)
    raw_type = _type_str(result)
    shaped = _parse_shaped_type(raw_type)
    shape = _shape_tuple(result) or (shaped.shape if shaped else ())
    element_type = shaped.element_type if shaped else _pointer_element_type(raw_type)
    const_value = op.get_constant_value() if op_name == "arith.constant" else None
    base_arg_index, base_arg_name = _base_from_plans(operand_plans)
    ops = _unique_ops(tuple(name for plan in operand_plans if plan for name in plan.ops) + (op_name,))

    if value_id in arg_info:
        index, _name = arg_info[value_id]
        return _argument_value_plan(result, index)

    if op_name == "arith.constant":
        varying_dims = ()
        kind = "constant"
    elif op_name == "tt.get_program_id":
        varying_dims = ()
        kind = "program_id"
    elif op_name == "tt.make_range":
        varying_dims = tuple(range(len(shape)))
        kind = "make_range"
    elif op_name == "tt.expand_dims":
        varying_dims = _shift_dims_for_expand(operand_plans[0] if operand_plans else None, op.get_int_attr("axis") or 0)
        kind = "expand_dims"
    elif op_name in {"tt.broadcast", "tt.splat", "ttg.convert_layout"}:
        varying_dims = operand_plans[0].varying_dims if operand_plans and operand_plans[0] else ()
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
    elif op_name == "ttg.local_load":
        varying_dims = tuple(range(len(shape)))
        kind = "local_load"
    elif op_name == "tt.dot":
        varying_dims = tuple(range(len(shape)))
        kind = "dot"
    elif raw_type == "!ttg.async.token":
        varying_dims = ()
        kind = "token"
    else:
        varying_dims = _merge_varying_dims(*operand_plans)
        kind = op_name

    return _ValuePlan(
        value_id,
        kind,
        raw_type,
        shape,
        element_type,
        varying_dims,
        _variability(varying_dims),
        const_value,
        base_arg_index,
        base_arg_name,
        ops,
    )


def _build_value_plans(mod, kernel, ops):
    fn = mod.get_function(kernel.name)
    values = {}
    arg_info = {}
    for index in range(fn.get_num_args()):
        arg_value = fn.args(index)
        arg_info[_value_id(arg_value)] = (index, f"arg{index}")

    for index in range(fn.get_num_args()):
        arg_value = fn.args(index)
        arg_plan = _argument_value_plan(arg_value, index)
        values[arg_plan.value_id] = arg_plan

    for op in ops:
        operands = _op_operands(op)
        operand_plans = tuple(values.get(_value_id(operand)) for operand in operands)
        for result in _op_results(op):
            plan = _value_plan_from_result(op, result, operand_plans, arg_info)
            values[plan.value_id] = plan
    return values


def _memdesc_from_value(value, kind, name=None, base_value_id=None, ring_slots=None, slot=None):
    shaped = _parse_shaped_type(_type_str(value))
    if shaped is None:
        raise ValueError(f"tlx_wave bridge expected {kind} to have a TTGIR memdesc type, got {_type_str(value)}")
    if shaped.storage is None:
        raise ValueError(f"tlx_wave bridge expected {kind} memdesc to declare storage")
    shape = shaped.shape
    if kind == "allocation" and len(shape) >= 3:
        inferred_ring_slots = shape[0]
        tile_shape = shape[1:]
    else:
        inferred_ring_slots = ring_slots
        tile_shape = shape
    return _MemDescPlan(
        _value_id(value),
        kind,
        name,
        shape,
        tile_shape,
        shaped.element_type,
        shaped.layout,
        shaped.storage,
        shaped.mutable,
        base_value_id,
        inferred_ring_slots,
        slot,
    )


def _build_memdesc_plans(ops, values):
    memdescs = {}
    alloc_index = 0
    view_index = 0
    for op in ops:
        name = op.get_name()
        if name == "ttg.local_alloc":
            result = op.get_result(0)
            plan = _memdesc_from_value(result, "allocation", name=f"alloc{alloc_index}")
            alloc_index += 1
            memdescs[plan.value_id] = plan
        elif name == "ttg.memdesc_index":
            base = op.get_operand(0)
            slot_value = op.get_operand(1)
            base_plan = memdescs.get(_value_id(base))
            slot = _const_int(values.get(_value_id(slot_value)))
            result = op.get_result(0)
            plan = _memdesc_from_value(
                result,
                "view",
                name=f"view{view_index}",
                base_value_id=_value_id(base),
                ring_slots=base_plan.ring_slots if base_plan else None,
                slot=slot,
            )
            view_index += 1
            memdescs[plan.value_id] = plan
        elif name in {"ttg.memdesc_subslice", "ttg.memdesc_reinterpret", "ttg.memdesc_reshape", "ttg.memdesc_trans"}:
            raise ValueError(f"tlx_wave bridge does not yet support LDS view op {name}")
    return memdescs


def _build_layout_plans(ops):
    layouts = {}
    for op in ops:
        for result in _op_results(op):
            plan = _layout_plan(result, op.get_name())
            if plan is not None and plan.shape:
                layouts[plan.value_id] = plan
    return layouts


def _role_by_memdesc_view(ops, owners):
    roles = {}
    for op in ops:
        if op.get_name() != "tt.dot":
            continue
        for operand_index, role in ((0, "a"), (1, "b")):
            load_owner = owners.get(_value_id(op.get_operand(operand_index)))
            if load_owner is None or load_owner.get_name() != "ttg.local_load":
                continue
            roles[_value_id(load_owner.get_operand(0))] = role
    return roles


def _address_plan(role, address_value, values, owners, mask_value=None, memdesc_value=None, ring_slot=None):
    address = values.get(_value_id(address_value))
    if address is None:
        raise ValueError(f"tlx_wave bridge could not plan address value for role {role}")
    mask = values.get(_value_id(mask_value)) if mask_value is not None else None
    offset_value_id = None
    offset_variability = None
    offset_varying_dims = ()
    owner = owners.get(address.value_id)
    if owner is not None and owner.get_name() == "tt.addptr":
        offset = values.get(_value_id(owner.get_operand(1)))
        if offset is not None:
            offset_value_id = offset.value_id
            offset_variability = offset.variability
            offset_varying_dims = offset.varying_dims
    return _AddressExprPlan(
        role,
        address.value_id,
        _pointer_element_type(address.type),
        address.shape,
        address.variability,
        address.varying_dims,
        address.base_arg_index,
        address.base_arg_name,
        offset_value_id,
        offset_variability,
        offset_varying_dims,
        mask.value_id if mask else None,
        mask.variability if mask else None,
        mask.varying_dims if mask else (),
        _value_id(memdesc_value) if memdesc_value is not None else None,
        ring_slot,
        address.ops,
    )


def _build_address_plans(ops, values, memdescs, owners):
    roles = _role_by_memdesc_view(ops, owners)
    addresses = []
    async_index = 0
    for op in ops:
        if op.get_name() == "ttg.async_copy_global_to_local":
            address_value = op.get_operand(0)
            memdesc_value = op.get_operand(1)
            mask_value = op.get_operand(2) if op.get_num_operands() > 2 else None
            memdesc_id = _value_id(memdesc_value)
            role = roles.get(memdesc_id)
            if role is None:
                role = "a" if async_index % 2 == 0 else "b"
            memdesc = memdescs.get(memdesc_id)
            addresses.append(_address_plan(role, address_value, values, owners, mask_value, memdesc_value,
                                           memdesc.slot if memdesc else None))
            async_index += 1
        elif op.get_name() == "tt.store":
            address_value = op.get_operand(0)
            mask_value = op.get_operand(2) if op.get_num_operands() > 2 else None
            addresses.append(_address_plan("c", address_value, values, owners, mask_value))
    return tuple(addresses)


def _build_token_plans(ops):
    tokens = []
    for op in ops:
        name = op.get_name()
        if name == "ttg.async_copy_global_to_local":
            result_id = _value_id(op.get_result(0))
            tokens.append(
                _TokenPlan(
                    result_id,
                    "async_copy",
                    _value_id(op.get_operand(0)),
                    _value_id(op.get_operand(1)),
                    _value_id(op.get_operand(2)) if op.get_num_operands() > 2 else None,
                    (),
                    None,
                )
            )
        elif name == "ttg.async_commit_group":
            result_id = _value_id(op.get_result(0)) if op.get_num_results() else None
            tokens.append(
                _TokenPlan(result_id, "commit_group", None, None, None,
                           tuple(_value_id(operand) for operand in _op_operands(op)), None)
            )
        elif name == "ttg.async_wait":
            result_id = _value_id(op.get_result(0)) if op.get_num_results() else None
            tokens.append(_TokenPlan(result_id, "wait_group", None, None, None, (), op.get_int_attr("num")))
    return tuple(tokens)


def _validate_static_gemm_plan(plan, ops):
    supported = {
        "builtin.module",
        "tt.func",
        "tt.return",
        "arith.constant",
        "arith.addi",
        "arith.muli",
        "arith.andi",
        "arith.cmpi",
        "tt.get_program_id",
        "tt.make_range",
        "tt.expand_dims",
        "tt.broadcast",
        "tt.splat",
        "tt.addptr",
        "tt.store",
        "tt.dot",
        "ttg.local_alloc",
        "ttg.memdesc_index",
        "ttg.async_copy_global_to_local",
        "ttg.async_commit_group",
        "ttg.async_wait",
        "ttg.local_load",
        "ttg.convert_layout",
    }
    for op in ops:
        name = op.get_name()
        if name.startswith("ttng."):
            raise ValueError(f"tlx_wave bridge GEMM planner rejects TMEM/NVIDIA storage op {name}")
        if name not in supported:
            raise ValueError(f"tlx_wave bridge GEMM planner does not support TTGIR op {name}")

    for memdesc in plan.memdescs:
        if memdesc.storage != "#ttg.shared_memory":
            raise ValueError(
                f"tlx_wave bridge GEMM planner supports only SMEM memdescs, got storage {memdesc.storage}"
            )
        if memdesc.layout is None or not memdesc.layout.startswith("#ttg.swizzled_shared"):
            raise ValueError(f"tlx_wave bridge GEMM planner does not support LDS layout {memdesc.layout}")
        if any(dim <= 0 for dim in memdesc.shape):
            raise ValueError(f"tlx_wave bridge GEMM planner rejects dynamic LDS sizes in {memdesc.shape}")
        if memdesc.element_type != "f16":
            raise ValueError(
                f"tlx_wave bridge GEMM planner supports only f16 LDS GEMM operands, got {memdesc.element_type}"
            )
        if memdesc.kind == "allocation":
            if len(memdesc.shape) != 3 or len(memdesc.tile_shape) != 2:
                raise ValueError(
                    "tlx_wave bridge GEMM planner supports only static rank-2 tiles with static ring-buffer slots"
                )
        elif memdesc.kind == "view":
            if len(memdesc.tile_shape) != 2 or memdesc.slot is None:
                raise ValueError("tlx_wave bridge GEMM planner requires static rank-2 LDS views and static slots")

    seen_views = set()
    for memdesc in plan.memdescs:
        if memdesc.kind != "view":
            continue
        key = (memdesc.base_value_id, memdesc.slot)
        if key in seen_views:
            raise ValueError("tlx_wave bridge GEMM planner rejects storage alias overlap between LDS views")
        seen_views.add(key)

    for op in ops:
        if op.get_name() != "tt.dot":
            continue
        lhs = _element_type(_type_str(op.get_operand(0)))
        rhs = _element_type(_type_str(op.get_operand(1)))
        if lhs != "f16" or rhs != "f16":
            raise ValueError(f"tlx_wave bridge GEMM planner supports only f16 GEMM operands, got {lhs} x {rhs}")


def _build_bridge_plan(mod, kernel):
    ops = _walk_ops(mod)
    owners = _result_owner_map(ops)
    values_by_id = _build_value_plans(mod, kernel, ops)
    memdescs_by_id = _build_memdesc_plans(ops, values_by_id)
    layouts_by_id = _build_layout_plans(ops)
    addresses = _build_address_plans(ops, values_by_id, memdescs_by_id, owners)
    tokens = _build_token_plans(ops)
    dot_count = sum(op.get_name() == "tt.dot" for op in ops)
    async_copy_count = sum(op.get_name() == "ttg.async_copy_global_to_local" for op in ops)
    store_count = sum(op.get_name() == "tt.store" for op in ops)

    allocations = [memdesc for memdesc in memdescs_by_id.values() if memdesc.kind == "allocation"]
    ring_slots = allocations[0].ring_slots if allocations else None
    block_m = block_n = block_k = None
    if allocations and len(allocations[0].tile_shape) == 2:
        block_m, block_k = allocations[0].tile_shape
    if len(allocations) > 1 and len(allocations[1].tile_shape) == 2:
        _, block_n = allocations[1].tile_shape

    plan = _BridgePlan(
        "gemm" if dot_count else "generic",
        tuple(values_by_id.values()),
        addresses,
        tuple(layouts_by_id.values()),
        tuple(memdescs_by_id.values()),
        tokens,
        block_m,
        block_n,
        block_k,
        ring_slots,
        dot_count,
        async_copy_count,
        store_count,
    )
    if plan.kind == "gemm":
        _validate_static_gemm_plan(plan, ops)
    return plan


def _bridge_plan_metadata(plan):
    return {
        "kind": plan.kind,
        "block_m": plan.block_m,
        "block_n": plan.block_n,
        "block_k": plan.block_k,
        "ring_slots": plan.ring_slots,
        "dot_count": plan.dot_count,
        "async_copy_count": plan.async_copy_count,
        "store_count": plan.store_count,
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
    has_explicit_local_mem_access = bool(op.get_bool_attr("tlx.has_explicit_local_mem_access"))
    return _ModuleAttrs(target, num_ctas, num_warps, threads_per_warp, has_explicit_local_mem_access)


def _wave_arg_type(name, signature_type):
    if signature_type.startswith("*"):
        element_type = signature_type[1:]
        if not element_type:
            raise ValueError(f"tlx_wave bridge does not yet support pointer argument %{name} with empty element type")
        return f"!wave.ptr<#wave.global, {element_type}>", "pointer"
    if signature_type in {"i1", "i8", "i16", "i32", "i64", "index", "f16", "bf16", "f32"}:
        return signature_type, "scalar"
    raise ValueError(f"tlx_wave bridge does not yet support kernel argument %{name} with type {signature_type}")


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
        names = ", ".join(func.get_str_attr("sym_name") or "<unnamed>" for func in funcs) or "none"
        raise ValueError(f"tlx_wave bridge supports exactly one public tt.func kernel, found {len(funcs)} ({names})")

    op = funcs[0]
    name = op.get_str_attr("sym_name") or _entry_name(mod)
    fn = mod.get_function(name)
    signature = mod.get_function_signature(fn)
    if len(signature) != fn.get_num_args():
        raise ValueError(
            "tlx_wave bridge expected function signature length to match argument count, "
            f"got {len(signature)} signature entries for {fn.get_num_args()} args"
        )

    args = []
    for index, signature_type in enumerate(signature):
        name_for_arg = f"arg{index}"
        ttgir_type = str(fn.args(index).get_type())
        wave_type, kind = _wave_arg_type(name_for_arg, signature_type)
        args.append(_KernelArg(index, name_for_arg, signature_type, ttgir_type, wave_type, kind))
    return _Kernel(name, tuple(args), op.get_bool_attr("noinline"))


def _validate_target(options, attrs):
    if options.arch != "gfx950":
        raise ValueError(f"tlx_wave bridge only supports gfx950 Wave skeletons, got {options.arch}")
    if options.warp_size != 64:
        raise ValueError(f"tlx_wave bridge only supports wave64 inputs, got warp_size={options.warp_size}")
    if attrs.target != "hip:gfx950":
        raise ValueError(f"tlx_wave bridge only supports TTGIR target hip:gfx950, got {attrs.target}")
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


def _binding_type(signature_type, w):
    if signature_type.startswith("*"):
        return w.ptr_type(_binding_type(signature_type[1:], w))
    if signature_type == "i1":
        return w.i1()
    if signature_type == "i8":
        return w.i8()
    if signature_type == "i16":
        return w.IntegerType.get_signless(16)
    if signature_type == "i32":
        return w.i32()
    if signature_type == "i64":
        return w.i64()
    if signature_type == "index":
        return w.index_type()
    if signature_type == "f16":
        return w.f16()
    if signature_type == "bf16":
        return w.bf16()
    if signature_type == "f32":
        return w.f32()
    raise ValueError(f"tlx_wave bridge does not yet support Wave binding type {signature_type}")


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
            w, {
                "pointer_count": pointer_count,
                "scalar_count": scalar_count,
                "wave_size": attrs.threads_per_warp,
                "num_warps": attrs.num_warps,
            })
        if kernel.noinline is not None:
            func_attrs["tlx_wave.ttgir.noinline"] = _binding_bool_attr(w, kernel.noinline)

        arg_types = [_binding_type(arg.signature_type, w) for arg in kernel.args]
        module_builder.module.operation.attributes["waveamdmachine.target"] = w.StringAttr.get(target_triple)
        module_builder.module.operation.attributes["tlx_wave.source_target"] = w.StringAttr.get(attrs.target)
        module_builder.module.operation.attributes["tlx_wave.num_ctas"] = _binding_i32_attr(w, attrs.num_ctas)
        module_builder.module.operation.attributes["tlx_wave.num_warps"] = _binding_i32_attr(w, attrs.num_warps)
        module_builder.module.operation.attributes["tlx_wave.threads_per_warp"] = _binding_i32_attr(
            w, attrs.threads_per_warp)
        module_builder.module.operation.attributes["tlx_wave.has_explicit_local_mem_access"] = _binding_bool_attr(
            w, attrs.has_explicit_local_mem_access)
        module_builder.module.operation.attributes["tlx_wave.plan.kind"] = w.StringAttr.get(plan.kind)
        if plan.block_m is not None:
            module_builder.module.operation.attributes["tlx_wave.plan.block_m"] = _binding_i32_attr(w, plan.block_m)
        if plan.block_n is not None:
            module_builder.module.operation.attributes["tlx_wave.plan.block_n"] = _binding_i32_attr(w, plan.block_n)
        if plan.block_k is not None:
            module_builder.module.operation.attributes["tlx_wave.plan.block_k"] = _binding_i32_attr(w, plan.block_k)
        if plan.ring_slots is not None:
            module_builder.module.operation.attributes["tlx_wave.plan.ring_slots"] = _binding_i32_attr(
                w, plan.ring_slots)
        module_builder.module.operation.attributes["tlx_wave.plan.num_addresses"] = _binding_i32_attr(
            w, len(plan.addresses))
        module_builder.module.operation.attributes["tlx_wave.plan.num_memdescs"] = _binding_i32_attr(
            w, len(plan.memdescs))
        module_builder.module.operation.attributes["tlx_wave.plan.num_tokens"] = _binding_i32_attr(
            w, len(plan.tokens))
        with module_builder.function(kernel.name, arg_types, kernel=True, attrs=func_attrs):
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
        raise RuntimeError(f"tlx_wave generated Wave skeleton failed wave-opt verification: {detail}")


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
    metadata["tlx_wave_num_pointer_args"] = sum(arg.kind == "pointer" for arg in kernel.args)
    metadata["tlx_wave_num_scalar_args"] = sum(arg.kind == "scalar" for arg in kernel.args)
    metadata["tlx_wave_plan_kind"] = plan.kind
    metadata["tlx_wave_plan_num_addresses"] = len(plan.addresses)
    metadata["tlx_wave_plan_num_memdescs"] = len(plan.memdescs)
    metadata["tlx_wave_plan_num_tokens"] = len(plan.tokens)
    metadata["tlx_wave_plan_json"] = json.dumps(_bridge_plan_metadata(plan), sort_keys=True)
    wave_text, builder = _emit_wave_skeleton(kernel, attrs, plan)
    wave_opt = _wave_opt()
    _verify_wave_skeleton(wave_text, wave_opt)
    metadata["tlx_wave_wave_builder"] = builder
    metadata["tlx_wave_wave_opt"] = wave_opt
    return wave_text
