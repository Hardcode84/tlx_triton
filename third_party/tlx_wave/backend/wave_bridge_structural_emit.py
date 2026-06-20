"""Structural Wave MLIR emission for the new TLX Wave bridge.

This module is the bridge-local boundary to the Wave Python MLIR bindings.
Rewriters remain stateless and produce the value/effect objects from
``wave_bridge_rewriters``; this emitter consumes those objects and constructs
Wave IR with Python APIs instead of formatting operation text.
"""

from dataclasses import dataclass
from math import gcd
from pathlib import Path
import sys
import warnings

from .wave_bridge_conversion import (
    _ConversionRun,
    _ConvertedInputValue,
    _amd_mfma_encoding_info,
    _attr_bool,
    _attr_value,
    _blocked_encoding_info,
    _dot_operand_encoding_info,
    _mma_shape_for_parent,
    _tile_rep_count,
)
from .wave_bridge_rewriters import (
    _AssumeEffect,
    _AsyncCommitGroupValue,
    _AsyncCopyValue,
    _AsyncWaitEffect,
    _AsyncWaitValue,
    _BinaryValue,
    _BufferStoreEffect,
    _CompareValue,
    _ConstantValue,
    _CastValue,
    _DotValue,
    _ForValue,
    _ForwardValue,
    _IfEffect,
    _IfValue,
    _LocalAllocValue,
    _LocalLoadValue,
    _LocalStoreEffect,
    _LoadValue,
    _MaskAndValue,
    _MemdescIndexValue,
    _MinValue,
    _PointerAddValue,
    _ProgramIdValue,
    _RangeValue,
    _ReturnEffect,
    _StoreEffect,
    _UnaryTensorValue,
)


@dataclass(frozen=True)
class _StructuralEmission:
    text: str
    lds_size: int = 0


@dataclass(frozen=True)
class _CoordGroupFact:
    divisibility: int
    delta_min: int
    delta_max: int


class _UnsupportedStructuralEmission(ValueError):
    pass


def _can_emit_structural_wave_module(run: _ConversionRun):
    deferred_coordinate_values = _deferred_coordinate_value_ids(
        run.program,
        run.result,
    )
    if any(
        value_id not in deferred_coordinate_values and not _can_emit_structural_value(value)
        for value_id, value in run.result.values.items()
    ):
        return False
    if any(not _can_emit_structural_effect(effect) for effect in run.result.effects):
        return False
    return True


def _deferred_coordinate_value_ids(program, result):
    candidate_trees = []
    for value in result.values.values():
        if not _is_structural_buffer_dma_candidate(value):
            continue
        tree = set()
        for root_id in (value.source_offset_value_id, value.mask_value_id):
            if root_id is None:
                continue
            root_tree = _coordinate_tree_ids(result, root_id)
            if not root_tree:
                tree = set()
                break
            tree.update(root_tree)
        if not tree:
            continue
        candidate_trees.append(frozenset(tree))
    for effect in result.effects:
        if not _is_structural_buffer_store_candidate(effect):
            continue
        roots = [effect.offset_id]
        if effect.mask_id is not None:
            roots.append(effect.mask_id)
        tree = set()
        for root_id in roots:
            root_tree = _coordinate_tree_ids(result, root_id)
            if not root_tree:
                tree = set()
                break
            tree.update(root_tree)
        if not tree:
            continue
        candidate_trees.append(frozenset(tree))
    for effect in result.effects:
        if not _is_structural_fragment_store_candidate(effect):
            continue
        tree = set(_pointer_coordinate_tree_ids(result, effect.pointer_id))
        if not tree:
            continue
        if effect.mask_id is not None:
            mask_tree = _coordinate_tree_ids(result, effect.mask_id)
            if not mask_tree:
                continue
            tree.update(mask_tree)
        candidate_trees.append(frozenset(tree))
    for value in result.values.values():
        if not _is_structural_dot_operand_convert(value, result):
            continue
        candidate_trees.append(frozenset({value.operand_id}))
    deferred = set()
    for tree in candidate_trees:
        deferred.update(
            value_id
            for value_id in tree
            if program.values[value_id].type.kind == "tensor"
        )
    return frozenset(deferred)


def _coordinate_tree_ids(result, root_id):
    if root_id is None:
        return frozenset()
    value = result.values.get(root_id)
    if not _can_materialize_coordinate_value(value):
        return frozenset()
    tree = {root_id}
    for operand_id in _coordinate_value_operand_ids(value):
        child_tree = _coordinate_tree_ids(result, operand_id)
        if not child_tree:
            return frozenset()
        tree.update(child_tree)
    return frozenset(tree)


def _pointer_coordinate_tree_ids(result, root_id):
    if root_id is None:
        return frozenset()
    value = result.values.get(root_id)
    if isinstance(value, _PointerAddValue):
        base_tree = _pointer_coordinate_tree_ids(result, value.base_id)
        offset_tree = _coordinate_tree_ids(result, value.offset_id)
        if not base_tree or not offset_tree:
            return frozenset()
        return frozenset({root_id}) | base_tree | offset_tree
    if isinstance(value, _UnaryTensorValue):
        if value.op_name not in {"tt.broadcast", "tt.splat"}:
            return frozenset()
        child_tree = _pointer_coordinate_tree_ids(result, value.operand_id)
        if not child_tree:
            return frozenset()
        return frozenset({root_id}) | child_tree
    if isinstance(value, _ForwardValue):
        if value.op_name not in {"tt.expand_dims", "ttg.convert_layout"}:
            return frozenset()
        child_tree = _pointer_coordinate_tree_ids(result, value.operand_id)
        if not child_tree:
            return frozenset()
        return frozenset({root_id}) | child_tree
    if isinstance(value, _ConvertedInputValue):
        return frozenset({root_id})
    return frozenset()


def _coordinate_value_operand_ids(value):
    if isinstance(value, (_BinaryValue, _CompareValue, _MaskAndValue, _MinValue)):
        return (value.lhs_id, value.rhs_id)
    if isinstance(value, (_UnaryTensorValue, _ForwardValue, _CastValue)):
        return (value.operand_id,)
    return ()


def _can_materialize_coordinate_value(value):
    if isinstance(
        value,
        (
            _ConstantValue,
            _RangeValue,
            _ConvertedInputValue,
            _ProgramIdValue,
            _IfValue,
            _ForValue,
        ),
    ):
        return True
    if isinstance(value, _UnaryTensorValue):
        return value.op_name in {"tt.broadcast", "tt.splat"}
    if isinstance(value, _ForwardValue):
        return value.op_name in {"tt.expand_dims", "ttg.convert_layout"}
    if isinstance(value, _CastValue):
        return value.kind == "fpconvert"
    if isinstance(value, _CompareValue):
        return True
    if isinstance(value, _MaskAndValue):
        return True
    if isinstance(value, _MinValue):
        return True
    if isinstance(value, _BinaryValue):
        return value.kind in {
            "addi",
            "subi",
            "muli",
            "shli",
            "shrui",
            "shrsi",
            "andi",
            "ori",
            "xori",
            "divui",
            "remui",
            "divsi",
            "remsi",
        }
    return False


def _is_structural_buffer_dma_candidate(value):
    return (
        isinstance(value, _AsyncCopyValue)
        and value.op_name == "amdg.buffer_load_to_local"
        and value.source_address_value_id is not None
        and value.source_offset_value_id is not None
        and value.memdesc_value_id is not None
        and value.other_value_id is None
        and isinstance(value.source_address, _ConvertedInputValue)
        and isinstance(value.memdesc, (_LocalAllocValue, _MemdescIndexValue))
    )


def _can_emit_structural_loop_type(result_type):
    return getattr(result_type, "kind", None) in {
        "scalar",
        "pointer",
        "tensor",
        "mask",
        "fragment",
    }


def _is_structural_buffer_store_candidate(effect):
    if not isinstance(effect, _BufferStoreEffect):
        return False
    if getattr(effect.value, "converted_type", None) is None:
        return False
    if effect.value.converted_type.kind != "fragment":
        return False
    if not isinstance(effect.base, _ConvertedInputValue):
        return False
    if not _can_materialize_coordinate_value(effect.offset):
        return False
    if effect.mask is not None and not _can_materialize_coordinate_value(effect.mask):
        return False
    return True


def _is_structural_fragment_store_candidate(effect):
    return (
        isinstance(effect, _StoreEffect)
        and getattr(effect.value, "converted_type", None) is not None
        and effect.value.converted_type.kind == "fragment"
    )


def _can_emit_structural_effect(effect):
    if isinstance(effect, (_ReturnEffect, _StoreEffect, _AssumeEffect)):
        return True
    if _is_structural_buffer_store_candidate(effect):
        return True
    if isinstance(effect, (_AsyncWaitEffect, _LocalStoreEffect)):
        return True
    if isinstance(effect, _IfEffect):
        return (
            all(_can_emit_structural_value(nested) for nested in effect.then_values.values())
            and all(_can_emit_structural_value(nested) for nested in effect.else_values.values())
            and all(_can_emit_structural_effect(nested) for nested in effect.then_effects)
            and all(_can_emit_structural_effect(nested) for nested in effect.else_effects)
        )
    return False


def _can_emit_structural_value(value):
    if isinstance(
        value,
        (
            _ConvertedInputValue,
            _ConstantValue,
            _RangeValue,
            _ProgramIdValue,
            _UnaryTensorValue,
            _BinaryValue,
            _MaskAndValue,
            _MinValue,
            _PointerAddValue,
            _LocalAllocValue,
            _MemdescIndexValue,
            _AsyncCommitGroupValue,
            _AsyncWaitValue,
            _DotValue,
        ),
    ):
        return True
    if isinstance(value, _AsyncCopyValue):
        return _can_emit_structural_async_copy(value)
    if isinstance(value, _ForwardValue):
        return _is_type_preserving_forward(value) or _is_structural_dot_operand_convert(
            value,
            None,
        )
    if isinstance(value, _CastValue):
        return _can_emit_structural_cast(value)
    if isinstance(value, _CompareValue):
        return True
    if isinstance(value, _LoadValue):
        return value.mask_id is None or value.converted_type.component_count == 1
    if isinstance(value, _LocalLoadValue):
        return True
    if isinstance(value, _IfValue):
        return (
            value.converted_type.component_count == 1
            and all(_can_emit_structural_value(nested) for nested in value.then_values.values())
            and all(_can_emit_structural_value(nested) for nested in value.else_values.values())
        )
    if isinstance(value, _ForValue):
        return (
            not value.body_effects
            and all(_can_emit_structural_loop_type(result_type) for result_type in value.result_types)
            and all(
                _can_emit_structural_value(nested)
                for nested in value.body_values.values()
            )
        )
    return False


def _emit_structural_wave_module(run: _ConversionRun) -> _StructuralEmission:
    _require_structural_subset(run)
    dsl, ir = _load_wave_dsl()
    program = run.program
    result = run.result
    lds_offsets, lds_size = _compute_lds_layout(program)
    kernel_name = program.kernel_name or "kernel"
    target = program.target or "hip:gfx950"
    num_ctas = _int_or(program.num_ctas, 1)
    num_warps = _int_or(program.num_warps, 1)
    wave_size = _int_or(program.threads_per_warp, 64)
    target_triple = target.replace("hip:", "amdgcn-amd-amdhsa--")
    pointer_count = sum(
        program.values[value_id].type.kind == "pointer" for value_id in program.kernel_args
    )
    scalar_count = sum(
        program.values[value_id].type.kind == "scalar" for value_id in program.kernel_args
    )
    with dsl.ModuleBuilder() as module_builder:
        arg_types = [
            _parse_type(ir, result.values[value_id].converted_type.wave_type)
            for value_id in program.kernel_args
        ]
        module_attrs = module_builder.module.operation.attributes
        module_attrs["tlx_wave.has_explicit_local_mem_access"] = ir.Attribute.parse(
            "true" if program.has_explicit_local_mem_access else "false"
        )
        module_attrs["tlx_wave.new_bridge"] = ir.Attribute.parse("true")
        module_attrs["tlx_wave.num_ctas"] = ir.IntegerAttr.get(dsl.i32(), num_ctas)
        module_attrs["tlx_wave.num_warps"] = ir.IntegerAttr.get(dsl.i32(), num_warps)
        module_attrs["tlx_wave.source_target"] = ir.StringAttr.get(target)
        module_attrs["tlx_wave.threads_per_warp"] = ir.IntegerAttr.get(
            dsl.i32(), wave_size
        )
        module_attrs["waveamdmachine.target"] = ir.StringAttr.get(target_triple)

        func_attrs = {
            "tlx_wave.bridge.stage": ir.StringAttr.get("new-python-rewrite"),
            "tlx_wave.num_pointer_args": ir.IntegerAttr.get(dsl.i32(), pointer_count),
            "tlx_wave.num_scalar_args": ir.IntegerAttr.get(dsl.i32(), scalar_count),
            "tlx_wave.num_warps": ir.IntegerAttr.get(dsl.i32(), num_warps),
            "tlx_wave.source_op": ir.StringAttr.get("tt.func"),
            "tlx_wave.ttgir.noinline": ir.Attribute.parse(
                "true" if program.noinline else "false"
            ),
            "tlx_wave.wave_size": ir.IntegerAttr.get(dsl.i32(), wave_size),
        }
        if lds_size:
            func_attrs["wave.lds_size"] = ir.IntegerAttr.get(dsl.i64(), lds_size)
        with module_builder.function(
            kernel_name,
            arg_types,
            kernel=True,
            attrs=func_attrs,
        ) as builder:
            state = _StructuralState(
                dsl,
                ir,
                builder,
                program,
                result,
                lds_offsets,
                _assume_condition_value_ids(program, result),
                _deferred_coordinate_value_ids(program, result),
            )
            for arg_id, arg_value in zip(program.kernel_args, builder.args):
                state.values[arg_id] = arg_value
            _seed_assume_range_facts(state)
            _emit_structural_body(state)
        return _StructuralEmission(str(module_builder), lds_size)


@dataclass
class _StructuralState:
    dsl: object
    ir: object
    builder: object
    program: object
    result: object
    lds_offsets: dict[int, int]
    values: dict[int, object]
    mem_token: object | None
    assume_condition_values: frozenset[int]
    deferred_coordinate_values: frozenset[int]
    nonnegative_values: set[int]
    positive_values: set[int]
    bounded_kernel_args: set[int]
    lower_bounds: dict[int, int]

    def __init__(
        self,
        dsl,
        ir,
        builder,
        program,
        result,
        lds_offsets=None,
        assume_condition_values=frozenset(),
        deferred_coordinate_values=frozenset(),
    ):
        self.dsl = dsl
        self.ir = ir
        self.builder = builder
        self.program = program
        self.result = result
        self.lds_offsets = {} if lds_offsets is None else dict(lds_offsets)
        self.values = {}
        self.mem_token = None
        self.assume_condition_values = assume_condition_values
        self.deferred_coordinate_values = deferred_coordinate_values
        self.nonnegative_values = set()
        self.positive_values = set()
        self.bounded_kernel_args = set()
        self.lower_bounds = {}
        self.buffer_cache = {}


def _emit_structural_body(state: _StructuralState):
    for op in state.program.ops:
        for result_id in op.results:
            if result_id in state.assume_condition_values:
                continue
            if result_id in state.deferred_coordinate_values:
                continue
            value = state.result.values.get(result_id)
            if value is None:
                continue
            _emit_structural_value(state, value)
        for effect in _effects_for_op(state, op.index):
            if isinstance(effect, _ReturnEffect):
                if effect.operand_ids:
                    raise _UnsupportedStructuralEmission(
                        "tlx_wave structural emitter does not yet support return values"
                    )
                continue
            if isinstance(effect, _StoreEffect):
                _emit_store(state, effect)
                continue
            if isinstance(effect, _BufferStoreEffect):
                _emit_buffer_store(state, effect)
                continue
            if isinstance(effect, _AssumeEffect):
                _emit_assume(state, effect)
                continue
            if isinstance(effect, _AsyncWaitEffect):
                _emit_async_wait_effect(state, effect)
                continue
            if isinstance(effect, _LocalStoreEffect):
                _emit_local_store(state, effect)
                continue
            if isinstance(effect, _IfEffect):
                _emit_if_effect(state, effect)
                continue
            raise _UnsupportedStructuralEmission(
                f"tlx_wave structural emitter cannot lower effect {type(effect).__name__}"
            )


def _emit_structural_region_body(state: _StructuralState, op_indices, converted_values, effects=()):
    effects_by_op_index = {}
    for effect in effects:
        effects_by_op_index.setdefault(getattr(effect, "op_index", -1), []).append(effect)
    for op_index in op_indices:
        op = state.program.ops[op_index]
        if op.name == "scf.yield":
            continue
        for result_id in op.results:
            if result_id in state.assume_condition_values:
                continue
            if result_id in state.deferred_coordinate_values:
                continue
            value = converted_values.get(result_id)
            if value is None:
                continue
            _emit_structural_value_from_map(state, value, converted_values)
        for effect in effects_by_op_index.pop(op.index, ()):
            _emit_structural_effect(state, effect)
    for op_index in sorted(effects_by_op_index):
        for effect in effects_by_op_index[op_index]:
            _emit_structural_effect(state, effect)


def _emit_structural_effect(state: _StructuralState, effect):
    if isinstance(effect, _ReturnEffect):
        if effect.operand_ids:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter does not yet support return values"
            )
        return
    if isinstance(effect, _StoreEffect):
        _emit_store(state, effect)
        return
    if isinstance(effect, _BufferStoreEffect):
        _emit_buffer_store(state, effect)
        return
    if isinstance(effect, _AssumeEffect):
        _emit_assume(state, effect)
        return
    if isinstance(effect, _AsyncWaitEffect):
        _emit_async_wait_effect(state, effect)
        return
    if isinstance(effect, _LocalStoreEffect):
        _emit_local_store(state, effect)
        return
    if isinstance(effect, _IfEffect):
        _emit_if_effect(state, effect)
        return
    raise _UnsupportedStructuralEmission(
        f"tlx_wave structural emitter cannot lower effect {type(effect).__name__}"
    )


def _effects_for_op(state: _StructuralState, op_index: int):
    return tuple(
        effect
        for effect in state.result.effects
        if getattr(effect, "op_index", -1) == op_index
    )


def _emit_structural_value(state: _StructuralState, value):
    _emit_structural_value_from_map(state, value, state.result.values)


def _emit_structural_value_from_map(state: _StructuralState, value, converted_values):
    if value.value_id in state.values:
        return
    if isinstance(value, _ConvertedInputValue):
        return
    if isinstance(value, _ConstantValue):
        _emit_constant(state, value)
        return
    if isinstance(value, _RangeValue):
        _emit_range(state, value)
        return
    if isinstance(value, _ProgramIdValue):
        _emit_program_id(state, value)
        return
    if isinstance(value, _UnaryTensorValue):
        _emit_unary_tensor(state, value)
        return
    if isinstance(value, _BinaryValue):
        _emit_binary(state, value)
        return
    if isinstance(value, _CompareValue):
        _emit_compare(state, value)
        return
    if isinstance(value, _MaskAndValue):
        _emit_mask_and(state, value)
        return
    if isinstance(value, _MinValue):
        _emit_min(state, value)
        return
    if isinstance(value, _PointerAddValue):
        _emit_pointer_add(state, value)
        return
    if isinstance(value, _DotValue):
        _emit_dot(state, value)
        return
    if isinstance(value, _LocalAllocValue):
        _emit_local_alloc(state, value)
        return
    if isinstance(value, _MemdescIndexValue):
        _emit_memdesc_index(state, value)
        return
    if isinstance(value, _LocalLoadValue):
        _emit_local_load(state, value)
        return
    if isinstance(value, _AsyncCopyValue):
        _emit_async_copy(state, value)
        return
    if isinstance(value, _AsyncCommitGroupValue):
        _emit_async_commit_group(state, value)
        return
    if isinstance(value, _AsyncWaitValue):
        _emit_async_wait_value(state, value)
        return
    if isinstance(value, _ForwardValue):
        _emit_forward(state, value)
        return
    if isinstance(value, _CastValue):
        _emit_cast(state, value)
        return
    if isinstance(value, _LoadValue):
        _emit_load(state, value)
        return
    if isinstance(value, _IfValue):
        _emit_if(state, value, converted_values)
        return
    if isinstance(value, _ForValue):
        _emit_for(state, value, converted_values)
        return
    raise _UnsupportedStructuralEmission(
        f"tlx_wave structural emitter cannot lower value {type(value).__name__}"
    )


def _emit_constant(state: _StructuralState, value: _ConstantValue):
    source_type = state.program.values[value.value_id].type
    if value.converted_type.kind == "fragment":
        literal = _scalar_constant_literal(value)
        if literal not in (0, 0.0):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter supports fragment constants only "
                "for zero fill"
            )
        zero = state.builder.constant(state.dsl.i32(), 0)
        fragment_type = _parse_type(state.ir, value.converted_type.wave_type)
        state.values[value.value_id] = _pack(
            state.builder.fragment_fill(zero, fragment_type)
            for _ in range(value.converted_type.component_count)
        )
        return
    if source_type.kind == "tensor":
        literal = _scalar_constant_literal(value)
        if literal is None:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter supports tensor constants only for splats"
            )
        result_type = _parse_type(state.ir, value.converted_type.wave_type)
        if value.converted_type.kind == "mask":
            state.values[value.value_id] = _pack(
                _wave_mask_constant(state, result_type, bool(literal))
                for _ in range(value.converted_type.component_count)
            )
            return
        scalar_type = _source_element_type(state.dsl, source_type)
        const = state.builder.constant(scalar_type, literal)
        state.values[value.value_id] = _pack(
            state.builder.splat(const, scalar_type, _simd_width(state.dsl, result_type))
            for _ in range(value.converted_type.component_count)
        )
        return
    literal = _scalar_constant_literal(value)
    if source_type.kind == "scalar" and source_type.raw == "i1":
        state.values[value.value_id] = state.dsl.arith.ConstantOp(
            _parse_type(state.ir, value.converted_type.wave_type),
            state.ir.IntegerAttr.get(state.dsl.i1(), int(bool(literal))),
        ).result
        return
    state.values[value.value_id] = state.builder.constant(
        _parse_type(state.ir, value.converted_type.wave_type),
        literal,
    )


def _emit_range(state: _StructuralState, value: _RangeValue):
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    width = _simd_width(state.dsl, result_type)
    element_type = _simd_element_type(state.dsl, result_type)
    component_count = int(value.converted_type.component_count)
    if value.end - value.start > width * component_count:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot fit tt.make_range in SIMD components"
        )
    lane = state.builder.workitem_id(0, element_type, width)
    mapping = _rank1_component_lane_mapping(
        state,
        _source_type(state, value.value_id),
        component_count,
        width,
    )
    components = []
    for component in range(component_count):
        lane_scale, offset = mapping[component]
        index = lane
        if lane_scale != 1:
            scale = state.builder.constant(element_type, lane_scale)
            scale = state.builder.splat(scale, element_type, width)
            index = state.builder.muli(index, scale)
        start = int(value.start) + int(offset)
        if start:
            const = state.builder.constant(element_type, start)
            splat = state.builder.splat(const, element_type, width)
            index = state.builder.addi(index, splat)
        components.append(index)
    state.values[value.value_id] = _pack(components)


def _emit_program_id(state: _StructuralState, value: _ProgramIdValue):
    raw = state.builder.workgroup_id(value.axis)
    state.values[value.value_id] = state.builder.assume_range(raw, 0, 2147483647)
    state.nonnegative_values.add(value.value_id)


def _emit_unary_tensor(state: _StructuralState, value: _UnaryTensorValue):
    operand = state.values[value.operand_id]
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    result_count = int(value.converted_type.component_count)
    operand_components = _components(operand)
    if len(operand_components) == result_count and all(
        component.type == result_type for component in operand_components
    ):
        state.values[value.value_id] = operand
        return
    if len(operand_components) != 1:
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter cannot remap {value.op_name} components yet"
        )
    operand_value = operand_components[0]
    if operand_value.type == result_type:
        state.values[value.value_id] = _pack(operand_components * result_count)
        return
    if _is_simd_type(state.dsl, operand_value.type):
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter cannot remap SIMD {value.op_name} "
            f"type {operand_value.type} -> {result_type}"
        )
    if str(operand_value.type) == "i1" and value.converted_type.kind == "mask":
        true_mask = _wave_mask_constant(state, result_type, True)
        false_mask = _wave_mask_constant(state, result_type, False)
        state.values[value.value_id] = _pack(
            state.builder.select(operand_value, true_mask, false_mask)
            for _ in range(result_count)
        )
        return
    width = _simd_width(state.dsl, result_type)
    state.values[value.value_id] = _pack(
        state.builder.splat(operand_value, operand_value.type, width)
        for _ in range(result_count)
    )


def _emit_binary(state: _StructuralState, value: _BinaryValue):
    lhs = _components(state.values[value.lhs_id])
    rhs = _components(state.values[value.rhs_id])
    result_count = int(value.converted_type.component_count)
    if len(lhs) == 1 and result_count > 1:
        lhs = lhs * result_count
    if len(rhs) == 1 and result_count > 1:
        rhs = rhs * result_count
    if len(lhs) != result_count or len(rhs) != result_count:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot match binary component counts"
        )
    if _try_emit_pow2_signed_div_rem(state, value, lhs, rhs):
        return
    kind = _binary_kind(state.dsl, _effective_binary_kind(value, state))
    nsw = "nsw" in value.flags
    nuw = "nuw" in value.flags
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    state.values[value.value_id] = _pack(
        state.builder.binary(
            kind,
            _ensure_binary_operand_type(state, lhs_value, result_type),
            _ensure_binary_operand_type(state, rhs_value, result_type),
            nsw=nsw,
            nuw=nuw,
        )
        for lhs_value, rhs_value in zip(lhs, rhs)
    )


def _ensure_binary_operand_type(state: _StructuralState, value, result_type):
    if value.type == result_type:
        return value
    if _is_integer_or_index_type(value.type) and _is_integer_or_index_type(result_type):
        return _emit_int_convert(state, value, result_type)
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural emitter cannot adapt binary operand type "
        f"{value.type} -> {result_type}"
    )


def _emit_int_convert(state: _StructuralState, value, result_type):
    policy = _int_convert_policy(state, value.type, result_type)
    return state.builder.cast(
        value,
        result_type,
        state.dsl.CastKind.IntConvert,
        policy=policy,
    )


def _int_convert_policy(state: _StructuralState, source_type, target_type):
    source_bits = _integer_bit_width_for_type(state, source_type)
    target_bits = _integer_bit_width_for_type(state, target_type)
    if target_bits is None and source_bits is not None:
        return state.ir.Attribute.parse("{extension = #wave.cast_extension<zero>}")
    if (
        source_bits is not None
        and target_bits is not None
        and int(source_bits) < int(target_bits)
    ):
        return state.ir.Attribute.parse("{extension = #wave.cast_extension<zero>}")
    return None


def _integer_bit_width_for_type(state: _StructuralState, type_):
    if _is_simd_type(state.dsl, type_):
        type_ = _simd_element_type(state.dsl, type_)
    text = str(type_)
    if text == "index":
        return None
    if text.startswith("i") and text[1:].isdigit():
        return int(text[1:])
    return None


def _is_integer_or_index_type(type_):
    text = str(type_)
    if text == "index":
        return True
    if text.startswith("i") and text[1:].isdigit():
        return True
    if text.startswith("!wave.simd<") and "," in text:
        element = text[len("!wave.simd<") :].split(",", 1)[0].strip()
        return element == "index" or (element.startswith("i") and element[1:].isdigit())
    return False


def _emit_compare(state: _StructuralState, value: _CompareValue):
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    result_count = int(value.converted_type.component_count)
    lhs = _component_tuple_for_count(
        state.values[value.lhs_id], result_count, "compare lhs"
    )
    rhs = _component_tuple_for_count(
        state.values[value.rhs_id], result_count, "compare rhs"
    )
    state.values[value.value_id] = _pack(
        _emit_cmpi(state, value.predicate, lhs_value, rhs_value)
        for lhs_value, rhs_value in zip(lhs, rhs)
    )
    for component in _components(state.values[value.value_id]):
        if component.type != result_type:
            raise _UnsupportedStructuralEmission(
                f"tlx_wave structural emitter produced compare type {component.type}, "
                f"expected {result_type}"
            )


def _emit_mask_and(state: _StructuralState, value: _MaskAndValue):
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    result_count = int(value.converted_type.component_count)
    lhs = _component_tuple_for_count(
        state.values[value.lhs_id], result_count, "mask lhs"
    )
    rhs = _component_tuple_for_count(
        state.values[value.rhs_id], result_count, "mask rhs"
    )
    false_mask = _wave_mask_constant(state, result_type, False)
    state.values[value.value_id] = _pack(
        state.builder.select(lhs_value, rhs_value, false_mask)
        for lhs_value, rhs_value in zip(lhs, rhs)
    )


def _emit_min(state: _StructuralState, value: _MinValue):
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    result_count = int(value.converted_type.component_count)
    lhs = _component_tuple_for_count(
        state.values[value.lhs_id], result_count, "min lhs"
    )
    rhs = _component_tuple_for_count(
        state.values[value.rhs_id], result_count, "min rhs"
    )
    components = []
    for lhs_value, rhs_value in zip(lhs, rhs):
        if lhs_value.type != result_type or rhs_value.type != result_type:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter cannot lower min with incompatible "
                f"types: {lhs_value.type}, {rhs_value.type} -> {result_type}"
            )
        pred = _emit_cmpi(state, value.predicate, lhs_value, rhs_value)
        components.append(state.builder.select(pred, lhs_value, rhs_value))
    state.values[value.value_id] = _pack(components)


def _emit_pointer_add(state: _StructuralState, value: _PointerAddValue):
    result_count = int(value.converted_type.component_count)
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    base = _component_tuple_for_count(
        state.values[value.base_id], result_count, "ptr_add base"
    )
    offset = _component_tuple_for_count(
        state.values[value.offset_id], result_count, "ptr_add offset"
    )
    state.values[value.value_id] = _pack(
        state.builder.ptr_add(base_value, offset_value, result_type)
        for base_value, offset_value in zip(base, offset)
    )


def _emit_dot(state: _StructuralState, value: _DotValue):
    lhs_fragments = _fragment_tuple_for_tile(
        state,
        value.lhs_id,
        value.lhs_tile_shape,
        "tt.dot lhs",
    )
    rhs_fragments = _fragment_tuple_for_tile(
        state,
        value.rhs_id,
        value.rhs_tile_shape,
        "tt.dot rhs",
    )
    acc_fragments = _fragment_tuple_for_tile(
        state,
        value.acc_id,
        value.result_tile_shape,
        "tt.dot accumulator",
    )
    results = []
    for row in range(value.result_tile_shape[0]):
        for col in range(value.result_tile_shape[1]):
            acc = acc_fragments[_tile_index(value.result_tile_shape, row, col)]
            for k_tile in range(value.lhs_tile_shape[1]):
                lhs = lhs_fragments[_tile_index(value.lhs_tile_shape, row, k_tile)]
                rhs = rhs_fragments[_tile_index(value.rhs_tile_shape, k_tile, col)]
                acc = state.builder.mma(value.mma_kind, lhs, rhs, acc)
            results.append(acc)
    state.values[value.value_id] = _pack(results)


def _fragment_tuple_for_tile(state, value_id, tile_shape, context):
    value = _require_structural_value(state, value_id)
    expected = int(tile_shape[0]) * int(tile_shape[1])
    components = _components(value)
    if len(components) != expected:
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter expected {expected} fragments for "
            f"{context}, got {len(components)}"
        )
    return components


def _tile_index(tile_shape, row, col):
    return int(row) * int(tile_shape[1]) + int(col)


def _emit_forward(state: _StructuralState, value: _ForwardValue):
    if _is_structural_dot_operand_convert(value, state.result):
        _emit_converted_dot_operand_local_load(state, value)
        return
    operand = state.values[value.operand_id]
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    if isinstance(operand, tuple):
        if not all(component.type == result_type for component in operand):
            raise _UnsupportedStructuralEmission(
                f"tlx_wave structural emitter cannot forward {value.op_name} "
                "with changed component types"
            )
        state.values[value.value_id] = operand
        return
    if operand.type != result_type:
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter cannot forward {value.op_name} "
            f"{operand.type} -> {result_type}"
        )
    state.values[value.value_id] = operand


def _emit_cast(state: _StructuralState, value: _CastValue):
    if value.kind != "fpconvert":
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter cannot lower cast kind {value.kind}"
        )
    operand_type = getattr(value.operand, "converted_type", None)
    if (
        getattr(operand_type, "kind", None) == "fragment"
        and value.converted_type.kind == "fragment"
    ):
        state.values[value.value_id] = state.values[value.operand_id]
        return
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    result_count = int(value.converted_type.component_count)
    operands = _component_tuple_for_count(
        state.values[value.operand_id], result_count, "cast operand"
    )
    components = []
    for operand in operands:
        if operand.type == result_type:
            components.append(operand)
            continue
        components.append(
            state.builder.cast(operand, result_type, state.dsl.CastKind.FpConvert)
        )
    state.values[value.value_id] = _pack(components)


def _emit_load(state: _StructuralState, value: _LoadValue):
    result_count = int(value.converted_type.component_count)
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    pointers = _component_tuple_for_count(
        state.values[value.pointer_id], result_count, "load pointer"
    )
    masks = None
    if value.mask_id is not None:
        masks = _component_tuple_for_count(
            state.values[value.mask_id], result_count, "load mask"
        )
    others = None
    if value.other_id is not None:
        others = _component_tuple_for_count(
            state.values[value.other_id], result_count, "load other"
        )
    token = _ensure_mem_token(state)
    components = []
    for index, pointer in enumerate(pointers):
        active = _component_active_mask(state, value.value_id, index, result_type)
        mask = None if masks is None else masks[index]
        mask = _combine_optional_masks(state, active, mask)
        if mask is None:
            loaded, token = state.builder.load(pointer, result_type, after=token)
        else:
            other = (
                _zero_value(state, result_type)
                if others is None
                else _ensure_value_type(state, others[index], result_type, "load other")
            )
            loaded, token = _emit_masked_load_component(
                state, pointer, mask, other, result_type, token
            )
        components.append(loaded)
    state.mem_token = token
    state.values[value.value_id] = _pack(components)


def _emit_local_alloc(state: _StructuralState, value: _LocalAllocValue):
    source_type = _source_type(state, value.value_id)
    element_type = _source_element_type(state.dsl, source_type)
    offset = state.lds_offsets.get(value.value_id, 0)
    state.values[value.value_id] = state.builder.lds_base(element_type, offset=offset)


def _emit_memdesc_index(state: _StructuralState, value: _MemdescIndexValue):
    base = _require_structural_value(state, value.memdesc_id)
    result_type = _shared_pointer_type_for_source(state, value.value_id)
    if base.type != result_type:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot index memdesc with changed pointer "
            f"type: {base.type} -> {result_type}"
        )
    elements_per_slot = _memdesc_index_elements_per_slot(state, value)
    static_index = _constant_int_record(value.index)
    if static_index is not None:
        element_offset = int(static_index) * int(elements_per_slot)
        if element_offset == 0:
            state.values[value.value_id] = base
            return
        offset = state.builder.constant(state.dsl.i32(), element_offset)
    else:
        offset = _require_structural_value(state, value.index_id)
        if _is_simd_type(state.dsl, offset.type):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter cannot lower lane-varying memdesc_index"
            )
        if int(elements_per_slot) != 1:
            stride = state.builder.constant(offset.type, int(elements_per_slot))
            offset = state.builder.muli(offset, stride)
    state.values[value.value_id] = state.builder.ptr_add(base, offset, result_type)


def _emit_local_load(state: _StructuralState, value: _LocalLoadValue):
    if value.converted_type.kind == "fragment":
        _emit_fragment_local_load(state, value)
        return
    result_count = int(value.converted_type.component_count)
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    pointers = _local_linear_pointers(
        state,
        value.memdesc_id,
        value.value_id,
        result_type,
        result_count,
    )
    token = _memory_token_operand(state, value.token_id)
    components = []
    for index, pointer in enumerate(pointers):
        mask = _component_active_mask(state, value.value_id, index, result_type)
        if mask is None:
            loaded, token = state.builder.load(pointer, result_type, after=token)
        else:
            loaded, token = _emit_masked_load_component(
                state,
                pointer,
                mask,
                _zero_value(state, result_type),
                result_type,
                token,
            )
        components.append(loaded)
    state.mem_token = token
    state.values[value.value_id] = _pack(components)


def _emit_fragment_local_load(state: _StructuralState, value: _LocalLoadValue):
    _emit_fragment_local_load_from_memdesc(
        state,
        value.value_id,
        value.memdesc_id,
        value.token_id,
        value.converted_type,
    )


def _emit_converted_dot_operand_local_load(state: _StructuralState, value: _ForwardValue):
    local_load = state.result.values.get(value.operand_id)
    if not isinstance(local_load, _LocalLoadValue):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter can lower converted dot operands only "
            "from ttg.local_load"
        )
    _emit_fragment_local_load_from_memdesc(
        state,
        value.value_id,
        local_load.memdesc_id,
        local_load.token_id,
        value.converted_type,
    )


def _emit_fragment_local_load_from_memdesc(
    state,
    value_id,
    memdesc_id,
    token_id,
    converted_type,
):
    source_type = _source_type(state, value_id)
    memdesc_type = _source_type(state, memdesc_id)
    padded = _padded_shared_memdesc_info(memdesc_type)
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if (
        padded is None
        and not _is_identity_shared_memdesc(memdesc_type)
        and swizzled is None
    ):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower fragment ttg.local_load "
            "from non-identity shared layout yet; source encoding="
            f"{source_type.encoding}, memdesc encoding={memdesc_type.encoding}"
        )
    dot = _dot_operand_encoding_info(source_type)
    if dot is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment local_load only for "
            f"#ttg.dot_op tensors, got {source_type.encoding}"
        )
    mma = _mma_shape_for_parent(dot.parent, "ttg.local_load")
    if dot.op_idx == 0:
        tile_shape = (
            _tile_rep_count(source_type.shape[0], mma.output_tile_shape[0]),
            _tile_rep_count(source_type.shape[1], mma.k_dim),
        )
    else:
        tile_shape = (
            _tile_rep_count(source_type.shape[0], mma.k_dim),
            _tile_rep_count(source_type.shape[1], mma.output_tile_shape[1]),
        )
    base = _ptr_cast(
        state,
        _require_structural_value(state, memdesc_id),
        _parse_type(state.ir, "!wave.ptr<#wave.shared, i32>"),
    )
    fragment_type = _parse_type(state.ir, converted_type.wave_type)
    input_token = _memory_token_operand(state, token_id)
    fragments = []
    load_tokens = []
    for row in range(tile_shape[0]):
        for col in range(tile_shape[1]):
            tile_offsets = _dot_operand_tile_offsets(dot, mma, row, col)
            if _can_emit_b16_transpose_fragment_load(
                source_type,
                memdesc_type,
                dot,
                mma,
            ):
                fragment, load_token = _emit_b16_transpose_fragment_load_tile(
                    state,
                    _require_structural_value(state, memdesc_id),
                    source_type,
                    memdesc_type,
                    fragment_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            elif padded is not None:
                fragment, load_token = _emit_padded_fragment_load_tile(
                    state,
                    base,
                    source_type,
                    memdesc_type,
                    padded,
                    fragment_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            elif swizzled is not None and not _is_identity_shared_memdesc(memdesc_type):
                fragment, load_token = _emit_swizzled_fragment_load_tile(
                    state,
                    base,
                    source_type,
                    memdesc_type,
                    swizzled,
                    fragment_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            else:
                fragment, load_token = _emit_dense_fragment_load_tile(
                    state,
                    base,
                    source_type,
                    fragment_type,
                    input_token,
                    tile_offsets,
                    mma.operand_registers,
                    mma.wave_size,
                )
            fragments.append(fragment)
            load_tokens.append(load_token)
    expected = int(converted_type.component_count)
    if len(fragments) != expected:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter fragment local_load tile mismatch: "
            f"loaded {len(fragments)}, expected {expected}"
        )
    if load_tokens:
        state.mem_token = state.builder.join(*load_tokens)
    state.values[value_id] = _pack(fragments)


def _emit_dense_fragment_load_tile(
    state,
    base,
    source_type,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    tile_base = _dense_tile_base_dwords(source_type, tile_offsets)
    offset = _emit_lane_scaled_offset(
        state,
        int(wave_size),
        int(registers),
        tile_base,
    )
    upper = _fragment_load_dword_offset_upper(source_type, registers)
    offset = state.builder.assume_range(offset, 0, upper)
    return _emit_fragment_dword_load(state, base, offset, fragment_type, token, registers, wave_size)


def _emit_padded_fragment_load_tile(
    state,
    base,
    source_type,
    memdesc_type,
    padded,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    offset = _emit_padded_fragment_dword_offset(
        state,
        source_type,
        memdesc_type,
        padded,
        tile_offsets,
        registers,
        wave_size,
    )
    return _emit_fragment_dword_load(state, base, offset, fragment_type, token, registers, wave_size)


def _emit_swizzled_fragment_load_tile(
    state,
    base,
    source_type,
    memdesc_type,
    swizzled,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    dword_offset = _emit_swizzled_fragment_dword_offset(
        state,
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        registers,
        wave_size,
    )
    return _emit_fragment_dword_load(
        state,
        base,
        dword_offset,
        fragment_type,
        token,
        registers,
        wave_size,
    )


def _emit_fragment_dword_load(
    state,
    base,
    offset,
    fragment_type,
    token,
    registers,
    wave_size,
):
    ptr_type = _parse_type(
        state.ir,
        f"!wave.simd<!wave.ptr<#wave.shared, i32>, {int(wave_size)}>",
    )
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(int(registers), state.dsl.i32()),
        width=int(wave_size),
    )
    pointer = state.builder.ptr_add(base, offset, ptr_type)
    loaded, token = state.builder.load(pointer, load_type, after=token)
    return state.builder.fragment_pack(loaded, fragment_type), token


def _can_emit_b16_transpose_fragment_load(source_type, memdesc_type, dot, mma):
    if dot.op_idx != 1:
        return False
    if tuple(mma.instr_shape) != (32, 32, 16):
        return False
    if int(mma.wave_size) != 64 or int(mma.operand_registers) != 4:
        return False
    if source_type.element_type not in {"f16", "bf16"}:
        return False
    if source_type.element_byte_width != 2 or memdesc_type.element_byte_width != 2:
        return False
    if memdesc_type.element_type != source_type.element_type:
        return False
    if _padded_shared_memdesc_info(memdesc_type) is not None:
        return True
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    return (
        swizzled is not None
        and int(swizzled["vec"]) == 8
        and int(swizzled["per_phase"]) == 4
        and int(swizzled["max_phase"]) == 4
        and tuple(swizzled["order"]) == (1, 0)
    )


def _emit_b16_transpose_fragment_load_tile(
    state,
    base,
    source_type,
    memdesc_type,
    fragment_type,
    token,
    tile_offsets,
    registers,
    wave_size,
):
    elements_per_lane = int(registers) * (4 // int(memdesc_type.element_byte_width))
    if elements_per_lane != 8:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower b16 transpose fragment "
            f"load with {elements_per_lane} elements per lane"
        )
    _validate_transpose_fragment_packets(
        source_type,
        memdesc_type,
        tile_offsets,
        elements_per_lane,
        int(memdesc_type.element_byte_width),
        int(wave_size),
    )
    element_type = _source_element_type(state.dsl, source_type)
    base_type = state.dsl.ptr_type(element_type, state.dsl.shared_address_space())
    base = _ptr_cast(state, base, base_type)
    ptr_type = state.dsl.simd_type(base_type, width=int(wave_size))
    load_type = state.dsl.simd_type(
        state.dsl.vector_type(4, element_type),
        width=int(wave_size),
    )
    component_type = state.dsl.simd_type(element_type, width=int(wave_size))
    components = []
    for chunk in range(2):
        chunk_offsets = _transpose_fragment_chunk_tile_offsets(tile_offsets, chunk)
        offset = _emit_fragment_element_offset(
            state,
            source_type,
            memdesc_type,
            chunk_offsets,
            elements_per_lane,
            wave_size,
            access_elements=4,
        )
        pointer = state.builder.ptr_add(base, offset, ptr_type)
        loaded, token = state.builder.transpose_load(pointer, load_type, after=token)
        for component in range(4):
            components.append(
                state.dsl.wave.ExtractOp(
                    component_type,
                    loaded,
                    int(component),
                ).result
            )
    packed_type = state.dsl.simd_type(
        state.dsl.vector_type(8, element_type),
        width=int(wave_size),
    )
    packed = state.dsl.wave.PackOp(packed_type, components).result
    return state.builder.fragment_pack(packed, fragment_type), token


def _transpose_fragment_chunk_tile_offsets(tile_offsets, chunk):
    offsets = list(tile_offsets)
    offsets[-1] += 4 * int(chunk)
    return tuple(offsets)


def _emit_fragment_element_offset(
    state,
    source_type,
    memdesc_type,
    tile_offsets,
    elements_per_lane,
    wave_size,
    *,
    access_elements,
):
    padded = _padded_shared_memdesc_info(memdesc_type)
    if padded is not None:
        offset = _emit_padded_fragment_element_offset(
            state,
            source_type,
            memdesc_type,
            padded,
            tile_offsets,
            elements_per_lane,
            wave_size,
        )
    else:
        swizzled = _swizzled_shared_memdesc_info(memdesc_type)
        if swizzled is not None and not _is_identity_shared_memdesc(memdesc_type):
            offset = _emit_swizzled_fragment_index_offset(
                state,
                source_type,
                memdesc_type,
                swizzled,
                tile_offsets,
                elements_per_lane,
                wave_size,
                elements_per_offset_unit=1,
            )
        else:
            offset = _emit_dense_fragment_element_offset(
                state,
                source_type,
                tile_offsets,
                elements_per_lane,
                wave_size,
            )
    upper = _fragment_load_element_offset_upper(memdesc_type, access_elements)
    return state.builder.assume_range(offset, 0, upper)


def _emit_dense_fragment_element_offset(
    state,
    source_type,
    tile_offsets,
    elements_per_lane,
    wave_size,
):
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    return _emit_lane_scaled_offset(
        state,
        int(wave_size),
        int(elements_per_lane),
        tile_base,
    )


def _emit_padded_fragment_element_offset(
    state,
    source_type,
    memdesc_type,
    padded,
    tile_offsets,
    elements_per_lane,
    wave_size,
):
    byte_width = memdesc_type.element_byte_width or source_type.element_byte_width
    if byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower padded transpose "
            "fragment load without known element byte width"
        )
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    lane = state.builder.workitem_id(0, state.dsl.i32(), int(wave_size))
    lane = state.builder.assume_range(lane, 0, int(wave_size) - 1)
    logical = _emit_simd_binary_const(state, "muli", lane, int(elements_per_lane))
    if tile_base:
        logical = _emit_simd_add_const(state, logical, tile_base)
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        quotient = _emit_simd_binary_const(state, "divui", logical, int(interval))
        pad = _emit_simd_binary_const(state, "muli", quotient, int(padding))
        encoded = state.builder.addi(encoded, pad)
    return encoded


def _emit_padded_fragment_dword_offset(
    state,
    source_type,
    memdesc_type,
    padded,
    tile_offsets,
    registers,
    wave_size,
):
    byte_width = memdesc_type.element_byte_width or source_type.element_byte_width
    if byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower padded fragment load "
            "without known element byte width"
        )
    elements_per_lane = int(registers) * (4 // int(byte_width))
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    _validate_padded_fragment_packet(
        padded,
        tile_base,
        elements_per_lane,
        int(byte_width),
    )
    affine = _padded_fragment_affine_dword_offset(
        padded,
        tile_base,
        elements_per_lane,
        int(byte_width),
        int(wave_size),
    )
    upper = _fragment_load_dword_offset_upper(memdesc_type, registers)
    if affine is not None:
        offset = _emit_lane_scaled_offset(
            state,
            int(wave_size),
            affine[0],
            affine[1],
        )
        return state.builder.assume_range(offset, 0, upper)
    lane = state.builder.workitem_id(0, state.dsl.i32(), int(wave_size))
    lane = state.builder.assume_range(lane, 0, int(wave_size) - 1)
    logical = _emit_simd_binary_const(state, "muli", lane, int(elements_per_lane))
    if tile_base:
        logical = _emit_simd_add_const(state, logical, tile_base)
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        quotient = _emit_simd_binary_const(state, "divui", logical, int(interval))
        pad = _emit_simd_binary_const(state, "muli", quotient, int(padding))
        encoded = state.builder.addi(encoded, pad)
    elements_per_dword = 4 // int(byte_width)
    if elements_per_dword != 1:
        encoded = _emit_simd_binary_const(
            state,
            "divui",
            encoded,
            elements_per_dword,
        )
    return state.builder.assume_range(encoded, 0, upper)


def _emit_swizzled_fragment_dword_offset(
    state,
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    registers,
    wave_size,
):
    byte_width = memdesc_type.element_byte_width or source_type.element_byte_width
    if byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower swizzled fragment load "
            "without known element byte width"
        )
    if 4 % int(byte_width):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports swizzled fragment loads only "
            f"when element byte width divides a dword, got {byte_width}"
        )
    elements_per_lane = int(registers) * (4 // int(byte_width))
    _validate_swizzled_fragment_packet(
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        elements_per_lane,
        int(byte_width),
        int(wave_size),
    )
    elements_per_dword = 4 // int(byte_width)
    offset = _emit_swizzled_fragment_index_offset(
        state,
        source_type,
        memdesc_type,
        swizzled,
        tile_offsets,
        elements_per_lane,
        wave_size,
        elements_per_offset_unit=elements_per_dword,
    )
    upper = _fragment_load_dword_offset_upper(memdesc_type, registers)
    return state.builder.assume_range(offset, 0, upper)


def _emit_swizzled_fragment_index_offset(
    state,
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    elements_per_lane,
    wave_size,
    *,
    elements_per_offset_unit,
):
    if len(memdesc_type.shape) != 2 or len(source_type.shape) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports swizzled fragment loads only "
            "for rank-2 source and memdesc tensors"
        )
    lane = state.builder.workitem_id(0, state.dsl.i32(), int(wave_size))
    lane = state.builder.assume_range(lane, 0, int(wave_size) - 1)
    logical = _emit_simd_binary_const(
        state,
        "muli",
        lane,
        int(elements_per_lane),
    )
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    if tile_base:
        logical = _emit_simd_add_const(state, logical, tile_base)
    cols = int(memdesc_type.shape[-1])
    vec = int(swizzled["vec"])
    row = _emit_simd_binary_const(state, "divui", logical, cols)
    col = _emit_simd_binary_const(state, "remui", logical, cols)
    row_phase = _emit_simd_binary_const(
        state,
        "divui",
        row,
        int(swizzled["per_phase"]),
    )
    phase = _emit_simd_binary_const(
        state,
        "remui",
        row_phase,
        int(swizzled["max_phase"]),
    )
    col_group = _emit_simd_binary_const(state, "divui", col, vec)
    swizzled_group = state.builder.binary(
        _binary_kind(state.dsl, "xori"),
        col_group,
        phase,
    )
    swizzled_base = _emit_simd_binary_const(state, "muli", swizzled_group, vec)
    col_in_vec = _emit_simd_binary_const(state, "remui", col, vec)
    swizzled_col = state.builder.addi(swizzled_base, col_in_vec)
    row_scaled = _emit_simd_binary_const(state, "muli", row, cols)
    element = state.builder.addi(row_scaled, swizzled_col)
    unit = int(elements_per_offset_unit)
    if unit != 1:
        element = _emit_simd_binary_const(state, "divui", element, unit)
    return element


def _padded_fragment_affine_dword_offset(
    padded,
    tile_base,
    elements_per_lane,
    byte_width,
    wave_size,
):
    elements_per_dword = 4 // int(byte_width)
    max_lane_elements = (int(wave_size) - 1) * int(elements_per_lane)
    pad_elements = 0
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        interval = int(interval)
        padding = int(padding)
        if int(tile_base) % interval + max_lane_elements >= interval:
            return None
        pad_elements += (int(tile_base) // interval) * padding
    constant_elements = int(tile_base) + pad_elements
    if int(elements_per_lane) % elements_per_dword:
        return None
    if constant_elements % elements_per_dword:
        return None
    return (
        int(elements_per_lane) // elements_per_dword,
        constant_elements // elements_per_dword,
    )


def _emit_lane_scaled_offset(state, wave_size, scale, constant):
    lane = state.builder.workitem_id(0, state.dsl.i32(), int(wave_size))
    lane = state.builder.assume_range(lane, 0, int(wave_size) - 1)
    offset = lane
    if int(scale) != 1:
        offset = _emit_simd_binary_const(state, "muli", offset, int(scale))
    if int(constant) != 0:
        offset = _emit_simd_add_const(state, offset, int(constant))
    return offset


def _fragment_load_dword_offset_upper(source_type, registers):
    total_bytes = _local_alloc_size_bytes(source_type)
    access_bytes = int(registers) * 4
    upper_bytes = total_bytes - access_bytes
    if upper_bytes < 0:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot bound fragment load offset: "
            f"{access_bytes} byte load exceeds {total_bytes} byte allocation"
        )
    return upper_bytes // 4


def _fragment_load_element_offset_upper(source_type, access_elements):
    if source_type.element_byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot bound fragment load element "
            "offset: unknown element byte width"
        )
    total_elements = _local_alloc_size_bytes(source_type) // int(
        source_type.element_byte_width
    )
    upper = total_elements - int(access_elements)
    if upper < 0:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot bound fragment load element "
            f"offset: {access_elements} element load exceeds allocation"
        )
    return upper


def _validate_swizzled_fragment_packet(
    source_type,
    memdesc_type,
    swizzled,
    tile_offsets,
    elements_per_lane,
    byte_width,
    wave_size,
):
    if 4 % int(byte_width):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports swizzled fragment loads only "
            f"when element byte width divides a dword, got {byte_width}"
        )
    tile_base = _dense_tile_base_elements(source_type, tile_offsets)
    for lane in range(int(wave_size)):
        first = None
        for element in range(int(elements_per_lane)):
            logical = tile_base + lane * int(elements_per_lane) + element
            coords = _row_major_coords_static(source_type.shape, logical)
            for dim, coord in enumerate(coords):
                if coord < 0 or coord >= int(memdesc_type.shape[dim]):
                    raise _UnsupportedStructuralEmission(
                        "tlx_wave structural emitter cannot lower swizzled "
                        f"fragment load: coordinate {coords} exceeds "
                        f"memdesc shape {memdesc_type.shape}"
                    )
            byte_offset = _swizzled_shared_static_byte_offset(
                memdesc_type,
                coords,
                swizzled,
            )
            if first is None:
                first = byte_offset
                if first % 4:
                    raise _UnsupportedStructuralEmission(
                        "tlx_wave structural emitter cannot lower swizzled "
                        f"fragment load: packet base byte offset {first} is "
                        "not dword-aligned"
                    )
                continue
            expected = first + element * int(byte_width)
            if byte_offset != expected:
                raise _UnsupportedStructuralEmission(
                    "tlx_wave structural emitter cannot lower swizzled "
                    f"fragment load: lane {lane} packet is not physically "
                    "contiguous"
                )


def _validate_transpose_fragment_packets(
    source_type,
    memdesc_type,
    tile_offsets,
    elements_per_lane,
    byte_width,
    wave_size,
):
    for chunk in range(2):
        chunk_offsets = _transpose_fragment_chunk_tile_offsets(tile_offsets, chunk)
        tile_base = _dense_tile_base_elements(source_type, chunk_offsets)
        for lane in range(int(wave_size)):
            first = None
            for element in range(4):
                logical = tile_base + lane * int(elements_per_lane) + element
                coords = _row_major_coords_static(source_type.shape, logical)
                for dim, coord in enumerate(coords):
                    if coord < 0 or coord >= int(memdesc_type.shape[dim]):
                        raise _UnsupportedStructuralEmission(
                            "tlx_wave structural emitter cannot lower "
                            "transpose fragment load: coordinate "
                            f"{coords} exceeds memdesc shape {memdesc_type.shape}"
                        )
                byte_offset = _memdesc_static_byte_offset(memdesc_type, coords)
                if first is None:
                    first = byte_offset
                    if first % 8:
                        raise _UnsupportedStructuralEmission(
                            "tlx_wave structural emitter cannot lower "
                            "transpose fragment load: packet base byte "
                            f"offset {first} is not 8-byte aligned"
                        )
                    continue
                expected = first + element * int(byte_width)
                if byte_offset != expected:
                    raise _UnsupportedStructuralEmission(
                        "tlx_wave structural emitter cannot lower transpose "
                        f"fragment load: lane {lane} packet is not physically "
                        "contiguous"
                    )


def _validate_padded_fragment_packet(padded, tile_base, elements_per_lane, byte_width):
    if 4 % int(byte_width):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports padded fragment loads only "
            f"when element byte width divides a dword, got {byte_width}"
        )
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        if int(interval) % int(elements_per_lane):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter cannot lower padded fragment load "
                f"whose {elements_per_lane}-element packets can cross padding "
                f"interval {interval}"
            )
        if int(tile_base) % int(elements_per_lane):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter cannot lower padded fragment load "
                f"with tile base {tile_base} not aligned to packet width "
                f"{elements_per_lane}"
            )
        if int(padding) % (4 // int(byte_width)):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter cannot lower padded fragment load "
                f"with padding {padding} not divisible by dword elements"
            )


def _dot_operand_tile_offsets(info, mma, row, col):
    if info.op_idx == 0:
        return (
            int(row) * int(mma.output_tile_shape[0]),
            int(col) * int(mma.k_dim),
        )
    return (
        int(row) * int(mma.k_dim),
        int(col) * int(mma.output_tile_shape[1]),
    )


def _dense_tile_base_dwords(source_type, tile_offsets):
    if source_type.element_byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot compute fragment tile offset "
            f"for {source_type.raw}: unknown element byte width"
        )
    element_offset = _dense_tile_base_elements(source_type, tile_offsets)
    byte_offset = element_offset * int(source_type.element_byte_width)
    if byte_offset % 4:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot compute fragment tile offset: "
            f"{byte_offset} byte offset is not i32 aligned"
        )
    return byte_offset // 4


def _dense_tile_base_elements(source_type, tile_offsets):
    element_offset = 0
    stride = 1
    for dim in reversed(range(len(source_type.shape))):
        element_offset += int(tile_offsets[dim]) * stride
        stride *= int(source_type.shape[dim])
    return element_offset


def _emit_local_store(state: _StructuralState, effect: _LocalStoreEffect):
    if getattr(effect.value.converted_type, "kind", None) == "fragment":
        _emit_fragment_local_store(state, effect)
        return
    value_components = _components(state.values[effect.value_id])
    result_count = len(value_components)
    value_type = value_components[0].type
    pointers = _local_linear_pointers(
        state,
        effect.memdesc_id,
        effect.value_id,
        value_type,
        result_count,
    )
    token = _memory_token_operand(state, effect.token_id)
    for index, (value_component, pointer) in enumerate(zip(value_components, pointers)):
        mask = _component_active_mask(state, effect.value_id, index, value_type)
        if mask is None:
            token = state.builder.store(value_component, pointer, after=token)
        else:
            token = _emit_masked_store_component(
                state,
                value_component,
                pointer,
                mask,
                token,
            )
    state.mem_token = state.builder.barrier(token)


def _emit_async_copy(state: _StructuralState, value: _AsyncCopyValue):
    if _can_emit_structural_buffer_dma_async_copy(value):
        _emit_buffer_async_copy_dma(state, value)
        return
    if _can_emit_structural_dma_async_copy(value):
        _emit_async_copy_dma(state, value)
        return
    if not _can_emit_structural_async_copy(value):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower this async copy yet"
        )
    source_ptrs = _components(state.values[value.source_address_value_id])
    source_type = _source_type(state, value.source_address_value_id)
    result_count = len(source_ptrs)
    element_type = source_type.pointee_type or source_type.element_type
    if element_type is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot infer async copy element type"
        )
    value_type = _parse_type(
        state.ir,
        f"!wave.simd<{element_type}, {_simd_width(state.dsl, source_ptrs[0].type)}>",
    )
    dest_ptrs = _local_linear_pointers(
        state,
        value.memdesc_value_id,
        value.source_address_value_id,
        value_type,
        result_count,
    )
    masks = None
    if value.mask_value_id is not None:
        masks = _component_tuple_for_count(
            state.values[value.mask_value_id],
            result_count,
            "async copy mask",
        )
    others = None
    if value.other_value_id is not None:
        others = _component_tuple_for_count(
            state.values[value.other_value_id],
            result_count,
            "async copy other",
        )
    token = _ensure_mem_token(state)
    for index, (source_ptr, dest_ptr) in enumerate(zip(source_ptrs, dest_ptrs)):
        mask = None if masks is None else masks[index]
        other = None if others is None else _ensure_value_type(
            state,
            others[index],
            value_type,
            "async copy other",
        )
        token = _emit_async_copy_component(
            state,
            source_ptr,
            dest_ptr,
            mask,
            other,
            value_type,
            token,
        )
    state.mem_token = token
    state.values[value.value_id] = token


def _emit_async_copy_dma(state: _StructuralState, value: _AsyncCopyValue):
    source_type = _source_type(state, value.source_address_value_id)
    memdesc_type = _source_type(state, value.memdesc_value_id)
    if source_type.kind != "tensor" or memdesc_type.kind != "memdesc":
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter expected tensor-to-memdesc DMA"
        )
    if len(source_type.shape) != 1 or tuple(source_type.shape) != tuple(memdesc_type.shape):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA currently supports rank-1 matching shapes only"
        )
    packet_bytes = _structural_dma_packet_bytes(source_type)
    packet_elements = packet_bytes // int(source_type.element_byte_width)
    element_count = _product(source_type.shape)
    width = int(state.result.values[value.source_address_value_id].converted_type.lane_width or 64)
    if element_count != width * packet_elements:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA currently supports exactly one wave of packets"
        )
    scalar_base_id = _simple_dma_scalar_base_id(state, value.source_address_value_id)
    if scalar_base_id is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA expected tt.addptr(tt.splat(base), tt.make_range)"
        )
    base = _require_structural_value(state, scalar_base_id)
    lane = state.builder.workitem_id(0, state.dsl.i32(), width)
    lane = state.builder.assume_range(lane, 0, width - 1)
    offset = lane
    if packet_elements != 1:
        scale = state.builder.constant(state.dsl.i32(), packet_elements)
        scale = state.builder.splat(scale, state.dsl.i32(), width)
        offset = state.builder.muli(lane, scale)
    offset = state.builder.assume_range(offset, 0, element_count - packet_elements)
    source_ptr_type = _parse_type(
        state.ir,
        f"!wave.simd<!wave.ptr<#wave.global, {source_type.pointee_type}>, {width}>",
    )
    source = state.builder.ptr_add(base, offset, source_ptr_type)
    dest = _ptr_cast(
        state,
        _require_structural_value(state, value.memdesc_value_id),
        _parse_type(state.ir, "!wave.ptr<#wave.shared, i32>"),
    )
    token = state.builder.dma_load_lds(
        source,
        dest,
        after=_ensure_mem_token(state),
        bytes=packet_bytes,
    )
    state.mem_token = token
    state.values[value.value_id] = token


def _emit_buffer_async_copy_dma(state: _StructuralState, value: _AsyncCopyValue):
    lowering = _select_structural_buffer_dma_lowering(state, value)
    pointer_type = lowering["pointer_type"]
    packet_bytes = lowering["packet_bytes"]
    width = lowering["width"]
    token = _ensure_mem_token(state)
    buffer = _buffer_pointer_for_base(state, value.source_address_value_id)
    source_ptr_type = _parse_type(
        state.ir,
        f"!wave.simd<!wave.ptr<#waveamd.buffer, {pointer_type.pointee_type}>, {width}>",
    )
    base_dest = _ptr_cast(
        state,
        _require_structural_value(state, value.memdesc_value_id),
        _parse_type(state.ir, "!wave.ptr<#wave.shared, i32>"),
    )
    for component in range(lowering["component_count"]):
        coords = _structural_dma_packet_start_coords(state, lowering, component)
        offset = _materialize_coordinate_value(
            state,
            value.source_offset_value_id,
            coords,
        )
        offset = _ensure_coordinate_simd_i32(state, offset, width)
        offset_upper = _buffer_offset_upper_bound(pointer_type, packet_bytes)
        if offset_upper is not None:
            offset = state.builder.assume_range(offset, 0, offset_upper)
        source = state.builder.ptr_add(buffer, offset, source_ptr_type)
        dest = _structural_dma_destination_pointer(
            state,
            base_dest,
            lowering,
            component,
        )
        mask = None
        if lowering["mask_value_id"] is not None:
            mask = _materialize_coordinate_value(
                state,
                lowering["mask_value_id"],
                coords,
            )
        token = _emit_buffer_dma_load_lds(
            state,
            source,
            dest,
            mask,
            token,
            packet_bytes,
        )
    state.mem_token = token
    state.values[value.value_id] = token


def _emit_buffer_dma_load_lds(state, source, dest, mask, token, packet_bytes):
    if mask is None:
        return state.builder.dma_load_lds(
            source,
            dest,
            after=token,
            bytes=packet_bytes,
        )
    mem_token_type = state.dsl.mem_token_type()
    with state.builder.where(mask, [mem_token_type]) as where:
        inner_token = state.builder.dma_load_lds(
            source,
            dest,
            after=token,
            bytes=packet_bytes,
        )
        state.builder.yield_([inner_token])
        with where.otherwise():
            state.builder.yield_([token])
    return where.results_[0]


def _select_structural_buffer_dma_lowering(
    state: _StructuralState,
    value: _AsyncCopyValue,
):
    pointer_type = _source_type(state, value.source_address_value_id)
    source_type = _source_type(state, value.source_offset_value_id)
    memdesc_type = _source_type(state, value.memdesc_value_id)
    if pointer_type.kind != "pointer" or pointer_type.pointee_type is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA expected typed base pointer"
        )
    if source_type.kind != "tensor" or memdesc_type.kind != "memdesc":
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA expected tensor offsets to memdesc"
        )
    if tuple(source_type.shape) != tuple(memdesc_type.shape):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA currently supports matching "
            "offset/destination shapes only"
        )
    if pointer_type.pointee_type != memdesc_type.element_type:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA element type mismatch: "
            f"{pointer_type.pointee_type} vs {memdesc_type.element_type}"
        )
    if not _can_materialize_coordinate_value(
        state.result.values.get(value.source_offset_value_id)
    ):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA cannot materialize source offsets "
            "at packet coordinates"
        )
    packet_bytes = _structural_dma_packet_bytes(pointer_type)
    packet_elements = packet_bytes // int(pointer_type.element_byte_width)
    _require_structural_dma_source_packet_contiguous(
        state,
        value,
        source_type,
        packet_elements,
    )
    if _is_structural_buffer_dma_candidate(value):
        _require_structural_dma_source_offset_nonnegative(state, value, source_type)
    width, cta_threads = _structural_dma_thread_layout(source_type)
    total_elements = _product(source_type.shape)
    if total_elements % packet_elements:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA expected packet-divisible tensor"
        )
    destination_remap = _structural_dma_destination_remap(
        memdesc_type,
        packet_elements,
        packet_bytes,
    )
    if destination_remap is None:
        _require_structural_dma_destination_packets(
            memdesc_type,
            packet_elements,
            packet_bytes,
            width,
        )
    if value.mask_value_id is not None:
        _require_structural_dma_packet_mask_uniform(
            state,
            value.mask_value_id,
            source_type.shape,
            packet_elements,
        )
    total_packets = total_elements // packet_elements
    packet_stride = cta_threads if total_packets >= cta_threads else width
    if total_packets % packet_stride:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer DMA expected packet count to be a full "
            "wave/CTA-thread multiple"
        )
    return {
        "pointer_type": pointer_type,
        "source_shape": tuple(int(dim) for dim in source_type.shape),
        "memdesc_type": memdesc_type,
        "packet_bytes": int(packet_bytes),
        "packet_elements": int(packet_elements),
        "width": int(width),
        "cta_threads": int(packet_stride),
        "component_count": total_packets // packet_stride,
        "mask_value_id": value.mask_value_id,
        "destination_remap": destination_remap,
    }


def _structural_dma_packet_start_coords(state, lowering, component):
    if lowering.get("destination_remap") is not None:
        return _emit_remapped_structural_dma_packet_start_coords(
            state,
            lowering,
            component,
        )
    width = lowering["width"]
    lane = state.builder.workitem_id(0, state.dsl.i32(), width)
    lane = state.builder.assume_range(lane, 0, width - 1)
    packet_index = _emit_simd_add_const(
        state,
        lane,
        int(component) * int(lowering["cta_threads"]),
    )
    logical = _emit_simd_binary_const(
        state,
        "muli",
        packet_index,
        lowering["packet_elements"],
    )
    return _emit_row_major_coords_from_linear(
        state,
        logical,
        lowering["source_shape"],
    )


def _emit_remapped_structural_dma_packet_start_coords(state, lowering, component):
    remap = lowering["destination_remap"]
    if remap["kind"] != "swizzled_physical_packets":
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA cannot emit unknown destination remap "
            f"{remap['kind']}"
        )
    width = lowering["width"]
    lane = state.builder.workitem_id(0, state.dsl.i32(), width)
    lane = state.builder.assume_range(lane, 0, width - 1)
    packet = _emit_simd_add_const(
        state,
        lane,
        int(component) * int(lowering["cta_threads"]),
    )
    groups_per_row = int(remap["groups_per_row"])
    row = _emit_simd_binary_const(state, "divui", packet, groups_per_row)
    physical_col_group = _emit_simd_binary_const(
        state,
        "remui",
        packet,
        groups_per_row,
    )
    row_phase_outer = _emit_simd_binary_const(
        state,
        "divui",
        row,
        int(remap["per_phase"]),
    )
    phase = _emit_simd_binary_const(
        state,
        "remui",
        row_phase_outer,
        int(remap["max_phase"]),
    )
    logical_col_group = state.builder.binary(
        _binary_kind(state.dsl, "xori"),
        physical_col_group,
        phase,
    )
    col = _emit_simd_binary_const(
        state,
        "muli",
        logical_col_group,
        int(remap["vec"]),
    )
    return (row, col)


def _structural_dma_thread_layout(source_type):
    attr = source_type.encoding_attr
    if attr is not None and _attr_bool(attr, "is_blocked_encoding"):
        threads_per_warp = tuple(
            int(value) for value in _attr_value(attr, "get_blocked_threads_per_warp")
        )
        warps_per_cta = tuple(
            int(value) for value in _attr_value(attr, "get_blocked_warps_per_cta")
        )
        width = _product(threads_per_warp)
        cta_threads = width * _product(warps_per_cta)
    elif attr is not None and _attr_bool(attr, "is_linear_encoding"):
        block_bases = tuple(
            _linear_basis_vector(basis)
            for basis in _attr_value(attr, "get_linear_block_bases")
        )
        if block_bases:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural DMA does not support linear block bases "
                f"{block_bases}"
            )
        lane_bases = tuple(
            _linear_basis_vector(basis)
            for basis in _attr_value(attr, "get_linear_lane_bases")
        )
        warp_bases = tuple(
            _linear_basis_vector(basis)
            for basis in _attr_value(attr, "get_linear_warp_bases")
        )
        width = 1 << len(lane_bases)
        cta_threads = width * (1 << len(warp_bases))
    else:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA expected blocked or linear source offset "
            "encoding"
        )
    if width <= 0 or cta_threads <= 0:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA expected non-empty source layout"
        )
    return width, cta_threads


def _linear_basis_vector(basis):
    return tuple(int(value) for value in basis)


def _require_structural_dma_source_packet_contiguous(
    state,
    value,
    source_type,
    packet_elements,
):
    if int(packet_elements) <= 1:
        return
    inner_dim = len(source_type.shape) - 1
    coord_coeffs = tuple(
        1 if dim == inner_dim else 0 for dim in range(len(source_type.shape))
    )
    coeff = _coordinate_inner_coeff(
        state,
        value.source_offset_value_id,
        coord_coeffs,
        packet_elements,
    )
    if coeff is None or coeff[0] != 1:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA cannot prove source packets are contiguous "
            "along the innermost dimension"
        )


def _require_structural_dma_source_offset_nonnegative(state, value, source_type):
    if _coordinate_nonnegative(
        state,
        value.source_offset_value_id,
        tuple(True for _ in source_type.shape),
    ):
        return
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural DMA cannot prove source offsets are nonnegative"
    )


def _require_structural_dma_packet_mask_uniform(
    state,
    mask_value_id,
    shape,
    packet_elements,
):
    if _mask_uniform_for_coord_group(
        state,
        mask_value_id,
        _dma_packet_coord_group_facts(shape, packet_elements),
    ):
        return
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural DMA cannot prove mask is uniform over packet"
    )


def _dma_packet_coord_group_facts(shape, packet_elements):
    inner_dim = len(tuple(shape)) - 1
    return tuple(
        _CoordGroupFact(int(packet_elements), 0, int(packet_elements) - 1)
        if dim == inner_dim
        else _CoordGroupFact(1, 0, 0)
        for dim in range(len(tuple(shape)))
    )


def _coordinate_inner_coeff(state, value_id, coord_coeffs, packet_elements=None):
    source_type = _source_type(state, value_id)
    value = state.result.values.get(value_id)
    if value is None:
        return None
    if source_type.kind != "tensor":
        return _scalar_inner_coeff(value)
    if isinstance(value, _ConstantValue):
        return 0, _constant_int_record(value)
    if isinstance(value, _RangeValue):
        if len(coord_coeffs) != 1:
            return None
        return int(coord_coeffs[0]), int(value.start)
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return _coordinate_inner_coeff(state, value.operand_id, (), packet_elements)
        if value.op_name == "tt.broadcast":
            operand_coeffs = _broadcast_operand_coord_coeffs(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_coeffs,
            )
            if operand_coeffs is None:
                return None
            return _coordinate_inner_coeff(
                state,
                value.operand_id,
                operand_coeffs,
                packet_elements,
            )
        return None
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _coordinate_inner_coeff(
                state,
                value.operand_id,
                coord_coeffs[:axis] + coord_coeffs[axis + 1 :],
                packet_elements,
            )
        if value.op_name == "ttg.convert_layout":
            return _coordinate_inner_coeff(
                state,
                value.operand_id,
                coord_coeffs,
                packet_elements,
            )
        return None
    if isinstance(value, _CastValue):
        return _coordinate_inner_coeff(
            state,
            value.operand_id,
            coord_coeffs,
            packet_elements,
        )
    if isinstance(value, _BinaryValue):
        if value.kind in {"remsi", "remui"}:
            rem = _coordinate_rem_inner_coeff(
                state,
                value,
                coord_coeffs,
                packet_elements,
            )
            if rem is not None:
                return rem
        lhs = _coordinate_inner_coeff(
            state,
            value.lhs_id,
            coord_coeffs,
            packet_elements,
        )
        rhs = _coordinate_inner_coeff(
            state,
            value.rhs_id,
            coord_coeffs,
            packet_elements,
        )
        return _combine_inner_coeffs(value.kind, lhs, rhs)
    return None


def _coordinate_rem_inner_coeff(state, value, coord_coeffs, packet_elements):
    lhs = _coordinate_inner_coeff(state, value.lhs_id, coord_coeffs, packet_elements)
    rhs = _coordinate_inner_coeff(state, value.rhs_id, coord_coeffs, packet_elements)
    if lhs is None or rhs is None:
        return None
    lhs_coeff, _lhs_const = lhs
    rhs_coeff, _rhs_const = rhs
    if rhs_coeff != 0:
        return None
    if lhs_coeff == 0:
        return 0, None
    if lhs_coeff != 1 or packet_elements is None or int(packet_elements) <= 1:
        return None
    coord_facts = _packet_coord_facts_for_inner_coeffs(coord_coeffs, packet_elements)
    if coord_facts is None:
        return None
    lhs_fact = _value_div_delta_for_coord_group(state, value.lhs_id, coord_facts)
    rhs_fact = _value_div_delta_for_coord_group(state, value.rhs_id, coord_facts)
    packet_elements = int(packet_elements)
    if (
        lhs_fact is not None
        and rhs_fact is not None
        and lhs_fact.delta_min == 0
        and lhs_fact.delta_max == packet_elements - 1
        and rhs_fact.delta_min == 0
        and rhs_fact.delta_max == 0
        and _divisibility_covers(lhs_fact.divisibility, packet_elements)
        and _divisibility_covers(rhs_fact.divisibility, packet_elements)
    ):
        return 1, None
    return None


def _packet_coord_facts_for_inner_coeffs(coord_coeffs, packet_elements):
    facts = []
    seen_varying_dim = False
    for coeff in coord_coeffs:
        coeff = int(coeff)
        if coeff == 0:
            facts.append(_CoordGroupFact(1, 0, 0))
        elif coeff == 1 and not seen_varying_dim:
            facts.append(_CoordGroupFact(int(packet_elements), 0, int(packet_elements) - 1))
            seen_varying_dim = True
        else:
            return None
    return tuple(facts)


def _scalar_inner_coeff(value):
    if isinstance(value, _ConstantValue):
        return 0, _constant_int_record(value)
    if isinstance(value, _CastValue):
        return 0, None
    if isinstance(value, _BinaryValue):
        lhs = _scalar_inner_coeff(value.lhs)
        rhs = _scalar_inner_coeff(value.rhs)
        combined = _combine_inner_coeffs(value.kind, lhs, rhs)
        return (0, None) if combined is None else combined
    return 0, None


def _combine_inner_coeffs(kind, lhs, rhs):
    if lhs is None or rhs is None:
        return None
    lhs_coeff, lhs_const = lhs
    rhs_coeff, rhs_const = rhs
    if kind == "addi":
        const = None if lhs_const is None or rhs_const is None else lhs_const + rhs_const
        return lhs_coeff + rhs_coeff, const
    if kind == "subi":
        const = None if lhs_const is None or rhs_const is None else lhs_const - rhs_const
        return lhs_coeff - rhs_coeff, const
    if kind == "muli":
        if lhs_const is not None:
            const = None if rhs_const is None else lhs_const * rhs_const
            return lhs_const * rhs_coeff, const
        if rhs_const is not None:
            const = None if lhs_const is None else lhs_const * rhs_const
            return rhs_const * lhs_coeff, const
        if lhs_coeff == 0 and rhs_coeff == 0:
            return 0, None
    return None


def _broadcast_operand_coord_coeffs(operand_shape, result_shape, coord_coeffs):
    if len(operand_shape) > len(result_shape):
        return None
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, coeff in zip(
        operand_shape,
        result_shape[leading:],
        coord_coeffs[leading:],
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(int(coeff))
        elif int(operand_dim) == 1:
            mapped.append(0)
        else:
            return None
    return tuple(mapped)


def _coordinate_nonnegative(state, value_id, coord_nonnegative):
    source_type = _source_type(state, value_id)
    value = state.result.values.get(value_id)
    if value is None:
        return False
    if source_type.kind != "tensor":
        lower = _scalar_lower_bound(state, value_id)
        return lower is not None and lower >= 0
    if value_id in state.nonnegative_values:
        return True
    if isinstance(value, _ConstantValue):
        literal = _constant_int_record(value)
        return literal is not None and literal >= 0
    if isinstance(value, _RangeValue):
        return len(coord_nonnegative) == 1 and bool(coord_nonnegative[0]) and value.start >= 0
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            lower = _scalar_lower_bound(state, value.operand_id)
            return lower is not None and lower >= 0
        if value.op_name == "tt.broadcast":
            operand_nonnegative = _broadcast_operand_coord_nonnegative(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_nonnegative,
            )
            return (
                operand_nonnegative is not None
                and _coordinate_nonnegative(
                    state,
                    value.operand_id,
                    operand_nonnegative,
                )
            )
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _coordinate_nonnegative(
                state,
                value.operand_id,
                coord_nonnegative[:axis] + coord_nonnegative[axis + 1 :],
            )
        if value.op_name == "ttg.convert_layout":
            return _coordinate_nonnegative(state, value.operand_id, coord_nonnegative)
    if isinstance(value, _CastValue):
        return _coordinate_nonnegative(state, value.operand_id, coord_nonnegative)
    if isinstance(value, _BinaryValue):
        return _binary_coordinate_nonnegative(
            state,
            value,
            coord_nonnegative,
        )
    if isinstance(value, _MinValue):
        return _coordinate_nonnegative(
            state,
            value.lhs_id,
            coord_nonnegative,
        ) and _coordinate_nonnegative(state, value.rhs_id, coord_nonnegative)
    return False


def _binary_coordinate_nonnegative(state, value, coord_nonnegative):
    lhs_nonnegative = _coordinate_nonnegative(state, value.lhs_id, coord_nonnegative)
    rhs_nonnegative = _coordinate_nonnegative(state, value.rhs_id, coord_nonnegative)
    lhs_const = _constant_int_for_value(state, value.lhs_id)
    rhs_const = _constant_int_for_value(state, value.rhs_id)
    if value.kind == "addi":
        return lhs_nonnegative and rhs_nonnegative
    if value.kind == "subi":
        lhs_lower = _coordinate_lower_bound(state, value.lhs_id, coord_nonnegative)
        return lhs_lower is not None and rhs_const is not None and lhs_lower >= rhs_const
    if value.kind == "muli":
        return (lhs_nonnegative and rhs_nonnegative) or (
            lhs_const is not None and rhs_const is not None and lhs_const * rhs_const >= 0
        )
    if value.kind in {"divui", "remui"}:
        return True
    if value.kind in {"divsi", "remsi"}:
        return lhs_nonnegative and _coordinate_positive(
            state,
            value.rhs_id,
            coord_nonnegative,
        )
    return False


def _coordinate_positive(state, value_id, coord_nonnegative):
    source_type = _source_type(state, value_id)
    value = state.result.values.get(value_id)
    if value is None:
        return False
    if source_type.kind != "tensor":
        lower = _scalar_lower_bound(state, value_id)
        return lower is not None and lower > 0
    lower = _coordinate_lower_bound(state, value_id, coord_nonnegative)
    return lower is not None and lower > 0


def _coordinate_lower_bound(state, value_id, coord_nonnegative):
    source_type = _source_type(state, value_id)
    value = state.result.values.get(value_id)
    if source_type.kind != "tensor":
        return _scalar_lower_bound(state, value_id)
    if isinstance(value, _ConstantValue):
        return _constant_int_record(value)
    if isinstance(value, _RangeValue):
        if len(coord_nonnegative) == 1 and bool(coord_nonnegative[0]):
            return int(value.start)
        return None
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return _scalar_lower_bound(state, value.operand_id)
        if value.op_name == "tt.broadcast":
            operand_nonnegative = _broadcast_operand_coord_nonnegative(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coord_nonnegative,
            )
            if operand_nonnegative is None:
                return None
            return _coordinate_lower_bound(state, value.operand_id, operand_nonnegative)
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _coordinate_lower_bound(
                state,
                value.operand_id,
                coord_nonnegative[:axis] + coord_nonnegative[axis + 1 :],
            )
        if value.op_name == "ttg.convert_layout":
            return _coordinate_lower_bound(state, value.operand_id, coord_nonnegative)
    if isinstance(value, _CastValue):
        return _coordinate_lower_bound(state, value.operand_id, coord_nonnegative)
    if isinstance(value, _MinValue):
        lhs = _coordinate_lower_bound(state, value.lhs_id, coord_nonnegative)
        rhs = _coordinate_lower_bound(state, value.rhs_id, coord_nonnegative)
        if lhs is not None and rhs is not None:
            return min(lhs, rhs)
        return None
    if isinstance(value, _BinaryValue):
        return _binary_coordinate_lower_bound(state, value, coord_nonnegative)
    return 0 if _coordinate_nonnegative(state, value_id, coord_nonnegative) else None


def _binary_coordinate_lower_bound(state, value, coord_nonnegative):
    lhs = _coordinate_lower_bound(state, value.lhs_id, coord_nonnegative)
    rhs = _coordinate_lower_bound(state, value.rhs_id, coord_nonnegative)
    rhs_const = _constant_int_for_value(state, value.rhs_id)
    lhs_const = _constant_int_for_value(state, value.lhs_id)
    if value.kind == "addi" and lhs is not None and rhs is not None:
        return lhs + rhs
    if value.kind == "subi" and lhs is not None and rhs_const is not None:
        return lhs - rhs_const
    if value.kind == "muli":
        if lhs is not None and rhs is not None and lhs >= 0 and rhs >= 0:
            return lhs * rhs
        if lhs_const is not None and rhs is not None:
            return lhs_const * rhs
        if rhs_const is not None and lhs is not None:
            return rhs_const * lhs
    if value.kind in {"divui", "remui"}:
        return 0
    if value.kind in {"divsi", "remsi"}:
        if _coordinate_nonnegative(
            state,
            value.lhs_id,
            coord_nonnegative,
        ) and _coordinate_positive(state, value.rhs_id, coord_nonnegative):
            return 0
    return None


def _broadcast_operand_coord_nonnegative(
    operand_shape,
    result_shape,
    coord_nonnegative,
):
    if len(operand_shape) > len(result_shape):
        return None
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, nonnegative in zip(
        operand_shape,
        result_shape[leading:],
        coord_nonnegative[leading:],
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(bool(nonnegative))
        elif int(operand_dim) == 1:
            mapped.append(True)
        else:
            return None
    return tuple(mapped)


def _emit_row_major_coords_from_linear(state, linear, shape):
    remaining = linear
    coords = [None for _ in shape]
    for dim in reversed(range(len(shape))):
        extent = int(shape[dim])
        if dim == 0:
            coords[dim] = remaining
            continue
        coords[dim] = _emit_simd_binary_const(state, "remui", remaining, extent)
        remaining = _emit_simd_binary_const(state, "divui", remaining, extent)
    return tuple(coords)


def _emit_simd_add_const(state, value, constant):
    constant = int(constant)
    if constant == 0:
        return value
    return _emit_simd_binary_const(state, "addi", value, constant)


def _emit_simd_binary_const(state, kind, value, constant):
    width = _simd_width(state.dsl, value.type)
    const = state.builder.constant(state.dsl.i32(), int(constant))
    splat = state.builder.splat(const, state.dsl.i32(), width)
    return state.builder.binary(_binary_kind(state.dsl, kind), value, splat)


def _materialize_coordinate_value(state, value_id, coords):
    value = state.result.values.get(value_id)
    width = _coordinate_width(state, coords)
    if isinstance(value, _RangeValue):
        if len(coords) != 1:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer expected rank-1 "
                "coordinate for tt.make_range"
            )
        coord = _ensure_coordinate_simd_i32(state, coords[0], width)
        return _emit_simd_add_const(state, coord, int(value.start))
    if isinstance(value, _ConstantValue):
        literal = _scalar_constant_literal(value)
        if isinstance(literal, bool):
            result_type = _parse_type(state.ir, f"!wave.mask<{int(width)}>")
            return _wave_mask_constant(state, result_type, literal)
        if not isinstance(literal, int):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer supports only "
                "integer constants"
            )
        const = state.builder.constant(state.dsl.i32(), literal)
        return state.builder.splat(const, state.dsl.i32(), width)
    if isinstance(value, _ConvertedInputValue):
        raw = _require_structural_value(state, value_id)
        return _ensure_coordinate_simd_i32(state, raw, width)
    if isinstance(value, (_ProgramIdValue, _IfValue, _ForValue)):
        raw = _require_structural_value(state, value_id)
        if isinstance(raw, tuple):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer expected scalar "
                f"value for {type(value).__name__}"
            )
        return _ensure_coordinate_simd_i32(state, raw, width)
    if isinstance(value, _UnaryTensorValue):
        return _materialize_coordinate_unary_tensor(state, value, coords)
    if isinstance(value, _ForwardValue):
        return _materialize_coordinate_forward(state, value, coords)
    if isinstance(value, _CastValue):
        if value.kind != "fpconvert":
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer cannot lower "
                f"cast kind {value.kind}"
            )
        source = _materialize_coordinate_value(state, value.operand_id, coords)
        target_type = _parse_type(state.ir, value.converted_type.wave_type)
        if source.type == target_type:
            return source
        return state.builder.cast(source, target_type, state.dsl.CastKind.FpConvert)
    if isinstance(value, _BinaryValue):
        lhs = _materialize_coordinate_value(state, value.lhs_id, coords)
        rhs = _materialize_coordinate_value(state, value.rhs_id, coords)
        lhs = _ensure_coordinate_simd_i32(state, lhs, width)
        rhs = _ensure_coordinate_simd_i32(state, rhs, width)
        return state.builder.binary(
            _binary_kind(state.dsl, _effective_binary_kind(value, state)),
            lhs,
            rhs,
            nsw="nsw" in value.flags,
            nuw="nuw" in value.flags,
        )
    if isinstance(value, _CompareValue):
        lhs = _materialize_coordinate_value(state, value.lhs_id, coords)
        rhs = _materialize_coordinate_value(state, value.rhs_id, coords)
        if lhs.type != rhs.type:
            width = _coordinate_width(state, coords)
            lhs = _ensure_coordinate_simd_i32(state, lhs, width)
            rhs = _ensure_coordinate_simd_i32(state, rhs, width)
        return _emit_cmpi(state, value.predicate, lhs, rhs)
    if isinstance(value, _MaskAndValue):
        lhs = _materialize_coordinate_value(state, value.lhs_id, coords)
        rhs = _materialize_coordinate_value(state, value.rhs_id, coords)
        if lhs.type != rhs.type:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer cannot combine "
                f"mask types {lhs.type} and {rhs.type}"
            )
        false_mask = _wave_mask_constant(state, lhs.type, False)
        return state.builder.select(lhs, rhs, false_mask)
    if isinstance(value, _MinValue):
        lhs = _materialize_coordinate_value(state, value.lhs_id, coords)
        rhs = _materialize_coordinate_value(state, value.rhs_id, coords)
        width = _coordinate_width(state, coords)
        lhs = _ensure_coordinate_simd_i32(state, lhs, width)
        rhs = _ensure_coordinate_simd_i32(state, rhs, width)
        pred = _emit_cmpi(state, value.predicate, lhs, rhs)
        return state.builder.select(pred, lhs, rhs)
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural coordinate materializer cannot lower "
        f"{type(value).__name__}"
    )


def _materialize_coordinate_unary_tensor(state, value, coords):
    if value.op_name == "tt.broadcast":
        operand_coords = _broadcast_operand_coords(
            state,
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
            coords,
        )
        return _materialize_coordinate_value(state, value.operand_id, operand_coords)
    if value.op_name == "tt.splat":
        operand = _require_structural_value(state, value.operand_id)
        width = _coordinate_width(state, coords)
        return _ensure_coordinate_simd_i32(state, operand, width)
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural coordinate materializer cannot lower unary op "
        f"{value.op_name}"
    )


def _materialize_coordinate_forward(state, value, coords):
    if value.op_name == "tt.expand_dims":
        axis = _expand_dims_axis(
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
        )
        return _materialize_coordinate_value(
            state,
            value.operand_id,
            coords[:axis] + coords[axis + 1 :],
        )
    if value.op_name == "ttg.convert_layout":
        return _materialize_coordinate_value(state, value.operand_id, coords)
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural coordinate materializer cannot lower forward op "
        f"{value.op_name}"
    )


def _broadcast_operand_coords(state, operand_shape, result_shape, coords):
    if len(operand_shape) > len(result_shape):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural coordinate materializer cannot broadcast from "
            f"{operand_shape} to {result_shape}"
        )
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, coord in zip(
        operand_shape, result_shape[leading:], coords[leading:]
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(coord)
        elif int(operand_dim) == 1:
            mapped.append(_zero_coordinate_like(state, coord))
        else:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer cannot map "
                f"broadcast coordinate from {operand_shape} to {result_shape}"
            )
    return tuple(mapped)


def _expand_dims_axis(operand_shape, result_shape):
    candidates = []
    for axis in range(len(result_shape)):
        if int(result_shape[axis]) != 1:
            continue
        if tuple(result_shape[:axis] + result_shape[axis + 1 :]) == tuple(operand_shape):
            candidates.append(axis)
    if len(candidates) != 1:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural coordinate materializer cannot derive "
            f"expand_dims axis for {operand_shape} -> {result_shape}"
        )
    return candidates[0]


def _zero_coordinate_like(state, coord):
    width = _simd_width_for_value_type(coord.type)
    const = state.builder.constant(state.dsl.i32(), 0)
    return state.builder.splat(const, state.dsl.i32(), width)


def _coordinate_width(state, coords):
    if not coords:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural coordinate materializer needs tensor coordinates"
        )
    return _simd_width(state.dsl, coords[0].type)


def _ensure_coordinate_simd_i32(state, value, width):
    target_type = _parse_type(state.ir, f"!wave.simd<i32, {int(width)}>")
    if value.type == target_type:
        return value
    if _is_simd_type(state.dsl, value.type):
        if str(_simd_element_type(state.dsl, value.type)) == "index":
            return _emit_int_convert(state, value, target_type)
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural coordinate materializer expected i32 SIMD, "
            f"got {value.type}"
        )
    if str(value.type) != "i32":
        if str(value.type) == "index":
            value = _emit_int_convert(state, value, state.dsl.i32())
        else:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural coordinate materializer expected i32 scalar, "
                f"got {value.type}"
            )
    return state.builder.splat(value, state.dsl.i32(), width)


def _simd_width_for_value_type(type_):
    text = str(type_)
    if not text.startswith("!wave.simd<") or "," not in text:
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural coordinate materializer expected SIMD type, got {type_}"
        )
    return int(text.rsplit(",", 1)[1].rstrip(">").strip())


def _structural_dma_destination_pointer(state, base, lowering, component):
    byte_offset = _structural_dma_destination_component_byte_offset(
        lowering,
        component,
    )
    if byte_offset % 4:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA destination is not dword aligned"
        )
    dword_offset = byte_offset // 4
    if dword_offset == 0:
        return base
    const = state.builder.constant(state.dsl.i32(), dword_offset)
    return state.builder.ptr_add(base, const, base.type)


def _structural_dma_destination_component_byte_offset(lowering, component):
    packet_index = int(component) * int(lowering["cta_threads"])
    if lowering.get("destination_remap") is not None:
        return packet_index * int(lowering["packet_bytes"])
    return _dma_packet_static_byte_offset(
        lowering["memdesc_type"],
        lowering["packet_elements"],
        packet_index,
    )


def _require_structural_dma_destination_packets(
    memdesc_type,
    packet_elements,
    packet_bytes,
    width,
):
    total_elements = _product(memdesc_type.shape)
    if total_elements % int(packet_elements):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA destination tensor element count is not "
            "packet-divisible"
        )
    total_packets = total_elements // int(packet_elements)
    packet_offsets = []
    for packet_index in range(total_packets):
        coords = _dma_packet_start_coords_static(
            memdesc_type.shape,
            packet_elements,
            packet_index,
        )
        packet_offsets.append(
            _require_dma_packet_contiguous_from_coords(
                memdesc_type,
                coords,
                packet_elements,
                packet_bytes,
            )
        )
    for chunk_start in range(0, total_packets, int(width)):
        chunk = packet_offsets[chunk_start : chunk_start + int(width)]
        if len(chunk) != int(width):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural DMA destination final packet chunk is not "
                "a full wave"
            )
        first = chunk[0]
        for lane, byte_offset in enumerate(chunk[1:], start=1):
            expected = first + lane * int(packet_bytes)
            if byte_offset != expected:
                raise _UnsupportedStructuralEmission(
                    "tlx_wave structural DMA destination packet starts are not "
                    "whole-wave contiguous"
                )


def _dma_packet_start_coords_static(shape, packet_elements, packet_index):
    return _row_major_coords_static(
        tuple(int(dim) for dim in shape),
        int(packet_index) * int(packet_elements),
    )


def _require_dma_packet_contiguous_from_coords(
    memdesc_type,
    coords,
    packet_elements,
    packet_bytes,
):
    first = None
    inner_dim = len(memdesc_type.shape) - 1
    for element in range(int(packet_elements)):
        packet_coords = list(coords)
        packet_coords[inner_dim] += element
        if packet_coords[inner_dim] >= int(memdesc_type.shape[inner_dim]):
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural DMA packet crosses the innermost dimension"
            )
        byte_offset = _memdesc_static_byte_offset(memdesc_type, tuple(packet_coords))
        if first is None:
            first = byte_offset
            if first % 4:
                raise _UnsupportedStructuralEmission(
                    "tlx_wave structural DMA packet starts at unaligned byte offset"
                )
            continue
        expected = first + element * int(memdesc_type.element_byte_width)
        if byte_offset != expected:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural DMA packet is not physically contiguous"
            )
    if int(packet_elements) * int(memdesc_type.element_byte_width) != int(packet_bytes):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA packet byte width does not match element width"
        )
    return first


def _dma_packet_static_byte_offset(memdesc_type, packet_elements, packet_index):
    coords = _dma_packet_start_coords_static(
        memdesc_type.shape,
        packet_elements,
        packet_index,
    )
    return _memdesc_static_byte_offset(memdesc_type, coords)


def _memdesc_static_byte_offset(memdesc_type, coords):
    if memdesc_type.element_byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA cannot address memdesc with unknown element width"
        )
    if _is_identity_shared_memdesc(memdesc_type):
        return _row_major_linear_static(memdesc_type.shape, coords) * int(
            memdesc_type.element_byte_width
        )
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if swizzled is not None:
        return _swizzled_shared_static_byte_offset(
            memdesc_type,
            coords,
            swizzled,
        )
    padded = _padded_shared_memdesc_info(memdesc_type)
    if padded is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA destination has unsupported shared layout"
        )
    if padded["order"] != tuple(reversed(range(len(memdesc_type.shape)))):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA destination has unsupported padded order"
        )
    logical = _row_major_linear_static(memdesc_type.shape, coords)
    encoded = logical
    for interval, padding in zip(padded["intervals"], padded["paddings"]):
        encoded += (logical // int(interval)) * int(padding)
    return encoded * int(memdesc_type.element_byte_width)


def _swizzled_shared_static_byte_offset(memdesc_type, coords, swizzled):
    _validate_swizzled_shared_memdesc(memdesc_type, swizzled)
    if len(memdesc_type.shape) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA supports only rank-2 swizzled shared "
            f"memdescs, got {memdesc_type.shape}"
        )
    row = int(coords[-2])
    col = int(coords[-1])
    rows = int(memdesc_type.shape[-2])
    cols = int(memdesc_type.shape[-1])
    if row < 0 or row >= rows or col < 0 or col >= cols:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA swizzled coordinate exceeds shape "
            f"{coords} vs {memdesc_type.shape}"
        )
    phase = (row // int(swizzled["per_phase"])) % int(swizzled["max_phase"])
    col_group = col // int(swizzled["vec"])
    swizzled_col = (col_group ^ phase) * int(swizzled["vec"]) + (
        col % int(swizzled["vec"])
    )
    return (row * cols + swizzled_col) * int(memdesc_type.element_byte_width)


def _is_identity_shared_memdesc(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_swizzled_shared_encoding"):
        return False
    order = tuple(int(value) for value in _attr_value(attr, "get_swizzled_shared_order"))
    return (
        int(_attr_value(attr, "get_swizzled_shared_vec")) == 1
        and int(_attr_value(attr, "get_swizzled_shared_per_phase")) == 1
        and int(_attr_value(attr, "get_swizzled_shared_max_phase")) == 1
        and order == tuple(reversed(range(len(source_type.shape))))
    )


def _swizzled_shared_memdesc_info(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_swizzled_shared_encoding"):
        return None
    info = {
        "vec": int(_attr_value(attr, "get_swizzled_shared_vec")),
        "per_phase": int(_attr_value(attr, "get_swizzled_shared_per_phase")),
        "max_phase": int(_attr_value(attr, "get_swizzled_shared_max_phase")),
        "order": tuple(
            int(value) for value in _attr_value(attr, "get_swizzled_shared_order")
        ),
    }
    if _is_identity_shared_memdesc(source_type):
        return info
    _validate_swizzled_shared_memdesc(source_type, info)
    return info


def _validate_swizzled_shared_memdesc(memdesc_type, swizzled):
    if len(memdesc_type.shape) < 2 or tuple(swizzled["order"]) != (1, 0):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA supports only rank-2 order=[1, 0] "
            "swizzled shared memdescs"
        )
    if (
        int(swizzled["vec"]) <= 0
        or int(swizzled["per_phase"]) <= 0
        or int(swizzled["max_phase"]) <= 0
    ):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA found invalid swizzled shared parameters"
        )
    if int(memdesc_type.shape[-1]) % int(swizzled["vec"]):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural DMA swizzled columns are not divisible by vec"
        )


def _structural_dma_destination_remap(memdesc_type, packet_elements, packet_bytes):
    if _is_identity_shared_memdesc(memdesc_type):
        return None
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if swizzled is None:
        return None
    if len(tuple(memdesc_type.shape)) != 2:
        return None
    if tuple(swizzled["order"]) != (1, 0):
        return None
    if int(packet_elements) != int(swizzled["vec"]):
        return None
    if memdesc_type.element_byte_width is None:
        return None
    if int(packet_bytes) != int(swizzled["vec"]) * int(memdesc_type.element_byte_width):
        return None
    rows, cols = (int(dim) for dim in memdesc_type.shape)
    if rows <= 0 or cols <= 0 or cols % int(swizzled["vec"]):
        return None
    return {
        "kind": "swizzled_physical_packets",
        "vec": int(swizzled["vec"]),
        "per_phase": int(swizzled["per_phase"]),
        "max_phase": int(swizzled["max_phase"]),
        "groups_per_row": cols // int(swizzled["vec"]),
    }


def _padded_shared_memdesc_info(source_type):
    attr = source_type.encoding_attr
    if attr is None or not _attr_bool(attr, "is_padded_shared_encoding"):
        return None
    return {
        "intervals": tuple(
            int(value) for value in _attr_value(attr, "get_padded_shared_intervals")
        ),
        "paddings": tuple(
            int(value) for value in _attr_value(attr, "get_padded_shared_paddings")
        ),
        "order": tuple(
            int(value) for value in _attr_value(attr, "get_padded_shared_order")
        ),
    }


def _row_major_coords_static(shape, linear):
    coords = [0 for _ in shape]
    remaining = int(linear)
    for dim in reversed(range(len(shape))):
        if dim == 0:
            coords[dim] = remaining
        else:
            extent = int(shape[dim])
            coords[dim] = remaining % extent
            remaining //= extent
    return tuple(coords)


def _row_major_linear_static(shape, coords):
    linear = 0
    stride = 1
    for dim in reversed(range(len(shape))):
        linear += int(coords[dim]) * stride
        stride *= int(shape[dim])
    return linear


def _emit_async_copy_component(
    state,
    source_ptr,
    dest_ptr,
    mask,
    other,
    value_type,
    token,
):
    if mask is None:
        loaded, load_token = state.builder.load(source_ptr, value_type, after=token)
        return state.builder.store(loaded, dest_ptr, after=load_token)
    mem_token_type = state.dsl.mem_token_type()
    with state.builder.where(mask, [mem_token_type]) as where:
        loaded, load_token = state.builder.load(source_ptr, value_type, after=token)
        store_token = state.builder.store(loaded, dest_ptr, after=load_token)
        state.builder.yield_([store_token])
        with where.otherwise():
            if other is None:
                state.builder.yield_([token])
            else:
                state.builder.yield_(
                    [state.builder.store(other, dest_ptr, after=token)]
                )
    return where.results_[0]


def _emit_async_commit_group(state: _StructuralState, value: _AsyncCommitGroupValue):
    tokens = tuple(
        _require_structural_value(state, token_id)
        for token_id in value.member_token_ids
        if token_id in state.values
    )
    if not tokens:
        token = _ensure_mem_token(state)
    elif len(tokens) == 1:
        token = tokens[0]
    else:
        token = state.builder.join(*tokens)
    state.mem_token = token
    state.values[value.value_id] = token


def _emit_async_wait_effect(state: _StructuralState, effect: _AsyncWaitEffect):
    _emit_async_wait_token(state, effect)


def _emit_async_wait_value(state: _StructuralState, value: _AsyncWaitValue):
    state.values[value.value_id] = _emit_async_wait_token(state, value)


def _emit_async_wait_token(state: _StructuralState, effect):
    deps = _async_wait_dependencies(state, effect)
    if not deps:
        return _ensure_mem_token(state)
    state.builder.wait(*deps)
    token = state.builder.barrier(*deps)
    state.mem_token = token
    return token


def _async_wait_dependencies(state: _StructuralState, effect):
    if effect.input_token_ids:
        return tuple(
            _require_structural_value(state, token_id)
            for token_id in effect.input_token_ids
        )
    deps = []
    for group_index in effect.waited_group_indices:
        group = state.program.token_graph.groups[group_index]
        if group.token_value_id is not None and group.token_value_id in state.values:
            deps.append(state.values[group.token_value_id])
    return tuple(dict.fromkeys(deps))


def _emit_if(state: _StructuralState, value: _IfValue, converted_values):
    if all(result_id in state.values for result_id in value.result_ids):
        return
    condition = state.values[value.condition_id]
    if str(condition.type) != "i1":
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter expected scalar i1 scf.if condition, "
            f"got {condition.type}"
        )
    result_values = tuple(converted_values[result_id] for result_id in value.result_ids)
    result_types = tuple(
        _parse_type(state.ir, result_value.converted_type.wave_type)
        for result_value in result_values
    )
    with state.builder.if_(condition, result_types, otherwise=True) as if_builder:
        then_state = _child_state(state)
        _emit_structural_region_body(
            then_state, value.then_op_indices, value.then_values
        )
        state.builder.yield_(
            [
                _require_structural_value(then_state, value_id)
                for value_id in value.then_yield_ids
            ]
        )
        with if_builder.otherwise():
            else_state = _child_state(state)
            _emit_structural_region_body(
                else_state, value.else_op_indices, value.else_values
            )
            state.builder.yield_(
                [
                    _require_structural_value(else_state, value_id)
                    for value_id in value.else_yield_ids
                ]
            )
    for result_id, result in zip(value.result_ids, if_builder.results):
        state.values[result_id] = result


def _emit_if_effect(state: _StructuralState, effect: _IfEffect):
    condition = state.values[effect.condition_id]
    if str(condition.type) != "i1":
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter expected scalar i1 scf.if condition, "
            f"got {condition.type}"
        )
    has_else = bool(effect.else_op_indices or effect.else_effects)
    with state.builder.if_(condition, otherwise=has_else) as if_builder:
        then_state = _child_state(state)
        _emit_structural_region_body(
            then_state,
            effect.then_op_indices,
            effect.then_values,
            effect.then_effects,
        )
        if has_else:
            with if_builder.otherwise():
                else_state = _child_state(state)
                _emit_structural_region_body(
                    else_state,
                    effect.else_op_indices,
                    effect.else_values,
                    effect.else_effects,
                )


def _emit_for(state: _StructuralState, value: _ForValue, converted_values):
    if all(result_id in state.values for result_id in value.result_ids):
        return
    if value.body_effects:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter does not yet support side-effecting scf.for"
        )
    lower = _index_bound(state, value.lower_id)
    upper = _index_bound(state, value.upper_id)
    step = _index_bound(state, value.step_id)
    init_args = []
    iter_arg_counts = []
    for iter_arg_id, result_type in zip(value.iter_arg_ids, value.result_types):
        count = _loop_value_component_count(result_type)
        init_args.extend(
            _component_tuple_for_count(
                _require_structural_value(state, iter_arg_id),
                count,
                "scf.for iter_arg",
            )
        )
        iter_arg_counts.append(count)
    with state.builder.for_loop(lower, upper, step, init_args) as for_op:
        body_state = _child_state(
            state,
            _region_result_for_values(
                {
                    **state.result.values,
                    **value.body_values,
                },
                value.body_effects,
            ),
        )
        body_state.values[value.body_arg_ids[0]] = for_op.induction_variable
        iv_lower = _scalar_lower_bound(state, value.lower_id)
        if iv_lower is not None:
            body_state.lower_bounds[value.body_arg_ids[0]] = int(iv_lower)
            if int(iv_lower) >= 0:
                body_state.nonnegative_values.add(value.body_arg_ids[0])
            if int(iv_lower) > 0:
                body_state.positive_values.add(value.body_arg_ids[0])
        inner_iter_args = tuple(for_op.inner_iter_args)
        offset = 0
        for body_arg_id, count in zip(value.body_arg_ids[1:], iter_arg_counts):
            body_state.values[body_arg_id] = _pack(
                inner_iter_args[offset : offset + count]
            )
            offset += count
        _emit_structural_region_body(
            body_state, value.body_op_indices, value.body_values
        )
        yielded = []
        for yield_id, count in zip(value.body_yield_ids, iter_arg_counts):
            yielded.extend(
                _component_tuple_for_count(
                    _require_structural_value(body_state, yield_id),
                    count,
                    "scf.for yield",
                )
            )
        state.builder.yield_(yielded)
    results = tuple(for_op.results)
    offset = 0
    for result_id, count in zip(value.result_ids, iter_arg_counts):
        state.values[result_id] = _pack(results[offset : offset + count])
        offset += count


def _loop_value_component_count(result_type):
    return max(1, int(getattr(result_type, "component_count", 1)))


def _emit_store(state: _StructuralState, effect: _StoreEffect):
    if getattr(effect.value.converted_type, "kind", None) == "fragment":
        _emit_fragment_global_store(state, effect)
        return
    value_components = _components(state.values[effect.value_id])
    pointer_components = _components(state.values[effect.pointer_id])
    mask_components = (
        () if effect.mask_id is None else _components(state.values[effect.mask_id])
    )
    component_count = max(len(value_components), len(pointer_components), len(mask_components))
    value_components = _component_tuple_for_count(
        state.values[effect.value_id], component_count, "store value"
    )
    pointer_components = _component_tuple_for_count(
        state.values[effect.pointer_id], component_count, "store pointer"
    )
    if effect.mask_id is not None:
        mask_components = _component_tuple_for_count(
            state.values[effect.mask_id], component_count, "store mask"
        )
    token = _ensure_mem_token(state)
    for index, (value, pointer) in enumerate(zip(value_components, pointer_components)):
        active = _component_active_mask(state, effect.value_id, index, value.type)
        mask = None if effect.mask_id is None else mask_components[index]
        mask = _combine_optional_masks(state, active, mask)
        if mask is None:
            token = state.builder.store(value, pointer, after=token)
        else:
            token = _emit_masked_store_component(
                state, value, pointer, mask, token
            )
    state.mem_token = token


def _emit_fragment_global_store(state: _StructuralState, effect: _StoreEffect):
    logical_type = _source_type(state, effect.value_id)
    if logical_type.kind != "tensor" or len(logical_type.shape) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment stores only for "
            f"rank-2 tensor values, got {logical_type.raw}"
        )
    if logical_type.element_type not in {"f16", "f32"}:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment stores only for "
            f"f16/f32 values, got {logical_type.raw}"
        )
    fragments = _components(_require_structural_value(state, effect.value_id))
    if not fragments:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter expected at least one store fragment"
        )
    frag = state.dsl.FragmentType(fragments[0].type)
    if int(frag.role) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports stores only from accumulator "
            f"fragments, got {fragments[0].type}"
        )
    mma = _fragment_store_mma_shape(state, effect.value_id)
    tile_rows = _tile_rep_count(logical_type.shape[0], mma.output_tile_shape[0])
    tile_cols = _tile_rep_count(logical_type.shape[1], mma.output_tile_shape[1])
    expected_fragments = int(tile_rows) * int(tile_cols)
    if len(fragments) != expected_fragments:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter fragment store tile mismatch: "
            f"got {len(fragments)} fragment(s), expected {expected_fragments}"
        )
    token = _ensure_mem_token(state)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            fragment = fragments[_tile_index((tile_rows, tile_cols), tile_row, tile_col)]
            frag = state.dsl.FragmentType(fragment.type)
            store_vector = _emit_fragment_store_vector(state, fragment, logical_type, frag)
            tile_offsets = (
                tile_row * mma.output_tile_shape[0],
                tile_col * mma.output_tile_shape[1],
            )
            for component, coords in enumerate(
                _emit_fragment_component_coords(state, frag, tile_offsets)
            ):
                value = _emit_fragment_store_component_value(
                    state,
                    store_vector,
                    logical_type,
                    frag,
                    component,
                )
                pointer = _materialize_pointer_coordinate(
                    state,
                    effect.pointer_id,
                    coords,
                    value.type,
                )
                mask = (
                    None
                    if effect.mask_id is None
                    else _materialize_coordinate_value(state, effect.mask_id, coords)
                )
                if mask is None:
                    token = state.builder.store(value, pointer, after=token)
                else:
                    token = _emit_masked_store_component(
                        state,
                        value,
                        pointer,
                        mask,
                        token,
                    )
    state.mem_token = token


def _emit_fragment_local_store(state: _StructuralState, effect: _LocalStoreEffect):
    logical_type = _source_type(state, effect.value_id)
    memdesc_type = _source_type(state, effect.memdesc_id)
    if logical_type.kind != "tensor" or len(logical_type.shape) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment local_store only "
            f"for rank-2 tensor values, got {logical_type.raw}"
        )
    if logical_type.element_type not in {"f16", "f32"}:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment local_store only "
            f"for f16/f32 values, got {logical_type.raw}"
        )
    if memdesc_type.element_type != logical_type.element_type:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot lower fragment local_store "
            "with mismatched value/memdesc element types: "
            f"{logical_type.element_type} vs {memdesc_type.element_type}"
        )
    fragments = _components(_require_structural_value(state, effect.value_id))
    if not fragments:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter expected at least one local_store fragment"
        )
    frag = state.dsl.FragmentType(fragments[0].type)
    if int(frag.role) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports local stores only from "
            f"accumulator fragments, got {fragments[0].type}"
        )
    mma = _fragment_store_mma_shape(state, effect.value_id)
    tile_rows = _tile_rep_count(logical_type.shape[0], mma.output_tile_shape[0])
    tile_cols = _tile_rep_count(logical_type.shape[1], mma.output_tile_shape[1])
    expected_fragments = int(tile_rows) * int(tile_cols)
    if len(fragments) != expected_fragments:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter fragment local_store tile mismatch: "
            f"got {len(fragments)} fragment(s), expected {expected_fragments}"
        )
    token = _memory_token_operand(state, effect.token_id)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            fragment = fragments[_tile_index((tile_rows, tile_cols), tile_row, tile_col)]
            frag = state.dsl.FragmentType(fragment.type)
            store_vector = _emit_fragment_store_vector(state, fragment, logical_type, frag)
            tile_offsets = (
                tile_row * mma.output_tile_shape[0],
                tile_col * mma.output_tile_shape[1],
            )
            for component, coords in enumerate(
                _emit_fragment_component_coords(state, frag, tile_offsets)
            ):
                value = _emit_fragment_store_component_value(
                    state,
                    store_vector,
                    logical_type,
                    frag,
                    component,
                )
                pointer = _materialize_local_store_pointer(
                    state,
                    effect.memdesc_id,
                    memdesc_type,
                    coords,
                    value.type,
                )
                token = state.builder.store(value, pointer, after=token)
    state.mem_token = state.builder.barrier(token)


def _emit_buffer_store(state: _StructuralState, effect: _BufferStoreEffect):
    if getattr(effect.value.converted_type, "kind", None) != "fragment":
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter currently supports amdg.buffer_store "
            "only for fragment values"
        )
    _emit_fragment_buffer_store(state, effect)


def _emit_fragment_buffer_store(state: _StructuralState, effect: _BufferStoreEffect):
    logical_type = _source_type(state, effect.value_id)
    if logical_type.kind != "tensor" or len(logical_type.shape) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment buffer stores only "
            f"for rank-2 tensor values, got {logical_type.raw}"
        )
    if logical_type.element_type not in {"f16", "f32"}:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports fragment buffer stores only "
            f"for f16/f32 values, got {logical_type.raw}"
        )
    fragments = _components(_require_structural_value(state, effect.value_id))
    if not fragments:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter expected at least one store fragment"
        )
    frag = state.dsl.FragmentType(fragments[0].type)
    if int(frag.role) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports buffer stores only from "
            f"accumulator fragments, got {fragments[0].type}"
        )
    mma = _fragment_store_mma_shape(state, effect.value_id)
    tile_rows = _tile_rep_count(logical_type.shape[0], mma.output_tile_shape[0])
    tile_cols = _tile_rep_count(logical_type.shape[1], mma.output_tile_shape[1])
    expected_fragments = int(tile_rows) * int(tile_cols)
    if len(fragments) != expected_fragments:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter fragment buffer store tile mismatch: "
            f"got {len(fragments)} fragment(s), expected {expected_fragments}"
        )
    token = _ensure_mem_token(state)
    for tile_row in range(tile_rows):
        for tile_col in range(tile_cols):
            fragment = fragments[_tile_index((tile_rows, tile_cols), tile_row, tile_col)]
            frag = state.dsl.FragmentType(fragment.type)
            store_vector = _emit_fragment_store_vector(state, fragment, logical_type, frag)
            tile_offsets = (
                tile_row * mma.output_tile_shape[0],
                tile_col * mma.output_tile_shape[1],
            )
            component_coords = _emit_fragment_component_coords(state, frag, tile_offsets)
            component = 0
            while component < len(component_coords):
                packed = _try_emit_buffer_fragment_vector_store(
                    state,
                    effect,
                    logical_type,
                    frag,
                    tile_offsets,
                    component_coords,
                    store_vector,
                    component,
                    token,
                )
                if packed is not None:
                    token, count = packed
                    component += count
                    continue
                coords = component_coords[component]
                value = _emit_fragment_store_component_value(
                    state,
                    store_vector,
                    logical_type,
                    frag,
                    component,
                )
                token = _emit_buffer_store_component(
                    state,
                    effect,
                    coords,
                    value,
                    token,
                )
                component += 1
    state.mem_token = token


def _fragment_store_mma_shape(state: _StructuralState, value_id: int):
    source_type = _fragment_physical_source_type(state, value_id)
    parent = _amd_mfma_encoding_info(source_type.encoding_attr)
    if parent is None:
        parent = _blocked_encoding_info(source_type.encoding_attr)
    if parent is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter expected fragment store value to "
            "originate from #ttg.amd_mfma or a supported blocked MFMA parent, "
            f"got {source_type.raw}"
        )
    return _mma_shape_for_parent(parent, "fragment buffer store")


def _fragment_physical_source_type(state: _StructuralState, value_id: int):
    source_type = _source_type(state, value_id)
    if (
        _amd_mfma_encoding_info(source_type.encoding_attr) is not None
        or _blocked_encoding_info(source_type.encoding_attr) is not None
    ):
        return source_type
    value = state.result.values.get(value_id)
    if isinstance(value, _ForwardValue) and value.op_name == "ttg.convert_layout":
        return _fragment_physical_source_type(state, value.operand_id)
    if isinstance(value, _CastValue):
        cast_type = _source_type(state, value.value_id)
        if (
            _amd_mfma_encoding_info(cast_type.encoding_attr) is not None
            or _blocked_encoding_info(cast_type.encoding_attr) is not None
        ):
            return cast_type
        return _fragment_physical_source_type(state, value.operand_id)
    return source_type


def _emit_fragment_store_vector(state, fragment, logical_type, frag):
    unpack_type = state.dsl.simd_type(
        state.dsl.vector_type(int(frag.registers), frag.element_type),
        width=int(frag.wave_size),
    )
    unpack = state.dsl.waveamd.FragmentUnpackOp(unpack_type, fragment).result
    logical_element = _source_element_type(state.dsl, logical_type)
    if str(frag.element_type) == str(logical_element):
        return unpack
    store_vector_type = state.dsl.simd_type(
        state.dsl.vector_type(int(frag.registers), logical_element),
        width=int(frag.wave_size),
    )
    return state.builder.cast(unpack, store_vector_type, state.dsl.CastKind.FpConvert)


def _emit_fragment_store_component_value(
    state,
    store_vector,
    logical_type,
    frag,
    component,
):
    extracted_type = state.dsl.simd_type(
        _source_element_type(state.dsl, logical_type),
        width=int(frag.wave_size),
    )
    return state.dsl.wave.ExtractOp(extracted_type, store_vector, int(component)).result


def _emit_fragment_component_coords(state, frag, tile_offsets):
    roots = _fragment_component_coord_roots(state, frag)
    width = int(frag.wave_size)
    return tuple(
        (
            _emit_simd_add_const(
                state,
                _ensure_coordinate_simd_i32(state, row, width),
                tile_offsets[0],
            ),
            _emit_simd_add_const(
                state,
                _ensure_coordinate_simd_i32(state, col, width),
                tile_offsets[1],
            ),
        )
        for row, col in roots
    )


def _fragment_component_coord_roots(state, frag):
    width = int(frag.wave_size)
    lane = state.builder.workitem_id(0, state.dsl.i32(), width)
    lane = state.builder.assume_range(lane, 0, width - 1)
    roots = []
    registers = int(frag.registers)
    lane_sym = state.dsl.sym("lane")
    if registers == 4:
        for component in range(registers):
            roots.append(
                (
                    state.builder.index_expr(
                        state.dsl.floor(lane_sym / 16) * 2 + component // 2,
                        {lane_sym: lane},
                    ),
                    state.builder.index_expr(
                        state.dsl.mod(lane_sym, 16) * 2 + component % 2,
                        {lane_sym: lane},
                    ),
                )
            )
    elif registers == 16:
        for component in range(registers):
            roots.append(
                (
                    state.builder.index_expr(
                        state.dsl.mod(lane_sym, 32),
                        {lane_sym: lane},
                    ),
                    state.builder.index_expr(
                        state.dsl.floor(lane_sym / 32) * 4
                        + (component % 4)
                        + 8 * (component // 4),
                        {lane_sym: lane},
                    ),
                )
            )
    else:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot map accumulator fragment with "
            f"{registers} registers"
        )
    return tuple(roots)


def _try_emit_buffer_fragment_vector_store(
    state,
    effect,
    logical_type,
    frag,
    tile_offsets,
    component_coords,
    store_vector,
    component,
    token,
):
    count = _fragment_vector_store_count(
        effect,
        logical_type,
        frag,
        component,
        len(component_coords),
    )
    if count <= 1:
        return None
    coord_facts = _fragment_store_coord_group_facts(
        frag,
        tile_offsets,
        component,
        count,
    )
    if effect.mask_id is not None and not _mask_uniform_for_coord_group(
        state,
        effect.mask_id,
        coord_facts,
    ):
        return None
    value = _emit_fragment_store_vector_group(
        state,
        store_vector,
        logical_type,
        frag,
        component,
        count,
    )
    token = _emit_buffer_store_component(
        state,
        effect,
        component_coords[component],
        value,
        token,
        use_oob_select=True,
    )
    return token, count


def _fragment_vector_store_count(effect, logical_type, frag, component, component_count):
    if logical_type.kind != "tensor" or logical_type.element_byte_width is None:
        return 1
    if effect.contiguity is None or int(effect.contiguity) <= 1:
        return 1
    count = int(effect.contiguity)
    if int(component) % count:
        return 1
    if int(component) + count > min(int(frag.registers), int(component_count)):
        return 1
    if count * int(logical_type.element_byte_width) > 16:
        return 1
    return count


def _emit_fragment_store_vector_group(
    state,
    store_vector,
    logical_type,
    frag,
    component,
    count,
):
    if int(component) == 0 and int(count) == int(frag.registers):
        return store_vector
    values = []
    for index in range(int(component), int(component) + int(count)):
        values.append(
            _emit_fragment_store_component_value(
                state,
                store_vector,
                logical_type,
                frag,
                index,
            )
        )
    packed_type = state.dsl.simd_type(
        state.dsl.vector_type(int(count), _source_element_type(state.dsl, logical_type)),
        width=int(frag.wave_size),
    )
    return state.dsl.wave.PackOp(packed_type, values).result


def _fragment_store_coord_group_facts(frag, tile_offsets, component, count):
    components = tuple(range(int(component), int(component) + int(count)))
    if int(frag.registers) == 4:
        row_deltas = tuple(comp // 2 for comp in components)
        col_deltas = tuple(comp % 2 for comp in components)
        return (
            _coord_group_fact(tile_offsets[0], 2, row_deltas),
            _coord_group_fact(tile_offsets[1], 2, col_deltas),
        )
    if int(frag.registers) == 16:
        row_deltas = tuple(0 for _ in components)
        col_deltas = tuple((comp % 4) + 8 * (comp // 4) for comp in components)
        return (
            _coord_group_fact(tile_offsets[0], 1, row_deltas),
            _coord_group_fact(tile_offsets[1], 4, col_deltas),
        )
    return tuple(_CoordGroupFact(1, 0, 0) for _ in range(len(tuple(tile_offsets))))


def _coord_group_fact(tile_offset, base_divisibility, deltas):
    deltas = tuple(int(delta) for delta in deltas)
    base_div = gcd(abs(int(tile_offset)), abs(int(base_divisibility)))
    if base_div == 0:
        base_div = abs(int(base_divisibility)) or 1
    return _CoordGroupFact(base_div, min(deltas), max(deltas))


def _mask_uniform_for_coord_group(state, mask_id, coord_facts):
    value = state.result.values.get(mask_id)
    if isinstance(value, _ConstantValue):
        return isinstance(_scalar_constant_literal(value), bool)
    if isinstance(value, _ForwardValue):
        if value.op_name == "ttg.convert_layout":
            return _mask_uniform_for_coord_group(state, value.operand_id, coord_facts)
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
            )
            return _mask_uniform_for_coord_group(
                state,
                value.operand_id,
                coord_facts[:axis] + coord_facts[axis + 1 :],
            )
    if isinstance(value, _UnaryTensorValue) and value.op_name == "tt.broadcast":
        operand_facts = _broadcast_operand_coord_facts(
            _source_type(state, value.operand_id).shape,
            _source_type(state, value.value_id).shape,
            coord_facts,
        )
        return _mask_uniform_for_coord_group(state, value.operand_id, operand_facts)
    if isinstance(value, _MaskAndValue):
        return _mask_uniform_for_coord_group(
            state,
            value.lhs_id,
            coord_facts,
        ) and _mask_uniform_for_coord_group(state, value.rhs_id, coord_facts)
    if isinstance(value, _CompareValue):
        return _compare_uniform_for_coord_group(state, value, coord_facts)
    return False


def _compare_uniform_for_coord_group(state, value, coord_facts):
    lhs = _value_div_delta_for_coord_group(state, value.lhs_id, coord_facts)
    rhs = _value_div_delta_for_coord_group(state, value.rhs_id, coord_facts)
    if lhs is None or rhs is None:
        return False
    if value.predicate in {"slt", "sle", "ult", "ule"}:
        return _ordered_compare_uniform(lhs, rhs)
    if value.predicate in {"sgt", "sge", "ugt", "uge"}:
        return _ordered_compare_uniform(rhs, lhs)
    if value.predicate in {"eq", "ne"}:
        return lhs.delta_min == lhs.delta_max == rhs.delta_min == rhs.delta_max == 0
    return False


def _ordered_compare_uniform(varying, bound):
    if bound.delta_min != 0 or bound.delta_max != 0:
        return False
    if varying.delta_min < 0:
        return False
    span = varying.delta_max - varying.delta_min
    if span == 0:
        return True
    alignment = int(span) + 1
    return _divisibility_covers(
        varying.divisibility,
        alignment,
    ) and _divisibility_covers(bound.divisibility, alignment)


def _value_div_delta_for_coord_group(state, value_id, coord_facts):
    source_type = _source_type(state, value_id)
    value = state.result.values.get(value_id)
    if isinstance(value, _ConstantValue):
        literal = _constant_int_record(value)
        if literal is None:
            return None
        return _CoordGroupFact(abs(int(literal)), 0, 0)
    if isinstance(value, _RangeValue):
        if len(coord_facts) != 1:
            return None
        coord = coord_facts[0]
        return _CoordGroupFact(
            _gcd_divisibility(coord.divisibility, abs(int(value.start))),
            coord.delta_min,
            coord.delta_max,
        )
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            return _value_div_delta_for_coord_group(state, value.operand_id, ())
        if value.op_name == "tt.broadcast":
            operand_facts = _broadcast_operand_coord_facts(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
                coord_facts,
            )
            return _value_div_delta_for_coord_group(
                state,
                value.operand_id,
                operand_facts,
            )
    if isinstance(value, _ForwardValue):
        if value.op_name == "ttg.convert_layout":
            return _value_div_delta_for_coord_group(state, value.operand_id, coord_facts)
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                _source_type(state, value.value_id).shape,
            )
            return _value_div_delta_for_coord_group(
                state,
                value.operand_id,
                coord_facts[:axis] + coord_facts[axis + 1 :],
            )
    if isinstance(value, _BinaryValue):
        result = _binary_div_delta_for_coord_group(state, value, coord_facts)
        if result is not None:
            return result
        if source_type.kind == "scalar":
            return _CoordGroupFact(_source_divisibility(source_type), 0, 0)
        return None
    if isinstance(value, _CastValue):
        return _value_div_delta_for_coord_group(state, value.operand_id, coord_facts)
    if source_type.kind == "scalar":
        return _CoordGroupFact(_source_divisibility(source_type), 0, 0)
    return None


def _binary_div_delta_for_coord_group(state, value, coord_facts):
    lhs = _value_div_delta_for_coord_group(state, value.lhs_id, coord_facts)
    rhs = _value_div_delta_for_coord_group(state, value.rhs_id, coord_facts)
    if lhs is None or rhs is None:
        return None
    if value.kind == "addi":
        return _CoordGroupFact(
            _gcd_divisibility(lhs.divisibility, rhs.divisibility),
            lhs.delta_min + rhs.delta_min,
            lhs.delta_max + rhs.delta_max,
        )
    if value.kind == "subi":
        return _CoordGroupFact(
            _gcd_divisibility(lhs.divisibility, rhs.divisibility),
            lhs.delta_min - rhs.delta_max,
            lhs.delta_max - rhs.delta_min,
        )
    if value.kind == "muli":
        lhs_const = _constant_int_for_value(state, value.lhs_id)
        rhs_const = _constant_int_for_value(state, value.rhs_id)
        if lhs_const is not None:
            return _scale_coord_group_fact(rhs, lhs_const)
        if rhs_const is not None:
            return _scale_coord_group_fact(lhs, rhs_const)
    return None


def _scale_coord_group_fact(fact, scale):
    scale = int(scale)
    lo = fact.delta_min * scale
    hi = fact.delta_max * scale
    return _CoordGroupFact(
        _scale_divisibility(fact.divisibility, scale),
        min(lo, hi),
        max(lo, hi),
    )


def _broadcast_operand_coord_facts(operand_shape, result_shape, coord_facts):
    if len(operand_shape) > len(result_shape):
        return ()
    leading = len(result_shape) - len(operand_shape)
    mapped = []
    for operand_dim, result_dim, fact in zip(
        operand_shape,
        result_shape[leading:],
        coord_facts[leading:],
    ):
        if int(operand_dim) == int(result_dim):
            mapped.append(fact)
        elif int(operand_dim) == 1:
            mapped.append(_CoordGroupFact(0, 0, 0))
        else:
            return ()
    return tuple(mapped)


def _constant_int_for_value(state, value_id):
    value = state.result.values.get(value_id)
    if not isinstance(value, _ConstantValue):
        return None
    return _constant_int_record(value)


def _source_divisibility(source_type):
    if source_type.divisibility is None or int(source_type.divisibility) <= 0:
        return 1
    return int(source_type.divisibility)


def _gcd_divisibility(lhs, rhs):
    lhs = abs(int(lhs))
    rhs = abs(int(rhs))
    if lhs == 0:
        return rhs
    if rhs == 0:
        return lhs
    return gcd(lhs, rhs)


def _scale_divisibility(divisibility, scale):
    divisibility = abs(int(divisibility))
    scale = abs(int(scale))
    if divisibility == 0 or scale == 0:
        return 0
    return divisibility * scale


def _divisibility_covers(divisibility, alignment):
    alignment = abs(int(alignment))
    if alignment <= 1:
        return True
    divisibility = abs(int(divisibility))
    return divisibility == 0 or divisibility % alignment == 0


def _emit_buffer_store_component(
    state: _StructuralState,
    effect: _BufferStoreEffect,
    coords,
    value,
    token,
    *,
    use_oob_select=False,
):
    if effect.mask_id is None:
        pointer = _materialize_buffer_store_pointer(
            state,
            effect,
            coords,
            value.type,
        )
        return state.builder.store(value, pointer, after=token)
    mask = _materialize_coordinate_value(state, effect.mask_id, coords)
    if use_oob_select:
        pointer = _materialize_masked_buffer_store_pointer(
            state,
            effect,
            coords,
            value.type,
            mask,
        )
        return state.builder.store(value, pointer, after=token)
    buffer = _buffer_pointer_for_base(state, effect.base_id)
    mem_token_type = state.dsl.mem_token_type()
    with state.builder.where(mask, [mem_token_type]) as where:
        pointer = _materialize_buffer_store_pointer(
            state,
            effect,
            coords,
            value.type,
            buffer=buffer,
        )
        inner_token = state.builder.store(value, pointer, after=token)
        state.builder.yield_([inner_token])
    return where.results_[0]


def _materialize_masked_buffer_store_pointer(
    state: _StructuralState,
    effect: _BufferStoreEffect,
    coords,
    value_type,
    mask,
):
    buffer = _buffer_pointer_for_base(state, effect.base_id)
    width = _coordinate_width(state, coords)
    offset = _materialize_coordinate_value(state, effect.offset_id, coords)
    offset = _ensure_coordinate_simd_i32(state, offset, width)
    access_byte_width = _store_access_byte_width(state, value_type)
    upper = _buffer_offset_upper_bound(
        _source_type(state, effect.base_id),
        access_byte_width,
    )
    if upper is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot prove masked amdg.buffer_store "
            f"offset bounds for base type {_source_type(state, effect.base_id).raw}"
        )
    offset = state.builder.assume_range(offset, 0, upper)
    source_type = _source_type(state, effect.base_id)
    pointer_type = _parse_type(
        state.ir,
        f"!wave.simd<!wave.ptr<#waveamd.buffer, {source_type.pointee_type}>, {width}>",
    )
    true_pointer = state.builder.ptr_add(buffer, offset, pointer_type)
    oob_const = state.builder.constant(state.dsl.i32(), int(upper) + 1)
    oob_offset = state.builder.splat(oob_const, state.dsl.i32(), width)
    oob_pointer = state.builder.ptr_add(buffer, oob_offset, pointer_type)
    return state.builder.select(mask, true_pointer, oob_pointer)


def _materialize_buffer_store_pointer(
    state: _StructuralState,
    effect: _BufferStoreEffect,
    coords,
    value_type,
    *,
    buffer=None,
):
    if buffer is None:
        buffer = _buffer_pointer_for_base(state, effect.base_id)
    width = _coordinate_width(state, coords)
    offset = _materialize_coordinate_value(state, effect.offset_id, coords)
    offset = _ensure_coordinate_simd_i32(state, offset, width)
    upper = _buffer_offset_upper_bound(
        _source_type(state, effect.base_id),
        _store_access_byte_width(state, value_type),
    )
    if upper is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot prove amdg.buffer_store "
            f"offset bounds for base type {_source_type(state, effect.base_id).raw}"
        )
    offset = state.builder.assume_range(offset, 0, upper)
    source_type = _source_type(state, effect.base_id)
    pointer_type = _parse_type(
        state.ir,
        f"!wave.simd<!wave.ptr<#waveamd.buffer, {source_type.pointee_type}>, {width}>",
    )
    return state.builder.ptr_add(buffer, offset, pointer_type)


def _store_access_byte_width(state: _StructuralState, value_type):
    element_type = _simd_element_type(state.dsl, value_type)
    byte_width = _scalar_type_byte_width(str(element_type))
    if byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot infer store access width for "
            f"{value_type}"
        )
    return byte_width


def _scalar_type_byte_width(element_type: str):
    if element_type.startswith("vector<") and element_type.endswith(">"):
        inner = element_type.removeprefix("vector<").removesuffix(">")
        count_text, scalar = inner.split("x", 1)
        scalar_width = _scalar_type_byte_width(scalar.strip())
        if scalar_width is None:
            return None
        return int(count_text.strip()) * scalar_width
    return {
        "i1": 1,
        "i8": 1,
        "i16": 2,
        "f16": 2,
        "bf16": 2,
        "i32": 4,
        "f32": 4,
        "index": 4,
        "i64": 8,
        "f64": 8,
    }.get(element_type)


def _emit_assume(state: _StructuralState, effect: _AssumeEffect):
    facts = _assume_fact_records(effect.predicate, _bounded_scalar_value_ids(state))
    for value_id, kind, constant in facts:
        expr = _assume_expr(
            state.dsl,
            kind,
            constant,
            canonicalize_strict=_is_signed_integer_kernel_arg(state, value_id),
        )
        if expr is None:
            continue
        _record_scalar_fact(state, value_id, kind, constant)
        if value_id not in state.values and value_id in state.assume_condition_values:
            continue
        value = _require_structural_value(state, value_id)
        if isinstance(value, tuple):
            continue
        if _is_simd_type(state.dsl, value.type) or str(value.type).startswith("!wave.mask<"):
            continue
        value = _ensure_signed_bound_assume(state, value_id, value)
        state.values[value_id] = state.builder.assume(value, [expr], name="x")


def _emit_masked_load_component(state, pointer, mask, other, result_type, token):
    mem_token_type = state.dsl.mem_token_type()
    with state.builder.where(mask, [result_type, mem_token_type]) as where:
        loaded, inner_token = state.builder.load(pointer, result_type, after=token)
        state.builder.yield_([loaded, inner_token])
        with where.otherwise():
            state.builder.yield_([other, token])
    return tuple(where.results_)


def _emit_masked_store_component(state, value, pointer, mask, token):
    mem_token_type = state.dsl.mem_token_type()
    with state.builder.where(mask, [mem_token_type]) as where:
        inner_token = state.builder.store(value, pointer, after=token)
        state.builder.yield_([inner_token])
    return where.results_[0]


def _materialize_pointer_coordinate(state, value_id, coords, value_type, *, as_base=False):
    value = state.result.values.get(value_id)
    width = _coordinate_width(state, coords)
    source_type = _source_type(state, value_id)
    if isinstance(value, _ConvertedInputValue):
        pointer = _require_structural_value(state, value_id)
        if as_base:
            return pointer
        if _is_simd_type(state.dsl, pointer.type):
            return pointer
        return state.builder.splat(pointer, pointer.type, width)
    if isinstance(value, _UnaryTensorValue):
        if value.op_name == "tt.splat":
            pointer = _require_structural_value(state, value.operand_id)
            if as_base:
                return pointer
            if _is_simd_type(state.dsl, pointer.type):
                return pointer
            return state.builder.splat(pointer, pointer.type, width)
        if value.op_name == "tt.broadcast":
            operand_coords = _broadcast_operand_coords(
                state,
                _source_type(state, value.operand_id).shape,
                source_type.shape,
                coords,
            )
            return _materialize_pointer_coordinate(
                state,
                value.operand_id,
                operand_coords,
                value_type,
                as_base=as_base,
            )
    if isinstance(value, _ForwardValue):
        if value.op_name == "tt.expand_dims":
            axis = _expand_dims_axis(
                _source_type(state, value.operand_id).shape,
                source_type.shape,
            )
            return _materialize_pointer_coordinate(
                state,
                value.operand_id,
                coords[:axis] + coords[axis + 1 :],
                value_type,
                as_base=as_base,
            )
        if value.op_name == "ttg.convert_layout":
            return _materialize_pointer_coordinate(
                state,
                value.operand_id,
                coords,
                value_type,
                as_base=as_base,
            )
    if isinstance(value, _PointerAddValue):
        base = _materialize_pointer_coordinate(
            state,
            value.base_id,
            coords,
            value_type,
            as_base=True,
        )
        offset = _materialize_coordinate_value(state, value.offset_id, coords)
        offset = _ensure_coordinate_simd_i32(state, offset, width)
        return state.builder.ptr_add(base, offset, _simd_pointer_type_for_value(state, value_type, width))
    if value_id in state.values:
        components = _components(state.values[value_id])
        if len(components) == 1:
            pointer = components[0]
            if _is_simd_type(state.dsl, pointer.type):
                return pointer
            return state.builder.splat(pointer, pointer.type, width)
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural emitter cannot materialize pointer coordinate for "
        f"value {value_id} ({type(value).__name__})"
    )


def _simd_pointer_type_for_value(state, value_type, width):
    value_element = _simd_element_type(state.dsl, value_type)
    return state.dsl.simd_type(
        state.dsl.ptr_type(value_element, state.dsl.global_address_space()),
        width=int(width),
    )


def _materialize_local_store_pointer(state, memdesc_id, memdesc_type, coords, value_type):
    width = _coordinate_width(state, coords)
    element_type = _simd_element_type(state.dsl, value_type)
    base_type = state.dsl.ptr_type(element_type, state.dsl.shared_address_space())
    base = _ptr_cast(state, _require_structural_value(state, memdesc_id), base_type)
    offset = _materialize_memdesc_element_offset(state, memdesc_type, coords)
    pointer_type = state.dsl.simd_type(base_type, width=int(width))
    return state.builder.ptr_add(base, offset, pointer_type)


def _materialize_memdesc_element_offset(state, memdesc_type, coords):
    if memdesc_type.element_byte_width is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot address memdesc with unknown "
            "element width"
        )
    if _is_identity_shared_memdesc(memdesc_type):
        return _materialize_row_major_element_offset(state, memdesc_type.shape, coords)
    swizzled = _swizzled_shared_memdesc_info(memdesc_type)
    if swizzled is not None:
        return _materialize_swizzled_element_offset(state, memdesc_type, swizzled, coords)
    padded = _padded_shared_memdesc_info(memdesc_type)
    if padded is not None:
        logical = _materialize_row_major_element_offset(state, memdesc_type.shape, coords)
        encoded = logical
        for interval, padding in zip(padded["intervals"], padded["paddings"]):
            quotient = _emit_simd_binary_const(state, "divui", logical, int(interval))
            pad = _emit_simd_binary_const(state, "muli", quotient, int(padding))
            encoded = state.builder.addi(encoded, pad)
        return encoded
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural emitter cannot address unsupported shared layout "
        f"{memdesc_type.raw}"
    )


def _materialize_row_major_element_offset(state, shape, coords):
    if len(shape) != len(coords):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot materialize row-major offset "
            f"for shape {shape} with {len(coords)} coordinate(s)"
        )
    width = _coordinate_width(state, coords)
    offset = None
    stride = 1
    for dim in reversed(range(len(shape))):
        coord = _ensure_coordinate_simd_i32(state, coords[dim], width)
        term = coord if stride == 1 else _emit_simd_binary_const(state, "muli", coord, stride)
        offset = term if offset is None else state.builder.addi(term, offset)
        stride *= int(shape[dim])
    if offset is None:
        const = state.builder.constant(state.dsl.i32(), 0)
        return state.builder.splat(const, state.dsl.i32(), width)
    return offset


def _materialize_swizzled_element_offset(state, memdesc_type, swizzled, coords):
    if len(memdesc_type.shape) != 2 or len(coords) != 2:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports swizzled local_store "
            "pointers only for rank-2 memdescs"
        )
    width = _coordinate_width(state, coords)
    row = _ensure_coordinate_simd_i32(state, coords[-2], width)
    col = _ensure_coordinate_simd_i32(state, coords[-1], width)
    cols = int(memdesc_type.shape[-1])
    row_phase = _emit_simd_binary_const(
        state,
        "divui",
        row,
        int(swizzled["per_phase"]),
    )
    phase = _emit_simd_binary_const(
        state,
        "remui",
        row_phase,
        int(swizzled["max_phase"]),
    )
    col_group = _emit_simd_binary_const(state, "divui", col, int(swizzled["vec"]))
    swizzled_group = state.builder.binary(
        _binary_kind(state.dsl, "xori"),
        col_group,
        phase,
    )
    swizzled_base = _emit_simd_binary_const(
        state,
        "muli",
        swizzled_group,
        int(swizzled["vec"]),
    )
    col_in_vec = _emit_simd_binary_const(state, "remui", col, int(swizzled["vec"]))
    swizzled_col = state.builder.addi(swizzled_base, col_in_vec)
    row_scaled = _emit_simd_binary_const(state, "muli", row, cols)
    return state.builder.addi(row_scaled, swizzled_col)


def _local_linear_pointers(
    state: _StructuralState,
    memdesc_id: int,
    tensor_value_id: int,
    value_type,
    component_count: int,
):
    _validate_linear_tensor_components(
        state, tensor_value_id, value_type, component_count
    )
    base = _require_structural_value(state, memdesc_id)
    memdesc_type = _source_type(state, memdesc_id)
    value_element = str(_simd_element_type(state.dsl, value_type))
    if memdesc_type.element_type != value_element:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter cannot map local memory with mismatched "
            f"element types: {memdesc_type.element_type} vs {value_element}"
        )
    width = _simd_width(state.dsl, value_type)
    offset_type = _parse_type(state.ir, f"!wave.simd<i32, {width}>")
    ptr_type = _parse_type(
        state.ir,
        f"!wave.simd<!wave.ptr<#wave.shared, {value_element}>, {width}>",
    )
    lane = state.builder.workitem_id(0, state.dsl.i32(), width)
    mapping = _rank1_component_lane_mapping(
        state,
        _source_type(state, tensor_value_id),
        component_count,
        width,
    )
    pointers = []
    for component in range(int(component_count)):
        lane_scale, component_offset = mapping[component]
        offset = lane
        if lane_scale != 1:
            scale = state.builder.constant(state.dsl.i32(), lane_scale)
            scale = state.builder.splat(scale, state.dsl.i32(), width)
            offset = state.builder.muli(offset, scale)
        if component_offset:
            shift = state.builder.constant(state.dsl.i32(), component_offset)
            shift = state.builder.splat(shift, state.dsl.i32(), width)
            offset = state.builder.addi(offset, shift)
        if offset.type != offset_type:
            raise _UnsupportedStructuralEmission(
                "tlx_wave structural emitter produced unexpected local offset type "
                f"{offset.type}, expected {offset_type}"
            )
        pointers.append(state.builder.ptr_add(base, offset, ptr_type))
    return tuple(pointers)


def _component_active_mask(state: _StructuralState, value_id: int, component: int, value_type):
    source_type = _source_type(state, value_id)
    if source_type.kind != "tensor" or len(tuple(source_type.shape)) != 1:
        return None
    component_count = int(state.result.values[value_id].converted_type.component_count)
    width = _simd_width(state.dsl, value_type)
    mapping = _rank1_component_lane_mapping(state, source_type, component_count, width)
    lane_scale, component_offset = mapping[int(component)]
    element_count = _product(source_type.shape)
    if lane_scale * width + component_offset <= element_count:
        return None
    lane = state.builder.workitem_id(0, state.dsl.i32(), width)
    index = lane
    if lane_scale != 1:
        scale = state.builder.constant(state.dsl.i32(), lane_scale)
        scale = state.builder.splat(scale, state.dsl.i32(), width)
        index = state.builder.muli(index, scale)
    if component_offset:
        offset = state.builder.constant(state.dsl.i32(), component_offset)
        offset = state.builder.splat(offset, state.dsl.i32(), width)
        index = state.builder.addi(index, offset)
    bound = state.builder.constant(state.dsl.i32(), element_count)
    bound = state.builder.splat(bound, state.dsl.i32(), width)
    return state.builder.cmpi("ult", index, bound)


def _rank1_component_lane_mapping(
    state: _StructuralState,
    source_type,
    component_count: int,
    lane_width: int,
):
    if len(tuple(source_type.shape)) == 1:
        attr = source_type.encoding_attr
        if attr is not None and _attr_bool(attr, "is_blocked_encoding"):
            mapping = _rank1_blocked_component_lane_mapping(
                attr,
                component_count,
                lane_width,
            )
            if mapping is not None:
                return mapping
    return tuple((1, component * int(lane_width)) for component in range(component_count))


def _rank1_blocked_component_lane_mapping(attr, component_count, lane_width):
    size_per_thread = tuple(
        int(value) for value in _attr_value(attr, "get_blocked_size_per_thread")
    )
    if len(size_per_thread) != 1:
        return None
    elements_per_thread = int(size_per_thread[0])
    if elements_per_thread <= 0:
        return None
    if int(component_count) == elements_per_thread:
        return tuple(
            (elements_per_thread, component) for component in range(component_count)
        )
    if elements_per_thread == 1:
        return tuple((1, component * int(lane_width)) for component in range(component_count))
    return None


def _combine_optional_masks(state: _StructuralState, lhs, rhs):
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    false_mask = _wave_mask_constant(state, lhs.type, False)
    return state.builder.select(lhs, rhs, false_mask)


def _validate_linear_tensor_components(
    state: _StructuralState,
    value_id: int,
    value_type,
    component_count: int,
):
    source_type = _source_type(state, value_id)
    if source_type.kind != "tensor":
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter expected tensor value for flat local "
            f"memory access, got {source_type.raw}"
        )
    element_count = _product(source_type.shape)
    width = _simd_width(state.dsl, value_type)
    if element_count > width * int(component_count):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter supports local load/store only for "
            "whole-component flat tensors; got "
            f"{source_type.raw} with {element_count} elements for {value_type}"
        )


def _shared_pointer_type_for_source(state: _StructuralState, value_id: int):
    source_type = _source_type(state, value_id)
    element_type = source_type.element_type or "i8"
    return _parse_type(state.ir, f"!wave.ptr<#wave.shared, {element_type}>")


def _memdesc_index_elements_per_slot(state: _StructuralState, value):
    result_type = _source_type(state, value.value_id)
    return _product(result_type.alloc_shape or result_type.shape or (1,))


def _memory_token_operand(state: _StructuralState, token_id):
    if token_id is None:
        return _ensure_mem_token(state)
    return _require_structural_value(state, token_id)


def _can_emit_structural_async_copy(value: _AsyncCopyValue):
    if _can_emit_structural_buffer_dma_async_copy(value):
        return True
    if _can_emit_structural_dma_async_copy(value):
        return True
    return (
        value.op_name == "ttg.async_copy_global_to_local"
        and value.source_address_value_id is not None
        and value.source_offset_value_id is None
        and value.memdesc_value_id is not None
        and value.other_value_id is not None
    )


def _can_emit_structural_dma_async_copy(value: _AsyncCopyValue):
    if (
        value.op_name != "ttg.async_copy_global_to_local"
        or value.source_address_value_id is None
        or value.source_offset_value_id is not None
        or value.memdesc_value_id is None
        or value.mask_value_id is not None
        or value.other_value_id is not None
    ):
        return False
    if not isinstance(value.source_address, _PointerAddValue):
        return False
    if not isinstance(value.source_address.base, _UnaryTensorValue):
        return False
    if value.source_address.base.op_name != "tt.splat":
        return False
    if not isinstance(value.source_address.offset, _RangeValue):
        return False
    return isinstance(value.memdesc, _LocalAllocValue)


def _can_emit_structural_buffer_dma_async_copy(value: _AsyncCopyValue):
    if not _is_structural_buffer_dma_candidate(value):
        return False
    return _can_materialize_coordinate_value(value.source_offset) and (
        value.mask_value_id is None or _can_materialize_coordinate_value(value.mask)
    )


def _structural_dma_packet_bytes(source_type):
    if source_type.element_byte_width == 2 and source_type.pointee_type in {"f16", "bf16"}:
        return 16
    if source_type.element_byte_width in {1, 2, 4}:
        return 4
    raise _UnsupportedStructuralEmission(
        "tlx_wave structural DMA does not support source element type "
        f"{source_type.pointee_type}"
    )


def _simple_dma_scalar_base_id(state: _StructuralState, source_value_id):
    value = state.result.values.get(source_value_id)
    if not isinstance(value, _PointerAddValue):
        return None
    if not isinstance(value.offset, _RangeValue):
        return None
    if value.offset.start != 0 or value.offset.end != _product(_source_type(state, source_value_id).shape):
        return None
    base = value.base
    if not isinstance(base, _UnaryTensorValue) or base.op_name != "tt.splat":
        return None
    base_type = _source_type(state, base.operand_id)
    if base_type.kind != "pointer" or base_type.address_space == 3:
        return None
    return base.operand_id


def _ptr_cast(state: _StructuralState, value, result_type):
    if value.type == result_type:
        return value
    return state.dsl.wave.PtrCastOp(result_type, value).result


def _buffer_pointer_for_base(state: _StructuralState, base_id: int):
    cached = state.buffer_cache.get(base_id)
    if cached is not None:
        return cached
    source_type = _source_type(state, base_id)
    if source_type.kind != "pointer" or source_type.pointee_type is None:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer memory expected typed base pointer"
        )
    pointer_range = source_type.pointer_range
    if pointer_range is None or int(pointer_range) <= 0 or int(pointer_range) > 32:
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer memory expected tt.pointer_range in 1..32"
        )
    base = _require_structural_value(state, base_id)
    if isinstance(base, tuple):
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural buffer memory expected uniform base pointer"
        )
    range_bytes = (1 << (int(pointer_range) - 1)) - 1
    range_value = state.builder.constant(state.dsl.i32(), range_bytes)
    buffer_type = _parse_type(
        state.ir, f"!wave.ptr<#waveamd.buffer, {source_type.pointee_type}>"
    )
    buffer = state.builder.make_buffer(base, range_value, buffer_type)
    state.buffer_cache[base_id] = buffer
    return buffer


def _buffer_offset_upper_bound(source_type, access_byte_width: int):
    pointer_range = source_type.pointer_range
    element_byte_width = source_type.element_byte_width
    if (
        pointer_range is None
        or int(pointer_range) <= 0
        or int(pointer_range) > 32
        or element_byte_width is None
        or int(element_byte_width) <= 0
        or int(access_byte_width) <= 0
    ):
        return None
    byte_upper = (1 << (int(pointer_range) - 1)) - 1
    byte_upper -= int(access_byte_width)
    if byte_upper < 0:
        return None
    return byte_upper // int(element_byte_width)


def _require_structural_subset(run: _ConversionRun):
    if not _can_emit_structural_wave_module(run):
        unsupported_values = sorted(
            {
                type(value).__name__
                for value in run.result.values.values()
                if not isinstance(
                    value,
                    (
                        _ConvertedInputValue,
                        _ConstantValue,
                        _RangeValue,
                        _ProgramIdValue,
                        _UnaryTensorValue,
                        _BinaryValue,
                        _CompareValue,
                        _MaskAndValue,
                        _MinValue,
                        _PointerAddValue,
                        _ForwardValue,
                        _CastValue,
                        _LoadValue,
                        _LocalAllocValue,
                        _MemdescIndexValue,
                        _LocalLoadValue,
                        _AsyncCopyValue,
                        _AsyncCommitGroupValue,
                        _AsyncWaitValue,
                        _DotValue,
                        _IfValue,
                        _ForValue,
                    ),
                )
            }
        )
        unsupported_effects = sorted(
            {
                type(effect).__name__
                for effect in run.result.effects
                if not isinstance(
                    effect,
                    (
                        _ReturnEffect,
                        _StoreEffect,
                        _AssumeEffect,
                        _AsyncWaitEffect,
                        _BufferStoreEffect,
                        _LocalStoreEffect,
                        _IfEffect,
                    ),
                )
            }
        )
        detail = ", ".join(unsupported_values + unsupported_effects)
        raise _UnsupportedStructuralEmission(
            "tlx_wave structural emitter does not support this conversion run"
            + (f": {detail}" if detail else "")
        )


def _is_type_preserving_forward(value: _ForwardValue):
    operand_type = getattr(getattr(value.operand, "converted_type", None), "wave_type", None)
    operand_components = int(
        getattr(getattr(value.operand, "converted_type", None), "component_count", 1)
    )
    result_type = value.converted_type.wave_type
    result_components = int(value.converted_type.component_count)
    return operand_type == result_type and operand_components == result_components


def _is_structural_dot_operand_convert(value, result):
    if not isinstance(value, _ForwardValue) or value.op_name != "ttg.convert_layout":
        return False
    if getattr(value.converted_type, "kind", None) != "fragment":
        return False
    if result is None:
        return isinstance(value.operand, _LocalLoadValue)
    return isinstance(result.values.get(value.operand_id), _LocalLoadValue)


def _can_emit_structural_cast(value: _CastValue):
    operand_type = getattr(value.operand, "converted_type", None)
    if value.kind != "fpconvert":
        return False
    if getattr(operand_type, "kind", None) == "fragment":
        return value.converted_type.kind == "fragment"
    return True


def _assume_condition_value_ids(program, result):
    assume_effects = tuple(
        effect for effect in result.effects if isinstance(effect, _AssumeEffect)
    )
    if not assume_effects:
        return frozenset()
    candidates = set()
    for effect in assume_effects:
        _collect_assume_condition_dependencies(
            result.values,
            effect.predicate_id,
            candidates,
            set(),
        )
    assume_effect_ids = {id(effect) for effect in assume_effects}
    consumers = _value_consumers(result)
    source_consumers = _source_value_consumers(program)
    protected_fact_targets = _shared_condition_fact_targets(
        program,
        assume_effects,
        source_consumers,
    )
    skipped = set()
    changed = True
    while changed:
        changed = False
        for value_id in candidates:
            if value_id in skipped:
                continue
            if value_id in protected_fact_targets:
                continue
            if not _source_consumers_are_assume_only(
                program,
                source_consumers.get(value_id, ()),
                candidates,
            ):
                continue
            value_consumers = consumers.get(value_id, ())
            if value_consumers and all(
                (kind == "effect" and consumer in assume_effect_ids)
                or (kind == "value" and consumer in skipped)
                for kind, consumer in value_consumers
            ):
                skipped.add(value_id)
                changed = True
    return frozenset(skipped)


def _source_value_consumers(program):
    consumers = {}
    for op in program.ops:
        for operand_id in op.operands:
            consumers.setdefault(operand_id, set()).add(op.index)
    return consumers


def _shared_condition_fact_targets(program, assume_effects, source_consumers):
    protected = set()
    for effect in assume_effects:
        if not any(
            program.ops[op_index].name == "scf.if" and not program.ops[op_index].results
            for op_index in source_consumers.get(effect.predicate_id, ())
        ):
            continue
        protected.update(
            value_id
            for value_id, _kind, _constant in _assume_fact_records(effect.predicate)
        )
    return protected


def _source_consumers_are_assume_only(program, op_indices, candidates):
    for op_index in op_indices:
        op = program.ops[op_index]
        if op.name == "llvm.intr.assume":
            continue
        if op.name == "scf.if" and not op.results:
            continue
        if op.results and all(result_id in candidates for result_id in op.results):
            continue
        return False
    return True


def _collect_assume_condition_dependencies(converted_values, value_id, out, seen):
    if value_id in seen:
        return
    seen.add(value_id)
    out.add(value_id)
    value = converted_values.get(value_id)
    if value is None:
        return
    for operand_id in _record_operand_ids(value):
        _collect_assume_condition_dependencies(converted_values, operand_id, out, seen)


def _value_consumers(result):
    consumers = {}
    _add_value_consumers(consumers, result.values)
    _add_effect_consumers(consumers, result.effects)
    return consumers


def _add_value_consumers(consumers, values):
    for result_id, value in values.items():
        for operand_id in _record_operand_ids(value):
            consumers.setdefault(operand_id, set()).add(("value", result_id))
        if isinstance(value, _IfValue):
            _add_value_consumers(consumers, value.then_values)
            _add_value_consumers(consumers, value.else_values)
        elif isinstance(value, _ForValue):
            _add_value_consumers(consumers, value.body_values)
            _add_effect_consumers(consumers, value.body_effects)


def _add_effect_consumers(consumers, effects):
    for effect in effects:
        for operand_id in _effect_operand_ids(effect):
            consumers.setdefault(operand_id, set()).add(("effect", id(effect)))
        if isinstance(effect, _IfEffect):
            _add_value_consumers(consumers, effect.then_values)
            _add_value_consumers(consumers, effect.else_values)
            _add_effect_consumers(consumers, effect.then_effects)
            _add_effect_consumers(consumers, effect.else_effects)


def _record_operand_ids(value):
    if isinstance(value, (_UnaryTensorValue, _ForwardValue, _CastValue)):
        return (value.operand_id,)
    if isinstance(value, (_BinaryValue, _CompareValue, _MaskAndValue, _MinValue)):
        return (value.lhs_id, value.rhs_id)
    if isinstance(value, _PointerAddValue):
        return (value.base_id, value.offset_id)
    if isinstance(value, _IfValue):
        return (value.condition_id,) + value.then_yield_ids + value.else_yield_ids
    if isinstance(value, _ForValue):
        return (
            value.lower_id,
            value.upper_id,
            value.step_id,
        ) + value.iter_arg_ids + value.body_yield_ids
    if isinstance(value, _LoadValue):
        ids = [value.pointer_id]
        if value.mask_id is not None:
            ids.append(value.mask_id)
        if value.other_id is not None:
            ids.append(value.other_id)
        return tuple(ids)
    if isinstance(value, _MemdescIndexValue):
        return (value.memdesc_id, value.index_id)
    if isinstance(value, _LocalLoadValue):
        return (value.memdesc_id,) if value.token_id is None else (
            value.memdesc_id,
            value.token_id,
        )
    if isinstance(value, _AsyncCopyValue):
        ids = []
        for value_id in (
            value.source_address_value_id,
            value.source_offset_value_id,
            value.memdesc_value_id,
            value.mask_value_id,
            value.other_value_id,
        ):
            if value_id is not None:
                ids.append(value_id)
        return tuple(ids)
    if isinstance(value, _AsyncCommitGroupValue):
        return value.member_token_ids
    if isinstance(value, _AsyncWaitValue):
        return value.input_token_ids
    return ()


def _effect_operand_ids(effect):
    if isinstance(effect, _StoreEffect):
        ids = [effect.pointer_id, effect.value_id]
        if effect.mask_id is not None:
            ids.append(effect.mask_id)
        return tuple(ids)
    if isinstance(effect, _BufferStoreEffect):
        ids = [effect.value_id, effect.base_id, effect.offset_id]
        if effect.mask_id is not None:
            ids.append(effect.mask_id)
        return tuple(ids)
    if isinstance(effect, _LocalStoreEffect):
        return (effect.value_id, effect.memdesc_id) if effect.token_id is None else (
            effect.value_id,
            effect.memdesc_id,
            effect.token_id,
        )
    if isinstance(effect, _AsyncWaitEffect):
        return effect.input_token_ids
    if isinstance(effect, _AssumeEffect):
        return (effect.predicate_id,)
    if isinstance(effect, _IfEffect):
        return (effect.condition_id,)
    if isinstance(effect, _ReturnEffect):
        return effect.operand_ids
    return ()


def _load_wave_dsl():
    third_party = Path(__file__).resolve().parents[2]
    wave_python = third_party / "wave" / "build" / "wave-build" / "python_packages" / "wave_mlir"
    if not wave_python.exists():
        raise RuntimeError(
            "tlx_wave structural emitter requires the built Wave MLIR Python package "
            f"at {wave_python}"
        )
    path = str(wave_python)
    if path not in sys.path:
        sys.path.insert(0, path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Attribute builder for .* is already registered",
            category=RuntimeWarning,
        )
        from mlir import ir
        from mlir.dialects import wave_dsl as dsl
    return dsl, ir


def _parse_type(ir, type_text: str):
    return ir.Type.parse(type_text)


def _bind_kernel_arg(state: _StructuralState, value_id: int, arg_value):
    bounds = _signed_int_bounds(state.result.values[value_id].converted_type.wave_type)
    if bounds is None:
        return arg_value
    lower, upper = bounds
    return state.builder.assume_range(arg_value, lower, upper)


def _signed_int_bounds(wave_type):
    if wave_type == "i8":
        return -(1 << 7), (1 << 7) - 1
    if wave_type == "i16":
        return -(1 << 15), (1 << 15) - 1
    if wave_type == "i32":
        return -(1 << 31), (1 << 31) - 1
    return None


def _source_element_type(dsl, source_type):
    element = source_type.element_type or source_type.raw
    return {
        "i1": dsl.i1,
        "i8": dsl.i8,
        "i16": dsl.i16,
        "i32": dsl.i32,
        "i64": dsl.i64,
        "index": dsl.index_type,
        "f16": dsl.f16,
        "bf16": dsl.bf16,
        "f32": dsl.f32,
    }[element]()


def _simd_width(dsl, type_):
    return int(dsl.SimdType(type_).width)


def _simd_element_type(dsl, type_):
    return dsl.SimdType(type_).element_type


def _is_simd_type(dsl, type_):
    return bool(dsl.SimdType.isinstance(type_))


def _region_result_for_values(values, effects=()):
    return type(
        "_StructuralRegionResult",
        (),
        {
            "values": values,
            "effects": effects,
        },
    )()


def _child_state(state: _StructuralState, result=None):
    child = _StructuralState(
        state.dsl,
        state.ir,
        state.builder,
        state.program,
        state.result if result is None else result,
        state.lds_offsets,
        state.assume_condition_values,
        state.deferred_coordinate_values,
    )
    child.values = dict(state.values)
    child.mem_token = state.mem_token
    child.nonnegative_values = set(state.nonnegative_values)
    child.positive_values = set(state.positive_values)
    child.bounded_kernel_args = set(state.bounded_kernel_args)
    child.lower_bounds = dict(state.lower_bounds)
    child.buffer_cache = state.buffer_cache
    return child


def _require_structural_value(state: _StructuralState, value_id):
    if value_id not in state.values:
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter could not materialize value {value_id}"
        )
    return state.values[value_id]


def _wave_mask_constant(state: _StructuralState, result_type, value: bool):
    attr = state.ir.Attribute.parse("true" if value else "false")
    return state.dsl.wave.ConstantOp(result_type, attr).result


def _zero_value(state: _StructuralState, result_type):
    if not _is_simd_type(state.dsl, result_type):
        return state.builder.constant(result_type, _zero_literal(result_type))
    element_type = _simd_element_type(state.dsl, result_type)
    width = _simd_width(state.dsl, result_type)
    zero = state.builder.constant(element_type, _zero_literal(element_type))
    return state.builder.splat(zero, element_type, width)


def _zero_literal(type_):
    if str(type_) in {"f16", "bf16", "f32", "f64"}:
        return 0.0
    return 0


def _ensure_value_type(state: _StructuralState, value, result_type, context: str):
    if value.type == result_type:
        return value
    if not _is_simd_type(state.dsl, value.type) and _is_simd_type(
        state.dsl, result_type
    ):
        return state.builder.splat(
            value,
            value.type,
            _simd_width(state.dsl, result_type),
        )
    raise _UnsupportedStructuralEmission(
        f"tlx_wave structural emitter cannot adapt {context} type "
        f"{value.type} -> {result_type}"
    )


def _index_bound(state: _StructuralState, value_id):
    constant = _constant_int_for_value(state, value_id)
    if constant is not None:
        return state.builder.constant(state.dsl.index_type(), int(constant))
    value = _require_structural_value(state, value_id)
    if str(value.type) == "index":
        return value
    if not _is_integer_or_index_type(value.type):
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter cannot use {value.type} as scf.for bound"
        )
    x = state.dsl.sym("x")
    return state.builder.index_expr(
        x,
        {x: value},
        result_type=state.dsl.index_type(),
    )


def _ensure_mem_token(state: _StructuralState):
    if state.mem_token is None:
        state.mem_token = state.builder.token()
    return state.mem_token


def _emit_cmpi(state: _StructuralState, predicate: str, lhs, rhs):
    if _is_simd_type(state.dsl, lhs.type):
        return state.builder.cmpi(predicate, lhs, rhs)
    return state.dsl.arith.CmpIOp(
        state.dsl.CmpIPredicate[predicate],
        lhs,
        rhs,
    ).result


def _component_tuple_for_count(value, count: int, context: str):
    components = _components(value)
    if len(components) == int(count):
        return components
    if len(components) == 1:
        return components * int(count)
    raise _UnsupportedStructuralEmission(
        f"tlx_wave structural emitter cannot remap {context} components: "
        f"got {len(components)}, expected {int(count)}"
    )


def _binary_kind(dsl, kind: str):
    return {
        "addi": dsl.BinaryKind.AddI,
        "subi": dsl.BinaryKind.SubI,
        "muli": dsl.BinaryKind.MulI,
        "shli": dsl.BinaryKind.ShLI,
        "shrui": dsl.BinaryKind.ShRUI,
        "shrsi": dsl.BinaryKind.ShRSI,
        "andi": dsl.BinaryKind.AndI,
        "ori": dsl.BinaryKind.OrI,
        "xori": dsl.BinaryKind.XOrI,
        "divui": dsl.BinaryKind.DivUI,
        "divsi": dsl.BinaryKind.DivSI,
        "remui": dsl.BinaryKind.RemUI,
        "remsi": dsl.BinaryKind.RemSI,
    }[kind]


def _effective_binary_kind(value: _BinaryValue, state: _StructuralState):
    if value.kind not in {"divsi", "remsi"}:
        return value.kind
    if value.converted_type.kind != "scalar":
        return value.kind
    lhs_lower = _scalar_lower_bound(state, value.lhs_id)
    if lhs_lower is None or lhs_lower < 0:
        return value.kind
    if value.rhs_id not in state.positive_values:
        return value.kind
    rhs_const = _constant_int_record(value.rhs)
    if rhs_const is not None and _is_positive_power_of_two(rhs_const):
        return value.kind
    return {"divsi": "divui", "remsi": "remui"}[value.kind]


def _try_emit_pow2_signed_div_rem(state: _StructuralState, value: _BinaryValue, lhs, rhs):
    if value.kind not in {"divsi", "remsi"}:
        return False
    if value.converted_type.kind != "scalar" or int(value.converted_type.component_count) != 1:
        return False
    divisor = _constant_int_record(value.rhs)
    if not _is_positive_power_of_two(divisor):
        return False
    lhs_lower = _scalar_lower_bound(state, value.lhs_id)
    if lhs_lower is None or lhs_lower < 0:
        return False
    lhs_value = lhs[0]
    x = state.dsl.sym("x")
    lhs_value = state.builder.assume(
        lhs_value,
        [x >= int(lhs_lower), x <= 2147483647],
        name="x",
    )
    result_type = _parse_type(state.ir, value.converted_type.wave_type)
    if str(result_type) != "index":
        state.values[value.value_id] = state.builder.binary(
            _binary_kind(
                state.dsl,
                "divui" if value.kind == "divsi" else "remui",
            ),
            _ensure_binary_operand_type(state, lhs_value, result_type),
            _ensure_binary_operand_type(state, rhs[0], result_type),
        )
        return True
    i32 = state.dsl.i32()
    lhs_i32 = (
        lhs_value
        if str(lhs_value.type) == "i32"
        else _emit_int_convert(state, lhs_value, i32)
    )
    rhs_i32 = (
        rhs[0]
        if str(rhs[0].type) == "i32"
        else _emit_int_convert(state, rhs[0], i32)
    )
    result_i32 = state.builder.binary(
        _binary_kind(
            state.dsl,
            "divui" if value.kind == "divsi" else "remui",
        ),
        lhs_i32,
        rhs_i32,
    )
    state.values[value.value_id] = _emit_int_convert(state, result_i32, result_type)
    return True


def _record_scalar_fact(state: _StructuralState, value_id: int, kind: str, constant):
    if kind not in {"eq", "sgt", "sge"} or constant is None:
        return
    if kind == "sgt" and int(constant) >= -1:
        state.nonnegative_values.add(value_id)
    if kind == "sge" and int(constant) >= 0:
        state.nonnegative_values.add(value_id)
    if kind == "eq" and int(constant) >= 0:
        state.nonnegative_values.add(value_id)
    if kind == "sgt" and int(constant) >= 0:
        state.positive_values.add(value_id)
        state.nonnegative_values.add(value_id)
    if kind == "sge" and int(constant) >= 1:
        state.positive_values.add(value_id)
        state.nonnegative_values.add(value_id)
    if kind == "eq" and int(constant) > 0:
        state.positive_values.add(value_id)
        state.nonnegative_values.add(value_id)
    lower = _fact_lower_bound(kind, int(constant))
    if lower is not None:
        current = state.lower_bounds.get(value_id)
        state.lower_bounds[value_id] = lower if current is None else max(current, lower)


def _seed_assume_range_facts(state: _StructuralState):
    for effect in state.result.effects:
        if not isinstance(effect, _AssumeEffect):
            continue
        for value_id, kind, constant in _assume_fact_records(
            effect.predicate,
            _bounded_scalar_value_ids(state),
        ):
            _record_scalar_fact(state, value_id, kind, constant)


def _fact_lower_bound(kind: str, constant: int):
    if kind == "eq":
        return constant
    if kind == "sge":
        return constant
    if kind == "sgt":
        return constant + 1
    return None


def _scalar_lower_bound(state: _StructuralState, value_id: int, seen=frozenset()):
    if value_id in state.lower_bounds:
        return state.lower_bounds[value_id]
    if value_id in seen:
        return None
    value = state.result.values.get(value_id)
    if value is None:
        return None
    if isinstance(value, _ConstantValue):
        return _constant_int_record(value)
    if isinstance(value, _ProgramIdValue):
        return 0
    if isinstance(value, (_ForwardValue, _CastValue)):
        return _scalar_lower_bound(state, value.operand_id, seen | {value_id})
    if isinstance(value, _MinValue):
        lhs = _scalar_lower_bound(state, value.lhs_id, seen | {value_id})
        rhs = _scalar_lower_bound(state, value.rhs_id, seen | {value_id})
        if lhs is not None and rhs is not None:
            return min(lhs, rhs)
        return None
    if isinstance(value, _BinaryValue):
        return _binary_lower_bound(state, value, seen | {value_id})
    return None


def _binary_lower_bound(state: _StructuralState, value: _BinaryValue, seen):
    lhs = _scalar_lower_bound(state, value.lhs_id, seen)
    rhs = _scalar_lower_bound(state, value.rhs_id, seen)
    if value.kind == "addi" and lhs is not None and rhs is not None:
        return lhs + rhs
    if value.kind == "muli":
        lhs_const = _constant_int_record(value.lhs)
        rhs_const = _constant_int_record(value.rhs)
        if lhs is not None and rhs is not None and lhs >= 0 and rhs >= 0:
            return lhs * rhs
        if rhs_const is not None and rhs_const >= 0 and lhs is not None:
            return lhs * rhs_const
        if lhs_const is not None and lhs_const >= 0 and rhs is not None:
            return rhs * lhs_const
    return None


def _ensure_signed_bound_assume(state: _StructuralState, value_id: int, value):
    bounds = (
        _signed_int_bounds(state.result.values[value_id].converted_type.wave_type)
        if value_id in state.program.kernel_args
        else None
    )
    if bounds is None or value_id in state.bounded_kernel_args:
        return value
    lower, upper = bounds
    assumed = state.builder.assume_range(value, lower, upper)
    state.bounded_kernel_args.add(value_id)
    state.values[value_id] = assumed
    return assumed


def _is_signed_integer_kernel_arg(state: _StructuralState, value_id: int):
    if value_id not in state.program.kernel_args:
        return False
    return _signed_int_bounds(state.result.values[value_id].converted_type.wave_type) is not None


def _is_positive_power_of_two(value):
    if value is None:
        return False
    value = int(value)
    return value > 0 and value & (value - 1) == 0


def _assume_fact_records(predicate, bounded_value_ids=frozenset()):
    raw_facts = _raw_assume_facts(predicate)
    facts_by_id = {}
    for value_id, kind, constant in raw_facts:
        facts_by_id.setdefault(value_id, set()).add((kind, constant))
    filtered = []
    seen = set()
    for fact in raw_facts:
        value_id, kind, _constant = fact
        if kind == "pow2":
            predicates = facts_by_id.get(value_id, set())
            if ("sgt", 0) not in predicates:
                continue
            if ("sle", 2147483647) not in predicates and value_id not in bounded_value_ids:
                continue
        if fact in seen:
            continue
        seen.add(fact)
        filtered.append(fact)
    return tuple(filtered)


def _bounded_scalar_value_ids(state: _StructuralState):
    return frozenset(
        value_id
        for value_id in state.program.kernel_args
        if _signed_int_bounds(state.result.values[value_id].converted_type.wave_type)
        is not None
    )


def _raw_assume_facts(predicate):
    if isinstance(predicate, _MaskAndValue):
        return _raw_assume_facts(predicate.lhs) + _raw_assume_facts(predicate.rhs)
    if isinstance(predicate, _BinaryValue) and predicate.kind == "andi":
        return _raw_assume_facts(predicate.lhs) + _raw_assume_facts(predicate.rhs)
    pow2_fact = _power_of_two_assume_fact(predicate)
    if pow2_fact is not None:
        return (pow2_fact,)
    divisible_fact = _divisibility_assume_fact(predicate)
    if divisible_fact is not None:
        return (divisible_fact,)
    simple_fact = _simple_assume_fact(predicate)
    return () if simple_fact is None else (simple_fact,)


def _divisibility_assume_fact(predicate):
    if not isinstance(predicate, _CompareValue) or predicate.predicate != "eq":
        return None
    lhs_const = _constant_int_record(predicate.lhs)
    rhs_const = _constant_int_record(predicate.rhs)
    if rhs_const == 0:
        rem = predicate.lhs
    elif lhs_const == 0:
        rem = predicate.rhs
    else:
        return None
    if not isinstance(rem, _BinaryValue) or rem.kind != "remsi":
        return None
    divisor = _constant_int_record(rem.rhs)
    if divisor is None or int(divisor) == 0:
        return None
    return rem.lhs_id, "divisible", abs(int(divisor))


def _power_of_two_assume_fact(predicate):
    if not isinstance(predicate, _CompareValue) or predicate.predicate != "eq":
        return None
    lhs_const = _constant_int_record(predicate.lhs)
    rhs_const = _constant_int_record(predicate.rhs)
    if rhs_const == 0:
        bits = predicate.lhs
    elif lhs_const == 0:
        bits = predicate.rhs
    else:
        return None
    if not isinstance(bits, _BinaryValue) or bits.kind != "andi":
        return None
    lhs_base = _sub_one_base_id(bits.lhs)
    rhs_base = _sub_one_base_id(bits.rhs)
    if lhs_base is not None and lhs_base == bits.rhs_id:
        return lhs_base, "pow2", None
    if rhs_base is not None and rhs_base == bits.lhs_id:
        return rhs_base, "pow2", None
    return None


def _sub_one_base_id(value):
    if not isinstance(value, _BinaryValue) or value.kind != "subi":
        return None
    rhs_const = _constant_int_record(value.rhs)
    if rhs_const != 1:
        return None
    return value.lhs_id


def _simple_assume_fact(predicate):
    if isinstance(predicate, _ConstantValue):
        return None
    if not isinstance(predicate, _CompareValue):
        return None
    lhs_const = _constant_int_record(predicate.lhs)
    rhs_const = _constant_int_record(predicate.rhs)
    if rhs_const is not None:
        return _predicate_fact(predicate.lhs_id, predicate.predicate, rhs_const)
    if lhs_const is not None:
        inverted = _invert_predicate(predicate.predicate)
        if inverted is None:
            return None
        return _predicate_fact(predicate.rhs_id, inverted, lhs_const)
    return None


def _predicate_fact(value_id, predicate, constant):
    if predicate not in {"eq", "slt", "sle", "sgt", "sge"}:
        return None
    return value_id, predicate, int(constant)


def _invert_predicate(predicate):
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


def _constant_int_record(value):
    if isinstance(value, _ConstantValue):
        literal = _scalar_constant_literal(value)
        if isinstance(literal, int) and not isinstance(literal, bool):
            return literal
    return None


def _assume_expr(dsl, kind, constant, *, canonicalize_strict=False):
    x = dsl.sym_ctx.sym("x")
    if kind == "eq":
        return dsl.sym_ctx.eq(x, int(constant))
    if kind == "ne":
        return dsl.sym_ctx.ne(x, int(constant))
    if kind == "slt":
        constant = int(constant)
        return x <= constant - 1 if canonicalize_strict else x < constant
    if kind == "sle":
        return x <= int(constant)
    if kind == "sgt":
        constant = int(constant)
        return x >= constant + 1 if canonicalize_strict else x > constant
    if kind == "sge":
        return x >= int(constant)
    if kind == "pow2":
        return dsl.sym_ctx.eq(x & (x - 1), 0)
    if kind == "divisible":
        return dsl.sym_ctx.eq(x % int(constant), 0)
    return None


def _scalar_constant_literal(value: _ConstantValue):
    literal = value.literal
    if isinstance(literal, str) and literal.startswith("dense<"):
        literal = literal[len("dense<") :].split(">", 1)[0]
    if isinstance(literal, str):
        text = literal.strip()
        if "," in text:
            return None
        if text == "true":
            return True
        if text == "false":
            return False
        try:
            return int(text, 0)
        except ValueError:
            pass
        try:
            return float(text)
        except ValueError:
            pass
    return literal


def _components(value):
    return value if isinstance(value, tuple) else (value,)


def _pack(values):
    values = tuple(values)
    if len(values) == 1:
        return values[0]
    return values


def _compute_lds_layout(program):
    offsets = {}
    cursor = 0
    for op in program.ops:
        if op.name != "ttg.local_alloc" or not op.results:
            continue
        value_id = op.results[0]
        size = _local_alloc_size_bytes(program.values[value_id].type)
        offsets[value_id] = cursor
        cursor = _align_to(cursor + size, 16)
    return offsets, cursor


def _local_alloc_size_bytes(source_type):
    byte_width = source_type.element_byte_width
    if byte_width is None:
        raise _UnsupportedStructuralEmission(
            f"tlx_wave structural emitter cannot size LDS allocation "
            f"{source_type.raw}: unknown element byte width"
        )
    return _product(source_type.alloc_shape or source_type.shape or (1,)) * int(byte_width)


def _source_type(state: _StructuralState, value_id: int):
    return state.program.values[value_id].type


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def _align_to(value, alignment):
    if value == 0:
        return 0
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _int_or(value, default):
    return default if value is None else int(value)
