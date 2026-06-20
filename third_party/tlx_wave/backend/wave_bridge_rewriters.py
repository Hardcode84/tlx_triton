"""Stateless TTGIR op rewriters for the new TLX Wave conversion path."""

from dataclasses import dataclass
import re

from .wave_bridge_conversion import (
    _ConvertedInputValue,
    _OpRewritePattern,
    _RegionYield,
    _RewriteResult,
    _RewriterRegistry,
    _attr_bool,
    _attr_value,
    _dot_operand_encoding_info,
    _mma_shape_for_parent,
    _tile_rep_count,
)


@dataclass(frozen=True)
class _ConstantValue:
    value_id: int
    literal: object
    raw_literal: str | None
    converted_type: object


@dataclass(frozen=True)
class _RangeValue:
    value_id: int
    start: int
    end: int
    converted_type: object


@dataclass(frozen=True)
class _ProgramIdValue:
    value_id: int
    axis: int
    converted_type: object


@dataclass(frozen=True)
class _UnaryTensorValue:
    value_id: int
    op_name: str
    operand_id: int
    operand: object
    converted_type: object


@dataclass(frozen=True)
class _BinaryValue:
    value_id: int
    op_name: str
    kind: str
    lhs_id: int
    rhs_id: int
    lhs: object
    rhs: object
    converted_type: object
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CompareValue:
    value_id: int
    predicate: str
    lhs_id: int
    rhs_id: int
    lhs: object
    rhs: object
    converted_type: object


@dataclass(frozen=True)
class _MaskAndValue:
    value_id: int
    lhs_id: int
    rhs_id: int
    lhs: object
    rhs: object
    converted_type: object


@dataclass(frozen=True)
class _MinValue:
    value_id: int
    op_name: str
    predicate: str
    lhs_id: int
    rhs_id: int
    lhs: object
    rhs: object
    converted_type: object


@dataclass(frozen=True)
class _PointerAddValue:
    value_id: int
    base_id: int
    offset_id: int
    base: object
    offset: object
    converted_type: object


@dataclass(frozen=True)
class _DotValue:
    value_id: int
    lhs_id: int
    rhs_id: int
    acc_id: int
    lhs: object
    rhs: object
    acc: object
    mma_kind: str
    lhs_tile_shape: tuple[int, int]
    rhs_tile_shape: tuple[int, int]
    result_tile_shape: tuple[int, int]
    converted_type: object


@dataclass(frozen=True)
class _ReturnEffect:
    operand_ids: tuple[int, ...]
    operands: tuple[object, ...]
    op_index: int = -1


@dataclass(frozen=True)
class _AssumeEffect:
    predicate_id: int
    predicate: object
    op_index: int = -1


@dataclass(frozen=True)
class _IfEffect:
    condition_id: int
    condition: object
    then_op_indices: tuple[int, ...]
    else_op_indices: tuple[int, ...]
    then_values: dict[int, object]
    else_values: dict[int, object]
    then_effects: tuple[object, ...]
    else_effects: tuple[object, ...]
    op_index: int = -1


@dataclass(frozen=True)
class _IfValue:
    value_id: int
    result_ids: tuple[int, ...]
    result_index: int
    condition_id: int
    condition: object
    then_op_indices: tuple[int, ...]
    else_op_indices: tuple[int, ...]
    then_values: dict[int, object]
    else_values: dict[int, object]
    then_yield_ids: tuple[int, ...]
    else_yield_ids: tuple[int, ...]
    converted_type: object


@dataclass(frozen=True)
class _ForValue:
    value_id: int
    result_ids: tuple[int, ...]
    result_index: int
    lower_id: int
    upper_id: int
    step_id: int
    iter_arg_ids: tuple[int, ...]
    body_arg_ids: tuple[int, ...]
    lower: object
    upper: object
    step: object
    iter_args: tuple[object, ...]
    body_op_indices: tuple[int, ...]
    body_values: dict[int, object]
    body_effects: tuple[object, ...]
    body_yield_ids: tuple[int, ...]
    result_types: tuple[object, ...]
    converted_type: object


@dataclass(frozen=True)
class _ForEffect:
    lower_id: int
    upper_id: int
    step_id: int
    iter_arg_ids: tuple[int, ...]
    body_arg_ids: tuple[int, ...]
    lower: object
    upper: object
    step: object
    iter_args: tuple[object, ...]
    body_op_indices: tuple[int, ...]
    body_values: dict[int, object]
    body_effects: tuple[object, ...]
    body_yield_ids: tuple[int, ...]
    result_types: tuple[object, ...]
    op_index: int = -1


@dataclass(frozen=True)
class _LocalAllocValue:
    value_id: int
    converted_type: object


@dataclass(frozen=True)
class _MemdescIndexValue:
    value_id: int
    memdesc_id: int
    index_id: int
    memdesc: object
    index: object
    converted_type: object


@dataclass(frozen=True)
class _AsyncCopyValue:
    value_id: int
    op_name: str
    source_address_value_id: int | None
    source_offset_value_id: int | None
    memdesc_value_id: int | None
    mask_value_id: int | None
    other_value_id: int | None
    source_address: object | None
    source_offset: object | None
    memdesc: object | None
    mask: object | None
    other: object | None
    token_node: object


@dataclass(frozen=True)
class _AsyncCommitGroupValue:
    value_id: int
    group_index: int
    member_token_ids: tuple[int, ...]
    member_tokens: tuple[object, ...]
    token_node: object


@dataclass(frozen=True)
class _AsyncWaitEffect:
    wait_group: int | None
    waited_group_indices: tuple[int, ...]
    input_token_ids: tuple[int, ...]
    input_tokens: tuple[object, ...]
    op_index: int = -1


@dataclass(frozen=True)
class _AsyncWaitValue:
    value_id: int
    wait_group: int | None
    waited_group_indices: tuple[int, ...]
    input_token_ids: tuple[int, ...]
    input_tokens: tuple[object, ...]
    token_node: object


@dataclass(frozen=True)
class _LoadValue:
    value_id: int
    pointer_id: int
    mask_id: int | None
    other_id: int | None
    pointer: object
    mask: object | None
    other: object | None
    converted_type: object


@dataclass(frozen=True)
class _ForwardValue:
    value_id: int
    op_name: str
    operand_id: int
    operand: object
    converted_type: object


@dataclass(frozen=True)
class _CastValue:
    value_id: int
    op_name: str
    kind: str
    operand_id: int
    operand: object
    converted_type: object


@dataclass(frozen=True)
class _LocalLoadValue:
    value_id: int
    memdesc_id: int
    token_id: int | None
    memdesc: object
    token: object | None
    converted_type: object


@dataclass(frozen=True)
class _StoreEffect:
    pointer_id: int
    value_id: int
    mask_id: int | None
    pointer: object
    value: object
    mask: object | None
    op_index: int = -1


@dataclass(frozen=True)
class _BufferStoreEffect:
    value_id: int
    base_id: int
    offset_id: int
    mask_id: int | None
    value: object
    base: object
    offset: object
    mask: object | None
    contiguity: int | None = None
    op_index: int = -1


@dataclass(frozen=True)
class _LocalStoreEffect:
    value_id: int
    memdesc_id: int
    token_id: int | None
    value: object
    memdesc: object
    token: object | None
    op_index: int = -1


class _ConstantRewrite(_OpRewritePattern):
    op_name = "arith.constant"

    def rewrite(self, context, op, values):
        _require_result_count(op, 1)
        result_id = op.results[0]
        source_type = context.value(result_id).type
        raw_literal = _raw_constant_literal(op)
        return _RewriteResult(
            {
                result_id: _ConstantValue(
                    result_id,
                    _constant_literal(raw_literal, source_type),
                    raw_literal,
                    context.converted_type(result_id),
                )
            }
        )


class _MakeRangeRewrite(_OpRewritePattern):
    op_name = "tt.make_range"

    def rewrite(self, context, op, values):
        _require_result_count(op, 1)
        result_id = op.results[0]
        return _RewriteResult(
            {
                result_id: _RangeValue(
                    result_id,
                    _required_int_attr(op, "start"),
                    _required_int_attr(op, "end"),
                    context.converted_type(result_id),
                )
            }
        )


class _ProgramIdRewrite(_OpRewritePattern):
    op_name = "tt.get_program_id"

    def rewrite(self, context, op, values):
        _require_result_count(op, 1)
        result_id = op.results[0]
        axis = _required_int_attr(op, "axis")
        if axis not in (0, 1, 2):
            raise ValueError(f"tlx_wave conversion cannot lower tt.get_program_id axis {axis}")
        return _RewriteResult(
            {
                result_id: _ProgramIdValue(
                    result_id,
                    axis,
                    context.converted_type(result_id),
                )
            }
        )


class _SplatRewrite(_OpRewritePattern):
    op_name = "tt.splat"

    def rewrite(self, context, op, values):
        return _rewrite_unary_tensor(context, op, values)


class _BroadcastRewrite(_OpRewritePattern):
    op_name = "tt.broadcast"

    def rewrite(self, context, op, values):
        return _rewrite_unary_tensor(context, op, values)


class _AddIRewrite(_OpRewritePattern):
    op_name = "arith.addi"
    kind = "addi"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _MulIRewrite(_OpRewritePattern):
    op_name = "arith.muli"
    kind = "muli"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _SubIRewrite(_OpRewritePattern):
    op_name = "arith.subi"
    kind = "subi"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _DivSIRewrite(_OpRewritePattern):
    op_name = "arith.divsi"
    kind = "divsi"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _RemSIRewrite(_OpRewritePattern):
    op_name = "arith.remsi"
    kind = "remsi"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _DivUIRewrite(_OpRewritePattern):
    op_name = "arith.divui"
    kind = "divui"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _RemUIRewrite(_OpRewritePattern):
    op_name = "arith.remui"
    kind = "remui"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _MinSIRewrite(_OpRewritePattern):
    op_name = "arith.minsi"
    predicate = "sle"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 2)
        _require_result_count(op, 1)
        result_id = op.results[0]
        lhs_id, rhs_id = op.operands
        return _RewriteResult(
            {
                result_id: _MinValue(
                    result_id,
                    op.name,
                    self.predicate,
                    lhs_id,
                    rhs_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    context.converted_type(result_id),
                )
            }
        )


class _CmpIRewrite(_OpRewritePattern):
    op_name = "arith.cmpi"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 2)
        _require_result_count(op, 1)
        result_id = op.results[0]
        lhs_id, rhs_id = op.operands
        return _RewriteResult(
            {
                result_id: _CompareValue(
                    result_id,
                    _cmpi_predicate(op),
                    lhs_id,
                    rhs_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    context.converted_type(result_id),
                )
            }
        )


class _AndIRewrite(_OpRewritePattern):
    op_name = "arith.andi"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 2)
        _require_result_count(op, 1)
        result_id = op.results[0]
        lhs_id, rhs_id = op.operands
        converted_type = context.converted_type(result_id)
        if converted_type.kind != "mask":
            return _rewrite_binary(context, op, values, "andi")
        return _RewriteResult(
            {
                result_id: _MaskAndValue(
                    result_id,
                    lhs_id,
                    rhs_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    converted_type,
                )
            }
        )


class _OrIRewrite(_OpRewritePattern):
    op_name = "arith.ori"
    kind = "ori"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _XorIRewrite(_OpRewritePattern):
    op_name = "arith.xori"
    kind = "xori"

    def rewrite(self, context, op, values):
        return _rewrite_binary(context, op, values, self.kind)


class _AddPtrRewrite(_OpRewritePattern):
    op_name = "tt.addptr"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 2)
        _require_result_count(op, 1)
        result_id = op.results[0]
        base_id, offset_id = op.operands
        return _RewriteResult(
            {
                result_id: _PointerAddValue(
                    result_id,
                    base_id,
                    offset_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    context.converted_type(result_id),
                )
            }
        )


class _LocalAllocRewrite(_OpRewritePattern):
    op_name = "ttg.local_alloc"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 0)
        _require_result_count(op, 1)
        result_id = op.results[0]
        return _RewriteResult(
            {
                result_id: _LocalAllocValue(
                    result_id,
                    context.converted_type(result_id),
                )
            }
        )


class _MemdescIndexRewrite(_OpRewritePattern):
    op_name = "ttg.memdesc_index"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 2)
        _require_result_count(op, 1)
        result_id = op.results[0]
        memdesc_id, index_id = op.operands
        return _RewriteResult(
            {
                result_id: _MemdescIndexValue(
                    result_id,
                    memdesc_id,
                    index_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    context.converted_type(result_id),
                )
            }
        )


class _AsyncCopyGlobalToLocalRewrite(_OpRewritePattern):
    op_name = "ttg.async_copy_global_to_local"

    def rewrite(self, context, op, values):
        return _rewrite_async_copy(context, op, values)


class _BufferLoadToLocalRewrite(_OpRewritePattern):
    op_name = "amdg.buffer_load_to_local"

    def rewrite(self, context, op, values):
        return _rewrite_async_copy(context, op, values)


class _AsyncCommitGroupRewrite(_OpRewritePattern):
    op_name = "ttg.async_commit_group"

    def rewrite(self, context, op, values):
        _require_result_count(op, 1)
        result_id = op.results[0]
        node = context.token_node(result_id)
        if node is None or node.committed_group_index is None:
            raise ValueError(
                "tlx_wave conversion cannot rewrite ttg.async_commit_group "
                "without a token-graph commit node"
            )
        group = context.program.token_graph.groups[node.committed_group_index]
        return _RewriteResult(
            {
                result_id: _AsyncCommitGroupValue(
                    result_id,
                    group.index,
                    group.member_token_ids,
                    tuple(values[token_id] for token_id in group.member_token_ids),
                    node,
                )
            }
        )


class _AsyncWaitRewrite(_OpRewritePattern):
    op_name = "ttg.async_wait"

    def rewrite(self, context, op, values):
        node = _single_token_event_node(context, op)
        if node.value_id is not None:
            return _RewriteResult(
                {
                    node.value_id: _AsyncWaitValue(
                        node.value_id,
                        node.wait_group,
                        node.waited_group_indices,
                        node.input_token_ids,
                        tuple(values[token_id] for token_id in node.input_token_ids),
                        node,
                    )
                }
            )
        return _RewriteResult(
            effects=(
                _AsyncWaitEffect(
                    node.wait_group,
                    node.waited_group_indices,
                    node.input_token_ids,
                    tuple(values[token_id] for token_id in node.input_token_ids),
                    op.index,
                ),
            )
        )


class _LoadRewrite(_OpRewritePattern):
    op_name = "tt.load"

    def rewrite(self, context, op, values):
        if not op.operands:
            raise ValueError("tlx_wave conversion expected tt.load pointer operand")
        _require_result_count(op, 1)
        result_id = op.results[0]
        pointer_id = op.operands[0]
        mask_id = op.operands[1] if len(op.operands) > 1 else None
        other_id = op.operands[2] if len(op.operands) > 2 else None
        return _RewriteResult(
            {
                result_id: _LoadValue(
                    result_id,
                    pointer_id,
                    mask_id,
                    other_id,
                    _operand(values, op, 0),
                    _optional_value(values, mask_id),
                    _optional_value(values, other_id),
                    context.converted_type(result_id),
                )
            }
        )


class _ExpandDimsRewrite(_OpRewritePattern):
    op_name = "tt.expand_dims"

    def rewrite(self, context, op, values):
        return _rewrite_forward(context, op, values)


class _ConvertLayoutRewrite(_OpRewritePattern):
    op_name = "ttg.convert_layout"

    def rewrite(self, context, op, values):
        _validate_convert_layout_component_mapping(context, op)
        return _rewrite_forward(context, op, values)


class _TruncFRewrite(_OpRewritePattern):
    op_name = "arith.truncf"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 1)
        _require_result_count(op, 1)
        result_id = op.results[0]
        operand_id = op.operands[0]
        return _RewriteResult(
            {
                result_id: _CastValue(
                    result_id,
                    op.name,
                    "fpconvert",
                    operand_id,
                    _operand(values, op, 0),
                    context.converted_type(result_id),
                )
            }
        )


class _StoreRewrite(_OpRewritePattern):
    op_name = "tt.store"

    def rewrite(self, context, op, values):
        if len(op.operands) < 2:
            raise ValueError("tlx_wave conversion expected tt.store pointer/value operands")
        mask_id = op.operands[2] if len(op.operands) > 2 else None
        return _RewriteResult(
            effects=(
                _StoreEffect(
                    op.operands[0],
                    op.operands[1],
                    mask_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    _optional_value(values, mask_id),
                    op.index,
                ),
            )
        )


class _BufferStoreRewrite(_OpRewritePattern):
    op_name = "amdg.buffer_store"

    def rewrite(self, context, op, values):
        _require_default_cache_modifier(op, "amdg.buffer_store")
        segments = op.attrs.get("operandSegmentSizes")
        if segments is None:
            raise ValueError(
                "tlx_wave conversion expected amdg.buffer_store "
                "operandSegmentSizes attribute"
            )
        if len(segments) != 5:
            raise ValueError(
                "tlx_wave conversion expected amdg.buffer_store "
                f"operandSegmentSizes with five entries, got {segments}"
            )
        if sum(segments) != len(op.operands):
            raise ValueError(
                "tlx_wave conversion found inconsistent amdg.buffer_store "
                f"operandSegmentSizes={segments} for {len(op.operands)} operands"
            )
        if segments[0] != 1 or segments[1] != 1 or segments[2] != 1:
            raise ValueError(
                "tlx_wave conversion expected amdg.buffer_store value, base "
                f"pointer, and offsets operands, got operandSegmentSizes={segments}"
            )
        if segments[3] != 0:
            raise ValueError(
                "tlx_wave conversion cannot lower amdg.buffer_store with "
                "boundary-check operands yet"
            )
        if segments[4] not in (0, 1):
            raise ValueError(
                "tlx_wave conversion expected optional single amdg.buffer_store "
                f"mask operand, got operandSegmentSizes={segments}"
            )
        index = 0
        value_id = op.operands[index]
        index += segments[0]
        base_id = op.operands[index]
        index += segments[1]
        offset_id = op.operands[index]
        index += segments[2]
        index += segments[3]
        mask_id = op.operands[index] if segments[4] else None
        return _RewriteResult(
            effects=(
                _BufferStoreEffect(
                    value_id,
                    base_id,
                    offset_id,
                    mask_id,
                    values[value_id],
                    values[base_id],
                    values[offset_id],
                    _optional_value(values, mask_id),
                    _optional_int_attr(op.attrs.get("contiguity")),
                    op.index,
                ),
            )
        )


class _AssumeRewrite(_OpRewritePattern):
    op_name = "llvm.intr.assume"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 1)
        if op.results:
            raise ValueError("tlx_wave conversion expected llvm.intr.assume with no results")
        predicate_id = op.operands[0]
        return _RewriteResult(
            effects=(
                _AssumeEffect(
                    predicate_id,
                    _operand(values, op, 0),
                    op.index,
                ),
            )
        )


class _ScfIfRewrite(_OpRewritePattern):
    op_name = "scf.if"

    def rewrite(self, context, op, values):
        _require_operand_count(op, 1)
        condition_id = op.operands[0]
        if len(op.regions) not in (1, 2):
            raise ValueError(
                f"tlx_wave conversion expected scf.if with one or two regions, "
                f"got {len(op.regions)}"
            )
        if op.results and len(op.regions) != 2:
            raise ValueError(
                "tlx_wave conversion cannot lower result-bearing scf.if "
                "without an else region"
            )
        then_result = context.convert_region(op, 0, values)
        else_result = (
            context.convert_region(op, 1, values)
            if len(op.regions) == 2
            else None
        )
        if op.results and (
            then_result.effects or (else_result is not None and else_result.effects)
        ):
            raise ValueError(
                "tlx_wave conversion cannot lower side effects inside scf.if yet"
            )
        if len(then_result.yield_operand_ids) != len(op.results):
            raise ValueError(
                "tlx_wave conversion expected scf.if then yield to match "
                f"{len(op.results)} result(s)"
            )
        if else_result is not None and len(else_result.yield_operand_ids) != len(op.results):
            raise ValueError(
                "tlx_wave conversion expected scf.if else yield to match "
                f"{len(op.results)} result(s)"
            )
        if not op.results:
            if not then_result.effects and (
                else_result is None or not else_result.effects
            ):
                return _RewriteResult()
            return _RewriteResult(
                effects=(
                    _IfEffect(
                        condition_id,
                        _operand(values, op, 0),
                        op.regions[0],
                        op.regions[1] if len(op.regions) == 2 else (),
                        then_result.values,
                        {} if else_result is None else else_result.values,
                        then_result.effects,
                        () if else_result is None else else_result.effects,
                        op.index,
                    ),
                )
            )
        return _RewriteResult(
            {
                result_id: _IfValue(
                    result_id,
                    tuple(op.results),
                    result_index,
                    condition_id,
                    _operand(values, op, 0),
                    op.regions[0],
                    op.regions[1],
                    then_result.values,
                    else_result.values,
                    then_result.yield_operand_ids,
                    else_result.yield_operand_ids,
                    context.converted_type(result_id),
                )
                for result_index, result_id in enumerate(op.results)
            }
        )


class _ScfForRewrite(_OpRewritePattern):
    op_name = "scf.for"

    def rewrite(self, context, op, values):
        if len(op.operands) < 3:
            raise ValueError("tlx_wave conversion expected scf.for with bounds")
        if len(op.regions) != 1:
            raise ValueError(
                f"tlx_wave conversion expected scf.for with one body region, "
                f"got {len(op.regions)}"
            )
        iter_arg_ids = tuple(op.operands[3:])
        if len(iter_arg_ids) != len(op.results):
            raise ValueError(
                "tlx_wave conversion expected scf.for iter_args/result count "
                f"to match, got {len(iter_arg_ids)} iter_args and "
                f"{len(op.results)} result(s)"
            )
        body_arg_ids = op.region_args[0] if op.region_args else ()
        if len(body_arg_ids) != 1 + len(iter_arg_ids):
            raise ValueError(
                "tlx_wave conversion expected scf.for body args to match "
                f"induction plus iter_args, got {len(body_arg_ids)} body arg(s) "
                f"for {len(iter_arg_ids)} iter_arg(s)"
            )
        body_values = dict(values)
        body_values[body_arg_ids[0]] = _ConvertedInputValue(
            body_arg_ids[0],
            "scf.for.induction",
            context.converted_type(body_arg_ids[0]),
        )
        for iter_arg_id, body_arg_id in zip(iter_arg_ids, body_arg_ids[1:]):
            body_values[body_arg_id] = values[iter_arg_id]
        body_result = context.convert_region(op, 0, body_values)
        if len(body_result.yield_operand_ids) != len(iter_arg_ids):
            raise ValueError(
                "tlx_wave conversion expected scf.for yield to match "
                f"{len(iter_arg_ids)} iter_arg(s)"
            )
        result_types = tuple(context.converted_type(result_id) for result_id in op.results)
        common = dict(
            lower_id=op.operands[0],
            upper_id=op.operands[1],
            step_id=op.operands[2],
            iter_arg_ids=iter_arg_ids,
            body_arg_ids=tuple(body_arg_ids),
            lower=_operand(values, op, 0),
            upper=_operand(values, op, 1),
            step=_operand(values, op, 2),
            iter_args=tuple(values[value_id] for value_id in iter_arg_ids),
            body_op_indices=op.regions[0],
            body_values=body_result.values,
            body_effects=body_result.effects,
            body_yield_ids=body_result.yield_operand_ids,
            result_types=result_types,
        )
        if not op.results:
            return _RewriteResult(effects=(_ForEffect(op_index=op.index, **common),))
        return _RewriteResult(
            {
                result_id: _ForValue(
                    result_id,
                    tuple(op.results),
                    result_index,
                    converted_type=result_types[result_index],
                    **common,
                )
                for result_index, result_id in enumerate(op.results)
            }
        )


class _ScfYieldRewrite(_OpRewritePattern):
    op_name = "scf.yield"

    def rewrite(self, context, op, values):
        return _RewriteResult(
            effects=(
                _RegionYield(
                    tuple(op.operands),
                    tuple(values[value_id] for value_id in op.operands),
                    op.index,
                ),
            )
        )


class _SchedBarrierRewrite(_OpRewritePattern):
    op_name = "rocdl.sched.barrier"

    def rewrite(self, context, op, values):
        if op.results:
            raise ValueError(
                "tlx_wave conversion expected rocdl.sched.barrier with no results"
            )
        return _RewriteResult()


class _SchedGroupBarrierRewrite(_SchedBarrierRewrite):
    op_name = "rocdl.sched.group.barrier"


class _LocalLoadRewrite(_OpRewritePattern):
    op_name = "ttg.local_load"

    def rewrite(self, context, op, values):
        if not op.operands:
            raise ValueError("tlx_wave conversion expected ttg.local_load memdesc operand")
        _require_result_count(op, 1)
        result_id = op.results[0]
        memdesc_id = op.operands[0]
        token_id = op.operands[1] if len(op.operands) > 1 else None
        return _RewriteResult(
            {
                result_id: _LocalLoadValue(
                    result_id,
                    memdesc_id,
                    token_id,
                    _operand(values, op, 0),
                    _optional_value(values, token_id),
                    context.converted_type(result_id),
                )
            }
        )


class _LocalStoreRewrite(_OpRewritePattern):
    op_name = "ttg.local_store"

    def rewrite(self, context, op, values):
        if len(op.operands) < 2:
            raise ValueError(
                "tlx_wave conversion expected ttg.local_store value/memdesc operands"
            )
        token_id = op.operands[2] if len(op.operands) > 2 else None
        return _RewriteResult(
            effects=(
                _LocalStoreEffect(
                    op.operands[0],
                    op.operands[1],
                    token_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    _optional_value(values, token_id),
                    op.index,
                ),
            )
        )


class _DotRewrite(_OpRewritePattern):
    op_name = "tt.dot"

    def rewrite(self, context, op, values):
        if len(op.operands) < 3:
            raise ValueError("tlx_wave conversion expected tt.dot with 3 operands")
        _require_result_count(op, 1)
        lhs_id, rhs_id, acc_id = op.operands[:3]
        result_id = op.results[0]
        lhs_type = context.value(lhs_id).type
        rhs_type = context.value(rhs_id).type
        acc_type = context.value(acc_id).type
        result_type = context.value(result_id).type
        lhs_info = _require_dot_operand_info(lhs_type, 0, "tt.dot lhs")
        rhs_info = _require_dot_operand_info(rhs_type, 1, "tt.dot rhs")
        if lhs_info.parent != rhs_info.parent:
            raise ValueError(
                "tlx_wave conversion expected tt.dot operands with matching "
                "MFMA parent layouts"
            )
        mma = _mma_shape_for_parent(lhs_info.parent, "tt.dot")
        mma_kind = _mma_kind(lhs_type.element_type, rhs_type.element_type, lhs_info, rhs_info, mma)
        _validate_dot_shapes(lhs_type, rhs_type, acc_type, result_type, mma)
        return _RewriteResult(
            {
                result_id: _DotValue(
                    result_id,
                    lhs_id,
                    rhs_id,
                    acc_id,
                    _operand(values, op, 0),
                    _operand(values, op, 1),
                    _operand(values, op, 2),
                    mma_kind,
                    (
                        _tile_rep_count(lhs_type.shape[0], mma.output_tile_shape[0]),
                        _tile_rep_count(lhs_type.shape[1], mma.k_dim),
                    ),
                    (
                        _tile_rep_count(rhs_type.shape[0], mma.k_dim),
                        _tile_rep_count(rhs_type.shape[1], mma.output_tile_shape[1]),
                    ),
                    (
                        _tile_rep_count(result_type.shape[0], mma.output_tile_shape[0]),
                        _tile_rep_count(result_type.shape[1], mma.output_tile_shape[1]),
                    ),
                    context.converted_type(result_id),
                )
            }
        )


class _ReturnRewrite(_OpRewritePattern):
    op_name = "tt.return"

    def rewrite(self, context, op, values):
        return _RewriteResult(
            effects=(
                _ReturnEffect(
                    tuple(op.operands),
                    tuple(values[value_id] for value_id in op.operands),
                    op.index,
                ),
            )
        )


def _basic_rewriter_registry():
    return _RewriterRegistry.from_patterns(
        (
            _ConstantRewrite(),
            _MakeRangeRewrite(),
            _ProgramIdRewrite(),
            _SplatRewrite(),
            _BroadcastRewrite(),
            _AddIRewrite(),
            _MulIRewrite(),
            _SubIRewrite(),
            _DivSIRewrite(),
            _RemSIRewrite(),
            _DivUIRewrite(),
            _RemUIRewrite(),
            _MinSIRewrite(),
            _CmpIRewrite(),
            _AndIRewrite(),
            _OrIRewrite(),
            _XorIRewrite(),
            _AddPtrRewrite(),
            _LocalAllocRewrite(),
            _MemdescIndexRewrite(),
            _AsyncCopyGlobalToLocalRewrite(),
            _BufferLoadToLocalRewrite(),
            _AsyncCommitGroupRewrite(),
            _AsyncWaitRewrite(),
            _LoadRewrite(),
            _ExpandDimsRewrite(),
            _ConvertLayoutRewrite(),
            _TruncFRewrite(),
            _StoreRewrite(),
            _BufferStoreRewrite(),
            _AssumeRewrite(),
            _ScfIfRewrite(),
            _ScfForRewrite(),
            _ScfYieldRewrite(),
            _SchedBarrierRewrite(),
            _SchedGroupBarrierRewrite(),
            _LocalLoadRewrite(),
            _LocalStoreRewrite(),
            _DotRewrite(),
            _ReturnRewrite(),
        )
    )


def _rewrite_unary_tensor(context, op, values):
    _require_operand_count(op, 1)
    _require_result_count(op, 1)
    result_id = op.results[0]
    operand_id = op.operands[0]
    return _RewriteResult(
        {
            result_id: _UnaryTensorValue(
                result_id,
                op.name,
                operand_id,
                _operand(values, op, 0),
                context.converted_type(result_id),
            )
        }
    )


def _rewrite_binary(context, op, values, kind):
    _require_operand_count(op, 2)
    _require_result_count(op, 1)
    result_id = op.results[0]
    lhs_id, rhs_id = op.operands
    return _RewriteResult(
        {
            result_id: _BinaryValue(
                result_id,
                op.name,
                kind,
                lhs_id,
                rhs_id,
                _operand(values, op, 0),
                _operand(values, op, 1),
                context.converted_type(result_id),
                _overflow_flags(op),
            )
        }
    )


def _rewrite_forward(context, op, values):
    _require_operand_count(op, 1)
    _require_result_count(op, 1)
    result_id = op.results[0]
    operand_id = op.operands[0]
    return _RewriteResult(
        {
            result_id: _ForwardValue(
                result_id,
                op.name,
                operand_id,
                _operand(values, op, 0),
                context.converted_type(result_id),
            )
        }
    )


def _rewrite_async_copy(context, op, values):
    _require_result_count(op, 1)
    result_id = op.results[0]
    node = context.token_node(result_id)
    if node is None:
        raise ValueError(
            f"tlx_wave conversion cannot rewrite {op.name} without a token graph node"
        )
    return _RewriteResult(
        {
            result_id: _AsyncCopyValue(
                result_id,
                op.name,
                node.source_address_value_id,
                node.source_offset_value_id,
                node.memdesc_value_id,
                node.mask_value_id,
                node.other_value_id,
                _optional_value(values, node.source_address_value_id),
                _optional_value(values, node.source_offset_value_id),
                _optional_value(values, node.memdesc_value_id),
                _optional_value(values, node.mask_value_id),
                _optional_value(values, node.other_value_id),
                node,
            )
        }
    )


def _validate_convert_layout_component_mapping(context, op):
    _require_operand_count(op, 1)
    _require_result_count(op, 1)
    source_type = context.value(op.operands[0]).type
    result_type = context.value(op.results[0]).type
    if source_type.kind != "tensor" or result_type.kind != "tensor":
        return
    if tuple(source_type.shape) != tuple(result_type.shape):
        return
    source_count = int(context.converted_type(op.operands[0]).component_count)
    result_count = int(context.converted_type(op.results[0]).component_count)
    component_count = max(source_count, result_count)
    if component_count <= 1:
        return
    lane_width = int(context.converted_type(op.results[0]).lane_width or 64)
    source_mapping = _rank1_component_lane_mapping(
        source_type, component_count, lane_width
    )
    result_mapping = _rank1_component_lane_mapping(
        result_type, component_count, lane_width
    )
    if source_mapping is None or result_mapping is None:
        return
    if source_mapping != result_mapping:
        raise ValueError(
            "tlx_wave conversion cannot lower ttg.convert_layout requiring "
            "cross-lane remap: source component mapping "
            f"{source_mapping} -> result component mapping {result_mapping}"
        )


def _rank1_component_lane_mapping(source_type, component_count, lane_width):
    if len(tuple(source_type.shape)) != 1:
        return None
    attr = source_type.encoding_attr
    if attr is None:
        return None
    if _attr_bool(attr, "is_blocked_encoding"):
        return _rank1_blocked_component_lane_mapping(
            attr, component_count, lane_width
        )
    if _attr_bool(attr, "is_linear_encoding"):
        return _rank1_linear_component_lane_mapping(
            attr, component_count
        )
    return None


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


def _rank1_linear_component_lane_mapping(attr, component_count):
    register_bases = _rank1_linear_bases(attr, "get_linear_register_bases")
    lane_bases = _rank1_linear_bases(attr, "get_linear_lane_bases")
    if register_bases is None or lane_bases is None:
        return None
    lane_coeff = _rank1_lane_coeff(lane_bases)
    if lane_coeff is None:
        return None
    if not register_bases:
        if int(component_count) != 1:
            return None
        return ((lane_coeff, 0),)
    if 1 << len(register_bases) != int(component_count):
        return None
    return tuple(
        (
            lane_coeff,
            sum(
                register_bases[bit]
                for bit in range(len(register_bases))
                if component & (1 << bit)
            ),
        )
        for component in range(component_count)
    )


def _rank1_linear_bases(attr, method):
    bases = _attr_value(attr, method)
    if bases is None:
        return ()
    result = []
    for basis in bases:
        vector = tuple(int(value) for value in basis)
        if len(vector) != 1:
            return None
        result.append(vector[0])
    return tuple(result)


def _rank1_lane_coeff(lane_bases):
    lane_bases = tuple(int(value) for value in lane_bases)
    if not lane_bases:
        return 0
    coeff = lane_bases[0]
    if coeff <= 0:
        return None
    for bit, basis in enumerate(lane_bases):
        if int(basis) != coeff * (1 << bit):
            return None
    return coeff


def _single_token_event_node(context, op):
    matches = tuple(
        node
        for node in context.program.token_graph.nodes
        if node.op_index == op.index and node.op_name == op.name
    )
    if len(matches) != 1:
        raise ValueError(
            f"tlx_wave conversion expected one token graph node for {op.name} "
            f"at op index {op.index}, got {len(matches)}"
        )
    return matches[0]


def _operand(values, op, index):
    value_id = op.operands[index]
    if value_id not in values:
        raise ValueError(
            f"tlx_wave conversion cannot rewrite {op.name}: operand {index} "
            f"value {value_id} has not been converted"
        )
    return values[value_id]


def _optional_value(values, value_id):
    if value_id is None:
        return None
    if value_id not in values:
        raise ValueError(
            "tlx_wave conversion cannot rewrite op: operand value "
            f"{value_id} has not been converted"
        )
    return values[value_id]


def _optional_int_attr(value):
    if value is None:
        return None
    return int(value)


def _require_default_cache_modifier(op, context):
    cache = op.attrs.get("cache")
    if cache is None or int(cache) == 1:
        return
    raise ValueError(
        f"tlx_wave conversion cannot lower {context} with cacheModifier={cache}; "
        "Wave lowering does not support cache modifiers yet"
    )


def _require_dot_operand_info(source_type, expected_op_idx, context):
    info = _dot_operand_encoding_info(source_type)
    if info is None:
        raise ValueError(
            f"tlx_wave conversion expected #ttg.dot_op AMD MFMA encoding for {context}"
        )
    if info.op_idx != expected_op_idx:
        raise ValueError(
            f"tlx_wave conversion expected {context} opIdx {expected_op_idx}, "
            f"got {info.op_idx}"
        )
    return info


def _mma_kind(lhs_element, rhs_element, lhs_info, rhs_info, mma):
    if lhs_element != rhs_element or lhs_element not in {"f16", "bf16"}:
        raise ValueError(
            "tlx_wave conversion supports tt.dot MFMA only for f16 x f16 or "
            "bf16 x bf16 operands in "
            f"lhs #ttg.dot_op<{{opIdx = {lhs_info.op_idx}}}> and "
            f"rhs #ttg.dot_op<{{opIdx = {rhs_info.op_idx}}}>, "
            f"got {lhs_element} x {rhs_element}"
        )
    return f"mfma.f32.{mma.kind_suffix}.{lhs_element}"


def _validate_dot_shapes(lhs_type, rhs_type, acc_type, result_type, mma):
    if len(lhs_type.shape) != 2 or len(rhs_type.shape) != 2:
        raise ValueError("tlx_wave conversion expected rank-2 tt.dot operands")
    if len(result_type.shape) != 2 or len(acc_type.shape) != 2:
        raise ValueError("tlx_wave conversion expected rank-2 tt.dot accumulator/result")
    if lhs_type.shape[1] != rhs_type.shape[0]:
        raise ValueError(
            "tlx_wave conversion expected tt.dot lhs K to match rhs K, got "
            f"{lhs_type.shape[1]} and {rhs_type.shape[0]}"
        )
    expected_result = (int(lhs_type.shape[0]), int(rhs_type.shape[1]))
    if tuple(result_type.shape) != expected_result:
        raise ValueError(
            "tlx_wave conversion expected tt.dot result shape "
            f"{expected_result}, got {result_type.shape}"
        )
    if tuple(acc_type.shape) != expected_result:
        raise ValueError(
            "tlx_wave conversion expected tt.dot accumulator shape "
            f"{expected_result}, got {acc_type.shape}"
        )
    if result_type.element_type != "f32" or acc_type.element_type != "f32":
        raise ValueError(
            "tlx_wave conversion supports tt.dot only with f32 accumulator/result"
        )
    _tile_rep_count(lhs_type.shape[0], mma.output_tile_shape[0])
    _tile_rep_count(lhs_type.shape[1], mma.k_dim)
    _tile_rep_count(rhs_type.shape[0], mma.k_dim)
    _tile_rep_count(rhs_type.shape[1], mma.output_tile_shape[1])
    _tile_rep_count(result_type.shape[0], mma.output_tile_shape[0])
    _tile_rep_count(result_type.shape[1], mma.output_tile_shape[1])


def _exact_div(lhs, rhs, context):
    lhs = int(lhs)
    rhs = int(rhs)
    if rhs == 0 or lhs % rhs:
        raise ValueError(
            f"tlx_wave conversion expected {context}={lhs} to be a multiple of {rhs}"
        )
    return lhs // rhs


def _require_operand_count(op, expected):
    if len(op.operands) != expected:
        raise ValueError(
            f"tlx_wave conversion expected {op.name} to have {expected} "
            f"operand(s), got {len(op.operands)}"
        )


def _require_result_count(op, expected):
    if len(op.results) != expected:
        raise ValueError(
            f"tlx_wave conversion expected {op.name} to have {expected} "
            f"result(s), got {len(op.results)}"
        )


def _required_int_attr(op, name):
    value = op.attrs.get(name)
    if value is None:
        raise ValueError(f"tlx_wave conversion expected {op.name} attr {name}")
    return int(value)


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


def _cmpi_predicate(op):
    raw_predicate = op.attrs.get("predicate")
    if raw_predicate is None:
        raise ValueError("tlx_wave conversion expected arith.cmpi predicate")
    if isinstance(raw_predicate, str):
        if raw_predicate in _CMPI_PREDICATES.values():
            return raw_predicate
        if raw_predicate.isdigit():
            raw_predicate = int(raw_predicate)
    predicate = _CMPI_PREDICATES.get(int(raw_predicate))
    if predicate is None:
        raise ValueError(
            f"tlx_wave conversion cannot lower arith.cmpi predicate {raw_predicate}"
        )
    return predicate


def _raw_constant_literal(op):
    attr = op.attrs.get("value")
    return None if attr is None else str(attr)


def _constant_literal(raw_literal, source_type):
    if raw_literal is None:
        return None
    text = raw_literal.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    if _is_integer_source_type(source_type):
        match = re.match(r"^[+-]?\d+", text)
        if match:
            return int(match.group(0))
    if _is_float_source_type(source_type):
        match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
        if match:
            return float(match.group(0))
    return raw_literal


def _is_integer_source_type(source_type):
    raw = source_type.element_type or source_type.raw
    return raw == "index" or raw.startswith("i")


def _is_float_source_type(source_type):
    raw = source_type.element_type or source_type.raw
    return raw in {"f16", "bf16", "f32", "f64"}


def _overflow_flags(op):
    for name in ("overflowFlags", "overflow_flags"):
        flags = op.attrs.get(name)
        if flags is None:
            continue
        text = str(flags)
        return tuple(part for part in re.split(r"[\s,|]+", text) if part)
    return ()
