"""Fact analysis for the TLX Wave converter."""

from dataclasses import dataclass, field
import re


STAGE = "facts"


@dataclass(frozen=True)
class Fact:
    fact_id: int
    kind: str
    subject_value_id: int
    predicate: str
    lower: int | None = None
    upper: int | None = None
    width: int | None = None
    signedness: str | None = None
    provenance: str = ""
    source_op_index: int | None = None
    mask_scope: int | None = None


@dataclass(frozen=True)
class FactProgram:
    facts: tuple[Fact, ...]
    by_value: dict[int, tuple[int, ...]]
    tensor_affine: dict[int, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorAffineTerm:
    kind: str
    coefficient: int = 1
    dim: int | None = None
    scalar_value_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TensorAffine:
    value_id: int
    shape: tuple[int, ...]
    terms: tuple[TensorAffineTerm, ...]


def analyze_facts(source_program, type_layout_program):
    del type_layout_program
    facts = []
    _add_type_width_facts(source_program, facts)
    _add_pointer_range_facts(source_program, facts)
    _add_assume_facts(source_program, facts)
    tensor_affine = _analyze_tensor_affine(source_program)
    by_value = {}
    for fact in facts:
        by_value.setdefault(fact.subject_value_id, []).append(fact.fact_id)
    return FactProgram(
        tuple(facts),
        {value_id: tuple(fact_ids) for value_id, fact_ids in by_value.items()},
        tensor_affine,
    )


def facts_for_value(fact_program, value_id):
    return tuple(fact_program.facts[fact_id] for fact_id in fact_program.by_value.get(value_id, ()))


def _add_type_width_facts(source_program, facts):
    for value in source_program.values.values():
        width = _integer_width(value.type.raw)
        if width is None:
            continue
        lower = -(1 << (width - 1))
        upper = (1 << (width - 1)) - 1
        _append_fact(
            facts,
            "range",
            value.value_id,
            "signed_width",
            lower=lower,
            upper=upper,
            width=width,
            signedness="signed",
            provenance=f"type:{value.type.raw}",
            source_op_index=value.owner_op_index,
        )


def _add_pointer_range_facts(source_program, facts):
    for value in source_program.values.values():
        pointer_range = value.type.pointer_range
        if value.type.kind != "pointer" or pointer_range is None:
            continue
        pointer_range = int(pointer_range)
        if pointer_range <= 0 or pointer_range > 32:
            continue
        _append_fact(
            facts,
            "pointer_byte_range",
            value.value_id,
            "byte_interval",
            lower=0,
            upper=(1 << (pointer_range - 1)) - 1,
            width=32,
            signedness="signed",
            provenance="arg:tt.pointer_range",
        )


def _add_assume_facts(source_program, facts):
    op_by_result = {
        result_id: op
        for op in source_program.ops
        for result_id in op.results
    }
    for op in source_program.ops:
        if op.name != "llvm.intr.assume" or len(op.operands) != 1:
            continue
        fact = _assume_compare_fact(source_program, op_by_result, op.operands[0], op.index)
        if fact is None:
            continue
        _append_fact(facts, **fact)


def _analyze_tensor_affine(source_program):
    tensor_affine = {}
    scalar_constants = {}
    for op in source_program.ops:
        if op.name == "arith.constant":
            _record_constant_affine(source_program, op, tensor_affine, scalar_constants)
            continue
        if op.name == "tt.make_range":
            _record_make_range_affine(source_program, op, tensor_affine)
            continue
        if op.name == "tt.splat":
            _record_splat_affine(source_program, op, tensor_affine, scalar_constants)
            continue
        if op.name == "tt.expand_dims":
            _record_expand_dims_affine(source_program, op, tensor_affine)
            continue
        if op.name == "tt.broadcast":
            _record_broadcast_affine(source_program, op, tensor_affine)
            continue
        if op.name in {"arith.addi", "arith.muli"}:
            _record_binary_affine(source_program, op, tensor_affine, scalar_constants)
            continue
    return tensor_affine


def _record_constant_affine(source_program, op, tensor_affine, scalar_constants):
    if len(op.results) != 1:
        return
    value_id = op.results[0]
    value = _constant_literal(op.attrs.get("value"))
    if value is None:
        return
    source_type = source_program.values[value_id].type
    if source_type.kind == "scalar" and isinstance(value, int):
        scalar_constants[value_id] = value
        return
    if source_type.kind == "tensor" and isinstance(value, int):
        tensor_affine[value_id] = TensorAffine(
            value_id,
            tuple(source_type.shape),
            (TensorAffineTerm("const", value),),
        )


def _record_make_range_affine(source_program, op, tensor_affine):
    if len(op.results) != 1:
        return
    value_id = op.results[0]
    source_type = source_program.values[value_id].type
    if source_type.kind != "tensor" or len(source_type.shape) != 1:
        return
    start = int(op.attrs.get("start", 0))
    terms = []
    if start:
        terms.append(TensorAffineTerm("const", start))
    terms.append(TensorAffineTerm("dim", 1, 0))
    tensor_affine[value_id] = TensorAffine(value_id, tuple(source_type.shape), tuple(terms))


def _record_splat_affine(source_program, op, tensor_affine, scalar_constants):
    if len(op.operands) != 1 or len(op.results) != 1:
        return
    value_id = op.results[0]
    source_type = source_program.values[value_id].type
    if source_type.kind != "tensor":
        return
    operand = op.operands[0]
    constant = scalar_constants.get(operand)
    term = (
        TensorAffineTerm("const", constant)
        if constant is not None
        else TensorAffineTerm("scalar", 1, None, (operand,))
    )
    tensor_affine[value_id] = TensorAffine(value_id, tuple(source_type.shape), (term,))


def _record_expand_dims_affine(source_program, op, tensor_affine):
    if len(op.operands) != 1 or len(op.results) != 1:
        return
    source = tensor_affine.get(op.operands[0])
    if source is None:
        return
    value_id = op.results[0]
    result_type = source_program.values[value_id].type
    axis = int(op.attrs.get("axis", 0))
    terms = []
    for term in source.terms:
        if term.dim is None:
            terms.append(term)
            continue
        dim = int(term.dim)
        terms.append(
            TensorAffineTerm(
                term.kind,
                term.coefficient,
                dim if dim < axis else dim + 1,
                term.scalar_value_ids,
            )
        )
    tensor_affine[value_id] = TensorAffine(value_id, tuple(result_type.shape), tuple(terms))


def _record_broadcast_affine(source_program, op, tensor_affine):
    if len(op.operands) != 1 or len(op.results) != 1:
        return
    source = tensor_affine.get(op.operands[0])
    if source is None:
        return
    value_id = op.results[0]
    result_shape = tuple(source_program.values[value_id].type.shape)
    if len(source.shape) != len(result_shape):
        return
    tensor_affine[value_id] = TensorAffine(value_id, result_shape, source.terms)


def _record_binary_affine(source_program, op, tensor_affine, scalar_constants):
    if len(op.operands) != 2 or len(op.results) != 1:
        return
    result_type = source_program.values[op.results[0]].type
    if result_type.kind != "tensor":
        return
    lhs = tensor_affine.get(op.operands[0])
    rhs = tensor_affine.get(op.operands[1])
    if op.name == "arith.addi":
        terms = _add_affine_terms(
            lhs,
            rhs,
            op.operands,
            scalar_constants,
        )
    else:
        terms = _mul_affine_terms(lhs, rhs)
    if terms is None:
        return
    tensor_affine[op.results[0]] = TensorAffine(
        op.results[0],
        tuple(result_type.shape),
        _canonical_terms(terms),
    )


def _add_affine_terms(lhs, rhs, operands, scalar_constants):
    terms = []
    if lhs is not None:
        terms.extend(lhs.terms)
    else:
        term = _scalar_operand_term(operands[0], scalar_constants)
        if term is None:
            return None
        terms.append(term)
    if rhs is not None:
        terms.extend(rhs.terms)
    else:
        term = _scalar_operand_term(operands[1], scalar_constants)
        if term is None:
            return None
        terms.append(term)
    return terms


def _mul_affine_terms(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    lhs_uniform = _uniform_affine_term(lhs)
    rhs_uniform = _uniform_affine_term(rhs)
    if lhs_uniform is not None:
        return _scale_affine_terms(rhs, lhs_uniform)
    if rhs_uniform is not None:
        return _scale_affine_terms(lhs, rhs_uniform)
    return None


def _scalar_operand_term(value_id, scalar_constants):
    constant = scalar_constants.get(value_id)
    if constant is not None:
        return TensorAffineTerm("const", constant)
    return TensorAffineTerm("scalar", 1, None, (value_id,))


def _uniform_affine_term(affine):
    if len(affine.terms) != 1:
        return None
    term = affine.terms[0]
    return term if term.dim is None else None


def _scale_affine_terms(affine, factor):
    result = []
    for term in affine.terms:
        scaled = _scale_affine_term(term, factor)
        if scaled is None:
            return None
        result.append(scaled)
    return result


def _scale_affine_term(term, factor):
    if factor.kind == "const":
        return TensorAffineTerm(
            term.kind,
            term.coefficient * factor.coefficient,
            term.dim,
            term.scalar_value_ids,
        )
    if factor.kind != "scalar":
        return None
    scalar = factor.scalar_value_ids[0]
    if term.kind == "const":
        return TensorAffineTerm("scalar", term.coefficient * factor.coefficient, None, (scalar,))
    if term.kind == "dim":
        return TensorAffineTerm("dim_scalar", term.coefficient * factor.coefficient, term.dim, (scalar,))
    if term.kind == "scalar":
        ids = (*term.scalar_value_ids, scalar)
        return TensorAffineTerm("scalar_product", term.coefficient * factor.coefficient, None, ids)
    return None


def _canonical_terms(terms):
    merged = {}
    for term in terms:
        key = (term.kind, term.dim, term.scalar_value_ids)
        merged[key] = merged.get(key, 0) + int(term.coefficient)
    result = []
    for (kind, dim, scalar_value_ids), coefficient in merged.items():
        if coefficient:
            result.append(TensorAffineTerm(kind, coefficient, dim, scalar_value_ids))
    result.sort(key=lambda term: (term.kind, -1 if term.dim is None else term.dim, term.scalar_value_ids))
    return tuple(result)


def _assume_compare_fact(source_program, op_by_result, predicate_id, assume_op_index):
    compare = op_by_result.get(predicate_id)
    if compare is None or compare.name != "arith.cmpi" or len(compare.operands) != 2:
        return None
    predicate = _cmpi_predicate(compare)
    lhs_id, rhs_id = compare.operands
    lhs_const = _constant_int(source_program, op_by_result, lhs_id)
    rhs_const = _constant_int(source_program, op_by_result, rhs_id)
    if rhs_const is not None and lhs_const is None:
        return _compare_bound_fact(
            source_program,
            lhs_id,
            predicate,
            rhs_const,
            assume_op_index,
        )
    if lhs_const is not None and rhs_const is None:
        inverted = _invert_predicate(predicate)
        if inverted is None:
            return None
        return _compare_bound_fact(
            source_program,
            rhs_id,
            inverted,
            lhs_const,
            assume_op_index,
        )
    return None


def _compare_bound_fact(source_program, value_id, predicate, constant, assume_op_index):
    width = _integer_width(source_program.values[value_id].type.raw)
    if predicate == "eq":
        lower = upper = int(constant)
    elif predicate == "sge":
        lower, upper = int(constant), None
    elif predicate == "sgt":
        lower, upper = int(constant) + 1, None
    elif predicate == "sle":
        lower, upper = None, int(constant)
    elif predicate == "slt":
        lower, upper = None, int(constant) - 1
    else:
        return None
    return {
        "kind": "range",
        "subject_value_id": value_id,
        "predicate": predicate,
        "lower": lower,
        "upper": upper,
        "width": width,
        "signedness": "signed",
        "provenance": "llvm.intr.assume",
        "source_op_index": assume_op_index,
    }


def _constant_int(source_program, op_by_result, value_id):
    op = op_by_result.get(value_id)
    if op is None or op.name != "arith.constant":
        return None
    attr = op.attrs.get("value")
    if attr is None:
        return None
    text = str(attr).strip()
    if text in {"true", "false"}:
        return None
    match = re.match(r"([+-]?\d+)", text)
    if match is None:
        return None
    return int(match.group(1), 0)


def _constant_literal(value):
    if value is None:
        return None
    text = str(value).strip()
    dense = re.fullmatch(r"dense<([^>]*)>\s*(?::.*)?", text, re.DOTALL)
    if dense is not None:
        text = dense.group(1).strip()
    else:
        text = text.split(":", 1)[0].strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return None
    match = re.fullmatch(r"([+-]?(?:0[xX][0-9a-fA-F]+|\d+))", text)
    return None if match is None else int(match.group(1), 0)


def _append_fact(
    facts,
    kind,
    subject_value_id,
    predicate,
    *,
    lower=None,
    upper=None,
    width=None,
    signedness=None,
    provenance="",
    source_op_index=None,
    mask_scope=None,
):
    facts.append(
        Fact(
            len(facts),
            kind,
            subject_value_id,
            predicate,
            lower,
            upper,
            width,
            signedness,
            provenance,
            source_op_index,
            mask_scope,
        )
    )


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
        return None
    if isinstance(raw_predicate, str):
        if raw_predicate in _CMPI_PREDICATES.values():
            return raw_predicate
        if raw_predicate.isdigit():
            raw_predicate = int(raw_predicate)
    return _CMPI_PREDICATES.get(int(raw_predicate))


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


def _integer_width(raw_type):
    if raw_type == "index":
        return None
    match = re.fullmatch(r"i([0-9]+)", str(raw_type))
    if match is None:
        return None
    width = int(match.group(1))
    return width if width > 1 else None
