"""Target-program verifier for the TLX Wave converter."""

from . import domains
from .diagnostics import fail


STAGE = "verification"

_PROOF_DEPENDENT_OPS = frozenset(
    {"assume", "buffer_load_to_local", "buffer_load", "buffer_store"}
)


def verify_target_program(
    target_program,
    *,
    source_program=None,
    fact_program=None,
    token_program=None,
):
    _verify_target_value_ids(target_program)
    _verify_ops(target_program, fact_program, source_program)
    if source_program is not None:
        _verify_source_results_covered(source_program, target_program)
    if token_program is not None and source_program is not None:
        _verify_memory_effects_tokenized(source_program, token_program)
    return True


def _verify_target_value_ids(target_program):
    for expected_id, value in enumerate(target_program.values):
        if value.target_value_id != expected_id:
            fail(
                "TLXW_VERIFY_VALUE_ID",
                STAGE,
                f"target value id {value.target_value_id} does not match "
                f"position {expected_id}",
                target_value_id=value.target_value_id,
            )


def _verify_ops(target_program, fact_program, source_program):
    value_count = len(target_program.values)
    op_count = len(target_program.ops)
    facts_by_id = _facts_by_id(fact_program)
    for expected_id, op in enumerate(target_program.ops):
        if op.target_op_id != expected_id:
            fail(
                "TLXW_VERIFY_OP_ID",
                STAGE,
                f"target op id {op.target_op_id} does not match position {expected_id}",
                target_op_id=op.target_op_id,
            )
        if op.kind not in domains.all_target_ops():
            fail(
                "TLXW_VERIFY_UNKNOWN_TARGET_OP",
                STAGE,
                f"target op {op.target_op_id} has unknown kind {op.kind}",
                target_op_id=op.target_op_id,
            )
        for target_value_id in (*op.operands, *op.results):
            if target_value_id < 0 or target_value_id >= value_count:
                fail(
                    "TLXW_VERIFY_UNKNOWN_TARGET_VALUE",
                    STAGE,
                    f"target op {op.target_op_id} references missing "
                    f"value {target_value_id}",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
        _verify_attrs(op)
        if len(op.fact_target_ids) != len(op.fact_ids):
            fail(
                "TLXW_VERIFY_FACT_TARGET_COUNT",
                STAGE,
                f"target op {op.target_op_id} has {len(op.fact_ids)} facts "
                f"but {len(op.fact_target_ids)} fact targets",
                target_op_id=op.target_op_id,
            )
        for target_value_id in op.fact_target_ids:
            if target_value_id < 0 or target_value_id >= value_count:
                fail(
                    "TLXW_VERIFY_UNKNOWN_FACT_TARGET",
                    STAGE,
                    f"target op {op.target_op_id} references missing fact "
                    f"target value {target_value_id}",
                    target_op_id=op.target_op_id,
                    target_value_id=target_value_id,
                )
        for fact_id in op.fact_ids:
            if fact_id not in facts_by_id:
                fail(
                    "TLXW_VERIFY_UNKNOWN_FACT",
                    STAGE,
                    f"target op {op.target_op_id} references missing fact {fact_id}",
                    target_op_id=op.target_op_id,
                    fact_id=fact_id,
                )
        for fact_id, target_value_id in zip(op.fact_ids, op.fact_target_ids):
            _verify_fact_target_compatible(
                target_program,
                op,
                facts_by_id[fact_id],
                target_value_id,
            )
        if op.kind == "layout_convert":
            _verify_layout_convert_fact_policy(op)
            if source_program is not None:
                _verify_layout_convert_source_op(op, source_program)
        if op.kind in _PROOF_DEPENDENT_OPS and not op.fact_ids:
            fail(
                "TLXW_VERIFY_MISSING_FACT",
                STAGE,
                f"target op {op.target_op_id} ({op.kind}) requires fact provenance",
                target_op_id=op.target_op_id,
            )
    _verify_region_op_ids(target_program, op_count)


def _verify_layout_convert_fact_policy(op):
    attrs = _attrs_dict(op)
    policy = attrs.get("fact_policy")
    if policy not in {"preserve_equivalent", "invalidate_layout_sensitive"}:
        fail(
            "TLXW_VERIFY_LAYOUT_FACT_POLICY",
            STAGE,
            "layout_convert requires an explicit fact_policy",
            target_op_id=op.target_op_id,
        )
    if policy == "invalidate_layout_sensitive" and op.fact_ids:
        fail(
            "TLXW_VERIFY_LAYOUT_FACT_POLICY",
            STAGE,
            "layout_convert that invalidates layout-sensitive facts must not "
            "carry fact ids",
            target_op_id=op.target_op_id,
        )


def _verify_layout_convert_source_op(op, source_program):
    if op.source_op_index is None:
        fail(
            "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE",
            STAGE,
            "layout_convert target op must come from source ttg.convert_layout",
            target_op_id=op.target_op_id,
        )
    try:
        source_op = source_program.ops[int(op.source_op_index)]
    except IndexError:
        fail(
            "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE",
            STAGE,
            "layout_convert target op references an unknown source op",
            target_op_id=op.target_op_id,
            source_op_index=op.source_op_index,
        )
    if source_op.name != "ttg.convert_layout":
        fail(
            "TLXW_VERIFY_LAYOUT_CONVERT_SOURCE",
            STAGE,
            "layout_convert target op must come from source ttg.convert_layout, "
            f"not {source_op.name}",
            target_op_id=op.target_op_id,
            source_op_index=op.source_op_index,
        )


def _verify_fact_target_compatible(target_program, op, fact, target_value_id):
    value = target_program.values[target_value_id]
    if value.source_value_id is not None:
        if value.source_value_id == fact.subject_value_id:
            return
        fail(
            "TLXW_VERIFY_FACT_TARGET",
            STAGE,
            f"fact {fact.fact_id} applies to source value "
            f"{fact.subject_value_id}, not target value {target_value_id}",
            target_op_id=op.target_op_id,
            target_value_id=target_value_id,
            fact_id=fact.fact_id,
        )
    source_targets = target_program.source_value_targets.get(fact.subject_value_id)
    if source_targets is None or target_value_id in source_targets:
        return
    fail(
        "TLXW_VERIFY_FACT_TARGET",
        STAGE,
        f"fact {fact.fact_id} target value {target_value_id} is not mapped "
        f"from source value {fact.subject_value_id}",
        target_op_id=op.target_op_id,
        target_value_id=target_value_id,
        fact_id=fact.fact_id,
    )


def _verify_attrs(op):
    names = set()
    for attr in op.attrs:
        if attr.name in names:
            fail(
                "TLXW_VERIFY_DUPLICATE_ATTR",
                STAGE,
                f"target op {op.target_op_id} has duplicate attr {attr.name}",
                target_op_id=op.target_op_id,
            )
        names.add(attr.name)
        if not _is_schema_value(attr.value):
            fail(
                "TLXW_VERIFY_NON_SCHEMA_ATTR",
                STAGE,
                f"target op {op.target_op_id} attr {attr.name} is not schema data",
                target_op_id=op.target_op_id,
            )


def _attrs_dict(op):
    return {attr.name: attr.value for attr in op.attrs}


def _verify_region_op_ids(target_program, op_count):
    for region in target_program.regions:
        for target_op_id in region.op_ids:
            if target_op_id < 0 or target_op_id >= op_count:
                fail(
                    "TLXW_VERIFY_UNKNOWN_REGION_OP",
                    STAGE,
                    f"target region {region.target_region_id} references "
                    f"missing op {target_op_id}",
                    target_op_id=target_op_id,
                )


def _verify_source_results_covered(source_program, target_program):
    erased = set(target_program.erased_source_values)
    for op in source_program.ops:
        for source_value_id in op.results:
            targets = target_program.source_value_targets.get(source_value_id, ())
            if source_value_id in erased:
                continue
            if len(targets) != 1:
                fail(
                    "TLXW_VERIFY_SOURCE_RESULT_COVERAGE",
                    STAGE,
                    f"source result {source_value_id} has {len(targets)} "
                    "target values",
                    source_op_index=op.index,
                    source_value_id=source_value_id,
                )


def _verify_memory_effects_tokenized(source_program, token_program):
    effect_op_indices = {effect.op_index for effect in token_program.memory_effects}
    for op in source_program.ops:
        if op.name in {
            "tt.load",
            "tt.store",
            "ttg.async_copy_global_to_local",
            "amdg.buffer_load",
            "amdg.buffer_load_to_local",
            "amdg.buffer_store",
            "ttg.local_load",
            "ttg.local_store",
        }:
            if op.index not in effect_op_indices:
                fail(
                    "TLXW_VERIFY_UNTOKENIZED_MEMORY_EFFECT",
                    STAGE,
                    f"memory op {op.name} has no memory effect",
                    source_op_index=op.index,
                )


def _facts_by_id(fact_program):
    if fact_program is None:
        return {}
    return {fact.fact_id: fact for fact in fact_program.facts}


def _is_schema_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        return all(_is_schema_value(item) for item in value)
    if isinstance(value, frozenset):
        return all(_is_schema_value(item) for item in value)
    return False
