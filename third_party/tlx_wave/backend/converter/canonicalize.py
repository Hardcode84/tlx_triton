"""Target-program canonicalization for the TLX Wave converter."""

from . import target_ir


def canonicalize_target_program(target_program):
    if len(target_program.regions) != 1:
        return target_program
    return _share_div_rem_pairs(target_program)


def _share_div_rem_pairs(target_program):
    divs_by_key = {}
    for op in target_program.ops:
        operation = _binary_operation(op)
        if operation in {"divsi", "divui"}:
            divs_by_key.setdefault((_div_rem_flavor(operation), op.operands), op)

    if not any(
        _binary_operation(op) in {"remsi", "remui"}
        and (_div_rem_flavor(_binary_operation(op)), op.operands) in divs_by_key
        for op in target_program.ops
    ):
        return target_program

    values = list(target_program.values)
    ops = []
    skipped_op_ids = set()
    emitted_div_ids = set()

    def append_op(op, **updates):
        new_op = target_ir.TargetOp(
            len(ops),
            updates.get("kind", op.kind),
            tuple(updates.get("operands", op.operands)),
            tuple(updates.get("results", op.results)),
            _attrs_tuple(updates.get("attrs", target_ir.attrs_dict(op))),
            tuple(updates.get("fact_ids", op.fact_ids)),
            tuple(updates.get("fact_target_ids", op.fact_target_ids)),
            tuple(updates.get("layout_map_ids", op.layout_map_ids)),
            tuple(updates.get("region_ids", op.region_ids)),
            updates.get("source_op_index", op.source_op_index),
        )
        ops.append(new_op)
        return new_op

    for op in target_program.ops:
        if op.target_op_id in skipped_op_ids:
            continue

        operation = _binary_operation(op)
        if operation not in {"remsi", "remui"}:
            append_op(op)
            if operation in {"divsi", "divui"}:
                emitted_div_ids.add(op.target_op_id)
            continue

        div_op = divs_by_key.get((_div_rem_flavor(operation), op.operands))
        if div_op is None:
            append_op(op)
            continue

        if div_op.target_op_id not in emitted_div_ids:
            append_op(div_op)
            emitted_div_ids.add(div_op.target_op_id)
            skipped_op_ids.add(div_op.target_op_id)

        lhs, rhs = op.operands
        rem_result = _single_result(op)
        product_value_id = len(values)
        values.append(
            target_ir.TargetValue(
                product_value_id,
                target_program.values[rem_result].type,
                debug_name=f"rem_product_{rem_result}",
            )
        )
        attrs = target_ir.attrs_dict(op)
        source_width = attrs.get("source_width")
        binary_attrs = {}
        if source_width is not None:
            binary_attrs["source_width"] = source_width
        append_op(
            op,
            operands=(div_op.results[0], rhs),
            results=(product_value_id,),
            attrs={**binary_attrs, "operation": "muli"},
            fact_ids=(),
            fact_target_ids=(),
            layout_map_ids=(),
        )
        append_op(
            op,
            operands=(lhs, product_value_id),
            attrs={**binary_attrs, "operation": "subi"},
        )

    return target_ir.TargetProgram(
        tuple(values),
        tuple(ops),
        _renumber_regions(target_program, ops),
        dict(target_program.source_value_targets),
        dict(target_program.erased_source_values),
        target_program.kernel,
    )


def _renumber_regions(target_program, ops):
    if not target_program.regions:
        return ()
    regions = list(target_program.regions)
    first = regions[0]
    regions[0] = target_ir.TargetRegion(
        first.target_region_id,
        tuple(op.target_op_id for op in ops),
        first.block_arg_ids,
        first.yield_value_ids,
    )
    return tuple(regions)


def _binary_operation(op):
    if op.kind != "binary":
        return None
    return target_ir.attrs_dict(op).get("operation")


def _div_rem_flavor(operation):
    if operation.endswith("si"):
        return "si"
    if operation.endswith("ui"):
        return "ui"
    return None


def _single_result(op):
    if len(op.results) != 1:
        raise AssertionError("binary target ops must have one result")
    return op.results[0]


def _attrs_tuple(attrs):
    return tuple(
        target_ir.TargetAttr(str(name), value) for name, value in sorted(attrs.items())
    )
