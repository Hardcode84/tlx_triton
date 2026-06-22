"""Lowering-domain ownership for the staged TLX Wave converter."""

from dataclasses import dataclass


STAGE = "domains"


@dataclass(frozen=True)
class LoweringDomain:
    name: str
    source_ops: tuple[str, ...]
    target_ops: tuple[str, ...]


LOWERING_DOMAINS = (
    LoweringDomain(
        "arithmetic_control",
        (
            "arith.constant",
            "arith.addi",
            "arith.subi",
            "arith.muli",
            "arith.andi",
            "arith.ori",
            "arith.xori",
            "arith.divsi",
            "arith.divui",
            "arith.remsi",
            "arith.remui",
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.cmpi",
            "arith.minsi",
            "llvm.intr.assume",
            "tt.make_range",
            "tt.splat",
            "tt.addptr",
            "tt.broadcast",
            "tt.expand_dims",
            "tt.get_program_id",
            "rocdl.sched.barrier",
            "scf.for",
            "scf.if",
            "tt.return",
        ),
        (
            "constant",
            "binary",
            "float_binary",
            "cmpi",
            "minsi",
            "assume",
            "make_range",
            "splat",
            "broadcast",
            "addptr",
            "expand_dims",
            "program_id",
            "for_loop",
            "if",
            "select",
            "return",
        ),
    ),
    LoweringDomain(
        "memory_dma",
        (
            "amdg.buffer_load_to_local",
            "amdg.buffer_load",
            "ttg.async_commit_group",
            "ttg.async_wait",
        ),
        (
            "buffer_load_to_local",
            "buffer_load",
            "token",
            "async_commit_group",
            "async_wait",
        ),
    ),
    LoweringDomain(
        "generic_memory",
        (
            "tt.load",
            "tt.store",
        ),
        (
            "load",
            "store",
        ),
    ),
    LoweringDomain(
        "local_memory_layout",
        (
            "ttg.local_alloc",
            "ttg.memdesc_index",
            "ttg.local_load",
            "ttg.convert_layout",
        ),
        (
            "local_alloc",
            "memdesc_index",
            "local_load_fragment",
            "layout_convert",
        ),
    ),
    LoweringDomain(
        "mfma_fragment",
        (
            "arith.constant",
            "arith.truncf",
            "tt.dot",
        ),
        (
            "fragment_fill",
            "mma",
            "fragment_truncf",
        ),
    ),
    LoweringDomain(
        "store_epilogue",
        ("amdg.buffer_store",),
        ("buffer_store",),
    ),
)

DOMAIN_NAMES = tuple(domain.name for domain in LOWERING_DOMAINS)
_DOMAINS_BY_NAME = {domain.name: domain for domain in LOWERING_DOMAINS}


def source_domains_for_op(op_name):
    return tuple(
        domain.name for domain in LOWERING_DOMAINS if op_name in domain.source_ops
    )


def target_domain_for_op(op_kind):
    for domain in LOWERING_DOMAINS:
        if op_kind in domain.target_ops:
            return domain.name
    return None


def source_ops_for_domain(domain_name):
    return _DOMAINS_BY_NAME[domain_name].source_ops


def target_ops_for_domain(domain_name):
    return _DOMAINS_BY_NAME[domain_name].target_ops


def all_source_ops():
    return frozenset(
        op_name for domain in LOWERING_DOMAINS for op_name in domain.source_ops
    )


def all_target_ops():
    return frozenset(
        op_kind for domain in LOWERING_DOMAINS for op_kind in domain.target_ops
    )
