"""Token and memory-effect graph for the TLX Wave converter."""

from dataclasses import dataclass

from .diagnostics import fail

STAGE = "tokens"

_ASYNC_COPY_OPS = frozenset({"ttg.async_copy_global_to_local", "amdg.buffer_load_to_local"})
_TOKEN_CONTROL_OPS = frozenset({"ttg.async_commit_group", "ttg.async_wait"})
_TOKEN_OPS = _ASYNC_COPY_OPS | _TOKEN_CONTROL_OPS
_MEMORY_OPS = _ASYNC_COPY_OPS | frozenset({
    "amdg.buffer_load",
    "amdg.buffer_store",
    "tt.load",
    "tt.store",
    "ttg.local_load",
    "ttg.local_store",
})


@dataclass(frozen=True)
class TokenNode:
    node_id: int
    op_index: int
    op_name: str
    value_id: int | None
    input_token_ids: tuple[int, ...]
    source_address_value_id: int | None = None
    source_offset_value_id: int | None = None
    memdesc_value_id: int | None = None
    mask_value_id: int | None = None
    other_value_id: int | None = None
    wait_group: int | None = None
    committed_group_id: int | None = None
    waited_group_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenGroup:
    group_id: int
    commit_op_index: int
    token_value_id: int | None
    member_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LoopTokenCarry:
    loop_op_index: int
    init_source_value_id: int | None
    yield_source_value_id: int
    add_issue_dependency: bool
    issue_dependency_op_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryEffect:
    effect_id: int
    op_index: int
    op_name: str
    kind: str
    address_space: str
    address_value_id: int | None = None
    offset_value_id: int | None = None
    value_value_id: int | None = None
    mask_value_id: int | None = None
    token_node_id: int | None = None
    cache_modifier: str | None = None
    volatile: bool = False
    ordering: str | None = None
    sync_scope: str | None = None
    alias_class: str = "unknown"
    depends_on_effect_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TokenProgram:
    nodes: tuple[TokenNode, ...]
    groups: tuple[TokenGroup, ...]
    memory_effects: tuple[MemoryEffect, ...]
    node_ids_by_value_id: dict[int, int]
    users_by_value_id: dict[int, tuple[int, ...]]
    loop_token_carries_by_op: dict[int, tuple[LoopTokenCarry, ...]]

    def node_for_value(self, value_id):
        node_id = self.node_ids_by_value_id.get(value_id)
        return None if node_id is None else self.nodes[node_id]

    def users_for_value(self, value_id):
        return tuple(self.nodes[node_id] for node_id in self.users_by_value_id.get(value_id, ()))


def build_token_program(source_program, type_layout_program):
    del type_layout_program
    nodes = []
    groups = []
    memory_effects = []
    node_ids_by_value = {}
    users_by_value = {}
    open_async_tokens = []
    committed_groups = []
    dependency_frontier = _DependencyFrontier()

    for op in source_program.ops:
        token_node_id = None
        if _needs_token_node(source_program, op):
            node, group, open_async_tokens, committed_groups = _build_token_node(
                source_program,
                op,
                len(nodes),
                len(groups),
                tuple(open_async_tokens),
                tuple(committed_groups),
            )
            nodes.append(node)
            token_node_id = node.node_id
            if node.value_id is not None:
                node_ids_by_value[node.value_id] = node.node_id
            if group is not None:
                groups.append(group)
            for input_token_id in node.input_token_ids:
                _append_user(users_by_value, input_token_id, node.node_id)
            if group is not None:
                for member_token_id in group.member_token_ids:
                    _append_user(users_by_value, member_token_id, node.node_id)
            if op.name == "ttg.async_wait":
                for group_id in node.waited_group_ids:
                    waited_group = groups[group_id]
                    if waited_group.token_value_id is not None:
                        _append_user(
                            users_by_value,
                            waited_group.token_value_id,
                            node.node_id,
                        )
                    for member_token_id in waited_group.member_token_ids:
                        _append_user(users_by_value, member_token_id, node.node_id)

        if op.name in _MEMORY_OPS:
            memory_effects.extend(
                _memory_effects_for_op(
                    source_program,
                    op,
                    token_node_id,
                    len(memory_effects),
                    dependency_frontier,
                ))

    nodes = tuple(nodes)
    groups = tuple(groups)
    return TokenProgram(
        nodes,
        groups,
        tuple(memory_effects),
        node_ids_by_value,
        {value_id: tuple(node_ids)
         for value_id, node_ids in users_by_value.items()},
        _loop_token_carries_by_op(source_program, nodes, groups),
    )


def _loop_token_carries_by_op(source_program, nodes, groups):
    groups_by_id = {group.group_id: group for group in groups}
    groups_by_value_id = {group.token_value_id: group for group in groups if group.token_value_id is not None}
    nodes_by_value_id = {node.value_id: node for node in nodes if node.value_id is not None}
    carries_by_op = {}
    for op in source_program.ops:
        if op.name != "scf.for" or len(op.region_ids) != 1:
            continue
        body_op_indices = _region_op_indices_recursive(source_program, op.region_ids[0])
        carries = _loop_token_carries_for_body(
            op,
            body_op_indices,
            nodes,
            groups,
            groups_by_id,
            groups_by_value_id,
            nodes_by_value_id,
        )
        if carries:
            carries_by_op[op.index] = carries
    return carries_by_op


def _loop_token_carries_for_body(
    op,
    body_op_indices,
    nodes,
    groups,
    groups_by_id,
    groups_by_value_id,
    nodes_by_value_id,
):
    external_wait_issue_pairs = _loop_external_wait_issue_pairs(
        nodes,
        groups,
        groups_by_id,
        body_op_indices,
    )
    waited_external_tokens = _dedupe_preserving_order(
        init_token_id for init_token_id, _issue_token_id in external_wait_issue_pairs)
    externally_waited_body_tokens = _externally_waited_body_tokens(
        nodes,
        groups_by_id,
        body_op_indices,
    )
    if waited_external_tokens:
        issue_tokens_by_init = {
            init_token_id: issue_token_id
            for init_token_id, issue_token_id in external_wait_issue_pairs
        }
        if (len(externally_waited_body_tokens) != len(waited_external_tokens)
                or len(issue_tokens_by_init) != len(waited_external_tokens)):
            fail(
                "TLXW_OP_UNSUPPORTED_FOR_TOKENS",
                STAGE,
                "scf.for async token carry requires each externally waited group "
                "to map to one loop-exit body group and one subsequent body issue",
                source_op_index=op.index,
            )
        return tuple(
            LoopTokenCarry(
                loop_op_index=op.index,
                init_source_value_id=init_source_value_id,
                yield_source_value_id=yield_source_value_id,
                add_issue_dependency=True,
                issue_dependency_op_indices=_group_issue_dependency_op_indices(
                    groups_by_value_id,
                    nodes_by_value_id,
                    issue_tokens_by_init[init_source_value_id],
                ),
            ) for init_source_value_id, yield_source_value_id in zip(
                waited_external_tokens,
                externally_waited_body_tokens,
            ))
    return tuple(LoopTokenCarry(
        op.index,
        None,
        body_token,
        False,
    ) for body_token in externally_waited_body_tokens)


def _loop_external_wait_issue_pairs(nodes, groups, groups_by_id, body_op_indices):
    body_groups = tuple(group for group in sorted(groups, key=lambda group: group.commit_op_index)
                        if group.commit_op_index in body_op_indices and group.token_value_id is not None)
    body_groups_after_index = 0
    assigned_body_tokens = set()
    committed_queue = []
    groups_by_commit = {group.commit_op_index: group for group in groups}
    nodes_by_op = {node.op_index: node for node in nodes}
    token_op_indices = sorted(set(groups_by_commit) | set(nodes_by_op))
    pairs = []
    for op_index in token_op_indices:
        group = groups_by_commit.get(op_index)
        if group is not None:
            committed_queue.append(group)
        node = nodes_by_op.get(op_index)
        if node is None or node.op_name != "ttg.async_wait":
            continue
        external_waited_tokens = []
        for group_id in node.waited_group_ids:
            waited_group = groups_by_id[group_id]
            if (node.op_index not in body_op_indices or waited_group.commit_op_index in body_op_indices
                    or waited_group.token_value_id is None):
                continue
            external_waited_tokens.append(waited_group.token_value_id)
        for init_token_id in external_waited_tokens:
            issue_group = next(
                (queued_group for queued_group in committed_queue
                 if queued_group.commit_op_index in body_op_indices and queued_group.token_value_id is not None
                 and queued_group.token_value_id not in assigned_body_tokens),
                None,
            )
            while issue_group is None and body_groups_after_index < len(body_groups):
                candidate = body_groups[body_groups_after_index]
                body_groups_after_index += 1
                if (candidate.commit_op_index <= node.op_index or candidate.token_value_id in assigned_body_tokens):
                    continue
                issue_group = candidate
            if issue_group is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_FOR_TOKENS",
                    STAGE,
                    "scf.for async token carry could not find a body issue "
                    "for an external wait",
                    source_op_index=node.op_index,
                )
            pairs.append((init_token_id, issue_group.token_value_id))
            assigned_body_tokens.add(issue_group.token_value_id)
        if node.waited_group_ids:
            waited_group_ids = set(node.waited_group_ids)
            committed_queue = [
                queued_group for queued_group in committed_queue if queued_group.group_id not in waited_group_ids
            ]
    return tuple(pairs)


def _externally_waited_body_tokens(nodes, groups_by_id, body_op_indices):
    body_tokens = []
    for node in sorted(nodes, key=lambda node: node.op_index):
        if node.op_index in body_op_indices or node.op_name != "ttg.async_wait":
            continue
        for group_id in node.waited_group_ids:
            group = groups_by_id[group_id]
            if group.commit_op_index not in body_op_indices or group.token_value_id is None:
                continue
            body_tokens.append(group.token_value_id)
    return _dedupe_preserving_order(body_tokens)


def _group_issue_dependency_op_indices(
    groups_by_value_id,
    nodes_by_value_id,
    token_value_id,
):
    group = groups_by_value_id.get(token_value_id)
    if group is None:
        return ()
    return _dedupe_preserving_order(node.op_index
                                    for member_token_id in group.member_token_ids
                                    for node in (nodes_by_value_id.get(member_token_id), )
                                    if node is not None)


def _region_op_indices_recursive(source_program, region_id):
    result = []
    for op_index in source_program.regions[region_id].op_indices:
        result.append(op_index)
        for child_region_id in source_program.ops[op_index].region_ids:
            result.extend(_region_op_indices_recursive(source_program, child_region_id))
    return frozenset(result)


def _dedupe_preserving_order(values):
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _needs_token_node(source_program, op):
    return (op.name in _TOKEN_OPS or _first_token_result(source_program, op) is not None
            or bool(_input_token_ids(source_program, op)))


def _build_token_node(
    source_program,
    op,
    node_id,
    next_group_id,
    open_async_tokens,
    committed_groups,
):
    if op.name in {"ttg.async_commit_group", "ttg.async_wait"}:
        _require_token_operands(source_program, op)

    value_id = _first_token_result(source_program, op)
    input_token_ids = _input_token_ids(source_program, op)
    token_fields = _token_fields(source_program, op)
    group = None
    committed_group_id = None
    waited_group_ids = ()
    next_open_async_tokens = list(open_async_tokens)
    next_committed_groups = list(committed_groups)

    if op.name in _ASYNC_COPY_OPS:
        if value_id is None:
            fail(
                "TLXW_TOKEN_MISSING_RESULT",
                STAGE,
                f"{op.name} must produce an async token",
                source_op_index=op.index,
            )
        next_open_async_tokens.append(value_id)

    if op.name == "ttg.async_commit_group":
        if input_token_ids:
            member_token_ids = input_token_ids
            committed = set(member_token_ids)
            next_open_async_tokens = [token_id for token_id in next_open_async_tokens if token_id not in committed]
        else:
            member_token_ids = tuple(next_open_async_tokens)
            next_open_async_tokens = []
        committed_group_id = next_group_id
        group = TokenGroup(
            committed_group_id,
            op.index,
            value_id,
            tuple(member_token_ids),
        )
        next_committed_groups.append(group)

    if op.name == "ttg.async_wait":
        wait_group = token_fields["wait_group"]
        if wait_group is not None and wait_group < 0:
            fail(
                "TLXW_TOKEN_MALFORMED_WAIT",
                STAGE,
                f"ttg.async_wait requires a nonnegative wait group, got {wait_group}",
                source_op_index=op.index,
            )
        if not input_token_ids:
            waited_group_ids = _waited_group_ids(next_committed_groups, wait_group)
            if waited_group_ids:
                waited = set(waited_group_ids)
                next_committed_groups = [group for group in next_committed_groups if group.group_id not in waited]

    node = TokenNode(
        node_id,
        op.index,
        op.name,
        value_id,
        tuple(input_token_ids),
        committed_group_id=committed_group_id,
        waited_group_ids=tuple(waited_group_ids),
        **token_fields,
    )
    return node, group, next_open_async_tokens, next_committed_groups


def _token_fields(source_program, op):
    if op.name == "ttg.async_copy_global_to_local":
        return _global_async_copy_fields(op)
    if op.name == "amdg.buffer_load_to_local":
        return _buffer_async_copy_fields(op)
    if op.name == "ttg.async_wait":
        return {
            "source_address_value_id": None,
            "source_offset_value_id": None,
            "memdesc_value_id": None,
            "mask_value_id": None,
            "other_value_id": None,
            "wait_group": _int_attr(op, "num"),
        }
    del source_program
    return {
        "source_address_value_id": None,
        "source_offset_value_id": None,
        "memdesc_value_id": None,
        "mask_value_id": None,
        "other_value_id": None,
        "wait_group": None,
    }


def _global_async_copy_fields(op):
    segments = _operand_segments(op, 4, (1, 1, 1 if len(op.operands) > 2 else 0, 0))
    _require_operand_count(op, segments)
    mask_index = int(segments[0]) + int(segments[1])
    other_index = mask_index + int(segments[2])
    return {
        "source_address_value_id": _operand_or_none(op, 0),
        "source_offset_value_id": None,
        "memdesc_value_id": _operand_or_none(op, 1),
        "mask_value_id": (_operand_or_none(op, mask_index) if int(segments[2]) else None),
        "other_value_id": (_operand_or_none(op, other_index) if int(segments[3]) else None),
        "wait_group": None,
    }


def _buffer_async_copy_fields(op):
    segments = _operand_segments(
        op,
        6,
        (1, 1, 1, 1 if len(op.operands) > 3 else 0, 0, 0),
    )
    _require_operand_count(op, segments)
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    return {
        "source_address_value_id": _operand_or_none(op, base_index),
        "source_offset_value_id": _operand_or_none(op, offset_index),
        "memdesc_value_id": _operand_or_none(op, 0),
        "mask_value_id": (_operand_or_none(op, mask_index) if int(segments[3]) else None),
        "other_value_id": (_operand_or_none(op, other_index) if int(segments[4]) else None),
        "wait_group": None,
    }


class _DependencyFrontier:

    def __init__(self):
        self._last_writes_by_domain = {}
        self._reads_since_write_by_domain = {}

    def dependencies_for(
        self,
        *,
        kind,
        address_space,
        volatile=False,
        ordering=None,
        sync_scope=None,
    ):
        domains = _alias_domains_for_query(address_space, self._known_domains())
        if _effect_is_barrier_like(volatile, ordering, sync_scope):
            return _dedupe_effect_ids(effect_id for domain in domains for effect_id in (
                *self._last_writes_by_domain.get(domain, ()),
                *self._reads_since_write_by_domain.get(domain, ()),
            ))
        if kind == "read":
            return _dedupe_effect_ids(effect_id for domain in domains
                                      for effect_id in self._last_writes_by_domain.get(domain, ()))
        if kind == "write":
            return _dedupe_effect_ids(effect_id for domain in domains for effect_id in (
                *self._last_writes_by_domain.get(domain, ()),
                *self._reads_since_write_by_domain.get(domain, ()),
            ))
        return ()

    def record(self, effect):
        domain = _alias_domain(effect.address_space)
        if _effect_is_barrier_like(
                effect.volatile,
                effect.ordering,
                effect.sync_scope,
        ) or effect.kind == "write":
            if domain == "unknown":
                self._last_writes_by_domain.clear()
                self._reads_since_write_by_domain.clear()
            self._last_writes_by_domain[domain] = (effect.effect_id, )
            self._reads_since_write_by_domain[domain] = ()
            return
        if effect.kind == "read":
            self._reads_since_write_by_domain[domain] = (
                *self._reads_since_write_by_domain.get(domain, ()),
                effect.effect_id,
            )

    def _known_domains(self):
        return frozenset((
            *self._last_writes_by_domain.keys(),
            *self._reads_since_write_by_domain.keys(),
        ))


def _memory_effects_for_op(
    source_program,
    op,
    token_node_id,
    next_effect_id,
    dependency_frontier,
):
    fields = _token_fields(source_program, op) if op.name in _ASYNC_COPY_OPS else {}
    if op.name == "ttg.async_copy_global_to_local":
        return _effect_pair(
            source_program,
            op,
            token_node_id,
            fields["source_address_value_id"],
            None,
            fields["memdesc_value_id"],
            fields["mask_value_id"],
            next_effect_id,
            dependency_frontier,
            read_space="global",
        )
    if op.name == "amdg.buffer_load_to_local":
        return _effect_pair(
            source_program,
            op,
            token_node_id,
            fields["source_address_value_id"],
            fields["source_offset_value_id"],
            fields["memdesc_value_id"],
            fields["mask_value_id"],
            next_effect_id,
            dependency_frontier,
            read_space="buffer",
        )
    if op.name == "tt.load":
        mask_value_id = _operand_or_none(op, 1) if len(op.operands) > 1 else None
        return (_memory_effect(
            source_program,
            op,
            "read",
            _pointer_address_space(source_program, _operand_or_none(op, 0)),
            _operand_or_none(op, 0),
            None,
            None,
            mask_value_id,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "tt.store":
        mask_value_id = _operand_or_none(op, 2) if len(op.operands) > 2 else None
        return (_memory_effect(
            source_program,
            op,
            "write",
            _pointer_address_space(source_program, _operand_or_none(op, 0)),
            _operand_or_none(op, 0),
            None,
            _operand_or_none(op, 1),
            mask_value_id,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "ttg.local_load":
        return (_memory_effect(
            source_program,
            op,
            "read",
            "local",
            _operand_or_none(op, 0),
            None,
            _operand_or_none(op, 1) if len(op.operands) > 1 else None,
            None,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "ttg.local_store":
        return (_memory_effect(
            source_program,
            op,
            "write",
            "local",
            _operand_or_none(op, 1),
            None,
            _operand_or_none(op, 0),
            None,
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "amdg.buffer_load":
        fields = _buffer_load_fields(op)
        return (_memory_effect(
            source_program,
            op,
            "read",
            "buffer",
            fields["base_value_id"],
            fields["offset_value_id"],
            None,
            fields["mask_value_id"],
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    if op.name == "amdg.buffer_store":
        fields = _buffer_store_fields(op)
        return (_memory_effect(
            source_program,
            op,
            "write",
            "buffer",
            fields["base_value_id"],
            fields["offset_value_id"],
            fields["value_value_id"],
            fields["mask_value_id"],
            token_node_id,
            next_effect_id,
            dependency_frontier,
        ), )
    return ()


def _effect_pair(
    source_program,
    op,
    token_node_id,
    source_address_value_id,
    source_offset_value_id,
    memdesc_value_id,
    mask_value_id,
    next_effect_id,
    dependency_frontier,
    *,
    read_space,
):
    read = _memory_effect(
        source_program,
        op,
        "read",
        read_space,
        source_address_value_id,
        source_offset_value_id,
        None,
        mask_value_id,
        token_node_id,
        next_effect_id,
        dependency_frontier,
    )
    write = _memory_effect(
        source_program,
        op,
        "write",
        "local",
        memdesc_value_id,
        None,
        None,
        mask_value_id,
        token_node_id,
        next_effect_id + 1,
        dependency_frontier,
        explicit_dependency_ids=(read.effect_id, ),
    )
    return read, write


def _memory_effect(
        source_program,
        op,
        kind,
        address_space,
        address_value_id,
        offset_value_id,
        value_value_id,
        mask_value_id,
        token_node_id,
        effect_id,
        dependency_frontier,
        explicit_dependency_ids=(),
):
    del source_program
    volatile = bool(op.attrs.get("volatile", False))
    ordering = _attr_or_none(op, "ordering")
    sync_scope = _attr_or_none(op, "syncscope")
    effect = MemoryEffect(
        effect_id,
        op.index,
        op.name,
        kind,
        address_space,
        address_value_id,
        offset_value_id,
        value_value_id,
        mask_value_id,
        token_node_id,
        _cache_modifier(op),
        volatile,
        ordering,
        sync_scope,
        "unknown",
        _dedupe_effect_ids((
            *dependency_frontier.dependencies_for(
                kind=kind,
                address_space=address_space,
                volatile=volatile,
                ordering=ordering,
                sync_scope=sync_scope,
            ),
            *explicit_dependency_ids,
        )),
    )
    dependency_frontier.record(effect)
    return effect


def _alias_domain(address_space):
    if address_space in {"global", "buffer"}:
        return "global"
    if address_space == "local":
        return "local"
    return "unknown"


def _alias_domains_for_query(address_space, known_domains):
    domain = _alias_domain(address_space)
    if domain == "unknown":
        return tuple(sorted(known_domains | {"global", "local", "unknown"}))
    return (domain, "unknown")


def _effect_is_barrier_like(volatile, ordering, sync_scope):
    return bool(volatile or ordering or sync_scope)


def _dedupe_effect_ids(effect_ids):
    result = []
    seen = set()
    for effect_id in effect_ids:
        effect_id = int(effect_id)
        if effect_id in seen:
            continue
        seen.add(effect_id)
        result.append(effect_id)
    return tuple(sorted(result))


def _buffer_load_fields(op):
    segments = _operand_segments(op, 5, None)
    _require_operand_count(op, segments)
    if segments[0] != 1 or segments[1] != 1:
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_load requires base pointer and offsets operands",
            source_op_index=op.index,
        )
    if segments[2] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_load supports at most one stride operand",
            source_op_index=op.index,
        )
    if segments[3] not in (0, 1) or segments[4] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_load supports at most one mask and one other operand",
            source_op_index=op.index,
        )
    offset_index = int(segments[0])
    stride_index = offset_index + int(segments[1])
    mask_index = stride_index + int(segments[2])
    other_index = mask_index + int(segments[3])
    return {
        "base_value_id": _operand_or_none(op, 0),
        "offset_value_id": _operand_or_none(op, offset_index),
        "stride_value_id": _operand_or_none(op, stride_index) if segments[2] else None,
        "mask_value_id": _operand_or_none(op, mask_index) if segments[3] else None,
        "other_value_id": _operand_or_none(op, other_index) if segments[4] else None,
    }


def _waited_group_ids(committed_groups, keep_count):
    keep_count = max(0, int(keep_count or 0))
    wait_count = max(0, len(committed_groups) - keep_count)
    return tuple(group.group_id for group in committed_groups[:wait_count])


def _first_token_result(source_program, op):
    for value_id in op.results:
        if _value_is_token(source_program, value_id):
            return value_id
    return None


def _input_token_ids(source_program, op):
    return tuple(value_id for value_id in op.operands if _value_is_token(source_program, value_id))


def _require_token_operands(source_program, op):
    for value_id in op.operands:
        if not _value_is_token(source_program, value_id):
            fail(
                "TLXW_TOKEN_NON_TOKEN_DEPENDENCY",
                STAGE,
                f"{op.name} operand {value_id} is not a token",
                source_op_index=op.index,
                source_value_id=value_id,
            )


def _value_is_token(source_program, value_id):
    value = source_program.values.get(value_id)
    return value is not None and value.type.kind == "token"


def _operand_segments(op, expected_len, default):
    segments = op.attrs.get("operandSegmentSizes")
    if segments is None:
        if default is None:
            fail(
                "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
                STAGE,
                f"{op.name} expected operandSegmentSizes",
                source_op_index=op.index,
            )
        segments = default
    segments = tuple(int(segment) for segment in segments)
    if len(segments) != expected_len:
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} expected {expected_len} operand segments, got {segments}",
            source_op_index=op.index,
        )
    if any(segment < 0 for segment in segments):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments must be nonnegative, got {segments}",
            source_op_index=op.index,
        )
    return segments


def _buffer_store_fields(op):
    segments = _operand_segments(op, 5, None)
    _require_operand_count(op, segments)
    if segments[0] != 1 or segments[1] != 1 or segments[2] != 1:
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_store requires value, base pointer, and offsets operands",
            source_op_index=op.index,
        )
    if segments[3] != 0:
        fail(
            "TLXW_TOKEN_UNSUPPORTED_BUFFER_BOUNDARY_CHECK",
            STAGE,
            "amdg.buffer_store boundary-check operands are not supported yet",
            source_op_index=op.index,
        )
    if segments[4] not in (0, 1):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            "amdg.buffer_store supports at most one mask operand",
            source_op_index=op.index,
        )
    base_index = int(segments[0])
    offset_index = base_index + int(segments[1])
    mask_index = offset_index + int(segments[2]) + int(segments[3])
    return {
        "value_value_id": _operand_or_none(op, 0),
        "base_value_id": _operand_or_none(op, base_index),
        "offset_value_id": _operand_or_none(op, offset_index),
        "mask_value_id": _operand_or_none(op, mask_index) if segments[4] else None,
    }


def _require_operand_count(op, segments):
    if sum(segments) != len(op.operands):
        fail(
            "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS",
            STAGE,
            f"{op.name} operand segments {segments} do not match "
            f"{len(op.operands)} operands",
            source_op_index=op.index,
        )


def _operand_or_none(op, index):
    return op.operands[index] if index is not None and index < len(op.operands) else None


def _pointer_address_space(source_program, value_id):
    value = source_program.values.get(value_id)
    if value is None:
        return "unknown"
    address_space = value.type.address_space
    if address_space in {3, "3"}:
        return "local"
    if address_space in {1, "1"}:
        return "global"
    return "unknown"


def _append_user(users_by_value, value_id, node_id):
    users = users_by_value.setdefault(value_id, [])
    if node_id not in users:
        users.append(node_id)


def _int_attr(op, name):
    value = op.attrs.get(name)
    return None if value is None else int(value)


def _attr_or_none(op, name):
    value = op.attrs.get(name)
    return None if value is None else str(value)


def _cache_modifier(op):
    value = op.attrs.get("cacheModifier")
    if value is None:
        value = op.attrs.get("cache")
    return None if value is None else str(value)
