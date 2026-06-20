"""Token-dependency planning for the next TLX Wave conversion path.

The new converter will build a stream of token events before rewriting ops, and
this module turns those events into explicit async group/window facts.
"""

from dataclasses import dataclass


_ASYNC_COPY_OPS = frozenset(
    {"ttg.async_copy_global_to_local", "amdg.buffer_load_to_local"}
)


@dataclass(frozen=True)
class _TokenEvent:
    op_index: int
    op_name: str
    value_id: int | None = None
    input_token_ids: tuple[int, ...] = ()
    source_address_value_id: int | None = None
    source_offset_value_id: int | None = None
    memdesc_value_id: int | None = None
    mask_value_id: int | None = None
    other_value_id: int | None = None
    wait_group: int | None = None


@dataclass(frozen=True)
class _TokenNode:
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
    committed_group_index: int | None = None
    waited_group_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _TokenGroup:
    index: int
    commit_op_index: int
    token_value_id: int | None
    member_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class _TokenGraph:
    nodes: tuple[_TokenNode, ...]
    groups: tuple[_TokenGroup, ...]
    by_value_id: dict[int, _TokenNode]
    users_by_value_id: dict[int, tuple[_TokenNode, ...]]

    def node_for_value(self, value_id):
        return self.by_value_id.get(value_id)

    def users_for_value(self, value_id):
        return self.users_by_value_id.get(value_id, ())


def _value_type_kind(value):
    source_type = getattr(value, "type", None)
    if source_type is not None and hasattr(source_type, "kind"):
        return source_type.kind
    return getattr(value, "type_kind", None)


def _value_is_token_id(value_id, values_by_id):
    value = values_by_id.get(value_id)
    return value is not None and _value_type_kind(value) == "token"


def _first_token_result(op, values_by_id):
    for value_id in op.results:
        if _value_is_token_id(value_id, values_by_id):
            return value_id
    return None


def _input_token_ids(op, values_by_id):
    return tuple(
        value_id
        for value_id in op.operands
        if _value_is_token_id(value_id, values_by_id)
    )


def _async_event_fields(op):
    if op.name == "ttg.async_copy_global_to_local":
        segments = op.attrs.get("operandSegmentSizes")
        if segments is None:
            segments = (1, 1, 1 if len(op.operands) > 2 else 0, 0)
        if len(segments) != 4:
            raise ValueError(
                "tlx_wave token graph expected ttg.async_copy_global_to_local "
                f"operandSegmentSizes with four entries, got {segments}"
            )
        mask_index = int(segments[0]) + int(segments[1])
        other_index = mask_index + int(segments[2])
        return {
            "source_address_value_id": op.operands[0] if len(op.operands) > 0 else None,
            "source_offset_value_id": None,
            "memdesc_value_id": op.operands[1] if len(op.operands) > 1 else None,
            "mask_value_id": (
                op.operands[mask_index]
                if int(segments[2]) and mask_index < len(op.operands)
                else None
            ),
            "other_value_id": (
                op.operands[other_index]
                if int(segments[3]) and other_index < len(op.operands)
                else None
            ),
        }
    if op.name == "amdg.buffer_load_to_local":
        segments = op.attrs.get("operandSegmentSizes")
        if segments is None:
            segments = (1, 1, 1, 1 if len(op.operands) > 3 else 0, 0, 0)
        if len(segments) != 6:
            raise ValueError(
                "tlx_wave token graph expected amdg.buffer_load_to_local "
                f"operandSegmentSizes with six entries, got {segments}"
            )
        ptr_index = int(segments[0])
        offsets_index = ptr_index + int(segments[1])
        mask_index = offsets_index + int(segments[2])
        other_index = mask_index + int(segments[3])
        return {
            "source_address_value_id": (
                op.operands[ptr_index] if ptr_index < len(op.operands) else None
            ),
            "source_offset_value_id": (
                op.operands[offsets_index] if offsets_index < len(op.operands) else None
            ),
            "memdesc_value_id": op.operands[0] if len(op.operands) > 0 else None,
            "mask_value_id": (
                op.operands[mask_index]
                if int(segments[3]) and mask_index < len(op.operands)
                else None
            ),
            "other_value_id": (
                op.operands[other_index]
                if int(segments[4]) and other_index < len(op.operands)
                else None
            ),
        }
    return {}


def _collect_token_events(ops, values_by_id):
    events = []
    for op in ops:
        value_id = _first_token_result(op, values_by_id)
        input_token_ids = _input_token_ids(op, values_by_id)
        if (
            value_id is None
            and not input_token_ids
            and op.name not in {"ttg.async_wait", "ttg.async_commit_group"}
        ):
            continue
        events.append(
            _TokenEvent(
                op.index,
                op.name,
                value_id=value_id,
                input_token_ids=input_token_ids,
                wait_group=op.attrs.get("num") if op.name == "ttg.async_wait" else None,
                **_async_event_fields(op),
            )
        )
    return tuple(events)


def _token_group_waited_indices(committed_groups, keep_count):
    keep_count = max(0, int(keep_count or 0))
    wait_count = max(0, len(committed_groups) - keep_count)
    return tuple(group.index for group in committed_groups[:wait_count])


def _append_user(users, token_id, node):
    if token_id is None:
        return
    token_users = users.setdefault(token_id, [])
    if not any(user.op_index == node.op_index for user in token_users):
        token_users.append(node)


def _event_node(event, *, committed_group_index=None, waited_group_indices=()):
    return _TokenNode(
        event.op_index,
        event.op_name,
        event.value_id,
        tuple(event.input_token_ids),
        event.source_address_value_id,
        event.source_offset_value_id,
        event.memdesc_value_id,
        event.mask_value_id,
        event.other_value_id,
        event.wait_group,
        committed_group_index,
        tuple(waited_group_indices),
    )


def _build_token_graph(events):
    """Build a conservative async token graph from ordered token events.

    Count-only waits are represented against committed async groups.  Explicit
    token operands stay as direct token dependencies.  This is the first step
    toward multibuffer-aware conversion: rewriters get group/window facts from
    this graph instead of reconstructing producer chains.
    """

    nodes = []
    groups = []
    by_value = {}
    users = {}
    open_async_tokens = []
    committed_groups = []

    for event in events:
        committed_group_index = None
        waited_group_indices = ()

        if event.op_name in _ASYNC_COPY_OPS and event.value_id is not None:
            open_async_tokens.append(event.value_id)

        if event.op_name == "ttg.async_commit_group":
            if event.input_token_ids:
                member_tokens = tuple(event.input_token_ids)
                committed_ids = set(member_tokens)
                open_async_tokens = [
                    token_id
                    for token_id in open_async_tokens
                    if token_id not in committed_ids
                ]
            else:
                member_tokens = tuple(open_async_tokens)
                open_async_tokens = []
            committed_group_index = len(groups)
            group = _TokenGroup(
                committed_group_index,
                event.op_index,
                event.value_id,
                member_tokens,
            )
            groups.append(group)
            committed_groups.append(group)

        if event.op_name == "ttg.async_wait":
            if event.input_token_ids:
                waited_group_indices = ()
            else:
                waited_group_indices = _token_group_waited_indices(
                    committed_groups, event.wait_group
                )
                if waited_group_indices:
                    committed_groups = committed_groups[len(waited_group_indices) :]

        node = _event_node(
            event,
            committed_group_index=committed_group_index,
            waited_group_indices=waited_group_indices,
        )
        nodes.append(node)
        if node.value_id is not None:
            by_value[node.value_id] = node
        for input_token_id in node.input_token_ids:
            _append_user(users, input_token_id, node)
        if event.op_name == "ttg.async_commit_group":
            for member_token_id in groups[committed_group_index].member_token_ids:
                _append_user(users, member_token_id, node)
        if event.op_name == "ttg.async_wait":
            for group_index in waited_group_indices:
                group = groups[group_index]
                if group.token_value_id is not None:
                    _append_user(users, group.token_value_id, node)
                for member_token_id in group.member_token_ids:
                    _append_user(users, member_token_id, node)

    return _TokenGraph(
        tuple(nodes),
        tuple(groups),
        by_value,
        {value_id: tuple(value_users) for value_id, value_users in users.items()},
    )
