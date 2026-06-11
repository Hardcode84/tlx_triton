import os
import subprocess
import sys
from pathlib import Path

from .wave_bridge_plan import (
    _GFX950_DOT_PARENT_LAYOUT,
    _GFX950_F16_MMA_KIND,
    _GFX950_MMA_M,
    _GFX950_MMA_N,
    _GFX950_MMA_REGS,
    _GFX950_MMA_SHAPE,
    _GFX950_MMA_WAVE,
    _GFX950_SHARED_LAYOUT,
    _BlockedEncodingInfo,
    _DotOperandEncodingInfo,
    _SwizzledSharedEncodingInfo,
    _WaveAsyncStats,
    _async_address_by_token,
    _bridge_stage,
    _compute_lds_layout,
    _has_local_loads_after_wait,
    _local_load_address_by_result,
    _memdesc_size_bytes,
    _memdescs_by_id,
    _target_triple,
    _values_by_id,
)


def _attr_method(attr, method):
    if attr is None:
        return None
    fn = getattr(attr, method, None)
    if fn is None:
        raise RuntimeError(
            "tlx_wave bridge requires typed TTGIR encoding Attribute helpers; "
            f"missing {method}. Rebuild Triton after updating python/src/ir.cc."
        )
    return fn


def _attr_bool(attr, method):
    fn = _attr_method(attr, method)
    return bool(fn()) if fn is not None else False


def _attr_value(attr, method):
    fn = _attr_method(attr, method)
    return fn() if fn is not None else None


def _int_tuple(values):
    if values is None:
        raise ValueError("missing typed TTGIR encoding field")
    return tuple(int(value) for value in values)


def _blocked_encoding_info(attr, raw_encoding, context):
    if attr is None or not _attr_bool(attr, "is_blocked_encoding"):
        raise ValueError(
            f"tlx_wave bridge expected blocked TTGIR encoding for {context}, "
            f"got {raw_encoding}"
        )
    return _BlockedEncodingInfo(
        _int_tuple(_attr_value(attr, "get_blocked_size_per_thread")),
        _int_tuple(_attr_value(attr, "get_blocked_threads_per_warp")),
        _int_tuple(_attr_value(attr, "get_blocked_warps_per_cta")),
        _int_tuple(_attr_value(attr, "get_blocked_order")),
    )


def _swizzled_shared_encoding_info(attr, raw_encoding, context):
    if attr is None or not _attr_bool(attr, "is_swizzled_shared_encoding"):
        raise ValueError(
            "tlx_wave bridge expected #ttg.swizzled_shared encoding for "
            f"{context}, got {raw_encoding}"
        )
    return _SwizzledSharedEncodingInfo(
        int(_attr_value(attr, "get_swizzled_shared_vec")),
        int(_attr_value(attr, "get_swizzled_shared_per_phase")),
        int(_attr_value(attr, "get_swizzled_shared_max_phase")),
        _int_tuple(_attr_value(attr, "get_swizzled_shared_order")),
    )


def _same_blocked_encoding(lhs, rhs):
    return (
        lhs.size_per_thread == rhs.size_per_thread
        and lhs.threads_per_warp == rhs.threads_per_warp
        and lhs.warps_per_cta == rhs.warps_per_cta
        and lhs.order == rhs.order
    )


def _dot_operand_encoding_info(value, context):
    attr = value.encoding_attr
    if attr is None or not _attr_bool(attr, "is_dot_operand_encoding"):
        raise ValueError(
            f"tlx_wave bridge expected #ttg.dot_op encoding for {context}, "
            f"got {value.encoding}"
        )
    parent_attr = _attr_value(attr, "get_dot_operand_parent")
    return _DotOperandEncodingInfo(
        int(_attr_value(attr, "get_dot_operand_op_idx")),
        int(_attr_value(attr, "get_dot_operand_k_width")),
        _blocked_encoding_info(parent_attr, value.encoding, f"{context} parent"),
    )



def _source_pointee_type(kernel, address):
    if address.base_arg_index is None:
        raise ValueError(
            "tlx_wave bridge cannot lower async copy without a source pointer base"
        )
    if address.base_arg_index >= len(kernel.args):
        raise ValueError(
            f"tlx_wave bridge async copy references missing kernel arg {address.base_arg_index}"
        )
    pointee_type = kernel.args[address.base_arg_index].ttgir_type_obj.get_pointee_type()
    if pointee_type is None:
        raise ValueError(
            f"tlx_wave bridge async copy source %{address.base_arg_name} is not a pointer"
        )
    return pointee_type


def _dma_packet_bytes(address, lds_offset):
    if address.element_byte_width is None:
        return None
    if lds_offset % 4 != 0:
        return None
    if address.element_byte_width in (2, 4):
        return 4
    if address.element_byte_width == 16:
        return 16
    return None


def _emit_async_source_ptr(builder, kernel, address, w, width):
    pointee_type = _source_pointee_type(kernel, address)
    element_type = _binding_type(pointee_type, w)
    lane = builder.lane_id(width=width)
    source_base = builder.args[address.base_arg_index]
    return (
        builder.ptr_add(source_base, lane, w.simd_ptr_type(element_type, width=width)),
        lane,
        element_type,
    )


def _emit_async_copy(
    builder, kernel, address, lds_layout, after_token, w, width, stats
):
    if address.memdesc_value_id is None:
        raise ValueError("tlx_wave bridge cannot lower async copy without LDS memdesc")
    if address.memdesc_value_id not in lds_layout.offsets:
        raise ValueError(
            f"tlx_wave bridge has no LDS placement for memdesc {address.memdesc_value_id}"
        )

    lds_offset = lds_layout.offsets[address.memdesc_value_id]
    source, lane, element_type = _emit_async_source_ptr(
        builder, kernel, address, w, width
    )
    dma_bytes = _dma_packet_bytes(address, lds_offset)
    stats.async_copies += 1
    if dma_bytes is not None:
        destination = builder.lds_base(w.i32(), offset=lds_offset)
        stats.dma_load_lds += 1
        return builder.dma_load_lds(
            source, destination, after=after_token, bytes=dma_bytes
        )

    destination_base = builder.lds_base(element_type, offset=lds_offset)
    destination = builder.ptr_add(
        destination_base,
        lane,
        w.simd_ptr_type(element_type, w.shared_address_space(), width),
    )
    values, load_token = builder.load(
        source, w.simd_type(element_type, width), after=after_token
    )
    stats.load_store_fallbacks += 1
    return builder.store(values, destination, after=load_token)


def _join_tokens(builder, tokens, stats):
    if not tokens:
        return builder.token()
    stats.joins += 1
    return builder.join(*tokens)


def _emit_async_tokens(builder, kernel, attrs, plan, lds_layout, w, stats):
    address_by_token = _async_address_by_token(plan)
    needs_shared_ready_token = _has_local_loads_after_wait(plan)
    pending_copy_tokens = []
    committed_groups = []
    last_order_token = None
    ready_tokens = []

    for token in plan.tokens:
        if token.op == "ttg.async_copy_global_to_local":
            address = address_by_token.get(token.value_id)
            if address is None:
                raise ValueError("tlx_wave bridge could not match async copy token")
            if last_order_token is None:
                last_order_token = builder.token()
            last_order_token = _emit_async_copy(
                builder,
                kernel,
                address,
                lds_layout,
                last_order_token,
                w,
                attrs.threads_per_warp,
                stats,
            )
            pending_copy_tokens.append(last_order_token)
        elif token.op == "ttg.async_commit_group":
            group = _join_tokens(builder, tuple(pending_copy_tokens), stats)
            pending_copy_tokens.clear()
            committed_groups.append(group)
            last_order_token = group
            stats.commit_groups += 1
        elif token.op == "ttg.async_wait":
            keep_groups = token.wait_group or 0
            wait_count = max(0, len(committed_groups) - keep_groups)
            if wait_count:
                waited_groups = tuple(committed_groups[:wait_count])
                wait_token = _join_tokens(builder, waited_groups, stats)
                builder.wait(wait_token)
                stats.waits += 1
                committed_groups = committed_groups[wait_count:]
                last_order_token = wait_token
                if needs_shared_ready_token:
                    last_order_token = builder.barrier(wait_token)
                    stats.barriers += 1
            ready_tokens.append(
                last_order_token if last_order_token is not None else builder.token()
            )
    return tuple(ready_tokens)


def _fragment_type_for_dot_operand(info, w):
    if info.op_idx not in (0, 1):
        raise ValueError(
            f"tlx_wave bridge supports #ttg.dot_op opIdx 0/1, got {info.op_idx}"
        )
    return w.fragment_type(
        info.op_idx,
        w.f16(),
        rows=_GFX950_MMA_M,
        columns=_GFX950_MMA_N,
        wave_size=_GFX950_MMA_WAVE,
        registers=_GFX950_MMA_REGS,
    )


def _acc_fragment_type(w):
    return w.fragment_type(
        2,
        w.f32(),
        rows=_GFX950_MMA_M,
        columns=_GFX950_MMA_N,
        wave_size=_GFX950_MMA_WAVE,
        registers=_GFX950_MMA_REGS,
    )


def _unsupported_local_load(value, reason, memdesc=None):
    message = (
        "tlx_wave bridge cannot lower ttg.local_load for WaveAMD MMA: "
        f"{reason}. original TTGIR encoding: {value.encoding}"
    )
    if memdesc is not None:
        message += f"; memdesc encoding: {memdesc.encoding}"
    raise ValueError(message)


def _validate_supported_dot_local_load_layout(value, memdesc, info):
    if not _same_blocked_encoding(info.parent, _GFX950_DOT_PARENT_LAYOUT):
        _unsupported_local_load(
            value,
            "current flat gfx950 fragment loader supports only dot operand "
            "parent layout sizePerThread=(2, 2), threadsPerWarp=(4, 16), "
            "warpsPerCTA=(4, 1), order=(1, 0); "
            f"got sizePerThread={info.parent.size_per_thread}, "
            f"threadsPerWarp={info.parent.threads_per_warp}, "
            f"warpsPerCTA={info.parent.warps_per_cta}, order={info.parent.order}",
            memdesc,
        )
    try:
        shared = _swizzled_shared_encoding_info(
            memdesc.encoding_attr, memdesc.encoding, "ttg.local_load memdesc"
        )
    except ValueError as exc:
        _unsupported_local_load(value, str(exc), memdesc)
    if shared != _GFX950_SHARED_LAYOUT:
        _unsupported_local_load(
            value,
            "current flat gfx950 fragment loader supports only "
            "#ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, "
            "order = [1, 0]}> for f16 32x32 dot operands; "
            f"got vec={shared.vec}, perPhase={shared.per_phase}, "
            f"maxPhase={shared.max_phase}, order={shared.order}",
            memdesc,
        )


def _validate_dot_local_load(value, memdesc, info):
    if value.element_type != "f16" or value.element_byte_width != 2:
        _unsupported_local_load(
            value, f"expected f16 dot operand, got {value.element_type}"
        )
    if value.shape != _GFX950_MMA_SHAPE:
        _unsupported_local_load(
            value, f"expected static 32x32 operand shape, got {value.shape}"
        )
    if info.op_idx not in (0, 1):
        _unsupported_local_load(value, f"expected opIdx 0/1, got {info.op_idx}")
    if info.k_width not in (0, 4, 32):
        _unsupported_local_load(
            value, f"expected kWidth 0, 4, or 32 for gfx950 f16 MFMA, got {info.k_width}"
        )
    if memdesc.element_type != "f16" or memdesc.element_byte_width != 2:
        _unsupported_local_load(
            value,
            f"expected f16 shared memdesc source, got {memdesc.element_type}",
            memdesc,
        )
    if memdesc.shape != _GFX950_MMA_SHAPE:
        _unsupported_local_load(
            value,
            f"expected 32x32 shared memdesc source, got {memdesc.shape}",
            memdesc,
        )
    _validate_supported_dot_local_load_layout(value, memdesc, info)


def _emit_memdesc_i32_ptr(builder, value, memdesc, memdescs, lds_layout, w):
    if memdesc.kind == "allocation":
        if memdesc.value_id not in lds_layout.offsets:
            _unsupported_local_load(value, "allocation has no LDS placement", memdesc)
        return builder.lds_base(w.i32(), offset=lds_layout.offsets[memdesc.value_id])

    if memdesc.view_op != "ttg.memdesc_index" or memdesc.static_index is None:
        _unsupported_local_load(
            value, f"unsupported memdesc view {memdesc.view_op}", memdesc
        )
    if memdesc.base_value_id is None or memdesc.base_value_id not in memdescs:
        _unsupported_local_load(value, "memdesc view has no known base", memdesc)

    base_ptr = _emit_memdesc_i32_ptr(
        builder, value, memdescs[memdesc.base_value_id], memdescs, lds_layout, w
    )
    slot_bytes = _memdesc_size_bytes(memdesc)
    if slot_bytes % 4:
        _unsupported_local_load(
            value, f"memdesc view size {slot_bytes} is not dword aligned", memdesc
        )
    view_offset = builder.index_expr(
        w.sym_ctx.int_(memdesc.static_index * (slot_bytes // 4))
    )
    return builder.ptr_add(base_ptr, view_offset)


def _emit_local_load_fragment(
    builder,
    address,
    value,
    memdesc,
    memdescs,
    info,
    lds_layout,
    after_token,
    w,
    stats,
):
    _validate_dot_local_load(value, memdesc, info)
    if address.memdesc_value_id not in lds_layout.offsets:
        raise ValueError(
            f"tlx_wave bridge has no LDS placement for memdesc {address.memdesc_value_id}"
        )

    lane = builder.lane_id(width=_GFX950_MMA_WAVE)
    lane_sym = w.sym(f"tlx_local_load_{value.value_id}_lane")
    lane_offset = builder.index_expr(
        lane_sym * _GFX950_MMA_REGS,
        {lane_sym: lane},
    )
    base = _emit_memdesc_i32_ptr(builder, value, memdesc, memdescs, lds_layout, w)
    ptr = builder.ptr_add(
        base,
        lane_offset,
        w.simd_ptr_type(w.i32(), w.shared_address_space(), _GFX950_MMA_WAVE),
    )
    fragment, token = builder.fragment_load(
        ptr, _fragment_type_for_dot_operand(info, w), after=after_token
    )
    stats.local_loads += 1
    stats.fragment_packs += 1
    return fragment, token


def _emit_accumulator_fragment(builder, value, wave_values, w, stats):
    if value.value_id in wave_values:
        return wave_values[value.value_id]
    if (
        value.producer == "arith.constant"
        and value.type_kind == "tensor"
        and value.element_type == "f32"
        and value.shape == _GFX950_MMA_SHAPE
        and value.const_value in (0, 0.0)
    ):
        zero = builder.constant(w.i32(), 0)
        fragment = builder.fragment_fill(zero, _acc_fragment_type(w))
        wave_values[value.value_id] = fragment
        stats.fragment_fills += 1
        return fragment
    raise ValueError(
        "tlx_wave bridge supports tt.dot accumulators only from prior tt.dot "
        "results or zero f32 tensor constants; "
        f"got producer={value.producer}, type={value.type}, encoding={value.encoding}"
    )


def _validate_dot_op(operands, acc, result):
    role_values = {}
    role_infos = {}
    for value in operands:
        info = _dot_operand_encoding_info(value, "tt.dot operand")
        if info.op_idx not in (0, 1):
            raise ValueError(
                "tlx_wave bridge expected tt.dot operands with #ttg.dot_op "
                f"opIdx 0/1, got {info.op_idx}; operand encoding: {value.encoding}"
            )
        if info.op_idx in role_values:
            raise ValueError(
                "tlx_wave bridge found duplicate #ttg.dot_op opIdx for tt.dot; "
                f"operand encoding: {value.encoding}"
            )
        role_values[info.op_idx] = value
        role_infos[info.op_idx] = info
    if set(role_values) != {0, 1}:
        encodings = ", ".join(value.encoding or "<none>" for value in operands)
        raise ValueError(
            "tlx_wave bridge expected tt.dot operands with #ttg.dot_op opIdx "
            f"0 and 1; operand encodings: {encodings}"
        )

    if any(value.element_type != "f16" for value in role_values.values()):
        encodings = ", ".join(value.encoding or "<none>" for value in operands)
        raise ValueError(
            "tlx_wave bridge supports only f16 x f16 tt.dot operands for "
            f"{_GFX950_F16_MMA_KIND}; operand encodings: {encodings}"
        )
    if any(value.shape != _GFX950_MMA_SHAPE for value in role_values.values()):
        shapes = ", ".join(str(value.shape) for value in operands)
        encodings = ", ".join(value.encoding or "<none>" for value in operands)
        raise ValueError(
            "tlx_wave bridge supports only static 32x32 tt.dot operands; "
            f"shapes={shapes}; operand encodings: {encodings}"
        )
    if not _same_blocked_encoding(role_infos[0].parent, role_infos[1].parent):
        raise ValueError(
            "tlx_wave bridge expected matching tt.dot operand parent layouts; "
            f"role0 encoding: {role_values[0].encoding}; "
            f"role1 encoding: {role_values[1].encoding}"
        )

    result_layout = _blocked_encoding_info(
        result.encoding_attr, result.encoding, "tt.dot result"
    )
    if not _same_blocked_encoding(role_infos[0].parent, result_layout):
        raise ValueError(
            "tlx_wave bridge expected tt.dot result layout to match dot operand parent; "
            f"result encoding: {result.encoding}; "
            f"operand encoding: {role_values[0].encoding}"
        )
    if result.element_type != "f32" or result.shape != _GFX950_MMA_SHAPE:
        raise ValueError(
            "tlx_wave bridge supports only f32 32x32 tt.dot results; "
            f"got type={result.type}, encoding={result.encoding}"
        )
    if acc.type_kind == "tensor":
        acc_layout = _blocked_encoding_info(
            acc.encoding_attr, acc.encoding, "tt.dot accumulator"
        )
        if not _same_blocked_encoding(result_layout, acc_layout):
            raise ValueError(
                "tlx_wave bridge expected tt.dot accumulator layout to match result; "
                f"accumulator encoding: {acc.encoding}; result encoding: {result.encoding}"
            )
    return role_values


def _emit_dot_op(builder, op, values, wave_values, w, stats):
    if len(op.operands) < 3 or len(op.results) != 1:
        raise ValueError("tlx_wave bridge expected tt.dot with 3 operands and 1 result")
    operands = tuple(values[operand] for operand in op.operands[:2])
    acc = values[op.operands[2]]
    result = values[op.results[0]]
    role_values = _validate_dot_op(operands, acc, result)
    if (
        role_values[0].value_id not in wave_values
        or role_values[1].value_id not in wave_values
    ):
        raise ValueError(
            "tlx_wave bridge supports tt.dot only when both operands are "
            "lowered from ttg.local_load; "
            f"role0 encoding: {role_values[0].encoding}; "
            f"role1 encoding: {role_values[1].encoding}"
        )
    acc_fragment = _emit_accumulator_fragment(builder, acc, wave_values, w, stats)
    dot = builder.mma(
        _GFX950_F16_MMA_KIND,
        wave_values[role_values[0].value_id],
        wave_values[role_values[1].value_id],
        acc_fragment,
    )
    wave_values[result.value_id] = dot
    stats.mmas += 1


def _emit_ordered_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats):
    values = _values_by_id(plan)
    memdescs = _memdescs_by_id(plan)
    address_by_token = _async_address_by_token(plan)
    local_loads = _local_load_address_by_result(plan)
    wave_values = {}
    pending_copy_tokens = []
    committed_groups = []
    last_order_token = None

    for op in plan.ops:
        if op.name == "ttg.async_copy_global_to_local":
            token_id = op.results[0] if op.results else None
            address = address_by_token.get(token_id)
            if address is None:
                raise ValueError("tlx_wave bridge could not match async copy token")
            if last_order_token is None:
                last_order_token = builder.token()
            last_order_token = _emit_async_copy(
                builder,
                kernel,
                address,
                lds_layout,
                last_order_token,
                w,
                attrs.threads_per_warp,
                stats,
            )
            if token_id is not None:
                wave_values[token_id] = last_order_token
            pending_copy_tokens.append(last_order_token)
        elif op.name == "ttg.async_commit_group":
            group = _join_tokens(builder, tuple(pending_copy_tokens), stats)
            pending_copy_tokens.clear()
            committed_groups.append(group)
            last_order_token = group
            for result_id in op.results:
                wave_values[result_id] = group
            stats.commit_groups += 1
        elif op.name == "ttg.async_wait":
            keep_groups = int(op.attrs.get("num", 0) or 0)
            wait_count = max(0, len(committed_groups) - keep_groups)
            if wait_count:
                waited_groups = tuple(committed_groups[:wait_count])
                wait_token = _join_tokens(builder, waited_groups, stats)
                builder.wait(wait_token)
                stats.waits += 1
                committed_groups = committed_groups[wait_count:]
                last_order_token = builder.barrier(wait_token)
                stats.barriers += 1
            ready_token = (
                last_order_token if last_order_token is not None else builder.token()
            )
            for result_id in op.results:
                wave_values[result_id] = ready_token
        elif op.name == "ttg.local_load":
            result_id = op.results[0] if op.results else None
            if result_id is None:
                raise ValueError("tlx_wave bridge expected ttg.local_load result")
            value = values[result_id]
            info = _dot_operand_encoding_info(value, "ttg.local_load result")
            address = local_loads.get(result_id)
            if address is None or address.memdesc_value_id is None:
                _unsupported_local_load(value, "missing shared memdesc source")
            memdesc = memdescs[address.memdesc_value_id]
            after = (
                wave_values.get(address.token_value_id)
                if address.token_value_id is not None
                else last_order_token
            )
            fragment, last_order_token = _emit_local_load_fragment(
                builder,
                address,
                value,
                memdesc,
                memdescs,
                info,
                lds_layout,
                after,
                w,
                stats,
            )
            wave_values[result_id] = fragment
        elif op.name == "tt.dot":
            _emit_dot_op(builder, op, values, wave_values, w, stats)


def _emit_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats):
    if plan.op_counts.get("tt.dot", 0):
        _emit_ordered_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats)
    else:
        _emit_async_tokens(builder, kernel, attrs, plan, lds_layout, w, stats)



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
        "tlx_wave.bridge.stage": w.StringAttr.get(attrs["bridge_stage"]),
        "tlx_wave.source_op": w.StringAttr.get("tt.func"),
        "tlx_wave.num_pointer_args": _binding_i32_attr(w, attrs["pointer_count"]),
        "tlx_wave.num_scalar_args": _binding_i32_attr(w, attrs["scalar_count"]),
        "tlx_wave.wave_size": _binding_i32_attr(w, attrs["wave_size"]),
        "tlx_wave.num_warps": _binding_i32_attr(w, attrs["num_warps"]),
        "tlx_wave.async.num_copies": _binding_i32_attr(w, attrs["async_copy_count"]),
        "tlx_wave.async.num_waits": _binding_i32_attr(w, attrs["async_wait_count"]),
    }


def _emit_wave_skeleton_with_bindings(kernel, attrs, plan):
    w = _load_wave_dsl()
    target_triple = _target_triple(attrs)
    pointer_count = sum(arg.kind == "pointer" for arg in kernel.args)
    scalar_count = sum(arg.kind == "scalar" for arg in kernel.args)
    lds_layout = _compute_lds_layout(plan)
    stats = _WaveAsyncStats(lds_size_bytes=lds_layout.size_bytes)
    bridge_stage = _bridge_stage(plan)
    with w.module() as module_builder:
        func_attrs = _binding_attrs(
            w,
            {
                "bridge_stage": bridge_stage,
                "pointer_count": pointer_count,
                "scalar_count": scalar_count,
                "wave_size": attrs.threads_per_warp,
                "num_warps": attrs.num_warps,
                "async_copy_count": plan.op_counts.get(
                    "ttg.async_copy_global_to_local", 0
                ),
                "async_wait_count": plan.op_counts.get("ttg.async_wait", 0),
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
        module_builder.module.operation.attributes["tlx_wave.plan.num_layouts"] = (
            _binding_i32_attr(w, len(plan.layouts))
        )
        module_builder.module.operation.attributes["tlx_wave.plan.num_tokens"] = (
            _binding_i32_attr(w, len(plan.tokens))
        )
        with module_builder.function(
            kernel.name,
            arg_types,
            kernel=True,
            lds_size=lds_layout.size_bytes if lds_layout.size_bytes else None,
            attrs=func_attrs,
        ) as builder:
            _emit_wave_body(builder, kernel, attrs, plan, lds_layout, w, stats)

        module_builder.module.operation.attributes["tlx_wave.lds_size_bytes"] = (
            _binding_i32_attr(w, lds_layout.size_bytes)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.async_copies"] = (
            _binding_i32_attr(w, stats.async_copies)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.dma_load_lds"] = (
            _binding_i32_attr(w, stats.dma_load_lds)
        )
        module_builder.module.operation.attributes[
            "tlx_wave.emitted.load_store_fallbacks"
        ] = _binding_i32_attr(w, stats.load_store_fallbacks)
        module_builder.module.operation.attributes["tlx_wave.emitted.joins"] = (
            _binding_i32_attr(w, stats.joins)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.waits"] = (
            _binding_i32_attr(w, stats.waits)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.barriers"] = (
            _binding_i32_attr(w, stats.barriers)
        )
        module_builder.module.operation.attributes["tlx_wave.emitted.local_loads"] = (
            _binding_i32_attr(w, stats.local_loads)
        )
        module_builder.module.operation.attributes[
            "tlx_wave.emitted.fragment_packs"
        ] = _binding_i32_attr(w, stats.fragment_packs)
        module_builder.module.operation.attributes[
            "tlx_wave.emitted.fragment_fills"
        ] = _binding_i32_attr(w, stats.fragment_fills)
        module_builder.module.operation.attributes["tlx_wave.emitted.mmas"] = (
            _binding_i32_attr(w, stats.mmas)
        )
        return str(module_builder.module), stats


def _emit_wave_skeleton(kernel, attrs, plan):
    wave_text, stats = _emit_wave_skeleton_with_bindings(kernel, attrs, plan)
    return wave_text, "wave-dsl", stats


def _verify_wave_module(wave_text, wave_opt):
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
            f"tlx_wave generated Wave module failed wave-opt verification: {detail}"
        )

