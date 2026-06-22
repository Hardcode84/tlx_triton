"""Structural layout-remap helpers for TLX Wave conversion."""

from dataclasses import replace

from .diagnostics import fail
from . import layouts


STAGE = "op_conversion"

_DISTRIBUTED_REMAP_KINDS = frozenset({"blocked", "linear"})
_DISTRIBUTED_REMAP_REPRESENTATIONS = frozenset(
    {
        "mask",
        "mask_tuple",
        "per_lane_pointer",
        "pointer_tuple",
        "simd",
        "simd_tuple",
    }
)


def register_remap(operand, result, operand_layout, result_layout, op):
    if operand_layout is None or result_layout is None:
        return None
    if not (operand_layout.kind == "amd_mfma" and result_layout.kind == "blocked"):
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if result.type.representation not in {"simd", "simd_tuple"}:
        return None

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    source_registers_per_component = layouts.mfma_registers_per_component(
        operand_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_scalar_count = (
        int(operand.type.component_count) * source_registers_per_component
    )
    if source_scalar_count != source_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA convert_layout source component model does not match "
            "the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if int(result.type.component_count) != result_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "MFMA to blocked convert_layout requires a warp-aware result "
            f"component model: result has {int(result.type.component_count)} "
            f"components but the per-wave register layout has "
            f"{result_register_count}",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
    )

    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
        )
        for result_register in range(result_register_count)
    )

    simple_remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result.value_id,
    )
    if simple_remap is not None:
        return {
            "source_component_count": int(operand.type.component_count),
            "source_registers_per_component": int(source_registers_per_component),
            **simple_remap,
        }

    exchange_remap = _cta_exchange_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        op,
        result.value_id,
    )
    return {
        "mode": "cta_exchange_register_remap",
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": int(source_registers_per_component),
        **exchange_remap,
    }


def distributed_remap(operand, result, operand_layout, result_layout, op):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if operand.type.kind != result.type.kind:
        return None
    if operand.type.representation not in _DISTRIBUTED_REMAP_REPRESENTATIONS:
        return None
    if result.type.representation not in _DISTRIBUTED_REMAP_REPRESENTATIONS:
        return None

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    description = (
        f"{operand_layout.kind} to {result_layout.kind} convert_layout"
    )
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        f"{description} source layout",
    )
    _require_injective_layout(
        result_layout_ll,
        op,
        result.value_id,
        f"{description} result layout",
    )
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    if int(operand.type.component_count) != source_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} source component model does not match the "
            "distributed register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    if int(result.type.component_count) != result_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} result component model does not match the "
            "distributed register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description=description,
    )
    result_sources = tuple(
        _sources_for_result_slot(
            result_layout_ll,
            result_register,
            source_by_coord,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description=description,
        )
        for result_register in range(result_register_count)
    )
    remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        1,
        op,
        result.value_id,
        description=description,
    )
    if remap is None:
        _reject_distributed_movement(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description,
        )
    if remap["mode"] == "cross_lane_register_remap" and result.type.representation not in {
        "simd",
        "simd_tuple",
    }:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires cross-lane movement for "
            f"{result.type.representation}; only Wave SIMD payloads can be "
            "shuffled",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    return {
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": 1,
        **remap,
    }


def dot_operand_fragment_pack(operand, result, operand_layout, result_layout, op):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind != "dot_operand":
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    if operand.type.representation not in {"simd", "simd_tuple"}:
        return None
    if result.type.representation not in {"fragment", "fragment_tuple"}:
        return None
    if tuple(operand_layout.shape) != tuple(result_layout.shape):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires matching "
            "source and result shapes",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    parent = result_layout.properties.get("parent_properties", {})
    instr_shape = tuple(int(value) for value in parent.get("instr_shape", ()))
    warps_per_cta = tuple(int(value) for value in parent.get("warps_per_cta", ()))
    if instr_shape not in {(16, 16, 32), (32, 32, 16)} or len(warps_per_cta) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires a supported "
            "MFMA parent layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    registers = _dot_operand_fragment_registers(
        result.type.element_type,
        instr_shape,
        op,
    )
    elements_per_lane = _fragment_elements_per_lane(
        result.type.element_type,
        registers,
        op,
        result.value_id,
    )
    k_width = int(result_layout.properties.get("k_width", 0))
    if k_width != int(elements_per_lane):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires kWidth to "
            "match the fragment payload width; chunked kWidth payloads are "
            f"not implemented yet: kWidth={k_width}, "
            f"payload_width={int(elements_per_lane)}",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    source_layout = _distributed_linear_layout(operand_layout, op)
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        "distributed to dot_operand source layout",
    )
    source_component_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    if int(operand.type.component_count) != source_component_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand source component model does not "
            "match the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    source_warp_count = _layout_warp_count(operand_layout)
    result_warp_count = _dot_operand_parent_warp_count(result_layout)
    if int(source_warp_count) != int(result_warp_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand convert_layout requires matching "
            "source and dot-parent wave counts",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )
    cta_warp_count = int(source_warp_count)
    source_component_count = int(source_component_count)
    cta_thread_count = int(lane_width) * int(cta_warp_count)

    source_store_bases = []
    source_store_coefficients = []
    for source_component in range(source_component_count):
        store_offsets = []
        for source_warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                coords = layouts.linear_layout_coords(
                    source_layout,
                    source_component,
                    lane,
                    warp=source_warp,
                )
                store_offsets.append(
                    _checked_dense_linear_offset(
                        operand_layout.shape,
                        coords,
                        op,
                        operand.value_id,
                        "distributed to dot_operand source store",
                    )
                )
        base, coefficients = _fit_bit_affine_offsets(
            store_offsets,
            cta_thread_count,
            op,
            operand.value_id,
            description="distributed to dot_operand source store convert_layout",
        )
        source_store_bases.append(int(base))
        source_store_coefficients.append(tuple(int(value) for value in coefficients))

    component_vector_load_bases = []
    component_vector_load_coefficients = []
    for component in range(int(result.type.component_count)):
        vector_load_offsets = []
        for result_warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                base = _dot_operand_payload_linear(
                    result_layout,
                    component,
                    0,
                    lane,
                    result_warp,
                    elements_per_lane,
                    instr_shape,
                    warps_per_cta,
                    op,
                )
                end = _dot_operand_payload_linear(
                    result_layout,
                    component,
                    int(elements_per_lane) - 1,
                    lane,
                    result_warp,
                    elements_per_lane,
                    instr_shape,
                    warps_per_cta,
                    op,
                )
                if int(end) != int(base) + int(elements_per_lane) - 1:
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        "distributed to dot_operand payload is not dense "
                        "contiguous in logical scratch",
                        source_op_index=op.index,
                        source_value_id=result.value_id,
                    )
                vector_load_offsets.append(int(base))
        base, coefficients = _fit_bit_affine_offsets(
            vector_load_offsets,
            cta_thread_count,
            op,
            result.value_id,
            description="distributed to dot_operand vector payload convert_layout",
        )
        component_vector_load_bases.append(int(base))
        component_vector_load_coefficients.append(
            tuple(int(value) for value in coefficients)
        )

    return {
        "cta_thread_count": int(cta_thread_count),
        "element_type": result.type.element_type,
        "elements_per_lane": int(elements_per_lane),
        "fragment_vector_load_bases": tuple(component_vector_load_bases),
        "fragment_vector_load_coefficients": tuple(component_vector_load_coefficients),
        "mode": "dot_operand_fragment_pack",
        "payload_mode": "vector",
        "registers": int(registers),
        "role": int(result_layout.properties["op_idx"]),
        "rows": int(instr_shape[0]),
        "columns": int(instr_shape[1]),
        "source_component_count": int(operand.type.component_count),
        "source_store_bases": tuple(source_store_bases),
        "source_store_coefficients": tuple(source_store_coefficients),
        "scratch_element_count": _product(result_layout.shape),
    }


def distributed_to_mfma_base_remap(operand, result, operand_layout, result_layout, op):
    if operand_layout is None or result_layout is None:
        return None
    if operand_layout.kind not in _DISTRIBUTED_REMAP_KINDS:
        return None
    if result_layout.kind != "amd_mfma":
        return None
    if operand.type.element_type != result.type.element_type:
        return None
    value_remap = (
        operand.type.representation in {"simd", "simd_tuple"}
        and result.type.representation in {"fragment", "fragment_tuple"}
        and operand.type.element_type != "i1"
    )
    mask_remap = (
        operand.type.representation in {"mask", "mask_tuple"}
        and result.type.representation in {"mask", "mask_tuple"}
        and operand.type.element_type == "i1"
    )
    if not value_remap and not mask_remap:
        return None
    if tuple(operand_layout.shape) != tuple(result_layout.shape):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA base convert_layout requires matching "
            "source and result shapes",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    source_layout = _distributed_linear_layout(operand_layout, op)
    result_layout_ll = _distributed_linear_layout(result_layout, op)
    description = "distributed to MFMA base convert_layout"
    _require_injective_layout(
        source_layout,
        op,
        operand.value_id,
        f"{description} source layout",
    )
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    if int(operand.type.component_count) != int(source_register_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA base source component model does not match "
            "the source register layout",
            source_op_index=op.index,
            source_value_id=operand.value_id,
        )
    registers_per_component = layouts.mfma_registers_per_component(
        result_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    if int(result.type.component_count) * int(registers_per_component) != int(
        result_register_count
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to MFMA base result component model does not match "
            "the MFMA register layout",
            source_op_index=op.index,
            source_value_id=result.value_id,
        )

    lane_width = int(result.type.lane_width or operand.type.lane_width or 64)
    cta_warp_count = max(
        _layout_warp_count(operand_layout),
        _layout_warp_count(result_layout),
    )
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        operand.value_id,
        description=description,
    )
    result_sources = []
    component_vectors_are_contiguous = True
    for component in range(int(result.type.component_count)):
        base_register = int(component) * int(registers_per_component)
        if not _mfma_component_vector_is_contiguous(
            result_layout_ll,
            base_register,
            int(registers_per_component),
            lane_width,
            cta_warp_count,
        ):
            component_vectors_are_contiguous = False
        result_sources.append(
            _sources_for_result_slot(
                result_layout_ll,
                base_register,
                source_by_coord,
                lane_width,
                cta_warp_count,
                op,
                result.value_id,
                description=description,
            )
        )
    result_sources = tuple(result_sources)

    if value_remap and result.type.element_type == "f32":
        scalar_result_sources = _mfma_scalar_result_sources(
            result_layout_ll,
            source_by_coord,
            int(result.type.component_count),
            int(registers_per_component),
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description,
        )
        return _mfma_vector_register_remap_attrs(
            result,
            result_layout,
            scalar_result_sources,
            lane_width,
            cta_warp_count,
            int(operand.type.component_count),
            source_registers_per_component=1,
            registers_per_component=int(registers_per_component),
            op=op,
            result_value_id=result.value_id,
            description=description,
        )

    if not component_vectors_are_contiguous:
        _reject_distributed_movement(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description,
        )

    remap = _simple_register_remap(
        result_sources,
        lane_width,
        cta_warp_count,
        1,
        op,
        result.value_id,
        description=description,
    )
    if remap is not None and (
        remap["mode"] == "same_lane_register_remap" or value_remap
    ):
        return {
            "source_component_count": int(operand.type.component_count),
            "source_registers_per_component": 1,
            **remap,
        }

    return {
        "mode": "cta_exchange_register_remap",
        "source_component_count": int(operand.type.component_count),
        "source_registers_per_component": 1,
        **_cta_exchange_register_remap(
            result_sources,
            lane_width,
            cta_warp_count,
            op,
            result.value_id,
            description=description,
        ),
    }


def _mfma_scalar_result_sources(
    result_layout,
    source_by_coord,
    result_component_count,
    registers_per_component,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    description,
):
    scalar_result_sources = []
    for component in range(int(result_component_count)):
        base_register = int(component) * int(registers_per_component)
        for element in range(int(registers_per_component)):
            scalar_result_sources.append(
                _sources_for_result_slot(
                    result_layout,
                    base_register + int(element),
                    source_by_coord,
                    lane_width,
                    cta_warp_count,
                    op,
                    result_value_id,
                    description=description,
                )
            )
    return tuple(scalar_result_sources)


def _mfma_vector_register_remap_attrs(
    result,
    result_layout,
    scalar_result_sources,
    lane_width,
    cta_warp_count,
    source_component_count,
    *,
    source_registers_per_component,
    registers_per_component,
    op,
    result_value_id,
    description,
):
    instr_shape = tuple(
        int(entry) for entry in result_layout.properties.get("instr_shape", ())
    )
    scalar_remap = _simple_register_remap(
        scalar_result_sources,
        lane_width,
        cta_warp_count,
        source_registers_per_component,
        op,
        result_value_id,
        description=description,
        allow_fallback=True,
    )
    attrs = {
        "mode": "mfma_vector_register_remap",
        "columns": int(instr_shape[1]),
        "registers": int(registers_per_component),
        "role": 2,
        "rows": int(instr_shape[0]),
        "scalar_result_component_count": len(scalar_result_sources),
        "source_component_count": int(source_component_count),
        "source_registers_per_component": int(source_registers_per_component),
        "vector_length": int(registers_per_component),
    }
    if scalar_remap is not None:
        return {
            **attrs,
            "scalar_mode": scalar_remap["mode"],
            "scalar_source_element_indices": tuple(
                scalar_remap["source_element_indices"]
            ),
            "scalar_source_indices": tuple(scalar_remap["source_indices"]),
            **{
                key: value
                for key, value in scalar_remap.items()
                if key.startswith("source_lane_")
            },
        }
    return {
        **attrs,
        "scratch_reuse_lds": True,
        **_cta_exchange_register_remap(
            scalar_result_sources,
            lane_width,
            cta_warp_count,
            op,
            result_value_id,
            description=description,
        ),
    }


def mfma_accumulator_to_native_remap(value, value_layout, op):
    return _mfma_accumulator_layout_remap(
        value,
        value_layout,
        op,
        source_is_native=False,
        description="result layout to native MFMA accumulator",
    )


def mfma_native_accumulator_remap(value, value_layout, op):
    return _mfma_accumulator_layout_remap(
        value,
        value_layout,
        op,
        source_is_native=True,
        description="native MFMA accumulator to result layout",
    )


def _mfma_accumulator_layout_remap(
    value,
    value_layout,
    op,
    *,
    source_is_native,
    description,
):
    if value_layout is None or value_layout.kind != "amd_mfma":
        return None
    if value.type.representation not in {"fragment", "fragment_tuple"}:
        return None
    instr_shape = tuple(
        int(entry) for entry in value_layout.properties.get("instr_shape", ())
    )
    if instr_shape not in {(16, 16, 32), (32, 32, 16)}:
        return None
    if not bool(value_layout.properties.get("is_transposed", False)):
        return None

    native_layout = replace(
        value_layout,
        properties={**value_layout.properties, "is_transposed": False},
    )
    native_layout_ll = _distributed_linear_layout(native_layout, op)
    value_layout_ll = _distributed_linear_layout(value_layout, op)
    source_layout = native_layout_ll if source_is_native else value_layout_ll
    result_layout_ll = value_layout_ll if source_is_native else native_layout_ll
    registers_per_component = layouts.mfma_registers_per_component(
        value_layout,
        stage=STAGE,
        source_op_index=op.index,
    )
    source_register_count = layouts.linear_layout_in_dim_size(
        source_layout,
        "register",
    )
    result_register_count = layouts.linear_layout_in_dim_size(
        result_layout_ll,
        "register",
    )
    expected_register_count = (
        int(value.type.component_count) * int(registers_per_component)
    )
    if int(source_register_count) != expected_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "native MFMA accumulator source component model does not match "
            "the native register layout",
            source_op_index=op.index,
            source_value_id=value.value_id,
        )
    if int(result_register_count) != expected_register_count:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "native MFMA accumulator result component model does not match "
            "the result register layout",
            source_op_index=op.index,
            source_value_id=value.value_id,
        )

    lane_width = int(value.type.lane_width or value_layout.lane_width or 64)
    cta_warp_count = _layout_warp_count(value_layout)
    source_by_coord = _source_slots_by_coord(
        source_layout,
        source_register_count,
        lane_width,
        cta_warp_count,
        op,
        value.value_id,
        description=description,
        allow_replicated_warps=True,
    )
    scalar_result_sources = []
    for component in range(int(value.type.component_count)):
        base_register = int(component) * int(registers_per_component)
        for element in range(int(registers_per_component)):
            scalar_result_sources.append(
                _sources_for_result_slot(
                    result_layout_ll,
                    base_register + int(element),
                    source_by_coord,
                    lane_width,
                    cta_warp_count,
                    op,
                    value.value_id,
                    description=description,
                )
            )
    scalar_result_sources = tuple(scalar_result_sources)
    scalar_remap = _simple_register_remap(
        scalar_result_sources,
        lane_width,
        cta_warp_count,
        registers_per_component,
        op,
        value.value_id,
        description=description,
        allow_fallback=True,
    )
    attrs = {
        "mode": "mfma_vector_register_remap",
        "columns": int(instr_shape[1]),
        "registers": int(registers_per_component),
        "role": 2,
        "rows": int(instr_shape[0]),
        "scalar_result_component_count": len(scalar_result_sources),
        "source_component_count": int(value.type.component_count),
        "source_registers_per_component": int(registers_per_component),
        "vector_length": int(registers_per_component),
    }
    if scalar_remap is not None:
        return {
            **attrs,
            "scalar_mode": scalar_remap["mode"],
            "scalar_source_element_indices": tuple(
                scalar_remap["source_element_indices"]
            ),
            "scalar_source_indices": tuple(scalar_remap["source_indices"]),
            **{
                key: value
                for key, value in scalar_remap.items()
                if key.startswith("source_lane_")
            },
        }
    return {
        **attrs,
        **_cta_exchange_register_remap(
            scalar_result_sources,
            lane_width,
            cta_warp_count,
            op,
            value.value_id,
            description=description,
        ),
    }


def reject_unsupported_pair(operand_layout, result_layout, op):
    operand_kind = "none" if operand_layout is None else operand_layout.kind
    result_kind = "none" if result_layout is None else result_layout.kind
    if operand_kind in {"slice", "dot_operand"} or result_kind in {"slice", "dot_operand"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{operand_kind} to {result_kind} convert_layout requires parent "
            "layout movement support",
            source_op_index=op.index,
        )
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{operand_kind} to {result_kind} convert_layout has unknown "
        "movement class",
        source_op_index=op.index,
    )


def _source_slots_by_coord(
    source_layout,
    source_register_count,
    lane_width,
    cta_warp_count,
    op,
    source_value_id,
    *,
    description="MFMA convert_layout",
    allow_replicated_warps=False,
):
    source_by_coord = {}
    for source_warp in range(int(cta_warp_count)):
        for source_register in range(int(source_register_count)):
            for lane in range(int(lane_width)):
                coords = layouts.linear_layout_coords(
                    source_layout,
                    source_register,
                    lane,
                    warp=source_warp,
                )
                source = (source_warp, lane, source_register)
                if coords in source_by_coord:
                    if allow_replicated_warps:
                        existing = source_by_coord[coords]
                        if existing and isinstance(existing[0], tuple):
                            source_by_coord[coords] = (
                                *existing,
                                tuple(int(value) for value in source),
                            )
                        else:
                            source_by_coord[coords] = (
                                tuple(int(value) for value in existing),
                                tuple(int(value) for value in source),
                            )
                        continue
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} source layout is not injective "
                        "within the CTA distributed map",
                        source_op_index=op.index,
                        source_value_id=source_value_id,
                    )
                source_by_coord[coords] = tuple(int(value) for value in source)
    return source_by_coord


def _sources_for_result_slot(
    result_layout,
    result_register,
    source_by_coord,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    sources = []
    for result_warp in range(int(cta_warp_count)):
        for lane in range(int(lane_width)):
            coords = layouts.linear_layout_coords(
                result_layout,
                result_register,
                lane,
                warp=result_warp,
            )
            source = source_by_coord.get(coords)
            if source is None:
                fail(
                    "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                    STAGE,
                    f"{description} result coordinate is not "
                    "covered by the source distributed layout",
                    source_op_index=op.index,
                    source_value_id=result_value_id,
                )
            if source and isinstance(source[0], tuple):
                same_warp_sources = [
                    candidate
                    for candidate in source
                    if int(candidate[0]) == int(result_warp)
                ]
                if not same_warp_sources:
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} replicated source coordinate is not "
                        "available in the result wave",
                        source_op_index=op.index,
                        source_value_id=result_value_id,
                    )
                source = same_warp_sources[0]
            sources.append(tuple(int(value) for value in source))
    return tuple(sources)


def _mfma_component_vector_is_contiguous(
    result_layout,
    base_register,
    registers_per_component,
    lane_width,
    cta_warp_count,
):
    if int(registers_per_component) <= 1:
        return True
    for result_warp in range(int(cta_warp_count)):
        for lane in range(int(lane_width)):
            base_coords = layouts.linear_layout_coords(
                result_layout,
                int(base_register),
                lane,
                warp=result_warp,
            )
            for element in range(1, int(registers_per_component)):
                coords = layouts.linear_layout_coords(
                    result_layout,
                    int(base_register) + int(element),
                    lane,
                    warp=result_warp,
                )
                expected = (
                    base_coords[0],
                    base_coords[1] + int(element),
                )
                if tuple(coords) != expected:
                    return False
    return True


def _simple_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    source_registers_per_component,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
    allow_fallback=False,
):
    source_indices = []
    source_element_indices = []
    source_lane_maps = []
    for sources in result_sources:
        if len(sources) != int(cta_warp_count) * int(lane_width):
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a malformed "
                "source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        lane_maps = []
        registers = []
        for result_warp in range(int(cta_warp_count)):
            wave_lane_map = []
            for lane in range(int(lane_width)):
                source_warp, source_lane, source_register = sources[
                    result_warp * int(lane_width) + lane
                ]
                if source_warp != result_warp:
                    return None
                registers.append(source_register)
                wave_lane_map.append(source_lane)
            lane_maps.append(tuple(int(lane) for lane in wave_lane_map))

        first_source = registers[0]
        if not all(source == first_source for source in registers):
            return None
        first_lane_map = lane_maps[0]
        if not all(lane_map == first_lane_map for lane_map in lane_maps):
            if allow_fallback:
                return None
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires a wave-varying source "
                "lane map; explicit CTA-wave remap support is required",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        if all(source_lane == lane for lane, source_lane in enumerate(first_lane_map)):
            source_lane_map = None
        else:
            source_lane_map = first_lane_map
        source_indices.append(first_source // int(source_registers_per_component))
        source_element_indices.append(first_source % int(source_registers_per_component))
        source_lane_maps.append(source_lane_map)

    if allow_fallback:
        lane_map_attrs = _classify_source_lane_maps_or_none(
            source_lane_maps,
            lane_width,
        )
        if lane_map_attrs is None and any(
            lane_map is not None for lane_map in source_lane_maps
        ):
            return None
    else:
        lane_map_attrs = _classify_source_lane_maps(
            source_lane_maps,
            lane_width,
            op,
            result_value_id,
            description=description,
        )
    return {
        "mode": "cross_lane_register_remap"
        if lane_map_attrs is not None
        else "same_lane_register_remap",
        "source_element_indices": tuple(source_element_indices),
        "source_indices": tuple(source_indices),
        **(lane_map_attrs or {}),
    }


def _classify_source_lane_maps_or_none(source_lane_maps, lane_width):
    concrete_maps = [lane_map for lane_map in source_lane_maps if lane_map is not None]
    if not concrete_maps:
        return None
    first = concrete_maps[0]
    if not all(lane_map == first for lane_map in concrete_maps):
        return None
    if len(first) != int(lane_width):
        return None
    if any(int(lane) < 0 or int(lane) >= int(lane_width) for lane in first):
        return None
    kind, attrs = _classify_lane_map(first, lane_width)
    if kind is None:
        return None
    return {
        "source_lane_map": tuple(int(lane) for lane in first),
        "source_lane_map_kind": kind,
        **attrs,
    }


def _cta_exchange_register_remap(
    result_sources,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    cta_thread_count = int(lane_width) * int(cta_warp_count)
    groups = {}
    max_group_slots = 0
    for result_index, sources in enumerate(result_sources):
        if len(sources) != cta_thread_count:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced a malformed CTA source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        source_slots = tuple(sorted({int(source[2]) for source in sources}))
        if not source_slots:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} produced an empty CTA source map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
        max_group_slots = max(max_group_slots, len(source_slots))
        source_slot_indices = {
            source_slot: index for index, source_slot in enumerate(source_slots)
        }
        load_offsets = []
        for source_warp, source_lane, source_register in sources:
            load_offsets.append(
                source_slot_indices[int(source_register)] * cta_thread_count
                + int(source_warp) * int(lane_width)
                + int(source_lane)
            )
        base, coefficients = _fit_bit_affine_offsets(
            load_offsets,
            cta_thread_count,
            op,
            result_value_id,
        )
        groups.setdefault(source_slots, []).append(
            (int(result_index), int(base), tuple(int(value) for value in coefficients))
        )

    exchange_groups = []
    for source_slots, result_entries in groups.items():
        exchange_groups.append(
            (
                tuple(int(slot) for slot in source_slots),
                tuple(int(entry[0]) for entry in result_entries),
                tuple(int(entry[1]) for entry in result_entries),
                tuple(
                    tuple(int(value) for value in entry[2])
                    for entry in result_entries
                ),
            )
        )
    return {
        "cta_thread_count": int(cta_thread_count),
        "exchange_groups": tuple(exchange_groups),
        "scratch_element_count": int(max_group_slots) * int(cta_thread_count),
    }


def _reject_distributed_movement(
    result_sources,
    lane_width,
    cta_warp_count,
    op,
    result_value_id,
    description,
):
    for sources in result_sources:
        for result_warp in range(int(cta_warp_count)):
            for lane in range(int(lane_width)):
                source_warp, _source_lane, _source_register = sources[
                    result_warp * int(lane_width) + lane
                ]
                if int(source_warp) != int(result_warp):
                    fail(
                        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                        STAGE,
                        f"{description} requires cross-warp movement",
                        source_op_index=op.index,
                        source_value_id=result_value_id,
                    )
        source_registers = {int(source[2]) for source in sources}
        if len(source_registers) > 1:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires per-lane source component selection",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{description} has unknown movement class",
        source_op_index=op.index,
        source_value_id=result_value_id,
    )


def _require_injective_layout(linear, op, source_value_id, description):
    if linear.is_injective():
        return
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        f"{description} is non-injective",
        source_op_index=op.index,
        source_value_id=source_value_id,
    )


def _fit_bit_affine_offsets(
    load_offsets,
    cta_thread_count,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    load_offsets = tuple(int(offset) for offset in load_offsets)
    if len(load_offsets) != int(cta_thread_count):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed CTA exchange load map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if int(cta_thread_count) <= 0 or int(cta_thread_count) & (
        int(cta_thread_count) - 1
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} CTA exchange requires a power-of-two CTA "
            "thread count",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    base = load_offsets[0]
    coefficients = []
    for bit in range(int(cta_thread_count).bit_length() - 1):
        coefficients.append(load_offsets[1 << bit] - base)
    for thread in range(int(cta_thread_count)):
        expected = base
        for bit, coefficient in enumerate(coefficients):
            if thread & (1 << bit):
                expected += int(coefficient)
        if load_offsets[thread] != expected:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                f"{description} requires a non-bit-affine CTA exchange "
                "load map",
                source_op_index=op.index,
                source_value_id=result_value_id,
            )
    return int(base), tuple(int(value) for value in coefficients)


def _dot_operand_fragment_registers(element_type, instr_shape, op):
    if element_type in {"f16", "bf16"} and instr_shape in {
        (16, 16, 32),
        (32, 32, 16),
    }:
        return 4
    fail(
        "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
        STAGE,
        "distributed to dot_operand fragment registers are not known for "
        f"element_type={element_type}, instr_shape={instr_shape}",
        source_op_index=op.index,
    )


def _fragment_elements_per_lane(element_type, registers, op, source_value_id):
    element_bits = {"f16": 16, "bf16": 16}.get(element_type)
    if element_bits is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"distributed to dot_operand does not support {element_type} fragments",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return int(registers) * 32 // int(element_bits)


def _dot_operand_parent_warp_count(layout):
    parent = layout.properties.get("parent_properties", {})
    warps_per_cta = tuple(int(value) for value in parent.get("warps_per_cta", ()))
    result = 1
    for value in warps_per_cta:
        result *= max(1, int(value))
    return result


def _dense_linear_offset(shape, coords):
    shape = tuple(int(dim) for dim in shape)
    coords = tuple(int(coord) for coord in coords)
    if len(shape) != len(coords):
        raise AssertionError("rank mismatch in dense linear offset")
    linear = 0
    for coord, extent in zip(coords, shape):
        linear = linear * int(extent) + int(coord)
    return int(linear)


def _dot_operand_payload_linear(
    layout,
    component,
    element,
    lane,
    result_warp,
    elements_per_lane,
    instr_shape,
    warps_per_cta,
    op,
):
    coords = _dot_operand_payload_coords(
        layout,
        component,
        element,
        lane,
        result_warp,
        elements_per_lane,
        instr_shape,
        warps_per_cta,
        op,
    )
    return _dense_linear_offset(layout.shape, coords)


def _checked_dense_linear_offset(shape, coords, op, source_value_id, description):
    shape = tuple(int(dim) for dim in shape)
    coords = tuple(int(coord) for coord in coords)
    if len(shape) != len(coords) or any(
        int(coord) < 0 or int(coord) >= int(extent)
        for coord, extent in zip(coords, shape)
    ):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} coordinate exceeds tensor shape",
            source_op_index=op.index,
            source_value_id=source_value_id,
        )
    return _dense_linear_offset(shape, coords)


def _dot_operand_payload_coords(
    layout,
    component,
    element,
    lane,
    result_warp,
    elements_per_lane,
    instr_shape,
    warps_per_cta,
    op,
):
    shape = tuple(int(dim) for dim in layout.shape)
    if len(shape) < 2:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand requires rank-2 tensors",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    op_idx = int(layout.properties.get("op_idx", -1))
    component = int(component)
    if op_idx == 0:
        k_tiles = _ceil_div(shape[1], instr_shape[2])
        if int(layout.component_count) % k_tiles:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "A dot_operand component count is not divisible by K tiles",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        per_wave_m_tiles = int(layout.component_count) // k_tiles
        m_tile = component // k_tiles
        k_tile = component % k_tiles
        row = m_tile * int(instr_shape[0])
        col = k_tile * int(instr_shape[2])
        row_base = row + (
            _dot_operand_wave_tile_coord(result_warp, warps_per_cta, "m")
            * per_wave_m_tiles
            * int(instr_shape[0])
        )
        linear = (
            row_base * shape[1]
            + col
            + int(lane) * int(elements_per_lane)
            + int(element)
        )
        row = linear // shape[1]
        col = linear % shape[1]
    elif op_idx == 1:
        k_tiles = _ceil_div(shape[0], instr_shape[2])
        if int(layout.component_count) % k_tiles:
            fail(
                "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
                STAGE,
                "B dot_operand component count is not divisible by K tiles",
                source_op_index=op.index,
                source_value_id=layout.value_id,
            )
        per_wave_n_tiles = int(layout.component_count) // k_tiles
        n_tile = component // k_tiles
        k_tile = component % k_tiles
        row = k_tile * int(instr_shape[2])
        col = n_tile * int(instr_shape[1])
        col_base = col + (
            _dot_operand_wave_tile_coord(result_warp, warps_per_cta, "n")
            * per_wave_n_tiles
            * int(instr_shape[1])
        )
        linear = int(lane) * int(elements_per_lane) + int(element)
        linear += row * shape[1] + col_base
        row = linear // shape[1]
        col = linear % shape[1]
    else:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"unsupported dot operand index {op_idx}",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    coords = (int(row), int(col))
    if any(coord < 0 or coord >= extent for coord, extent in zip(coords, shape)):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            "distributed to dot_operand payload coordinate exceeds tensor shape",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return coords


def _dot_operand_wave_tile_coord(result_warp, warps_per_cta, axis):
    warps_n = max(1, int(warps_per_cta[1]))
    if axis == "m":
        wave_coord = int(result_warp) // warps_n
        return wave_coord
    if axis == "n":
        return int(result_warp) % warps_n
    return 0


def _ceil_div(lhs, rhs):
    return (int(lhs) + int(rhs) - 1) // int(rhs)


def _product(values):
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _classify_source_lane_maps(
    source_lane_maps,
    lane_width,
    op,
    result_value_id,
    *,
    description="MFMA to blocked convert_layout",
):
    concrete_maps = [lane_map for lane_map in source_lane_maps if lane_map is not None]
    if not concrete_maps:
        return None
    first = concrete_maps[0]
    if not all(lane_map == first for lane_map in concrete_maps):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires per-component source "
            "lane maps; explicit lane-map remap support is required",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if len(first) != int(lane_width):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced a malformed source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    if any(int(lane) < 0 or int(lane) >= int(lane_width) for lane in first):
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} produced an out-of-range source "
            "lane map",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    kind, attrs = _classify_lane_map(first, lane_width)
    if kind is None:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"{description} requires a non-affine source "
            "lane map; Wave shuffle emission only supports affine and "
            "2-D transpose lane maps",
            source_op_index=op.index,
            source_value_id=result_value_id,
        )
    return {
        "source_lane_map": tuple(int(lane) for lane in first),
        "source_lane_map_kind": kind,
        **attrs,
    }


def _classify_lane_map(lane_map, lane_width):
    lane_width = int(lane_width)
    identity = tuple(range(lane_width))
    if tuple(lane_map) == identity:
        return None, {}
    if lane_width > 1:
        stride = int(lane_map[1]) - int(lane_map[0])
        base = int(lane_map[0])
        if all(int(value) == base + lane * stride for lane, value in enumerate(lane_map)):
            return "affine", {
                "source_lane_affine_base": int(base),
                "source_lane_affine_stride": int(stride),
            }
    for inner in _lane_width_factors(lane_width):
        outer = lane_width // inner
        transposed = tuple((lane % inner) * outer + lane // inner for lane in range(lane_width))
        if tuple(lane_map) == transposed:
            return "transpose", {
                "source_lane_transpose_inner": int(inner),
                "source_lane_transpose_outer": int(outer),
            }
    return None, {}


def _lane_width_factors(lane_width):
    lane_width = int(lane_width)
    for factor in range(2, lane_width):
        if lane_width % factor == 0:
            yield factor


def _distributed_linear_layout(layout, op):
    if layout.kind not in {"blocked", "linear", "amd_mfma"}:
        fail(
            "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT",
            STAGE,
            f"layout {layout.kind} is not converted through linear-layout remap",
            source_op_index=op.index,
            source_value_id=layout.value_id,
        )
    return layouts.distributed_linear_layout(
        layout,
        stage=STAGE,
        source_op_index=op.index,
    )


def _layout_warp_count(layout):
    return layouts.layout_warp_count(layout)
