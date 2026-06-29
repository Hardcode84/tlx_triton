"""Type and layout conversion for the TLX Wave converter."""

from dataclasses import dataclass

from .layouts import LayoutMap, build_layout_map

_MMA_FRAGMENT_ELEMENT_TYPES = frozenset({"bf16", "f16", "f32"})


@dataclass(frozen=True)
class ConvertedType:
    kind: str
    representation: str
    element_type: str | None = None
    lane_width: int | None = None
    component_count: int = 1


@dataclass(frozen=True)
class ConvertedValue:
    value_id: int
    type: ConvertedType
    layout_map_id: int | None = None


@dataclass(frozen=True)
class TypeLayoutProgram:
    values: dict[int, ConvertedValue]
    layouts: tuple[LayoutMap, ...]


def convert_source_program(program):
    lane_width = int(program.kernel.threads_per_warp or 64)
    layouts = []
    converted_values = {}
    for value_id, source_value in program.values.items():
        layout = build_layout_map(len(layouts), value_id, source_value.type, lane_width)
        layout_id = None
        if layout is not None:
            layout_id = layout.layout_map_id
            layouts.append(layout)
        converted_values[value_id] = ConvertedValue(
            value_id,
            _convert_type(source_value.type, layout, lane_width),
            layout_id,
        )
    return TypeLayoutProgram(converted_values, tuple(layouts))


def _convert_type(source_type, layout, lane_width):
    if source_type.kind == "scalar":
        return ConvertedType("scalar", "scalar", source_type.element_type or source_type.raw)
    if source_type.kind == "pointer":
        return ConvertedType("pointer", "uniform_pointer", source_type.pointee_type)
    if source_type.kind == "token":
        return ConvertedType("token", "token")
    if source_type.kind == "memdesc":
        return ConvertedType("memdesc", "memdesc", source_type.element_type)
    if source_type.kind == "tensor":
        component_count = 1 if layout is None else int(layout.component_count)
        if source_type.element_type == "i1":
            return ConvertedType(
                "mask",
                "mask" if component_count == 1 else "mask_tuple",
                source_type.element_type,
                lane_width,
                component_count,
            )
        if source_type.pointee_type is not None:
            return ConvertedType(
                "pointer",
                "per_lane_pointer" if component_count == 1 else "pointer_tuple",
                source_type.pointee_type,
                lane_width,
                component_count,
            )
        if (layout is not None and layout.kind in {"dot_operand", "amd_mfma"}
                and source_type.element_type in _MMA_FRAGMENT_ELEMENT_TYPES):
            return ConvertedType(
                "fragment",
                "fragment" if component_count == 1 else "fragment_tuple",
                source_type.element_type,
                lane_width,
                component_count,
            )
        return ConvertedType(
            "tensor",
            "simd" if component_count == 1 else "simd_tuple",
            source_type.element_type,
            lane_width,
            component_count,
        )
    return ConvertedType("unsupported", "unsupported")
