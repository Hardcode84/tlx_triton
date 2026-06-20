import ast
from contextlib import contextmanager
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import triton
from triton._C.libtriton import ir
import triton.language as tl
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile, make_backend

if "tlx_wave" in backends:
    from triton.backends.tlx_wave import compiler as tlx_wave_compiler
    from triton.backends.tlx_wave import driver as tlx_wave_driver
    from triton.backends.tlx_wave import wave_bridge_tools
    from triton.backends.tlx_wave.converter import diagnostics as converter_diagnostics
    from triton.backends.tlx_wave.converter import canonicalize as converter_canonicalize
    from triton.backends.tlx_wave.converter import emission as converter_emission
    from triton.backends.tlx_wave.converter import facts as converter_facts
    from triton.backends.tlx_wave.converter import op_conversion as converter_op_conversion
    from triton.backends.tlx_wave.converter import pipeline as converter_pipeline
    from triton.backends.tlx_wave.converter import source_import as converter_source_import
    from triton.backends.tlx_wave.converter import source_ir as converter_source_ir
    from triton.backends.tlx_wave.converter import target_ir as converter_target_ir
    from triton.backends.tlx_wave.converter import tokens as converter_tokens
    from triton.backends.tlx_wave.converter import types as converter_types
    from triton.backends.tlx_wave.converter import verifier as converter_verifier
else:
    tlx_wave_compiler = None
    tlx_wave_driver = None
    wave_bridge_tools = None
    converter_diagnostics = None
    converter_canonicalize = None
    converter_emission = None
    converter_facts = None
    converter_op_conversion = None
    converter_pipeline = None
    converter_source_import = None
    converter_source_ir = None
    converter_target_ir = None
    converter_tokens = None
    converter_types = None
    converter_verifier = None


pytestmark = pytest.mark.skipif(
    "tlx_wave" not in backends, reason="tlx_wave backend is not installed"
)

GFX942_WAVE = GPUTarget("tlx_wave", "gfx942", 64)
GFX950_WAVE = GPUTarget("tlx_wave", "gfx950", 64)
_TLX_WAVE_RUNTIME_ARCHES = {"gfx942", "gfx950"}


def _asm_text(compiled, artifact):
    text = compiled.asm[artifact]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return text


def _require_tlx_wave_runtime_target():
    torch = pytest.importorskip("torch")
    try:
        active_driver = triton.runtime.driver.active
        device = active_driver.get_current_device()
        properties = active_driver.utils.get_device_properties(device)
    except Exception as exc:
        pytest.skip(f"requires an active HIP runtime for TLX Wave launch tests: {exc}")
    arch = str(properties.get("arch", "")).split(":")[0]
    if arch not in _TLX_WAVE_RUNTIME_ARCHES:
        pytest.skip(
            "requires physical gfx942/gfx950 hardware for TLX Wave launch "
            f"tests, got {arch or 'unknown'}"
        )
    if not torch.cuda.is_available():
        pytest.skip("requires torch.cuda/ROCm for TLX Wave launch tests")
    return torch, arch


@contextmanager
def _active_tlx_wave_driver():
    previous_driver = triton.runtime.driver.active
    triton.runtime.driver.set_active(tlx_wave_driver.TLXWaveDriver())
    try:
        yield
    finally:
        triton.runtime.driver.set_active(previous_driver)


def test_tlx_wave_converter_import_stage_boundary_is_static():
    package_root = (
        Path(__file__).resolve().parents[4]
        / "third_party"
        / "tlx_wave"
        / "backend"
        / "converter"
    )
    forbidden_prefixes = (
        "triton.backends.tlx_wave.wave_bridge",
        "third_party.tlx_wave.backend.wave_bridge",
    )
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)
        assert not any(
            module.startswith(forbidden_prefixes) for module in imports
        ), (path, imports)

    assert "wave_bridge" not in converter_source_ir.__dict__
    assert "wave_bridge" not in converter_diagnostics.__dict__
    assert "wave_bridge" not in converter_canonicalize.__dict__
    assert "wave_bridge" not in converter_source_import.__dict__
    assert "wave_bridge" not in converter_tokens.__dict__
    assert "wave_bridge" not in converter_target_ir.__dict__
    assert "wave_bridge" not in converter_op_conversion.__dict__
    assert "wave_bridge" not in converter_verifier.__dict__
    assert "wave_bridge" not in converter_emission.__dict__
    assert "wave_bridge" not in converter_pipeline.__dict__


def test_tlx_wave_converter_op_rewriters_do_not_accept_source_program():
    tree = ast.parse(
        Path(converter_op_conversion.__file__).read_text(),
        filename=converter_op_conversion.__file__,
    )
    allowed_source_program_functions = {
        "convert_ops",
        "_build_conversion_input",
        "_memdesc_infos",
        "_constant_ints",
    }
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in allowed_source_program_functions:
            continue
        arg_names = [arg.arg for arg in node.args.args]
        if "source_program" in arg_names:
            offenders.append(f"{node.name}:argument")
        if any(
            isinstance(child, ast.Name) and child.id == "source_program"
            for child in ast.walk(node)
        ):
            offenders.append(f"{node.name}:body")

    assert not offenders


def test_tlx_wave_converter_import_stage_builds_source_snapshot(tmp_path):
    local_func = """
  tt.func public @converter_import(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32, tt.divisibility = 16 : i32},
      %arg1: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg1, %zero : i32
    %value = scf.if %positive -> (i32) {
      %one = arith.constant 1 : i32
      scf.yield %one : i32
    } else {
      %two = arith.constant 2 : i32
      scf.yield %two : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2)

    program = converter_source_import.import_source_program(mod)

    assert isinstance(program, converter_source_ir.SourceProgram)
    assert program.kernel.name == "converter_import"
    assert program.kernel.num_warps == 2
    assert len(program.kernel.arg_ids) == 2
    pointer_arg = program.values[program.kernel.arg_ids[0]]
    assert pointer_arg.type.kind == "pointer"
    assert pointer_arg.type.pointer_range == 32
    assert pointer_arg.type.divisibility == 16

    if_op = next(op for op in program.ops if op.name == "scf.if")
    assert len(if_op.region_ids) == 2
    assert all(program.regions[region_id].parent_op_index == if_op.index for region_id in if_op.region_ids)
    assert [
        program.ops[index].name
        for index in program.regions[if_op.region_ids[0]].op_indices
    ] == ["arith.constant", "scf.yield"]
    assert [
        program.ops[index].name
        for index in program.regions[if_op.region_ids[1]].op_indices
    ] == ["arith.constant", "scf.yield"]
    assert all(
        program.values[result_id].owner_op_index == op.index
        for op in program.ops
        for result_id in op.results
    )
    assert not any(hasattr(value, "users") for value in program.values.values())
    del ctx


def test_tlx_wave_converter_import_stage_reports_structured_diagnostics(tmp_path):
    local_func = """
  tt.func private @not_public() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_source_import.import_source_program(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_IMPORT_KERNEL_COUNT"
    assert diagnostic.stage == "import"
    assert diagnostic.no_fallback is True
    assert "no_fallback" in str(diagnostic)
    del ctx


def test_tlx_wave_converter_type_layout_stage_converts_source_snapshot(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_type_layout(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<128xi1, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)

    converted = converter_types.convert_source_program(source)

    range_op = next(op for op in source.ops if op.name == "tt.make_range")
    range_value = converted.values[range_op.results[0]]
    range_layout = converted.layouts[range_value.layout_map_id]
    assert range_value.type.representation == "simd_tuple"
    assert range_value.type.component_count == 2
    assert range_layout.kind == "blocked"
    assert range_layout.properties["size_per_thread"] == (2,)

    mask_op = next(op for op in source.ops if op.name == "tt.splat" and source.values[op.results[0]].type.element_type == "i1")
    assert converted.values[mask_op.results[0]].type.representation == "mask_tuple"

    addptr_op = next(op for op in source.ops if op.name == "tt.addptr")
    assert converted.values[addptr_op.results[0]].type.representation == "pointer_tuple"
    assert not hasattr(converted, "target_ops")
    del ctx


def test_tlx_wave_converter_type_layout_stage_rejects_unknown_encoding():
    class UnknownEncoding:
        pass

    source_type = converter_source_ir.SourceType(
        "tensor<4xf32, #unknown>",
        "tensor",
        shape=(4,),
        element_type="f32",
        encoding_attr=UnknownEncoding(),
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_layout", threads_per_warp=64),
        (),
        {1: converter_source_ir.SourceValue(1, source_type)},
        (converter_source_ir.SourceRegion(0, ()),),
        0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_types.convert_source_program(program)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TYPE_UNSUPPORTED_LAYOUT"
    assert diagnostic.stage == "type_layout"
    assert diagnostic.source_value_id == 1
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_type_layout_stage_supports_slice_encoding(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_slice_layout() attributes {noinline = false} {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #slice>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)
    source = converter_source_import.import_source_program(mod)

    converted = converter_types.convert_source_program(source)

    range_op = next(op for op in source.ops if op.name == "tt.make_range")
    range_value = converted.values[range_op.results[0]]
    layout = converted.layouts[range_value.layout_map_id]
    assert layout.kind == "slice"
    assert layout.properties["dim"] == 1
    assert layout.properties["parent_kind"] == "blocked"
    assert layout.properties["parent_properties"]["warps_per_cta"] == (8, 1)
    del ctx


def test_tlx_wave_converter_fact_stage_extracts_provenance_facts(tmp_path):
    local_func = """
  tt.func public @converter_facts(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %stride: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %nonnegative = arith.cmpi sge, %stride, %zero : i32
    llvm.intr.assume %nonnegative : i1
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    fact_program = converter_facts.analyze_facts(source, converted)

    pointer_arg_id, stride_id = source.kernel.arg_ids
    pointer_facts = converter_facts.facts_for_value(fact_program, pointer_arg_id)
    assert any(
        fact.kind == "pointer_byte_range"
        and fact.lower == 0
        and fact.upper == (1 << 31) - 1
        and fact.width == 32
        and fact.signedness == "signed"
        and fact.provenance == "arg:tt.pointer_range"
        for fact in pointer_facts
    )

    stride_facts = converter_facts.facts_for_value(fact_program, stride_id)
    assert any(
        fact.kind == "range"
        and fact.predicate == "signed_width"
        and fact.lower == -(1 << 31)
        and fact.upper == (1 << 31) - 1
        and fact.width == 32
        and fact.provenance == "type:i32"
        for fact in stride_facts
    )
    assume_op = next(op for op in source.ops if op.name == "llvm.intr.assume")
    assert any(
        fact.kind == "range"
        and fact.predicate == "sge"
        and fact.lower == 0
        and fact.upper is None
        and fact.width == 32
        and fact.provenance == "llvm.intr.assume"
        and fact.source_op_index == assume_op.index
        for fact in stride_facts
    )
    assert not hasattr(fact_program, "target_ops")
    del ctx


def test_tlx_wave_converter_fact_stage_does_not_infer_overflowing_mul(tmp_path):
    local_func = """
  tt.func public @converter_no_mul_fact(%x: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %x_nonnegative = arith.cmpi sge, %x, %zero : i32
    llvm.intr.assume %x_nonnegative : i1
    %prod = arith.muli %x, %x : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    fact_program = converter_facts.analyze_facts(source, converted)

    product_op = next(op for op in source.ops if op.name == "arith.muli")
    product_facts = converter_facts.facts_for_value(fact_program, product_op.results[0])
    assert not any(
        fact.kind == "range"
        and fact.predicate != "signed_width"
        and fact.lower is not None
        and fact.lower >= 0
        for fact in product_facts
    )
    del ctx


def test_tlx_wave_converter_token_stage_builds_async_groups_and_effects(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_tokens(%arg0: !tt.ptr<f16>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<f16>, #blocked>, tensor<64xi32, #blocked>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<64xi1, #blocked>
    %token = ttg.async_copy_global_to_local %ptr, %alloc mask %mask : tensor<64x!tt.ptr<f16>, #blocked> -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    copy_op = next(op for op in source.ops if op.name == "ttg.async_copy_global_to_local")
    group_op = next(op for op in source.ops if op.name == "ttg.async_commit_group")
    wait_op = next(op for op in source.ops if op.name == "ttg.async_wait")
    copy_node = token_program.node_for_value(copy_op.results[0])
    group_node = token_program.node_for_value(group_op.results[0])
    wait_node = token_program.node_for_value(wait_op.results[0])
    assert copy_node.source_address_value_id == copy_op.operands[0]
    assert copy_node.memdesc_value_id == copy_op.operands[1]
    assert copy_node.mask_value_id == copy_op.operands[2]
    assert token_program.groups[group_node.committed_group_id].member_token_ids == copy_op.results
    assert wait_node.input_token_ids == group_op.results
    assert token_program.users_for_value(copy_op.results[0]) == (group_node,)
    assert token_program.users_for_value(group_op.results[0]) == (wait_node,)

    assert [(effect.kind, effect.address_space) for effect in token_program.memory_effects] == [
        ("read", "global"),
        ("write", "local"),
    ]
    read_effect, write_effect = token_program.memory_effects
    assert read_effect.address_value_id == copy_op.operands[0]
    assert write_effect.address_value_id == copy_op.operands[1]
    assert read_effect.mask_value_id == write_effect.mask_value_id == copy_op.operands[2]
    assert read_effect.token_node_id == write_effect.token_node_id == copy_node.node_id
    assert write_effect.depends_on_effect_ids == (read_effect.effect_id,)
    assert not hasattr(token_program, "target_ops")
    del ctx


def test_tlx_wave_converter_token_stage_orders_generic_memory_effects(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_memory_order(%ptr: tensor<64x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %loaded = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %loaded : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    load_effect, store_effect = token_program.memory_effects
    assert (load_effect.op_name, load_effect.kind) == ("tt.load", "read")
    assert (store_effect.op_name, store_effect.kind) == ("tt.store", "write")
    assert store_effect.depends_on_effect_ids == (load_effect.effect_id,)
    assert load_effect.alias_class == store_effect.alias_class == "unknown"
    assert token_program.nodes == ()
    del ctx


def test_tlx_wave_converter_token_stage_orders_local_memory_effects(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_local_effects(%value: tensor<64xf32, #blocked>) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    ttg.local_store %value, %alloc : tensor<64xf32, #blocked> -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %loaded = ttg.local_load %alloc : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    store_effect, load_effect = token_program.memory_effects
    assert (store_effect.op_name, store_effect.kind, store_effect.address_space) == (
        "ttg.local_store",
        "write",
        "local",
    )
    assert (load_effect.op_name, load_effect.kind, load_effect.address_space) == (
        "ttg.local_load",
        "read",
        "local",
    )
    assert load_effect.depends_on_effect_ids == (store_effect.effect_id,)
    assert token_program.nodes == ()
    del ctx


def test_tlx_wave_converter_token_stage_records_buffer_store_effect():
    value_type = converter_source_ir.SourceType(
        "tensor<64xf16>",
        "tensor",
        shape=(64,),
        element_type="f16",
    )
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f16>",
        "pointer",
        pointee_type="f16",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64,),
        element_type="i32",
    )
    mask_type = converter_source_ir.SourceType(
        "tensor<64xi1>",
        "tensor",
        shape=(64,),
        element_type="i1",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("buffer_store_effect"),
        (
            converter_source_ir.SourceOp(
                0,
                "amdg.buffer_store",
                operands=(1, 2, 3, 4),
                attrs={
                    "cacheModifier": "none",
                    "operandSegmentSizes": (1, 1, 1, 0, 1),
                },
            ),
        ),
        {
            1: converter_source_ir.SourceValue(1, value_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, pointer_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, offset_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, mask_type, producer_name="arg3"),
        },
        (converter_source_ir.SourceRegion(0, (0,)),),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    (effect,) = token_program.memory_effects
    assert (effect.op_name, effect.kind, effect.address_space) == (
        "amdg.buffer_store",
        "write",
        "buffer",
    )
    assert effect.value_value_id == 1
    assert effect.address_value_id == 2
    assert effect.offset_value_id == 3
    assert effect.mask_value_id == 4
    assert effect.cache_modifier == "none"


def test_tlx_wave_converter_token_stage_records_buffer_load_effect():
    result_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64,),
        element_type="f32",
    )
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f32>",
        "pointer",
        pointee_type="f32",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64,),
        element_type="i32",
    )
    mask_type = converter_source_ir.SourceType(
        "tensor<64xi1>",
        "tensor",
        shape=(64,),
        element_type="i1",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("buffer_load_effect"),
        (
            converter_source_ir.SourceOp(
                0,
                "amdg.buffer_load",
                operands=(1, 2, 3, 4),
                results=(5,),
                attrs={
                    "cache": 1,
                    "operandSegmentSizes": (1, 1, 0, 1, 1),
                },
            ),
        ),
        {
            1: converter_source_ir.SourceValue(1, pointer_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, offset_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, mask_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, result_type, producer_name="arg3"),
            5: converter_source_ir.SourceValue(5, result_type, producer_name="load"),
        },
        (converter_source_ir.SourceRegion(0, (0,)),),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    (effect,) = token_program.memory_effects
    assert (effect.op_name, effect.kind, effect.address_space) == (
        "amdg.buffer_load",
        "read",
        "buffer",
    )
    assert effect.address_value_id == 1
    assert effect.offset_value_id == 2
    assert effect.value_value_id is None
    assert effect.mask_value_id == 3


def test_tlx_wave_converter_token_stage_reports_malformed_segments():
    pointer_type = converter_source_ir.SourceType(
        "!tt.ptr<f16>",
        "pointer",
        pointee_type="f16",
        address_space=1,
    )
    memdesc_type = converter_source_ir.SourceType(
        "!ttg.memdesc<64xf16>",
        "memdesc",
        shape=(64,),
        element_type="f16",
    )
    token_type = converter_source_ir.SourceType("!tt.async.token", "token")
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_segments"),
        (
            converter_source_ir.SourceOp(
                0,
                "ttg.async_copy_global_to_local",
                operands=(1, 2),
                results=(3,),
                attrs={"operandSegmentSizes": (1, 1, 0)},
            ),
        ),
        {
            1: converter_source_ir.SourceValue(1, pointer_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, memdesc_type, producer_name="alloc"),
            3: converter_source_ir.SourceValue(
                3,
                token_type,
                owner_op_index=0,
                producer_name="ttg.async_copy_global_to_local",
            ),
        },
        (converter_source_ir.SourceRegion(0, (0,)),),
        0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_tokens.build_token_program(program, None)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TOKEN_MALFORMED_OPERAND_SEGMENTS"
    assert diagnostic.stage == "tokens"
    assert diagnostic.source_op_index == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_token_stage_rejects_non_token_dependencies():
    scalar_type = converter_source_ir.SourceType("i32", "scalar")
    token_type = converter_source_ir.SourceType("!tt.async.token", "token")
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("bad_token_dep"),
        (
            converter_source_ir.SourceOp(
                0,
                "ttg.async_commit_group",
                operands=(1,),
                results=(2,),
            ),
        ),
        {
            1: converter_source_ir.SourceValue(1, scalar_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(
                2,
                token_type,
                owner_op_index=0,
                producer_name="ttg.async_commit_group",
            ),
        },
        (converter_source_ir.SourceRegion(0, (0,)),),
        0,
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_tokens.build_token_program(program, None)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TOKEN_NON_TOKEN_DEPENDENCY"
    assert diagnostic.stage == "tokens"
    assert diagnostic.source_op_index == 0
    assert diagnostic.source_value_id == 1
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_op_stage_lowers_basic_dataflow(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_op(%arg0: i32, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %arg0, %zero : i32
    %pred = arith.cmpi sge, %sum, %zero : i32
    llvm.intr.assume %pred : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    facts = converter_facts.analyze_facts(source, converted)
    tokens = converter_tokens.build_token_program(source, converted)

    target = converter_op_conversion.convert_ops(source, converted, facts, tokens)

    assert converter_verifier.verify_target_program(
        target,
        source_program=source,
        fact_program=facts,
        token_program=tokens,
    )
    assert [op.kind for op in target.ops] == [
        "constant",
        "binary",
        "cmpi",
        "assume",
        "make_range",
        "splat",
        "addptr",
        "return",
    ]
    binary_op = next(op for op in target.ops if op.kind == "binary")
    assert converter_target_ir.attrs_dict(binary_op) == {
        "operation": "addi",
        "source_width": 32,
    }
    assume_op = next(op for op in target.ops if op.kind == "assume")
    assert assume_op.fact_ids
    range_op = next(op for op in target.ops if op.kind == "make_range")
    assert converter_target_ir.attrs_dict(range_op) == {"end": 64, "start": 0}
    assert not any(
        callable(attr.value)
        for op in target.ops
        for attr in op.attrs
    )
    assert not hasattr(target, "source_program")
    assert not hasattr(target, "target_ops")
    del ctx


def test_tlx_wave_converter_materializes_operand_assumes_before_arithmetic(tmp_path):
    local_func = """
  tt.func public @converter_early_assume(%arg0: i32) attributes {noinline = false} {
    %c255 = arith.constant 255 : i32
    %sum = arith.addi %arg0, %c255 : i32
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg0, %zero : i32
    llvm.intr.assume %positive : i1
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    add_op = next(
        op
        for op in output.target_program.ops
        if op.kind == "binary"
        and converter_target_ir.attrs_dict(op)["operation"] == "addi"
    )
    assert add_op.fact_ids
    wave = output.emitted_module.text
    assert wave.index("wave.assume %arg0") < wave.index("wave.binary addi")
    del ctx


def test_tlx_wave_converter_preserves_explicit_arith_overflow_flags(tmp_path):
    local_func = """
  tt.func public @converter_overflow_flags(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %sum = arith.addi %arg0, %arg1 overflow<nsw> : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    binary_op = next(op for op in output.target_program.ops if op.kind == "binary")
    assert converter_target_ir.attrs_dict(binary_op)["nsw"] is True
    assert "wave.binary addi" in output.emitted_module.text
    assert "overflow<nsw>" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_float_add(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_float_add() attributes {noinline = false} {
    %lhs = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %rhs = arith.constant dense<2.000000e+00> : tensor<64xf32, #blocked>
    %sum = arith.addf %lhs, %rhs : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    float_op = next(op for op in output.target_program.ops if op.kind == "float_binary")
    assert converter_target_ir.attrs_dict(float_op)["operation"] == "addf"
    assert "wave.fadd" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_rejects_scalar_float_add(tmp_path):
    local_func = """
  tt.func public @converter_scalar_float_add(%arg0: f32, %arg1: f32) attributes {noinline = false} {
    %sum = arith.addf %arg0, %arg1 : f32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_FLOAT_BINARY"
    assert diagnostic.stage == "op_conversion"
    assert diagnostic.no_fallback is True
    del ctx


def test_tlx_wave_converter_canonicalizes_div_before_rem_pair():
    target = _target_div_rem_program(("divsi", "remsi"))

    canonical = converter_canonicalize.canonicalize_target_program(target)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"]
        for op in canonical.ops
        if op.kind == "binary"
    ]
    assert operations == ["divsi", "muli", "subi"]
    assert canonical.ops[0].results == (2,)
    assert canonical.ops[1].operands == (2, 1)
    assert canonical.ops[2].operands == (0, 4)
    assert canonical.ops[2].results == (3,)


def test_tlx_wave_converter_moves_div_before_earlier_rem_pair():
    target = _target_div_rem_program(("remsi", "divsi"))

    canonical = converter_canonicalize.canonicalize_target_program(target)

    operations = [
        converter_target_ir.attrs_dict(op)["operation"]
        for op in canonical.ops
        if op.kind == "binary"
    ]
    assert operations == ["divsi", "muli", "subi"]
    assert canonical.ops[0].results == (3,)
    assert canonical.ops[1].operands == (3, 1)
    assert canonical.ops[2].operands == (0, 4)
    assert canonical.ops[2].results == (2,)


def _target_div_rem_program(operations):
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    values = tuple(
        converter_target_ir.TargetValue(value_id, scalar_i32)
        for value_id in range(4)
    )
    result_ids = {"divsi": 2, "divui": 2, "remsi": 3, "remui": 3}
    if operations[0].startswith("rem"):
        result_ids = {"remsi": 2, "remui": 2, "divsi": 3, "divui": 3}
    ops = tuple(
        converter_target_ir.TargetOp(
            index,
            "binary",
            operands=(0, 1),
            results=(result_ids[operation],),
            attrs=(
                converter_target_ir.TargetAttr("operation", operation),
                converter_target_ir.TargetAttr("source_width", 32),
            ),
        )
        for index, operation in enumerate(operations)
    )
    return converter_target_ir.TargetProgram(
        values,
        ops,
        (converter_target_ir.TargetRegion(0, tuple(range(len(ops)))),),
        {},
        {},
    )


def test_tlx_wave_converter_op_stage_rejects_unsupported_memory_op():
    pointer_type = converter_source_ir.SourceType(
        "tensor<64x!tt.ptr<f32>>",
        "tensor",
        shape=(64,),
        pointee_type="f32",
        address_space=1,
    )
    value_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64,),
        element_type="f32",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("unsupported_load", arg_ids=(1,)),
        (
            converter_source_ir.SourceOp(
                0,
                "tt.load",
                operands=(1,),
                results=(2,),
            ),
        ),
        {
            1: converter_source_ir.SourceValue(
                1,
                pointer_type,
                producer_name="arg0",
                argument_index=0,
            ),
            2: converter_source_ir.SourceValue(
                2,
                value_type,
                owner_op_index=0,
                producer_name="tt.load",
            ),
        },
        (converter_source_ir.SourceRegion(0, (0,)),),
        0,
    )
    converted = converter_types.convert_source_program(program)
    facts = converter_facts.analyze_facts(program, converted)
    tokens = converter_tokens.build_token_program(program, converted)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_op_conversion.convert_ops(program, converted, facts, tokens)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED"
    assert diagnostic.stage == "op_conversion"
    assert diagnostic.source_op_index == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_missing_fact():
    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(
                0,
                converter_target_ir.TargetType("mask", "mask", "i1"),
            ),
        ),
        (
            converter_target_ir.TargetOp(
                0,
                "assume",
                operands=(0,),
            ),
        ),
        (converter_target_ir.TargetRegion(0, (0,)),),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_MISSING_FACT"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_unknown_target_value():
    target = converter_target_ir.TargetProgram(
        (),
        (
            converter_target_ir.TargetOp(
                0,
                "return",
                operands=(99,),
            ),
        ),
        (converter_target_ir.TargetRegion(0, (0,)),),
        {},
        {},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_UNKNOWN_TARGET_VALUE"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.target_value_id == 99
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_emission_stage_emits_basic_wave_module(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_emit(%arg0: i32, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %arg0, %zero : i32
    %pred = arith.cmpi sge, %sum, %zero : i32
    llvm.intr.assume %pred : i1
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %base = tt.splat %arg1 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>, #blocked>
    %ptr = tt.addptr %base, %range : tensor<64x!tt.ptr<i32>, #blocked>, tensor<64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    output = converter_pipeline.convert_ttgir_to_wave(mod)
    emitted = output.emitted_module

    assert "@converter_emit" in emitted.text
    assert "tlx_wave.new_converter" in emitted.text
    assert "gpu.module @kernels" in emitted.text
    assert "gpu.kernel" in emitted.text
    assert "wave.binary" in emitted.text
    assert "wave.assume" in emitted.text
    assert "wave.ptr_add" in emitted.text
    assert "wave_bridge" not in emitted.text
    assert output.target_program.kernel.name == "converter_emit"
    del ctx


def test_tlx_wave_backend_wave_stage_uses_staged_converter(tmp_path, monkeypatch):
    local_func = """
  tt.func public @backend_wave_stage(%arg0: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %sum = arith.addi %arg0, %zero : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(),
    )

    assert "tlx_wave.new_converter" in wave_artifact
    assert "gpu.module @kernels" in wave_artifact
    assert "gpu.kernel" in wave_artifact
    assert "wave_bridge" not in wave_artifact
    assert metadata["name"] == "backend_wave_stage"
    assert metadata["tlx_wave_status"] == "emitted_wave_staged_converter"
    assert metadata["tlx_wave_bridge_stage"] == "staged-converter"
    assert metadata["tlx_wave_wave_builder"] == "staged-converter"
    assert metadata["tlx_wave_plan_kind"] == "staged-converter"
    assert metadata["tlx_wave_ttgir_target"] == "hip:gfx950"
    assert metadata["tlx_wave_num_kernel_args"] == 1
    assert metadata["tlx_wave_num_scalar_args"] == 1
    assert metadata["tlx_wave_num_pointer_args"] == 0
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_hash_includes_wave_opt_sha(monkeypatch):
    backend = tlx_wave_compiler.TLXWaveBackend(GFX950_WAVE)
    first_sha = "1" * 64
    second_sha = "2" * 64

    monkeypatch.setattr(tlx_wave_compiler, "_wave_opt_sha256", lambda: first_sha)
    first_hash = backend.hash()

    monkeypatch.setattr(tlx_wave_compiler, "_wave_opt_sha256", lambda: second_sha)
    second_hash = backend.hash()

    assert f"wave-opt-sha256={first_sha}" in first_hash
    assert f"wave-opt-sha256={second_sha}" in second_hash
    assert first_hash != second_hash


def test_tlx_wave_driver_load_binary_delegates_hsaco():
    calls = []

    class FakeHIPUtils:
        def load_binary(self, name, kernel, shared, device):
            calls.append((name, kernel, shared, device))
            return "module", "function", 10, 0, 1024

    utils = tlx_wave_driver._TLXWaveUtils(FakeHIPUtils())

    result = utils.load_binary("kernel_name", b"\x7fELFpayload", 128, 0)

    assert result == ("module", "function", 10, 0, 1024)
    assert calls == [("kernel_name", b"\x7fELFpayload", 128, 0)]


def test_tlx_wave_driver_load_binary_rejects_wave_text():
    class FakeHIPUtils:
        def load_binary(self, name, kernel, shared, device):
            raise AssertionError("non-HSACO artifact should not reach HIP")

    utils = tlx_wave_driver._TLXWaveUtils(FakeHIPUtils())

    with pytest.raises(RuntimeError, match="expected HSACO bytes"):
        utils.load_binary("kernel_name", "gpu.module @kernels {}", 0, 0)

    with pytest.raises(RuntimeError, match="expected an ELF HSACO object"):
        utils.load_binary("kernel_name", b"gpu.module @kernels {}", 0, 0)


def test_tlx_wave_backend_compile_uses_staged_converter():
    src = ASTSource(fn=_tlx_wave_stage_only_kernel, signature={}, constexprs={})

    compiled = triton_compile(src, target=GFX950_WAVE)
    wave_artifact = _asm_text(compiled, "wave")
    hsaco = compiled.asm["hsaco"]

    assert "tlx_wave.new_converter" in wave_artifact
    assert "gpu.module @kernels" in wave_artifact
    assert "gpu.kernel" in wave_artifact
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert compiled.kernel == hsaco
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_binary_stage == "wave-compile-kernels"
    assert compiled.metadata.tlx_wave_hsaco_size_bytes == len(hsaco)
    assert compiled.metadata.tlx_wave_bridge_stage == "staged-converter"
    assert compiled.metadata.tlx_wave_wave_builder == "staged-converter"
    assert compiled.metadata.tlx_wave_plan_kind == "staged-converter"
    assert compiled.metadata.tlx_wave_num_kernel_args == 0
    assert compiled.metadata.tlx_wave_plan_num_ops >= 1
    _run_wave_verify(wave_artifact)
    binary_module = _run_wave_compile_kernels(wave_artifact)
    assert "gpu.binary @kernels" in binary_module
    assert "gpu.module @kernels" not in binary_module


def test_tlx_wave_backend_compile_lowers_masked_global_load_store():
    src = ASTSource(
        fn=_tlx_wave_add_one_kernel,
        signature={
            "x": "*fp32",
            "y": "*fp32",
            "n": "i32",
            "BLOCK": "constexpr",
        },
        constexprs={"BLOCK": 64},
        attrs={
            (0,): [["tt.pointer_range", 32]],
            (1,): [["tt.pointer_range", 32]],
        },
    )

    compiled = triton_compile(src, target=GFX950_WAVE)
    wave_artifact = _asm_text(compiled, "wave")
    hsaco = compiled.asm["hsaco"]

    assert "wave.where" in wave_artifact
    assert "wave.fadd" in wave_artifact
    assert "wave.load" in wave_artifact
    assert "wave.store" in wave_artifact
    assert "waveamd.make_buffer" in wave_artifact
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_binary_stage == "wave-compile-kernels"


def test_tlx_wave_runtime_launches_no_memory_kernel():
    torch, _arch = _require_tlx_wave_runtime_target()

    with _active_tlx_wave_driver():
        _tlx_wave_stage_only_kernel[(1,)]()
        torch.cuda.synchronize()


def test_tlx_wave_runtime_launches_masked_global_memory_kernel():
    torch, _arch = _require_tlx_wave_runtime_target()
    block = 64
    n = 37
    x = torch.arange(block, device="cuda", dtype=torch.float32)
    y = torch.full((block,), -1.0, device="cuda", dtype=torch.float32)
    expected = torch.full_like(y, -1.0)
    expected[:n] = x[:n] + 1.0

    with _active_tlx_wave_driver():
        _tlx_wave_add_one_kernel[(1,)](x, y, n, BLOCK=block)
        torch.cuda.synchronize()

    torch.testing.assert_close(y, expected)


def test_tlx_wave_converter_pipeline_lowers_program_id(tmp_path):
    local_func = """
  tt.func public @converter_program_id() attributes {noinline = false} {
    %pid = tt.get_program_id x : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["program_id", "return"]
    assert "wave.workgroup_id" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_pure_value_if(tmp_path):
    local_func = """
  tt.func public @converter_pure_if(%arg0: i32) attributes {noinline = false} {
    %zero = arith.constant 0 : i32
    %positive = arith.cmpi sgt, %arg0, %zero : i32
    %value = scf.if %positive -> (i32) {
      %one = arith.constant 1 : i32
      %then = arith.addi %arg0, %one : i32
      scf.yield %then : i32
    } else {
      %two = arith.constant 2 : i32
      %else = arith.addi %arg0, %two : i32
      scf.yield %else : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "constant",
        "cmpi",
        "constant",
        "binary",
        "constant",
        "binary",
        "select",
        "return",
    ]
    assert "wave.select" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_signed_min(tmp_path):
    local_func = """
  tt.func public @converter_signed_min(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %min = arith.minsi %arg0, %arg1 : i32
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["minsi", "return"]
    assert "arith.cmpi" in output.emitted_module.text
    assert "wave.select" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_local_alloc(tmp_path):
    preamble = """
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_local_alloc() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["local_alloc", "return"]
    local_alloc = output.target_program.ops[0]
    assert converter_target_ir.attrs_dict(local_alloc) == {
        "allocation_bytes": 128,
        "byte_offset": 0,
        "element_type": "f16",
        "shape": (64,),
    }
    assert output.emitted_module.lds_size == 128
    assert "wave.lds_size = 128 : i64" in output.emitted_module.text
    assert "wave.lds_base" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_memdesc_index(tmp_path):
    preamble = """
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_memdesc_index() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x64xf16, #shared, #smem, mutable>
    %slot = arith.constant 1 : i32
    %view = ttg.memdesc_index %alloc[%slot] : !ttg.memdesc<2x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "local_alloc",
        "constant",
        "memdesc_index",
        "return",
    ]
    assert converter_target_ir.attrs_dict(output.target_program.ops[2]) == {
        "elements_per_slot": 64,
        "static_lds_byte_offset": 128,
    }
    assert output.emitted_module.lds_size == 256
    assert "wave.lds_size = 256 : i64" in output.emitted_module.text
    assert "{offset = 128 : i64}" in output.emitted_module.text
    assert "wave.ptr_add" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_buffer_load_to_local_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "local_alloc",
        "make_range",
        "buffer_load_to_local",
        "async_commit_group",
        "async_wait",
        "return",
    ]
    attrs = converter_target_ir.attrs_dict(output.target_program.ops[2])
    assert attrs["mode"] == "dma_packet_lds"
    assert attrs["component_count"] == 1
    assert attrs["destination_component_offsets"] == (0,)
    assert attrs["packet_bytes"] == 16
    assert attrs["packet_elements"] == 8
    assert attrs["range_bytes"] == 2147483647
    assert attrs["source_offset_range"] == (0, 1073741816)
    assert "waveamd.make_buffer" in output.emitted_module.text
    assert "wave.assume" in output.emitted_module.text
    assert "wave.load" not in output.emitted_module.text
    assert "wave.store" not in output.emitted_module.text
    assert "waveamd.dma_load_lds" in output.emitted_module.text
    assert "wave.wait" in output.emitted_module.text
    assert "wave.barrier" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_masked_buffer_store_with_oob_select(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_buffer_store(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 1 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op,) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["has_mask"] is True
    assert attrs["mask_mode"] == "select_oob_offset"
    assert attrs["inactive_offset"] == 1073741824
    assert attrs["offset_range"] == (0, 1073741823)
    assert "wave.where" not in output.emitted_module.text
    assert output.emitted_module.text.count("wave.ptr_add") == 2
    assert "wave.select" in output.emitted_module.text

    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.buffer_store_b16" in machine
    assert "waveamdmachine.exec_if" not in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_masked_buffer_load_with_other(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_buffer_load(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f32> {tt.pointer_range = 32 : i32},
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask, %other {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    amdg.buffer_store %loaded, %arg1[%range], %mask {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op,) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert attrs["has_mask"] is True
    assert attrs["has_other"] is True
    assert attrs["mask_mode"] == "exec_where"
    assert attrs["inactive_offset"] == 536870912
    assert attrs["offset_range"] == (0, 536870911)
    assert output.emitted_module.text.count("waveamd.make_buffer") == 2
    assert output.emitted_module.text.count("wave.load") == 1
    assert "wave.where" in output.emitted_module.text
    assert "wave.select" in output.emitted_module.text

    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.buffer_load_b32" in machine
    assert "waveamdmachine.buffer_store_b32" in machine
    assert "waveamdmachine.exec_if" in machine
    del ctx


def test_tlx_wave_converter_masks_buffer_load_offset_assumes(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_small_buffer_load(
      %arg0: !tt.ptr<f32> {tt.pointer_range = 3 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %one = arith.constant dense<1> : tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %one : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask, %other {contiguity = 1 : i32} : tensor<64xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op,) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert attrs["offset_range"] == (0, 0)
    assert attrs["inactive_offset"] == 1
    wave = output.emitted_module.text
    assert wave.index("wave.where") < wave.index("wave.assume") < wave.index("wave.load")
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b32" in machine
    del ctx


def test_tlx_wave_converter_pipeline_groups_mult_warp_padded_dma(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.padded_shared<[512:+16] {order = [0], shape = [4096]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_padded_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4096xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<4096xi32, #blocked>] -> <4096xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    attrs = converter_target_ir.attrs_dict(output.target_program.ops[2])
    assert attrs["mode"] == "dma_packet_lds"
    assert attrs["component_count"] == 1
    assert attrs["component_thread_count"] == 512
    assert attrs["destination_component_offsets"] == (0,)
    assert attrs["destination_wave_count"] == 8
    assert attrs["destination_wave_stride_dwords"] == 264
    assert output.emitted_module.text.count("waveamd.dma_load_lds") == 1
    assert "wave.read_first" in output.emitted_module.text
    assert "wave.index_expr" not in output.emitted_module.text
    assert "wave.binary shrui" in output.emitted_module.text
    assert "wave.binary muli" in output.emitted_module.text
    assert "c264_i32" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_warp_tiled_mfma_dot(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_warp_tiled_mfma_dot() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<256x64xf16, #dot0> * tensor<64x128xf16, #dot1> -> tensor<256x128xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    attrs_by_kind = {
        op.kind: converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind in {"fragment_fill", "mma"}
    }
    assert attrs_by_kind["fragment_fill"]["component_count"] == 16
    assert attrs_by_kind["mma"]["m_tiles"] == 4
    assert attrs_by_kind["mma"]["n_tiles"] == 4
    assert attrs_by_kind["mma"]["k_tiles"] == 2
    local_load_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "local_load_fragment"
    ]
    assert [attrs["component_count"] for attrs in local_load_attrs] == [8, 8]
    assert [attrs["wave_tile_axis"] for attrs in local_load_attrs] == ["m", "n"]
    assert output.emitted_module.text.count("waveamd.fragment_fill") == 16
    assert output.emitted_module.text.count('waveamd.mma "mfma.f32.16x16x32.f16"') == 32
    del ctx


def test_tlx_wave_converter_pipeline_lowers_mfma32_transpose_load(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_mfma32_transpose_load() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #dot1>
    %acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<32x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    local_load_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "local_load_fragment"
    ]

    assert [attrs["load_mode"] for attrs in local_load_attrs] == [
        "swizzled_fragment_load",
        "b16_transpose",
    ]
    assert output.emitted_module.text.count("wave.load") == 2
    assert output.emitted_module.text.count("waveamd.transpose_load") == 4
    assert output.emitted_module.text.count("wave.pack") == 2
    assert output.emitted_module.text.count('waveamd.mma "mfma.f32.32x32x16.f16"') == 2
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert machine.count("waveamdmachine.ds_load_tuple_b32") == 2
    assert machine.count("waveamdmachine.ds_read_tr_b64_b16") == 4
    del ctx


def test_tlx_wave_converter_records_b16_transpose_chunk_deltas(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.padded_shared<[512:+16] {order = [1, 0], shape = [64, 128]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_b16_transpose_chunk_deltas() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %rhs = ttg.local_load %alloc : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #dot1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    local_load_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "local_load_fragment"
    ]

    assert len(local_load_attrs) == 1
    assert local_load_attrs[0]["load_mode"] == "b16_transpose"
    assert local_load_attrs[0]["chunk_element_deltas"] == ((0, 4),) * 8
    assert output.emitted_module.text.count("waveamd.transpose_load") == 16
    del ctx


def test_tlx_wave_converter_pipeline_lowers_same_representation_expand_dims(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_expand_dims() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #slice>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<64xi32, #slice> -> tensor<1x64xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "make_range",
        "expand_dims",
        "return",
    ]
    assert "tt.expand_dims" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_blocked_broadcast(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_broadcast() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %expanded = tt.expand_dims %range {axis = 1 : i32} : tensor<128xi32, #slice> -> tensor<128x1xi32, #blocked>
    %broadcast = tt.broadcast %expanded : tensor<128x1xi32, #blocked> -> tensor<128x2xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "make_range",
        "expand_dims",
        "broadcast",
        "return",
    ]
    assert "tt.broadcast" not in output.emitted_module.text
    del ctx


@triton.jit
def _tlx_wave_stage_only_kernel():
    pid = tl.program_id(0)
    tl.assume(pid >= 0)


@triton.jit
def _tlx_wave_add_one_kernel(x, y, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(x + offs, mask=mask, other=0.0)
    tl.store(y + offs, vals + 1.0, mask=mask)


def _minimal_ttgir(
    public_funcs,
    target="hip:gfx950",
    threads_per_warp=64,
    num_ctas=1,
    num_warps=4,
    preamble="",
):
    return f"""
{preamble}
module attributes {{tlx.has_explicit_local_mem_access = true, "ttg.num-ctas" = {num_ctas} : i32, "ttg.num-warps" = {num_warps} : i32, ttg.target = "{target}", "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
{public_funcs}
}}
"""


def _tlx_wave_options(arch="gfx950", warp_size=64):
    return SimpleNamespace(arch=arch, warp_size=warp_size)


def test_tlx_wave_backend_defaults_and_accepts_mfma_options():
    backend = make_backend(GFX950_WAVE)

    assert backend.parse_options({}).matrix_instr_nonkdim == 0
    assert backend.parse_options({"matrix_instr_nonkdim": 32}).matrix_instr_nonkdim == 32
    with pytest.warns(UserWarning, match="kpack is deprecated"):
        assert backend.parse_options({"kpack": 2}).kpack == 1
    assert make_backend(GFX942_WAVE).parse_options({}).matrix_instr_nonkdim == 0
    assert make_backend(GFX942_WAVE).parse_options({"kpack": 2}).kpack == 2


def _parse_ttgir(
    tmp_path,
    public_funcs,
    target="hip:gfx950",
    threads_per_warp=64,
    num_ctas=1,
    num_warps=4,
    preamble="",
):
    ctx = ir.context()
    ir.load_dialects(ctx)
    make_backend(GFX950_WAVE).load_dialects(ctx)
    path = tmp_path / "tlx_wave_test.mlir"
    path.write_text(
        _minimal_ttgir(
            public_funcs,
            target,
            threads_per_warp,
            num_ctas,
            num_warps,
            preamble,
        )
    )
    return ir.parse_mlir_module(str(path), ctx), ctx


def _run_waveamd_to_machine(wave_artifact):
    result = subprocess.run(
        [wave_bridge_tools._wave_opt(), "-", "--waveamd-to-machine"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_verify(wave_artifact):
    result = subprocess.run(
        [wave_bridge_tools._wave_opt(), "-", "--verify-diagnostics"],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout


def _run_wave_compile_kernels(wave_artifact):
    result = subprocess.run(
        [
            wave_bridge_tools._wave_opt(),
            "-",
            "--waveamd-to-machine",
            "--waveamd-abi-lowering",
            "--waveamd-reg-alloc",
            "--waveamd-insert-hazard-waits",
            "--waveamd-resource-info",
            "--waveamd-metadata",
            "--wave-compile-kernels=features=",
        ],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout
