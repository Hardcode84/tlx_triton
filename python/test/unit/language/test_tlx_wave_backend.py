import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

import triton
from triton._C.libtriton import ir
import triton.language as tl
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, compile as triton_compile, make_backend
from triton.runtime.jit import MockTensor

if "tlx_wave" in backends:
    from triton.backends.tlx_wave import compiler as tlx_wave_compiler
    from triton.backends.tlx_wave import driver as tlx_wave_driver
    from triton.backends.tlx_wave import wave_bridge_tools
    from triton.backends.tlx_wave.converter import diagnostics as converter_diagnostics
    from triton.backends.tlx_wave.converter import canonicalize as converter_canonicalize
    from triton.backends.tlx_wave.converter import domains as converter_domains
    from triton.backends.tlx_wave.converter import emission as converter_emission
    from triton.backends.tlx_wave.converter import facts as converter_facts
    from triton.backends.tlx_wave.converter import coordinates as converter_coordinates
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
    converter_domains = None
    converter_emission = None
    converter_facts = None
    converter_coordinates = None
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


def _tlx_wave_physical_arch(properties):
    return str(properties.get("arch", "")).split(":")[0]


def _tlx_wave_runtime_skip_reason(arch):
    supported = "/".join(sorted(_TLX_WAVE_RUNTIME_ARCHES))
    return (
        f"requires physical {supported} hardware for TLX Wave launch tests, "
        f"got {arch or 'unknown'}; this is a runtime launch guard, not a "
        "Wave HSACO generation failure. Compile-only TLX Wave tests may target "
        "gfx942/gfx950 without matching local hardware."
    )


def _require_tlx_wave_runtime_target():
    torch = pytest.importorskip("torch")
    try:
        active_driver = triton.runtime.driver.active
        device = active_driver.get_current_device()
        properties = active_driver.utils.get_device_properties(device)
    except Exception as exc:
        pytest.skip(f"requires an active HIP runtime for TLX Wave launch tests: {exc}")
    arch = _tlx_wave_physical_arch(properties)
    if arch not in _TLX_WAVE_RUNTIME_ARCHES:
        pytest.skip(_tlx_wave_runtime_skip_reason(arch))
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


@contextmanager
def _active_amd_driver():
    from triton.backends.amd import driver as amd_driver

    previous_driver = triton.runtime.driver.active
    triton.runtime.driver.set_active(amd_driver.HIPDriver())
    try:
        yield
    finally:
        triton.runtime.driver.set_active(previous_driver)


@contextmanager
def _tlx_wave_compile_driver(monkeypatch):
    previous_default = triton.runtime.driver._default
    previous_active = triton.runtime.driver._active
    monkeypatch.setenv("TRITON_DEFAULT_BACKEND", "tlx_wave")
    try:
        triton.runtime.driver._default = None
        triton.runtime.driver._active = None
        active_driver = triton.runtime.driver.active
    except RuntimeError as exc:
        pytest.skip(f"requires active TLX Wave compile driver: {exc}")
    try:
        yield active_driver
    finally:
        triton.runtime.driver._default = previous_default
        triton.runtime.driver._active = previous_active


def _load_tlx_gfx9_gemm_module(version_dir, module_name=None):
    repo_root = Path(__file__).resolve().parents[4]
    kernel_path = (
        repo_root
        / "third_party"
        / "tlx"
        / "tutorials"
        / "gfx9_gemm"
        / "a16w16"
        / version_dir
        / "matmul_kernel.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name or f"_tlx_wave_test_{version_dir}",
        kernel_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tlx_gfx9_gemm_kernel(version_dir, function_name):
    module = _load_tlx_gfx9_gemm_module(
        version_dir,
        f"_tlx_wave_test_{version_dir}_{function_name}",
    )
    return getattr(module, function_name)


def _compile_tlx_gfx9_gemm_kernel(tmp_path, monkeypatch, case):
    torch = pytest.importorskip("torch")
    if case.get("disables_post_misched", False):
        monkeypatch.setenv("TRITON_DISABLE_POST_MISCHED", "1")
    kernel = _load_tlx_gfx9_gemm_kernel(case["version_dir"], case["function_name"])

    m = n = k = 256
    a = MockTensor(torch.float16, [m, k])
    b = MockTensor(torch.float16, [k, n])
    c = MockTensor(torch.float16, [m, n])
    a_strides = a.stride()
    b_strides = b.stride()
    c_strides = c.stride()

    with (
        _tlx_wave_compile_driver(monkeypatch),
        triton.knobs.cache.scope(),
        triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{case['version_dir']}-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        return kernel.warmup(
            a,
            b,
            c,
            m,
            n,
            k,
            a_strides[0],
            a_strides[1],
            b_strides[0],
            b_strides[1],
            c_strides[0],
            c_strides[1],
            BLOCK_M=256,
            BLOCK_N=256,
            BLOCK_K=64,
            num_warps=case["num_warps"],
            num_stages=1,
            matrix_instr_nonkdim=16,
            grid=(1,),
            **case.get("extra_meta", {}),
        )


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
    assert "wave_bridge" not in converter_domains.__dict__
    assert "wave_bridge" not in converter_canonicalize.__dict__
    assert "wave_bridge" not in converter_source_import.__dict__
    assert "wave_bridge" not in converter_tokens.__dict__
    assert "wave_bridge" not in converter_target_ir.__dict__
    assert "wave_bridge" not in converter_op_conversion.__dict__
    assert "wave_bridge" not in converter_verifier.__dict__
    assert "wave_bridge" not in converter_emission.__dict__
    assert "wave_bridge" not in converter_pipeline.__dict__


def test_tlx_wave_converter_lowering_domains_are_pure_policy():
    tree = ast.parse(
        Path(converter_domains.__file__).read_text(),
        filename=converter_domains.__file__,
    )
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)

    assert imports == ["dataclasses"]


def test_tlx_wave_converter_lowering_domains_cover_dispatch():
    assert converter_domains.DOMAIN_NAMES == (
        "arithmetic_control",
        "memory_dma",
        "generic_memory",
        "local_memory_layout",
        "mfma_fragment",
        "store_epilogue",
    )
    assert converter_domains.source_domains_for_op("arith.addi") == (
        "arithmetic_control",
    )
    assert converter_domains.source_domains_for_op("arith.constant") == (
        "arithmetic_control",
        "mfma_fragment",
    )
    assert converter_domains.source_domains_for_op("amdg.buffer_load_to_local") == (
        "memory_dma",
    )
    assert converter_domains.source_domains_for_op("tt.load") == ("generic_memory",)
    assert converter_domains.source_domains_for_op("tt.store") == ("generic_memory",)
    assert converter_domains.source_domains_for_op("rocdl.sched.barrier") == (
        "arithmetic_control",
    )
    assert converter_domains.target_domain_for_op("local_load_fragment") == (
        "local_memory_layout"
    )
    assert converter_domains.target_domain_for_op("mma") == "mfma_fragment"
    assert converter_domains.target_domain_for_op("load") == "generic_memory"
    assert converter_domains.target_domain_for_op("store") == "generic_memory"
    assert converter_domains.target_domain_for_op("buffer_store") == "store_epilogue"
    assert (
        converter_op_conversion._SUPPORTED_SOURCE_OPS
        == converter_domains.all_source_ops()
    )
    assert set(converter_emission._TARGET_EMITTERS) == (
        converter_domains.all_target_ops()
    )


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


def test_tlx_wave_converter_make_range_uses_slice_coordinates(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_slice_make_range() attributes {noinline = false} {
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #slice>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)
    token_program = converter_tokens.build_token_program(source, converted)

    target = converter_op_conversion.convert_ops(
        source,
        converted,
        fact_program,
        token_program,
    )

    (range_op,) = [op for op in target.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    assert attrs["coordinate_mode"] == "bit_affine_workitem"
    assert attrs["component_bases"] == (0,)
    assert attrs["workitem_coefficients"] == (1, 2, 4, 8, 16, 0)
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


def test_tlx_wave_converter_fact_stage_invalidates_convert_layout_affine(tmp_path):
    preamble = """
#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_fact_layout_invalidation() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked0>
    %converted = ttg.convert_layout %range : tensor<128xi32, #blocked0> -> tensor<128xi32, #blocked1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)

    convert_op = next(op for op in source.ops if op.name == "ttg.convert_layout")
    source_value_id = convert_op.operands[0]
    result_value_id = convert_op.results[0]
    assert converted.values[source_value_id].layout_map_id != converted.values[
        result_value_id
    ].layout_map_id
    assert source_value_id in fact_program.tensor_affine
    assert result_value_id not in fact_program.tensor_affine
    assert converter_facts.facts_for_value(fact_program, result_value_id) == ()
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


def test_tlx_wave_converter_token_stage_uses_memory_frontier(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_memory_frontier(%ptr: tensor<64x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %loaded0 = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    %loaded1 = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %ptr, %loaded0 : tensor<64x!tt.ptr<f32>, #blocked>
    %loaded2 = tt.load %ptr : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)

    token_program = converter_tokens.build_token_program(source, converted)

    read0, read1, write, read2 = token_program.memory_effects
    assert (read0.kind, read1.kind, write.kind, read2.kind) == (
        "read",
        "read",
        "write",
        "read",
    )
    assert read0.depends_on_effect_ids == ()
    assert read1.depends_on_effect_ids == ()
    assert write.depends_on_effect_ids == (read0.effect_id, read1.effect_id)
    assert read2.depends_on_effect_ids == (write.effect_id,)
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


def test_tlx_wave_converter_token_stage_treats_global_and_buffer_as_may_alias():
    value_type = converter_source_ir.SourceType(
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
    tensor_pointer_type = converter_source_ir.SourceType(
        "tensor<64x!tt.ptr<f32>>",
        "tensor",
        shape=(64,),
        element_type="!tt.ptr<f32>",
        address_space=1,
    )
    offset_type = converter_source_ir.SourceType(
        "tensor<64xi32>",
        "tensor",
        shape=(64,),
        element_type="i32",
    )
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("global_buffer_alias"),
        (
            converter_source_ir.SourceOp(
                0,
                "tt.load",
                operands=(1,),
                results=(4,),
            ),
            converter_source_ir.SourceOp(
                1,
                "amdg.buffer_store",
                operands=(5, 2, 3),
                attrs={
                    "cacheModifier": "none",
                    "operandSegmentSizes": (1, 1, 1, 0, 0),
                },
            ),
        ),
        {
            1: converter_source_ir.SourceValue(1, tensor_pointer_type, producer_name="arg0"),
            2: converter_source_ir.SourceValue(2, pointer_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, offset_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, value_type, producer_name="load"),
            5: converter_source_ir.SourceValue(5, value_type, producer_name="arg3"),
        },
        (converter_source_ir.SourceRegion(0, (0, 1)),),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    load_effect, store_effect = token_program.memory_effects
    assert (load_effect.address_space, store_effect.address_space) == (
        "global",
        "buffer",
    )
    assert store_effect.depends_on_effect_ids == (load_effect.effect_id,)


def test_tlx_wave_converter_token_stage_treats_unknown_space_as_may_alias():
    value_type = converter_source_ir.SourceType(
        "tensor<64xf32>",
        "tensor",
        shape=(64,),
        element_type="f32",
    )
    unknown_tensor_pointer_type = converter_source_ir.SourceType(
        "tensor<64x!tt.ptr<f32>>",
        "tensor",
        shape=(64,),
        element_type="!tt.ptr<f32>",
    )
    buffer_pointer_type = converter_source_ir.SourceType(
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
    program = converter_source_ir.SourceProgram(
        converter_source_ir.KernelInfo("unknown_alias"),
        (
            converter_source_ir.SourceOp(
                0,
                "tt.load",
                operands=(1,),
                results=(4,),
            ),
            converter_source_ir.SourceOp(
                1,
                "amdg.buffer_store",
                operands=(5, 2, 3),
                attrs={
                    "cacheModifier": "none",
                    "operandSegmentSizes": (1, 1, 1, 0, 0),
                },
            ),
            converter_source_ir.SourceOp(
                2,
                "tt.store",
                operands=(1, 5),
            ),
        ),
        {
            1: converter_source_ir.SourceValue(
                1,
                unknown_tensor_pointer_type,
                producer_name="arg0",
            ),
            2: converter_source_ir.SourceValue(2, buffer_pointer_type, producer_name="arg1"),
            3: converter_source_ir.SourceValue(3, offset_type, producer_name="arg2"),
            4: converter_source_ir.SourceValue(4, value_type, producer_name="load"),
            5: converter_source_ir.SourceValue(5, value_type, producer_name="arg3"),
        },
        (converter_source_ir.SourceRegion(0, (0, 1, 2)),),
        0,
    )

    token_program = converter_tokens.build_token_program(program, None)

    unknown_read, buffer_write, unknown_write = token_program.memory_effects
    assert unknown_read.address_space == "unknown"
    assert buffer_write.address_space == "buffer"
    assert unknown_write.address_space == "unknown"
    assert buffer_write.depends_on_effect_ids == (unknown_read.effect_id,)
    assert unknown_write.depends_on_effect_ids == (
        unknown_read.effect_id,
        buffer_write.effect_id,
    )


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
    assert assume_op.fact_target_ids
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
    assert add_op.fact_target_ids == (output.target_program.kernel.arg_target_ids[0],)
    wave = output.emitted_module.text
    assert wave.index("wave.assume %arg0") < wave.index("wave.binary addi")
    del ctx


def test_tlx_wave_converter_emits_facts_without_source_provenance(tmp_path):
    local_func = """
  tt.func public @converter_stripped_facts(%arg0: i32) attributes {noinline = false} {
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
    stripped_values = tuple(
        converter_target_ir.TargetValue(value.target_value_id, value.type)
        for value in output.target_program.values
    )
    stripped_target = converter_target_ir.TargetProgram(
        stripped_values,
        output.target_program.ops,
        output.target_program.regions,
        {},
        {},
        output.target_program.kernel,
    )

    stripped_emitted = converter_emission.emit_wave_module(
        stripped_target,
        output.fact_program,
    )

    assert stripped_emitted.text == output.emitted_module.text
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


def test_tlx_wave_converter_pipeline_ignores_sched_barrier(tmp_path):
    local_func = """
  tt.func public @converter_sched_barrier() attributes {noinline = false} {
    rocdl.sched.barrier 0 {triton.warp_pipeline.border = "stage0", triton.warp_pipeline.priority = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == ["return"]
    assert "rocdl.sched.barrier" not in output.emitted_module.text
    assert "wave.barrier" not in output.emitted_module.text
    assert "s_barrier" not in output.emitted_module.text
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


def test_tlx_wave_converter_op_stage_lowers_generic_load():
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

    target = converter_op_conversion.convert_ops(program, converted, facts, tokens)

    (load_op,) = target.ops
    assert load_op.kind == "load"
    assert load_op.operands == (0,)
    assert load_op.results == (1,)
    assert converter_target_ir.attrs_dict(load_op) == {
        "component_count": 1,
        "element_type": "f32",
        "has_mask": False,
        "has_other": False,
        "lane_width": 64,
        "mask_mode": "none",
    }


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


def test_tlx_wave_converter_verifier_rejects_missing_fact_target():
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(0, scalar_i32, source_value_id=0),
        ),
        (
            converter_target_ir.TargetOp(
                0,
                "assume",
                fact_ids=(0,),
            ),
        ),
        (converter_target_ir.TargetRegion(0, (0,)),),
        {0: (0,)},
        {},
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "range", 0, "sge", lower=0),),
        {0: (0,)},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target, fact_program=fact_program)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_FACT_TARGET_COUNT"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_incompatible_fact_target():
    scalar_i32 = converter_target_ir.TargetType("scalar", "scalar", "i32")
    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(0, scalar_i32, source_value_id=0),
            converter_target_ir.TargetValue(1, scalar_i32, source_value_id=1),
        ),
        (
            converter_target_ir.TargetOp(
                0,
                "assume",
                fact_ids=(0,),
                fact_target_ids=(1,),
            ),
        ),
        (converter_target_ir.TargetRegion(0, (0,)),),
        {0: (0,), 1: (1,)},
        {},
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "range", 0, "sge", lower=0),),
        {0: (0,)},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(target, fact_program=fact_program)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_FACT_TARGET"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.target_value_id == 1
    assert diagnostic.fact_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_layout_convert_without_fact_policy():
    builder = converter_target_ir.TargetBuilder()
    tensor = converter_target_ir.TargetType("tensor", "simd", "f32", 64, 1)
    operand = builder.add_value(tensor, source_value_id=0)
    result = builder.add_value(tensor, source_value_id=1)
    builder.add_op(
        "layout_convert",
        operands=(operand,),
        results=(result,),
        attrs={"mode": "alias", "result_component_count": 1},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(builder.build())

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_LAYOUT_FACT_POLICY"
    assert diagnostic.stage == "verification"
    assert diagnostic.target_op_id == 0
    assert diagnostic.no_fallback is True


def test_tlx_wave_converter_verifier_rejects_invalidating_layout_convert_facts():
    builder = converter_target_ir.TargetBuilder()
    tensor = converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1)
    operand = builder.add_value(tensor, source_value_id=0)
    result = builder.add_value(tensor, source_value_id=1)
    builder.add_op(
        "layout_convert",
        operands=(operand,),
        results=(result,),
        attrs={
            "fact_policy": "invalidate_layout_sensitive",
            "mode": "same_lane_register_remap",
            "result_component_count": 1,
        },
        fact_ids=(0,),
        fact_target_ids=(operand,),
    )
    fact_program = converter_facts.FactProgram(
        (converter_facts.Fact(0, "range", 0, "signed_width", lower=0),),
        {0: (0,)},
    )

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_verifier.verify_target_program(
            builder.build(),
            fact_program=fact_program,
        )

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_VERIFY_LAYOUT_FACT_POLICY"
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


def test_tlx_wave_converter_emits_workgroup_shape_from_ttgir(tmp_path):
    local_func = """
  tt.func public @converter_workgroup_shape() attributes {noinline = false} {
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_warps=4,
        threads_per_warp=64,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)
    wave_artifact = output.emitted_module.text

    assert "wave.workgroup_size = array<i32: 256, 1, 1>" in wave_artifact
    assert "gpu.known_block_size = array<i32: 256, 1, 1>" in wave_artifact
    assert "wave.waves_per_workgroup = 4 : i64" in wave_artifact
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
    assert metadata["tlx_wave_workgroup_size"] == 64
    _run_wave_verify(wave_artifact)
    del ctx


def test_tlx_wave_backend_wave_stage_keeps_fixed_lds_out_of_launch_shared(
    tmp_path,
):
    preamble = """
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @backend_wave_stage_fixed_lds() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    metadata = {}

    wave_artifact = tlx_wave_compiler.TLXWaveBackend.make_wave(
        mod,
        metadata,
        _tlx_wave_options(),
    )

    assert "wave.lds_size = 128 : i64" in wave_artifact
    assert metadata["shared"] == 0
    assert metadata["tlx_wave_launch_shared_bytes"] == 0
    assert metadata["tlx_wave_lds_size_bytes"] == 128
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

    assert "stage8-staged-converter-hsaco-static-lds" in first_hash
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


def test_tlx_wave_runtime_skip_reason_is_test_only_targeting_policy():
    assert _TLX_WAVE_RUNTIME_ARCHES == {"gfx942", "gfx950"}
    assert _tlx_wave_physical_arch({"arch": "gfx1100:sramecc+:xnack-"}) == "gfx1100"

    reason = _tlx_wave_runtime_skip_reason("gfx1100")

    assert "physical gfx942/gfx950 hardware" in reason
    assert "runtime launch guard" in reason
    assert "not a Wave HSACO generation failure" in reason
    assert "Compile-only TLX Wave tests may target gfx942/gfx950" in reason


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


@pytest.mark.parametrize(
    "case",
    [
        {
            "version_dir": "v6_loop_unroll",
            "function_name": "v6_loop_unroll",
            "num_warps": 4,
            "expected_failure": (
                "waveamd-reg-alloc",
                "wave.lds_size = 135040",
            ),
        },
        {
            "version_dir": "v7_slice",
            "function_name": "v7_slice",
            "num_warps": 4,
        },
        {
            "version_dir": "v8_warp_pipeline",
            "function_name": "v8_warp_pipeline",
            "num_warps": 8,
            "disables_post_misched": True,
        },
        {
            "version_dir": "v9_beyond_hotloop",
            "function_name": "v9_beyond_hotloop",
            "num_warps": 8,
            "disables_post_misched": True,
            "extra_meta": {"GROUP_SIZE_M": 4, "NUM_XCDS": 8, "GRID_MN": 1},
        },
    ],
    ids=lambda case: case["version_dir"],
)
def test_tlx_wave_backend_compiles_gfx9_gemm_v6_to_v9_to_hsaco(
    tmp_path,
    monkeypatch,
    case,
):
    if "expected_failure" in case:
        try:
            compiled = _compile_tlx_gfx9_gemm_kernel(tmp_path, monkeypatch, case)
        except RuntimeError as exc:
            detail = str(exc)
            for expected in case["expected_failure"]:
                assert expected in detail
            return
    else:
        compiled = _compile_tlx_gfx9_gemm_kernel(tmp_path, monkeypatch, case)
    wave_artifact = _asm_text(compiled, "wave")
    hsaco = compiled.asm["hsaco"]

    assert "tlx_wave.new_converter" in wave_artifact
    assert "gpu.kernel" in wave_artifact
    assert isinstance(hsaco, bytes)
    assert hsaco.startswith(b"\x7fELF")
    assert compiled.kernel == hsaco
    assert compiled.metadata.tlx_wave_status == "emitted_wave_staged_converter"
    assert compiled.metadata.tlx_wave_binary_stage == "wave-compile-kernels"
    assert compiled.metadata.tlx_wave_hsaco_size_bytes == len(hsaco)
    assert compiled.metadata.tlx_wave_ttgir_target == "hip:gfx950"
    assert compiled.metadata.shared == 0
    assert compiled.metadata.tlx_wave_launch_shared_bytes == 0
    assert compiled.metadata.tlx_wave_lds_size_bytes > 0
    assert compiled.metadata.tlx_wave_num_mmas > 0
    assert compiled.metadata.tlx_wave_num_dma_load_lds > 0


@pytest.mark.parametrize(
    "case_name,b_layout",
    [
        ("contiguous_b", "contiguous"),
        pytest.param(
            "transposed_b",
            "transposed",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                reason=(
                    "v9 TLX Wave lowering currently miscomputes benchmark-style "
                    "B tensors with shape (K, N) and stride (1, K)"
                ),
                strict=True,
            ),
        ),
    ],
)
def test_tlx_wave_runtime_gfx950_v9_e2e(tmp_path, case_name, b_layout):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 v9 e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "v9_beyond_hotloop",
        f"_tlx_wave_v9_runtime_{case_name}",
    )

    m = n = k = 256
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    if b_layout == "contiguous":
        b = torch.randn((k, n), device=device, dtype=torch.float16)
    else:
        b = torch.randn((n, k), device=device, dtype=torch.float16).T
    assert b.shape == (k, n)

    with (
        _active_tlx_wave_driver(),
        triton.knobs.cache.scope(),
        triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / f"{case_name}-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b)
        torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=1e-1, rtol=0)


def test_tlx_runtime_gfx950_v9_amd_backend_transposed_b_e2e(tmp_path):
    torch, arch = _require_tlx_wave_runtime_target()
    if arch != "gfx950":
        pytest.skip(f"requires physical gfx950 hardware for gfx950 v9 e2e, got {arch}")
    tutorial = _load_tlx_gfx9_gemm_module(
        "v9_beyond_hotloop",
        "_tlx_v9_amd_backend_transposed_b",
    )

    m = n = k = 256
    device = torch.device("cuda")
    torch.manual_seed(0)
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((n, k), device=device, dtype=torch.float16).T
    assert b.shape == (k, n)
    assert b.stride() == (1, k)

    with (
        _active_amd_driver(),
        triton.knobs.cache.scope(),
        triton.knobs.runtime.scope(),
    ):
        triton.knobs.cache.dir = str(tmp_path / "amd-backend-transposed-b-cache")
        triton.knobs.runtime.override_arch = "gfx950"
        got = tutorial.matmul(a, b)
        torch.cuda.synchronize()

    expected = torch.matmul(a, b)
    torch.cuda.synchronize()
    torch.testing.assert_close(got, expected, atol=1e-1, rtol=0)


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


def test_tlx_wave_converter_pipeline_lowers_dynamic_for_with_iter_args(tmp_path):
    local_func = """
  tt.func public @converter_dynamic_for(%arg0: i32, %arg1: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg0 step %c1_i32 iter_args(%acc = %arg1) -> (i32)  : i32 {
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "constant",
        "constant",
        "binary",
        "for_loop",
        "return",
    ]
    for_op = next(op for op in output.target_program.ops if op.kind == "for_loop")
    assert len(for_op.region_ids) == 1
    assert len(output.target_program.regions[for_op.region_ids[0]].block_arg_ids) == 2
    assert "scf.for" in output.emitted_module.text
    assert "iter_args" in output.emitted_module.text
    assert "scf.yield" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_carries_mask_payload_across_dynamic_for(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_dynamic_for_mask_payload(%limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %init = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %carried = scf.for %i = %c0 to %c64 step %c1 iter_args(%mask = %init) -> (tensor<64xi1, #blocked>) {
      %next = arith.andi %mask, %init : tensor<64xi1, #blocked>
      scf.yield %next : tensor<64xi1, #blocked>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert "scf.for" in output.emitted_module.text
    assert "scf.yield" in output.emitted_module.text
    assert "arith.andi" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_normalizes_carried_mask_init_to_payload(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_dynamic_for_mask_constant_init(%limit: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %init = arith.constant dense<true> : tensor<64xi1, #blocked>
    %carried = scf.for %i = %c0 to %c64 step %c1 iter_args(%mask = %init) -> (tensor<64xi1, #blocked>) {
      %active = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
      %next = arith.andi %mask, %active : tensor<64xi1, #blocked>
      scf.yield %next : tensor<64xi1, #blocked>
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert "scf.for" in output.emitted_module.text
    assert "scf.yield" in output.emitted_module.text
    assert "arith.andi" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_nested_dynamic_for_with_iter_args(
    tmp_path,
):
    local_func = """
  tt.func public @converter_nested_dynamic_for(
      %arg0: i32,
      %arg1: i32,
      %arg2: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg0 step %c1_i32 iter_args(%outer_acc = %arg2) -> (i32)  : i32 {
      %inner = scf.for %j = %c0_i32 to %arg1 step %c1_i32 iter_args(%inner_acc = %outer_acc) -> (i32)  : i32 {
        %ij = arith.addi %i, %j : i32
        %next_inner = arith.addi %inner_acc, %ij : i32
        scf.yield %next_inner : i32
      }
      %next_outer = arith.addi %inner, %i : i32
      scf.yield %next_outer : i32
    }
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    for_ops = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert len(for_ops) == 2
    assert len(output.target_program.regions) == 3
    assert len(output.target_program.regions[for_ops[0].region_ids[0]].block_arg_ids) == 2
    assert len(output.target_program.regions[for_ops[1].region_ids[0]].block_arg_ids) == 2
    assert output.emitted_module.text.count("scf.for") == 2
    assert output.emitted_module.text.count("scf.yield") == 2
    del ctx


def test_tlx_wave_converter_pipeline_carries_async_token_across_dynamic_for(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_async_for(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %wait = ttg.async_wait {num = 1 : i32}
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    body_dma = [
        op
        for op in output.target_program.ops
        if op.kind == "buffer_load_to_local" and op.source_op_index is not None
    ][1]
    attrs = converter_target_ir.attrs_dict(body_dma)
    assert attrs["mode"] == "dma_packet_lds"
    assert attrs["issue_dependency_count"] == 1
    wave = output.emitted_module.text
    loop_match = re.search(
        r"scf\.for .*iter_args\(.*(?P<token_arg>%arg\d+) = %[^\)]*"
        r"\) -> \(.*!wave\.mem\.token\)",
        wave,
    )
    assert loop_match is not None
    token_arg = loop_match.group("token_arg")
    loop_body = wave[loop_match.end() :]
    assert "waveamd.dma_load_lds" in loop_body
    assert f"after {token_arg}" in loop_body
    assert f"wave.wait {token_arg}" in loop_body
    assert loop_body.index(f"after {token_arg}") < loop_body.index(
        f"wave.wait {token_arg}"
    )
    del ctx


def test_tlx_wave_converter_pipeline_carries_loop_issued_async_token_to_final_wait(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_async_for_final_wait(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg1 step %c1_i32 iter_args(%acc = %c0_i32) -> (i32)  : i32 {
      %body = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
      %body_group = ttg.async_commit_group tokens %body
      %next = arith.addi %acc, %i : i32
      scf.yield %next : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops].count("token") == 1
    (dma_op,) = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local"
    ]
    attrs = converter_target_ir.attrs_dict(dma_op)
    assert attrs["mode"] == "dma_packet_lds"
    assert attrs["issue_dependency_count"] == 0
    wave = output.emitted_module.text
    loop_match = re.search(
        r"scf\.for .*iter_args\(.*(?P<token_arg>%arg\d+) = %[^\)]*"
        r"\) -> \(.*!wave\.mem\.token\)",
        wave,
    )
    assert loop_match is not None
    loop_body = wave[loop_match.end() :]
    assert f"after {loop_match.group('token_arg')}" not in loop_body
    assert re.search(r"}\n\s+wave\.wait %\d+#\d+ : !wave\.mem\.token", wave)
    del ctx


def test_tlx_wave_converter_pipeline_carries_async_tokens_through_nested_for(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_nested_async_for(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg1: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %arg2: i32,
      %arg3: i32) attributes {noinline = false} {
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %warmup_a = amdg.buffer_load_to_local %arg0[%range] into %alloc_a : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_b = amdg.buffer_load_to_local %arg1[%range] into %alloc_b : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %warmup_group = ttg.async_commit_group tokens %warmup_a, %warmup_b
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %sum = scf.for %i = %c0_i32 to %arg2 step %c1_i32 iter_args(%outer_acc = %c0_i32) -> (i32)  : i32 {
      %inner = scf.for %j = %c0_i32 to %arg3 step %c1_i32 iter_args(%inner_acc = %outer_acc) -> (i32)  : i32 {
        %body_a = amdg.buffer_load_to_local %arg0[%range] into %alloc_a : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
        %body_b = amdg.buffer_load_to_local %arg1[%range] into %alloc_b : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
        %body_group = ttg.async_commit_group tokens %body_a, %body_b
        %wait = ttg.async_wait {num = 1 : i32}
        %ij = arith.addi %i, %j : i32
        %next_inner = arith.addi %inner_acc, %ij : i32
        scf.yield %next_inner : i32
      }
      scf.yield %inner : i32
    }
    %final_wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    for_ops = [op for op in output.target_program.ops if op.kind == "for_loop"]
    assert len(for_ops) == 2
    dma_ops = [
        op
        for op in output.target_program.ops
        if op.kind == "buffer_load_to_local" and op.source_op_index is not None
    ]
    assert len(dma_ops) == 4
    assert [
        converter_target_ir.attrs_dict(op)["issue_dependency_count"]
        for op in dma_ops
    ] == [0, 0, 1, 1]
    wave = output.emitted_module.text
    loop_matches = list(
        re.finditer(
            r"scf\.for .*iter_args\(.*(?P<token_arg>%arg\d+) = %[^\)]*"
            r"\) -> \(.*!wave\.mem\.token\)",
            wave,
        )
    )
    assert len(loop_matches) == 2
    inner_token_arg = loop_matches[-1].group("token_arg")
    inner_body = wave[loop_matches[-1].end() :]
    assert inner_body.count(f"after {inner_token_arg}") == 2
    assert inner_body.count(f"wave.wait {inner_token_arg}") == 1
    assert inner_body.index(f"after {inner_token_arg}") < inner_body.index(
        f"wave.wait {inner_token_arg}"
    )
    assert wave.count("waveamd.dma_load_lds") == 4
    assert wave.count("scf.for") == 2
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
        "element_byte_width": 2,
        "elements_per_slot": 64,
        "static_lds_byte_offset": 128,
    }
    assert output.emitted_module.lds_size == 256
    assert "wave.lds_size = 256 : i64" in output.emitted_module.text
    assert "{offset = 128 : i64}" in output.emitted_module.text
    assert "wave.ptr_add" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_pipeline_lowers_padded_memdesc_index_stride(tmp_path):
    preamble = """
#shared = #ttg.padded_shared<[512:+16] {order = [1, 0], shape = [256, 64]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_padded_memdesc_index() attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %view0 = ttg.memdesc_index %alloc[%zero] : !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %view1 = ttg.memdesc_index %alloc[%one] : !ttg.memdesc<2x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    local_alloc_attrs = converter_target_ir.attrs_dict(output.target_program.ops[0])
    assert local_alloc_attrs["allocation_bytes"] == 67520
    assert output.emitted_module.lds_size == 67520
    assert "wave.lds_size = 67520 : i64" in output.emitted_module.text

    first_view_attrs = converter_target_ir.attrs_dict(output.target_program.ops[3])
    second_view_attrs = converter_target_ir.attrs_dict(output.target_program.ops[4])
    assert first_view_attrs["elements_per_slot"] == 16880
    assert first_view_attrs["static_lds_byte_offset"] == 0
    assert second_view_attrs["elements_per_slot"] == 16880
    assert second_view_attrs["static_lds_byte_offset"] == 33760
    assert "{offset = 33760 : i64}" in output.emitted_module.text
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


def test_tlx_wave_converter_lowers_dynamic_memdesc_index_packet_dma_destination(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_dynamic_memdesc_dma(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<2x512xf16, #shared, #smem, mutable>
    %view = ttg.memdesc_index %alloc[%stage] : !ttg.memdesc<2x512xf16, #shared, #smem, mutable> -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %view : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "local_alloc",
        "memdesc_index",
        "make_range",
        "buffer_load_to_local",
        "async_commit_group",
        "async_wait",
        "return",
    ]
    memdesc_attrs = converter_target_ir.attrs_dict(output.target_program.ops[1])
    assert memdesc_attrs["element_byte_width"] == 2
    assert memdesc_attrs["elements_per_slot"] == 512
    assert memdesc_attrs["static_lds_byte_offset"] is None
    load_attrs = converter_target_ir.attrs_dict(output.target_program.ops[3])
    assert load_attrs["mode"] == "dma_packet_lds"
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_lds_b128" in machine
    del ctx


def test_tlx_wave_converter_lowers_masked_dma_eligible_buffer_load_to_local_fallback(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_buffer_load_to_local(%arg0: !tt.ptr<f16> {tt.pointer_range = 2 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<512xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %one = arith.constant dense<1> : tensor<512xi32, #blocked>
    %mask = arith.cmpi slt, %range, %one : tensor<512xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<512xi32, #blocked>] -> <512xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op,) = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local"
    ]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    assert attrs["mode"] == "scalarized_load_store"
    assert attrs["has_mask"] is True
    assert attrs["mask_mode"] == "exec_where"
    assert attrs["offset_range"] == (0, 0)
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert "wave.load" in wave
    assert "wave.store" in wave
    assert "wave.where" in wave
    assert wave.index("wave.where") < wave.index("wave.assume") < wave.index("wave.load")
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_lowers_masked_scalar_buffer_load_to_local_fallback(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_scalar_buffer_load_to_local(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %limit = arith.constant dense<32> : tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit : tensor<64xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op,) = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local"
    ]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    assert attrs["mode"] == "scalarized_load_store"
    assert attrs["has_mask"] is True
    assert attrs["component_count"] == 1
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert wave.count("wave.load") == 1
    assert wave.count("wave.store") == 1
    assert "wave.where" in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_lowers_splat_i1_buffer_load_to_local_mask(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_splat_i1_buffer_load_to_local_mask(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32},
      %stage: i32) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %two = arith.constant 2 : i32
    %three = arith.constant 3 : i32
    %active_a = arith.cmpi ne, %stage, %two : i32
    %active_b = arith.cmpi ne, %stage, %three : i32
    %mask_a = tt.splat %active_a : i1 -> tensor<64xi1, #blocked>
    %mask_b = tt.splat %active_b : i1 -> tensor<64xi1, #blocked>
    %mask = arith.andi %mask_a, %mask_b : tensor<64xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert "scf.if" in wave
    assert "wave.select" in wave
    assert "arith.andi" not in wave
    assert "i1 -> !wave.simd<i1" not in wave
    assert "wave.where" not in wave
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_scalarized_buffer_load_to_local_swizzled_order01(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 2, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_swizzled_scalarized_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xf16, #shared, #smem, mutable>
    %zero = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<8x8xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%zero] mask = %mask into %alloc : <f16>[tensor<8x8xi32, #blocked>] -> <8x8xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op,) = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local"
    ]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    assert attrs["mode"] == "scalarized_load_store"
    assert attrs["destination_offset_mode"] == "layout_coordinates"
    assert attrs["destination_coordinate_shape"] == (8, 8)
    assert attrs["destination_shared_layout"] == "swizzled"
    assert attrs["destination_swizzled_order"] == (0, 1)
    assert attrs["destination_swizzled_vec"] == 2
    assert "destination_component_offsets" not in attrs
    wave = output.emitted_module.text
    assert "waveamd.dma_load_lds" not in wave
    assert wave.count("wave.load") == 1
    assert wave.count("wave.store") == 1
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_b16" in machine
    del ctx


def test_tlx_wave_converter_rejects_unsupported_scalarized_swizzled_layout(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_rejects_swizzled_scalarized_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<8x8xf16, #shared, #smem, mutable>
    %zero = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<8x8xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%zero] mask = %mask into %alloc : <f16>[tensor<8x8xi32, #blocked>] -> <8x8xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_LOCAL_LOAD"
    message = str(diagnostic)
    assert "order=(0, 1)" in message
    assert "vec=4" in message
    assert "per_phase=1" in message
    assert "max_phase=4" in message
    del ctx


def test_tlx_wave_converter_rejects_buffer_load_to_local_other_fallback(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_other(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %true = arith.constant dense<true> : tensor<64xi1, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %true other = %other into %alloc : <f16>[tensor<64xi32, #blocked>] tensor<64xf16, #blocked> -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_BUFFER_ASYNC"
    assert "other fallback is not converted yet" in str(diagnostic)
    del ctx


def test_tlx_wave_converter_rejects_buffer_load_to_local_cache_modifier(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_cache(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<64xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %true = arith.constant dense<true> : tensor<64xi1, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %true cacheModifier = cv into %alloc : <f16>[tensor<64xi32, #blocked>] -> <64xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CACHE_MODIFIER"
    assert "cacheModifier=" in str(diagnostic)
    del ctx


def test_tlx_wave_converter_pipeline_joins_independent_dma_packets(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_buffer_load_to_local_join(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1024xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] into %alloc : <f16>[tensor<1024xi32, #blocked>] -> <1024xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    attrs = converter_target_ir.attrs_dict(output.target_program.ops[2])
    assert attrs["mode"] == "dma_packet_lds"
    assert attrs["component_count"] == 2
    assert attrs["destination_component_offsets"] == (0, 512)
    assert output.emitted_module.text.count("waveamd.dma_load_lds") == 2
    assert output.emitted_module.text.count("wave.join") == 2
    del ctx


def test_tlx_wave_converter_lowers_mult_warp_blocked_make_range_structurally(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
"""
    local_func = """
  tt.func public @converter_mult_warp_blocked_range() attributes {noinline = false} {
    %range = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op,) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    (result_id,) = range_op.results
    result_type = output.target_program.values[result_id].type
    assert result_type.component_count == 8
    assert attrs["coordinate_mode"] == "affine_workitem"
    assert attrs["component_bases"] == tuple(range(8))
    assert attrs["workitem_stride"] == 8
    assert "wave.binary muli" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_linear_make_range_with_block_basis(
    tmp_path,
):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = [[64]]}>
"""
    local_func = """
  tt.func public @converter_linear_range_block_basis() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(
        tmp_path,
        local_func,
        num_ctas=2,
        num_warps=1,
        preamble=preamble,
    )

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op,) = [op for op in output.target_program.ops if op.kind == "make_range"]
    (result_id,) = range_op.results
    assert output.target_program.values[result_id].type.component_count == 1
    assert "wave.workitem_id" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_bit_affine_linear_make_range(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[32], [16], [8], [4], [2], [1]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_bit_affine_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op,) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    assert attrs["coordinate_mode"] == "bit_affine_workitem"
    assert attrs["component_bases"] == (0,)
    assert attrs["workitem_coefficients"] == (32, 16, 8, 4, 2, 1)
    assert "wave.binary shrui" in output.emitted_module.text
    assert "wave.binary andi" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_replicated_generic_linear_make_range(tmp_path):
    preamble = """
#linear = #ttg.generic_linear<{register = [[1], [2]], lane = [[4], [8], [16], [32], [64], [0]], warp = [[64], [128]], block = []}>
"""
    local_func = """
  tt.func public @converter_replicated_generic_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    range_op = next(op for op in source.ops if op.name == "tt.make_range")
    value = converted.values[range_op.results[0]]
    layout = converted.layouts[value.layout_map_id]

    assert value.type.component_count == 4
    assert layout.properties["coordinate_domain"]["coverage"] == "replicated"
    assert layout.properties["coordinate_domain"]["covered_elements"] == 256
    assert layout.properties["coordinate_domain"]["duplicate_slots"] > 0

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (target_range_op,) = [
        op for op in output.target_program.ops if op.kind == "make_range"
    ]
    attrs = converter_target_ir.attrs_dict(target_range_op)
    assert attrs["coordinate_mode"] == "layout_coordinates"
    assert attrs["component_coordinate_bases"] == ((0,), (1,), (2,), (3,))
    assert attrs["workitem_coordinate_coefficients"] == (
        (4,),
        (8,),
        (16,),
        (32,),
        (64,),
        (0,),
        (64,),
        (128,),
    )
    assert "wave.binary xori" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_keeps_overlapping_xor_basis_out_of_affine_path(tmp_path):
    preamble = """
#linear = #ttg.generic_linear<{register = [[1]], lane = [[3], [6], [12], [24], [48], [96]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_overlapping_xor_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (range_op,) = [op for op in output.target_program.ops if op.kind == "make_range"]
    attrs = converter_target_ir.attrs_dict(range_op)
    assert attrs["coordinate_mode"] == "layout_coordinates"
    assert attrs["component_coordinate_bases"] == ((0,), (1,))
    assert attrs["workitem_coordinate_coefficients"] == (
        (3,),
        (6,),
        (12,),
        (24,),
        (48,),
        (96,),
    )
    assert "wave.binary xori" in output.emitted_module.text
    assert attrs.get("workitem_stride") is None
    del ctx


def test_tlx_wave_converter_materializes_rank2_blocked_coordinates(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
"""
    local_func = """
  tt.func public @converter_rank2_coordinate_layout() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<8x8xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    constant_op = next(op for op in source.ops if op.name == "arith.constant")
    value = converted.values[constant_op.results[0]]
    layout = converted.layouts[value.layout_map_id]

    plan = converter_coordinates.layout_coordinate_plan(
        layout,
        value.type.component_count,
        value.type.lane_width,
        1,
        constant_op,
        value.value_id,
    )

    assert plan.shape == (8, 8)
    assert plan.component_bases == ((0, 0),)
    assert plan.workitem_coefficients == (
        (1, 0),
        (2, 0),
        (4, 0),
        (0, 1),
        (0, 2),
        (0, 4),
    )

    target = converter_target_ir.TargetProgram(
        (
            converter_target_ir.TargetValue(
                0,
                converter_target_ir.TargetType("tensor", "simd", "i32", 64, 1),
            ),
        ),
        (
            converter_target_ir.TargetOp(
                0,
                "make_range",
                results=(0,),
                attrs=(
                    converter_target_ir.TargetAttr("start", 0),
                    converter_target_ir.TargetAttr("end", 64),
                    converter_target_ir.TargetAttr("coordinate_mode", "layout_coordinates"),
                    converter_target_ir.TargetAttr("coordinate_shape", plan.shape),
                    converter_target_ir.TargetAttr(
                        "component_coordinate_bases",
                        plan.component_bases,
                    ),
                    converter_target_ir.TargetAttr(
                        "workitem_coordinate_coefficients",
                        plan.workitem_coefficients,
                    ),
                ),
            ),
        ),
        (converter_target_ir.TargetRegion(0, (0,)),),
        {},
        {},
    )
    emitted = converter_emission.emit_wave_module(target)

    assert "wave.binary shrui" in emitted.text
    assert "wave.binary andi" in emitted.text
    assert "wave.binary muli" in emitted.text
    del ctx


def test_tlx_wave_converter_rejects_non_injective_linear_make_range(tmp_path):
    preamble = """
#linear = #ttg.linear<{register = [], lane = [[0], [0], [0], [0], [0], [0]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_non_injective_linear_range() attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_TYPE_UNSUPPORTED_LAYOUT"
    assert diagnostic.stage == "type_layout"
    text = str(diagnostic)
    assert "unsupported distributed layout coordinate domain duplicate_partial" in text
    assert "bases={'register': (), 'lane': ((0,), (0,), (0,), (0,), (0,), (0,))" in text
    del ctx


def test_tlx_wave_converter_lowers_blocked_to_linear_same_lane_payloads(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#linear = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_blocked_to_linear_payloads(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %converted_value = ttg.convert_layout %value : tensor<64xf32, #blocked> -> tensor<64xf32, #linear>
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<64xi1, #blocked>
    %converted_mask = ttg.convert_layout %mask : tensor<64xi1, #blocked> -> tensor<64xi1, #linear>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %converted_ptr = ttg.convert_layout %base : tensor<64x!tt.ptr<f32>, #blocked> -> tensor<64x!tt.ptr<f32>, #linear>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    convert_ops = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    assert len(convert_ops) == 3
    for convert_op in convert_ops:
        attrs = converter_target_ir.attrs_dict(convert_op)
        assert attrs["mode"] == "same_lane_register_remap"
        assert attrs["fact_policy"] == "invalidate_layout_sensitive"
        assert attrs["source_component_count"] == 1
        assert attrs["source_indices"] == (0,)
        assert attrs["source_element_indices"] == (0,)
        assert attrs["source_registers_per_component"] == 1
    assert "wave.shuffle" not in output.emitted_module.text
    assert "wave.extract" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_blocked_component_reorder(tmp_path):
    preamble = """
#source = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [1, 1, 64], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#result = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [1, 1, 64], warpsPerCTA = [1, 1, 1], order = [2, 0, 1]}>
"""
    local_func = """
  tt.func public @converter_blocked_component_reorder() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<2x2x64xi32, #source>
    %converted = ttg.convert_layout %value : tensor<2x2x64xi32, #source> -> tensor<2x2x64xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "same_lane_register_remap"
    assert attrs["source_component_count"] == 4
    assert attrs["source_indices"] == (0, 2, 1, 3)
    assert attrs["source_element_indices"] == (0, 0, 0, 0)
    assert "wave.shuffle" not in output.emitted_module.text
    assert "wave.extract" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_lowers_blocked_cross_lane_transpose(tmp_path):
    preamble = """
#source = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#result = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
"""
    local_func = """
  tt.func public @converter_blocked_cross_lane_transpose() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<8x8xi32, #source>
    %converted = ttg.convert_layout %value : tensor<8x8xi32, #source> -> tensor<8x8xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "cross_lane_register_remap"
    assert attrs["source_lane_map_kind"] == "transpose"
    assert attrs["source_lane_transpose_inner"] == 8
    assert attrs["source_lane_transpose_outer"] == 8
    assert attrs["source_lane_map"][:10] == (0, 8, 16, 24, 32, 40, 48, 56, 1, 9)
    assert output.emitted_module.text.count("wave.shuffle") == 1
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert machine.count("waveamdmachine.ds_bpermute_b32") == 1
    del ctx


def test_tlx_wave_converter_lowers_linear_alias_convert_layout(tmp_path):
    preamble = """
#source = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
#result = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_linear_alias() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<64xi32, #source>
    %converted = ttg.convert_layout %value : tensor<64xi32, #source> -> tensor<64xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "alias"
    assert attrs["fact_policy"] == "preserve_equivalent"
    assert "wave.shuffle" not in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_rejects_multi_warp_linear_remap(tmp_path):
    preamble = """
#source = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0]], block = []}>
#result = #ttg.generic_linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 1]], block = []}>
"""
    local_func = """
  tt.func public @converter_multi_warp_linear_remap() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<128x2xi32, #source>
    %converted = ttg.convert_layout %value : tensor<128x2xi32, #source> -> tensor<128x2xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=2, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT"
    assert "linear to linear convert_layout requires per-lane source component selection" in str(
        diagnostic
    )
    del ctx


def test_tlx_wave_converter_rejects_non_affine_linear_lane_remap(tmp_path):
    preamble = """
#source = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16], [32]], warp = [], block = []}>
#result = #ttg.linear<{register = [], lane = [[32], [16], [8], [4], [2], [1]], warp = [], block = []}>
"""
    local_func = """
  tt.func public @converter_non_affine_linear_remap() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<64xi32, #source>
    %converted = ttg.convert_layout %value : tensor<64xi32, #source> -> tensor<64xi32, #result>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT"
    assert "linear to linear convert_layout requires a non-affine source lane map" in str(
        diagnostic
    )
    del ctx


def test_tlx_wave_converter_rejects_slice_parent_layout_remap(tmp_path):
    preamble = """
#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice0 = #ttg.slice<{dim = 0, parent = #parent}>
#slice1 = #ttg.slice<{dim = 1, parent = #parent}>
"""
    local_func = """
  tt.func public @converter_slice_parent_layout_remap() attributes {noinline = false} {
    %value = arith.constant dense<0> : tensor<64xi32, #slice0>
    %converted = ttg.convert_layout %value : tensor<64xi32, #slice0> -> tensor<64xi32, #slice1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT"
    assert "slice to slice convert_layout requires parent layout movement support" in str(
        diagnostic
    )
    del ctx


def test_tlx_wave_converter_dispatches_blocked_to_mfma_base_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_blocked_to_mfma_base_remap() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %converted = ttg.convert_layout %value : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    fact_program = converter_facts.analyze_facts(source, converted)
    token_program = converter_tokens.build_token_program(source, converted)

    target = converter_op_conversion.convert_ops(
        source,
        converted,
        fact_program,
        token_program,
    )

    (convert_op,) = [op for op in target.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "mfma_vector_register_remap"
    assert attrs["scalar_mode"] == "cross_lane_register_remap"
    assert attrs["fact_policy"] == "invalidate_layout_sensitive"
    assert attrs["result_component_count"] == 1
    assert attrs["scalar_result_component_count"] == 4
    assert attrs["source_component_count"] == 4
    assert attrs["source_registers_per_component"] == 1
    assert attrs["vector_length"] == 4
    assert attrs["scalar_source_indices"] == (0, 1, 2, 3)
    assert attrs["scalar_source_element_indices"] == (0, 0, 0, 0)
    assert attrs["source_lane_map_kind"] == "transpose"
    assert attrs["source_lane_transpose_inner"] == 16
    assert attrs["source_lane_transpose_outer"] == 4
    assert attrs["mode"] != "component_group_first"
    del ctx


def test_tlx_wave_converter_dispatches_tiled_blocked_to_mfma_base_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_tiled_blocked_to_mfma_base_remap() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #blocked>
    %converted = ttg.convert_layout %value : tensor<256x128xf32, #blocked> -> tensor<256x128xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [
        op for op in output.target_program.ops if op.kind == "layout_convert"
    ]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "mfma_vector_register_remap"
    assert attrs["result_component_count"] == 16
    assert attrs["source_component_count"] == 64
    assert attrs["scalar_result_component_count"] == 64
    assert attrs["source_registers_per_component"] == 1
    assert attrs["vector_length"] == 4
    assert attrs["mode"] != "component_group_first"
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_layout_remap_scratch_attrs_are_mode_specific():
    conversion_input = SimpleNamespace(
        value_element_byte_widths={7: 2, 8: None},
        lds_size=33,
    )
    op = SimpleNamespace(index=0)
    simd_result = SimpleNamespace(
        value_id=7,
        type=SimpleNamespace(representation="simd"),
    )
    mask_result = SimpleNamespace(
        value_id=8,
        type=SimpleNamespace(representation="mask_tuple"),
    )

    unrelated = {
        "mode": "same_lane_register_remap",
        "scratch_element_count": 16,
    }
    assert (
        converter_op_conversion._add_layout_remap_scratch_attrs(
            unrelated,
            conversion_input,
            simd_result,
            op,
        )
        == unrelated
    )

    dot_attrs = converter_op_conversion._add_layout_remap_scratch_attrs(
        {
            "mode": "dot_operand_fragment_pack",
            "scratch_element_count": 16,
        },
        conversion_input,
        simd_result,
        op,
    )
    assert dot_attrs["scratch_byte_offset"] == 48
    assert dot_attrs["scratch_allocation_bytes"] == 32

    mask_attrs = converter_op_conversion._add_layout_remap_scratch_attrs(
        {
            "mode": "cta_exchange_register_remap",
            "scratch_element_count": 16,
            "scratch_reuse_lds": True,
        },
        conversion_input,
        mask_result,
        op,
    )
    assert mask_attrs["scratch_byte_offset"] == 48
    assert mask_attrs["scratch_allocation_bytes"] == 64

    empty_lds_input = SimpleNamespace(
        value_element_byte_widths={8: None},
        lds_size=0,
    )
    empty_lds_mask_attrs = converter_op_conversion._add_layout_remap_scratch_attrs(
        {
            "mode": "cta_exchange_register_remap",
            "scratch_element_count": 16,
            "scratch_reuse_lds": True,
        },
        empty_lds_input,
        mask_result,
        op,
    )
    assert empty_lds_mask_attrs["scratch_byte_offset"] == 0
    assert empty_lds_mask_attrs["scratch_allocation_bytes"] == 64


def test_tlx_wave_converter_packs_blocked_accumulator_remap_for_dot(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_blocked_accumulator_remap_for_dot() attributes {noinline = false} {
    %a_alloc = ttg.local_alloc : () -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable>
    %b_alloc = ttg.local_alloc : () -> !ttg.memdesc<32x16xf16, #shared, #smem, mutable>
    %lhs = ttg.local_load %a_alloc : !ttg.memdesc<16x32xf16, #shared, #smem, mutable> -> tensor<16x32xf16, #dot0>
    %rhs = ttg.local_load %b_alloc : !ttg.memdesc<32x16xf16, #shared, #smem, mutable> -> tensor<32x16xf16, #dot1>
    %base = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %acc = ttg.convert_layout %base : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #mma>
    %dot = tt.dot %lhs, %rhs, %acc : tensor<16x32xf16, #dot0> * tensor<32x16xf16, #dot1> -> tensor<16x16xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    convert_attrs = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "layout_convert"
    ]
    (attrs,) = [
        attrs
        for attrs in convert_attrs
        if attrs["mode"] == "mfma_vector_register_remap"
        and attrs.get("scalar_mode") == "cross_lane_register_remap"
    ]
    assert attrs["mode"] == "mfma_vector_register_remap"
    assert attrs["scalar_mode"] == "cross_lane_register_remap"
    native_convert_ops = [
        op
        for op in output.target_program.ops
        if op.kind == "layout_convert"
        and "scratch_element_count" in converter_target_ir.attrs_dict(op)
    ]
    native_convert_attrs = [
        converter_target_ir.attrs_dict(op) for op in native_convert_ops
    ]
    assert [attrs["mode"] for attrs in native_convert_attrs] == [
        "mfma_vector_register_remap",
        "mfma_vector_register_remap",
    ]
    assert [attrs["result_component_count"] for attrs in native_convert_attrs] == [
        1,
        1,
    ]
    assert [attrs["scalar_result_component_count"] for attrs in native_convert_attrs] == [
        4,
        4,
    ]
    (mma_op,) = [op for op in output.target_program.ops if op.kind == "mma"]
    assert native_convert_ops[0].results[0] == mma_op.operands[2]
    assert mma_op.results[0] == native_convert_ops[1].operands[0]
    wave = output.emitted_module.text
    assert 'waveamd.fragment_pack' in wave
    assert 'waveamd.mma "mfma.f32.16x16x32.f16"' in wave
    _run_waveamd_to_machine(wave)
    del ctx


def test_tlx_wave_converter_emits_mfma32_vector_accumulator_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma32_vector_accumulator_remap() attributes {noinline = false} {
    %value = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %converted = ttg.convert_layout %value : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #mma>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [
        op for op in output.target_program.ops if op.kind == "layout_convert"
    ]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "mfma_vector_register_remap"
    assert attrs["result_component_count"] == 1
    assert attrs["scalar_result_component_count"] == 16
    assert attrs["vector_length"] == 16
    wave = output.emitted_module.text
    assert "waveamd.fragment_pack" in wave
    assert "vector<16xf32>" in wave
    _run_wave_verify(wave)
    del ctx


def test_tlx_wave_converter_classifies_mfma_to_blocked_epilogue_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_mfma_to_blocked_epilogue_remap() attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c = arith.truncf %acc : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)
    source = converter_source_import.import_source_program(mod)
    converted = converter_types.convert_source_program(source)
    convert_layout_op = next(op for op in source.ops if op.name == "ttg.convert_layout")
    converted_result = converted.values[convert_layout_op.results[0]]
    assert converted_result.type.component_count == 256

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "cta_exchange_register_remap"
    assert attrs["fact_policy"] == "invalidate_layout_sensitive"
    assert attrs["result_component_count"] == 256
    assert attrs["source_component_count"] == 64
    assert attrs["source_registers_per_component"] == 4
    assert attrs["cta_thread_count"] == 256
    assert attrs["scratch_element_count"] == 2048
    assert attrs["scratch_byte_offset"] == 0
    assert attrs["scratch_allocation_bytes"] == 4096
    assert len(attrs["exchange_groups"]) == 32
    first_group = attrs["exchange_groups"][0]
    assert first_group[0] == (0, 4, 8, 12, 16, 20, 24, 28)
    assert first_group[1] == (0, 4, 8, 12, 16, 20, 24, 28)
    assert first_group[2] == (0, 16, 8, 24, 128, 144, 136, 152)
    assert first_group[3][0] == (32, 64, 256, 512, 1024, 1, 2, 4)
    assert output.emitted_module.lds_size == 4096
    assert output.emitted_module.text.count("wave.store") == 256
    assert output.emitted_module.text.count("wave.barrier") == 64
    assert output.emitted_module.text.count("wave.load") == 256
    assert "wave.wait" not in output.emitted_module.text
    _run_wave_verify(output.emitted_module.text)
    del ctx


def test_tlx_wave_converter_lowers_same_lane_mfma_to_blocked_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_same_lane_mfma_to_blocked_remap() attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c = arith.truncf %acc : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "same_lane_register_remap"
    assert attrs["fact_policy"] == "invalidate_layout_sensitive"
    assert attrs["source_indices"] == (0, 0, 0, 0)
    assert attrs["source_element_indices"] == (0, 1, 2, 3)
    assert output.emitted_module.text.count("wave.extract") == 4
    del ctx


def test_tlx_wave_converter_lowers_cross_lane_mfma_to_blocked_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_cross_lane_mfma_to_blocked_remap() attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c = arith.truncf %acc : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
    %converted = ttg.convert_layout %c : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (convert_op,) = [op for op in output.target_program.ops if op.kind == "layout_convert"]
    attrs = converter_target_ir.attrs_dict(convert_op)
    assert attrs["mode"] == "cross_lane_register_remap"
    assert attrs["fact_policy"] == "invalidate_layout_sensitive"
    assert attrs["source_indices"] == (0, 0, 0, 0)
    assert attrs["source_element_indices"] == (0, 1, 2, 3)
    assert attrs["source_lane_map_kind"] == "transpose"
    assert attrs["source_lane_transpose_inner"] == 4
    assert attrs["source_lane_transpose_outer"] == 16
    assert attrs["source_lane_map"][:8] == (0, 16, 32, 48, 1, 17, 33, 49)
    assert output.emitted_module.text.count("wave.extract") == 4
    assert output.emitted_module.text.count("wave.shuffle") == 4
    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert machine.count("waveamdmachine.ds_bpermute_b32") == 4
    del ctx


def test_tlx_wave_converter_rejects_fragment_cross_lane_remap(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 32], isTransposed = true}>
"""
    local_func = """
  tt.func public @converter_reject_fragment_cross_lane_remap() attributes {noinline = false} {
    %acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %converted = ttg.convert_layout %acc : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT"
    assert "fragment-backed f32 MFMA convert_layout" in str(diagnostic)
    assert "fragment unpack" in str(diagnostic)
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
    assert attrs["inactive_byte_offset"] == 2147483648
    assert attrs["inactive_offset"] == 1073741824
    assert attrs["offset_range"] == (0, 1073741823)
    assert "wave.where" not in output.emitted_module.text
    assert output.emitted_module.text.count("wave.ptr_add") == 2
    assert "wave.select" in output.emitted_module.text

    machine = _run_waveamd_to_machine(output.emitted_module.text)
    assert "waveamdmachine.buffer_store_b16" in machine
    assert "waveamdmachine.exec_if" not in machine
    del ctx


def test_tlx_wave_converter_masks_wide_buffer_store_with_oob_select(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_wide_masked_buffer_store(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<64xf16, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 4 : i32} : tensor<64xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op,) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["mask_mode"] == "select_oob_offset"
    assert attrs["access_element_count"] == 4
    assert attrs["inactive_byte_offset"] == 2147483648
    assert attrs["offset_range"] == (0, 1073741820)
    assert attrs["inactive_offset"] == 1073741824
    assert "wave.select" in output.emitted_module.text
    del ctx


def test_tlx_wave_converter_keeps_buffer_store_components_independent(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_independent_buffer_store_components(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %value = arith.constant dense<0.000000e+00> : tensor<128xf16, #blocked>
    amdg.buffer_store %value, %arg0[%range] {contiguity = 1 : i32} : tensor<128xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    store_lines = [
        line
        for line in wave.splitlines()
        if "wave.store" in line and "#waveamd.buffer" in line
    ]
    assert len(store_lines) == 2
    assert all(" after " not in line for line in store_lines)

    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.buffer_store_b16") == 2
    assert machine.count("waveamdmachine.s_waitcnt_vscnt") <= 1
    del ctx


def test_tlx_wave_converter_masks_byte_buffer_store_with_triton_oob_sentinel(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_byte_masked_buffer_store(%arg0: !tt.ptr<i8> {tt.pointer_range = 32 : i32}, %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %value = arith.constant dense<0> : tensor<64xi8, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    amdg.buffer_store %value, %arg0[%range], %mask {contiguity = 1 : i32} : tensor<64xi8, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (store_op,) = [op for op in output.target_program.ops if op.kind == "buffer_store"]
    attrs = converter_target_ir.attrs_dict(store_op)
    assert attrs["mask_mode"] == "select_oob_offset"
    assert attrs["inactive_byte_offset"] == 2147483648
    assert attrs["inactive_offset"] == 2147483648
    wave = output.emitted_module.text
    assert "wave.select" in wave
    assert "arith.constant -2147483648" not in wave

    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_store_b8" in machine
    assert "waveamdmachine.exec_if" not in machine
    del ctx


def test_tlx_wave_converter_pipeline_lowers_raw_masked_load_store(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_raw_load_store(
      %arg0: !tt.ptr<f32>,
      %arg1: !tt.ptr<f32>,
      %limit: i32) attributes {noinline = false} {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %src_base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %dst_base = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
    %src_ptr = tt.addptr %src_base, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %dst_ptr = tt.addptr %dst_base, %range : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
    %limit_splat = tt.splat %limit : i32 -> tensor<64xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit_splat : tensor<64xi32, #blocked>
    %other = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %loaded = tt.load %src_ptr, %mask, %other : tensor<64x!tt.ptr<f32>, #blocked>
    tt.store %dst_ptr, %loaded, %mask : tensor<64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op,) = [op for op in output.target_program.ops if op.kind == "load"]
    (store_op,) = [op for op in output.target_program.ops if op.kind == "store"]
    load_attrs = converter_target_ir.attrs_dict(load_op)
    store_attrs = converter_target_ir.attrs_dict(store_op)
    assert load_attrs["has_mask"] is True
    assert load_attrs["has_other"] is True
    assert load_attrs["mask_mode"] == "exec_where"
    assert store_attrs["has_mask"] is True
    assert store_attrs["mask_mode"] == "exec_where"
    wave = output.emitted_module.text
    assert "waveamd.make_buffer" not in wave
    assert wave.count("wave.load") == 1
    assert wave.count("wave.store") == 1
    assert wave.count("wave.where") == 2
    assert "wave.select" in wave
    binary_module = _run_wave_compile_kernels(wave)
    assert "gpu.binary @kernels" in binary_module
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
    assert attrs["inactive_byte_offset"] == 2147483648
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


def test_tlx_wave_converter_vectorizes_contiguous_f16_buffer_load(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_vector_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range] {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_op,) = [op for op in output.target_program.ops if op.kind == "buffer_load"]
    attrs = converter_target_ir.attrs_dict(load_op)
    assert attrs["access_element_count"] == 8
    wave = output.emitted_module.text
    assert wave.count("wave.load") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    assert wave.count("wave.extract") == 8
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_vectorizes_packet_uniform_masked_f16_buffer_load(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_masked_vector_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<512xi1, #blocked>
    %loaded = amdg.buffer_load %arg0[%range], %mask {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert wave.count("wave.load") == 1
    assert "!wave.simd<vector<8xf16>, 64>" in wave
    assert wave.count("wave.extract") == 8
    machine = _run_waveamd_to_machine(wave)
    assert "waveamdmachine.buffer_load_tuple_b32" in machine
    assert "waveamdmachine.buffer_load_b16" not in machine
    del ctx


def test_tlx_wave_converter_keeps_buffer_load_packets_inside_contiguity_groups(
    tmp_path,
):
    assert (
        converter_emission._buffer_load_packet_elements(
            {"access_element_count": 3, "element_byte_width": 2}
        )
        == 1
    )
    assert (
        converter_emission._buffer_load_packet_elements(
            {"access_element_count": 5, "element_byte_width": 2}
        )
        == 1
    )
    assert (
        converter_emission._buffer_load_packet_elements(
            {"access_element_count": 10, "element_byte_width": 2}
        )
        == 2
    )

    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
"""
    local_func = """
  tt.func public @converter_nondividing_contiguity_buffer_load(
      %arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %loaded = amdg.buffer_load %arg0[%range] {contiguity = 5 : i32} : tensor<512xf16, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    wave = output.emitted_module.text
    assert "!wave.simd<vector<" not in wave
    assert wave.count("wave.load") == 8
    machine = _run_waveamd_to_machine(wave)
    assert machine.count("waveamdmachine.buffer_load_b16") == 8
    assert "waveamdmachine.buffer_load_tuple_b32" not in machine
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
    assert attrs["inactive_byte_offset"] == 2147483648
    assert attrs["inactive_offset"] == 536870912
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


def test_tlx_wave_converter_scalarized_masked_dma_preserves_rank1_padded_offsets(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
#shared = #ttg.padded_shared<[512:+16] {order = [0], shape = [4096]}>
#smem = #ttg.shared_memory
"""
    local_func = """
  tt.func public @converter_masked_padded_scalarized_dma(%arg0: !tt.ptr<f16> {tt.pointer_range = 32 : i32}) attributes {noinline = false} {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<4096xf16, #shared, #smem, mutable>
    %range = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked>
    %limit = arith.constant dense<4096> : tensor<4096xi32, #blocked>
    %mask = arith.cmpi slt, %range, %limit : tensor<4096xi32, #blocked>
    %token = amdg.buffer_load_to_local %arg0[%range] mask = %mask into %alloc : <f16>[tensor<4096xi32, #blocked>] -> <4096xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %token
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (load_to_local_op,) = [
        op for op in output.target_program.ops if op.kind == "buffer_load_to_local"
    ]
    attrs = converter_target_ir.attrs_dict(load_to_local_op)
    assert attrs["mode"] == "scalarized_load_store"
    assert attrs["component_count"] == 8
    assert attrs["destination_component_offsets"] == tuple(range(8))
    assert attrs["destination_lane_stride_elements"] == 8
    assert attrs["destination_wave_stride_elements"] == 528
    assert output.emitted_module.text.count("wave.store") == 8
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
    assert [attrs["load_mode"] for attrs in local_load_attrs] == [
        "indexed_fragment_load",
        "indexed_fragment_load",
    ]
    assert local_load_attrs[1]["source_shape"] == (32, 16)
    assert local_load_attrs[1]["memdesc_shape"] == (64, 128)
    assert [attrs["wave_tile_axis"] for attrs in local_load_attrs] == ["m", "n"]
    assert [attrs["wave_tile_stride_elements"] for attrs in local_load_attrs] == [
        1024,
        16,
    ]
    assert local_load_attrs[0]["component_tile_offsets"] == (
        (0, 0),
        (0, 32),
        (64, 0),
        (64, 32),
        (128, 0),
        (128, 32),
        (192, 0),
        (192, 32),
    )
    assert local_load_attrs[1]["component_tile_offsets"] == (
        (0, 0),
        (32, 0),
        (0, 32),
        (32, 32),
        (0, 64),
        (32, 64),
        (0, 96),
        (32, 96),
    )
    wave = output.emitted_module.text
    assert "64*floor(1/2*Mod(wi, 64))" in wave
    assert "native_register_layout" not in wave
    assert wave.count("waveamd.fragment_fill") == 16
    assert wave.count('waveamd.mma "mfma.f32.16x16x32.f16"') == 32
    del ctx


def test_tlx_wave_converter_packs_blocked_dot_operand_parent_layout(tmp_path):
    preamble = """
#blocked_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
"""
    local_func = """
  tt.func public @converter_blocked_dot_operand_parent_layout() attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #blocked_a>
    %b = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked_b>
    %a_dot = ttg.convert_layout %a : tensor<256x64xf16, #blocked_a> -> tensor<256x64xf16, #dot0>
    %b_dot = ttg.convert_layout %b : tensor<64x256xf16, #blocked_b> -> tensor<64x256xf16, #dot1>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    layout_converts = [
        converter_target_ir.attrs_dict(op)
        for op in output.target_program.ops
        if op.kind == "layout_convert"
    ]
    assert [attrs["mode"] for attrs in layout_converts] == [
        "dot_operand_fragment_pack",
        "dot_operand_fragment_pack",
    ]
    assert [attrs["payload_mode"] for attrs in layout_converts] == ["vector", "vector"]
    assert [attrs["result_component_count"] for attrs in layout_converts] == [16, 16]
    assert all(len(attrs["fragment_vector_load_bases"]) == 16 for attrs in layout_converts)
    wave = output.emitted_module.text
    assert wave.count("wave.store") == sum(
        attrs["source_component_count"] for attrs in layout_converts
    )
    assert wave.count("wave.load") == 32
    assert wave.count("waveamd.fragment_pack") == 32
    assert "vector<8xf16>" in wave
    del ctx


def test_tlx_wave_converter_rejects_chunked_blocked_dot_operand_pack(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>
"""
    local_func = """
  tt.func public @converter_rejects_chunked_dot_operand_parent_layout() attributes {noinline = false} {
    %a = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #blocked>
    %a_dot = ttg.convert_layout %a : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #dot0>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=4, preamble=preamble)

    with pytest.raises(converter_diagnostics.Diagnostic) as exc_info:
        converter_pipeline.convert_ttgir_to_wave(mod)

    diagnostic = exc_info.value
    assert diagnostic.code == "TLXW_OP_UNSUPPORTED_CONVERT_LAYOUT"
    assert "requires kWidth to match the fragment payload width" in str(diagnostic)
    assert "kWidth=4" in str(diagnostic)
    assert "payload_width=8" in str(diagnostic)
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
    wave = output.emitted_module.text
    assert wave.count("wave.index_expr") == 6
    assert "xor(" in wave
    assert "wave.binary andi" not in wave
    assert "wave.binary shrui" not in wave
    assert wave.count("wave.load") == 2
    assert wave.count("waveamd.transpose_load") == 4
    assert all(
        " after " not in line
        for line in wave.splitlines()
        if "waveamd.transpose_load" in line
    )
    assert wave.count("wave.pack") == 2
    assert wave.count('waveamd.mma "mfma.f32.32x32x16.f16"') == 2
    del ctx


def test_tlx_wave_converter_records_b16_transpose_chunk_deltas(tmp_path):
    preamble = """
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>
#shared = #ttg.padded_shared<[4:+16] {order = [1, 0], shape = [64, 128]}>
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
    assert local_load_attrs[0]["chunk_element_deltas"] == ((0, 2560),) * 8
    wave = output.emitted_module.text
    assert '<"2560 + 5120*floor' in wave
    assert '<"2560 + ' in wave
    assert '<"20 + ' not in wave
    assert '<"100 + 40*Mod' not in wave
    assert wave.count("waveamd.transpose_load") == 16
    assert all(
        " after " not in line
        for line in wave.splitlines()
        if "waveamd.transpose_load" in line
    )
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


def test_tlx_wave_converter_pipeline_lowers_pointer_splat_expand_dims(tmp_path):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_pointer_expand_dims(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #slice>
    %expanded = tt.expand_dims %base {axis = 0 : i32} : tensor<64x!tt.ptr<f32>, #slice> -> tensor<1x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=1, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    assert [op.kind for op in output.target_program.ops] == [
        "splat",
        "expand_dims",
        "return",
    ]
    assert "tt.expand_dims" not in output.emitted_module.text
    assert "wave.splat" in output.emitted_module.text
    assert "!wave.simd<!wave.ptr<#wave.global" in output.emitted_module.text
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


def test_tlx_wave_converter_pipeline_lowers_blocked_column_broadcast_components(
    tmp_path,
):
    preamble = """
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 0, parent = #blocked}>
"""
    local_func = """
  tt.func public @converter_column_broadcast() attributes {noinline = false} {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #slice>
    %expanded = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32, #slice> -> tensor<1x128xi32, #blocked>
    %broadcast = tt.broadcast %expanded : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
    tt.return
  }
"""
    mod, ctx = _parse_ttgir(tmp_path, local_func, num_warps=8, preamble=preamble)

    output = converter_pipeline.convert_ttgir_to_wave(mod)

    (broadcast_op,) = [
        op for op in output.target_program.ops if op.kind == "broadcast"
    ]
    attrs = converter_target_ir.attrs_dict(broadcast_op)
    assert attrs["component_sources"] == tuple(index % 8 for index in range(64))
    assert "tt.broadcast" not in output.emitted_module.text
    del ctx


def test_tlx_wave_emits_broadcast_component_sources():
    source_type = converter_target_ir.TargetType(
        "tensor",
        "simd_tuple",
        "i32",
        64,
        component_count=2,
    )
    result_type = converter_target_ir.TargetType(
        "tensor",
        "simd_tuple",
        "i32",
        64,
        component_count=4,
    )
    program = converter_target_ir.TargetProgram(
        values=(
            converter_target_ir.TargetValue(0, source_type),
            converter_target_ir.TargetValue(1, result_type),
        ),
        ops=(),
        regions=(converter_target_ir.TargetRegion(0),),
        source_value_targets={},
        erased_source_values={},
    )
    state = converter_emission._EmissionState(
        None,
        None,
        None,
        program,
        None,
        {0: ("a", "b")},
        uniform_pointer_bases={0: ("base_a", "base_b")},
    )
    op = converter_target_ir.TargetOp(
        0,
        "broadcast",
        operands=(0,),
        results=(1,),
        attrs=(
            converter_target_ir.TargetAttr("component_sources", (0, 1, 0, 1)),
        ),
    )

    converter_emission._emit_broadcast(state, op)

    assert state.values[1] == ("a", "b", "a", "b")
    assert state.uniform_pointer_bases[1] == ("base_a", "base_b", "base_a", "base_b")


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
    wave_opt = wave_bridge_tools._wave_opt()
    result = subprocess.run(
        [wave_opt, "-", *wave_bridge_tools._wave_hsaco_pipeline_args(wave_opt, "gfx950")],
        input=wave_artifact,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout
