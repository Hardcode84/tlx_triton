import triton
import triton.language as tl

import torch
import math
import pytest

_BLOCK_SIZE = 16


def test_llvm_ir_to_bitcode():
    from triton._C.libtriton import llvm

    bitcode = llvm.to_bitcode("define void @kernel() { ret void }")
    assert isinstance(bitcode, bytes)
    assert bitcode.startswith(b"BC\xc0\xde")
    assert b"\x00" in bitcode


def test_llvm_ir_to_bitcode_reports_invalid_ir():
    from triton._C.libtriton import llvm

    with pytest.raises(RuntimeError, match="failed to parse LLVM IR.*expected top-level entity"):
        llvm.to_bitcode("invalid LLVM IR")


@triton.jit
def add_helper(x, y):
    return x + y


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    n_elements,
    out_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    x2d = x[None, :]
    x1d = tl.reshape(x2d, [BLOCK_SIZE])
    output = add_helper(x1d, y)
    tl.store(out_ptr + offsets, output, mask=mask)


def test_module_walk(device):
    """
    Test the MLIR bindings exposed for the out-of-tree walk.
    """

    def walk_fn(op):
        name = op.get_name()
        for i in range(op.get_num_results()):
            op.get_result(i).id()
        for i in range(op.get_num_operands()):
            op.get_operand(i).id()
        for i in range(op.get_num_regions()):
            op.get_region(i).id()
        block = op.get_block()
        if block is not None:
            block.id()
            for i in range(block.get_num_arguments()):
                block.get_argument(i)
        if name == "tt.func":
            op.get_str_attr("sym_name")
        if name == "tt.call":
            op.get_flat_symbol_ref_attr("callee")
        if name == "tt.make_range":
            assert 0 == op.get_int_attr("start")
            assert _BLOCK_SIZE == op.get_int_attr("end")
        if name == "arith.constant":
            val = op.get_constant_value()
            assert isinstance(val, int)
        if name == "tt.expand_dims":
            shape = op.get_result(0).get_shape()
            assert shape == [1, _BLOCK_SIZE]
        if name == "tt.reshape":
            in_shape = op.get_operand(0).get_shape()
            out_shape = op.get_result(0).get_shape()
            assert in_shape == [1, _BLOCK_SIZE]
            assert out_shape == [_BLOCK_SIZE]

    kernel = add_kernel
    args = [
        torch.empty((32, 32), device=device),  # in_ptr0
        torch.empty((32, 32), device=device),  # in_ptr1
        1024,  # n_elements
        torch.empty((32, 32), device=device),  # out_ptr
        _BLOCK_SIZE,  # BLOCK_SIZE
    ]
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    src = triton.compiler.compiler.ASTSource(
        fn=kernel,
        signature={kernel.arg_names[i]: triton.runtime.jit.mangle_type(arg)
                   for i, arg in enumerate(args)},
        constexprs={kernel.arg_names[i]: arg
                    for i, arg in enumerate(args)
                    if not isinstance(arg, torch.Tensor)},
    )

    context = triton._C.libtriton.ir.context()
    options = backend.parse_options(dict())
    codegen_fns = dict()
    module_map = backend.get_module_map()
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
    ttir_module.walk(walk_fn)


def test_python_func_in_visit_call(device):

    @triton.jit
    def test_py_call_const_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        log2e: tl.constexpr = math.log2(math.e)
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = x * log2e
        tl.store(out_ptr + offsets, output, mask=mask)

    x = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    test_py_call_const_kernel[(4, )](x, out, 4, 4)


@pytest.mark.parametrize("scalarize", [False, True])
def test_scalarize_packed_fops_llvm_pipeline(tmp_path, scalarize):
    from triton._C.libtriton import ir, llvm

    # NVIDIA requests this pass explicitly: its module does not yet have a
    # target triple when optimize_module runs. Other backends retain the default.
    source = tmp_path / "fma.mlir"
    source.write_text("""
module {
  llvm.func @fma_lane(%a: vector<2xf32>, %b: vector<2xf32>, %c: vector<2xf32>) -> f32 {
    %v = llvm.call_intrinsic "llvm.fma"(%a, %b, %c) : (vector<2xf32>, vector<2xf32>, vector<2xf32>) -> vector<2xf32>
    %index = llvm.mlir.constant(1 : i32) : i32
    %r = llvm.extractelement %v[%index : i32] : vector<2xf32>
    llvm.return %r : f32
  }
}
""")
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    llvm_context = llvm.context()
    llvm_module = llvm.to_module(module, llvm_context)
    options = {"scalarize_packed_fops": True} if scalarize else {}
    llvm.optimize_module(llvm_module, llvm.OPTIMIZE_O3, **options)
    expected = "call float @llvm.fma.f32" if scalarize else "call <2 x float> @llvm.fma.v2f32"
    assert expected in str(llvm_module)


@pytest.mark.parametrize("size", [1, 2])
@pytest.mark.parametrize("buffer", [False, True])
def test_reencode_blocked_load(tmp_path, size, buffer):
    from triton._C.libtriton import ir
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget

    source = tmp_path / "load.mlir"
    source.write_text("""
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @load(%ptr: !tt.ptr<f16>, %offset: tensor<512xi32, #blocked>, %mask: tensor<512xi1, #blocked>, %other: tensor<512xf16, #blocked>) -> tensor<512xf16, #blocked> {
    %value = amdg.buffer_load %ptr[%offset], %mask, %other {contiguity = 8 : i32} : tensor<512xf16, #blocked>
    tt.return %value : tensor<512xf16, #blocked>
  }
}
""")
    if not buffer:
        source.write_text(source.read_text().replace(
            '%value = amdg.buffer_load %ptr[%offset], %mask, %other {contiguity = 8 : i32} : tensor<512xf16, #blocked>',
            '%base = tt.splat %ptr : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>, #blocked>\n'
            '%pointers = tt.addptr %base, %offset : tensor<512x!tt.ptr<f16>, #blocked>, tensor<512xi32, #blocked>\n'
            '%value = tt.load %pointers, %mask, %other : tensor<512x!tt.ptr<f16>, #blocked>'))
    context = ir.context()
    ir.load_dialects(context)
    HIPBackend(GPUTarget("hip", "gfx950", 64)).load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    loads = []
    module.walk(lambda op: loads.append(op) if op.get_name() == ("amdg.buffer_load" if buffer else "tt.load") else None)
    (load, ) = loads
    with pytest.raises(ValueError, match="matching rank"):
        load.reencode_blocked_load([1, 1])
    with pytest.raises(ValueError, match="power of two"):
        load.reencode_blocked_load([3])
    rewritten = load.reencode_blocked_load([size])
    if buffer:
        assert rewritten.get_int_attr("contiguity") == size
    assert rewritten.get_result(0).get_type().get_encoding().get_blocked_size_per_thread() == [size]
    assert str(module).count("ttg.convert_layout") == 4
    assert module.verify()
