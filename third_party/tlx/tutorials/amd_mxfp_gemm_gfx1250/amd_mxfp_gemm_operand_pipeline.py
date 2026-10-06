"""Four-wave MXFP8 GEMM with operand prefetch across K steps.

Four payload/scale LDS slots feed one rolling A register set and two B sets.
One spare 64-row A fragment lets the final fragment of the next K step load
before its current value is consumed. A two-step loop rotates the physical
register banks without copying the operands at each backedge.

Each K128 step computes sixteen 64x64 subtiles: 64 native scaled WMMAs per
wave, with b128 payload reads and FP32 output stores. Payload and scale slots
share a lifetime so the compiler can use one workgroup reuse barrier.

Run through ``bench.py --operand-pipeline``. This experimental path requires
E4M3 inputs, packed block-32 scales, full 256x256 tiles, four waves, and
K >= 512 divisible by 256.
"""
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def _data(ad, bd, ab, bb, slot, pred=True):
    tlx.async_amd_descriptor_load_fused(
        ((tlx.update_tensor_descriptor(ad, pred=pred, clamp_bounds=False), tlx.local_view(ab, slot), 5),
         (tlx.update_tensor_descriptor(bd, pred=pred, clamp_bounds=False), tlx.local_view(bb, slot), 10)))


@triton.jit
def _scales(ad, bd, ab, bb, slot, pred=True):
    tlx.async_amd_descriptor_load_fused(
        ((tlx.update_tensor_descriptor(ad, pred=pred, clamp_bounds=False), tlx.local_view(ab, slot), 5),
         (tlx.update_tensor_descriptor(bd, pred=pred, clamp_bounds=False), tlx.local_view(bb, slot), 10)))


@triton.jit
def _load_scale(buf, slot, row: tl.constexpr, IS_A: tl.constexpr):
    view = tlx.local_reshape(tlx.local_view(buf, slot), [2, 1, 32, 4, 4])
    view = tlx.local_reshape(tlx.local_trans(view, (0, 3, 2, 1, 4)), [256, 4])
    if IS_A:
        layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, )), stride=((4, 0, 128), (1, )))
    else:
        layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, )), stride=((4, 128, 0), (1, )))
    return tlx.local_load(tlx.local_slice(view, [row * 64, 0], [64, 4]), layout=layout, relaxed=True)


@triton.jit
def _a(buf, slot, row: tl.constexpr, LAYOUT: tl.constexpr):
    return tlx.local_load(tlx.local_slice(tlx.local_view(buf, slot), [row * 64, 0], [64, 128]), layout=LAYOUT,
                          relaxed=True)


@triton.jit
def _b(buf, slot, col: tl.constexpr, LAYOUT: tl.constexpr):
    view = tlx.local_slice(tlx.local_view(buf, slot), [col * 64, 0], [64, 128])
    return tlx.local_load(tlx.local_trans(view), layout=LAYOUT, relaxed=True)


@triton.jit
def _row(acc, row: tl.constexpr, a, sa, bs, scales, DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr):
    offset: tl.constexpr = row * 4
    values = ()
    for col in tl.static_range(4):
        values += (tl.dot_scaled(a, sa, DTYPE_A, bs[col], scales[col], DTYPE_B, acc[offset + col]), )
    return acc[:offset] + values + acc[offset + 4:]


@triton.jit
def mxgemm_tdm_operand_pipeline_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale,
    b_scale,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_scale,
    DTYPE_A: tl.constexpr,
    DTYPE_B: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr = True,
    WITH_A_SCALE: tl.constexpr = True,
    SCHEDULE: tl.constexpr = "sliceMNK",
    TDM_FUSION: tl.constexpr = "partial",
    L2_PREFETCH_DISTANCE: tl.constexpr = -1,
    TDM_SPLIT: tl.constexpr = False,
):
    tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128 and NUM_BUFFERS == 4)
    tl.static_assert(DTYPE_A == "e4m3" and DTYPE_B == "e4m3" and TRANSPOSE_B and SCALE_PRESHUFFLE and WITH_A_SCALE)
    tl.static_assert(SCALE_BLOCK == 32 and TDM_FUSION == "partial" and not TDM_SPLIT)
    tl.static_assert(SCHEDULE == "sliceMNK" and L2_PREFETCH_DISTANCE == -1)
    MMA: tl.constexpr = tlx.amd_wmma_layout(((0, 2), (2, 0)), ((0, 1), (1, 0)))
    DA: tl.constexpr = tlx.dot_operand_layout(0, MMA, 16)
    DB: tl.constexpr = tlx.dot_operand_layout(1, MMA, 16)
    pid = tl.program_id(0)
    num_m, num_n = M // 256, N // 256
    group = pid // (GROUP_SIZE_M * num_n)
    group_m = tl.minimum(num_m - group * GROUP_SIZE_M, GROUP_SIZE_M)
    pm = group * GROUP_SIZE_M + pid % group_m
    pn = pid % (GROUP_SIZE_M * num_n) // group_m
    ad = tl.make_tensor_descriptor(a_ptr + pm * 256 * stride_am, [M, K], [stride_am, 1], [256, 128])
    bd = tl.make_tensor_descriptor(b_ptr + pn * 256 * stride_bn, [N, K], [stride_bn, 1], [256, 128])
    sad = tl.make_tensor_descriptor(a_scale + pm * 2 * stride_scale, [M // 128, K * 4], [stride_scale, 1], [2, 512])
    sbd = tl.make_tensor_descriptor(b_scale + pn * 2 * stride_scale, [N // 128, K * 4], [stride_scale, 1], [2, 512])
    data_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [256, 128])
    scale_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], [2, 512])
    ab = tlx.local_alloc((256, 128), tlx.dtype_of(a_ptr), 4, layout=data_layout)
    bb = tlx.local_alloc((256, 128), tlx.dtype_of(b_ptr), 4, layout=data_layout)
    sab = tlx.local_alloc((2, 512), tl.uint8, 4, layout=scale_layout)
    sbb = tlx.local_alloc((2, 512), tl.uint8, 4, layout=scale_layout)
    for slot in tl.static_range(4):
        _scales(sad, sbd, sab, sbb, slot)
        _data(ad, bd, ab, bb, slot)
        ad = tlx.update_tensor_descriptor(ad, add_offsets=[0, 128], clamp_bounds=False)
        bd = tlx.update_tensor_descriptor(bd, add_offsets=[0, 128], clamp_bounds=False)
        sad = tlx.update_tensor_descriptor(sad, add_offsets=[0, 512], clamp_bounds=False)
        sbd = tlx.update_tensor_descriptor(sbd, add_offsets=[0, 512], clamp_bounds=False)
    tlx.async_amd_descriptor_wait(6)
    a = ()
    b = ()
    sa, sb = (), ()
    for i in tl.static_range(4):
        a += (_a(ab, 0, i, DA), )
        b += (_b(bb, 0, i, DB), )
        sa += (_load_scale(sab, 0, i, True), )
        sb += (_load_scale(sbb, 0, i, False), )
    acc = ()
    for _ in tl.static_range(16):
        acc += (tlx.require_layout(tl.zeros((64, 64), tl.float32), MMA), )
    slot = 0
    k_iters = K // 128
    tl.assume(k_iters >= 4)
    tl.assume(k_iters % 2 == 0)
    for k in tl.range(0, k_iters, loop_unroll_factor=2):
        next_slot = (slot + 1) & 3
        # Every operand of the current K step was read in the preceding
        # iteration. Return its LDS slot before prefetching the next step.
        _scales(sad, sbd, sab, sbb, slot, k + 4 < k_iters)
        _data(ad, bd, ab, bb, slot, k + 4 < k_iters)
        tlx.async_amd_descriptor_wait(6)
        next_sa, next_sb = (), ()
        for i in tl.static_range(4):
            next_sa += (_load_scale(sab, next_slot, i, True), )
            next_sb += (_load_scale(sbb, next_slot, i, False), )
        ad = tlx.update_tensor_descriptor(ad, add_offsets=[0, 128], clamp_bounds=False)
        bd = tlx.update_tensor_descriptor(bd, add_offsets=[0, 128], clamp_bounds=False)
        sad = tlx.update_tensor_descriptor(sad, add_offsets=[0, 512], clamp_bounds=False)
        sbd = tlx.update_tensor_descriptor(sbd, add_offsets=[0, 512], clamp_bounds=False)
        next_a, next_b = (), ()
        for row in tl.static_range(4):
            # A0..A2 reuse their bank one row after its last use. A3 has
            # one spare fragment so its next value can be read in row zero.
            if row == 0:
                next_a3 = _a(ab, next_slot, 3, DA)
            else:
                next_a += (_a(ab, next_slot, row - 1, DA), )
            next_b += (_b(bb, next_slot, row, DB), )
            acc = _row(acc, row, a[row], sa[row], b, sb, DTYPE_A, DTYPE_B)
            tlx.amd_sched_barrier()
        a, b, sa, sb = next_a + (next_a3, ), next_b, next_sa, next_sb
        slot = next_slot
    tlx.async_amd_descriptor_wait(0)
    out_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [256, 256])
    out = tlx.local_view(tlx.local_alloc((256, 256), tl.float32, 1, layout=out_layout), 0)
    for row in tl.static_range(4):
        for col in tl.static_range(4):
            tlx.local_store(tlx.local_slice(out, [row * 64, col * 64], [64, 64]), acc[row * 4 + col])
    cd = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [256, 256])
    tlx.async_amd_descriptor_store(cd, out, [pm * 256, pn * 256])
    tlx.async_amd_descriptor_wait(0)
