"""Four-wave MXFP8 GEMM with K256 transfers and native K128 computation.

Two payload/scale LDS stages feed two K128 register steps per transfer. The
input partitions and matrix-wave layout keep simultaneous reads in separate
LDS regions. Read the high half early enough to refill its stage while the
low half is still computing; defer the next stage's visibility wait until
two rows of independent high-half work have completed.

Scheduling groups distribute descriptor/address preparation and operand
reads across matrix issue windows. A0 and all B fragments are read first
because the next K128 step consumes them in its first row. Output retains
padded shared storage, b128 stores, and a single FP32 TDM transfer.

Use ``bench.py --operand-pipeline -BK 256``. This path requires E4M3 inputs,
packed block-32 scales, full 256x256 tiles, four waves, two payload/scale
stages, and K >= 512 divisible by 256. A cache prefetch distance of one or
two fetches each batch ahead of its LDS refill; -1 disables it.
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
def _load_scale(buf, slot, row: tl.constexpr, IS_A: tl.constexpr, start_k: tl.constexpr = 0):
    view = tlx.local_reshape(tlx.local_view(buf, slot), [2, 2, 32, 4, 4])
    view = tlx.local_reshape(tlx.local_trans(view, (0, 3, 2, 1, 4)), [256, 8])
    view = tlx.local_reshape(view, [2, 4, 32, 8])
    view = tlx.local_reshape(tlx.local_trans(view, (1, 0, 2, 3)), [256, 8])
    if IS_A:
        layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2), (4, )), stride=((4, 128, 128, 64), (1, )))
    else:
        layout: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, )), stride=((4, 128, 0), (1, )))
    return tlx.local_load(tlx.local_slice(view, [row * 64, start_k // 32], [64, 4]), layout=layout, relaxed=True)


@triton.jit
def _a(buf, slot, row: tl.constexpr, LAYOUT: tl.constexpr, start_k: tl.constexpr = 0):
    view = tlx.local_slice(tlx.local_view(buf, slot), [0, row * 32, start_k], [2, 32, 128])
    load_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2), (16, 4, 2)),
                                           stride=((128, 16, 4096, 2048), (1, 32, 4096)))
    value = tlx.local_load(view, layout=load_layout, relaxed=True)
    return tlx.require_layout(value.reshape([64, 128]), LAYOUT)


@triton.jit
def _b(buf, slot, col: tl.constexpr, LAYOUT: tl.constexpr, start_k: tl.constexpr = 0):
    view = tlx.local_slice(tlx.local_view(buf, slot), [0, col * 32, start_k], [2, 32, 128])
    load_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2), (16, 4, 2)),
                                           stride=((128, 16, 4096, 0), (1, 32, 2048)))
    value = tlx.local_load(view, layout=load_layout, relaxed=True)
    return tlx.require_layout(value.reshape([64, 128]).T, LAYOUT)


@triton.jit
def _schedule_reads(sync_id: tl.constexpr, WITH_SCALES: tl.constexpr):
    if sync_id == 0:
        for _ in tl.static_range(8):
            tlx.amd_sched_group_barrier(0x8, 1, sync_id)
            tlx.amd_sched_group_barrier(0x4, 6, sync_id)
        tlx.amd_sched_group_barrier(0x20, 4, sync_id)
        tlx.amd_sched_group_barrier(0x100, 6, sync_id)
        for _ in tl.static_range(8):
            tlx.amd_sched_group_barrier(0x8, 1, sync_id)
            tlx.amd_sched_group_barrier(0x4, 3, sync_id)
            tlx.amd_sched_group_barrier(0x100, 4, sync_id)
    else:
        if WITH_SCALES:
            tlx.amd_sched_group_barrier(0x100, 4, sync_id)
        for _ in tl.static_range(16):
            tlx.amd_sched_group_barrier(0x8, 1, sync_id)
            tlx.amd_sched_group_barrier(0x100, 2, sync_id)


@triton.jit
def _row(acc, row: tl.constexpr, a, sa, bs, scales, DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr):
    offset: tl.constexpr = row * 4
    values = ()
    for col in tl.static_range(4):
        values += (tl.dot_scaled(a, sa, DTYPE_A, bs[col], scales[col], DTYPE_B, acc[offset + col]), )
    return acc[:offset] + values + acc[offset + 4:]


@triton.jit
def _prefetch_batch(ad, bd, k):
    # The caller clamps k to a complete input tile. Keeping these descriptors
    # invariant avoids decoding advancing pointers and selecting every lane's
    # address in the loop. No per-lane bounds check is needed for full tiles.
    tlx.amd_descriptor_prefetch_tensor(ad, [0, 0, k * 256], speculative=True)
    tlx.amd_descriptor_prefetch_tensor(bd, [0, 0, k * 256], speculative=True)


@triton.jit
def _step_late_wait(acc, a, b, sa, sb, ab, bb, sab, sbb, slot, DA: tl.constexpr, DB: tl.constexpr,
                    DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr):
    acc = _row(acc, 0, a[0], sa[0], b, sb, DTYPE_A, DTYPE_B)
    acc = _row(acc, 1, a[1], sa[1], b, sb, DTYPE_A, DTYPE_B)
    tlx.amd_sched_barrier()
    tlx.async_amd_descriptor_wait(2)
    next_sa, next_sb = (), ()
    for i in tl.static_range(4):
        next_sa += (_load_scale(sab, slot, i, True, 0), )
        next_sb += (_load_scale(sbb, slot, i, False, 0), )
    next_a0 = _a(ab, slot, 0, DA, 0)
    next_b = ()
    for col in tl.static_range(4):
        next_b += (_b(bb, slot, col, DB, 0), )
    acc = _row(acc, 2, a[2], sa[2], b, sb, DTYPE_A, DTYPE_B)
    tlx.amd_sched_group_barrier(0x100, 6, 2)
    for i in tl.static_range(16):
        tlx.amd_sched_group_barrier(0x8, 1, 2)
        tlx.amd_sched_group_barrier(0x100, 3 if i < 8 else 2, 2)
    tlx.amd_sched_barrier()
    next_a1 = _a(ab, slot, 1, DA, 0)
    next_a2 = _a(ab, slot, 2, DA, 0)
    next_a3 = _a(ab, slot, 3, DA, 0)
    acc = _row(acc, 3, a[3], sa[3], b, sb, DTYPE_A, DTYPE_B)
    for i in tl.static_range(16):
        tlx.amd_sched_group_barrier(0x8, 1, 3)
        tlx.amd_sched_group_barrier(0x100, 2 if i < 8 else 1, 3)
    tlx.amd_sched_barrier()
    return acc, (next_a0, next_a1, next_a2, next_a3), next_b, next_sa, next_sb


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
    tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 256 and NUM_BUFFERS == 2)
    tl.static_assert(DTYPE_A == "e4m3" and DTYPE_B == "e4m3" and TRANSPOSE_B and SCALE_PRESHUFFLE and WITH_A_SCALE)
    tl.static_assert(SCALE_BLOCK == 32 and TDM_FUSION == "partial" and not TDM_SPLIT)
    tl.static_assert(SCHEDULE == "sliceMNK")
    tl.static_assert(L2_PREFETCH_DISTANCE == -1 or L2_PREFETCH_DISTANCE == 1 or L2_PREFETCH_DISTANCE == 2)
    MMA: tl.constexpr = tlx.amd_wmma_layout(((2, 2), (1, 0)), ((0, 1), (2, 0)))
    DA: tl.constexpr = tlx.dot_operand_layout(0, MMA, 16)
    DB: tl.constexpr = tlx.dot_operand_layout(1, MMA, 16)
    pid = tl.program_id(0)
    num_m, num_n = M // 256, N // 256
    group = pid // (GROUP_SIZE_M * num_n)
    group_m = tl.minimum(num_m - group * GROUP_SIZE_M, GROUP_SIZE_M)
    pm = group * GROUP_SIZE_M + pid % group_m
    pn = pid % (GROUP_SIZE_M * num_n) // group_m
    ad = tl.make_tensor_descriptor(a_ptr + pm * 256 * stride_am, [M // 128, 128, K], [128 * stride_am, stride_am, 1],
                                   [2, 128, 256])
    bd = tl.make_tensor_descriptor(b_ptr + pn * 256 * stride_bn, [N // 128, 128, K], [128 * stride_bn, stride_bn, 1],
                                   [2, 128, 256])
    sad = tl.make_tensor_descriptor(a_scale + pm * 2 * stride_scale, [M // 128, K * 4], [stride_scale, 1], [2, 1024])
    sbd = tl.make_tensor_descriptor(b_scale + pn * 2 * stride_scale, [N // 128, K * 4], [stride_scale, 1], [2, 1024])
    data_piece: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [1, 128, 256],
                                                                                   [2, 1, 0])
    data_layout: tl.constexpr = tlx.partitioned_shared_layout_encoding(2, 1, 0, data_piece)
    scale_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], [2, 1024])
    ab = tlx.local_alloc((2, 128, 256), tlx.dtype_of(a_ptr), 2, layout=data_layout)
    bb = tlx.local_alloc((2, 128, 256), tlx.dtype_of(b_ptr), 2, layout=data_layout)
    sab = tlx.local_alloc((2, 1024), tl.uint8, 2, layout=scale_layout)
    sbb = tlx.local_alloc((2, 1024), tl.uint8, 2, layout=scale_layout)
    prefetch_ad, prefetch_bd = ad, bd
    for slot in tl.static_range(2):
        _scales(sad, sbd, sab, sbb, slot)
        _data(ad, bd, ab, bb, slot)
        ad = tlx.update_tensor_descriptor(ad, add_offsets=[0, 0, 256], clamp_bounds=False)
        bd = tlx.update_tensor_descriptor(bd, add_offsets=[0, 0, 256], clamp_bounds=False)
        sad = tlx.update_tensor_descriptor(sad, add_offsets=[0, 1024], clamp_bounds=False)
        sbd = tlx.update_tensor_descriptor(sbd, add_offsets=[0, 1024], clamp_bounds=False)
    if L2_PREFETCH_DISTANCE > 0:
        for ahead in tl.static_range(2, 2 + L2_PREFETCH_DISTANCE):
            _prefetch_batch(prefetch_ad, prefetch_bd, tl.minimum(ahead, K // 256 - 1))
    tlx.async_amd_descriptor_wait(2)
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
    k_iters = K // 256
    tl.assume(k_iters >= 2)
    for k in tl.range(k_iters, loop_unroll_factor=1):
        if L2_PREFETCH_DISTANCE > 0:
            _prefetch_batch(prefetch_ad, prefetch_bd, tl.minimum(k + 2 + L2_PREFETCH_DISTANCE, k_iters - 1))
        next_sa, next_sb = (), ()
        for i in tl.static_range(4):
            next_sa += (_load_scale(sab, slot, i, True, 128), )
            next_sb += (_load_scale(sbb, slot, i, False, 128), )
        next_a1 = _a(ab, slot, 1, DA, 128)
        next_a2 = _a(ab, slot, 2, DA, 128)
        next_a3 = _a(ab, slot, 3, DA, 128)
        next_b0 = _b(bb, slot, 0, DB, 128)
        acc = _row(acc, 0, a[0], sa[0], b, sb, DTYPE_A, DTYPE_B)
        _schedule_reads(0, True)
        tlx.amd_sched_barrier()
        next_a0 = _a(ab, slot, 0, DA, 128)
        next_b1 = _b(bb, slot, 1, DB, 128)
        next_b2 = _b(bb, slot, 2, DB, 128)
        next_b3 = _b(bb, slot, 3, DB, 128)
        acc = _row(acc, 1, a[1], sa[1], b, sb, DTYPE_A, DTYPE_B)
        _schedule_reads(1, False)
        tlx.amd_sched_barrier()
        acc = _row(acc, 2, a[2], sa[2], b, sb, DTYPE_A, DTYPE_B)
        tlx.amd_sched_barrier()
        _scales(sad, sbd, sab, sbb, slot, k + 2 < k_iters)
        _data(ad, bd, ab, bb, slot, k + 2 < k_iters)
        ad = tlx.update_tensor_descriptor(ad, add_offsets=[0, 0, 256], clamp_bounds=False)
        bd = tlx.update_tensor_descriptor(bd, add_offsets=[0, 0, 256], clamp_bounds=False)
        sad = tlx.update_tensor_descriptor(sad, add_offsets=[0, 1024], clamp_bounds=False)
        sbd = tlx.update_tensor_descriptor(sbd, add_offsets=[0, 1024], clamp_bounds=False)
        acc = _row(acc, 3, a[3], sa[3], b, sb, DTYPE_A, DTYPE_B)
        tlx.amd_sched_barrier()
        a = (next_a0, next_a1, next_a2, next_a3)
        b = (next_b0, next_b1, next_b2, next_b3)
        sa, sb = next_sa, next_sb
        next_slot = slot ^ 1
        acc, a, b, sa, sb = _step_late_wait(acc, a, b, sa, sb, ab, bb, sab, sbb, next_slot, DA, DB, DTYPE_A, DTYPE_B)
        slot = next_slot
    tlx.async_amd_descriptor_wait(0)
    out_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [256, 256])
    out = tlx.local_view(tlx.local_alloc((256, 256), tl.float32, 1, layout=out_layout), 0)
    out_view = tlx.local_reshape(out, [2, 4, 32, 2, 4, 32])
    out_view = tlx.local_reshape(tlx.local_trans(out_view, (1, 0, 2, 4, 3, 5)), [256, 256])
    for row in tl.static_range(4):
        for col in tl.static_range(4):
            tlx.local_store(tlx.local_slice(out_view, [row * 64, col * 64], [64, 64]), acc[row * 4 + col])
    cd = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [256, 256])
    tlx.async_amd_descriptor_store(cd, out, [pm * 256, pn * 256])
    tlx.async_amd_descriptor_wait(0)
