"""Persistent E4M3 GEMM with native WMMA fragments and pipelined operand reads.

One rolling A register set, two spare A fragments, and two B sets feed
sixty-four 32x32 accumulator subtiles. The first two rows execute by column
so the last B fragment has a later first use. Their matrix work covers the
final A reads before LDS reuse. The remaining rows interleave current matrix
work with next-step A/B reads.

A paired K loop rotates operand registers without backedge copies. Its peeled
tail prefetches the next output tile while the current tile finishes. Its
first operands stay in registers through the FP32 store phase, avoiding a
second load at tile entry. Two separate output slots use b128 LDS stores.

Run through ``bench.py --streamed-operands``. This experimental path requires
E4M3 inputs, preshuffled block-32 scales, full 256x256 tiles, three K128 buffers,
four waves, and K >= 384 divisible by 128.
"""

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

if __package__:
    from .amd_mxfp_gemm_tdm_pipelined import (
        _mxgemm_persistent_load,
        _mxgemm_remap_program_id,
        _mxgemm_tile_offsets,
        _operand_shared_layout,
        _scale_shared_layout,
    )
else:
    from amd_mxfp_gemm_tdm_pipelined import (
        _mxgemm_persistent_load,
        _mxgemm_remap_program_id,
        _mxgemm_tile_offsets,
        _operand_shared_layout,
        _scale_shared_layout,
    )


@triton.jit
def _pack(value):
    bits = value.to(tl.uint8, bitcast=True)
    even, odd = tl.split(tl.reshape(bits, (32, 64, 2)))
    b0, b2 = tl.split(tl.reshape(even, (32, 32, 2)))
    b1, b3 = tl.split(tl.reshape(odd, (32, 32, 2)))
    return b0.to(tl.uint32) | (b1.to(tl.uint32) << 8) | (b2.to(tl.uint32) << 16) | (b3.to(tl.uint32) << 24)


@triton.jit
def _unpack(value, IS_A: tl.constexpr):
    even = tl.join(value.to(tl.uint8), (value >> 16).to(tl.uint8))
    odd = tl.join((value >> 8).to(tl.uint8), (value >> 24).to(tl.uint8))
    result = tl.reshape(tl.join(even, odd), (32, 128)).to(tl.float8e4nv, bitcast=True)
    mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (1, 0)))
    if IS_A:
        return tlx.require_layout(result, tlx.dot_operand_layout(0, mma, 16))
    else:
        return tlx.require_layout(result.T, tlx.dot_operand_layout(1, mma, 16))


@triton.constexpr_function
def _scale_layout(IS_A):
    if IS_A:
        return tlx.layout(shape=((16, 2, 2, 2), (4, )), stride=((4, 0, 0, 64), (1, )))
    return tlx.layout(shape=((16, 2, 2, 2), (4, )), stride=((4, 0, 64, 0), (1, )))


@triton.jit
def _scale_pack(scale):
    even, odd = tl.split(tl.reshape(scale, (scale.shape[0], 2, 2)))
    b0, b2 = tl.split(even)
    b1, b3 = tl.split(odd)
    return b0.to(tl.uint32) | (b1.to(tl.uint32) << 8) | (b2.to(tl.uint32) << 16) | (b3.to(tl.uint32) << 24)


@triton.jit
def _scale_unpack(scale, IS_A: tl.constexpr):
    even = tl.join(scale.to(tl.uint8), (scale >> 16).to(tl.uint8))
    odd = tl.join((scale >> 8).to(tl.uint8), (scale >> 24).to(tl.uint8))
    return tlx.require_layout(tl.reshape(tl.join(even, odd), (32, 4)), _scale_layout(IS_A))


@triton.jit
def _scales(scale_buf, slot, IS_A: tl.constexpr):
    tl.assume(slot >= 0)
    tl.assume(slot < 3)
    scale_view = tlx.local_reshape(tlx.local_view(scale_buf, slot), [2, 1, 32, 4, 4])
    scale_view = tlx.local_reshape(tlx.local_trans(scale_view, (0, 3, 2, 1, 4)), [256, 4])
    if IS_A:
        layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2), (4, 8)), stride=((4, 0, 0, 64), (1, 128)))
    else:
        layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2), (4, 8)), stride=((4, 0, 64, 0), (1, 128)))
    packed = _scale_pack(tlx.local_load(scale_view, layout=layout, relaxed=True))
    lo, hi = tl.split(tl.reshape(packed, (2, 128)).T)
    ll, lh = tl.split(tl.reshape(lo, (2, 64)).T)
    hl, hh = tl.split(tl.reshape(hi, (2, 64)).T)
    p0, p1 = tl.split(tl.reshape(ll, (2, 32)).T)
    p2, p3 = tl.split(tl.reshape(lh, (2, 32)).T)
    p4, p5 = tl.split(tl.reshape(hl, (2, 32)).T)
    p6, p7 = tl.split(tl.reshape(hh, (2, 32)).T)
    return p0, p1, p2, p3, p4, p5, p6, p7


@triton.jit
def _operand(buf, slot, piece: tl.constexpr, IS_A: tl.constexpr):
    tl.assume(slot >= 0)
    tl.assume(slot < 3)
    view = tlx.local_slice(tlx.local_view(buf, slot), [piece * 32, 0], [32, 128])
    mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (1, 0)))
    if IS_A:
        value = tlx.local_load(view, layout=tlx.dot_operand_layout(0, mma, 16), relaxed=True)
    else:
        value = tlx.local_load(tlx.local_trans(view), layout=tlx.dot_operand_layout(1, mma, 16), relaxed=True).T
    return _pack(value)


@triton.jit
def _row(acc, row: tl.constexpr, a, sa, bs, scales):
    a = _unpack(a, True)
    sa = _scale_unpack(sa, True)
    values = ()
    mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (1, 0)))
    for col in tl.static_range(8):
        b = _unpack(bs[col], False)
        sb = _scale_unpack(scales[col], False)
        c = tlx.require_layout(acc[row * 8 + col], mma)
        values += (tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", c, tiles_per_warp=[1, 1]), )
    return acc[:row * 8] + values + acc[row * 8 + 8:]


@triton.jit
def _head(acc, a, sa, bs, scales):
    a0, a1 = _unpack(a[0], True), _unpack(a[1], True)
    sa0, sa1 = _scale_unpack(sa[0], True), _scale_unpack(sa[1], True)
    row0, row1 = (), ()
    mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (1, 0)))
    for col in tl.static_range(8):
        b = _unpack(bs[col], False)
        sb = _scale_unpack(scales[col], False)
        c0 = tlx.require_layout(acc[col], mma)
        c1 = tlx.require_layout(acc[8 + col], mma)
        row0 += (tlx.dot_scaled(a0, sa0, "e4m3", b, sb, "e4m3", c0, tiles_per_warp=[1, 1]), )
        row1 += (tlx.dot_scaled(a1, sa1, "e4m3", b, sb, "e4m3", c1, tiles_per_warp=[1, 1]), )
    return row0 + row1 + acc[16:]


@triton.jit
def _schedule_pair(sync_id: tl.constexpr):
    # Bulk scale reads lead the first payload pair. The final two rows also
    # prefetch the spare A fragments. Pair WMMAs to limit register-bank switches.
    if sync_id == 2:
        tlx.amd_sched_group_barrier(0x100, 4, sync_id)
    for _ in tl.static_range(4):
        tlx.amd_sched_group_barrier(0x100, 4 if sync_id >= 6 else 2, sync_id)
        tlx.amd_sched_group_barrier(0x8, 2, sync_id)


@triton.jit
def _step(acc, a, b, sa, sb, ad, bd, sad, sbd, ab, bb, sab, sbb, slot, load_m, load_n, load_k,
          CLUSTER_SIZE: tl.constexpr, CLUSTER_MULTICAST: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
          XCD_REMAP_MODE: tl.constexpr, CLUSTER_BARRIER_INTERVAL: tl.constexpr):
    # Interleave the first two rows by column. B7 is first consumed after
    # fourteen WMMAs. Two spare A fragments move the final reads into the
    # preceding rows, ahead of this independent matrix work and LDS reuse.
    acc = _head(acc, a, sa, b, sb)
    tlx.amd_sched_barrier()
    # Launch the new descriptors before waiting for the next readable stage.
    # A wait placed first would delay the refill whenever that older stage
    # is still in flight, shortening the memory-overlap window.
    _mxgemm_persistent_load(ad, bd, sad, sbd, ab, bb, sab, sbb, load_m, load_n, load_k, slot, 128, 1, 1, 3, True,
                            "partial", CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                            CLUSTER_BARRIER_INTERVAL)
    tlx.async_amd_descriptor_wait(4)
    next_slot = tl.where(slot == 2, 0, slot + 1)
    nsa, nsb = _scales(sab, next_slot, True), _scales(sbb, next_slot, False)
    na, nb = (), ()
    for row in tl.static_range(2, 8):
        av = _operand(ab, next_slot, row - 2, True)
        bv = _operand(bb, next_slot, row - 2 if row < 7 else 6, False)
        if row >= 6:
            extra_b = _operand(bb, next_slot, 5 if row == 6 else 7, False)
        if row == 6:
            last_a6 = _operand(ab, next_slot, 6, True)
        if row == 7:
            last_a7 = _operand(ab, next_slot, 7, True)
        acc = _row(acc, row, a[row], sa[row], b, sb)
        _schedule_pair(row)
        tlx.amd_sched_barrier()
        na, nb = na + (av, ), nb + (bv, )
        if row >= 6:
            nb += (extra_b, )
    tlx.amd_sched_barrier()
    return acc, na + (last_a6, last_a7), nb, nsa, nsb, next_slot


@triton.jit
def mxgemm_tdm_streamed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale,
    b_scale,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_as: tl.constexpr,
    stride_bs: tl.constexpr,
    DTYPE_A: tl.constexpr,
    DTYPE_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    TDM_FUSION: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    CROSS_TILE_PREFETCH: tl.constexpr = True,
    OUTPUT_STAGING: tl.constexpr = True,
    SCHED_MODE_2: tl.constexpr = False,
    XCD_REMAP_MODE: tl.constexpr = 0,
    NUM_XCDS: tl.constexpr = 8,
    XCD_CHUNK: tl.constexpr = 2,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    CLUSTER_BARRIER_INTERVAL: tl.constexpr = 4,
    REGISTER_PIPELINE: tl.constexpr = False,
    OUTPUT_TAIL_REUSE: tl.constexpr = False,
    FIRST_USE_PREFETCH: tl.constexpr = False,
):
    tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 128 and NUM_BUFFERS == 3)
    tl.static_assert(DTYPE_A == "e4m3" and DTYPE_B == "e4m3" and WITH_A_SCALE and TDM_FUSION == "partial")
    tl.static_assert(OUTPUT_STAGING and CROSS_TILE_PREFETCH and not REGISTER_PIPELINE and tlx.num_warps() == 4)
    tl.static_assert(not OUTPUT_TAIL_REUSE and not FIRST_USE_PREFETCH)
    tl.static_assert(M > 0 and N > 0 and M % 256 == 0 and N % 256 == 0 and K >= 384 and K % 128 == 0)
    if SCHED_MODE_2:
        tlx.amd_set_wave_sched_mode(1, offset=2, width=1)
    mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (1, 0)))
    ad = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [256, 128])
    bd = tl.make_tensor_descriptor(b_ptr, [N, K], [stride_bn, 1], [256, 128])
    sad = tl.make_tensor_descriptor(a_scale, [M // 128, K * 4], [stride_as, 1], [2, 512])
    sbd = tl.make_tensor_descriptor(b_scale, [N // 128, K * 4], [stride_bs, 1], [2, 512])
    ab = tlx.local_alloc((256, 128), tlx.dtype_of(a_ptr), 3, layout=_operand_shared_layout([256, 128]))
    bb = tlx.local_alloc((256, 128), tlx.dtype_of(b_ptr), 3, layout=_operand_shared_layout([256, 128]))
    sab = tlx.local_alloc((2, 512), tl.uint8, 3, layout=_scale_shared_layout([2, 512]))
    sbb = tlx.local_alloc((2, 512), tl.uint8, 3, layout=_scale_shared_layout([2, 512]))
    cb = tlx.local_alloc((64, 128), tl.float32, 2)
    num_m, num_n = M // 256, N // 256
    total_tiles = num_m * num_n
    tile = _mxgemm_remap_program_id(tl.program_id(0), NUM_PROGRAMS, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK)
    phase = 0
    off_m, off_n = _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_SIZE_M, 256, 256)
    for p in tl.static_range(3):
        _mxgemm_persistent_load(ad, bd, sad, sbd, ab, bb, sab, sbb, off_m, off_n, p, p, 128, 1, 1, 3, True, "partial",
                                CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL)
    tlx.async_amd_descriptor_wait(4)
    sa, sb = _scales(sab, phase, True), _scales(sbb, phase, False)
    a, b = (), ()
    for row in tl.static_range(8):
        av = _operand(ab, phase, row, True)
        bv = _operand(bb, phase, row, False)
        a, b = a + (av, ), b + (bv, )
    k_iters = K // 128
    while tile < total_tiles:
        # Establish a completed LDS epoch at the outer register-carry join.
        # Otherwise wait-count analysis can drain all reads again at the
        # inner loop header, before its delayed operand consumers.
        tlx.workgroup_barrier()
        off_m, off_n = _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_SIZE_M, 256, 256)
        acc = ()
        for _ in tl.static_range(64):
            acc += (tlx.require_layout(tl.zeros((32, 32), tl.float32), mma), )
        slot = phase
        # Keep descriptor coordinates invariant in the paired steady loop.
        # Only the last three or four steps can refill the next output tile.
        steady_end: tl.constexpr = (K // 128 - 3) // 2 * 2
        for k in tl.range(steady_end, loop_unroll_factor=2):
            acc, a, b, sa, sb, slot = _step(acc, a, b, sa, sb, ad, bd, sad, sbd, ab, bb, sab, sbb, slot, off_m, off_n,
                                            k + 3, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                            CLUSTER_BARRIER_INTERVAL)
        tlx.amd_sched_barrier()
        next_tile = tile + NUM_PROGRAMS
        # Final-tile refills are unused. A valid clamped address keeps the
        # descriptor counts and scheduling regions uniform through the tail.
        next_m, next_n = _mxgemm_tile_offsets(tl.minimum(next_tile, total_tiles - 1), num_m, num_n, GROUP_SIZE_M, 256,
                                              256)
        tlx.amd_sched_barrier()
        for k in tl.range(steady_end, k_iters, loop_unroll_factor=2):
            refill_k = k + 3
            next_input = refill_k >= k_iters
            load_k = tl.where(next_input, refill_k - k_iters, refill_k)
            load_m = tl.where(next_input, next_m, off_m)
            load_n = tl.where(next_input, next_n, off_n)
            acc, a, b, sa, sb, slot = _step(acc, a, b, sa, sb, ad, bd, sad, sbd, ab, bb, sab, sbb, slot, load_m, load_n,
                                            load_k, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                            CLUSTER_BARRIER_INTERVAL)
        tlx.amd_sched_barrier()
        cd = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [64, 128])
        # A previous tile leaves at most two output stores pending. The first
        # two K-step waits retire them, so both slots are free here (K >= 384).
        for part in tl.static_range(8):
            if part >= 2:
                tlx.async_amd_descriptor_wait(1)
            view = tlx.local_view(cb, part % 2)
            for r in tl.static_range(2):
                for c in tl.static_range(4):
                    value = acc[((part // 2) * 2 + r) * 8 + (part % 2) * 4 + c]
                    tlx.local_store(tlx.local_slice(view, [r * 32, c * 32], [32, 32]), value)
            tlx.async_amd_descriptor_store(cd, view, [off_m + (part // 2) * 64, off_n + (part % 2) * 128])
        phase = (phase + k_iters) % 3
        tile = next_tile
    tlx.async_amd_descriptor_wait(0)
    if CLUSTER_SIZE > 1 and CLUSTER_BARRIER_INTERVAL > 0:
        tlx.cluster_barrier()
