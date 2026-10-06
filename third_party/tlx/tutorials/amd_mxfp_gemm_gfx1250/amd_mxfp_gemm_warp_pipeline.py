"""Eight-wave MXFP8 GEMM with partitioned LDS and FP32 output.

The K256 loop uses two payload slots, three scale slots, and alternating
load/compute stages for two groups of four waves. TDM refills from the leading
group overlap computation in the trailing group. The two N-half accumulators
are stored through a shared FP32 output panel after the input pipeline drains.

Run through ``bench.py --warp-pipeline``. This experimental path requires E4M3
inputs, preshuffled block-32 scales, full 256x256 tiles, and K divisible by 512
with K >= 1024.
"""
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


def get_layouts():
    a_piece = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [128, 256], [1, 0])
    b_piece = tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], [1, 64, 256], [2, 1, 0])
    return dict(
        A_LAYOUT=tlx.partitioned_shared_layout_encoding(2, 1, 0, a_piece),
        B_LAYOUT=tlx.partitioned_shared_layout_encoding(2, 2, 1, b_piece),
        MMA_LAYOUT=tlx.amd_wmma_layout([[8, 4], [4, 0], [8, 0]], [[0, 1], [0, 2], [1, 0], [2, 0]]),
        AS_LAYOUT=tlx.layout(shape=((32, 2, 2, 2), (4, 2)), stride=((4, 512, 256, 512), (1, 128))),
        BS_LAYOUT=tlx.layout(shape=((32, 2, 4), (4, 2, 2)), stride=((4, 256, 0), (1, 128, 512))),
        B_LOAD_LAYOUT=tlx.layout(shape=((16, 2, 2, 4), (16, 4, 4)), stride=((1, 2048, 64, 0), (128, 4096, 16))),
    )


@triton.jit
def _load_scale(buf, slot, start_k: tl.constexpr, LAYOUT: tl.constexpr):
    view = tlx.local_reshape(tlx.local_view(buf, slot), [2, 2, 32, 4, 4])
    view = tlx.local_reshape(tlx.local_trans(view, (0, 3, 2, 1, 4)), [256, 8])
    return tlx.local_load(tlx.local_slice(view, [0, start_k], [256, 4]), layout=LAYOUT, relaxed=True)


@triton.jit
def _load_operands(a_buf, b_buf, as_buf, bs_buf, slot, scale_slot, start_k: tl.constexpr, DOT_A: tl.constexpr,
                   DOT_B: tl.constexpr, AS_LAYOUT: tl.constexpr, BS_LAYOUT: tl.constexpr, B_LOAD_LAYOUT: tl.constexpr):
    scale_a = _load_scale(as_buf, scale_slot, start_k // 32, AS_LAYOUT)
    scale_b = _load_scale(bs_buf, scale_slot, start_k // 32, BS_LAYOUT)
    scale_b0, scale_b1 = scale_b.reshape(2, 128, 4).permute(1, 2, 0).split()
    a = tlx.local_load(tlx.local_slice(tlx.local_view(a_buf, slot), [0, start_k], [256, 128]), layout=DOT_A,
                       relaxed=True)
    b0_view = tlx.local_slice(tlx.local_view(b_buf, slot), [0, 0, start_k], [1, 128, 128])
    b1_view = tlx.local_slice(tlx.local_view(b_buf, slot), [0, 128, start_k], [1, 128, 128])
    b0 = tlx.local_load(tlx.local_trans(b0_view, (2, 0, 1)), layout=B_LOAD_LAYOUT, relaxed=True)
    b1 = tlx.local_load(tlx.local_trans(b1_view, (2, 0, 1)), layout=B_LOAD_LAYOUT, relaxed=True)
    b0 = tlx.require_layout(b0.reshape(128, 128), DOT_B)
    b1 = tlx.require_layout(b1.reshape(128, 128), DOT_B)
    return a, scale_a, b0, scale_b0, b1, scale_b1


@triton.jit
def _issue_data(a_desc, b_desc, a_buf, b_buf, slot):
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), warp_used_hint=15)
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), warp_used_hint=15)


@triton.jit
def _issue_scales(as_desc, bs_desc, as_buf, bs_buf, slot):
    tlx.async_amd_descriptor_load_fused(
        ((as_desc, tlx.local_view(as_buf, slot), 3), (bs_desc, tlx.local_view(bs_buf, slot), 12)))


@triton.jit
def _consume(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, acc0, acc1, slot, scale_slot,
             REFILL_SCALE: tl.constexpr, DOT_A: tl.constexpr, DOT_B: tl.constexpr, AS_LAYOUT: tl.constexpr,
             BS_LAYOUT: tl.constexpr, B_LOAD_LAYOUT: tl.constexpr):
    with tlx.warp_pipeline_stage("load_low"):
        a0, as0, b0, bs0, b1, bs1 = _load_operands(a_buf, b_buf, as_buf, bs_buf, slot, scale_slot, 0, DOT_A, DOT_B,
                                                   AS_LAYOUT, BS_LAYOUT, B_LOAD_LAYOUT)
    with tlx.warp_pipeline_stage("compute_low"):
        acc0 = tl.dot_scaled(a0, as0, "e4m3", b0, bs0, "e4m3", acc0)
        acc1 = tl.dot_scaled(a0, as0, "e4m3", b1, bs1, "e4m3", acc1)
    with tlx.warp_pipeline_stage("load_high_advance"):
        a1, as1, b2, bs2, b3, bs3 = _load_operands(a_buf, b_buf, as_buf, bs_buf, slot, scale_slot, 128, DOT_A, DOT_B,
                                                   AS_LAYOUT, BS_LAYOUT, B_LOAD_LAYOUT)
        next_a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        next_b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
        next_as_desc = as_desc
        next_bs_desc = bs_desc
        if REFILL_SCALE:
            next_as_desc = tlx.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
            next_bs_desc = tlx.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    with tlx.warp_pipeline_stage("compute_high"):
        acc0 = tl.dot_scaled(a1, as1, "e4m3", b2, bs2, "e4m3", acc0)
        acc1 = tl.dot_scaled(a1, as1, "e4m3", b3, bs3, "e4m3", acc1)
    with tlx.warp_pipeline_stage("refill"):
        _issue_data(a_desc, b_desc, a_buf, b_buf, slot)
        if REFILL_SCALE:
            _issue_scales(as_desc, bs_desc, as_buf, bs_buf, scale_slot)
    return next_a_desc, next_b_desc, next_as_desc, next_bs_desc, acc0, acc1


@triton.jit
def _tail(a_buf, b_buf, as_buf, bs_buf, acc0, acc1, slot, scale_slot, DOT_A: tl.constexpr, DOT_B: tl.constexpr,
          AS_LAYOUT: tl.constexpr, BS_LAYOUT: tl.constexpr, B_LOAD_LAYOUT: tl.constexpr):
    for sub_k in tl.static_range(2):
        with tlx.warp_pipeline_stage("tail_load"):
            a, sa, b0, sb0, b1, sb1 = _load_operands(a_buf, b_buf, as_buf, bs_buf, slot, scale_slot, sub_k * 128, DOT_A,
                                                     DOT_B, AS_LAYOUT, BS_LAYOUT, B_LOAD_LAYOUT)
        with tlx.warp_pipeline_stage("tail_compute"):
            acc0 = tl.dot_scaled(a, sa, "e4m3", b0, sb0, "e4m3", acc0)
            acc1 = tl.dot_scaled(a, sa, "e4m3", b1, sb1, "e4m3", acc1)
    return acc0, acc1


@triton.jit
def mxfp8_warp_pipeline_kernel(a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr, M, N, K, stride_am, stride_bn, stride_cm,
                               stride_scale, A_LAYOUT: tl.constexpr, B_LAYOUT: tl.constexpr, MMA_LAYOUT: tl.constexpr,
                               AS_LAYOUT: tl.constexpr, BS_LAYOUT: tl.constexpr, B_LOAD_LAYOUT: tl.constexpr,
                               GROUP_M: tl.constexpr = 8):
    pid = tl.program_id(0)
    num_m = M // 256
    num_n = N // 256
    group = pid // (GROUP_M * num_n)
    group_m = tl.minimum(num_m - group * GROUP_M, GROUP_M)
    pid_m = group * GROUP_M + pid % group_m
    pid_n = pid % (GROUP_M * num_n) // group_m
    a_desc = tl.make_tensor_descriptor(a_ptr + pid_m * 256 * stride_am, [M, K], [stride_am, 1], [256, 256])
    b_desc = tl.make_tensor_descriptor(b_ptr + pid_n * 256 * stride_bn, [1, N, K], [256 * stride_bn, stride_bn, 1],
                                       [1, 256, 256])
    as_desc = tl.make_tensor_descriptor(as_ptr + pid_m * 2 * stride_scale, [M // 128, K * 4], [stride_scale, 1],
                                        [2, 1024])
    bs_desc = tl.make_tensor_descriptor(bs_ptr + pid_n * 2 * stride_scale, [N // 128, K * 4], [stride_scale, 1],
                                        [2, 1024])
    a_buf = tlx.local_alloc((256, 256), tlx.dtype_of(a_ptr), 2, layout=A_LAYOUT)
    b_buf = tlx.local_alloc((1, 256, 256), tlx.dtype_of(b_ptr), 2, layout=B_LAYOUT)
    scale_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], [2, 1024])
    as_buf = tlx.local_alloc((2, 1024), tl.uint8, 3, layout=scale_layout)
    bs_buf = tlx.local_alloc((2, 1024), tl.uint8, 3, layout=scale_layout)
    for slot in tl.static_range(2):
        _issue_scales(as_desc, bs_desc, as_buf, bs_buf, slot)
        as_desc = tlx.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
        bs_desc = tlx.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
        _issue_data(a_desc, b_desc, a_buf, b_buf, slot)
        a_desc = tlx.update_tensor_descriptor(a_desc, add_offsets=[0, 256])
        b_desc = tlx.update_tensor_descriptor(b_desc, add_offsets=[0, 0, 256])
    _issue_scales(as_desc, bs_desc, as_buf, bs_buf, 2)
    as_desc = tlx.update_tensor_descriptor(as_desc, add_offsets=[0, 1024])
    bs_desc = tlx.update_tensor_descriptor(bs_desc, add_offsets=[0, 1024])
    dot_a: tl.constexpr = tlx.dot_operand_layout(0, MMA_LAYOUT, 16)
    dot_b: tl.constexpr = tlx.dot_operand_layout(1, MMA_LAYOUT, 16)
    acc0 = tlx.require_layout(tl.zeros([256, 128], tl.float32), MMA_LAYOUT)
    acc1 = tlx.require_layout(tl.zeros([256, 128], tl.float32), MMA_LAYOUT)
    k_iters = K // 256
    tl.assume(k_iters >= 4)
    tl.assume(k_iters % 2 == 0)
    for _ in range((k_iters - 4) // 6):
        for inner in tl.static_range(6):
            tlx.async_amd_descriptor_wait(4)
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                                    as_buf, bs_buf, acc0, acc1, inner % 2, inner % 3,
                                                                    True, dot_a, dot_b, AS_LAYOUT, BS_LAYOUT,
                                                                    B_LOAD_LAYOUT)
    for pair in range((k_iters - 4) % 6 // 2):
        for inner in tl.static_range(2):
            tlx.async_amd_descriptor_wait(4)
            a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf,
                                                                    as_buf, bs_buf, acc0, acc1, inner,
                                                                    (pair * 2 + inner) % 3, True, dot_a, dot_b,
                                                                    AS_LAYOUT, BS_LAYOUT, B_LOAD_LAYOUT)
    for inner in tl.static_range(2):
        tlx.async_amd_descriptor_wait(4)
        a_desc, b_desc, as_desc, bs_desc, acc0, acc1 = _consume(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf,
                                                                bs_buf, acc0, acc1, inner, (k_iters - 4 + inner) % 3,
                                                                inner == 0, dot_a, dot_b, AS_LAYOUT, BS_LAYOUT,
                                                                B_LOAD_LAYOUT)
    tlx.async_amd_descriptor_wait(3)
    acc0, acc1 = _tail(a_buf, b_buf, as_buf, bs_buf, acc0, acc1, 0, (k_iters - 2) % 3, dot_a, dot_b, AS_LAYOUT,
                       BS_LAYOUT, B_LOAD_LAYOUT)
    tlx.async_amd_descriptor_wait(0)
    acc0, acc1 = _tail(a_buf, b_buf, as_buf, bs_buf, acc0, acc1, 1, (k_iters - 1) % 3, dot_a, dot_b, AS_LAYOUT,
                       BS_LAYOUT, B_LOAD_LAYOUT)
    out_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for([[128, 8]], [256, 128])
    out = tlx.local_view(tlx.local_alloc((256, 128), tl.float32, 1, layout=out_layout), 0)
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [256, 128])
    tlx.local_store(out, acc0)
    tlx.async_amd_descriptor_store(c_desc, out, [pid_m * 256, pid_n * 256])
    tlx.async_amd_descriptor_wait(0)
    tlx.local_store(out, acc1)
    tlx.async_amd_descriptor_store(c_desc, out, [pid_m * 256, pid_n * 256 + 128])
    tlx.async_amd_descriptor_wait(0)
