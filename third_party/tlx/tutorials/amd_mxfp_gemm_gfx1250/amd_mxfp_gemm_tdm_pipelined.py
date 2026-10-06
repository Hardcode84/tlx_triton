"""
MXFP TDM-pipelined GEMM for AMD gfx1250 using TLX.

This mirrors the structure of the Gluon MXFP GEMM example: descriptor-backed
loads for A/B and scales, optional A scales, L2 prefetch hints, and baseline or
sliced K/N/MNK compute schedules.

The optional persistent sliceMNK path carries its data/scale rings across
output tiles and prefetches the next tile during the final K-ring rotation.
It also supports 128-element K stages with up to four ring slots, allowing
earlier refills without increasing the input footprint of two 256-element stages.
With 256x256x128 tiles, ``--output_staging`` uses separate LDS output slots.
For 256x256x256 tiles, combine it with ``--no-cross_tile_prefetch`` to reuse
the A ring for four quadrant output stores. This supports two input buffers
for A8W8 and two or three for A8W4, without a separate output allocation.
For A8W8 with two input buffers, ``--register_pipeline`` overlaps two K128
register stages and uses the free A/B stage for four FP32 output panels.
It keeps one next-tile input stage prefetched while storing the current tile.
For example, add ``--persistent -M 4096 --num_programs 32`` to the default
standalone configuration to process two output tiles per workgroup.

The sweep in ``bench.py`` benchmarks MXFP8 x MXFP8 and MXFP8 x MXFP4 at
8192x8192x8192 and 8192x8192x4096 by default, using persistent 256x256x128
tiles, three input buffers, output staging, and partial TDM fusion.
"""
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


@tl.core.builtin
def _mxgemm_multicast_rendezvous(_semantic=None):
    # Match independent CTA requests. TDM refill/wait barriers protect local LDS.
    _semantic.builder.create_amd_cluster_barrier()


@triton.constexpr_function
def _operand_shared_layout(block_shape):
    block_k = block_shape[1]
    pad_interval = 256 if block_k <= 256 else block_k
    return tlx.padded_shared_layout_encoding.with_identity_for([[pad_interval, 16]], block_shape, [1, 0])


@triton.constexpr_function
def _scale_shared_layout(block_shape):
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], block_shape, [1, 0])


@triton.constexpr_function
def _b_shared_layout(block_shape, transpose_b):
    if transpose_b:
        return _operand_shared_layout(block_shape)
    return tlx.padded_shared_layout_encoding.with_identity_for([[block_shape[1], 16]], block_shape, [1, 0])


def is_gfx1250_available():
    try:
        target = triton.runtime.driver.active.get_current_target()
        return target.arch == "gfx1250"
    except Exception:
        return False


def pack_scale(x: torch.Tensor, preshuffle_factor: int = 128) -> torch.Tensor:
    if x is None:
        return x
    non_k, k_scale = x.shape
    scale_kwidth = 4 if k_scale >= 4 else k_scale
    num_chunk_m = non_k // preshuffle_factor
    num_chunk_k = k_scale // scale_kwidth
    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, scale_kwidth)
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    return x.view(non_k // preshuffle_factor, k_scale * preshuffle_factor)


def fp8e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.view(torch.uint8).to(torch.int32)
    scale = scale << 23
    return scale.view(torch.float32)


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block, M, N, K):
    if a_scale is None:
        a_scale_f32 = torch.full((M, K), 1.0, dtype=torch.float32)
    else:
        a_scale_f32 = fp8e8m0_to_float32(a_scale).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = fp8e8m0_to_float32(b_scale).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]
    return torch.matmul(a.to(torch.float32) * a_scale_f32, b.to(torch.float32) * b_scale_f32)


@triton.jit
def _mxgemm_position_load_descriptor(desc, offsets, pred):
    # Persistent loads inherit the enabled base descriptor's predicate.
    # Replacing it on every update can keep packed descriptor fields live
    # across the K loop and spill them into VGPR lanes.
    if pred is None:
        return tlx.update_tensor_descriptor(desc, add_offsets=offsets, clamp_bounds=False)
    else:
        return tlx.update_tensor_descriptor(desc, add_offsets=offsets, clamp_bounds=False, pred=pred)


@triton.jit
def _mxgemm_issue_load_a_scale(
    a_scale_desc,
    a_scale_buf,
    load_idx,
    pred,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    load_slot=None,
):
    if WITH_A_SCALE:
        if pred is None:
            pred = True
        slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
        if SCALE_PRESHUFFLE:
            scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.async_amd_descriptor_load(a_scale_desc, tlx.local_view(a_scale_buf, slot), scale_offsets, pred=pred,
                                      clamp_bounds=False)


@triton.jit
def _mxgemm_issue_load_b_scale(
    b_scale_desc,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    load_slot=None,
):
    if pred is None:
        pred = True
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]
    tlx.async_amd_descriptor_load(b_scale_desc, tlx.local_view(b_scale_buf, slot), scale_offsets, pred=pred,
                                  clamp_bounds=False)


@triton.jit
def _mxgemm_issue_load_a_data(
    a_desc,
    a_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    load_slot=None,
):
    if pred is None:
        pred = True
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), [0, load_idx * BLOCK_K_PACKED_A], pred=pred,
                                  clamp_bounds=False)


@triton.jit
def _mxgemm_issue_load_b_data(
    b_desc,
    b_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_B: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    load_slot=None,
):
    if pred is None:
        pred = True
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), b_offsets, pred=pred, clamp_bounds=False)


@triton.jit
def _mxgemm_issue_loads(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    a_buf,
    b_buf,
    a_scale_buf,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    TDM_FUSION: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    GROUP_SIZE_M: tl.constexpr = 8,
    XCD_REMAP_MODE: tl.constexpr = 0,
    load_slot=None,
    SCALES_FIRST: tl.constexpr = False,
):
    slot = load_idx % NUM_BUFFERS if load_slot is None else load_slot
    if CLUSTER_SIZE > 1 and CLUSTER_MULTICAST:
        tl.static_assert(WITH_A_SCALE and TDM_FUSION != "none")
        tl.static_assert(GROUP_SIZE_M == 4 or GROUP_SIZE_M == 8)
        rank = tlx.cluster_cta_rank()
        if XCD_REMAP_MODE == 2 and CLUSTER_SIZE == 4 and GROUP_SIZE_M == 4:
            a_mask, b_mask = 5 << (rank & 1), 3 << (rank & 2)
        else:
            # Without remapping, consecutive cluster CTAs cover consecutive
            # M tiles at the same N, so only B and its scales are shared.
            a_mask, b_mask = 0, (1 << CLUSTER_SIZE) - 1
        data_masks = [a_mask, b_mask]
        all_masks = [a_mask, b_mask, a_mask, b_mask]
    else:
        data_masks = None
        all_masks = None
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]

    # Spread each fused member over the same fraction of the workgroup when
    # two waves run on each SIMD. The additional waves load disjoint rows.
    WAVE_MASK_REPEAT: tl.constexpr = 0x11 if tlx.num_warps() == 8 else 1
    if TDM_FUSION == "4way":
        tl.static_assert(WITH_A_SCALE, "4-way TDM fusion requires WITH_A_SCALE")
        tlx.async_amd_descriptor_load_fused([
            (_mxgemm_position_load_descriptor(a_desc, [0, load_idx * BLOCK_K_PACKED_A],
                                              pred), tlx.local_view(a_buf, slot), 1 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_desc, b_offsets, pred), tlx.local_view(b_buf,
                                                                                       slot), 2 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(
                a_scale_buf, slot), 4 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(
                b_scale_buf, slot), 8 * WAVE_MASK_REPEAT),
        ], multicast_masks=all_masks)
    elif TDM_FUSION == "2way":
        tlx.async_amd_descriptor_load_fused([
            (_mxgemm_position_load_descriptor(a_desc, [0, load_idx * BLOCK_K_PACKED_A],
                                              pred), tlx.local_view(a_buf, slot), 3 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_desc, b_offsets, pred), tlx.local_view(b_buf,
                                                                                       slot), 12 * WAVE_MASK_REPEAT),
        ], multicast_masks=data_masks)
        if WITH_A_SCALE:
            tlx.async_amd_descriptor_load_fused([
                (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(
                    a_scale_buf, slot), 3 * WAVE_MASK_REPEAT),
                (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(
                    b_scale_buf, slot), 12 * WAVE_MASK_REPEAT),
            ], multicast_masks=data_masks)
        else:
            _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED,
                                       NUM_BUFFERS, SCALE_PRESHUFFLE, load_slot=slot)
    elif TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "partial TDM fusion requires WITH_A_SCALE")
        if SCALES_FIRST:
            # The register pipeline consumes scales early. Enqueue their small
            # transfers before the larger payload copies.
            tlx.async_amd_descriptor_load_fused([
                (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(
                    a_scale_buf, slot), 5 * WAVE_MASK_REPEAT),
                (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(
                    b_scale_buf, slot), 10 * WAVE_MASK_REPEAT),
            ], multicast_masks=data_masks)
        tlx.async_amd_descriptor_load_fused([
            (_mxgemm_position_load_descriptor(a_desc, [0, load_idx * BLOCK_K_PACKED_A],
                                              pred), tlx.local_view(a_buf, slot), 5 * WAVE_MASK_REPEAT),
            (_mxgemm_position_load_descriptor(b_desc, b_offsets, pred), tlx.local_view(b_buf,
                                                                                       slot), 10 * WAVE_MASK_REPEAT),
        ], multicast_masks=data_masks)
        if not SCALES_FIRST:
            tlx.async_amd_descriptor_load_fused([
                (_mxgemm_position_load_descriptor(a_scale_desc, scale_offsets, pred), tlx.local_view(
                    a_scale_buf, slot), 5 * WAVE_MASK_REPEAT),
                (_mxgemm_position_load_descriptor(b_scale_desc, scale_offsets, pred), tlx.local_view(
                    b_scale_buf, slot), 10 * WAVE_MASK_REPEAT),
            ], multicast_masks=data_masks)
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_FUSION must be one of: none, 2way, 4way, partial")
        _mxgemm_issue_load_a_scale(a_scale_desc, a_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE, WITH_A_SCALE, load_slot=slot)
        _mxgemm_issue_load_b_scale(b_scale_desc, b_scale_buf, load_idx, pred, BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS,
                                   SCALE_PRESHUFFLE, load_slot=slot)
        _mxgemm_issue_load_a_data(a_desc, a_buf, load_idx, pred, BLOCK_K_PACKED_A, NUM_BUFFERS, load_slot=slot)
        _mxgemm_issue_load_b_data(b_desc, b_buf, load_idx, pred, BLOCK_K_PACKED_B, NUM_BUFFERS, TRANSPOSE_B,
                                  load_slot=slot)
    return load_idx + 1


@triton.jit
def _mxgemm_issue_split_loads(
    a0_desc,
    a1_desc,
    b0_desc,
    b1_desc,
    a_scale_desc,
    b_scale_desc,
    a0_buf,
    a1_buf,
    b0_buf,
    b1_buf,
    a_scale_buf,
    b_scale_buf,
    load_idx,
    pred,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    TDM_FUSION: tl.constexpr,
):
    slot = load_idx % NUM_BUFFERS
    a_offsets = [0, load_idx * BLOCK_K_PACKED_A]
    if TRANSPOSE_B:
        b_offsets = [0, load_idx * BLOCK_K_PACKED_B]
    else:
        b_offsets = [load_idx * BLOCK_K_PACKED_B, 0]
    if SCALE_PRESHUFFLE:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED]
    else:
        scale_offsets = [0, load_idx * BLOCK_K_SCALE_PRESHUFFLED // 128]

    if TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "split partial TDM fusion requires WITH_A_SCALE")
        a0_load_desc = tlx.update_tensor_descriptor(a0_desc, add_offsets=a_offsets)
        a1_load_desc = tlx.update_tensor_descriptor(a1_desc, add_offsets=a_offsets)
        b0_load_desc = tlx.update_tensor_descriptor(b0_desc, add_offsets=b_offsets)
        b1_load_desc = tlx.update_tensor_descriptor(b1_desc, add_offsets=b_offsets)
        a_scale_load_desc = tlx.update_tensor_descriptor(a_scale_desc, add_offsets=scale_offsets)
        b_scale_load_desc = tlx.update_tensor_descriptor(b_scale_desc, add_offsets=scale_offsets)
        tlx.async_amd_descriptor_load_fused([
            (tlx.update_tensor_descriptor(a0_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(a0_buf, slot), 5),
            (tlx.update_tensor_descriptor(b0_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(b0_buf, slot), 10),
        ])
        tlx.async_amd_descriptor_load_fused([
            (tlx.update_tensor_descriptor(a1_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(a1_buf, slot), 5),
            (tlx.update_tensor_descriptor(b1_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(b1_buf, slot), 10),
        ])
        tlx.async_amd_descriptor_load_fused([
            (tlx.update_tensor_descriptor(a_scale_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(a_scale_buf, slot), 5),
            (tlx.update_tensor_descriptor(b_scale_load_desc, add_offsets=[0, 0], clamp_bounds=False,
                                          pred=pred), tlx.local_view(b_scale_buf, slot), 10),
        ])
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_SPLIT supports TDM_FUSION values: none, partial")
        if WITH_A_SCALE:
            a_scale_load_desc = tlx.update_tensor_descriptor(a_scale_desc, add_offsets=scale_offsets)
            tlx.async_amd_descriptor_load(a_scale_load_desc, tlx.local_view(a_scale_buf, slot), [0, 0], pred=pred,
                                          clamp_bounds=False)
        b_scale_load_desc = tlx.update_tensor_descriptor(b_scale_desc, add_offsets=scale_offsets)
        a0_load_desc = tlx.update_tensor_descriptor(a0_desc, add_offsets=a_offsets)
        a1_load_desc = tlx.update_tensor_descriptor(a1_desc, add_offsets=a_offsets)
        b0_load_desc = tlx.update_tensor_descriptor(b0_desc, add_offsets=b_offsets)
        b1_load_desc = tlx.update_tensor_descriptor(b1_desc, add_offsets=b_offsets)
        tlx.async_amd_descriptor_load(b_scale_load_desc, tlx.local_view(b_scale_buf, slot), [0, 0], pred=pred,
                                      clamp_bounds=False)
        tlx.async_amd_descriptor_load(a0_load_desc, tlx.local_view(a0_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(b0_load_desc, tlx.local_view(b0_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(a1_load_desc, tlx.local_view(a1_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(b1_load_desc, tlx.local_view(b1_buf, slot), [0, 0], pred=pred, clamp_bounds=False)
    return load_idx + 1


@triton.jit
def _mxgemm_issue_l2_prefetches(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    load_idx,
    pred,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if L2_PREFETCH_DISTANCE >= 0:
        prefetch_iteration = load_idx + L2_PREFETCH_DISTANCE
        if WITH_A_SCALE:
            if SCALE_PRESHUFFLE:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
            else:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
            tlx.amd_descriptor_prefetch_tensor(a_scale_desc, a_scale_offsets, pred=pred)
        if SCALE_PRESHUFFLE:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.amd_descriptor_prefetch_tensor(b_scale_desc, b_scale_offsets, pred=pred)
        tlx.amd_descriptor_prefetch_tensor(a_desc, [0, prefetch_iteration * BLOCK_K_PACKED_A], pred=pred)
        if TRANSPOSE_B:
            tlx.amd_descriptor_prefetch_tensor(b_desc, [0, prefetch_iteration * BLOCK_K_PACKED_B], pred=pred)
        else:
            tlx.amd_descriptor_prefetch_tensor(b_desc, [prefetch_iteration * BLOCK_K_PACKED_B, 0], pred=pred)


@triton.jit
def _mxgemm_issue_split_l2_prefetches(
    a0_desc,
    a1_desc,
    b0_desc,
    b1_desc,
    a_scale_desc,
    b_scale_desc,
    load_idx,
    pred,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if L2_PREFETCH_DISTANCE >= 0:
        prefetch_iteration = load_idx + L2_PREFETCH_DISTANCE
        if WITH_A_SCALE:
            if SCALE_PRESHUFFLE:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
            else:
                a_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
            tlx.amd_descriptor_prefetch_tensor(a_scale_desc, a_scale_offsets, pred=pred)
        if SCALE_PRESHUFFLE:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED]
        else:
            b_scale_offsets = [0, prefetch_iteration * BLOCK_K_SCALE_PRESHUFFLED // 128]
        tlx.amd_descriptor_prefetch_tensor(b_scale_desc, b_scale_offsets, pred=pred)
        tlx.amd_descriptor_prefetch_tensor(a0_desc, [0, prefetch_iteration * BLOCK_K_PACKED_A], pred=pred)
        tlx.amd_descriptor_prefetch_tensor(a1_desc, [0, prefetch_iteration * BLOCK_K_PACKED_A], pred=pred)
        if TRANSPOSE_B:
            b_offsets = [0, prefetch_iteration * BLOCK_K_PACKED_B]
        else:
            b_offsets = [prefetch_iteration * BLOCK_K_PACKED_B, 0]
        tlx.amd_descriptor_prefetch_tensor(b0_desc, b_offsets, pred=pred)
        tlx.amd_descriptor_prefetch_tensor(b1_desc, b_offsets, pred=pred)


@triton.jit
def _mxgemm_issue_l2_prefetches_prologue(
    a_desc,
    b_desc,
    a_scale_desc,
    b_scale_desc,
    load_idx,
    L2_PREFETCH_DISTANCE: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    BLOCK_K_PACKED_A: tl.constexpr,
    BLOCK_K_PACKED_B: tl.constexpr,
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    if L2_PREFETCH_DISTANCE >= 0:
        always = tl.full((), True, dtype=tl.int1)
        for i in tl.static_range(NUM_BUFFERS, NUM_BUFFERS + L2_PREFETCH_DISTANCE):
            _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always, i,
                                        BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B,
                                        SCALE_PRESHUFFLE, WITH_A_SCALE)


@triton.jit
def _mxgemm_load_a_operand(
    a_buf,
    a_scale_buf,
    wmma_idx,
    subtile_start_idx_m: tl.constexpr,
    subtile_start_idx_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIV_FACTOR_A: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_M_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_SUBTILES_M: tl.constexpr,
    NUM_SUBTILES_K: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
    SCALES_FIRST: tl.constexpr = False,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_m: tl.constexpr = BLOCK_M // NUM_SUBTILES_M
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_m: tl.constexpr = subtile_start_idx_m * subtile_m
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    a_view = tlx.local_view(a_buf, slot)
    if NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
        a_view = tlx.local_slice(a_view, [subtile_start_m, subtile_start_k // DIV_FACTOR_A],
                                 [subtile_m, subtile_k // DIV_FACTOR_A])
    if not SCALES_FIRST:
        a = tlx.local_load(a_view)

    if WITH_A_SCALE:
        if SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_reshape(
                tlx.local_view(a_scale_buf, slot),
                [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
            )
            scale_a_view = tlx.local_trans(scale_a_view, (0, 3, 2, 1, 4))
            scale_a_view = tlx.local_reshape(scale_a_view, [BLOCK_M, BLOCK_K_SCALE])
        else:
            scale_a_view = tlx.local_view(a_scale_buf, slot)
        if NUM_SUBTILES_M != 1 or NUM_SUBTILES_K != 1:
            scale_a_view = tlx.local_slice(scale_a_view, [subtile_start_m, subtile_start_k // SCALE_BLOCK],
                                           [subtile_m, subtile_k // SCALE_BLOCK])
        elif not SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_slice(scale_a_view, [0, 0], [BLOCK_M, BLOCK_K_SCALE])
        scale_a = tlx.local_load(scale_a_view)
    else:
        scale_a = tl.full((subtile_m, subtile_k // SCALE_BLOCK), 127, dtype=tl.uint8)

    if SCALES_FIRST:
        a = tlx.local_load(a_view)
    return a, scale_a


@triton.jit
def _mxgemm_load_a_operand_split(
    a0_buf,
    a1_buf,
    a_scale_buf,
    wmma_idx,
    subtile_start_idx_m: tl.constexpr,
    subtile_start_idx_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIV_FACTOR_A: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_M_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_SUBTILES_M: tl.constexpr,
    NUM_SUBTILES_K: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    WITH_A_SCALE: tl.constexpr,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_m: tl.constexpr = BLOCK_M // NUM_SUBTILES_M
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_m: tl.constexpr = subtile_start_idx_m * subtile_m
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    if subtile_start_idx_m == 0:
        a_view = tlx.local_view(a0_buf, slot)
    else:
        a_view = tlx.local_view(a1_buf, slot)
    if NUM_SUBTILES_K != 1:
        a_view = tlx.local_slice(a_view, [0, subtile_start_k // DIV_FACTOR_A], [subtile_m, subtile_k // DIV_FACTOR_A])
    a = tlx.local_load(a_view)

    if WITH_A_SCALE:
        if SCALE_PRESHUFFLE:
            scale_a_view = tlx.local_reshape(
                tlx.local_view(a_scale_buf, slot),
                [BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
            )
            scale_a_view = tlx.local_trans(scale_a_view, (0, 3, 2, 1, 4))
            scale_a_view = tlx.local_reshape(scale_a_view, [BLOCK_M, BLOCK_K_SCALE])
        else:
            scale_a_view = tlx.local_view(a_scale_buf, slot)
        scale_a_view = tlx.local_slice(scale_a_view, [subtile_start_m, subtile_start_k // SCALE_BLOCK],
                                       [subtile_m, subtile_k // SCALE_BLOCK])
        scale_a = tlx.local_load(scale_a_view)
    else:
        scale_a = tl.full((subtile_m, subtile_k // SCALE_BLOCK), 127, dtype=tl.uint8)

    return a, scale_a


@triton.jit
def _mxgemm_load_b_operand(
    b_buf,
    b_scale_buf,
    wmma_idx,
    subtile_start_idx_k: tl.constexpr,
    subtile_start_idx_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIV_FACTOR_B: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_N_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_SUBTILES_N: tl.constexpr,
    NUM_SUBTILES_K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
    SCALES_FIRST: tl.constexpr = False,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_n: tl.constexpr = BLOCK_N // NUM_SUBTILES_N
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_n: tl.constexpr = subtile_start_idx_n * subtile_n
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    b_view = tlx.local_view(b_buf, slot)
    if TRANSPOSE_B:
        if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_n, subtile_start_k // DIV_FACTOR_B],
                                     [subtile_n, subtile_k // DIV_FACTOR_B])
        b_view = tlx.local_trans(b_view)
    else:
        if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_k // DIV_FACTOR_B, subtile_start_n],
                                     [subtile_k // DIV_FACTOR_B, subtile_n])
    if not SCALES_FIRST:
        b = tlx.local_load(b_view)

    if SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_reshape(
            tlx.local_view(b_scale_buf, slot),
            [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
        )
        scale_b_view = tlx.local_trans(scale_b_view, (0, 3, 2, 1, 4))
        scale_b_view = tlx.local_reshape(scale_b_view, [BLOCK_N, BLOCK_K_SCALE])
    else:
        scale_b_view = tlx.local_view(b_scale_buf, slot)
    if NUM_SUBTILES_N != 1 or NUM_SUBTILES_K != 1:
        scale_b_view = tlx.local_slice(scale_b_view, [subtile_start_n, subtile_start_k // SCALE_BLOCK],
                                       [subtile_n, subtile_k // SCALE_BLOCK])
    elif not SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_slice(scale_b_view, [0, 0], [BLOCK_N, BLOCK_K_SCALE])
    scale_b = tlx.local_load(scale_b_view)

    if SCALES_FIRST:
        b = tlx.local_load(b_view)
    return b, scale_b


@triton.jit
def _mxgemm_load_b_operand_split(
    b0_buf,
    b1_buf,
    b_scale_buf,
    wmma_idx,
    subtile_start_idx_k: tl.constexpr,
    subtile_start_idx_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIV_FACTOR_B: tl.constexpr,
    SCALE_BLOCK: tl.constexpr,
    BLOCK_K_SCALE: tl.constexpr,
    BLOCK_N_PRESHUFFLED: tl.constexpr,
    SCALE_KWIDTH: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_SUBTILES_N: tl.constexpr,
    NUM_SUBTILES_K: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,
    SCALE_PRESHUFFLE: tl.constexpr,
):
    slot = wmma_idx % NUM_BUFFERS
    subtile_n: tl.constexpr = BLOCK_N // NUM_SUBTILES_N
    subtile_k: tl.constexpr = BLOCK_K // NUM_SUBTILES_K
    subtile_start_n: tl.constexpr = subtile_start_idx_n * subtile_n
    subtile_start_k: tl.constexpr = subtile_start_idx_k * subtile_k

    if subtile_start_idx_n == 0:
        b_view = tlx.local_view(b0_buf, slot)
    else:
        b_view = tlx.local_view(b1_buf, slot)
    if TRANSPOSE_B:
        if NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [0, subtile_start_k // DIV_FACTOR_B],
                                     [subtile_n, subtile_k // DIV_FACTOR_B])
        b = tlx.local_load(tlx.local_trans(b_view))
    else:
        if NUM_SUBTILES_K != 1:
            b_view = tlx.local_slice(b_view, [subtile_start_k // DIV_FACTOR_B, 0],
                                     [subtile_k // DIV_FACTOR_B, subtile_n])
        b = tlx.local_load(b_view)

    if SCALE_PRESHUFFLE:
        scale_b_view = tlx.local_reshape(
            tlx.local_view(b_scale_buf, slot),
            [BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, 128 // 4, 4, SCALE_KWIDTH],
        )
        scale_b_view = tlx.local_trans(scale_b_view, (0, 3, 2, 1, 4))
        scale_b_view = tlx.local_reshape(scale_b_view, [BLOCK_N, BLOCK_K_SCALE])
    else:
        scale_b_view = tlx.local_view(b_scale_buf, slot)
    scale_b_view = tlx.local_slice(scale_b_view, [subtile_start_n, subtile_start_k // SCALE_BLOCK],
                                   [subtile_n, subtile_k // SCALE_BLOCK])
    scale_b = tlx.local_load(scale_b_view)

    return b, scale_b


@triton.jit
def mxgemm_tdm_pipelined_kernel(
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
    SCHEDULE: tl.constexpr = "baseline",
    TDM_FUSION: tl.constexpr = "none",
    L2_PREFETCH_DISTANCE: tl.constexpr = -1,
    TDM_SPLIT: tl.constexpr = False,
):
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    BLOCK_K_PACKED_A: tl.constexpr = BLOCK_K // DIV_FACTOR_A
    BLOCK_K_PACKED_B: tl.constexpr = BLOCK_K // DIV_FACTOR_B
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_BLOCK
    SCALE_KWIDTH: tl.constexpr = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
    BLOCK_M_PRESHUFFLED: tl.constexpr = BLOCK_M // 128
    BLOCK_N_PRESHUFFLED: tl.constexpr = BLOCK_N // 128
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr = BLOCK_K_SCALE * 128
    STORE_INSTR_M: tl.constexpr = 32 if (DTYPE_A == "e2m1" and DTYPE_B == "e2m1") else 16
    STORE_INSTR_SHAPE: tl.constexpr = (STORE_INSTR_M, 16, 128)
    if SCALE_PRESHUFFLE:
        STORE_WARP_BASES: tl.constexpr = ((0, 2), (2, 0))
        STORE_REG_BASES: tl.constexpr = ((0, 1), (1, 0))
    else:
        STORE_WARP_BASES: tl.constexpr = ((0, 1), (1, 0))
        STORE_REG_BASES: tl.constexpr = ()
    if TDM_FUSION == "4way":
        tl.static_assert(WITH_A_SCALE, "4-way TDM fusion requires WITH_A_SCALE")
        NUM_LOADS_IN_BATCH: tl.constexpr = 1
    elif TDM_FUSION == "2way":
        NUM_LOADS_IN_BATCH: tl.constexpr = 2
    elif TDM_FUSION == "partial":
        tl.static_assert(WITH_A_SCALE, "partial TDM fusion requires WITH_A_SCALE")
        NUM_LOADS_IN_BATCH: tl.constexpr = 2
    else:
        tl.static_assert(TDM_FUSION == "none", "TDM_FUSION must be one of: none, 2way, 4way, partial")
        NUM_LOADS_IN_BATCH: tl.constexpr = 4 if WITH_A_SCALE else 3
    if SCHEDULE == "sliceMNK":
        NUM_SUBTILES_M: tl.constexpr = 2
        NUM_SUBTILES_N: tl.constexpr = 2
        # Native MXFP WMMAs consume K128; two K64 halves would double the
        # matrix instructions for a K128 input stage.
        NUM_SUBTILES_K: tl.constexpr = 1 if BLOCK_K == 128 else 2
    elif SCHEDULE == "sliceNK":
        NUM_SUBTILES_M: tl.constexpr = 1
        NUM_SUBTILES_N: tl.constexpr = 2
        NUM_SUBTILES_K: tl.constexpr = 2
    elif SCHEDULE == "sliceK":
        NUM_SUBTILES_M: tl.constexpr = 1
        NUM_SUBTILES_N: tl.constexpr = 1
        NUM_SUBTILES_K: tl.constexpr = 2
    else:
        tl.static_assert(SCHEDULE == "baseline")
        NUM_SUBTILES_M: tl.constexpr = 1
        NUM_SUBTILES_N: tl.constexpr = 1
        NUM_SUBTILES_K: tl.constexpr = 1
    if TDM_SPLIT:
        tl.static_assert(SCHEDULE == "sliceMNK", "TDM_SPLIT is only supported for the sliceMNK schedule")
        tl.static_assert(BLOCK_M % 2 == 0 and BLOCK_N % 2 == 0, "TDM_SPLIT requires even BLOCK_M and BLOCK_N")
        if TDM_FUSION == "partial":
            tl.static_assert(WITH_A_SCALE, "split partial TDM fusion requires WITH_A_SCALE")
            SPLIT_LOADS_IN_BATCH: tl.constexpr = 3
        else:
            tl.static_assert(TDM_FUSION == "none", "TDM_SPLIT supports TDM_FUSION values: none, partial")
            SPLIT_LOADS_IN_BATCH: tl.constexpr = 6 if WITH_A_SCALE else 5

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if TDM_SPLIT:
        HALF_BLOCK_M: tl.constexpr = BLOCK_M // 2
        HALF_BLOCK_N: tl.constexpr = BLOCK_N // 2
        a0_desc = tl.make_tensor_descriptor(
            a_ptr + pid_m * BLOCK_M * stride_am,
            shape=[M, K // DIV_FACTOR_A],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[HALF_BLOCK_M, BLOCK_K_PACKED_A],
        )
        a1_desc = tl.make_tensor_descriptor(
            a_ptr + pid_m * BLOCK_M * stride_am + HALF_BLOCK_M * stride_am,
            shape=[M, K // DIV_FACTOR_A],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[HALF_BLOCK_M, BLOCK_K_PACKED_A],
        )
        if TRANSPOSE_B:
            b0_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[N, K // DIV_FACTOR_B],
                strides=[stride_bn, tl.constexpr(1)],
                block_shape=[HALF_BLOCK_N, BLOCK_K_PACKED_B],
            )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn + HALF_BLOCK_N * stride_bn,
                shape=[N, K // DIV_FACTOR_B],
                strides=[stride_bn, tl.constexpr(1)],
                block_shape=[HALF_BLOCK_N, BLOCK_K_PACKED_B],
            )
            b0_shape: tl.constexpr = (HALF_BLOCK_N, BLOCK_K_PACKED_B)
            b1_shape: tl.constexpr = (HALF_BLOCK_N, BLOCK_K_PACKED_B)
        else:
            b0_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[K // DIV_FACTOR_B, N],
                strides=[stride_bk, tl.constexpr(1)],
                block_shape=[BLOCK_K_PACKED_B, HALF_BLOCK_N],
            )
            b1_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn + HALF_BLOCK_N * stride_bn,
                shape=[K // DIV_FACTOR_B, N],
                strides=[stride_bk, tl.constexpr(1)],
                block_shape=[BLOCK_K_PACKED_B, HALF_BLOCK_N],
            )
            b0_shape: tl.constexpr = (BLOCK_K_PACKED_B, HALF_BLOCK_N)
            b1_shape: tl.constexpr = (BLOCK_K_PACKED_B, HALF_BLOCK_N)
        a_split_layout: tl.constexpr = _operand_shared_layout([HALF_BLOCK_M, BLOCK_K_PACKED_A])
        b_split_layout: tl.constexpr = _b_shared_layout(b0_shape, TRANSPOSE_B)
        a0_buf = tlx.local_alloc((HALF_BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                                 layout=a_split_layout)
        a1_buf = tlx.local_alloc((HALF_BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                                 layout=a_split_layout)
        b0_buf = tlx.local_alloc(b0_shape, tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_split_layout)
        b1_buf = tlx.local_alloc(b1_shape, tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_split_layout)
    else:
        a_desc = tl.make_tensor_descriptor(
            a_ptr + pid_m * BLOCK_M * stride_am,
            shape=[M, K // DIV_FACTOR_A],
            strides=[stride_am, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_K_PACKED_A],
        )
        if TRANSPOSE_B:
            b_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[N, K // DIV_FACTOR_B],
                strides=[stride_bn, tl.constexpr(1)],
                block_shape=[BLOCK_N, BLOCK_K_PACKED_B],
            )
            b_layout: tl.constexpr = _operand_shared_layout([BLOCK_N, BLOCK_K_PACKED_B])
            b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K_PACKED_B), tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_layout)
        else:
            b_desc = tl.make_tensor_descriptor(
                b_ptr + pid_n * BLOCK_N * stride_bn,
                shape=[K // DIV_FACTOR_B, N],
                strides=[stride_bk, tl.constexpr(1)],
                block_shape=[BLOCK_K_PACKED_B, BLOCK_N],
            )
            b_layout: tl.constexpr = _b_shared_layout([BLOCK_K_PACKED_B, BLOCK_N], TRANSPOSE_B)
            b_buf = tlx.local_alloc((BLOCK_K_PACKED_B, BLOCK_N), tlx.dtype_of(b_ptr), NUM_BUFFERS, layout=b_layout)

    if SCALE_PRESHUFFLE:
        a_scale_desc = tl.make_tensor_descriptor(
            a_scale + pid_m * BLOCK_M_PRESHUFFLED * stride_scale,
            shape=[M // 128, K // SCALE_BLOCK * 128],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],
        )
        b_scale_desc = tl.make_tensor_descriptor(
            b_scale + pid_n * BLOCK_N_PRESHUFFLED * stride_scale,
            shape=[N // 128, K // SCALE_BLOCK * 128],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED],
        )
        a_scale_shape: tl.constexpr = (BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED)
        b_scale_shape: tl.constexpr = (BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED)
    else:
        BLOCK_K_SCALE_LOAD: tl.constexpr = 16 if BLOCK_K_SCALE < 16 else BLOCK_K_SCALE
        a_scale_desc = tl.make_tensor_descriptor(
            a_scale + pid_m * BLOCK_M * stride_scale,
            shape=[M, K // SCALE_BLOCK],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_K_SCALE_LOAD],
        )
        b_scale_desc = tl.make_tensor_descriptor(
            b_scale + pid_n * BLOCK_N * stride_scale,
            shape=[N, K // SCALE_BLOCK],
            strides=[stride_scale, tl.constexpr(1)],
            block_shape=[BLOCK_N, BLOCK_K_SCALE_LOAD],
        )
        a_scale_shape: tl.constexpr = (BLOCK_M, BLOCK_K_SCALE_LOAD)
        b_scale_shape: tl.constexpr = (BLOCK_N, BLOCK_K_SCALE_LOAD)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_offsets = (stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]).to(tl.int32)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SCHEDULE == "sliceMNK":
        c_desc = tl.make_tensor_descriptor(
            c_ptr,
            shape=[M, N],
            strides=[stride_cm, tl.constexpr(1)],
            block_shape=[BLOCK_M, BLOCK_N],
        )
        c_off_m = pid_m * BLOCK_M
        c_off_n = pid_n * BLOCK_N

    a_scale_layout: tl.constexpr = _scale_shared_layout(a_scale_shape)
    b_scale_layout: tl.constexpr = _scale_shared_layout(b_scale_shape)
    if not TDM_SPLIT:
        a_layout: tl.constexpr = _operand_shared_layout([BLOCK_M, BLOCK_K_PACKED_A])
        a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K_PACKED_A), tlx.dtype_of(a_ptr), NUM_BUFFERS, layout=a_layout)
    a_scale_buf = tlx.local_alloc(a_scale_shape, tlx.dtype_of(a_scale), NUM_BUFFERS, layout=a_scale_layout)
    b_scale_buf = tlx.local_alloc(b_scale_shape, tlx.dtype_of(b_scale), NUM_BUFFERS, layout=b_scale_layout)

    K_ITERS = tl.cdiv(K, BLOCK_K)
    epilogue_lb = K_ITERS - (NUM_BUFFERS - 1)
    load_idx = 0
    wmma_idx = 0

    if SCHEDULE == "baseline" or SCHEDULE == "sliceK":
        _mxgemm_issue_l2_prefetches_prologue(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, L2_PREFETCH_DISTANCE,
                                             NUM_BUFFERS, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED,
                                             TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)

    always = tl.full((), True, dtype=tl.int1)
    for _ in tl.static_range(NUM_BUFFERS - 1):
        if TDM_SPLIT:
            load_idx = _mxgemm_issue_split_loads(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc, a0_buf,
                                                 a1_buf, b0_buf, b1_buf, a_scale_buf, b_scale_buf, load_idx, always,
                                                 BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED,
                                                 NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION)
        else:
            load_idx = _mxgemm_issue_loads(
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                a_buf,
                b_buf,
                a_scale_buf,
                b_scale_buf,
                load_idx,
                always,
                BLOCK_K_PACKED_A,
                BLOCK_K_PACKED_B,
                BLOCK_K_SCALE_PRESHUFFLED,
                NUM_BUFFERS,
                TRANSPOSE_B,
                SCALE_PRESHUFFLE,
                WITH_A_SCALE,
                TDM_FUSION,
            )

    tl.assume(K_ITERS > 0)
    if SCHEDULE == "sliceMNK":
        SUBTILE_M: tl.constexpr = BLOCK_M // 2
        SUBTILE_N: tl.constexpr = BLOCK_N // 2
        if TDM_SPLIT:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 2) * SPLIT_LOADS_IN_BATCH)
            a00, scale_a00 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M, BLOCK_K,
                                                          DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                          SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                          SCALE_PRESHUFFLE, WITH_A_SCALE)
            b00, scale_b00 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N, BLOCK_K,
                                                          DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                          SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                          TRANSPOSE_B, SCALE_PRESHUFFLE)
            load_idx = _mxgemm_issue_split_loads(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc, a0_buf,
                                                 a1_buf, b0_buf, b1_buf, a_scale_buf, b_scale_buf, load_idx, always,
                                                 BLOCK_K_PACKED_A, BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED,
                                                 NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION)
        else:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 2) * NUM_LOADS_IN_BATCH)
            a00, scale_a00 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M, BLOCK_K, DIV_FACTOR_A,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                    WITH_A_SCALE)
            b00, scale_b00 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N, BLOCK_K, DIV_FACTOR_B,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                    SCALE_PRESHUFFLE)
            load_idx = _mxgemm_issue_loads(
                a_desc,
                b_desc,
                a_scale_desc,
                b_scale_desc,
                a_buf,
                b_buf,
                a_scale_buf,
                b_scale_buf,
                load_idx,
                always,
                BLOCK_K_PACKED_A,
                BLOCK_K_PACKED_B,
                BLOCK_K_SCALE_PRESHUFFLED,
                NUM_BUFFERS,
                TRANSPOSE_B,
                SCALE_PRESHUFFLE,
                WITH_A_SCALE,
                TDM_FUSION,
            )
        # Keep the first subtile's scales in the WMMA distribution across the
        # loop edge. Converting blocked loop-carried scales adds LDS exchanges
        # and barriers before the next iteration's first dot.
        PIN_LOOP_SCALES: tl.constexpr = (SCALE_PRESHUFFLE and SCALE_BLOCK == 32 and (BLOCK_K == 128 or BLOCK_K == 256)
                                         and STORE_INSTR_M == 16 and tlx.num_warps() == 4 and BLOCK_N == 256
                                         and (BLOCK_M == 128 or BLOCK_M == 256))
        if PIN_LOOP_SCALES:
            if BLOCK_M == 128:
                # A 64x128 subtile distributes all four waves along N.
                SCALE_A_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 4), (4, 2)), stride=((4, 0), (1, 128)))
                SCALE_B_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 4), (4, )), stride=((4, 128), (1, )))
            else:
                SCALE_A_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, 2)),
                                                               stride=((4, 0, 128), (1, 256)))
                SCALE_B_LOOP_LAYOUT: tl.constexpr = tlx.layout(shape=((32, 2, 2), (4, 2)),
                                                               stride=((4, 128, 0), (1, 256)))
            scale_a00 = tlx.require_layout(scale_a00, SCALE_A_LOOP_LAYOUT)
            scale_b00 = tlx.require_layout(scale_b00, SCALE_B_LOOP_LAYOUT)
        c00 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c01 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c10 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        c11 = tl.zeros((SUBTILE_M, SUBTILE_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            if NUM_SUBTILES_K == 1:
                # Make the short stage's refill predicate available before its math.
                pred_load = i + 1 - epilogue_lb
                pred_load = (pred_load >> 31) & 1
            c00 = tlx.dot_scaled(a00, scale_a00, DTYPE_A, b00, scale_b00, DTYPE_B, c00, tiles_per_warp=[2, 2])
            if TDM_SPLIT:
                b01, scale_b01 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 0, 1, BLOCK_N,
                                                              BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                              SCALE_PRESHUFFLE)
            else:
                b01, scale_b01 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 0, 1, BLOCK_N, BLOCK_K,
                                                        DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                        TRANSPOSE_B, SCALE_PRESHUFFLE)
            pred_prefetch = i - epilogue_lb
            pred_prefetch = (pred_prefetch >> 31) & 1
            if TDM_SPLIT:
                _mxgemm_issue_split_l2_prefetches(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc,
                                                  load_idx, always, L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A,
                                                  BLOCK_K_PACKED_B, BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B,
                                                  SCALE_PRESHUFFLE, WITH_A_SCALE)
            else:
                _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always,
                                            L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                            BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)
            c01 = tlx.dot_scaled(a00, scale_a00, DTYPE_A, b01, scale_b01, DTYPE_B, c01, tiles_per_warp=[2, 2])
            if TDM_SPLIT:
                a10, scale_a10 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 1, 0, BLOCK_M,
                                                              BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                              WITH_A_SCALE)
            else:
                a10, scale_a10 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 1, 0, BLOCK_M, BLOCK_K,
                                                        DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                        SCALE_PRESHUFFLE, WITH_A_SCALE)
            c10 = tlx.dot_scaled(a10, scale_a10, DTYPE_A, b00, scale_b00, DTYPE_B, c10, tiles_per_warp=[2, 2])
            if NUM_SUBTILES_K == 2:
                if TDM_SPLIT:
                    b10, scale_b10 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 1, 0, BLOCK_N,
                                                                  BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                                  SCALE_PRESHUFFLE)
                else:
                    b10, scale_b10 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 1, 0, BLOCK_N, BLOCK_K,
                                                            DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                            SCALE_PRESHUFFLE)
            c11 = tlx.dot_scaled(a10, scale_a10, DTYPE_A, b01, scale_b01, DTYPE_B, c11, tiles_per_warp=[2, 2])
            if NUM_SUBTILES_K == 2:
                if TDM_SPLIT:
                    a01, scale_a01 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 1, BLOCK_M,
                                                                  BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                                  WITH_A_SCALE)
                else:
                    a01, scale_a01 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 1, BLOCK_M, BLOCK_K,
                                                            DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                            WITH_A_SCALE)
                c00 = tlx.dot_scaled(a01, scale_a01, DTYPE_A, b10, scale_b10, DTYPE_B, c00, tiles_per_warp=[2, 2])
                if TDM_SPLIT:
                    b11, scale_b11 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 1, 1, BLOCK_N,
                                                                  BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                                  SCALE_PRESHUFFLE)
                else:
                    b11, scale_b11 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 1, 1, BLOCK_N, BLOCK_K,
                                                            DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                            SCALE_PRESHUFFLE)
                c01 = tlx.dot_scaled(a01, scale_a01, DTYPE_A, b11, scale_b11, DTYPE_B, c01, tiles_per_warp=[2, 2])
                if TDM_SPLIT:
                    a11, scale_a11 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 1, 1, BLOCK_M,
                                                                  BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                                  BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                                  NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                                  WITH_A_SCALE)
                else:
                    a11, scale_a11 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 1, 1, BLOCK_M, BLOCK_K,
                                                            DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                            BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                            NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                            WITH_A_SCALE)
            wmma_idx += 1
            if NUM_SUBTILES_K == 2:
                c10 = tlx.dot_scaled(a11, scale_a11, DTYPE_A, b10, scale_b10, DTYPE_B, c10, tiles_per_warp=[2, 2])
                c11 = tlx.dot_scaled(a11, scale_a11, DTYPE_A, b11, scale_b11, DTYPE_B, c11, tiles_per_warp=[2, 2])
                pred_load = i + 1 - epilogue_lb
                pred_load = (pred_load >> 31) & 1
            if TDM_SPLIT:
                load_idx = _mxgemm_issue_split_loads(a0_desc, a1_desc, b0_desc, b1_desc, a_scale_desc, b_scale_desc,
                                                     a0_buf, a1_buf, b0_buf, b1_buf, a_scale_buf, b_scale_buf, load_idx,
                                                     pred_load, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                                     BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B,
                                                     SCALE_PRESHUFFLE, WITH_A_SCALE, TDM_FUSION)
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * SPLIT_LOADS_IN_BATCH)
                a00, scale_a00 = _mxgemm_load_a_operand_split(a0_buf, a1_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M,
                                                              BLOCK_K, DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_M_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                              WITH_A_SCALE)
                b00, scale_b00 = _mxgemm_load_b_operand_split(b0_buf, b1_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N,
                                                              BLOCK_K, DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE,
                                                              BLOCK_N_PRESHUFFLED, SCALE_KWIDTH, NUM_BUFFERS,
                                                              NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                              SCALE_PRESHUFFLE)
            else:
                load_idx = _mxgemm_issue_loads(a_desc, b_desc, a_scale_desc, b_scale_desc, a_buf, b_buf, a_scale_buf,
                                               b_scale_buf, load_idx, pred_load, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                               BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                               WITH_A_SCALE, TDM_FUSION)
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
                a00, scale_a00 = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, 0, BLOCK_M, BLOCK_K,
                                                        DIV_FACTOR_A, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K,
                                                        SCALE_PRESHUFFLE, WITH_A_SCALE)
                b00, scale_b00 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, 0, 0, BLOCK_N, BLOCK_K,
                                                        DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                        SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                        TRANSPOSE_B, SCALE_PRESHUFFLE)
            if PIN_LOOP_SCALES:
                scale_a00 = tlx.require_layout(scale_a00, SCALE_A_LOOP_LAYOUT)
                scale_b00 = tlx.require_layout(scale_b00, SCALE_B_LOOP_LAYOUT)
        # Stage each quadrant in its native WMMA register order. Joining them
        # puts the quadrant selectors first and defeats b128 LDS vectorization.
        c_buf = tlx.local_alloc((BLOCK_M, BLOCK_N), tlx.dtype_of(c_ptr), 1)
        c_view = tlx.local_view(c_buf, 0)
        tlx.local_store(tlx.local_slice(c_view, [0, 0], [SUBTILE_M, SUBTILE_N]), c00.to(tlx.dtype_of(c_ptr)))
        tlx.local_store(tlx.local_slice(c_view, [0, SUBTILE_N], [SUBTILE_M, SUBTILE_N]), c01.to(tlx.dtype_of(c_ptr)))
        tlx.local_store(tlx.local_slice(c_view, [SUBTILE_M, 0], [SUBTILE_M, SUBTILE_N]), c10.to(tlx.dtype_of(c_ptr)))
        tlx.local_store(tlx.local_slice(c_view, [SUBTILE_M, SUBTILE_N], [SUBTILE_M, SUBTILE_N]),
                        c11.to(tlx.dtype_of(c_ptr)))
        tlx.async_amd_descriptor_store(c_desc, c_view, [c_off_m, c_off_n], clamp_bounds=True)
        tlx.async_amd_descriptor_wait(0)
    elif SCHEDULE == "sliceNK":
        SUBTILE_N: tl.constexpr = BLOCK_N // 2
        c0 = tl.zeros((BLOCK_M, SUBTILE_N), dtype=tl.float32)
        c1 = tl.zeros((BLOCK_M, SUBTILE_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = _mxgemm_issue_loads(a_desc, b_desc, a_scale_desc, b_scale_desc, a_buf, b_buf, a_scale_buf,
                                           b_scale_buf, load_idx, pred, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                           BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                           WITH_A_SCALE, TDM_FUSION)
            _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always,
                                        L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                        BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
            for kt in tl.static_range(2):
                a, scale_a = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, kt, BLOCK_M, BLOCK_K, DIV_FACTOR_A,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                    WITH_A_SCALE)
                b0, scale_b0 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, kt, 0, BLOCK_N, BLOCK_K,
                                                      DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                      SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                      TRANSPOSE_B, SCALE_PRESHUFFLE)
                b1, scale_b1 = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, kt, 1, BLOCK_N, BLOCK_K,
                                                      DIV_FACTOR_B, SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED,
                                                      SCALE_KWIDTH, NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K,
                                                      TRANSPOSE_B, SCALE_PRESHUFFLE)
                c0 = tlx.dot_scaled(a, scale_a, DTYPE_A, b0, scale_b0, DTYPE_B, c0, tiles_per_warp=[2, 2])
                c1 = tlx.dot_scaled(a, scale_a, DTYPE_A, b1, scale_b1, DTYPE_B, c1, tiles_per_warp=[2, 2])
            wmma_idx += 1
        acc = tl.join(c0, c1).permute(0, 2, 1).reshape((BLOCK_M, BLOCK_N))
        acc = tlx.require_amd_wmma_layout(acc, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                          instr_shape=STORE_INSTR_SHAPE)
        c_offsets_store = tlx.require_amd_wmma_layout(c_offsets, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                      instr_shape=STORE_INSTR_SHAPE)
        c_mask_store = tlx.require_amd_wmma_layout(c_mask, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                   instr_shape=STORE_INSTR_SHAPE)
        tlx.buffer_store(acc, c_ptr, c_offsets_store, mask=c_mask_store)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for i in tl.range(0, K_ITERS):
            pred = i - epilogue_lb
            pred = (pred >> 31) & 1
            load_idx = _mxgemm_issue_loads(a_desc, b_desc, a_scale_desc, b_scale_desc, a_buf, b_buf, a_scale_buf,
                                           b_scale_buf, load_idx, pred, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                           BLOCK_K_SCALE_PRESHUFFLED, NUM_BUFFERS, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                           WITH_A_SCALE, TDM_FUSION)
            _mxgemm_issue_l2_prefetches(a_desc, b_desc, a_scale_desc, b_scale_desc, load_idx, always,
                                        L2_PREFETCH_DISTANCE, BLOCK_K_PACKED_A, BLOCK_K_PACKED_B,
                                        BLOCK_K_SCALE_PRESHUFFLED, TRANSPOSE_B, SCALE_PRESHUFFLE, WITH_A_SCALE)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * NUM_LOADS_IN_BATCH)
            for kt in tl.static_range(NUM_SUBTILES_K):
                a, scale_a = _mxgemm_load_a_operand(a_buf, a_scale_buf, wmma_idx, 0, kt, BLOCK_M, BLOCK_K, DIV_FACTOR_A,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_M_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_M, NUM_SUBTILES_K, SCALE_PRESHUFFLE,
                                                    WITH_A_SCALE)
                b, scale_b = _mxgemm_load_b_operand(b_buf, b_scale_buf, wmma_idx, kt, 0, BLOCK_N, BLOCK_K, DIV_FACTOR_B,
                                                    SCALE_BLOCK, BLOCK_K_SCALE, BLOCK_N_PRESHUFFLED, SCALE_KWIDTH,
                                                    NUM_BUFFERS, NUM_SUBTILES_N, NUM_SUBTILES_K, TRANSPOSE_B,
                                                    SCALE_PRESHUFFLE)
                acc = tlx.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, acc, tiles_per_warp=[2, 2])
            wmma_idx += 1
        acc = tlx.require_amd_wmma_layout(acc, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                          instr_shape=STORE_INSTR_SHAPE)
        c_offsets_store = tlx.require_amd_wmma_layout(c_offsets, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                      instr_shape=STORE_INSTR_SHAPE)
        c_mask_store = tlx.require_amd_wmma_layout(c_mask, warp_bases=STORE_WARP_BASES, reg_bases=STORE_REG_BASES,
                                                   instr_shape=STORE_INSTR_SHAPE)
        tlx.buffer_store(acc, c_ptr, c_offsets_store, mask=c_mask_store)


DTYPE_TO_TRITON = {
    "float8_e5m2": "e5m2",
    "float8_e4m3": "e4m3",
    "float4": "e2m1",
}

_XCD_REMAP_MODES = {"none": 0, "balanced": 1, "chunked": 2}


@triton.jit
def _mxgemm_remap_program_id(pid, NUM_PROGRAMS: tl.constexpr, XCD_REMAP_MODE: tl.constexpr, NUM_XCDS: tl.constexpr,
                             XCD_CHUNK: tl.constexpr):
    tl.static_assert(NUM_XCDS > 0 and XCD_CHUNK > 0)
    if XCD_REMAP_MODE == 0 or NUM_XCDS == 1:
        return pid
    elif XCD_REMAP_MODE == 1:
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        per_xcd = NUM_PROGRAMS // NUM_XCDS
        extra = NUM_PROGRAMS % NUM_XCDS
        return xcd * per_xcd + tl.minimum(xcd, extra) + local_pid
    else:
        tl.static_assert(XCD_REMAP_MODE == 2)
        aligned = NUM_PROGRAMS // (NUM_XCDS * XCD_CHUNK) * (NUM_XCDS * XCD_CHUNK)
        if pid >= aligned:
            return pid
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        return (local_pid // XCD_CHUNK) * NUM_XCDS * XCD_CHUNK + xcd * XCD_CHUNK + local_pid % XCD_CHUNK


@triton.jit
def _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    group = tile // (GROUP_M * num_n)
    first_m = group * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    local = tile % (GROUP_M * num_n)
    return (first_m + local % group_m) * BM, (local // group_m) * BN


@triton.jit
def _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n, k, slot,
                            BK: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr, BUFFERS: tl.constexpr,
                            WITH_A_SCALE: tl.constexpr, FUSION: tl.constexpr, CLUSTER_SIZE: tl.constexpr,
                            CLUSTER_MULTICAST: tl.constexpr, GROUP_SIZE_M: tl.constexpr, XCD_REMAP_MODE: tl.constexpr,
                            CLUSTER_BARRIER_INTERVAL: tl.constexpr, REGISTER_PIPELINE: tl.constexpr = False):
    # Keep the K coordinate independent of the LDS ring slot. Adding (k - slot)
    # here and slot in the loader hides the invariant per-wave global base
    # from LLVM and repeats descriptor address work inside the K loop.
    ad = tlx.update_tensor_descriptor(a_desc, add_offsets=[off_m, 0], clamp_bounds=False)
    bd = tlx.update_tensor_descriptor(b_desc, add_offsets=[off_n, 0], clamp_bounds=False)
    asd = tlx.update_tensor_descriptor(as_desc, add_offsets=[off_m // 128, 0], clamp_bounds=False)
    bsd = tlx.update_tensor_descriptor(bs_desc, add_offsets=[off_n // 128, 0], clamp_bounds=False)
    # Barriers align requests for multicast; late requests receive separate loads.
    if CLUSTER_SIZE > 1:
        if CLUSTER_BARRIER_INTERVAL == 1:
            if REGISTER_PIPELINE:
                _mxgemm_multicast_rendezvous()
            else:
                tlx.cluster_barrier()
        elif CLUSTER_BARRIER_INTERVAL > 1:
            if k % CLUSTER_BARRIER_INTERVAL == 0:
                if REGISTER_PIPELINE:
                    _mxgemm_multicast_rendezvous()
                else:
                    tlx.cluster_barrier()
    _mxgemm_issue_loads(ad, bd, asd, bsd, a_buf, b_buf, as_buf, bs_buf, k, None, BK // DA, BK // DB, BK // 32 * 128,
                        BUFFERS, True, True, WITH_A_SCALE, FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M,
                        XCD_REMAP_MODE, load_slot=slot, SCALES_FIRST=REGISTER_PIPELINE)


@triton.jit
def _mxgemm_persistent_a(a_buf, as_buf, slot, m: tl.constexpr, k: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
                         DA: tl.constexpr, BUFFERS: tl.constexpr, WITH_A_SCALE: tl.constexpr,
                         PREFETCH_SCALES: tl.constexpr = False):
    tl.assume(slot >= 0)
    tl.assume(slot < BUFFERS)
    a, scale = _mxgemm_load_a_operand(
        a_buf, as_buf, slot, m, k, BM, BK, DA, 32, BK // 32, BM // 128, 4, BUFFERS, 2, BK // 128, True, WITH_A_SCALE,
        SCALES_FIRST=(PREFETCH_SCALES or ((BUFFERS == 4 or tlx.num_warps() == 8) and BK == 128)) and DA == 1
        and WITH_A_SCALE)
    # Preserve the scale distribution across loop edges. A generic blocked
    # layout here adds an LDS exchange and barriers before the first dot in
    # each K iteration.
    if tlx.num_warps() == 8:
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 4, 2), (4, BM // 128)),
                                                     stride=((4, 0, 128), (1, 256))))
    else:
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, BM // 128)),
                                                     stride=((4, 0, 128), (1, 256))))
    return a, scale


@triton.jit
def _mxgemm_persistent_b(b_buf, bs_buf, slot, k: tl.constexpr, n: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                         DB: tl.constexpr, BUFFERS: tl.constexpr, PREFETCH_SCALES: tl.constexpr = False):
    tl.assume(slot >= 0)
    tl.assume(slot < BUFFERS)
    b, scale = _mxgemm_load_b_operand(
        b_buf, bs_buf, slot, k, n, BN, BK, DB, 32, BK // 32, BN // 128, 4, BUFFERS, 2, BK // 128, True, True,
        SCALES_FIRST=(PREFETCH_SCALES or ((BUFFERS == 4 or tlx.num_warps() == 8) and BK == 128)) and DB == 1)
    if tlx.num_warps() == 8:
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 4, 2), (4, )), stride=((4, 128, 0), (1, ))))
    else:
        scale = tlx.require_layout(scale, tlx.layout(shape=((32, 2, 2), (4, BN // 128)),
                                                     stride=((4, 128, 0), (1, 256))))
    return b, scale


@triton.jit
def _mxgemm_persistent_compute(c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot,
                               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr,
                               DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr,
                               WITH_A_SCALE: tl.constexpr):
    c00 = tlx.dot_scaled(a00, sa00, DTYPE_A, b00, sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
    b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS)
    c01 = tlx.dot_scaled(a00, sa00, DTYPE_A, b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
    a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS, WITH_A_SCALE)
    c10 = tlx.dot_scaled(a10, sa10, DTYPE_A, b00, sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
    if BK == 128:
        # All operands for this stage are now in registers. The caller can
        # refill its LDS slot while the remaining matrix instructions execute.
        c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
    else:
        b10, sb10 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 1, 0, BN, BK, DB, BUFFERS)
        # Limit overlap between K halves for full A8W8 tiles to avoid spilling.
        # Preload the next A operand before the current half's last dot so the
        # scheduling boundary does not serialize its LDS load.
        LIMIT_FP8_LIFETIME: tl.constexpr = DA == 1 and DB == 1 and BM == 256 and BN == 256
        if LIMIT_FP8_LIFETIME:
            a01, sa01 = _mxgemm_persistent_a(a_buf, as_buf, slot, 0, 1, BM, BK, DA, BUFFERS, WITH_A_SCALE)
        c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
        if LIMIT_FP8_LIFETIME:
            tlx.amd_sched_barrier()
        else:
            a01, sa01 = _mxgemm_persistent_a(a_buf, as_buf, slot, 0, 1, BM, BK, DA, BUFFERS, WITH_A_SCALE)
        c00 = tlx.dot_scaled(a01, sa01, DTYPE_A, b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
        b11, sb11 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 1, 1, BN, BK, DB, BUFFERS)
        c01 = tlx.dot_scaled(a01, sa01, DTYPE_A, b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
        a11, sa11 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 1, BM, BK, DA, BUFFERS, WITH_A_SCALE)
        c10 = tlx.dot_scaled(a11, sa11, DTYPE_A, b10, sb10, DTYPE_B, c10, tiles_per_warp=[2, 2])
        c11 = tlx.dot_scaled(a11, sa11, DTYPE_A, b11, sb11, DTYPE_B, c11, tiles_per_warp=[2, 2])
    return c00, c01, c10, c11


@triton.jit
def _mxgemm_schedule_prefetch(sync_id: tl.constexpr, OPERAND_A: tl.constexpr = False):
    # Four waves have sixteen WMMAs per quadrant and sixteen operand reads.
    # Eight waves have eight WMMAs, sixteen A reads, and eight B reads.
    # Issue the scale early, leaving matrix work after the final payload read.
    GROUPS: tl.constexpr = 4 if tlx.num_warps() == 8 else 8
    READS: tl.constexpr = 4 if tlx.num_warps() == 8 and OPERAND_A else 2
    TAIL: tl.constexpr = 3 if tlx.num_warps() == 8 else 7
    for _ in tl.static_range(GROUPS):
        tlx.amd_sched_group_barrier(0x8, 1, sync_id)
        tlx.amd_sched_group_barrier(0x100, READS, sync_id)
    tlx.amd_sched_group_barrier(0x8, 1, sync_id)
    tlx.amd_sched_group_barrier(0x100, 1, sync_id)
    tlx.amd_sched_group_barrier(0x8, TAIL, sync_id)


@triton.jit
def _mxgemm_persistent_compute_top(c00, c01, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BM: tl.constexpr,
                                   BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr, DB: tl.constexpr,
                                   DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr,
                                   WITH_A_SCALE: tl.constexpr, GROUPED_PREFETCH: tl.constexpr = False):
    tl.static_assert(BK == 128)
    b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS)
    c00 = tlx.dot_scaled(a00, sa00, DTYPE_A, b00, sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
    if GROUPED_PREFETCH:
        _mxgemm_schedule_prefetch(101)
    tlx.amd_sched_barrier()
    a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS, WITH_A_SCALE)
    c01 = tlx.dot_scaled(a00, sa00, DTYPE_A, b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
    if GROUPED_PREFETCH:
        _mxgemm_schedule_prefetch(102, True)
    tlx.amd_sched_barrier()
    # Retain the bottom-half operands in registers so the caller can refill
    # this LDS slot and read the next stage before finishing C10/C11.
    return c00, c01, a10, sa10, b01, sb01


@triton.jit
def _mxgemm_persistent_compute_256(c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot,
                                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr,
                                   DB: tl.constexpr, DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr,
                                   BUFFERS: tl.constexpr, WITH_A_SCALE: tl.constexpr):
    tl.static_assert(BK == 256)
    c00 = tlx.dot_scaled(a00, sa00, DTYPE_A, b00, sb00, DTYPE_B, c00, tiles_per_warp=[2, 2])
    b01, sb01 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 0, 1, BN, BK, DB, BUFFERS)
    c01 = tlx.dot_scaled(a00, sa00, DTYPE_A, b01, sb01, DTYPE_B, c01, tiles_per_warp=[2, 2])
    a10, sa10 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 0, BM, BK, DA, BUFFERS, WITH_A_SCALE)
    c10 = tlx.dot_scaled(a10, sa10, DTYPE_A, b00, sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
    b10, sb10 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 1, 0, BN, BK, DB, BUFFERS)
    a01, sa01 = _mxgemm_persistent_a(a_buf, as_buf, slot, 0, 1, BM, BK, DA, BUFFERS, WITH_A_SCALE)
    if DB != 2 or tlx.num_warps() == 8:
        c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
    # Bound operand lifetimes before gathering the remaining second half.
    # Four-wave A8W4 keeps the first-half operands live across the refill.
    # Eight waves finish them here to fit two waves in each SIMD's registers.
    tlx.amd_sched_barrier()
    if DB != 2 or tlx.num_warps() == 8:
        c00 = tlx.dot_scaled(a01, sa01, DTYPE_A, b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
    b11, sb11 = _mxgemm_persistent_b(b_buf, bs_buf, slot, 1, 1, BN, BK, DB, BUFFERS)
    if DB != 2 or tlx.num_warps() == 8:
        c01 = tlx.dot_scaled(a01, sa01, DTYPE_A, b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
    a11, sa11 = _mxgemm_persistent_a(a_buf, as_buf, slot, 1, 1, BM, BK, DA, BUFFERS, WITH_A_SCALE)
    tlx.amd_sched_barrier()
    # All operands are in registers. Refill this slot and read the next stage
    # before finishing the deferred first- and second-half quadrants.
    return c00, c01, c10, c11, a01, sa01, a11, sa11, b10, sb10, b11, sb11, a10, sa10, b01, sb01


@triton.jit
def _mxgemm_persistent_store(c_ptr, acc, off_m, off_n, stride_cm, BM: tl.constexpr, BN: tl.constexpr,
                             INSTR_M: tl.constexpr, c_buf=None, C_ROWS: tl.constexpr = 64, C_SLOTS: tl.constexpr = 2,
                             QUADRANT: tl.constexpr = 0, C_COLS: tl.constexpr = 128):
    # Pin values so output extraction stays in registers. Direct stores also
    # pin their offsets below to avoid an implicit LDS transpose.
    WARP_BASES: tl.constexpr = ((0, 2), (0, 4), (2, 0)) if tlx.num_warps() == 8 else ((0, 2), (2, 0))
    acc = tlx.require_amd_wmma_layout(acc, warp_bases=WARP_BASES, reg_bases=((0, 1), (1, 0)),
                                      instr_shape=(INSTR_M, 16, 128))
    base = c_ptr + off_m.to(tl.int64) * stride_cm + off_n
    if c_buf is not None:
        tl.static_assert(BM == 128 and BN == 128 and INSTR_M == 16)
        desc = tl.make_tensor_descriptor(base, [BM, BN], [stride_cm, tl.constexpr(1)], [C_ROWS, C_COLS])
        if C_ROWS == 128:
            tlx.async_amd_descriptor_wait(C_SLOTS - 1)
            view = c_buf[QUADRANT % C_SLOTS]
            tlx.local_store(view, acc)
            tlx.async_amd_descriptor_store(desc, view, [0, 0], clamp_bounds=False)
        else:
            tl.static_assert(C_ROWS == 64 and C_SLOTS == 2 and (C_COLS == 64 or C_COLS == 128))
            for part in tl.static_range(2 * (BN // C_COLS)):
                # Each quadrant writes the two slots in order, so wait(1) retires
                # the previous use even across quadrant/tile boundaries.
                tlx.async_amd_descriptor_wait(1)
                lo, hi = tl.split(tl.reshape(acc, (2, 64, BN)).permute(1, 2, 0))
                if C_COLS == 128:
                    chunk = lo if part == 0 else hi
                    row, col = part * 64, 0
                else:
                    # N's 64-column split stays within each wave's registers.
                    # A 32-row split would require exchanging values across waves.
                    half = lo if part < 2 else hi
                    left, right = tl.split(tl.reshape(half, (64, 2, 64)).permute(0, 2, 1))
                    chunk = left if part % 2 == 0 else right
                    row, col = (part // 2) * 64, (part % 2) * 64
                view = c_buf[part % C_SLOTS]
                tlx.local_store(view, chunk)
                tlx.async_amd_descriptor_store(desc, view, [row, col], clamp_bounds=False)
    else:
        offsets = tl.arange(0, BM)[:, None] * stride_cm + tl.arange(0, BN)[None, :]
        offsets = tlx.require_amd_wmma_layout(offsets, warp_bases=WARP_BASES, reg_bases=((0, 1), (1, 0)),
                                              instr_shape=(INSTR_M, 16, 128))
        tlx.buffer_store(acc, base, offsets)


@triton.jit
def _mxgemm_schedule_register_pair(sync_id: tl.constexpr):
    # Two FP8 operands: 32 b128 payload reads and two paired scale reads.
    tlx.amd_sched_group_barrier(0x100, 2, sync_id)
    for _ in tl.static_range(16):
        tlx.amd_sched_group_barrier(0x8, 1, sync_id)
        tlx.amd_sched_group_barrier(0x100, 2, sync_id)


@triton.jit
def _mxgemm_register_next_half(c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1, a_buf, b_buf, as_buf, bs_buf,
                               next_slot, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr,
                               DB: tl.constexpr, DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr,
                               WITH_A_SCALE: tl.constexpr, HALF: tl.constexpr):
    # Keep at most six payload fragments live while consuming the two N panels.
    na0, nsa0 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, HALF, BM, BK, DA, BUFFERS, WITH_A_SCALE,
                                     PREFETCH_SCALES=True)
    c00 = tlx.dot_scaled(a0, sa0, DTYPE_A, b0, sb0, DTYPE_B, c00, tiles_per_warp=[2, 2])
    _mxgemm_schedule_prefetch(301)
    tlx.amd_sched_barrier()
    na1, nsa1 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 1, HALF, BM, BK, DA, BUFFERS, WITH_A_SCALE,
                                     PREFETCH_SCALES=True)
    c10 = tlx.dot_scaled(a1, sa1, DTYPE_A, b0, sb0, DTYPE_B, c10, tiles_per_warp=[2, 2])
    _mxgemm_schedule_prefetch(302)
    tlx.amd_sched_barrier()
    nb0, nsb0 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, HALF, 0, BN, BK, DB, BUFFERS, PREFETCH_SCALES=True)
    c01 = tlx.dot_scaled(a0, sa0, DTYPE_A, b1, sb1, DTYPE_B, c01, tiles_per_warp=[2, 2])
    _mxgemm_schedule_prefetch(303)
    tlx.amd_sched_barrier()
    nb1, nsb1 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, HALF, 1, BN, BK, DB, BUFFERS, PREFETCH_SCALES=True)
    c11 = tlx.dot_scaled(a1, sa1, DTYPE_A, b1, sb1, DTYPE_B, c11, tiles_per_warp=[2, 2])
    _mxgemm_schedule_prefetch(304)
    tlx.amd_sched_barrier()
    return c00, c01, c10, c11, na0, nsa0, na1, nsa1, nb0, nsb0, nb1, nsb1


@triton.jit
def _mxgemm_register_next_stage(c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1, a_buf, b_buf, as_buf, bs_buf,
                                next_slot, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DA: tl.constexpr,
                                DB: tl.constexpr, DTYPE_A: tl.constexpr, DTYPE_B: tl.constexpr, BUFFERS: tl.constexpr,
                                WITH_A_SCALE: tl.constexpr, HALF: tl.constexpr):
    na0, nsa0 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, HALF, BM, BK, DA, BUFFERS, WITH_A_SCALE,
                                     PREFETCH_SCALES=True)
    c10 = tlx.dot_scaled(a1, sa1, DTYPE_A, b0, sb0, DTYPE_B, c10, tiles_per_warp=[2, 2])
    _mxgemm_schedule_prefetch(401)
    tlx.amd_sched_barrier()
    # C00 needs B0 before C10 needs A1. Reading A1 first makes the dscnt
    # wait for B0 also cover requests whose consumers could run later.
    nb0, nsb0 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, HALF, 0, BN, BK, DB, BUFFERS, PREFETCH_SCALES=True)
    tlx.amd_sched_barrier(0xE)
    na1, nsa1 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 1, HALF, BM, BK, DA, BUFFERS, WITH_A_SCALE,
                                     PREFETCH_SCALES=True)
    c01 = tlx.dot_scaled(a0, sa0, DTYPE_A, b1, sb1, DTYPE_B, c01, tiles_per_warp=[2, 2])
    _mxgemm_schedule_register_pair(402)
    tlx.amd_sched_barrier()
    nb1, nsb1 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, HALF, 1, BN, BK, DB, BUFFERS, PREFETCH_SCALES=True)
    c11 = tlx.dot_scaled(a1, sa1, DTYPE_A, b1, sb1, DTYPE_B, c11, tiles_per_warp=[2, 2])
    _mxgemm_schedule_prefetch(403)
    tlx.amd_sched_barrier()
    return c00, c01, c10, c11, na0, nsa0, na1, nsa1, nb0, nsb0, nb1, nsb1


@triton.constexpr_function
def _mxgemm_register_output_layout():
    # Preserve the byte layout: 256 FP8 values + 16 padding bytes become
    # 64 FP32 values + four padding elements, within the same input slot.
    return tlx.padded_shared_layout_encoding.with_identity_for([[64, 4]], [256, 64], [1, 0])


@triton.jit
def _mxgemm_store_register_panels(c_ptr, c00, c01, c10, c11, off_m, off_n, stride_cm, c_a, c_b):
    base = c_ptr + off_m.to(tl.int64) * stride_cm + off_n
    desc = tl.make_tensor_descriptor(base, [256, 256], [stride_cm, tl.constexpr(1)], [256, 64])
    # Joining top and bottom quadrants only changes register indexing.
    # The free A/B stages form two output slots; wait(1) protects reuse.
    for part in tl.static_range(4):
        top = c00 if part < 2 else c01
        bot = c10 if part < 2 else c11
        left, right = tl.split(tl.reshape(top, (128, 2, 64)).permute(0, 2, 1))
        top_half = left if part % 2 == 0 else right
        left, right = tl.split(tl.reshape(bot, (128, 2, 64)).permute(0, 2, 1))
        bot_half = left if part % 2 == 0 else right
        panel = tl.reshape(tl.join(top_half, bot_half).permute(2, 0, 1), (256, 64))
        panel = tlx.require_amd_wmma_layout(panel, warp_bases=((0, 2), (2, 0)), reg_bases=((0, 1), (1, 0)),
                                            instr_shape=(16, 16, 128))
        view = c_a if part % 2 == 0 else c_b
        tlx.async_amd_descriptor_wait(1)
        tlx.local_store(view, panel)
        tlx.async_amd_descriptor_store(desc, view, [0, part * 64], clamp_bounds=False)


@triton.jit
def mxgemm_tdm_persistent_kernel(
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
    OUTPUT_STAGING: tl.constexpr = False,
    SCHED_MODE_2: tl.constexpr = False,
    XCD_REMAP_MODE: tl.constexpr = 0,
    NUM_XCDS: tl.constexpr = 8,
    XCD_CHUNK: tl.constexpr = 2,
    CLUSTER_SIZE: tl.constexpr = 1,
    CLUSTER_MULTICAST: tl.constexpr = True,
    CLUSTER_BARRIER_INTERVAL: tl.constexpr = 1,
    REGISTER_PIPELINE: tl.constexpr = False,
):
    """Full-tile, transposed-B sliceMNK GEMM with a persistent TDM ring."""
    if SCHED_MODE_2:
        # Allow WMMA queuing with one wave per SIMD; preserve mode bits 0-1.
        tlx.amd_set_wave_sched_mode(1, offset=2, width=1)
    tl.static_assert(BLOCK_M % 128 == 0 and BLOCK_N % 128 == 0 and (BLOCK_K == 128 or BLOCK_K == 256))
    tl.static_assert(NUM_BUFFERS >= 2 and NUM_PROGRAMS > 0 and GROUP_SIZE_M > 0)
    tl.static_assert(CLUSTER_SIZE == 1 or CLUSTER_SIZE == 2 or CLUSTER_SIZE == 4)
    tl.static_assert(CLUSTER_BARRIER_INTERVAL >= 0)
    if CLUSTER_SIZE > 1:
        tl.static_assert(XCD_REMAP_MODE == 0 or (XCD_REMAP_MODE == 2 and NUM_XCDS == 8 and XCD_CHUNK == 2))
        tl.static_assert(GROUP_SIZE_M == 4 or GROUP_SIZE_M == 8)
        tl.static_assert(NUM_PROGRAMS % 16 == 0)
    DA: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DB: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    tl.static_assert(tlx.num_warps() == 4 or tlx.num_warps() == 8)
    if tlx.num_warps() == 8:
        tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and DA == 1,
                         "eight-wave MXFP requires 256x256 M/N tiles and FP8 A")
    if REGISTER_PIPELINE:
        tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 256 and NUM_BUFFERS == 2)
        tl.static_assert(DA == 1 and DB == 1 and WITH_A_SCALE and tlx.num_warps() == 4)
        tl.static_assert(OUTPUT_STAGING and TDM_FUSION == "partial")
    OUTPUT_REUSE: tl.constexpr = OUTPUT_STAGING and BLOCK_K == 256 and not REGISTER_PIPELINE
    if OUTPUT_STAGING:
        tl.static_assert(BLOCK_M == 256 and BLOCK_N == 256 and DA == 1)
        if OUTPUT_REUSE:
            tl.static_assert(not CROSS_TILE_PREFETCH)
            tl.static_assert(NUM_BUFFERS == 2 or (NUM_BUFFERS == 3 and DB == 2))
        else:
            tl.static_assert(NUM_BUFFERS <= 4)
            tl.static_assert(tlx.num_warps() == 4 or NUM_BUFFERS <= 3,
                             "eight-wave output staging supports at most three input buffers")
    # Keep independent bottom-half work after the refill and next operand loads.
    # In particular, periodic cluster barriers prevent LLVM from moving earlier
    # WMMAs across that boundary to cover the load latency.
    DEFER_BOTTOM: tl.constexpr = OUTPUT_STAGING and BLOCK_K == 128
    GROUPED_PREFETCH: tl.constexpr = DEFER_BOTTOM and (NUM_BUFFERS == 4
                                                       or tlx.num_warps() == 8) and DA == 1 and DB == 1 and WITH_A_SCALE
    PREFETCH_B_AFTER_C10: tl.constexpr = GROUPED_PREFETCH or (DEFER_BOTTOM and tlx.num_warps() == 8)
    if TDM_FUSION == "4way":
        tl.static_assert(WITH_A_SCALE)
        LOADS: tl.constexpr = 1
    elif TDM_FUSION == "partial" or TDM_FUSION == "2way":
        tl.static_assert(TDM_FUSION != "partial" or WITH_A_SCALE)
        LOADS: tl.constexpr = 2
    else:
        tl.static_assert(TDM_FUSION == "none")
        LOADS: tl.constexpr = 4 if WITH_A_SCALE else 3

    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K // DA], [stride_am, tl.constexpr(1)], [BLOCK_M, BLOCK_K // DA])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K // DB], [stride_bn, tl.constexpr(1)], [BLOCK_N, BLOCK_K // DB])
    as_desc = tl.make_tensor_descriptor(a_scale, [M // 128, K // 32 * 128], [stride_as, tl.constexpr(1)],
                                        [BLOCK_M // 128, BLOCK_K // 32 * 128])
    bs_desc = tl.make_tensor_descriptor(b_scale, [N // 128, K // 32 * 128], [stride_bs, tl.constexpr(1)],
                                        [BLOCK_N // 128, BLOCK_K // 32 * 128])
    a_buf = tlx.local_alloc((BLOCK_M, BLOCK_K // DA), tlx.dtype_of(a_ptr), NUM_BUFFERS,
                            layout=_operand_shared_layout([BLOCK_M, BLOCK_K // DA]))
    b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K // DB), tlx.dtype_of(b_ptr), NUM_BUFFERS,
                            layout=_operand_shared_layout([BLOCK_N, BLOCK_K // DB]))
    as_buf = tlx.local_alloc((BLOCK_M // 128, BLOCK_K // 32 * 128), tlx.dtype_of(a_scale), NUM_BUFFERS,
                             layout=_scale_shared_layout([BLOCK_M // 128, BLOCK_K // 32 * 128]))
    bs_buf = tlx.local_alloc((BLOCK_N // 128, BLOCK_K // 32 * 128), tlx.dtype_of(b_scale), NUM_BUFFERS,
                             layout=_scale_shared_layout([BLOCK_N // 128, BLOCK_K // 32 * 128]))
    # Two 64x64 output slots leave room for a fourth BK128 input stage.
    C_COLS: tl.constexpr = 64 if NUM_BUFFERS == 4 else BLOCK_N // 2
    C_ROWS: tl.constexpr = 128 if OUTPUT_REUSE else 64
    C_SLOTS: tl.constexpr = NUM_BUFFERS if OUTPUT_REUSE else 2
    if REGISTER_PIPELINE:
        c_buf = None
    elif OUTPUT_REUSE:
        # One FP32 quadrant has the same byte size as a BK256 FP8 A slot.
        # Loads and stores use this storage in disjoint phases of each tile.
        c_buf = tlx.local_alloc((128, BLOCK_N // 2), tl.float32, NUM_BUFFERS, reuse=a_buf)
    else:
        c_buf = tlx.local_alloc((C_ROWS, C_COLS), tl.float32, 2) if OUTPUT_STAGING else None
    num_m, num_n = M // BLOCK_M, N // BLOCK_N
    total_tiles = num_m * num_n
    k_iters = K // BLOCK_K
    tl.assume(k_iters >= NUM_BUFFERS)
    tile = _mxgemm_remap_program_id(tl.program_id(0), NUM_PROGRAMS, XCD_REMAP_MODE, NUM_XCDS, XCD_CHUNK)
    phase = 0
    off_m, off_n = _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_SIZE_M, BLOCK_M, BLOCK_N)
    for p in tl.static_range(NUM_BUFFERS):
        _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n, p, p,
                                BLOCK_K, DA, DB, NUM_BUFFERS, WITH_A_SCALE, TDM_FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST,
                                GROUP_SIZE_M, XCD_REMAP_MODE, CLUSTER_BARRIER_INTERVAL, REGISTER_PIPELINE)

    while tile < total_tiles:
        off_m, off_n = _mxgemm_tile_offsets(tile, num_m, num_n, GROUP_SIZE_M, BLOCK_M, BLOCK_N)
        c00 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
        c01 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
        c10 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
        c11 = tl.zeros((BLOCK_M // 2, BLOCK_N // 2), tl.float32)
        if REGISTER_PIPELINE:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
            a0, sa0 = _mxgemm_persistent_a(a_buf, as_buf, phase, 0, 0, BLOCK_M, BLOCK_K, DA, NUM_BUFFERS, WITH_A_SCALE,
                                           PREFETCH_SCALES=True)
            b0, sb0 = _mxgemm_persistent_b(b_buf, bs_buf, phase, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS,
                                           PREFETCH_SCALES=True)
            b1, sb1 = _mxgemm_persistent_b(b_buf, bs_buf, phase, 0, 1, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS,
                                           PREFETCH_SCALES=True)
            a1, sa1 = _mxgemm_persistent_a(a_buf, as_buf, phase, 1, 0, BLOCK_M, BLOCK_K, DA, NUM_BUFFERS, WITH_A_SCALE,
                                           PREFETCH_SCALES=True)
            tlx.amd_sched_barrier()
            steady_end = k_iters - NUM_BUFFERS
            slot = phase
            for i in tl.range(steady_end, loop_unroll_factor=1):
                tl.assume(slot >= 0)
                tl.assume(slot < NUM_BUFFERS)
                next_slot = tl.where(slot == NUM_BUFFERS - 1, 0, slot + 1)
                c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1 = _mxgemm_register_next_half(
                    c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M,
                    BLOCK_N, BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS, WITH_A_SCALE, 1)
                # C00 covers completion of the current slot's last LDS reads.
                # The other three quadrants cover incoming operands after refill.
                c00 = tlx.dot_scaled(a0, sa0, DTYPE_A, b0, sb0, DTYPE_B, c00, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
                _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n,
                                        i + NUM_BUFFERS, slot, BLOCK_K, DA, DB, NUM_BUFFERS, WITH_A_SCALE, TDM_FUSION,
                                        CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                        CLUSTER_BARRIER_INTERVAL, REGISTER_PIPELINE)
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
                tlx.amd_sched_barrier()
                c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1 = _mxgemm_register_next_stage(
                    c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1, a_buf, b_buf, as_buf, bs_buf, next_slot,
                    BLOCK_M, BLOCK_N, BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS, WITH_A_SCALE, 0)
                slot = next_slot

            tlx.amd_sched_barrier()
            next_tile = tile + NUM_PROGRAMS
            has_next = next_tile < total_tiles
            next_m, next_n = _mxgemm_tile_offsets(tl.minimum(next_tile, total_tiles - 1), num_m, num_n, GROUP_SIZE_M,
                                                  BLOCK_M, BLOCK_N)
            tlx.amd_sched_barrier()
            for j in tl.static_range(NUM_BUFFERS):
                slot = (phase + steady_end + j) % NUM_BUFFERS
                c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1 = _mxgemm_register_next_half(
                    c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M,
                    BLOCK_N, BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS, WITH_A_SCALE, 1)
                c00 = tlx.dot_scaled(a0, sa0, DTYPE_A, b0, sb0, DTYPE_B, c00, tiles_per_warp=[2, 2])
                tlx.amd_sched_barrier()
                if CROSS_TILE_PREFETCH and has_next and j == 0:
                    _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m,
                                            next_n, j, slot, BLOCK_K, DA, DB, NUM_BUFFERS, WITH_A_SCALE, TDM_FUSION,
                                            CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                            CLUSTER_BARRIER_INTERVAL, REGISTER_PIPELINE)
                if j < NUM_BUFFERS - 1:
                    if CROSS_TILE_PREFETCH and has_next:
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
                    else:
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - j - 2) * LOADS)
                    tlx.amd_sched_barrier()
                    next_slot = (phase + steady_end + j + 1) % NUM_BUFFERS
                    c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1 = _mxgemm_register_next_stage(
                        c00, c01, c10, c11, a0, sa0, a1, sa1, b0, sb0, b1, sb1, a_buf, b_buf, as_buf, bs_buf, next_slot,
                        BLOCK_M, BLOCK_N, BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS, WITH_A_SCALE, 0)
                else:
                    c10 = tlx.dot_scaled(a1, sa1, DTYPE_A, b0, sb0, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c01 = tlx.dot_scaled(a0, sa0, DTYPE_A, b1, sb1, DTYPE_B, c01, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c11 = tlx.dot_scaled(a1, sa1, DTYPE_A, b1, sb1, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()

        else:
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
            a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, phase, 0, 0, BLOCK_M, BLOCK_K, DA, NUM_BUFFERS,
                                             WITH_A_SCALE)
            b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, phase, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
            steady_end = k_iters - NUM_BUFFERS
            slot = phase
            for i in tl.range(steady_end):
                # Carry the physical slot so the hot loop needs only an increment
                # and wrap, including when K does not divide the ring depth.
                tl.assume(slot >= 0)
                tl.assume(slot < NUM_BUFFERS)
                next_slot = tl.where(slot == NUM_BUFFERS - 1, 0, slot + 1)
                if DEFER_BOTTOM:
                    c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(c00, c01, a00, sa00, b00, sb00,
                                                                                    a_buf, b_buf, as_buf, bs_buf, slot,
                                                                                    BLOCK_M, BLOCK_N, BLOCK_K, DA, DB,
                                                                                    DTYPE_A, DTYPE_B, NUM_BUFFERS,
                                                                                    WITH_A_SCALE, GROUPED_PREFETCH)
                    last_b00, last_sb00 = b00, sb00
                elif OUTPUT_REUSE:
                    c00, c01, c10, c11, a01, sa01, a11, sa11, b10, sb10, b11, sb11, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_256(
                        c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                        BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS, WITH_A_SCALE)
                else:
                    c00, c01, c10, c11 = _mxgemm_persistent_compute(c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf,
                                                                    b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                                                                    BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS,
                                                                    WITH_A_SCALE)
                _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, off_m, off_n,
                                        i + NUM_BUFFERS, slot, BLOCK_K, DA, DB, NUM_BUFFERS, WITH_A_SCALE, TDM_FUSION,
                                        CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                        CLUSTER_BARRIER_INTERVAL, REGISTER_PIPELINE)
                tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
                a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BLOCK_M, BLOCK_K, DA, NUM_BUFFERS,
                                                 WITH_A_SCALE)
                if not PREFETCH_B_AFTER_C10:
                    b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                if OUTPUT_REUSE:
                    if DB == 2 and tlx.num_warps() == 4:
                        c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                        c00 = tlx.dot_scaled(a01, sa01, DTYPE_A, b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                        c01 = tlx.dot_scaled(a01, sa01, DTYPE_A, b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                    c10 = tlx.dot_scaled(a11, sa11, DTYPE_A, b10, sb10, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c11 = tlx.dot_scaled(a11, sa11, DTYPE_A, b11, sb11, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                if DEFER_BOTTOM:
                    c10 = tlx.dot_scaled(a10, sa10, DTYPE_A, last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    if GROUPED_PREFETCH:
                        _mxgemm_schedule_prefetch(103, True)
                    tlx.amd_sched_barrier()
                    if PREFETCH_B_AFTER_C10:
                        # C10 is the last user of old B0. Prefetching here permits
                        # register reuse and gives every quadrant one incoming operand.
                        b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB,
                                                         NUM_BUFFERS)
                    c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    if GROUPED_PREFETCH:
                        _mxgemm_schedule_prefetch(104)
                    tlx.amd_sched_barrier()

                slot = next_slot

            # Only construct the next tile's offsets after the steady loop. The
            # final ring rotation can refill slots as soon as their operands are read.
            tlx.amd_sched_barrier()
            next_tile = tile + NUM_PROGRAMS
            has_next = next_tile < total_tiles
            next_m, next_n = _mxgemm_tile_offsets(tl.minimum(next_tile, total_tiles - 1), num_m, num_n, GROUP_SIZE_M,
                                                  BLOCK_M, BLOCK_N)
            tlx.amd_sched_barrier()
            for j in tl.static_range(NUM_BUFFERS):
                slot = (phase + steady_end + j) % NUM_BUFFERS
                if DEFER_BOTTOM:
                    c00, c01, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_top(c00, c01, a00, sa00, b00, sb00,
                                                                                    a_buf, b_buf, as_buf, bs_buf, slot,
                                                                                    BLOCK_M, BLOCK_N, BLOCK_K, DA, DB,
                                                                                    DTYPE_A, DTYPE_B, NUM_BUFFERS,
                                                                                    WITH_A_SCALE, GROUPED_PREFETCH)
                    last_b00, last_sb00 = b00, sb00
                elif OUTPUT_REUSE:
                    c00, c01, c10, c11, a01, sa01, a11, sa11, b10, sb10, b11, sb11, a10, sa10, b01, sb01 = _mxgemm_persistent_compute_256(
                        c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf, b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                        BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS, WITH_A_SCALE)
                else:
                    c00, c01, c10, c11 = _mxgemm_persistent_compute(c00, c01, c10, c11, a00, sa00, b00, sb00, a_buf,
                                                                    b_buf, as_buf, bs_buf, slot, BLOCK_M, BLOCK_N,
                                                                    BLOCK_K, DA, DB, DTYPE_A, DTYPE_B, NUM_BUFFERS,
                                                                    WITH_A_SCALE)
                if CROSS_TILE_PREFETCH and has_next:
                    _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m,
                                            next_n, j, slot, BLOCK_K, DA, DB, NUM_BUFFERS, WITH_A_SCALE, TDM_FUSION,
                                            CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                            CLUSTER_BARRIER_INTERVAL, REGISTER_PIPELINE)
                if j < NUM_BUFFERS - 1:
                    if CROSS_TILE_PREFETCH and has_next:
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
                    else:
                        # No replacement batch was issued. Drain the shrinking
                        # queue far enough to make the next consumer slot ready.
                        tlx.async_amd_descriptor_wait((NUM_BUFFERS - j - 2) * LOADS)
                    next_slot = (phase + steady_end + j + 1) % NUM_BUFFERS
                    a00, sa00 = _mxgemm_persistent_a(a_buf, as_buf, next_slot, 0, 0, BLOCK_M, BLOCK_K, DA, NUM_BUFFERS,
                                                     WITH_A_SCALE)
                    b00, sb00 = _mxgemm_persistent_b(b_buf, bs_buf, next_slot, 0, 0, BLOCK_N, BLOCK_K, DB, NUM_BUFFERS)
                if OUTPUT_REUSE:
                    if DB == 2 and tlx.num_warps() == 4:
                        c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                        c00 = tlx.dot_scaled(a01, sa01, DTYPE_A, b10, sb10, DTYPE_B, c00, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                        c01 = tlx.dot_scaled(a01, sa01, DTYPE_A, b11, sb11, DTYPE_B, c01, tiles_per_warp=[2, 2])
                        tlx.amd_sched_barrier()
                    c10 = tlx.dot_scaled(a11, sa11, DTYPE_A, b10, sb10, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c11 = tlx.dot_scaled(a11, sa11, DTYPE_A, b11, sb11, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                if DEFER_BOTTOM:
                    c10 = tlx.dot_scaled(a10, sa10, DTYPE_A, last_b00, last_sb00, DTYPE_B, c10, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()
                    c11 = tlx.dot_scaled(a10, sa10, DTYPE_A, b01, sb01, DTYPE_B, c11, tiles_per_warp=[2, 2])
                    tlx.amd_sched_barrier()

        if REGISTER_PIPELINE:
            # Keep next-tile K=0 in one input stage. The other stage is free
            # for FP32 output; drain its consumers before loading next-tile K=1.
            tlx.amd_sched_barrier()
            tl.static_assert(OUTPUT_STAGING)
            free_slot = (phase + k_iters - 1) % NUM_BUFFERS
            c_a = tlx.local_reinterpret(a_buf[free_slot], tl.float32, [256, 64],
                                        layout=_mxgemm_register_output_layout())
            c_b = tlx.local_reinterpret(b_buf[free_slot], tl.float32, [256, 64],
                                        layout=_mxgemm_register_output_layout())
            _mxgemm_store_register_panels(c_ptr, c00, c01, c10, c11, off_m, off_n, stride_cm, c_a, c_b)
            tlx.async_amd_descriptor_wait(0)
            tlx.amd_sched_barrier()
        else:
            # Dedicated staging preserves prefetched inputs. BK256 instead reuses
            # A only after the final operand reads, then drains stores before reload.
            tlx.amd_sched_barrier()
            INSTR_M: tl.constexpr = 32 if DA == 2 and DB == 2 else 16
            _mxgemm_persistent_store(c_ptr, c00, off_m, off_n, stride_cm, BLOCK_M // 2, BLOCK_N // 2, INSTR_M, c_buf,
                                     C_ROWS, C_SLOTS, 0, C_COLS)
            _mxgemm_persistent_store(c_ptr, c01, off_m, off_n + BLOCK_N // 2, stride_cm, BLOCK_M // 2, BLOCK_N // 2,
                                     INSTR_M, c_buf, C_ROWS, C_SLOTS, 1, C_COLS)
            _mxgemm_persistent_store(c_ptr, c10, off_m + BLOCK_M // 2, off_n, stride_cm, BLOCK_M // 2, BLOCK_N // 2,
                                     INSTR_M, c_buf, C_ROWS, C_SLOTS, 2, C_COLS)
            _mxgemm_persistent_store(c_ptr, c11, off_m + BLOCK_M // 2, off_n + BLOCK_N // 2, stride_cm, BLOCK_M // 2,
                                     BLOCK_N // 2, INSTR_M, c_buf, C_ROWS, C_SLOTS, 3, C_COLS)
            if OUTPUT_REUSE:
                tlx.async_amd_descriptor_wait(0)
            tlx.amd_sched_barrier()
        # K=0 of the next tile occupies the first slot recycled by the tail:
        # (phase + k_iters - NUM_BUFFERS) % NUM_BUFFERS. Keep that phase even
        # when K's iteration count is not divisible by the ring depth.
        phase = (phase + k_iters) % NUM_BUFFERS
        tile = next_tile
        if (REGISTER_PIPELINE or not CROSS_TILE_PREFETCH) and has_next:
            for p in tl.static_range(1 if REGISTER_PIPELINE and CROSS_TILE_PREFETCH else 0, NUM_BUFFERS):
                _mxgemm_persistent_load(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, next_m, next_n,
                                        p, (phase + p) % NUM_BUFFERS, BLOCK_K, DA, DB, NUM_BUFFERS, WITH_A_SCALE,
                                        TDM_FUSION, CLUSTER_SIZE, CLUSTER_MULTICAST, GROUP_SIZE_M, XCD_REMAP_MODE,
                                        CLUSTER_BARRIER_INTERVAL, REGISTER_PIPELINE)
    if OUTPUT_STAGING:
        tlx.async_amd_descriptor_wait(0)
    if CLUSTER_SIZE > 1 and CLUSTER_BARRIER_INTERVAL > 0:
        tlx.async_amd_descriptor_wait(0)
        tlx.cluster_barrier()


def _mxgemm_persistent_programs(a, b, a_scale, b_scale, M, N, K, BM, BN, BK, num_buffers, group_m, transpose_b,
                                preshuffle, with_a_scale, schedule, fusion, split, l2_prefetch, num_programs):
    if schedule != "sliceMNK" or not transpose_b or not preshuffle or split or l2_prefetch != -1:
        raise ValueError("persistent MXFP requires sliceMNK, transposed B, preshuffled scales, "
                         "unsplit descriptors, and L2 prefetch disabled")
    if (BM not in (128, 256) or BN not in (128, 256) or BK not in (128, 256) or num_buffers not in (2, 3, 4)
            or group_m <= 0):
        raise ValueError("persistent MXFP requires 128/256 M, N, and K tiles, "
                         "2/3/4 buffers, and positive GROUP_SIZE_M")
    if min(M, N, K) <= 0 or M % BM or N % BN or K % BK or K // BK < num_buffers:
        raise ValueError("persistent MXFP requires nonempty full tiles and at least NUM_BUFFERS K tiles")
    if M != a.shape[0] or N != b.shape[0] or tuple(b_scale.shape) != (N // 128, K // 32 * 128):
        raise ValueError("persistent MXFP dimensions must match the data and preshuffled scales")
    if with_a_scale and (a_scale is None or tuple(a_scale.shape) != (M // 128, K // 32 * 128)):
        raise ValueError("persistent MXFP dimensions must match the preshuffled A scales")
    if fusion not in ("none", "2way", "4way", "partial") or (not with_a_scale and fusion in ("4way", "partial")):
        raise ValueError("invalid persistent MXFP fusion or missing A scales")
    if a.stride(1) != 1 or b.stride(1) != 1 or b_scale.stride(1) != 1:
        raise ValueError("persistent MXFP requires contiguous K dimensions")
    if with_a_scale and (a_scale is None or a_scale.stride(1) != 1):
        raise ValueError("persistent MXFP requires preshuffled A scales with contiguous K")
    if num_programs is not None and (not isinstance(num_programs, int) or num_programs <= 0):
        raise ValueError("NUM_PROGRAMS must be a positive integer")
    if num_programs is None:
        num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    return min(num_programs, (M // BM) * (N // BN))


def _mxfp_gemm_tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2 * M * N * K / (ms * 1e-3) / 1e12


def _validate_mxfp_scheduling(M, N, num_programs, *, persistent, block_m, block_n, group_m, with_a_scale, fusion,
                              xcd_remap, num_xcds, xcd_chunk, cluster_size, cluster_barrier_interval=1):
    if not isinstance(cluster_barrier_interval, int) or cluster_barrier_interval < 0:
        raise ValueError("CLUSTER_BARRIER_INTERVAL must be a nonnegative integer")
    if xcd_remap not in _XCD_REMAP_MODES:
        raise ValueError("XCD_REMAP must be none, balanced, or chunked")
    if num_xcds <= 0 or xcd_chunk <= 0:
        raise ValueError("NUM_XCDS and XCD_CHUNK must be positive")
    if cluster_size not in (1, 2, 4):
        raise ValueError("CLUSTER_SIZE must be 1, 2, or 4")
    if not persistent and (xcd_remap != "none" or cluster_size > 1):
        raise ValueError("XCD remapping and clustering require persistence")
    if cluster_size == 1:
        return
    if xcd_remap != "none" and (xcd_remap, num_xcds, xcd_chunk) != ("chunked", 8, 2):
        raise ValueError("clustering requires XCD_REMAP=none or chunked with NUM_XCDS=8 and XCD_CHUNK=2")
    if group_m not in (4, 8) or M % (group_m * block_m) or N % (2 * block_n):
        raise ValueError("clustering requires GROUP_SIZE_M=4 or 8, full M groups, and an even number of N tiles")
    if not with_a_scale or fusion not in ("partial", "2way", "4way"):
        raise ValueError("clustering requires A scales and partial, 2way, or 4way TDM fusion")
    # Every cluster member must execute the same number of persistent tiles,
    # including taking the same has_next branches around cluster barriers.
    tiles = (M // block_m) * (N // block_n)
    if not isinstance(num_programs, int) or num_programs <= 0 or num_programs % 16 or tiles % num_programs:
        raise ValueError("clustering requires NUM_PROGRAMS divisible by 16 and dividing the output tile count; "
                         "set NUM_PROGRAMS explicitly if necessary")


def _init_data(dtype: str, rows: int, cols: int):
    if dtype == "float4":
        return MXFP4Tensor(size=(rows, cols)).random()
    if dtype == "float8_e5m2":
        return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e5m2)
    if dtype == "float8_e4m3":
        return torch.randint(20, 40, (rows, cols), dtype=torch.uint8).view(torch.float8_e4m3fn)
    raise ValueError(f"unsupported dtype: {dtype}")


def mxgemm_tdm_pipelined(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    TRANSPOSE_B: bool = False,
    NUM_BUFFERS: int = 2,
    DTYPE_A: str = "e5m2",
    DTYPE_B: str = "e5m2",
    SCALE_PRESHUFFLE: bool = True,
    WITH_A_SCALE: bool = True,
    SCHEDULE: str = "baseline",
    L2_PREFETCH_DISTANCE: int = -1,
    M: int | None = None,
    N: int | None = None,
    K: int | None = None,
    TDM_FUSION: str = "none",
    TDM_SPLIT: bool = False,
    GROUP_SIZE_M: int = 8,
    BENCHMARK: str | None = None,
    BENCHMARK_NUM_ITERS: int = 32,
    PERSISTENT: bool = False,
    NUM_PROGRAMS: int | None = None,
    CROSS_TILE_PREFETCH: bool = True,
    OUTPUT_STAGING: bool = False,
    SCHED_MODE_2: bool = False,
    XCD_REMAP: str = "none",
    NUM_XCDS: int = 8,
    XCD_CHUNK: int = 2,
    CLUSTER_SIZE: int = 1,
    CLUSTER_MULTICAST: bool = True,
    CLUSTER_BARRIER_INTERVAL: int = 1,
    NUM_WARPS: int = 4,
    REGISTER_PIPELINE: bool = False,
    WARP_PIPELINE: bool = False,
    OPERAND_PIPELINE: bool = False,
) -> torch.Tensor:
    """Run MXFP GEMM, optionally with persistent full-tile sliceMNK scheduling.

    ``PERSISTENT`` requires transposed B, preshuffled scales, unsplit descriptors,
    128/256 M, N, and K tiles, two to four buffers, and L2 prefetch disabled.
    Four 128-element K stages fit the same input footprint as two 256-element
    stages and allow earlier ring refills. M/N/K must
    contain full tiles, with at least NUM_BUFFERS K tiles. ``NUM_PROGRAMS``
    defaults to the smaller of the CU count and the output tile count.
    ``CROSS_TILE_PREFETCH=False`` retains persistence but primes each tile
    after the previous output store, providing a control for input overlap.
    ``OUTPUT_STAGING`` requires persistence, 256x256 M/N tiles, and FP8 A.
    BK128 uses two dedicated output slots: 64x128 with 2/3 input buffers,
    or 64x64 with four input buffers and four waves.
    BK256 reuses the A ring for output and requires ``CROSS_TILE_PREFETCH=False``; it
    supports two input buffers for A8W8 and two or three for A8W4.
    ``REGISTER_PIPELINE`` selects the A8W8 256x256x256, two-buffer, four-wave
    register pipeline. It requires both scales, partial TDM fusion, and
    output staging. It retains one next-tile input stage and reuses the
    free A/B stage for four 256x64 FP32 output panels.
    ``SCHED_MODE_2`` optionally enables hardware WMMA queuing in the persistent
    kernel. It defaults to false, leaving the hardware scheduling mode unchanged.
    ``XCD_REMAP`` selects none, balanced, or chunked persistent program ordering.
    ``CLUSTER_SIZE=2|4`` groups independent workgroups for input/scale multicast;
    ``CLUSTER_MULTICAST=False`` keeps the cluster barriers with independent loads.
    ``CLUSTER_BARRIER_INTERVAL`` aligns requests every N input K blocks (default
    one), resetting at K=0 of each tile. Zero removes all cluster barriers.
    Barriers only improve multicast overlap; each CTA's local waits protect LDS.
    Clustering supports no remapping (sharing B and its scales), or chunked
    remapping with eight XCDs and chunk size two. Both require
    full M groups of four or eight tiles, even N tile counts, and fused loads
    with both scales. The program count must be divisible by 16 and divide the
    total tile count so cluster members take identical loop boundaries.
    ``NUM_WARPS=8`` experiments with two waves per SIMD on persistent
    256x256 tiles with FP8 A. Four input buffers currently require direct
    output stores with eight waves.
    ``WARP_PIPELINE`` selects the experimental nonpersistent E4M3 x E4M3
    eight-wave kernel, with two K256 payload slots and three scale slots.
    It requires 256x256 tiles, preshuffled scales, partial fusion, staged FP32
    output, and K divisible by 512 with K >= 1024.
    ``OPERAND_PIPELINE`` selects the experimental nonpersistent E4M3 x E4M3
    four-wave kernel with four K128 payload/scale slots, rolling A prefetch,
    and two B register sets. It requires 256x256 tiles, preshuffled scales,
    partial unsplit TDM, staged FP32 output, and K >= 512 divisible by 256.
    """
    if M is None:
        M = a.shape[0]
    if K is None:
        K = a.shape[1] * (2 if DTYPE_A == "e2m1" else 1)
    if N is None:
        if TRANSPOSE_B:
            N = b.shape[0]
        else:
            N = b.shape[1]
    if TRANSPOSE_B:
        Kb = b.shape[1] * (2 if DTYPE_B == "e2m1" else 1)
    else:
        Kb = b.shape[0] * (2 if DTYPE_B == "e2m1" else 1)
    assert K == Kb
    if OPERAND_PIPELINE:
        if not (not PERSISTENT and not REGISTER_PIPELINE and not WARP_PIPELINE and NUM_WARPS == 4 and NUM_BUFFERS == 4
                and (BLOCK_M, BLOCK_N, BLOCK_K) == (256, 256, 128) and DTYPE_A == DTYPE_B == "e4m3" and TRANSPOSE_B
                and SCALE_PRESHUFFLE and WITH_A_SCALE and OUTPUT_STAGING and TDM_FUSION == "partial" and not TDM_SPLIT
                and SCHEDULE == "sliceMNK" and L2_PREFETCH_DISTANCE == -1):
            raise ValueError("OPERAND_PIPELINE requires nonpersistent E4M3 x E4M3, 256x256x128 tiles, "
                             "four warps, four payload/scale buffers, transposed B, preshuffled scales, "
                             "partial unsplit TDM, sliceMNK, staged output, and no L2 prefetch")
        if min(M, N) <= 0 or M % 256 or N % 256 or K < 512 or K % 256 or GROUP_SIZE_M <= 0:
            raise ValueError("OPERAND_PIPELINE requires full M/N tiles, positive GROUP_SIZE_M, "
                             "and K >= 512 divisible by 256")
        if (a.shape != (M, K) or b.shape != (N, K) or a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn
                or a.stride(1) != 1 or b.stride(1) != 1 or a_scale is None or b_scale is None
                or a_scale.dtype != torch.uint8 or b_scale.dtype != torch.uint8 or not a_scale.is_contiguous()
                or not b_scale.is_contiguous() or a_scale.shape != (M // 128, K * 4)
                or b_scale.shape != (N // 128, K * 4)):
            raise ValueError("OPERAND_PIPELINE requires matching E4M3 input shapes, contiguous K, "
                             "and packed uint8 scales of shape (rows // 128, K * 4)")
        if __package__:
            from .amd_mxfp_gemm_operand_pipeline import mxgemm_tdm_operand_pipeline_kernel
        else:
            from amd_mxfp_gemm_operand_pipeline import mxgemm_tdm_operand_pipeline_kernel
    if WARP_PIPELINE:
        if not (not PERSISTENT and not REGISTER_PIPELINE and NUM_WARPS == 8 and NUM_BUFFERS == 2 and
                (BLOCK_M, BLOCK_N, BLOCK_K) == (256, 256, 256) and DTYPE_A == DTYPE_B == "e4m3" and TRANSPOSE_B
                and SCALE_PRESHUFFLE and WITH_A_SCALE and OUTPUT_STAGING and TDM_FUSION == "partial" and not TDM_SPLIT
                and SCHEDULE == "sliceMNK" and L2_PREFETCH_DISTANCE == -1):
            raise ValueError("WARP_PIPELINE requires nonpersistent E4M3 x E4M3, 256x256x256 tiles, "
                             "eight warps, two payload buffers, transposed B, preshuffled scales, "
                             "partial unsplit TDM, sliceMNK, staged output, and no L2 prefetch")
        if min(M, N) <= 0 or M % 256 or N % 256 or K < 1024 or K % 512:
            raise ValueError("WARP_PIPELINE requires full M/N tiles and K >= 1024 divisible by 512")
        if (a_scale is None or b_scale is None or a.stride(1) != 1 or b.stride(1) != 1 or not a_scale.is_contiguous()
                or not b_scale.is_contiguous() or a_scale.shape != (M // 128, K * 4)
                or b_scale.shape != (N // 128, K * 4)):
            raise ValueError("WARP_PIPELINE requires contiguous K and packed scales of shape (rows // 128, K * 4)")
        from amd_mxfp_gemm_warp_pipeline import get_layouts, mxfp8_warp_pipeline_kernel
        warp_layouts = get_layouts()
    if REGISTER_PIPELINE and not (PERSISTENT and OUTPUT_STAGING and WITH_A_SCALE and TDM_FUSION == "partial"
                                  and NUM_WARPS == 4 and BLOCK_M == 256 and BLOCK_N == 256 and BLOCK_K == 256
                                  and NUM_BUFFERS == 2 and DTYPE_A in ("e4m3", "e5m2") and DTYPE_B in ("e4m3", "e5m2")):
        raise ValueError("REGISTER_PIPELINE requires persistent A8W8, 256x256x256 tiles, two buffers, "
                         "four warps, both scales, partial TDM fusion, and output staging")
    if NUM_WARPS not in (4, 8):
        raise ValueError("MXFP supports four or eight warps")
    if NUM_WARPS == 8 and not WARP_PIPELINE:
        if not PERSISTENT or BLOCK_M != 256 or BLOCK_N != 256 or DTYPE_A == "e2m1":
            raise ValueError("eight-wave MXFP requires persistent 256x256 M/N tiles and FP8 A")
        if OUTPUT_STAGING and NUM_BUFFERS == 4:
            raise ValueError("eight-wave output staging supports at most three input buffers")
    if SCHED_MODE_2 and not PERSISTENT:
        raise ValueError("SCHED_MODE_2 requires persistence")
    if OUTPUT_STAGING and not (WARP_PIPELINE or OPERAND_PIPELINE):
        if not PERSISTENT or BLOCK_M != 256 or BLOCK_N != 256 or DTYPE_A == "e2m1":
            raise ValueError("output staging requires persistent 256x256 M/N tiles and FP8 A")
        if BLOCK_K == 128:
            if NUM_BUFFERS not in (2, 3, 4):
                raise ValueError("BK128 output staging requires 2/3/4 buffers")
        elif BLOCK_K == 256:
            if CROSS_TILE_PREFETCH and not REGISTER_PIPELINE:
                raise ValueError("BK256 output staging reuses the A ring and requires CROSS_TILE_PREFETCH=False")
            if NUM_BUFFERS != 2 and not (NUM_BUFFERS == 3 and DTYPE_B == "e2m1"):
                raise ValueError("BK256 output staging requires 2 buffers, or 3 buffers with FP4 B")
        else:
            raise ValueError("output staging requires BK128 or BK256")
    if PERSISTENT:
        assert K == a.shape[1] * (2 if DTYPE_A == "e2m1" else 1)
        NUM_PROGRAMS = _mxgemm_persistent_programs(a, b, a_scale, b_scale, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                   NUM_BUFFERS, GROUP_SIZE_M, TRANSPOSE_B, SCALE_PRESHUFFLE,
                                                   WITH_A_SCALE, SCHEDULE, TDM_FUSION, TDM_SPLIT, L2_PREFETCH_DISTANCE,
                                                   NUM_PROGRAMS)
    _validate_mxfp_scheduling(M, N, NUM_PROGRAMS, persistent=PERSISTENT, block_m=BLOCK_M, block_n=BLOCK_N,
                              group_m=GROUP_SIZE_M, with_a_scale=WITH_A_SCALE, fusion=TDM_FUSION, xcd_remap=XCD_REMAP,
                              num_xcds=NUM_XCDS, xcd_chunk=XCD_CHUNK, cluster_size=CLUSTER_SIZE,
                              cluster_barrier_interval=CLUSTER_BARRIER_INTERVAL)
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    a_scale_arg = a_scale if WITH_A_SCALE else b_scale
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    def run_kernel():
        if WARP_PIPELINE:
            return mxfp8_warp_pipeline_kernel[grid](a, b, c, a_scale, b_scale, M, N, K, a.stride(0), b.stride(0),
                                                    c.stride(0), a_scale.stride(0), **warp_layouts,
                                                    GROUP_M=GROUP_SIZE_M, num_warps=8, waves_per_eu=2)
        if PERSISTENT:
            return mxgemm_tdm_persistent_kernel[(NUM_PROGRAMS, )](
                a, b, c, a_scale_arg, b_scale, M, N, K, a.stride(0), b.stride(0), c.stride(0), a_scale_arg.stride(0),
                b_scale.stride(0), DTYPE_A=DTYPE_A, DTYPE_B=DTYPE_B, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_SIZE_M, NUM_BUFFERS=NUM_BUFFERS, WITH_A_SCALE=WITH_A_SCALE, TDM_FUSION=TDM_FUSION,
                NUM_PROGRAMS=NUM_PROGRAMS, CROSS_TILE_PREFETCH=CROSS_TILE_PREFETCH, OUTPUT_STAGING=OUTPUT_STAGING,
                REGISTER_PIPELINE=REGISTER_PIPELINE, SCHED_MODE_2=SCHED_MODE_2,
                XCD_REMAP_MODE=_XCD_REMAP_MODES[XCD_REMAP], NUM_XCDS=NUM_XCDS, XCD_CHUNK=XCD_CHUNK,
                CLUSTER_SIZE=CLUSTER_SIZE, CLUSTER_MULTICAST=CLUSTER_MULTICAST,
                CLUSTER_BARRIER_INTERVAL=CLUSTER_BARRIER_INTERVAL, num_warps=NUM_WARPS, waves_per_eu=NUM_WARPS // 4,
                ctas_per_cga=(CLUSTER_SIZE, 1, 1))
        kernel = mxgemm_tdm_operand_pipeline_kernel if OPERAND_PIPELINE else mxgemm_tdm_pipelined_kernel
        return kernel[grid](
            a,
            b,
            c,
            a_scale_arg,
            b_scale,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            stride_bk,
            stride_bn,
            c.stride(0),
            c.stride(1),
            b_scale.stride(0),
            DTYPE_A=DTYPE_A,
            DTYPE_B=DTYPE_B,
            SCALE_BLOCK=32,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            TRANSPOSE_B=TRANSPOSE_B,
            NUM_BUFFERS=NUM_BUFFERS,
            SCALE_PRESHUFFLE=SCALE_PRESHUFFLE,
            WITH_A_SCALE=WITH_A_SCALE,
            SCHEDULE=SCHEDULE,
            TDM_FUSION=TDM_FUSION,
            L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
            TDM_SPLIT=TDM_SPLIT,
            num_warps=4,
            waves_per_eu=1,
        )

    if BENCHMARK == "graph":
        ms = triton.testing.do_bench_cudagraph(run_kernel, rep=BENCHMARK_NUM_ITERS)
        print(f"execution time: {ms} ms, {_mxfp_gemm_tflops(ms, M, N, K):.2f} TFLOPS")
    elif BENCHMARK == "eager":
        ms = triton.testing.do_bench(run_kernel, warmup=30, rep=BENCHMARK_NUM_ITERS)
        print(f"execution time: {ms} ms, {_mxfp_gemm_tflops(ms, M, N, K):.2f} TFLOPS")
    else:
        run_kernel()
    return c


_DEFAULT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "GROUP_SIZE_M": 8,
    "NUM_BUFFERS": 2,
    "DTYPE_A": "e5m2",
    "DTYPE_B": "e5m2",
    "SCALE_BLOCK": 32,
    "TRANSPOSE_B": False,
    "num_warps": 4,
    "waves_per_eu": 1,
}


def matmul(a: torch.Tensor, b: torch.Tensor, a_scale: torch.Tensor, b_scale: torch.Tensor, config=None) -> torch.Tensor:
    """C = (A * a_scale) @ (B * b_scale) using a TDM-pipelined MXFP kernel on AMD gfx1250.

    ``a_scale`` / ``b_scale`` must already be pre-shuffled with :func:`pack_scale`.
    When ``config["TRANSPOSE_B"]`` is set, ``b`` is the ``[N, K]`` transposed layout.
    ``PERSISTENT``, ``NUM_PROGRAMS``, ``CROSS_TILE_PREFETCH``, ``OUTPUT_STAGING``, and ``SCHED_MODE_2``
    configure the persistent full-tile sliceMNK path; see :func:`mxgemm_tdm_pipelined`.
    ``XCD_REMAP``, ``NUM_XCDS``, ``XCD_CHUNK``, ``CLUSTER_SIZE``, and ``CLUSTER_MULTICAST``
    select persistent program remapping and input/scale sharing across workgroups.
    """
    cfg = dict(_DEFAULT_CONFIG)
    if config is not None:
        cfg.update(config)
    TRANSPOSE_B = cfg["TRANSPOSE_B"]

    M, packed_k = a.shape
    K = packed_k * (2 if cfg["DTYPE_A"] == "e2m1" else 1)
    if TRANSPOSE_B:
        N, Kb = b.shape
    else:
        Kb, N = b.shape
    Kb *= 2 if cfg["DTYPE_B"] == "e2m1" else 1
    assert K == Kb, f"K mismatch: A={a.shape}, B={b.shape}"

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]

    if cfg.get("REGISTER_PIPELINE", False) and not cfg.get("PERSISTENT", False):
        raise ValueError("REGISTER_PIPELINE requires persistence")
    if cfg.get("OUTPUT_STAGING", False) and not (cfg.get("PERSISTENT", False) or cfg.get("WARP_PIPELINE", False)
                                                 or cfg.get("OPERAND_PIPELINE", False)):
        raise ValueError("output staging requires persistence")
    if cfg.get("SCHED_MODE_2", False) and not cfg.get("PERSISTENT", False):
        raise ValueError("SCHED_MODE_2 requires persistence")
    if not cfg.get("PERSISTENT", False):
        _validate_mxfp_scheduling(M, N, cfg.get("NUM_PROGRAMS"), persistent=False, block_m=BLOCK_M, block_n=BLOCK_N,
                                  group_m=cfg["GROUP_SIZE_M"], with_a_scale=cfg.get("WITH_A_SCALE", True),
                                  fusion=cfg.get("TDM_FUSION", "none"), xcd_remap=cfg.get("XCD_REMAP", "none"),
                                  num_xcds=cfg.get("NUM_XCDS",
                                                   8), xcd_chunk=cfg.get("XCD_CHUNK",
                                                                         2), cluster_size=cfg.get("CLUSTER_SIZE", 1),
                                  cluster_barrier_interval=cfg.get("CLUSTER_BARRIER_INTERVAL", 1))
    if cfg.get("PERSISTENT", False) or cfg.get("WARP_PIPELINE", False) or cfg.get("OPERAND_PIPELINE", False):
        if (cfg["num_warps"] not in (4, 8) or cfg["waves_per_eu"] not in (1, cfg["num_warps"] // 4)
                or cfg["SCALE_BLOCK"] != 32):
            raise ValueError("pipelined MXFP requires four/eight warps, one/two waves per SIMD, and SCALE_BLOCK=32")
        return mxgemm_tdm_pipelined(
            a, b, a_scale, b_scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, TRANSPOSE_B=TRANSPOSE_B,
            NUM_BUFFERS=cfg["NUM_BUFFERS"], DTYPE_A=cfg["DTYPE_A"], DTYPE_B=cfg["DTYPE_B"],
            SCALE_PRESHUFFLE=cfg.get("SCALE_PRESHUFFLE",
                                     True), WITH_A_SCALE=cfg.get("WITH_A_SCALE",
                                                                 True), SCHEDULE=cfg.get("SCHEDULE", "baseline"),
            L2_PREFETCH_DISTANCE=cfg.get("L2_PREFETCH_DISTANCE",
                                         -1), TDM_FUSION=cfg.get("TDM_FUSION",
                                                                 "none"), TDM_SPLIT=cfg.get("TDM_SPLIT", False),
            GROUP_SIZE_M=cfg["GROUP_SIZE_M"], PERSISTENT=cfg.get("PERSISTENT",
                                                                 False), WARP_PIPELINE=cfg.get("WARP_PIPELINE", False),
            NUM_PROGRAMS=cfg.get("NUM_PROGRAMS"), CROSS_TILE_PREFETCH=cfg.get("CROSS_TILE_PREFETCH", True),
            OUTPUT_STAGING=cfg.get("OUTPUT_STAGING",
                                   False), SCHED_MODE_2=cfg.get("SCHED_MODE_2",
                                                                False), XCD_REMAP=cfg.get("XCD_REMAP", "none"),
            NUM_XCDS=cfg.get("NUM_XCDS", 8), XCD_CHUNK=cfg.get("XCD_CHUNK", 2), CLUSTER_SIZE=cfg.get("CLUSTER_SIZE", 1),
            CLUSTER_MULTICAST=cfg.get("CLUSTER_MULTICAST",
                                      True), CLUSTER_BARRIER_INTERVAL=cfg.get("CLUSTER_BARRIER_INTERVAL",
                                                                              1), NUM_WARPS=cfg["num_warps"],
            REGISTER_PIPELINE=cfg.get("REGISTER_PIPELINE", False), OPERAND_PIPELINE=cfg.get("OPERAND_PIPELINE", False))

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    stride_bk, stride_bn = (b.stride(0), b.stride(1)) if not TRANSPOSE_B else (b.stride(1), b.stride(0))
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    mxgemm_tdm_pipelined_kernel[grid](
        a,
        b,
        c,
        a_scale if cfg.get("WITH_A_SCALE", True) else b_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        stride_bk,
        stride_bn,
        c.stride(0),
        c.stride(1),
        b_scale.stride(0),
        DTYPE_A=cfg["DTYPE_A"],
        DTYPE_B=cfg["DTYPE_B"],
        SCALE_BLOCK=cfg["SCALE_BLOCK"],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=cfg["GROUP_SIZE_M"],
        TRANSPOSE_B=TRANSPOSE_B,
        NUM_BUFFERS=cfg["NUM_BUFFERS"],
        SCALE_PRESHUFFLE=cfg.get("SCALE_PRESHUFFLE", True),
        WITH_A_SCALE=cfg.get("WITH_A_SCALE", True),
        SCHEDULE=cfg.get("SCHEDULE", "baseline"),
        TDM_FUSION=cfg.get("TDM_FUSION", "none"),
        TDM_SPLIT=cfg.get("TDM_SPLIT", False),
        L2_PREFETCH_DISTANCE=cfg.get("L2_PREFETCH_DISTANCE", -1),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    return c


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the TLX AMD MXFP TDM GEMM tutorial kernel")
    parser.add_argument("-M", type=int, default=2048, help="problem M size")
    parser.add_argument("-N", type=int, default=1024, help="problem N size")
    parser.add_argument("-K", type=int, default=8192, help="problem K size")
    parser.add_argument("-BM", type=int, default=256, help="BLOCK_M")
    parser.add_argument("-BN", type=int, default=256, help="BLOCK_N")
    parser.add_argument("-BK", type=int, default=256, help="BLOCK_K")
    parser.add_argument("--num_warps", type=int, default=4, choices=[4, 8], help="kernel waves per workgroup")
    parser.add_argument("--num_buffers", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--group_size_m", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--dtype_a", type=str, default="float8_e4m3", choices=tuple(DTYPE_TO_TRITON))
    parser.add_argument("--dtype_b", type=str, default="float4", choices=tuple(DTYPE_TO_TRITON))
    parser.add_argument("--scale_preshuffled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with_a_scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--transpose_b", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--schedule", type=str, default="sliceMNK",
                        choices=["baseline", "sliceK", "sliceNK", "sliceMNK"])
    parser.add_argument("--tdm_fusion", type=str, default="partial", choices=["none", "2way", "4way", "partial"])
    parser.add_argument("--partial_tdm", action="store_true",
                        help="Alias for --tdm_fusion partial, matching the Gluon CLI spelling")
    parser.add_argument("--tdm_split", action="store_true")
    parser.add_argument("--persistent", action="store_true", help="use persistent full-tile sliceMNK scheduling")
    parser.add_argument("--register_pipeline", action="store_true",
                        help="use the A8W8 BK256/two-buffer register pipeline and four FP32 output panels")
    parser.add_argument("--warp_pipeline", action="store_true",
                        help="use the nonpersistent E4M3 eight-wave pipeline with two payload and three scale slots")
    parser.add_argument("--operand_pipeline", action="store_true",
                        help="use the nonpersistent E4M3 four-wave K128 pipeline with four payload/scale buffers")
    parser.add_argument("--sched_mode_2", action=argparse.BooleanOptionalAction, default=False,
                        help="set SCHED_MODE[2] for persistent WMMA queuing (default: disabled)")
    parser.add_argument("--output_staging", action="store_true", help="stage persistent FP32 output for TDM stores")
    parser.add_argument("--num_programs", type=int, default=None, help="persistent workgroup count (default: CU count)")
    parser.add_argument("--xcd_remap", choices=tuple(_XCD_REMAP_MODES), default="none")
    parser.add_argument("--num_xcds", type=int, default=8)
    parser.add_argument("--xcd_chunk", type=int, default=2)
    parser.add_argument("--cluster_size", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--cluster_barrier_interval", type=int, default=1,
                        help="align cluster requests every N input K blocks; 0 disables all cluster barriers")
    parser.add_argument("--cluster_multicast", action=argparse.BooleanOptionalAction, default=True,
                        help="share data and scales within a cluster; disable to keep only cluster synchronization")
    parser.add_argument("--cross_tile_prefetch", action=argparse.BooleanOptionalAction, default=True,
                        help="prefetch the next persistent tile during the current tile's K-loop tail")
    parser.add_argument("--l2_prefetch_distance", type=int, default=-1,
                        help="Prefetch distance in K iterations; -1 disables L2 prefetch")
    parser.add_argument("--benchmark_mode", choices=["eager", "graph", "none"], default="eager")
    parser.add_argument("--benchmark_num_iters", type=int, default=32,
                        help="timing repetition budget in milliseconds (not an iteration count)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.sched_mode_2 and not args.persistent:
        parser.error("--sched_mode_2 requires --persistent")

    torch.manual_seed(args.seed)
    a = _init_data(args.dtype_a, args.M, args.K)
    b = _init_data(args.dtype_b, args.K, args.N)
    if args.with_a_scale:
        a_scale = MXScaleTensor(size=(args.M, triton.cdiv(args.K, 32))).random(high=32.0).data
    else:
        a_scale = None
    b_scale = MXScaleTensor(size=(args.N, triton.cdiv(args.K, 32))).random(high=32.0).data

    a_scale_input = pack_scale(a_scale) if args.scale_preshuffled and a_scale is not None else a_scale
    b_scale_input = pack_scale(b_scale) if args.scale_preshuffled else b_scale
    if args.dtype_a == "float4":
        a = a.to_packed_tensor(dim=1)
    if args.dtype_b == "float4":
        b = b.to_packed_tensor(dim=0)

    a_d = a.data.contiguous().cuda() if args.dtype_a == "float4" else a.contiguous().cuda()
    if args.dtype_b == "float4":
        b_d = b.data.T.contiguous().cuda() if args.transpose_b else b.data.contiguous().cuda()
    else:
        b_d = b.T.contiguous().cuda() if args.transpose_b else b.contiguous().cuda()
    a_scale_d = a_scale_input.cuda() if a_scale_input is not None else None
    b_scale_d = b_scale_input.cuda()

    benchmark = None if args.benchmark_mode == "none" else args.benchmark_mode
    mxgemm_tdm_pipelined(
        a_d,
        b_d,
        a_scale_d,
        b_scale_d,
        args.BM,
        args.BN,
        args.BK,
        args.transpose_b,
        args.num_buffers,
        DTYPE_TO_TRITON[args.dtype_a],
        DTYPE_TO_TRITON[args.dtype_b],
        args.scale_preshuffled,
        args.with_a_scale,
        args.schedule,
        args.l2_prefetch_distance,
        args.M,
        args.N,
        args.K,
        "partial" if args.partial_tdm else args.tdm_fusion,
        args.tdm_split,
        args.group_size_m,
        benchmark,
        args.benchmark_num_iters,
        PERSISTENT=args.persistent,
        NUM_PROGRAMS=args.num_programs,
        CROSS_TILE_PREFETCH=args.cross_tile_prefetch,
        OUTPUT_STAGING=args.output_staging,
        SCHED_MODE_2=args.sched_mode_2,
        XCD_REMAP=args.xcd_remap,
        NUM_XCDS=args.num_xcds,
        XCD_CHUNK=args.xcd_chunk,
        CLUSTER_SIZE=args.cluster_size,
        CLUSTER_MULTICAST=args.cluster_multicast,
        CLUSTER_BARRIER_INTERVAL=args.cluster_barrier_interval,
        NUM_WARPS=args.num_warps,
        REGISTER_PIPELINE=args.register_pipeline,
        WARP_PIPELINE=args.warp_pipeline,
        OPERAND_PIPELINE=args.operand_pipeline,
    )
