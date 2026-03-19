"""
AMD GEMM V4 BT - Global Prefetch Pipeline with B pre-transposed (TLX)

Variant of amd_gemm_v4 where B is passed as B^T (N*K layout).

Instead of loading B as (K, N) and needing ds_read_b64_tr_b16 to
transpose during the LDS read, we load B^T with swapped indexing
so data lands directly in a (K, N) shaped LDS buffer with K as the
fast dimension. The AMD local_load lowering checks the shared memory
order and skips the transposing read when K is already contiguous
(order[0] == kDim).

No memdesc_trans, no ds_read_tr -- just regular ds_read_b128.
"""
import pytest
import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton._internal_testing import is_cuda, is_hip

DEVICE = triton.runtime.driver.active.get_active_torch_device()


B_K_CONTIGUOUS_LAYOUT = tlx.swizzled_shared_layout_encoding(
    vectorSize=1, perPhase=1, maxPhase=1,
    order=[0, 1],
    numCTAs=[1, 1], numCTAsPerCGA=[1, 1],
    numCTASplit=[1, 1], numCTAOrder=[1, 0],
)


@triton.jit
def matmul_kernel_v4_bt(
    a_ptr, bt_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_btn, stride_btk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    B_LAYOUT: tl.constexpr = B_K_CONTIGUOUS_LAYOUT,
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_btn > 0)
    tl.assume(stride_btk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    # A is M*K, loaded as (BLOCK_M, BLOCK_K) tiles.
    buffers_A = tlx.local_alloc((BLOCK_M, BLOCK_K), tlx.dtype_of(a_ptr), 2)
    # B^T is N*K in memory. We allocate (BLOCK_K, BLOCK_N) with order=[0,1]
    # (K-contiguous) so that data lands with K as the fast dimension in LDS.
    # This tells the AMD lowering to use regular ds_read instead of ds_read_tr.
    buffers_B = tlx.local_alloc((BLOCK_K, BLOCK_N), tlx.dtype_of(bt_ptr), 2, layout=B_LAYOUT)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A: standard (M, K) pointer tensor.
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B^T is (N, K) in memory. We index as (K, N) for the load:
    #   row dim = offs_k, stepping by stride_btk (= 1 for contiguous B^T)
    #   col dim = offs_bn, stepping by stride_btn (= K for contiguous B^T)
    b_ptrs = bt_ptr + offs_k[:, None] * stride_btk + offs_bn[None, :] * stride_btn

    iterMax = tl.cdiv(K, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- Prologue: async copy iteration 0 -> buffer 0 ---
    smem_a0 = tlx.local_view(buffers_A, 0)
    smem_b0 = tlx.local_view(buffers_B, 0)
    tok_a = tlx.async_load(a_ptrs, smem_a0, mask=offs_k[None, :] < K)
    tok_b = tlx.async_load(b_ptrs, smem_b0, mask=offs_k[:, None] < K)
    tlx.async_load_commit_group([tok_a, tok_b])
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_btk

    # --- Main loop: double-buffered async copy + compute ---
    for k in tl.range(0, iterMax - 1, num_stages=0):
        l_idx = k % 2
        g_idx = 1 - l_idx

        smem_ag = tlx.local_view(buffers_A, g_idx)
        smem_bg = tlx.local_view(buffers_B, g_idx)
        tok_a = tlx.async_load(a_ptrs, smem_ag, mask=offs_k[None, :] < K - (k + 1) * BLOCK_K)
        tok_b = tlx.async_load(b_ptrs, smem_bg, mask=offs_k[:, None] < K - (k + 1) * BLOCK_K)
        tlx.async_load_commit_group([tok_a, tok_b])

        tlx.async_load_wait_group(1)

        smem_al = tlx.local_view(buffers_A, l_idx)
        smem_bl = tlx.local_view(buffers_B, l_idx)
        a = tlx.local_load(smem_al)
        b = tlx.local_load(smem_bl)
        acc = tl.dot(a, b, acc)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_btk

    # --- Epilogue: drain last buffer ---
    tlx.async_load_wait_group(0)
    l_idx = (iterMax - 1) % 2
    smem_al = tlx.local_view(buffers_A, l_idx)
    smem_bl = tlx.local_view(buffers_B, l_idx)
    a = tlx.local_load(smem_al)
    b = tlx.local_load(smem_bl)
    acc = tl.dot(a, b, acc)

    c = acc.to(tlx.dtype_of(c_ptr))

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    """C = A @ B, but B is passed as B^T (N*K) for optimal LDS layout."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    num_warps = 4

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # Pre-transpose B for optimal LDS layout.
    bt = b.T.contiguous()

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)

    matmul_kernel_v4_bt[grid](
        a, bt, c,
        M, N, K,
        a.stride(0), a.stride(1),
        bt.stride(0), bt.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return c


@pytest.mark.skipif(
    not is_hip(),
    reason="Requires AMD GPU",
)
def test_op():
    torch.manual_seed(0)
    a = torch.randn((4096, 4096), device=DEVICE, dtype=torch.float16)
    b = torch.randn((4096, 4096), device=DEVICE, dtype=torch.float16)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a, b)
    rtol = 1e-2
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol)
    print("Correctness test passed!")


if __name__ == "__main__":
    test_op()
