"""
Batched MXFP8 GEMM for AMD gfx1250 using TLX: C[b] = A[b] @ B[b]^T.

Targets small-M batched products such as B=128, M=64, N=8192, K=8192, where
every batch has its own weight matrix. Each weight byte is used by one M tile
only, so the kernel streams B through a deep TDM ring and the WMMAs hide behind
the loads:

* One output tile is a whole batch's M rows by BLOCK_N columns. Folding the
  batch into the rows (A as [B*M, K], B as [B*N, K], C as [B*M, N]) keeps every
  tile inside one batch; tile (b, n) reads A rows b*M and B rows b*N + n*BLOCK_N.
* Programs are persistent over tiles ordered batch-major, so the programs that
  run one batch's N tiles at the same time share its A rows through L2.
* Loads run over the flattened (tile, K step) sequence of a program, so the
  next tile's first stages are in flight while the current tile finishes.
* B scales use the 128-row preshuffled layout (pack_scale); A scales stay
  row-major because M=64 does not fill a 128-row block.
* B and its scales load with a cache bypass (FUSION="split", B_CACHE=".cv")
  and A with the default policy, about 1% faster than one policy for both.
* C goes through LDS (STAGE_C). Stored straight from the WMMA layout, every
  store instruction writes 16 bytes into each of 32 rows; restaged, a wave
  writes whole rows, which took about 3% off the 128x64x8192x8192 problem.

``mxgemm_batched_split_n`` runs one problem on several devices (for example the
two gfx1250 DPX partitions) by splitting N: every device holds its own N shard
of B and its scales, a copy of A, and computes its N shard of C.

Run ``python amd_mxfp_batched_gemm.py --check`` for the default 128x64x8192x8192
problem on one device or ``--devices 0,1`` to split N across two.
"""
import argparse
import statistics
import threading

import torch

import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

SCALE_BLOCK = 32


@triton.constexpr_function
def _operand_layout(block_shape):
    # 16 bytes of padding per 256-byte row piece avoids LDS bank conflicts for
    # the WMMA operand reads (as in the persistent MXFP kernel).
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 16]], block_shape, [1, 0])


@triton.constexpr_function
def _scale_layout(block_shape):
    return tlx.padded_shared_layout_encoding.with_identity_for([[256, 8]], block_shape, [1, 0])


@triton.jit
def _position(desc, offsets, pred):
    return tlx.update_tensor_descriptor(desc, add_offsets=offsets, clamp_bounds=False, pred=pred)


@triton.jit
def _issue_step(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, step, total, pid, M: tl.constexpr,
                N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, K_ITERS: tl.constexpr,
                TILES_N: tl.constexpr, NUM_BUFFERS: tl.constexpr, NUM_PROGRAMS: tl.constexpr, FUSION: tl.constexpr,
                B_CACHE: tl.constexpr):
    # Load K step `step` of this program's flattened (tile, K step) sequence.
    # Past the end the loads are predicated off; they still count towards
    # TENSORcnt, so the ring's waits stay uniform.
    pred = step < total
    tile = pid + (step // K_ITERS) * NUM_PROGRAMS
    k = step % K_ITERS
    batch = tile // TILES_N
    row_b = batch * N + (tile % TILES_N) * BLOCK_N
    slot = step % NUM_BUFFERS
    BKS: tl.constexpr = BLOCK_K // 32
    a_off = [batch * M, k * BLOCK_K]
    b_off = [row_b, k * BLOCK_K]
    as_off = [batch * M, k * BKS]
    bs_off = [row_b // 128, k * BKS * 128]
    if FUSION == "split":
        # Two per wave and step: B with its scales, then A with its scales. B
        # is read once, so it can bypass the caches (B_CACHE) while A, which
        # every N tile of its batch reads, keeps the default policy.
        tlx.async_amd_descriptor_load_fused([
            (_position(b_desc, b_off, pred), tlx.local_view(b_buf, slot), 3),
            (_position(bs_desc, bs_off, pred), tlx.local_view(bs_buf, slot), 12),
        ], cache_modifier=B_CACHE)
        tlx.async_amd_descriptor_load_fused([
            (_position(a_desc, a_off, pred), tlx.local_view(a_buf, slot), 3),
            (_position(as_desc, as_off, pred), tlx.local_view(as_buf, slot), 12),
        ])
    elif FUSION == "4way":
        # One TDM instruction per wave and step: each wave fills one member.
        tlx.async_amd_descriptor_load_fused([
            (_position(a_desc, a_off, pred), tlx.local_view(a_buf, slot), 1),
            (_position(b_desc, b_off, pred), tlx.local_view(b_buf, slot), 2),
            (_position(as_desc, as_off, pred), tlx.local_view(as_buf, slot), 4),
            (_position(bs_desc, bs_off, pred), tlx.local_view(bs_buf, slot), 8),
        ])
    elif FUSION == "partial":
        # Two per wave and step: payloads, then scales, each split A | B.
        tlx.async_amd_descriptor_load_fused([
            (_position(a_desc, a_off, pred), tlx.local_view(a_buf, slot), 5),
            (_position(b_desc, b_off, pred), tlx.local_view(b_buf, slot), 10),
        ])
        tlx.async_amd_descriptor_load_fused([
            (_position(as_desc, as_off, pred), tlx.local_view(as_buf, slot), 5),
            (_position(bs_desc, bs_off, pred), tlx.local_view(bs_buf, slot), 10),
        ])
    else:
        tl.static_assert(FUSION == "none")
        tlx.async_amd_descriptor_load(as_desc, tlx.local_view(as_buf, slot), as_off, pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(bs_desc, tlx.local_view(bs_buf, slot), bs_off, pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(a_desc, tlx.local_view(a_buf, slot), a_off, pred=pred, clamp_bounds=False)
        tlx.async_amd_descriptor_load(b_desc, tlx.local_view(b_buf, slot), b_off, pred=pred, clamp_bounds=False)


@triton.jit
def mxgemm_batched_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale,
    b_scale,
    BATCH: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_BUFFERS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    TILES_PER_WARP: tl.constexpr = (4, 4),
    FUSION: tl.constexpr = "split",
    B_CACHE: tl.constexpr = ".cv",
    STAGE_C: tl.constexpr = True,
):
    """A [BATCH*M, K] e4m3, B [BATCH*N, K] e4m3 (K-contiguous, i.e. transposed),
    A scales [BATCH*M, K/32] e8m0, B scales preshuffled [BATCH*N/128, K/32*128],
    C [BATCH*M, N]."""
    tl.static_assert(M % 16 == 0 and M <= 128, "M must be a multiple of 16, at most 128")
    tl.static_assert(BLOCK_N % 128 == 0 and N % BLOCK_N == 0 and BLOCK_K % 128 == 0 and K % BLOCK_K == 0)
    tl.static_assert(NUM_BUFFERS >= 2 and K // BLOCK_K >= NUM_BUFFERS - 1)
    BKS: tl.constexpr = BLOCK_K // 32
    # TDM rows are at least 16 bytes; read 16 A-scale columns per stage and use
    # the stage's first BKS.
    BKS_LOAD: tl.constexpr = 16 if BKS < 16 else BKS
    K_ITERS: tl.constexpr = K // BLOCK_K
    TILES_N: tl.constexpr = N // BLOCK_N
    NUM_TILES: tl.constexpr = BATCH * TILES_N
    # TDM instructions per wave and K step. gfx1250 codegen waits for at most
    # ten outstanding TDM instructions before issuing another, which bounds the
    # ring depth that is actually in flight: fusing keeps all stages in flight.
    LOADS: tl.constexpr = 1 if FUSION == "4way" else 2 if FUSION == "partial" or FUSION == "split" else 4

    pid = tl.program_id(0)
    my_tiles = (NUM_TILES - 1 - pid) // NUM_PROGRAMS + 1
    total = my_tiles * K_ITERS

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[BATCH * M, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[BATCH * N, K], strides=[K, tl.constexpr(1)],
                                       block_shape=[BLOCK_N, BLOCK_K])
    as_desc = tl.make_tensor_descriptor(a_scale, shape=[BATCH * M, K // 32], strides=[K // 32, tl.constexpr(1)],
                                        block_shape=[M, BKS_LOAD])
    bs_desc = tl.make_tensor_descriptor(b_scale, shape=[BATCH * N // 128,
                                                        K // 32 * 128], strides=[K // 32 * 128,
                                                                                 tl.constexpr(1)],
                                        block_shape=[BLOCK_N // 128, BKS * 128])
    a_buf = tlx.local_alloc((M, BLOCK_K), tlx.dtype_of(a_ptr), NUM_BUFFERS, layout=_operand_layout([M, BLOCK_K]))
    b_buf = tlx.local_alloc((BLOCK_N, BLOCK_K), tlx.dtype_of(b_ptr), NUM_BUFFERS,
                            layout=_operand_layout([BLOCK_N, BLOCK_K]))
    as_buf = tlx.local_alloc((M, BKS_LOAD), tlx.dtype_of(a_scale), NUM_BUFFERS, layout=_scale_layout([M, BKS_LOAD]))
    bs_buf = tlx.local_alloc((BLOCK_N // 128, BKS * 128), tlx.dtype_of(b_scale), NUM_BUFFERS,
                             layout=_scale_layout([BLOCK_N // 128, BKS * 128]))

    if STAGE_C:
        c_view = tlx.local_view(tlx.local_alloc((M, BLOCK_N), tlx.dtype_of(c_ptr), 1), 0)

    for j in tl.static_range(NUM_BUFFERS - 1):
        _issue_step(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, j, total, pid, M, N, BLOCK_N,
                    BLOCK_K, K_ITERS, TILES_N, NUM_BUFFERS, NUM_PROGRAMS, FUSION, B_CACHE)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, BLOCK_N)
    for t in range(my_tiles):
        tile = pid + t * NUM_PROGRAMS
        batch = tile // TILES_N
        n0 = (tile % TILES_N) * BLOCK_N
        acc = tl.zeros((M, BLOCK_N), dtype=tl.float32)
        for k in range(K_ITERS):
            step = t * K_ITERS + k
            _issue_step(a_desc, b_desc, as_desc, bs_desc, a_buf, b_buf, as_buf, bs_buf, step + NUM_BUFFERS - 1, total,
                        pid, M, N, BLOCK_N, BLOCK_K, K_ITERS, TILES_N, NUM_BUFFERS, NUM_PROGRAMS, FUSION, B_CACHE)
            tlx.async_amd_descriptor_wait((NUM_BUFFERS - 1) * LOADS)
            slot = step % NUM_BUFFERS
            a = tlx.local_load(tlx.local_view(a_buf, slot))
            b = tlx.local_load(tlx.local_trans(tlx.local_view(b_buf, slot)))
            sa = tlx.local_load(tlx.local_slice(tlx.local_view(as_buf, slot), [0, 0], [M, BKS]))
            sb_view = tlx.local_reshape(tlx.local_view(bs_buf, slot), [BLOCK_N // 128, BKS // 4, 32, 4, 4])
            sb_view = tlx.local_reshape(tlx.local_trans(sb_view, (0, 3, 2, 1, 4)), [BLOCK_N, BKS])
            sb = tlx.local_load(sb_view)
            acc = tlx.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc, tiles_per_warp=TILES_PER_WARP)
        c_tile = c_ptr + (batch * M).to(tl.int64) * N + n0
        c = acc.to(c_ptr.dtype.element_ty)
        if STAGE_C:
            tlx.local_store(c_view, c)
            c = tlx.local_load(c_view)
        tl.store(c_tile + offs_m[:, None] * N + offs_n[None, :], c)
    tlx.async_amd_descriptor_wait(0)


def pack_b_scale(b_scale: torch.Tensor) -> torch.Tensor:
    """[B, N, K/32] e8m0 bytes -> preshuffled [B*N/128, K/32*128] (128-row blocks, 4 K scales per dword)."""
    batch, n, k_scale = b_scale.shape
    if n % 128 or k_scale % 4:
        raise ValueError("B scales need N % 128 == 0 and K/32 % 4 == 0")
    x = b_scale.reshape(batch * n // 128, 4, 32, k_scale // 4, 4).permute(0, 3, 2, 1, 4)
    return x.contiguous().view(batch * n // 128, k_scale * 128)


def mxgemm_batched(a, b, a_scale, b_scale_packed, out=None, *, block_n=256, block_k=256, num_buffers=3,
                   num_programs=None, out_dtype=torch.bfloat16, tiles_per_warp=(4, 4), fusion="split", b_cache=".cv",
                   stage_c=True):
    """C[b] = (A[b] * 2^(sA[b]-127)) @ (B[b] * 2^(sB[b]-127))^T for every batch b.

    a: [B, M, K] float8_e4m3fn; b: [B, N, K] float8_e4m3fn; a_scale: [B, M, K/32]
    uint8 (e8m0); b_scale_packed: pack_b_scale([B, N, K/32]). Returns [B, M, N].
    """
    batch, m, k = a.shape
    if b.shape[0] != batch or b.shape[2] != k or tuple(a_scale.shape) != (batch, m, k // SCALE_BLOCK):
        raise ValueError("batched MXFP shapes do not match")
    n = b.shape[1]
    if m % 16 or m > 128 or n % block_n or k % block_k or k // block_k < num_buffers - 1:
        raise ValueError("batched MXFP needs M % 16 == 0 (M <= 128), N % block_n == 0, K % block_k == 0, "
                         "and at least num_buffers - 1 K stages")
    if tuple(b_scale_packed.shape) != (batch * n // 128, k // SCALE_BLOCK * 128):
        raise ValueError("b_scale_packed must be pack_b_scale(b_scale)")
    for t in (a, b, a_scale, b_scale_packed):
        if not t.is_contiguous():
            raise ValueError("batched MXFP needs contiguous inputs")
    if out is None:
        out = torch.empty((batch, m, n), device=a.device, dtype=out_dtype)
    tiles = batch * (n // block_n)
    if num_programs is None:
        num_programs = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_programs = min(num_programs, tiles)
    mxgemm_batched_kernel[(num_programs, )](a.view(batch * m, k), b.view(batch * n, k), out.view(batch * m, n),
                                            a_scale.view(batch * m, k // 32), b_scale_packed, BATCH=batch, M=m, N=n,
                                            K=k, BLOCK_N=block_n, BLOCK_K=block_k, NUM_BUFFERS=num_buffers,
                                            NUM_PROGRAMS=num_programs, TILES_PER_WARP=tuple(tiles_per_warp),
                                            FUSION=fusion, B_CACHE=b_cache, STAGE_C=stage_c, num_warps=4,
                                            waves_per_eu=1)
    return out


def split_n_inputs(devices, b, b_scale):
    """Shard B [B, N, K] and its raw scales [B, N, K/32] by N over `devices`;
    returns per-device (B shard, packed scale shard)."""
    n = b.shape[1]
    if n % len(devices):
        raise ValueError("N must split evenly across devices")
    step = n // len(devices)
    return [(b[:,
               i * step:(i + 1) * step].contiguous().to(dev), pack_b_scale(b_scale[:, i * step:(i + 1) * step].to(dev)))
            for i, dev in enumerate(devices)]


def mxgemm_batched_split_n(devices, a_per_device, a_scale_per_device, shards, **kwargs):
    """Launch every device's N shard (inputs already resident); returns the per-device C shards."""
    outs = []
    for dev, a, sa, (b, sb) in zip(devices, a_per_device, a_scale_per_device, shards):
        with torch.cuda.device(dev):
            outs.append(mxgemm_batched(a, b, sa, sb, **kwargs))
    return outs


def reference(a, b, a_scale, b_scale, batches=None):
    """FP32 reference for selected batches: list of [M, N] tensors."""
    out = []
    for i in (range(a.shape[0]) if batches is None else batches):
        sa = torch.exp2(a_scale[i].float() - 127).repeat_interleave(SCALE_BLOCK, dim=1)
        sb = torch.exp2(b_scale[i].float() - 127).repeat_interleave(SCALE_BLOCK, dim=1)
        out.append((a[i].float() * sa) @ (b[i].float() * sb).T)
    return out


def _random_inputs(batch, m, n, k, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    a = torch.randint(20, 40, (batch, m, k), dtype=torch.uint8, device=device, generator=g)
    b = torch.randint(20, 40, (batch, n, k), dtype=torch.uint8, device=device, generator=g)
    sa = torch.randint(120, 130, (batch, m, k // 32), dtype=torch.uint8, device=device, generator=g)
    sb = torch.randint(120, 130, (batch, n, k // 32), dtype=torch.uint8, device=device, generator=g)
    return a.view(torch.float8_e4m3fn), b.view(torch.float8_e4m3fn), sa, sb


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-N", type=int, default=8192)
    p.add_argument("-K", type=int, default=8192)
    p.add_argument("--devices", default="0", help="comma-separated devices; N is split across them")
    p.add_argument("--block_n", type=int, default=256)
    p.add_argument("--block_k", type=int, default=256)
    p.add_argument("--num_buffers", type=int, default=3)
    p.add_argument("--num_programs", type=int, default=None)
    p.add_argument("--fusion", choices=("split", "4way", "partial", "none"), default="split")
    p.add_argument("--b_cache", default=".cv", help="cache modifier for B loads with --fusion split")
    p.add_argument("--no_stage_c", action="store_true", help="store C straight from the WMMA layout")
    p.add_argument("--check", action="store_true")
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()
    devices = [int(d) for d in args.devices.split(",")]
    batch, m, n, k = args.batch, args.M, args.N, args.K
    a, b, sa, sb = _random_inputs(batch, m, n, k, f"cuda:{devices[0]}")
    shards = split_n_inputs(devices, b, sb)
    a_dev = [a.to(d) for d in devices]
    sa_dev = [sa.to(d) for d in devices]
    kw = dict(block_n=args.block_n, block_k=args.block_k, num_buffers=args.num_buffers, num_programs=args.num_programs,
              fusion=args.fusion, b_cache=args.b_cache, stage_c=not args.no_stage_c)
    outs = mxgemm_batched_split_n(devices, a_dev, sa_dev, shards, **kw)
    for d in devices:
        torch.cuda.synchronize(d)
    if args.check:
        for bi in sorted({0, 1, batch // 2, batch - 1}):
            ref = reference(a, b, sa, sb, [bi])[0]
            got = torch.cat([o[bi].float().to(ref.device) for o in outs], dim=1)
            rel = ((got - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()
            print(f"batch {bi}: max rel err {rel:.3g}")
            assert rel < 1e-2, "output mismatch"
    times = {d: [] for d in devices}

    def run(d, i):
        with torch.cuda.device(d):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            mxgemm_batched(a_dev[i], shards[i][0], sa_dev[i], shards[i][1], out=outs[i], **kw)
            e.record()
            e.synchronize()
            times[d].append(s.elapsed_time(e))

    for _ in range(args.iters):
        threads = [threading.Thread(target=run, args=(d, i)) for i, d in enumerate(devices)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    ms = max(statistics.median(v) for v in times.values())
    flops = 2 * batch * m * n * k
    bytes_moved = batch * (n * k + n * k // 32 + m * k + m * k // 32) + batch * m * n * outs[0].element_size()
    print(f"batch={batch} M={m} N={n} K={k} devices={devices}: {ms:.3f} ms, {flops / ms / 1e9:.1f} TFLOPS, "
          f"{bytes_moved / ms / 1e9:.2f} TB/s (" +
          ", ".join(f"dev{d} {statistics.median(v):.3f}" for d, v in times.items()) + " ms)")


if __name__ == "__main__":
    main()
