"""Shared-memory base addresses whose constant does not fit the 16-bit ds_* offset
stay loop-invariant on gfx1250.

LICM hoists `@global_smem + C + laneOffset` out of a loop, but CodeGenPrepare
sinks it back next to every LDS access to fold it into the addressing mode; for
C >= 64 KiB the fold fails and each iteration recomputes the address with a VALU
add. The AMD backend's anchor-LDS-bases pass keeps one frozen base per lane
offset instead, so the loop only adds small immediates.
"""

import re

import triton
import triton.knobs as knobs
import triton.language as tl
import triton.language.extra.tlx as tlx
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile

TARGET = GPUTarget("hip", "gfx1250", 32)
ROWS, COLS = 64, 128  # 32 KiB of FP32 per slot; slot 2 starts at 64 KiB


@triton.jit
def kernel(X, Y, n, ROWS: tl.constexpr, COLS: tl.constexpr):
    offs = tl.arange(0, ROWS)[:, None] * COLS + tl.arange(0, COLS)[None, :]
    buf = tlx.local_alloc((ROWS, COLS), tl.float32, 3)
    acc = tl.load(X + offs)
    for _ in range(n):
        tlx.local_store(tlx.local_view(buf, 2), acc)
        acc = acc + tlx.local_load(tlx.local_view(buf, 2))
    tl.store(Y + offs, acc)


def compile_kernel(anchor):
    with knobs.amd.scope():
        knobs.amd.anchor_lds_bases = anchor
        src = ASTSource(fn=kernel, signature={"X": "*fp32", "Y": "*fp32", "n": "i32"},
                        constexprs={"ROWS": ROWS, "COLS": COLS})
        return triton_compile(src, target=TARGET, options={"num_warps": 4})


LARGE_GEP = re.compile(r"(%[\w.]+) = getelementptr inbounds(?: nuw)? i8, ptr addrspace\(3\) "
                       r"getelementptr \(i8, ptr addrspace\(3\) @global_smem, i32 (\d+)\), i32 %")


def large_offset_geps(llir):
    """LDS GEPs of `@global_smem + C + laneOffset` with C beyond the ds_* offset range."""
    return [name for name, c in LARGE_GEP.findall(llir) if int(c) >= 1 << 16]


def test_anchor_lds_bases_freezes_large_offset_bases():
    off = compile_kernel(False).asm["llir"]
    on = compile_kernel(True).asm["llir"]
    assert large_offset_geps(off), "expected LDS accesses at offsets of 64 KiB and more"
    assert "lds.anchor.fz = freeze" in on
    # Every remaining large-offset GEP is an anchor start; accesses use the frozen anchors.
    assert all(name.startswith("%lds.anchor") for name in large_offset_geps(on))
