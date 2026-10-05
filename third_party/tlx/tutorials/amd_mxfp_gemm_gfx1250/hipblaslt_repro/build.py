"""Build the published hipBLASLt gfx1250 A8W8 kernel with FP32 output.

Source: ROCm/rocm-libraries, hipblaslt-gfx1250-custom-kernels, pinned below.
The public 8Kx8Kx8K and 8Kx8Kx4K assembly files are byte-identical.
Only the output epilogue and kernel name/metadata change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.request

UPSTREAM_COMMIT = "74186f70c3861b4b17cbad886def36fdaf106374"
UPSTREAM_SHA256 = "70ecd7146cc22a99ce58cda7dcb1b14f601ac327162c2fc6fcf24c7e5edb6476"
UPSTREAM_PATH = ("projects/hipblaslt/tensilelite/"
                 "custom_MXFP8xMXFP8_BS1_8Kx8Kx8K_async_store_split_cluster_barrier_group_pack_spread_ds_clean_dep/"
                 "1_BenchmarkProblems/Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH_UserArgs_00/"
                 "00_Final/source/build_tmp/SOURCE/assembly/"
                 "Cijk_Alik_Bljk_F8F8S_MXAE8B32_MXBE8B32_BH_UserAryhb6oxamh7hWKFZpDAFtquh3nGS09H6NSjBxnQu3VBg=.s")
UPSTREAM_URL = f"https://raw.githubusercontent.com/ROCm/rocm-libraries/{UPSTREAM_COMMIT}/{UPSTREAM_PATH}"
DEFAULT_CACHE = Path.home() / ".cache" / "tlx-hipblaslt-mxfp-f32"
DEFAULT_LLVM = Path(os.environ.get("LLVM_SYSPATH", str(Path.home() / "llvm/llvm-build")))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def epilogue(kind, wait_count, odd_k_blocks=False):
    out = []

    def emit(s):
        out.extend(s.strip().splitlines())

    emit('''
// FP32 epilogue. D is column-major; alpha=1 and beta=0.
// Keep v0:7 and v16:23 alive: next-tile scale operands survive this epilogue.
label_FP32_output:
s_wait_dscnt 0
s_barrier_signal -1
s_barrier_wait -1
s_set_vgpr_msb 3
v_mov_b32 v11, v255
s_set_vgpr_msb 0
v_lshrrev_b32 v12, 5, v11
v_and_b32 v14, 1, v12
v_lshlrev_b32 v14, 4, v14
v_bfe_u32 v10, v11, 4, 1
v_lshl_add_u32 v14, v10, 3, v14
v_lshrrev_b32 v12, 1, v12
v_and_b32 v8, 15, v11
v_lshl_add_u32 v8, v12, 4, v8
// v14=m0 in [0,31], v8=n0 in [0,31].
s_lshl_b32 s88, s36, 2
s_lshl_b32 s14, s3, 8
s_mul_i32 s12, s14, s88
s_mul_hi_u32 s13, s14, s88
s_add_u32 s28, s24, s12
s_addc_u32 s29, s25, s13
s_lshl_b32 s12, s2, 10
s_add_u32 s28, s28, s12
s_addc_u32 s29, s29, 0
s_add_u32 s97, s97, 1
''')
    if kind == 'direct':
        emit('''
v_mul_lo_u32 v24, v8, s88
v_lshl_add_u32 v24, v14, 2, v24
s_lshl_b32 s92, s88, 5
''')
        for ns in range(8):
            for ms in range(8):
                for h in (0, 4):
                    reg = 32 + 64 * ns + 8 * ms + h
                    emit(f's_set_vgpr_msb {4*(reg//256)}')
                    emit(f'global_store_b128 v24, v[{reg%256}:{reg%256+3}], s[28:29] offset:{ms*128+h*4}')
            if ns != 7:
                emit('s_set_vgpr_msb 0\nv_add_nc_u32 v24, v24, s92')
        emit('s_wait_storecnt 0\ns_set_vgpr_msb 0')
    else:
        pitch = 1088
        slot_size = 64 * pitch
        if odd_k_blocks:
            # With an odd number of BK256 steps, each tile flips the retained
            # input half. s97 already names the next tile; use the other half.
            # s12/s13 are output temporaries; s93/s94 retain input XOR masks.
            emit('s_and_b32 s12, s97, 1\ns_cselect_b32 s12, 0, 143872')
        emit(f'''
v_mul_lo_u32 v15, v8, {pitch}
v_lshl_add_u32 v15, v14, 2, v15
// Consumers: 64 threads per column, 4 contiguous FP32 elements per thread.
v_and_b32 v12, 63, v11
v_lshlrev_b32 v12, 4, v12
v_lshrrev_b32 v10, 6, v11
v_mul_lo_u32 v24, v10, s88
v_add_nc_u32 v24, v24, v12
v_mad_u32_u24 v11, v10, {pitch}, v12
s_lshl_b32 s92, s88, 1
s_sub_u32 s92, s92, {2*pitch}
''')
        for i in range(1, 8):
            emit(f'v_add_nc_u32 v{24+i}, v{23+i}, s92')
        emit('s_lshl_b32 s92, s88, 4')
        for p in range(4):
            emit(f'label_FP32_panel_{p}:')
            if p >= 2 and (wait_count or p == 2):
                emit(f's_wait_asynccnt {wait_count}\ns_barrier_signal -1\ns_barrier_wait -1')
            slot = 143872 + (p % 2) * slot_size
            if odd_k_blocks:
                emit(f's_add_u32 s13, s12, {(p % 2) * slot_size}')
                slot = 's13'
            emit(f's_set_vgpr_msb 0\nv_add_nc_u32 v13, {slot}, v15')
            for ns in range(2):
                for ms in range(8):
                    for h in (0, 4):
                        reg = 32 + 64 * (2 * p + ns) + 8 * ms + h
                        offset = ns * 32 * pitch + ms * 128 + h * 4
                        emit(f's_set_vgpr_msb {4*(reg//256)}')
                        emit(f'ds_store_b128 v13, v[{reg%256}:{reg%256+3}] offset:{offset}')
            emit(f'''
s_set_vgpr_msb 0
v_add_nc_u32 v13, {slot}, v11
s_wait_dscnt 0
s_barrier_signal -1
s_barrier_wait -1
''')
            for group in range(4):
                for i in range(8):
                    emit(f'global_store_async_from_lds_b128 v{24+i}, v13, s[28:29] offset:{i*2*pitch}')
                emit('s_add_u32 s28, s28, s92\ns_addc_u32 s29, s29, 0')
                if group != 3:
                    emit(f'v_add_nc_u32 v13, {16*pitch}, v13')
        emit('s_set_vgpr_msb 0')
    return '\n'.join(out) + '\n\n'


def build(cache_dir=DEFAULT_CACHE, llvm=DEFAULT_LLVM, *, output="staged", k=8192):
    """Return the HSACO path and provenance for the FP32-only specialization."""
    if output not in ("staged", "direct"):
        raise ValueError("output must be staged or direct")
    if k < 768 or k % 256:
        raise ValueError("K must be a multiple of 256 and at least 768")
    cache_dir, llvm = Path(cache_dir).expanduser(), Path(llvm).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    upstream = cache_dir / "upstream.s"
    if not upstream.exists():
        data = urllib.request.urlopen(UPSTREAM_URL, timeout=30).read()
        if sha256(data) != UPSTREAM_SHA256:
            raise RuntimeError("Downloaded hipBLASLt assembly does not match the pinned SHA256")
        upstream.write_bytes(data)
    data = upstream.read_bytes()
    if sha256(data) != UPSTREAM_SHA256:
        raise RuntimeError(f"{upstream} does not match the pinned hipBLASLt assembly")
    src = data.decode()
    wait_count = 32 if output == "staged" else 0
    early_cluster_wait = output == "staged"
    odd_k_blocks = output == "staged" and k % 512 != 0
    name = f"hipblaslt_mxfp8_f32_{output}_{wait_count}"
    if early_cluster_wait:
        name += "_prebarrier"
    if odd_k_blocks:
        name += "_odd_k"
    original_name = re.search(r"^\.globl (\S+)", src, re.M)[1]
    begin = src.index("/* load store sgprs */")
    end = src.index("label_KernelEnd:", begin)
    tail = src[end:]
    output_code = epilogue(output, wait_count, odd_k_blocks)
    if early_cluster_wait:
        # Consume the outstanding cluster rendezvous before the output path
        # performs additional workgroup barriers. Retain every rendezvous.
        wait = "s_barrier_wait -3 // cluster wait for the last iteration\n"
        assert tail.count(wait) == 1
        tail = tail.replace(wait, "", 1)
        output_code = output_code.replace("label_FP32_output:\n", "label_FP32_output:\n" + wait)
    modified = (src[:begin] + output_code + tail).replace(original_name, name)
    for arg in ("D", "C"):
        modified = re.sub(
            r"(\.name: +" + arg + r"\n(?:(?!\.name:).)*?\.value_type: +)fp8",
            r"\g<1>f32",
            modified,
            flags=re.S,
        )
    path = cache_dir / name
    asm, obj, hsaco = (path.with_suffix(s) for s in (".s", ".o", ".hsaco"))
    asm.write_text(modified)
    subprocess.run([
        str(llvm / "bin/llvm-mc"), "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1250", "-filetype=obj",
        str(asm), "-o",
        str(obj)
    ], check=True)
    subprocess.run([str(llvm / "bin/ld.lld"), "-shared", str(obj), "-o", str(hsaco)], check=True)
    manifest = dict(
        name=name,
        upstream_commit=UPSTREAM_COMMIT,
        upstream_url=UPSTREAM_URL,
        upstream_sha256=UPSTREAM_SHA256,
        source_sha256=sha256(modified.encode()),
        hsaco_sha256=sha256(hsaco.read_bytes()),
        llvm=str(llvm.resolve()),
        output=output,
        async_wait_count=wait_count,
        cluster_wait_before_output=early_cluster_wait,
        staging_follows_ring=odd_k_blocks,
        resources=dict(vgprs=1024, sgprs=106, lds_bytes=287744, scratch_bytes=0),
        compute_source_unchanged=True,
        alpha=1,
        beta=0,
        output_dtype="float32",
        cluster=(4, 4, 1),
        block=(128, 1, 1),
        tiles_per_cta=4,
    )
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return hsaco, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--llvm", type=Path, default=DEFAULT_LLVM)
    parser.add_argument("--output", choices=("staged", "direct"), default="staged")
    parser.add_argument("-K", "--k", type=int, default=8192, help="K dimension for output-ring specialization")
    args = parser.parse_args()
    path, manifest = build(args.cache_dir, args.llvm, output=args.output, k=args.k)
    print(json.dumps(dict(hsaco=str(path), **manifest), indent=2))
