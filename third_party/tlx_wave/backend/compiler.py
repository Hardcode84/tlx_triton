import functools
from typing import Any

from triton import knobs
from triton._C.libtriton import amd, ir, passes, tlx
import triton.backends.amd.compiler as amd_compiler
from triton.backends.compiler import GPUTarget, Language

from . import wave_bridge


class TLXWaveBackend(amd_compiler.HIPBackend):
    """TLX-first AMD Wave backend scaffold.

    This backend deliberately reuses the HIP/TLX frontend and TTIR/TTGIR
    conversion path, then emits the current Wave IR handoff artifact.  The
    TTGIR-to-Wave bridge lives in Python so it can use Wave's own Python package
    and LLVM build without linking the two MLIR worlds in-process.
    """

    supports_native_tensor_specialization = False

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "tlx_wave"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "wave"

    def get_target_name(self, options) -> str:
        return f"tlx_wave:{options.arch}"

    def parse_options(self, opts) -> Any:
        opts = dict(opts)
        opts["backend_name"] = "tlx_wave"
        # Match the AMD/HIP backend contract: matrix_instr_nonkdim=0 leaves
        # Triton's AMD matmul pass free to derive the MFMA shape from the tile.
        # The bridge validates the concrete TTGIR layout before emitting Wave IR.
        options = super().parse_options(opts)
        if options.arch not in {"gfx942", "gfx950"}:
            raise ValueError(
                f"tlx_wave stage-1 scaffold only supports gfx942/gfx950, got {options.arch}"
            )
        if options.warp_size != 64:
            raise ValueError(
                f"tlx_wave {options.arch} expects wave64, got warp_size={options.warp_size}"
            )
        return options

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        tlx.tlx_passes.add_triton_tlx_fixup(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                            options.num_ctas, list((1, 1, 1)))
        passes.common.add_inliner(pm)
        # gfx942/gfx950 have no TDM path in this design; keep tensor descriptors on the
        # ordinary pointer path for the future Wave bridge.
        if not amd.supports_tdm(options.arch):
            passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        # Preserve TLX aliasing intent in concrete local_alias/memdesc form.
        tlx.tlx_passes.add_tlx_storage_alias_lowering(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, "tlx_wave.make_ttir")
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        # Continue to target hip:gfx950 for TTGIR construction because TritonGPU
        # and TLX passes use that target spelling to choose AMD-compatible
        # encodings before the Wave bridge takes over.
        passes.ttir.add_convert_to_ttgpuir(pm, f"hip:{options.arch}", options.num_warps, options.warp_size,
                                           options.num_ctas)
        pm.run(mod, "tlx_wave.make_ttgir_early")

        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttgpuir.add_coalesce(pm)
        passes.ttgpuir.add_f32_dot_tc(pm, False)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_thread_locality(pm)
        # Keep the HIP/TLX dot layout contract intact for the Wave bridge:
        # accelerated dot metadata drives require_layout insertion, propagation
        # retags the local_alloc/local_load chain, and the final cleanup removes
        # layout conversions left behind by the propagation step.
        amd.passes.ttgpuir.add_accelerate_matmul(pm, options.arch, options.matrix_instr_nonkdim, options.kpack)
        tlx.tlx_passes.add_tlx_insert_require_layout(pm)
        tlx.tlx_passes.add_tlx_propagate_layout(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "tlx_wave.make_ttgir")

        passes.convert.triton_lift_cf_to_scf(mod)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_canonicalizer(pm)
        tlx.tlx_passes.add_tlx_propagate_layout(pm)
        tlx.tlx_passes.add_tlx_rewrite_local_alias(pm)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "tlx_wave.make_ttgir_post_cf_lift")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_wave(src, metadata, options):
        return wave_bridge.stop_before_wave_lowering(src, metadata, options)

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("tlx_wave scaffold currently supports only Triton/TLX language input")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["wave"] = lambda src, metadata: self.make_wave(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    @functools.lru_cache()
    def hash(self):
        return f"{self.target}:stage3-ttgir-graph-plan"
