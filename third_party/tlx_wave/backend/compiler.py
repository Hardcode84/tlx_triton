from typing import Any

from triton import knobs
from triton._C.libtriton import amd, ir, passes, tlx
import triton.backends.amd.compiler as amd_compiler
from triton.backends.compiler import GPUTarget, Language

from .converter import pipeline as converter_pipeline
from .wave_bridge_tools import (
    _compile_wave_module_to_hsaco,
    _verify_wave_module,
    _wave_opt,
    _wave_opt_sha256,
)


def _module_has_new_bridge_blocker(mod):
    found = False

    def visit(op):
        nonlocal found
        name = op.get_name()
        segments = op.get_int_array_attr("operandSegmentSizes")
        if name == "ttg.async_copy_global_to_local" and segments is not None:
            if len(segments) >= 4 and int(segments[3]):
                found = True
                return False
        if name == "amdg.buffer_load_to_local" and segments is not None:
            if len(segments) >= 6 and (
                int(segments[3]) or int(segments[4]) or int(segments[5])
            ):
                found = True
                return False
        return True

    mod.walk(visit)
    return found


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
        self.binary_ext = "hsaco"

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
        if amd_compiler.is_async_copy_enabled(options.arch):
            amd.passes.ttgpuir.add_coalesce_async_copy(pm, options.arch)
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
        if not _module_has_new_bridge_blocker(mod):
            amd.passes.ttgpuir.add_optimize_epilogue(pm)
            amd.passes.ttgpuir.add_optimize_dot_operands(pm, options.arch)
            amd.passes.ttgpuir.add_hoist_layout_conversions(pm)
            amd.passes.ttgpuir.add_sink_layout_conversions(pm)
            if knobs.amd.use_buffer_ops:
                amd.passes.ttgpuir.add_canonicalize_pointers(pm)
                passes.common.add_canonicalizer(pm)
                amd.passes.ttgpuir.add_convert_to_buffer_ops(
                    pm,
                    options.arch,
                    knobs.amd.use_buffer_atomics,
                    knobs.amd.buffer_ops_analyze_small_tensor_range,
                )
                amd.passes.ttgpuir.add_optimize_buffer_op_ptr(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        pm.run(mod, "tlx_wave.make_ttgir_post_cf_lift")
        metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    @staticmethod
    def make_wave(src, metadata, options):
        output = converter_pipeline.convert_ttgir_to_wave(src)
        _validate_staged_converter_output(output, options)

        wave_text = output.emitted_module.text
        wave_opt = _wave_opt()
        _verify_wave_module(wave_text, wave_opt)
        _populate_staged_converter_metadata(metadata, output, options, wave_opt)
        return wave_text

    @staticmethod
    def make_hsaco(src, metadata, options):
        wave_opt = _wave_opt()
        hsaco = _compile_wave_module_to_hsaco(src, wave_opt)
        metadata["tlx_wave_binary_stage"] = "wave-compile-kernels"
        metadata["tlx_wave_hsaco_size_bytes"] = len(hsaco)
        return hsaco

    def add_stages(self, stages, options, language):
        if language != Language.TRITON:
            raise NotImplementedError("tlx_wave scaffold currently supports only Triton/TLX language input")
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options)
        stages["wave"] = lambda src, metadata: self.make_wave(src, metadata, options)
        stages["hsaco"] = lambda src, metadata: self.make_hsaco(src, metadata, options)
        if knobs.runtime.add_stages_inspection_hook is not None:
            knobs.runtime.add_stages_inspection_hook(self, stages, options, language, None)

    def hash(self):
        return (
            f"{self.target}:stage7-staged-converter-hsaco:"
            f"wave-opt-sha256={_wave_opt_sha256()}"
        )


def _validate_staged_converter_output(output, options):
    kernel = output.source_program.kernel
    expected_target = f"hip:{options.arch}"
    if kernel.target != expected_target:
        raise ValueError(
            f"tlx_wave staged converter expected TTGIR target {expected_target}, "
            f"got {kernel.target}"
        )
    if int(kernel.threads_per_warp or options.warp_size) != int(options.warp_size):
        raise ValueError(
            "tlx_wave staged converter saw inconsistent wave size: "
            f"TTGIR={kernel.threads_per_warp}, options={options.warp_size}"
        )


def _populate_staged_converter_metadata(metadata, output, options, wave_opt):
    source_kernel = output.source_program.kernel
    target_program = output.target_program
    emitted = output.emitted_module
    wave_text = emitted.text
    target_values = {
        value.target_value_id: value for value in target_program.values
    }
    arg_types = [
        target_values[target_id].type
        for target_id in target_program.kernel.arg_target_ids
    ]

    metadata["name"] = target_program.kernel.name
    metadata["shared"] = emitted.lds_size
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = "emitted_wave_staged_converter"
    metadata["tlx_wave_bridge_stage"] = "staged-converter"
    metadata["tlx_wave_wave_builder"] = "staged-converter"
    metadata["tlx_wave_emit_api"] = "structural-python"
    metadata["tlx_wave_plan_kind"] = "staged-converter"
    metadata["tlx_wave_arch"] = options.arch
    metadata["tlx_wave_ttgir_target"] = source_kernel.target
    metadata["tlx_wave_num_ctas"] = int(source_kernel.num_ctas or 1)
    metadata["tlx_wave_num_warps"] = int(source_kernel.num_warps or 1)
    metadata["tlx_wave_threads_per_warp"] = int(
        source_kernel.threads_per_warp or options.warp_size
    )
    metadata["tlx_wave_num_kernel_args"] = len(target_program.kernel.arg_target_ids)
    metadata["tlx_wave_num_pointer_args"] = sum(
        1 for target_type in arg_types if target_type.kind == "pointer"
    )
    metadata["tlx_wave_num_scalar_args"] = sum(
        1 for target_type in arg_types if target_type.kind == "scalar"
    )
    metadata["tlx_wave_plan_num_values"] = len(target_program.values)
    metadata["tlx_wave_plan_num_ops"] = len(target_program.ops)
    metadata["tlx_wave_lds_size_bytes"] = emitted.lds_size
    metadata["tlx_wave_num_mmas"] = wave_text.count("waveamd.mma")
    metadata["tlx_wave_num_wave_joins"] = wave_text.count("wave.join")
    metadata["tlx_wave_num_async_waits"] = wave_text.count("wave.wait")
    metadata["tlx_wave_num_dma_load_lds"] = wave_text.count("waveamd.dma_load_lds")
    metadata["tlx_wave_wave_opt"] = wave_opt
