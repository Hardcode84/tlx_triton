import json

from .wave_bridge_emit import (
    _emit_accumulator_fragment,
    _emit_wave_skeleton,
    _verify_wave_module,
    _wave_opt,
)
from .wave_bridge_plan import (
    _GFX950_F16_MMA_KIND,
    _GFX950_MMA_SHAPE,
    _bridge_plan_metadata,
    _bridge_stage,
    _build_bridge_plan,
    _entry_name,
    _kernel_from_module,
    _module_attrs,
    _status_for_stage,
    _validate_compute_support,
    _validate_target,
)


def stop_before_wave_lowering(mod, metadata, options):
    """Emit the first Wave/WaveAMD module from cutoff TTGIR.

    This stage preserves the public kernel ABI and lowers the supported
    TTGIR graph prefix into Wave/WaveAMD memory-token operations. For the
    supported dot/MMA op path it also lowers dot-operand local loads and
    f16/f32 tt.dot SSA flow into gfx950 WaveAMD MFMA fragments.
    """
    attrs = _module_attrs(mod)
    _validate_target(options, attrs)
    kernel = _kernel_from_module(mod)
    plan = _build_bridge_plan(mod, kernel)
    _validate_compute_support(attrs, plan)
    bridge_stage = _bridge_stage(plan)

    metadata["name"] = kernel.name or _entry_name(mod)
    metadata["shared"] = 0
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = _status_for_stage(bridge_stage)
    metadata["tlx_wave_bridge_stage"] = bridge_stage
    metadata["tlx_wave_arch"] = options.arch
    metadata["tlx_wave_ttgir_target"] = attrs.target
    metadata["tlx_wave_num_warps"] = attrs.num_warps
    metadata["tlx_wave_threads_per_warp"] = attrs.threads_per_warp
    metadata["tlx_wave_num_ctas"] = attrs.num_ctas
    metadata["tlx_wave_num_kernel_args"] = len(kernel.args)
    metadata["tlx_wave_num_pointer_args"] = sum(
        arg.kind == "pointer" for arg in kernel.args
    )
    metadata["tlx_wave_num_scalar_args"] = sum(
        arg.kind == "scalar" for arg in kernel.args
    )
    metadata["tlx_wave_plan_kind"] = plan.kind
    metadata["tlx_wave_plan_num_ops"] = len(plan.ops)
    metadata["tlx_wave_plan_num_values"] = len(plan.values)
    metadata["tlx_wave_plan_num_addresses"] = len(plan.addresses)
    metadata["tlx_wave_plan_num_memdescs"] = len(plan.memdescs)
    metadata["tlx_wave_plan_num_layouts"] = len(plan.layouts)
    metadata["tlx_wave_plan_num_layout_constraints"] = len(plan.layout_constraints)
    metadata["tlx_wave_plan_num_storage_aliases"] = len(plan.storage_aliases)
    metadata["tlx_wave_plan_num_tokens"] = len(plan.tokens)
    metadata["tlx_wave_plan_json"] = json.dumps(
        _bridge_plan_metadata(plan), sort_keys=True
    )
    wave_text, builder, wave_stats = _emit_wave_skeleton(kernel, attrs, plan)
    metadata["shared"] = wave_stats.lds_size_bytes
    metadata["tlx_wave_lds_size_bytes"] = wave_stats.lds_size_bytes
    metadata["tlx_wave_num_async_copies"] = wave_stats.async_copies
    metadata["tlx_wave_num_dma_load_lds"] = wave_stats.dma_load_lds
    metadata["tlx_wave_num_load_store_fallbacks"] = wave_stats.load_store_fallbacks
    metadata["tlx_wave_num_async_commit_groups"] = wave_stats.commit_groups
    metadata["tlx_wave_num_async_waits"] = wave_stats.waits
    metadata["tlx_wave_num_wave_joins"] = wave_stats.joins
    metadata["tlx_wave_num_wave_barriers"] = wave_stats.barriers
    metadata["tlx_wave_num_wave_local_loads"] = wave_stats.local_loads
    metadata["tlx_wave_num_fragment_packs"] = wave_stats.fragment_packs
    metadata["tlx_wave_num_fragment_fills"] = wave_stats.fragment_fills
    metadata["tlx_wave_num_mmas"] = wave_stats.mmas
    wave_opt = _wave_opt()
    _verify_wave_module(wave_text, wave_opt)
    metadata["tlx_wave_wave_builder"] = builder
    metadata["tlx_wave_wave_opt"] = wave_opt
    return wave_text
