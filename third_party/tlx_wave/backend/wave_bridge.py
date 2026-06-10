def _entry_name(mod):
    try:
        return mod.get_entry_func_name()
    except Exception:
        return "tlx_wave_kernel"


def stop_before_wave_lowering(mod, metadata, options):
    """Stage-1 placeholder for the Python TTGIR-to-Wave bridge.

    The real bridge will import Wave's Python package from the `third_party/wave`
    submodule and lower this TTGIR module to Wave/WaveAMD textual MLIR.  For now
    we intentionally stop at the handoff boundary and preserve the cutoff TTGIR
    as the `wave` artifact so compile/dump paths can inspect it.
    """
    metadata["name"] = _entry_name(mod)
    metadata["shared"] = 0
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1
    metadata["tlx_wave_status"] = "stopped_before_wave_lowering"
    metadata["tlx_wave_arch"] = options.arch
    return str(mod)
