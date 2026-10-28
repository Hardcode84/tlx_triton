"""Meta LLVM fork builds for facebookexperimental/triton.

cmake/llvm-info.json is OAI's pin and stays byte-identical to upstream, so
upstream picks apply cleanly. cmake/meta-llvm-info.json, when present, names
the Meta LLVM fork build, which must sit on OAI's pin. build_helpers.py calls
in here from one hook; keep Meta's dependency logic in this file.
"""

import json
import os
from typing import Optional

try:
    from . import build_helpers as bh
except ImportError:  # build_helpers.py runs as a script
    import build_helpers as bh


def enabled(base_dir: str) -> bool:
    return os.path.exists(os.path.join(base_dir, "cmake", "meta-llvm-info.json"))


def load_llvm_info(base_dir: Optional[str] = None) -> dict:
    # INVARIANT: compare against llvm-info.json, the LLVM the Triton sources
    # are written against. llvm-build-info.json only requests the next OAI
    # build and moves days before the sources do.
    cmake_dir = os.path.join(base_dir or bh.get_base_dir(), "cmake")
    with open(os.path.join(cmake_dir, "llvm-info.json"), "r") as llvm_info_file:
        oai_info = json.load(llvm_info_file)
    with open(os.path.join(cmake_dir, "meta-llvm-info.json"), "r") as meta_info_file:
        meta_info = json.load(meta_info_file)
    if meta_info["based_on_oai_llvm_hash"] != oai_info["llvm_hash"]:
        raise RuntimeError(f"cmake/meta-llvm-info.json is based on OAI LLVM {meta_info['based_on_oai_llvm_hash'][:12]} "
                           f"but cmake/llvm-info.json pins {oai_info['llvm_hash'][:12]}; "
                           "rebuild the Meta LLVM on the new OAI pin")
    return meta_info


def llvm_package_for_suffix(system_suffix: str, base_dir: Optional[str] = None) -> "bh.Package":
    llvm_info = load_llvm_info(base_dir)
    sha256sums = llvm_info["sha256sum"]
    if system_suffix not in sha256sums:
        raise RuntimeError(f"LLVM pre-compiled image is not available for {system_suffix} "
                           f"(published: {', '.join(sorted(sha256sums))}); set LLVM_SYSPATH to a local LLVM build")
    name = f"llvm-{llvm_info['llvm_hash'][:8]}-{system_suffix}-{llvm_info['build_number']}"
    return bh.Package(
        "llvm",
        name,
        f"{llvm_info['release_base_url'].rstrip('/')}/{name}.tar.gz",
        "LLVM_INCLUDE_DIRS",
        "LLVM_LIBRARY_DIR",
        "LLVM_SYSPATH",
        # Stable symlink that doesn't include the revision.
        sym_name=f"llvm-{system_suffix}",
        sha256sum=sha256sums[system_suffix],
    )


def llvm_package(helper_args: "bh.BuildHelperArgs") -> "bh.Package":
    system_suffix = bh.get_llvm_system_suffix(helper_args)
    if system_suffix is None:
        return bh.Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
    try:
        return llvm_package_for_suffix(system_suffix)
    except RuntimeError as error:
        if helper_args.llvm_syspath is None:
            raise
        print(f"{error}. Proceeding with LLVM_SYSPATH={helper_args.llvm_syspath}.")
        return bh.Package("llvm", "LLVM-C.lib", "", "LLVM_INCLUDE_DIRS", "LLVM_LIBRARY_DIR", "LLVM_SYSPATH")
