// RUN: triton-opt %s -split-input-file -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @global_prefetch_constant_pointer(%ptr: !tt.ptr<f16, "constant">) {
    // expected-error @+1 {{expects global memory pointers}}
    amdg.global_prefetch %ptr, speculative = true : !tt.ptr<f16, "constant">
    tt.return
  }
}
