// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // One prefetch per element each thread holds, at SE scope; the
  // non-speculative form sets the low hint bit.
  // CHECK-LABEL: llvm.func @global_prefetch_tensor
  // CHECK-COUNT-2: rocdl.global.prefetch %{{.*}}, 8 : !llvm.ptr<1>
  // CHECK-NOT:     rocdl.global.prefetch %{{.*}}, 8
  // CHECK-COUNT-2: rocdl.global.prefetch %{{.*}}, 9 : !llvm.ptr<1>
  // CHECK-NOT:     rocdl.global.prefetch
  tt.func @global_prefetch_tensor(%ptrs: tensor<256x!tt.ptr<i8>, #blocked>) {
    amdg.global_prefetch %ptrs, speculative = true : tensor<256x!tt.ptr<i8>, #blocked>
    amdg.global_prefetch %ptrs, speculative = false : tensor<256x!tt.ptr<i8>, #blocked>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @global_prefetch_scalar
  // CHECK:       rocdl.global.prefetch %{{.*}}, 8 : !llvm.ptr<1>
  // CHECK-NOT:   rocdl.global.prefetch
  tt.func @global_prefetch_scalar(%ptr: !tt.ptr<f16>) {
    amdg.global_prefetch %ptr, speculative = true : !tt.ptr<f16>
    tt.return
  }
}
