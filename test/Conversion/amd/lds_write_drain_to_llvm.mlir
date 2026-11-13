// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefix=GFX1250
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s --check-prefix=GFX942

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Each lane ORs zero into its own dword at the buffer's base with a
  // volatile atomic, then the wave waits for LDS.
  // GFX1250-LABEL: llvm.func @lds_write_drain
  // GFX1250:       %[[BASE:.+]] = llvm.extractvalue %arg0[0]
  // GFX1250:       %[[DWORDS:.+]] = llvm.mlir.constant(32 : i32) : i32
  // GFX1250:       %[[LANE:.+]] = llvm.urem %{{.+}}, %{{.+}} : i32
  // GFX1250:       %[[IDX:.+]] = llvm.urem %[[LANE]], %[[DWORDS]] : i32
  // GFX1250:       %[[PTR:.+]] = llvm.getelementptr %[[BASE]][%[[IDX]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
  // GFX1250-NOT:   llvm.cond_br
  // GFX1250:       llvm.atomicrmw volatile _or %[[PTR]], %{{.+}} syncscope("workgroup") monotonic <alignment = 4> : !llvm.ptr<3>, i32
  // GFX1250-NEXT:  rocdl.s.wait.dscnt 0
  // GFX1250-NEXT:  llvm.return
  // GFX942-LABEL: llvm.func @lds_write_drain
  // GFX942-NOT:   llvm.atomicrmw
  // GFX942-NOT:   rocdl.s.wait
  // GFX942:       llvm.return
  tt.func @lds_write_drain(%buf: !ttg.memdesc<64x128xf16, #shared, #smem, mutable>) {
    amdg.lds_write_drain %buf : !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A buffer smaller than 32 dwords limits the lanes to its own dwords.
  // GFX1250-LABEL: llvm.func @lds_write_drain_small
  // GFX1250:       %[[DWORDS:.+]] = llvm.mlir.constant(4 : i32) : i32
  // GFX1250:       llvm.urem %{{.+}}, %[[DWORDS]] : i32
  // GFX1250:       llvm.atomicrmw volatile _or
  // GFX942-LABEL: llvm.func @lds_write_drain_small
  // GFX942-NOT:   llvm.atomicrmw
  tt.func @lds_write_drain_small(%buf: !ttg.memdesc<8xf16, #shared, #smem, mutable>) {
    amdg.lds_write_drain %buf : !ttg.memdesc<8xf16, #shared, #smem, mutable>
    tt.return
  }
}
