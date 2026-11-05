// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1250" | FileCheck %s

// An explicitly selected WMMA must not take the generic scaled-dot fallback
// while its layout still carries TLX's inference and pinning wrappers.
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 128]}>
#result = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
#operand_a = #tlx.no_verify_layout<#ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
#operand_b = #tlx.no_verify_layout<#ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
#scale_a = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
#scale_b = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @wmma_dot_scaled_pinned
  // CHECK-NEXT: %[[DOT:.*]] = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %arg4 lhs = e4m3 rhs = e4m3
  // CHECK-NEXT: tt.return %[[DOT]]
  tt.func @wmma_dot_scaled_pinned(%a: tensor<32x128xf8E4M3FN, #operand_a>,
                                 %b: tensor<128x32xf8E4M3FN, #operand_b>,
                                 %sa: tensor<32x4xi8, #scale_a>,
                                 %sb: tensor<32x4xi8, #scale_b>,
                                 %c: tensor<32x32xf32, #result>) -> tensor<32x32xf32, #result> {
    %dot = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN, #operand_a>, tensor<32x4xi8, #scale_a> * tensor<128x32xf8E4M3FN, #operand_b>, tensor<32x4xi8, #scale_b> -> tensor<32x32xf32, #result>
    tt.return %dot : tensor<32x32xf32, #result>
  }
}

// -----

// Tile-hint validation must respect an already selected WMMA distribution.
// Four M tiles per wave fit this 1x4 wave layout; planning a new 2x2 layout
// would incorrectly require 128 rows. Cover both plain and TLX-pinned layouts.
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2]]}, instrShape = [16, 16, 128]}>
#result = #tlx.no_verify_layout<#tlx.user_layout<#mma>>
#operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
#scale_a = #ttg.linear<{register = [[0, 1], [0, 2], [16, 0], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#scale_b = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [32, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @wmma_dot_scaled_selected_tile_hint
  // CHECK-NEXT: %[[DOT:.*]] = tt.dot_scaled {{.*}}amdg.wmma_tiles_per_warp = array<i32: 4, 1>
  // CHECK-NEXT: tt.return %[[DOT]]
  tt.func @wmma_dot_scaled_selected_tile_hint(%a: tensor<64x128xf8E4M3FN, #operand_a>,
                                              %b: tensor<128x64xf8E4M3FN, #operand_b>,
                                              %sa: tensor<64x4xi8, #scale_a>,
                                              %sb: tensor<64x4xi8, #scale_b>,
                                              %c: tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #mma> {
    %dot = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 4, 1>} : tensor<64x128xf8E4M3FN, #operand_a>, tensor<64x4xi8, #scale_a> * tensor<128x64xf8E4M3FN, #operand_b>, tensor<64x4xi8, #scale_b> -> tensor<64x64xf32, #mma>
    tt.return %dot : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @wmma_dot_scaled_pinned_tile_hint
  // CHECK-NEXT: %[[DOT:.*]] = tt.dot_scaled {{.*}}amdg.wmma_tiles_per_warp = array<i32: 4, 1>
  // CHECK-NEXT: tt.return %[[DOT]]
  tt.func @wmma_dot_scaled_pinned_tile_hint(%a: tensor<64x128xf8E4M3FN, #operand_a>,
                                            %b: tensor<128x64xf8E4M3FN, #operand_b>,
                                            %sa: tensor<64x4xi8, #scale_a>,
                                            %sb: tensor<64x4xi8, #scale_b>,
                                            %c: tensor<64x64xf32, #result>) -> tensor<64x64xf32, #result> {
    %dot = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 4, 1>} : tensor<64x128xf8E4M3FN, #operand_a>, tensor<64x4xi8, #scale_a> * tensor<128x64xf8E4M3FN, #operand_b>, tensor<64x4xi8, #scale_b> -> tensor<64x64xf32, #result>
    tt.return %dot : tensor<64x64xf32, #result>
  }
}

// -----

// A register-only accumulator slice has a linear encoding before matmul
// acceleration. It must still select native WMMA operand and scale layouts.
#input = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#scales = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
#acc = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>
// CHECK-DAG: #[[$LINEAR_MMA:.+]] = #ttg.amd_wmma<{{.*}}instrShape = [16, 16, 128]{{.*}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @wmma_dot_scaled_linear_accumulator
  // CHECK: ttg.convert_layout %arg4 {{.*}} -> tensor<64x64xf32, #[[$LINEAR_MMA]]>
  // CHECK: tt.dot_scaled {{.*}} -> tensor<64x64xf32, #[[$LINEAR_MMA]]>
  // CHECK: ttg.convert_layout {{.*}} : tensor<64x64xf32, #[[$LINEAR_MMA]]> -> tensor<64x64xf32, #linear{{.*}}>
  // CHECK: tt.return
  tt.func @wmma_dot_scaled_linear_accumulator(%a: tensor<64x128xf8E4M3FN, #input>,
                                             %b: tensor<128x64xf8E4M3FN, #input>,
                                             %sa: tensor<64x4xi8, #scales>,
                                             %sb: tensor<64x4xi8, #scales>,
                                             %c: tensor<64x64xf32, #acc>) -> tensor<64x64xf32, #acc> {
    %dot = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 2, 2>} : tensor<64x128xf8E4M3FN, #input>, tensor<64x4xi8, #scales> * tensor<128x64xf8E4M3FN, #input>, tensor<64x4xi8, #scales> -> tensor<64x64xf32, #acc>
    tt.return %dot : tensor<64x64xf32, #acc>
  }

  // TLX defers verification of layouts inferred through register-only views.
  // CHECK-LABEL: tt.func @wmma_dot_scaled_wrapped_linear_accumulator
  // CHECK: ttg.convert_layout %arg4 {{.*}} -> tensor<64x64xf32, #[[$LINEAR_MMA]]>
  // CHECK: tt.dot_scaled {{.*}} -> tensor<64x64xf32, #[[$LINEAR_MMA]]>
  // CHECK: ttg.convert_layout {{.*}} : tensor<64x64xf32, #[[$LINEAR_MMA]]> -> tensor<64x64xf32, #tlx.no_verify_layout<#linear{{.*}}>>
  // CHECK: tt.return
  tt.func @wmma_dot_scaled_wrapped_linear_accumulator(%a: tensor<64x128xf8E4M3FN, #tlx.no_verify_layout<#input>>,
                                                     %b: tensor<128x64xf8E4M3FN, #tlx.no_verify_layout<#input>>,
                                                     %sa: tensor<64x4xi8, #tlx.no_verify_layout<#scales>>,
                                                     %sb: tensor<64x4xi8, #tlx.no_verify_layout<#scales>>,
                                                     %c: tensor<64x64xf32, #tlx.no_verify_layout<#acc>>) -> tensor<64x64xf32, #tlx.no_verify_layout<#acc>> {
    %dot = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e4m3 {fastMath = false, amdg.wmma_tiles_per_warp = array<i32: 2, 2>} : tensor<64x128xf8E4M3FN, #tlx.no_verify_layout<#input>>, tensor<64x4xi8, #tlx.no_verify_layout<#scales>> * tensor<128x64xf8E4M3FN, #tlx.no_verify_layout<#input>>, tensor<64x4xi8, #tlx.no_verify_layout<#scales>> -> tensor<64x64xf32, #tlx.no_verify_layout<#acc>>
    tt.return %dot : tensor<64x64xf32, #tlx.no_verify_layout<#acc>>
  }
}
