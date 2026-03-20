#include "IR/Dialect.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tlx-amd-insert-require-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace tlx = ::mlir::triton::tlx;

namespace mlir {
namespace triton {
namespace tlx {

#define GEN_PASS_DEF_TLXINSERTREQUIRELAYOUT
#include "tlx/dialect/include/Transforms/Passes.h.inc"

LogicalResult insertRequireLayout(ModuleOp m) {
  OpBuilder builder(m.getContext());
  LDBG("insertRequiredLayout\n");
  WalkResult result = m.walk([&](tt::DotOp dotOp) -> WalkResult {
    SetVector<Operation *> backwardSet;
    BackwardSliceOptions options;
    options.inclusive = false;
    options.omitUsesFromAbove = false;
    if (failed(mlir::getBackwardSlice(dotOp.getOperation(), &backwardSet,
                                      options))) {
      return WalkResult::interrupt();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "DotOp\n";
      dotOp.dump();
    });
    for (Operation *op : backwardSet) {
      if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(op)) {
        LLVM_DEBUG({
          llvm::dbgs() << "LocalLoadOp\n";
          localLoadOp.dump();
        });
        // Get the shared encoding for this local load op based on the dot op.
        bool incompatible = false;
        auto encoding = getSharedEncIfAllUsersAreDotEnc(
                            localLoadOp->getResult(0), incompatible)
                            .value_or(nullptr);
        if (encoding) {
          // If the source memdesc has a user-specified order that differs
          // from the derived one, rebuild the encoding with that order.
          // This lets the user signal K-contiguous data (order=[0,1]) for
          // pre-transposed B, which avoids ds_read_tr in the LLVM lowering.
          auto loadMemDescTy = op->getOperands()[0];
          if (auto srcType =
                  dyn_cast<ttg::MemDescType>(loadMemDescTy.getType())) {
            if (auto srcEnc = dyn_cast<ttg::SwizzledSharedEncodingAttr>(
                    srcType.getEncoding())) {
              if (srcEnc.getOrder() != encoding.getOrder()) {
                LDBG("Respecting user-specified order "
                     << srcEnc << " instead of derived " << encoding);
                encoding = ttg::SwizzledSharedEncodingAttr::get(
                    encoding.getContext(), encoding.getVec(),
                    encoding.getPerPhase(), encoding.getMaxPhase(),
                    srcEnc.getOrder(), encoding.getCTALayout());
              }
            }
          }
          LLVM_DEBUG({
            llvm::dbgs() << "SwizzledSharedEncodingAttr\n";
            encoding.dump();
          });
          builder.setInsertionPoint(localLoadOp);
          auto encodingAttr = mlir::cast<Attribute>(encoding);
          if (auto type = dyn_cast<ttg::MemDescType>(loadMemDescTy.getType())) {
            auto newType = ttg::MemDescType::get(
                type.getShape(), type.getElementType(), encodingAttr,
                type.getMemorySpace(), type.getMutableMemory());
            auto converLayoutOp = mlir::triton::tlx::RequireLayoutOp::create(builder, 
                op->getLoc(), newType, loadMemDescTy);
            localLoadOp->setOperand(0, converLayoutOp.getResult());
          }
        } else {
          // Cannot determine shared encoding for this local_load. This
          // happens when the load result feeds loop iter_args rather
          // than a dot operand directly. Skip it -- the layout will be
          // resolved by propagation from other local_loads that do feed
          // dots.
          LDBG("Skipping local_load without determinable dot encoding");
        }
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

struct TLXInsertRequireLayoutPass
    : public impl::TLXInsertRequireLayoutBase<TLXInsertRequireLayoutPass> {
public:
  using impl::TLXInsertRequireLayoutBase<
      TLXInsertRequireLayoutPass>::TLXInsertRequireLayoutBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(tlx::insertRequireLayout(m))) {
      signalPassFailure();
    }
  }
};

} // namespace tlx
} // namespace triton
} // namespace mlir
