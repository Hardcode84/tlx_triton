#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::triton::AMD {
namespace {
// Returns true for a producer-to-consumer dependency ordered by AsyncWait.
// AsyncWait does not release the consumed LDS slice for a later async write.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2,
                                       bool op1IsRead, bool op2IsRead,
                                       Allocation *allocation) {
  auto getAsyncDestinations = [](Operation *op) -> SmallVector<Value> {
    return llvm::TypeSwitch<Operation *, SmallVector<Value>>(op)
        .Case<triton::amdgpu::BufferLoadToLocalOp>(
            [](auto op) { return SmallVector<Value>{op.getDest()}; })
        .Case<triton::gpu::AsyncCopyGlobalToLocalOp,
              triton::amdgpu::AsyncTDMCopyGlobalToLocalOp>(
            [](auto op) { return SmallVector<Value>{op.getResult()}; })
        .Case<triton::amdgpu::AsyncTDMGatherOp>(
            [](auto op) { return SmallVector<Value>{op.getDst()}; })
        .Case<triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp>(
            [](auto op) { return llvm::to_vector(op.getDests()); })
        .Default([](Operation *) { return SmallVector<Value>{}; });
  };

  // Only filter a RAW dependency from a prior async LDS write to its local
  // consumer. In particular, never filter the opposite LocalLoad-to-prefetch
  // WAR dependency: a wait says nothing about consumer completion.
  auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op2);
  if (op1IsRead || !op2IsRead || !localLoad ||
      !isSyncedViaAsyncWait(localLoad)) {
    return false;
  }

  auto consumerBufferIds =
      allocation->getAllBufferIdsWithAliases(localLoad.getSrc());
  // A fused TDM copy can produce several independent buffers. Match each
  // destination, not just the first member. The wait marker covers the actual
  // producer; a later prefetch can alias the same allocation's other ring slot.
  return llvm::any_of(getAsyncDestinations(op1), [&](Value dest) {
    auto producerBufferIds = allocation->getAllBufferIdsWithAliases(dest);
    return llvm::any_of(producerBufferIds,
                        [&](auto id) { return consumerBufferIds.count(id); });
  });
}

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::amdgpu::InitBarrierOp,
                     triton::amdgpu::ArriveBarrierOp,
                     triton::amdgpu::AsyncCopyMbarrierArriveOp,
                     triton::amdgpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2, bool op1IsRead,
                  bool op2IsRead, Allocation *allocation) {
  return (filterAsyncLocalLoadsDependencies(op1, op2, op1IsRead, op2IsRead,
                                            allocation) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}

namespace {
// External model that stamps the marker interface onto an upstream ROCDL op we
// do not own. The interface has no methods, so the model body is empty.
template <typename OpT>
struct SchedulingBarrierModel
    : public ::mlir::triton::gpu::SchedulingBarrierOpInterface::ExternalModel<
          SchedulingBarrierModel<OpT>, OpT> {};
} // namespace

void registerSchedulingBarrierExternalModel(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *) {
    ROCDL::SchedBarrier::attachInterface<
        SchedulingBarrierModel<ROCDL::SchedBarrier>>(*ctx);
    ROCDL::SchedGroupBarrier::attachInterface<
        SchedulingBarrierModel<ROCDL::SchedGroupBarrier>>(*ctx);
  });
}
} // namespace mlir::triton::AMD
