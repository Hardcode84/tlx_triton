#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"

#define DEBUG_TYPE "tritonamdgpu-anchor-lds-bases"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

// DS instructions take one address VGPR plus an unsigned 16-bit immediate.
constexpr int64_t kDSOffsetLimit = 1 << 16;
// Constants of one anchored base stay this close, leaving room for the small
// offsets that later GEPs add before the access.
constexpr int64_t kWindow = 40000;
constexpr unsigned kLDSAddrSpace = 3;

struct Access {
  GetElementPtrInst *gep;
  int64_t offset;
};

// Peel `add nuw %x, C` and `or disjoint %x, C` chains off a lane offset.
Value *peelConstant(Value *v, int64_t &offset) {
  while (true) {
    Value *x;
    ConstantInt *c;
    if (match(v, m_NUWAdd(m_Value(x), m_ConstantInt(c))) ||
        match(v, m_DisjointOr(m_Value(x), m_ConstantInt(c)))) {
      offset += c->getSExtValue();
      v = x;
      continue;
    }
    return v;
  }
}

// Shared-memory accesses are `@global_smem + C + laneOffset` GEPs. LICM hoists
// them, but for C beyond the 16-bit DS offset CodeGenPrepare sinks the address
// back next to every access (to fold it into the addressing mode), so each
// loop iteration recomputes `C + laneOffset` with a VALU add. Materialize one
// frozen base per lane offset and 40 KB window of constants right after the
// lane offset is defined, and rewrite each access as that base plus a small
// constant, which ISel folds into the DS immediate. `freeze` is opaque to
// CodeGenPrepare's address matching.
bool anchorLDSBases(Function &F) {
  const DataLayout &dl = F.getDataLayout();
  MapVector<std::pair<GlobalVariable *, Value *>, SmallVector<Access>> groups;
  for (Instruction &inst : instructions(F)) {
    auto *gep = dyn_cast<GetElementPtrInst>(&inst);
    if (!gep || gep->getAddressSpace() != kLDSAddrSpace ||
        !gep->getSourceElementType()->isIntegerTy(8) ||
        gep->getNumIndices() != 1)
      continue;
    APInt baseOffset(dl.getIndexTypeSizeInBits(gep->getType()), 0);
    auto *gv = dyn_cast<GlobalVariable>(
        gep->getPointerOperand()->stripAndAccumulateConstantOffsets(
            dl, baseOffset, /*AllowNonInbounds=*/true));
    Value *index = gep->getOperand(1);
    if (!gv || isa<Constant>(index) || !index->getType()->isIntegerTy(32))
      continue;
    int64_t offset = baseOffset.getSExtValue();
    Value *root = peelConstant(index, offset);
    if (offset < kDSOffsetLimit)
      continue;
    groups[{gv, root}].push_back({gep, offset});
  }

  bool changed = false;
  Type *i8 = Type::getInt8Ty(F.getContext());
  for (auto &[key, accesses] : groups) {
    auto [gv, root] = key;
    llvm::sort(accesses, [](const Access &a, const Access &b) {
      return a.offset < b.offset;
    });
    // Anchors go right after the lane offset (after the PHIs of its block),
    // then after the previous anchor. Both positions are looked up when used,
    // since rewritten accesses are erased.
    Instruction *base = nullptr;
    int64_t baseOffset = 0;
    for (Access &access : accesses) {
      if (!base || access.offset - baseOffset >= kWindow) {
        IRBuilder<> builder(F.getContext());
        if (base)
          builder.SetInsertPoint(base->getNextNode());
        else if (auto *rootInst = dyn_cast<Instruction>(root))
          builder.SetInsertPoint(
              isa<PHINode>(rootInst)
                  ? rootInst->getParent()->getFirstInsertionPt()
                  : std::next(rootInst->getIterator()));
        else
          builder.SetInsertPoint(F.getEntryBlock().getFirstInsertionPt());
        baseOffset = access.offset;
        Value *start = builder.CreateInBoundsGEP(
            i8, gv, builder.getInt32(baseOffset), "lds.anchor.start");
        Value *lane = builder.CreateInBoundsGEP(i8, start, root, "lds.anchor");
        base = cast<Instruction>(builder.CreateFreeze(lane, "lds.anchor.fz"));
      }
      IRBuilder<> builder(access.gep);
      Value *replacement = builder.CreateInBoundsGEP(
          i8, base, builder.getInt32(access.offset - baseOffset));
      replacement->takeName(access.gep);
      access.gep->replaceAllUsesWith(replacement);
      access.gep->eraseFromParent();
      changed = true;
    }
  }
  return changed;
}

} // namespace

namespace mlir::triton::AMD {
bool runAnchorLDSBasesPass(llvm::Function &F) {
  bool changed = anchorLDSBases(F);
  assert(!llvm::verifyFunction(F) &&
         "expected function to verify successfully");
  return changed;
}
} // namespace mlir::triton::AMD
