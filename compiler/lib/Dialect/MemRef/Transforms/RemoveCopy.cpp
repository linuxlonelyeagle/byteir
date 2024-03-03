//===- RemoveCopy.cpp -----------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "byteir/Dialect/MemRef/Transforms/RemoveCopy.h"
#include "byteir/Dialect/MemRef/Utils/MemEffect.h"
#include "byteir/Utils/Hoist.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/ComposeSubView.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "PassDetail.h"

#define DEBUG_TYPE "remove-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace llvm;
using namespace mlir;
using namespace mlir::memref;

namespace {

// to check whether all uses of oldValue can be safely replaced with newValue
bool anyIncompatibleUse(Value oldValue, Value newValue) {
  return llvm::any_of(oldValue.getUses(),
                      [](OpOperand &operand) {
                        Operation *op = operand.getOwner();
                        Dialect *dialect = op->getDialect();
                        return llvm::isa<memref::CollapseShapeOp, func::CallOp>(
                                   op) ||
                               (dialect && dialect->getNamespace() == "byre");
                      }) &&
         (oldValue.getType() != newValue.getType());
}

bool allAliasesFromReshapeOp(llvm::ArrayRef<Value> aliases) {
  if (aliases.size() <= 1) 
    return false;
  if (!isa<memref::AllocOp>(aliases[0].getDefiningOp()))
    return false;
  for (size_t idx = 1; idx != aliases.size(); ++idx) {
    if (!isa<memref::CollapseShapeOp, memref::ExpandShapeOp>(aliases[idx].getDefiningOp())) 
      return false;
  }
  return true;
}

class RemoveCopyPattern : public OpRewritePattern<memref::CopyOp> {
public:
  RemoveCopyPattern(MLIRContext *context, DominanceInfo &dom)
      : OpRewritePattern(context), domInfo(dom) {}

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value src = copyOp.getSource();
    Value target = copyOp.getTarget();
    if (src == target)
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "match CopyOp " << copyOp << "\n");

    // only support at least one alloc for now
    /*if (!src.getDefiningOp<memref::AllocOp>() &&
        !target.getDefiningOp<memref::AllocOp>()) {
      return failure();
    }*/

    auto allocUseInTerminator = [](memref::AllocOp alloc) {
      for (auto user : alloc.getResult().getUsers()) {
        if (user->hasTrait<::mlir::OpTrait::IsTerminator>()) {
          return true;
        }
      }
      return false;
    };

    if (target.getType() != src.getType()) {
      // skip copy when it is used in a terminator
      if (auto srcAlloc = src.getDefiningOp<memref::AllocOp>()) {
        if (allocUseInTerminator(srcAlloc)) {
          return failure();
        }
      }
      if (auto targetAlloc = target.getDefiningOp<memref::AllocOp>()) {
        if (allocUseInTerminator(targetAlloc)) {
          return failure();
        }
      }
    }
    SmallVector<SmallVector<Value>, 2> aliases(2);
    getAllAlias(copyOp, aliases, /*skipNonOverlapedSubviews*/ true);
    /*llvm::outs() << "print copyOp:\n";
    copyOp.print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs() << "print src alias:\n";
    for (auto alias : aliases[0]) {
      alias.print(llvm::outs());
      llvm::outs() << "\n";
    }
    llvm::outs() << "print target alias:\n";
    for (auto alias : aliases[1]) {
      alias.print(llvm::outs());
      llvm::outs() << "\n";
    }*/
    llvm::DenseMap<Operation *, unsigned> opToIdx;
    unsigned idx = 0;
    copyOp->getBlock()->walk<WalkOrder::PreOrder>(
        [&](Operation *inner) { opToIdx[inner] = idx++; });

    SmallVector<OpMemEffectOrder, 2> memEffects(2);
    getMemEffects(memEffects, aliases, opToIdx, opToIdx[copyOp]);

    auto hasReadAfterWrite = [&](ArrayRef<Operation *> reads,
                                 ArrayRef<Operation *> writes) {
      for (auto read : reads) {
        for (auto write : writes) {
          if (opToIdx[write] < opToIdx[read]) {
            return true;
          } else if (opToIdx[write] == opToIdx[read]) {
            // we only support read-after-write in the same op
            // when that op is a copy
            return !isa<CopyOp>(write);
          }
        }
      }
      return false;
    };

    // check all read-after-write cases
    // except src write before copy and target read after copy
    if (hasReadAfterWrite(memEffects[0].before.reads,
                          memEffects[1].before.writes) ||
        hasReadAfterWrite(memEffects[1].before.reads,
                          memEffects[0].before.writes) ||
        hasReadAfterWrite(memEffects[0].after.reads,
                          memEffects[1].after.writes) ||
        hasReadAfterWrite(memEffects[1].after.reads,
                          memEffects[0].after.writes) ||
        hasReadAfterWrite(memEffects[0].after.reads,
                          memEffects[1].before.writes)) {
      LLVM_DEBUG(llvm::dbgs() << "failed at RAW\n");
      return failure();
    }

    auto isGlobalConstant = [](Value value) -> bool {
      if (auto getGlobalOp = value.getDefiningOp<memref::GetGlobalOp>()) {
        if (auto globalOp =
                SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
                    getGlobalOp, getGlobalOp.getNameAttr())) {
          return globalOp.getConstant();
        }
      }
      return false;
    };
    if (llvm::any_of(aliases[0], isGlobalConstant)) {
      LLVM_DEBUG(llvm::dbgs() << "failed to replace constant src");
      return failure();
    }
    // now it is legal to rewrite.
    // we prefer target alloc over src alloc in this implementation
    if (auto targetAlloc = target.getDefiningOp<memref::AllocOp>()) {
      if (auto srcDef = src.getDefiningOp()) {
        if (isa<memref::AllocOp, memref::SubViewOp>(srcDef))
          hoistUpOpInBlock(srcDef, domInfo);
      }

      // check whether targetAlloc replacible by src
      if (!domInfo.properlyDominates(src, targetAlloc)) {
        LLVM_DEBUG(llvm::dbgs() << "failed at src " << src
                                << " not dominated by " << targetAlloc << "\n");
        return failure();
      }

      if (!anyIncompatibleUse(target, src)) {
        rewriter.replaceOp(targetAlloc, {src});
        return success();
      }
    }
    if (auto srcAlloc = src.getDefiningOp<memref::AllocOp>()) {
      if (auto targetDef = target.getDefiningOp()) {
        if (isa<memref::AllocOp, memref::SubViewOp>(targetDef))
          hoistUpOpInBlock(targetDef, domInfo);
      }

      if (!domInfo.properlyDominates(target, srcAlloc)) {
        LLVM_DEBUG(llvm::dbgs() << "failed at target " << target
                                << " not dominated by " << srcAlloc << "\n");
        return failure();
      }
      
      if (!anyIncompatibleUse(src, target)) {
        rewriter.replaceOp(srcAlloc, {target});
        return success();
      }
    } /*else if (allAliasesFromReshapeOp(aliases[0])) { 
      auto srcAlloc = cast<memref::AllocOp>(aliases[0][0].getDefiningOp());
      if (auto targetDef = target.getDefiningOp()) {
        if (isa<memref::AllocOp, memref::SubViewOp>(targetDef))
          hoistUpOpInBlock(targetDef, domInfo);
      }
      
      if (!domInfo.properlyDominates(target, srcAlloc)) {
        LLVM_DEBUG(llvm::dbgs() << "failed at target " << target
                                << " not dominated by " << srcAlloc << "\n");
        return failure();
      }
      auto srcAllocType = srcAlloc.getType();
      auto targetType = target.getType().cast<MemRefType>();
      // create collapseShapeOp.
      if (srcAllocType.getRank() < targetType.getRank()) {
        auto srcAllocShape = srcAllocType.getShape();
        auto targetShape = targetType.getShape();
        llvm::SmallVector<ReassociationIndices> reassociation;
        int index = 0;
        for (auto targetDim : targetShape) {
          ReassociationIndices indices; 
          for (auto srcAllocDim : srcAllocShape) {
            while (targetDim >= 1) {
              targetDim = targetDim / srcAllocDim;
              indices.push_back(index++);
            }
          }
          reassociation.push_back(indices);
        }
        rewriter.setInsertionPoint(srcAlloc);
        auto collapseShapeOp = rewriter.create<memref::CollapseShapeOp>(srcAlloc.getLoc(), srcAllocType, target, reassociation);
        if (!anyIncompatibleUse(srcAlloc, collapseShapeOp)) 
          rewriter.replaceOp(srcAlloc, collapseShapeOp);
        if (!anyIncompatibleUse(src, target))
          rewriter.replaceOp(src.getDefiningOp(), target); 
        return success();
      } else if (srcAllocType.getRank() > targetType.getRank()) {
       return failure(); 
      }
    }*/
      /*if (auto collapseShapeOp = aliases[0][1].getDefiningOp<memref::CollapseShapeOp>()) {
        rewriter.setInsertionPoint(srcAlloc);
        auto expandShapeOp = rewriter.create<memref::ExpandShapeOp>(srcAlloc.getLoc(), srcAlloc.getType(), target, collapseShapeOp.getReassociationIndices());
        rewriter.replaceOp(srcAlloc, expandShapeOp);
        rewriter.replaceOp(collapseShapeOp, target);
        return success();
      } else if (auto expandShapeOp = aliases[0][1].getDefiningOp<memref::ExpandShapeOp>()) {
        rewriter.setInsertionPoint(srcAlloc);
        auto collapseShapeOp = rewriter.create<memref::CollapseShapeOp>(srcAlloc.getLoc(), srcAlloc.getType(), target, expandShapeOp.getReassociationIndices());
        rewriter.replaceOp(srcAlloc, collapseShapeOp);
        rewriter.replaceOp(expandShapeOp, target);
        return success();
      }*/
    /*else if (auto collapseShapeOp = src.getDefiningOp<memref::CollapseShapeOp>()) {       
      if (auto srcAlloc = collapseShapeOp.getSrc().getDefiningOp<memref::AllocOp>()) {
        if (!domInfo.properlyDominates(target, srcAlloc)) {
          LLVM_DEBUG(llvm::dbgs() << "failed at target " << target
                                  << " not dominated by " << srcAlloc << "\n");
          return failure();
        }
        rewriter.setInsertionPoint(srcAlloc);
        auto expandShape = rewriter.create<memref::ExpandShapeOp>(srcAlloc.getLoc(), srcAlloc.getType(), target, collapseShapeOp.getReassociationIndices());
        rewriter.replaceOp(srcAlloc, expandShape);
        rewriter.replaceOp(collapseShapeOp, {target});
        return success(); 
      }
    }*/
    return failure();
  }

private:
  DominanceInfo &domInfo;
};

struct RemoveCopyPass : public RemoveCopyBase<RemoveCopyPass> {
public:
  RemoveCopyPass() = default;
  void runOnOperation() override {

    func::FuncOp funcOp = getOperation();
    auto &domInfo = getAnalysis<DominanceInfo>();
    auto &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    populateRemoveCopyAfterBufferizationPattern(patterns, domInfo);

    // also insert related canonicalizer
    memref::AllocOp::getCanonicalizationPatterns(patterns, &ctx);
    memref::CopyOp::getCanonicalizationPatterns(patterns, &ctx);
    memref::SubViewOp::getCanonicalizationPatterns(patterns, &ctx);

    // To eliminate subview of subview where the second subview might have
    // incorrect strides.
    memref::populateComposeSubViewPatterns(patterns, &ctx);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(funcOp, frozenPatterns, grc))) {
      signalPassFailure();
    }
  }
};

} // namespace

void mlir::populateRemoveCopyAfterBufferizationPattern(
    RewritePatternSet &patterns, DominanceInfo &domInfo) {
  patterns.add<RemoveCopyPattern>(patterns.getContext(), domInfo);
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createRemoveCopyPass() {
  return std::make_unique<RemoveCopyPass>();
}
