//===- CclOps.cpp ---------------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;
using namespace mlir::ccl;

#include "byteir/Dialect/Ccl/IR/CclOpsDialect.cpp.inc"

namespace {

// Verify replica groups in collective communication operations.
LogicalResult
verifyReplicaGroups(std::optional<Location> location,
                    std::optional<ArrayRef<ReplicaGroupsIndices>> replicaGroups,
                    Value dynamicReplicaGroups) {
  if (dynamicReplicaGroups != nullptr && replicaGroups.has_value())
    return emitOptionalError(
        location,
        "dynamic_replica_groups and replica_groups can't exist simultaneously");

  if (dynamicReplicaGroups != nullptr) {
    ShapedType type = dynamicReplicaGroups.getType().cast<ShapedType>();
    if (!type.getElementType().isa<IndexType, IntegerType>())
      return emitOptionalError(
          location,
          "dynamic_replica_groups's element type should be index or integer");
    if (type.hasRank() && type.getRank() != 2)
      return emitOptionalError(
          location, "dynamic_replica_groups's rank should equal to 2");
  }

  if (replicaGroups.has_value()) {
    for (const ReplicaGroupsIndices &group : *replicaGroups) {
      llvm::SmallSet<int64_t, 8> replicaIdsSeen;
      for (int64_t replicaId : group) {
        if (!replicaIdsSeen.insert(replicaId).second) {
          return emitOptionalError(location, "replica id #", replicaId,
                                   " seen more than once");
        }
      }
    }
  }

  return success();
}

// Verify source/target index in p2p communication operations.
LogicalResult verifyP2PIndex(std::optional<Location> location,
                             std::optional<IntegerAttr> index,
                             Value dynamicIndex) {
  if (dynamicIndex != nullptr && index.has_value()) {
    return emitOptionalError(
        location, "dynamic_index and index can't exist simultaneously");
  }
  if (dynamicIndex == nullptr && !index.has_value()) {
    return emitOptionalError(
        location, "dynamic_index and index can't absent simultaneously");
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// ccl dialect.
//===----------------------------------------------------------------------===//

void CclDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "byteir/Dialect/Ccl/IR/CclOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ccl.wait
//===----------------------------------------------------------------------===//

LogicalResult
WaitOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr, OpaqueProperties,
                         RegionRange,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// ccl.all_reduce
//===----------------------------------------------------------------------===//

void mlir::ccl::AllReduceOp::build(::mlir::OpBuilder &builder,
                                   ::mlir::OperationState &result, Value src,
                                   StringAttr reduction,
                                   ArrayAttr replica_groups,
                                   IntegerAttr unique_id,
                                   ArrayRef<NamedAttribute> attributes) {
  result.addOperands(src);
  result.addAttribute(getReductionAttrName(result.name), reduction);
  result.addAttribute(getReplicaGroupsAttrName(result.name), replica_groups);
  result.addAttribute(getUniqueIdAttrName(result.name), unique_id);
  result.addAttributes(attributes);
  result.addTypes(src.getType());
}

LogicalResult
AllReduceOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                              ValueRange operands, DictionaryAttr,
                              OpaqueProperties, RegionRange,
                              SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.size() != 1)
    return emitOptionalError(
        location, "Expected operands' number equal to 1 for ccl.all_reduce");
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

LogicalResult AllReduceOp::verify() {
  auto reduction = getReduction();
  if (reduction != getRedOpSumName() && reduction != getRedOpProdName() &&
      reduction != getRedOpMinName() && reduction != getRedOpMaxName() &&
      reduction != getRedOpAvgName()) {
    return this->emitError("unknown reduction str: ") << reduction;
  }
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// ccl.all_gather
//===----------------------------------------------------------------------===//

LogicalResult AllGatherOp::verify() {
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// ccl.reduce_scatter
//===----------------------------------------------------------------------===//

LogicalResult ReduceScatterOp::verify() {
  auto reduction = getReduction();
  if (reduction != getRedOpSumName() && reduction != getRedOpProdName() &&
      reduction != getRedOpMinName() && reduction != getRedOpMaxName() &&
      reduction != getRedOpAvgName()) {
    return this->emitError("unknown reduction str: ") << reduction;
  }
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// ccl.all_to_all
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::verify() {
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

//===----------------------------------------------------------------------===//
// ccl.broadcast
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  return verifyReplicaGroups(getLoc(), getReplicaGroupsIndices(),
                             getDynamicReplicaGroups());
}

namespace {
// If two and more WaitOp's follow a BroadcastOp, the excess WaitOp's are
// eliminated and only one WaitOp is retained.
struct EliminateDuplicateWait : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = op.getOperation()->getNextNode();
    if (nextOp && isa<WaitOp>(nextOp)) {
      auto nextNextOp = nextOp->getNextNode();
      if (nextNextOp && isa<WaitOp>(nextNextOp)) {
        rewriter.replaceOp(nextNextOp, nextOp);
        return success();
      }
    }
    return failure();
  }
};

// A BroadcastOp with a synchronous of false plus a WaitOp equals a BroadcastOp
// with a synchronous of true.
struct CombineBroadcastAndWait : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = op.getOperation()->getNextNode();
    if (nextOp && isa<WaitOp>(nextOp) && op.getSynchronous() == false) {
      op.setSynchronous(true);
      rewriter.replaceOp(nextOp, op);
      return success();
    }
    return failure();
  }
};

// Eliminate the WaitOp after a BroadcastOp which synchronous is true.
struct EliminateUnnecessaryWait : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto nextOp = op.getOperation()->getNextNode();
    if (nextOp && isa<WaitOp>(nextOp) && op.getSynchronous() == true) {
      rewriter.replaceOp(nextOp, op);
      return success();
    }
    return failure();
  }
};
} // namespace

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  // clang-format off
  results.add<
  EliminateDuplicateWait,
  EliminateUnnecessaryWait,
  CombineBroadcastAndWait
  >(context);
  // clamg-format on
}

//===----------------------------------------------------------------------===//
// ccl.send
//===----------------------------------------------------------------------===//

LogicalResult SendOp::verify() {
  return verifyP2PIndex(getLoc(), getTargetIndexAttr(),
                        getDynamicTargetIndex());
}

LogicalResult
SendOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr, OpaqueProperties,
                         RegionRange,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// ccl.recv
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verify() {
  return verifyP2PIndex(getLoc(), getSourceIndexAttr(),
                        getDynamicSourceIndex());
}

LogicalResult
RecvOp::inferReturnTypes(MLIRContext *, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr, OpaqueProperties,
                         RegionRange,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "byteir/Dialect/Ccl/IR/CclOps.cpp.inc"
