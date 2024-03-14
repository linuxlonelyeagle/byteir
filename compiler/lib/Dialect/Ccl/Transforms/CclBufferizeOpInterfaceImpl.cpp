//===- CclBufferlize.cpp --------------------------------------*--- C++ -*-===//
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

#include "byteir/Dialect/Ccl/Transforms/CclBufferizeOpInterfaceImpl.h"
#include "PassDetail.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define DEBUG_TYPE "ccl-bufferize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

namespace {

using bufferization::AliasingValueList;
using bufferization::AnalysisState;
using bufferization::BufferizableOpInterface;
using bufferization::BufferizationOptions;
using bufferization::BufferRelation;

struct BroadcastOpInterface
    : public BufferizableOpInterface::ExternalModel<BroadcastOpInterface,
                                                    ccl::BroadcastOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &,
                                      const AnalysisState &) const {
    return {{op->getResult(0), BufferRelation::Equivalent, true}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto broadcastOp = cast<ccl::BroadcastOp>(op);
    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, broadcastOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();
    // Since the `getBuffer` later sets the `dynamicReplicaGroupsBuffer`, the
    // type here is `FailureOr<Value>`.It must be ensured that
    // dynamicReplicaGroupsBuffer has a value, as dynamicReplicaGroupsBuffer
    // will be used later to construct new ops.
    FailureOr<Value> dynamicReplicaGroupsBuffer = Value();
    if (broadcastOp.getDynamicReplicaGroups())
      dynamicReplicaGroupsBuffer =
          getBuffer(rewriter, broadcastOp.getDynamicReplicaGroups(), options);
    rewriter.setInsertionPoint(broadcastOp);
    rewriter.create<ccl::BroadcastOp>(op->getLoc(), TypeRange(), srcBuffer.value(), dynamicReplicaGroupsBuffer.value(), broadcastOp.getSynchronous(), broadcastOp.getReplicaGroupsAttr(), broadcastOp.getUniqueIdAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, {srcBuffer.value()});
    return success();
  }
};

struct SendOpInterface : public BufferizableOpInterface::ExternalModel<SendOpInterface, ccl::SendOp> {
   bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
    auto sendOp = cast<ccl::SendOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, sendOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure(); 
    rewriter.setInsertionPoint(sendOp);
    rewriter.create<ccl::SendOp>(sendOp->getLoc(), TypeRange(), srcBuffer.value(), sendOp.getDynamicTargetIndex(), sendOp.getSynchronousAttr(), sendOp.getTargetIndexAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, srcBuffer.value());
    return success();
  }
};

struct RecvOpInterface : public BufferizableOpInterface::ExternalModel<RecvOpInterface, ccl::RecvOp> {
   bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
    auto recvOp = cast<ccl::RecvOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, recvOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure(); 
    rewriter.setInsertionPoint(recvOp);
    rewriter.create<ccl::RecvOp>(recvOp->getLoc(), TypeRange(), srcBuffer.value(), recvOp.getDynamicSourceIndex(), recvOp.getSynchronousAttr(), recvOp.getSourceIndexAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, srcBuffer.value());
    return success();
  }
};

struct AllReduceOpInterface : public BufferizableOpInterface::ExternalModel<AllReduceOpInterface, ccl::AllReduceOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &, const AnalysisState &) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &, const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
    auto allReduceOp = cast<ccl::AllReduceOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, allReduceOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure(); 
    rewriter.setInsertionPoint(allReduceOp);
    auto memrefType = cast<MemRefType>(srcBuffer.value().getType());
    auto allocOp = rewriter.create<memref::AllocOp>(allReduceOp.getLoc(), memrefType);
    //rewriter.create<ccl::AllReduceOp>(allReduceOp->getLoc(), TypeRange(), srcBuffer.value(), recvOp.getDynamicSourceIndex(), recvOp.getSynchronousAttr(), recvOp.getSourceIndexAttr());
    bufferization::replaceOpWithBufferizedValues(rewriter, op, srcBuffer.value());*/
    return success();
  } 
};
} // namespace

void mlir::ccl::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ccl::CclDialect *dialect) {
    ccl::BroadcastOp::attachInterface<BroadcastOpInterface>(*ctx);
    ccl::SendOp::attachInterface<SendOpInterface>(*ctx);
    ccl::RecvOp::attachInterface<RecvOpInterface>(*ctx);
  });
}
