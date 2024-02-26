#include "byteir/Dialect/Ccl/Transforms/CclBufferlize.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "PassDetail.h"

#define DEBUG_TYPE "ccl-move-down"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;

namespace {

using bufferization::BufferizableOpInterface;
using bufferization::AliasingValueList;
using bufferization::AnalysisState;
using bufferization::BufferRelation;
using bufferization::BufferizationOptions;

struct BroadcastOpInterface : public BufferizableOpInterface::ExternalModel<BroadcastOpInterface, ccl::BroadcastOp> {
  bool bufferizesToMemoryRead(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *, OpOperand &, const AnalysisState &) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &, const AnalysisState &) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
    auto broadcastOp = cast<ccl::BroadcastOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, broadcastOp.getSrc(), options);
    FailureOr<Value> dynamicReplicaGroupsBuffer = Value();
    if (broadcastOp.getDynamicReplicaGroups().getImpl())
      dynamicReplicaGroupsBuffer = getBuffer(rewriter, broadcastOp.getDynamicReplicaGroups(), options);
    if (failed(srcBuffer))
      return failure();
    Type resultType;
    auto opResultType = broadcastOp.getType(0);
    if (auto rankedType = opResultType.dyn_cast<RankedTensorType>()) {
      resultType = MemRefType::get(rankedType.getShape(), rankedType.getElementType());
    } else if (auto unrankedType = opResultType.dyn_cast<UnrankedTensorType>()) {
      resultType = mlir::UnrankedMemRefType::get(unrankedType.getElementType(), 0);
    }
    broadcastOp = bufferization::replaceOpWithNewBufferizedOp<ccl::BroadcastOp>(rewriter, op, resultType, srcBuffer.value(), dynamicReplicaGroupsBuffer.value(), broadcastOp.getSynchronous(), broadcastOp.getReplicaGroupsAttr(), broadcastOp.getUniqueIdAttr());
    rewriter.replaceAllUsesWith(broadcastOp.getResult(), srcBuffer.value());
    rewriter.setInsertionPoint(broadcastOp);
    rewriter.create<ccl::BroadcastOp>(broadcastOp.getLoc(), TypeRange(), srcBuffer.value(), dynamicReplicaGroupsBuffer.value(), broadcastOp.getSynchronous(), broadcastOp.getReplicaGroupsAttr(), broadcastOp.getUniqueIdAttr());
    rewriter.eraseOp(broadcastOp);
    return success();
  }
};

struct CclBufferlizePass : public CclBufferlizeBase<CclBufferlizePass> {
  CclBufferlizePass() : CclBufferlizeBase() {}
  void runOnOperation() override {
    BufferizationOptions options = bufferization::getPartialBufferizationOptions();
    options.opFilter.allowDialect<ccl::CclDialect>();
    if (failed(bufferizeOp(getOperation(), options))) 
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ccl::CclDialect>();
    registry.addExtension(+[](MLIRContext *ctx, ccl::CclDialect * /*dialect*/) {
    ccl::BroadcastOp::attachInterface<BroadcastOpInterface>(*ctx);
    ctx->loadDialect<bufferization::BufferizationDialect>();
  });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createCclBufferlizePass() {
  return std::make_unique<CclBufferlizePass>();
}
