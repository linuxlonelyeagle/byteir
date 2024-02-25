#include "byteir/Dialect/Ccl/Transforms/CclBufferlize.h"
#include "byteir/Dialect/Ccl/IR/CclOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter, const BufferizationOptions &options) const {
    return success();
  }
};

struct CclBufferlizePass : public CclBufferlizeBase<CclBufferlizePass> {
  CclBufferlizePass() : CclBufferlizeBase() {}
  void runOnOperation() override {

  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ccl::CclDialect>();
    registry.addExtension(+[](MLIRContext *ctx, ccl::CclDialect * /*dialect*/) {
    ccl::BroadcastOp::attachInterface<BranchOpInterface>(*ctx);
  });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createCclBufferlizePass() {
  return std::make_unique<CclBufferlizePass>();
}