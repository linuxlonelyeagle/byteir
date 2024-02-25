#ifndef BYTEIR_DIALECT_CCL_TRANSFORMS_CCLBUFFERLIZE_H
#define BYTEIR_DIALECT_CCL_TRANSFORMS_CCLBUFFERLIZE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
std::unique_ptr<OperationPass<ModuleOp>> createCclBufferlizePass();
} // namespace mlir

#endif // BYTEIR_DIALECT_CCL_TRANSFORMS_CCLBUFFERLIZE_H 